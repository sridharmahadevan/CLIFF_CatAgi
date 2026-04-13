"""Export compact public example bundles from legacy BASKET/ROCKET artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from html import unescape
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _redact_local_paths(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        name = Path(raw).name
        return name or raw

    return re.sub(r"/(?:Users|tmp|var|private|home|Volumes)(?:/[^\s)>\"]+)+", _replace, text)


def _sanitize_string(text: str) -> str:
    return _redact_local_paths(text).strip()


def _sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _sanitize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return [_sanitize_payload(value) for value in payload]
    if isinstance(payload, str):
        return _sanitize_string(payload)
    return payload


def _slugify(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(tokens) or "item"


def _company_key(text: str) -> str:
    return _slugify(text)


def _copy_sanitized_text(source: Path, dest: Path, *, replacements: dict[str, str] | None = None) -> None:
    text = _sanitize_string(source.read_text(encoding="utf-8"))
    for old, new in (replacements or {}).items():
        text = text.replace(old, new)
    dest.write_text(text, encoding="utf-8")


def _strip_html(text: str) -> str:
    stripped = re.sub(r"<[^>]+>", " ", text)
    return _normalize_space(unescape(stripped))


def _extract_html_table_rows(html_text: str, heading: str) -> list[list[str]]:
    pattern = rf"<h[23][^>]*>\s*{re.escape(heading)}\s*</h[23]>\s*<table>(.*?)</table>"
    match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    rows: list[list[str]] = []
    for row_match in re.finditer(r"<tr>(.*?)</tr>", match.group(1), flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row_match.group(1), flags=re.IGNORECASE | re.DOTALL)
        cleaned = [_strip_html(cell) for cell in cells]
        if cleaned:
            rows.append(cleaned)
    return rows


def _render_markdown_table(rows: list[list[str]]) -> list[str]:
    if len(rows) < 2:
        return []
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return lines


def _read_company_index_row(index_summary_path: Path, company: str) -> dict[str, Any]:
    company_key = _company_key(company)
    rows = list(_load_json(index_summary_path))
    for row in rows:
        row_company = _company_key(str(row.get("company") or ""))
        if row_company == company_key:
            return dict(row)
    raise ValueError(f"Could not find company {company!r} in {index_summary_path}")


def _build_extractor_snapshot(summary_payload: dict[str, Any], manifest_payload: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_payload(
        {
            "source_mode": summary_payload.get("source_mode") or manifest_payload.get("source_mode"),
            "extractor_mode": summary_payload.get("extractor_mode"),
            "statement_globs": list(manifest_payload.get("statement_globs") or []),
            "n_statement_files": summary_payload.get("n_statement_files"),
            "n_company_years": summary_payload.get("n_company_years"),
            "n_statement_rows": summary_payload.get("n_statement_rows"),
            "n_extractions": summary_payload.get("n_extractions"),
            "n_steps": summary_payload.get("n_steps"),
            "avg_actions_per_plan": summary_payload.get("avg_actions_per_plan"),
            "macro_skill_count": summary_payload.get("macro_skill_count"),
            "action_vocab_size": summary_payload.get("action_vocab_size"),
            "paths": dict(manifest_payload.get("paths") or {}),
        }
    )


def _build_panel_reranking_summary(reranking_payload: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_payload(
        {
            "reward_mode": reranking_payload.get("reward_mode"),
            "financial_targets": list(reranking_payload.get("financial_targets") or []),
            "financial_horizon": reranking_payload.get("financial_horizon"),
            "n_rows": reranking_payload.get("n_rows"),
            "n_changed": reranking_payload.get("n_changed"),
            "changed_rate": reranking_payload.get("changed_rate"),
            "mean_score_gain": reranking_payload.get("mean_score_gain"),
            "changed_mean_score_gain": reranking_payload.get("changed_mean_score_gain"),
            "selected_sources": dict(reranking_payload.get("selected_sources") or {}),
            "selected_labels": dict(reranking_payload.get("selected_labels") or {}),
            "changed_by_company_top10": dict(reranking_payload.get("changed_by_company_top10") or {}),
            "inputs": dict(reranking_payload.get("inputs") or {}),
        }
    )


def _build_company_snapshot(
    *,
    company_row: dict[str, Any],
    company_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(company_row)
    payload["summary_metrics"] = dict(company_summary)
    return _sanitize_payload(payload)


def _filter_company_examples(
    reranking_payload: dict[str, Any],
    *,
    company: str,
    limit: int,
) -> list[dict[str, Any]]:
    company_key = _company_key(company)
    matches = []
    for example in list(reranking_payload.get("top_changed_examples") or []):
        if _company_key(str(example.get("company") or "")) == company_key:
            matches.append(dict(example))
        if len(matches) >= limit:
            break
    return list(_sanitize_payload(matches))


def _build_readme(
    *,
    company: str,
    company_snapshot: dict[str, Any],
    extractor_snapshot: dict[str, Any],
    panel_summary: dict[str, Any],
    top_examples: list[dict[str, Any]],
    has_psr: bool,
    has_timeline: bool,
) -> str:
    aggregate_plan = dict(company_snapshot.get("aggregate_plan") or {})
    financial_summary = dict(company_snapshot.get("financial_summary") or {})
    summary_metrics = dict(company_snapshot.get("summary_metrics") or {})

    lines = [
        "# BASKET/ROCKET Example Bundle",
        "",
        "This is a compact, GitHub-friendly export of a legacy BASKET/ROCKET company snapshot",
        "prepared for the CLIFF `basket_rocket_sec` route. The full extractor panel, raw PDFs,",
        "and large intermediate state were intentionally excluded.",
        "",
        "## Snapshot",
        "",
        f"- Company: {company.upper()}",
        f"- Route: basket_rocket_sec",
        f"- Source kind: legacy full-panel reranking snapshot",
        f"- Reward mode: {panel_summary.get('reward_mode') or 'unknown'}",
        f"- Financial horizon: {panel_summary.get('financial_horizon') or 'n/a'}",
        f"- Coverage years: {financial_summary.get('coverage_years') or aggregate_plan.get('year_count') or 'n/a'}",
        f"- Latest year: {financial_summary.get('latest_year') or 'n/a'}",
        f"- Statement rows for company: {summary_metrics.get('n_rows') or company_snapshot.get('n_rows') or 0}",
        f"- Changed statements for company: {summary_metrics.get('n_changed') or company_snapshot.get('n_changed') or 0}",
        "",
        "## Included Files",
        "",
        "- `extractor_snapshot.json`: compact metadata from the full-panel BASKET extraction pass",
        "- `panel_reranking_summary.json`: sanitized full-panel financial reranking summary",
        "- `company_summary.json`: company-level reranking and aggregate-plan snapshot",
        "- `top_changed_examples.json`: representative changed statements for this company",
        "- `visualizations/README.md`: GitHub-friendly index for the saved visualizations",
        "- `visualizations/company_reranking.md`: GitHub-renderable company reranking summary",
        "- `visualizations/aggregate_plans.md`: GitHub-renderable aggregate-plan summary",
        "- `visualizations/company_reranking.html`: original company reranking visualizer",
        "- `visualizations/aggregate_plans.html`: original aggregate-plan drilldown",
    ]
    if has_psr:
        lines.append("- `visualizations/psr_drilldown.md`: GitHub-renderable PSR summary")
        lines.append("- `visualizations/psr_drilldown.html`: original company PSR comparison page")
    if has_timeline:
        lines.append("- `images/timeline.png`: company timeline graphic referenced by the PSR drilldown")

    lines.extend(
        [
            "",
            "## Panel Context",
            "",
            f"- Full-panel extracted workflows: {extractor_snapshot.get('n_extractions') or 0}",
            f"- Full-panel statement rows: {extractor_snapshot.get('n_statement_rows') or 0}",
            f"- Full-panel reranked rows: {panel_summary.get('n_rows') or 0}",
            f"- Full-panel changed rows: {panel_summary.get('n_changed') or 0}",
            f"- Full-panel mean score gain: {panel_summary.get('mean_score_gain')}",
            f"- Company changed rate: {summary_metrics.get('changed_rate') or company_snapshot.get('changed_rate')}",
            "",
            "## Aggregate Plan Highlights",
            "",
        ]
    )

    top_actions = list(aggregate_plan.get("top_actions") or [])[:5]
    top_edges = list(aggregate_plan.get("top_edges") or [])[:5]
    if top_actions:
        lines.append("### Top Actions")
        for row in top_actions:
            lines.append(f"- {row.get('action')}: {row.get('count')}")
        lines.append("")
    if top_edges:
        lines.append("### Top Edges")
        for row in top_edges:
            lines.append(f"- {row.get('src')} -> {row.get('dst')}: {row.get('count')}")
        lines.append("")

    lines.extend(["## Representative Changed Statements", ""])
    if top_examples:
        for example in top_examples:
            lines.append(
                f"- {example.get('statement_id')}: "
                f"{' -> '.join(example.get('base_actions') or [])} => "
                f"{' -> '.join(example.get('selected_actions') or [])} "
                f"(gain={example.get('score_gain')})"
            )
    else:
        lines.append("- No company-specific changed statements were available in the exported top-example list.")

    lines.append("")
    return "\n".join(lines)


def _build_company_reranking_markdown(
    *,
    company_snapshot: dict[str, Any],
    top_examples: list[dict[str, Any]],
) -> str:
    aggregate_plan = dict(company_snapshot.get("aggregate_plan") or {})
    financial_summary = dict(company_snapshot.get("financial_summary") or {})
    summary_metrics = dict(company_snapshot.get("summary_metrics") or {})
    lines = [
        "# ROCKET Reranking Visualizer",
        "",
        f"- Company: {str(company_snapshot.get('company') or '').upper()}",
        f"- Ticker: {financial_summary.get('ticker') or 'n/a'}",
        f"- Statement rows: {company_snapshot.get('n_rows') or 0}",
        f"- Changed statements: {company_snapshot.get('n_changed') or 0}",
        f"- Changed rate: {company_snapshot.get('changed_rate')}",
        f"- Mean score gain: {company_snapshot.get('mean_score_gain')}",
        f"- Source mix: {company_snapshot.get('source_mix') or 'n/a'}",
        f"- Inserted mix: {company_snapshot.get('inserted_mix') or 'n/a'}",
        "",
        "## Financial Snapshot",
        "",
    ]
    for metric in list(financial_summary.get("snapshot_metrics") or []):
        lines.append(f"- {metric.get('label')}: {metric.get('value')}")
    if not list(financial_summary.get("snapshot_metrics") or []):
        lines.append("- No filing-year snapshot metrics were available.")
    lines.extend(["", "## Top Actions", ""])
    for row in list(aggregate_plan.get("top_actions") or [])[:8]:
        lines.append(f"- {row.get('action')}: {row.get('count')}")
    lines.extend(["", "## Top Edges", ""])
    for row in list(aggregate_plan.get("top_edges") or [])[:8]:
        lines.append(f"- {row.get('src')} -> {row.get('dst')}: {row.get('count')}")
    lines.extend(["", "## Changed Statement Examples", ""])
    for example in top_examples:
        lines.append(
            f"- `{example.get('statement_id')}` ({example.get('year')}): "
            f"{' -> '.join(example.get('base_actions') or [])} => "
            f"{' -> '.join(example.get('selected_actions') or [])} "
            f"(gain={example.get('score_gain')})"
        )
    if not top_examples:
        lines.append("- No representative changed statements were exported.")
    lines.extend(["", "## Year Distribution", ""])
    for year, count in list((summary_metrics.get("changed_by_year") or {}).items())[:24]:
        lines.append(f"- {year}: {count}")
    lines.append("")
    return "\n".join(lines)


def _build_aggregate_plans_markdown(*, company_snapshot: dict[str, Any]) -> str:
    aggregate_plan = dict(company_snapshot.get("aggregate_plan") or {})
    lines = [
        "# ROCKET Aggregate Plans",
        "",
        f"- Company: {str(company_snapshot.get('company') or '').upper()}",
        f"- Plan count: {aggregate_plan.get('plan_count') or 0}",
        f"- Statement count: {aggregate_plan.get('statement_count') or 0}",
        f"- Year count: {aggregate_plan.get('year_count') or 0}",
        f"- Year span: {aggregate_plan.get('year_span') or []}",
        "",
        "## Top Action Counts",
        "",
    ]
    for row in list(aggregate_plan.get("top_actions") or [])[:10]:
        lines.append(f"- {row.get('action')}: {row.get('count')}")
    lines.extend(["", "## Top Edge Counts", ""])
    for row in list(aggregate_plan.get("top_edges") or [])[:10]:
        lines.append(f"- {row.get('src')} -> {row.get('dst')}: {row.get('count')}")
    lines.extend(["", "## Top Sequences", ""])
    for row in list(aggregate_plan.get("top_sequences") or [])[:10]:
        lines.append(f"- {row.get('sequence')}: {row.get('count')}")
    lines.append("")
    return "\n".join(lines)


def _build_psr_markdown(*, company: str, psr_html: str, timeline_included: bool) -> str:
    score_rows = _extract_html_table_rows(psr_html, "ROCKET score summary")
    year_rows = _extract_html_table_rows(psr_html, "Year table")
    section_names = (
        "PSR baseline",
        "PSR max length 3",
        "PSR section-aware",
    )
    lines = [
        "# PSR Drilldown",
        "",
        f"- Company: {company.upper()}",
        "- This Markdown companion is included so GitHub can render the PSR snapshot directly.",
    ]
    if timeline_included:
        lines.extend(["", "## Timeline", "", "![Timeline](../images/timeline.png)"])
    if score_rows:
        lines.extend(["", "## ROCKET Score Summary", ""])
        lines.extend(_render_markdown_table(score_rows))
    if year_rows:
        lines.extend(["", "## Year Table", ""])
        lines.extend(_render_markdown_table(year_rows))
    for section_name in section_names:
        rows = _extract_html_table_rows(psr_html, section_name)
        if rows:
            lines.extend(["", f"## {section_name}", ""])
            lines.extend(_render_markdown_table(rows))
    lines.append("")
    return "\n".join(lines)


def _build_visualizations_readme(*, has_psr: bool) -> str:
    lines = [
        "# Visualization Views",
        "",
        "GitHub renders the Markdown companions in this folder more reliably than raw HTML blob pages.",
        "The original `.html` files are still included for local opening in a browser.",
        "",
        "## Recommended GitHub Views",
        "",
        "- [Company reranking](company_reranking.md)",
        "- [Aggregate plans](aggregate_plans.md)",
    ]
    if has_psr:
        lines.append("- [PSR drilldown](psr_drilldown.md)")
    lines.extend(
        [
            "",
            "## Original HTML Files",
            "",
            "- `company_reranking.html`",
            "- `aggregate_plans.html`",
        ]
    )
    if has_psr:
        lines.append("- `psr_drilldown.html`")
    lines.append("")
    return "\n".join(lines)


def export_basket_rocket_example(
    *,
    company: str,
    extractor_dir: Path,
    reranking_dir: Path,
    company_viz_dir: Path,
    output_dir: Path,
    force: bool = False,
    psr_company_dir: Path | None = None,
    top_examples_limit: int = 5,
) -> dict[str, Any]:
    """Export a compact public bundle from legacy BASKET/ROCKET company artifacts."""

    company_key = _company_key(company)
    extractor_dir = extractor_dir.resolve()
    reranking_dir = reranking_dir.resolve()
    company_viz_dir = company_viz_dir.resolve()
    output_dir = output_dir.resolve()
    psr_company_dir = psr_company_dir.resolve() if psr_company_dir else None

    extractor_summary_path = extractor_dir / "summary.json"
    extractor_manifest_path = extractor_dir / "plan_block_manifest.json"
    reranking_summary_path = reranking_dir / "reranked_summary.json"
    company_index_path = company_viz_dir / "index_summary.json"
    company_summary_path = company_viz_dir / f"rocket_reranking_visualizer___{company_key}_summary.json"
    company_html_path = company_viz_dir / f"rocket_reranking_visualizer___{company_key}.html"
    aggregate_html_path = company_viz_dir / f"rocket_aggregate_plans___{company_key}.html"

    for required in (
        extractor_summary_path,
        extractor_manifest_path,
        reranking_summary_path,
        company_index_path,
        company_summary_path,
        company_html_path,
        aggregate_html_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Expected artifact at {required}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir = output_dir / "visualizations"
    images_dir = output_dir / "images"
    visualizations_dir.mkdir()
    images_dir.mkdir()

    extractor_summary = dict(_load_json(extractor_summary_path))
    extractor_manifest = dict(_load_json(extractor_manifest_path))
    reranking_summary = dict(_load_json(reranking_summary_path))
    company_index_row = _read_company_index_row(company_index_path, company)
    company_summary = dict(_load_json(company_summary_path))

    extractor_snapshot = _build_extractor_snapshot(extractor_summary, extractor_manifest)
    panel_summary = _build_panel_reranking_summary(reranking_summary)
    company_snapshot = _build_company_snapshot(company_row=company_index_row, company_summary=company_summary)
    top_examples = _filter_company_examples(reranking_summary, company=company, limit=top_examples_limit)

    _write_json(output_dir / "extractor_snapshot.json", extractor_snapshot)
    _write_json(output_dir / "panel_reranking_summary.json", panel_summary)
    _write_json(output_dir / "company_summary.json", company_snapshot)
    _write_json(output_dir / "top_changed_examples.json", top_examples)

    _copy_sanitized_text(company_html_path, visualizations_dir / "company_reranking.html")
    _copy_sanitized_text(
        aggregate_html_path,
        visualizations_dir / "aggregate_plans.html",
        replacements={"href=\"index.html\"": 'href="company_reranking.html"'},
    )
    (visualizations_dir / "company_reranking.md").write_text(
        _build_company_reranking_markdown(
            company_snapshot=company_snapshot,
            top_examples=top_examples,
        ),
        encoding="utf-8",
    )
    (visualizations_dir / "aggregate_plans.md").write_text(
        _build_aggregate_plans_markdown(company_snapshot=company_snapshot),
        encoding="utf-8",
    )

    psr_html_included = False
    timeline_included = False
    psr_markdown = ""
    if psr_company_dir:
        psr_html_path = psr_company_dir / f"{company_key}.html"
        timeline_path = psr_company_dir / f"{company_key}_timeline.png"
        if psr_html_path.exists():
            psr_raw_html = _sanitize_string(psr_html_path.read_text(encoding="utf-8"))
            _copy_sanitized_text(
                psr_html_path,
                visualizations_dir / "psr_drilldown.html",
                replacements={
                    f"{company_key}_timeline.png": "../images/timeline.png",
                },
            )
            psr_html_included = True
            psr_markdown = _build_psr_markdown(
                company=company_key,
                psr_html=psr_raw_html.replace(f"{company_key}_timeline.png", "../images/timeline.png"),
                timeline_included=timeline_path.exists(),
            )
        if timeline_path.exists():
            shutil.copy2(timeline_path, images_dir / "timeline.png")
            timeline_included = True
        if psr_html_included:
            (visualizations_dir / "psr_drilldown.md").write_text(psr_markdown, encoding="utf-8")

    (visualizations_dir / "README.md").write_text(
        _build_visualizations_readme(has_psr=psr_html_included),
        encoding="utf-8",
    )

    readme_text = _build_readme(
        company=company,
        company_snapshot=company_snapshot,
        extractor_snapshot=extractor_snapshot,
        panel_summary=panel_summary,
        top_examples=top_examples,
        has_psr=psr_html_included,
        has_timeline=timeline_included,
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    manifest = {
        "bundle_version": 1,
        "route": "basket_rocket_sec",
        "source_kind": "legacy_fullpanel_snapshot",
        "company": company_key,
        "reward_mode": panel_summary.get("reward_mode"),
        "financial_horizon": panel_summary.get("financial_horizon"),
        "financial_targets": list(panel_summary.get("financial_targets") or []),
        "panel_row_count": panel_summary.get("n_rows"),
        "panel_changed_count": panel_summary.get("n_changed"),
        "company_row_count": company_summary.get("n_rows"),
        "company_changed_count": company_summary.get("n_changed"),
        "files": [
            "README.md",
            "extractor_snapshot.json",
            "panel_reranking_summary.json",
            "company_summary.json",
            "top_changed_examples.json",
            "example_manifest.json",
        ],
        "visualizations_dir": "visualizations",
        "images_dir": "images",
        "included_visualizations": [
            "README.md",
            "company_reranking.md",
            "company_reranking.html",
            "aggregate_plans.md",
            "aggregate_plans.html",
        ]
        + (["psr_drilldown.md", "psr_drilldown.html"] if psr_html_included else []),
        "included_images": ["timeline.png"] if timeline_included else [],
    }
    _write_json(output_dir / "example_manifest.json", manifest)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a compact public BASKET/ROCKET example bundle.")
    parser.add_argument("--company", required=True, help="Company slug or display name, for example adobe or coca_cola.")
    parser.add_argument("--extractor-dir", required=True, help="Path to the legacy BASKET extractor output directory.")
    parser.add_argument("--reranking-dir", required=True, help="Path to the legacy ROCKET reranking output directory.")
    parser.add_argument("--company-viz-dir", required=True, help="Path to the company visualizer directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the compact example bundle should be written.")
    parser.add_argument(
        "--psr-company-dir",
        default="",
        help="Optional directory containing per-company PSR drilldowns and timeline images.",
    )
    parser.add_argument(
        "--top-examples-limit",
        type=int,
        default=5,
        help="Maximum number of representative changed statements to export.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it already exists.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifest = export_basket_rocket_example(
        company=str(args.company),
        extractor_dir=Path(args.extractor_dir),
        reranking_dir=Path(args.reranking_dir),
        company_viz_dir=Path(args.company_viz_dir),
        output_dir=Path(args.output_dir),
        force=bool(args.force),
        psr_company_dir=Path(args.psr_company_dir) if str(args.psr_company_dir).strip() else None,
        top_examples_limit=int(args.top_examples_limit),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
