"""HTML workflow visualizations for FF2 BASKET/ROCKET SEC worksets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .repo_layout import resolve_brand_panel_root


@dataclass(frozen=True)
class BasketRocketVisualizationResult:
    """Generated visualization assets for a batch of FF2 BASKET/ROCKET worksets."""

    index_path: Path
    summary_path: Path
    company_summary_path: Path
    workset_pages: tuple[tuple[str, str], ...]
    aggregate_pages: tuple[tuple[str, str], ...]
    year_context_pages: tuple[tuple[str, str], ...]
    company_reranking_pages: tuple[tuple[str, str], ...]
    company_aggregate_pages: tuple[tuple[str, str], ...]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRAND_PANEL_ROOT = resolve_brand_panel_root()
_DEFAULT_FINANCIAL_PANEL_PATH = (
    _BRAND_PANEL_ROOT / "outputs" / "company_panel_26" / "rocket_company_outcomes.csv"
)
_FINANCIAL_SNAPSHOT_SPECS = (
    ("revenue", "Revenue", "money"),
    ("operating_margin", "Op margin", "pct"),
    ("free_cash_flow_margin", "FCF margin", "pct"),
    ("return_on_assets", "ROA", "pct"),
    ("debt_to_assets", "Debt/assets", "pct"),
)
_FINANCIAL_OUTCOME_SPECS = (
    ("revenue_yoy", "Revenue YoY", "pct"),
    ("operating_margin", "Op margin", "pct"),
    ("free_cash_flow_margin", "FCF margin", "pct"),
    ("return_on_assets", "ROA", "pct"),
    ("debt_to_assets", "Debt/assets", "pct"),
)


def _escape_html(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _slugify(value: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in value.lower())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _format_money_short(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    for threshold, suffix in (
        (1_000_000_000_000.0, "T"),
        (1_000_000_000.0, "B"),
        (1_000_000.0, "M"),
        (1_000.0, "K"),
    ):
        if abs_value >= threshold:
            return f"${value / threshold:.1f}{suffix}"
    return f"${value:,.0f}"


def _format_ratio(value: float | None, *, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "money":
        return _format_money_short(value)
    return f"{value * 100.0:+.1f}%"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_company_token(value: str) -> str:
    token = _slugify(value).replace("_", "")
    for suffix in ("inc", "corp", "corporation", "company", "co", "ltd", "plc", "holdings", "group"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
    return token


def _build_financial_panel_index(panel_rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    index: dict[tuple[str, int], dict[str, str]] = {}
    for row in panel_rows:
        company_id = str(row.get("company_id", "")).strip().lower()
        ticker = str(row.get("ticker", "")).strip().lower()
        year = int(_float_or_none(row.get("fiscal_year", 0)) or 0)
        if year <= 0:
            continue
        for key in {company_id, ticker, _normalize_company_token(company_id)}:
            if key:
                index[(key, year)] = row
    return index


def _financial_row_has_values(row: dict[str, str]) -> bool:
    if not row:
        return False
    if str(row.get("companyfacts_status", "")).strip() == "ok":
        return True
    for key, _label, _kind in _FINANCIAL_SNAPSHOT_SPECS + _FINANCIAL_OUTCOME_SPECS:
        if str(row.get(key, "")).strip():
            return True
    return False


def _collect_metric_values(row: dict[str, str], specs: tuple[tuple[str, str, str], ...]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for key, label, kind in specs:
        value = _float_or_none(row.get(key, ""))
        if value is None:
            continue
        values.append({"label": label, "value": _format_ratio(value, kind=kind)})
    return values


def _financial_lookup_keys(company: str, ticker: str = "") -> list[str]:
    keys = []
    if ticker:
        keys.append(str(ticker).strip().lower())
    company_key = str(company).strip().lower()
    if company_key:
        keys.append(company_key)
        keys.append(_normalize_company_token(company_key))
    deduped = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _resolve_financial_company_series(
    panel_index: dict[tuple[str, int], dict[str, str]],
    *,
    company: str,
    ticker: str = "",
) -> tuple[str, list[dict[str, str]]]:
    for key in _financial_lookup_keys(company, ticker):
        rows = [row for (panel_key, _year), row in panel_index.items() if panel_key == key]
        if rows:
            unique_rows = sorted(
                {
                    (str(row.get("company_id", "")), str(row.get("fiscal_year", ""))): row
                    for row in rows
                }.values(),
                key=lambda row: int(_float_or_none(row.get("fiscal_year", 0)) or 0),
            )
            return key, unique_rows
    return "", []


def _summarize_company_financial(
    company: str,
    panel_index: dict[tuple[str, int], dict[str, str]],
    *,
    ticker: str = "",
) -> dict[str, object]:
    _resolved_key, rows = _resolve_financial_company_series(panel_index, company=company, ticker=ticker)
    if not rows:
        return {}
    rows_with_financials = [row for row in rows if _financial_row_has_values(row)]
    latest_row = rows_with_financials[-1] if rows_with_financials else rows[-1]
    latest_year = int(_float_or_none(latest_row.get("fiscal_year", 0)) or 0)
    next_row = {}
    for row in rows:
        if int(_float_or_none(row.get("fiscal_year", 0)) or 0) == latest_year + 1:
            next_row = row
            break
    return {
        "coverage_years": len(rows_with_financials),
        "latest_year": latest_year,
        "ticker": str(latest_row.get("ticker", "")).strip(),
        "snapshot_metrics": _collect_metric_values(latest_row, _FINANCIAL_SNAPSHOT_SPECS),
        "outcome_metrics": _collect_metric_values(next_row, _FINANCIAL_OUTCOME_SPECS),
        "outcome_year": latest_year + 1 if _financial_row_has_values(next_row) else 0,
    }


def _summarize_company_financial_by_year(
    company: str,
    panel_index: dict[tuple[str, int], dict[str, str]],
    *,
    ticker: str = "",
) -> dict[int, dict[str, object]]:
    _resolved_key, rows = _resolve_financial_company_series(panel_index, company=company, ticker=ticker)
    by_year: dict[int, dict[str, object]] = {}
    rows_by_year = {int(_float_or_none(row.get("fiscal_year", 0)) or 0): row for row in rows}
    for year, row in sorted(rows_by_year.items()):
        if year <= 0:
            continue
        next_row = rows_by_year.get(year + 1, {})
        by_year[year] = {
            "year": year,
            "ticker": str(row.get("ticker", "")).strip(),
            "snapshot_metrics": _collect_metric_values(row, _FINANCIAL_SNAPSHOT_SPECS),
            "outcome_metrics": _collect_metric_values(next_row, _FINANCIAL_OUTCOME_SPECS),
            "outcome_year": year + 1 if _financial_row_has_values(next_row) else 0,
        }
    return by_year


def _sequence_string(actions: list[str]) -> str:
    return " -> ".join(actions) if actions else "(empty)"


def _inline_bar_chart(
    items: list[tuple[str, float]],
    *,
    width: int = 560,
    height: int = 220,
    bar_color: str = "#0f766e",
) -> str:
    if not items:
        return '<p class="empty-chart">No values available.</p>'
    margin_left = 48
    margin_right = 14
    margin_top = 18
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(value for _label, value in items)
    max_value = max(max_value, 1.0)
    slot_width = plot_width / max(len(items), 1)
    bar_width = min(44, slot_width * 0.66)

    parts = [
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#cbd5e1" stroke-width="1.5" />',
    ]
    for index, (label, value) in enumerate(items):
        x_center = margin_left + (slot_width * index) + (slot_width / 2)
        bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
        x = x_center - (bar_width / 2)
        y = margin_top + plot_height - bar_height
        value_label = _format_score(value) if value < 10 else str(int(round(value)))
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="10" fill="{bar_color}" />'
        )
        parts.append(
            f'<text x="{x_center:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="chart-value">{_escape_html(value_label)}</text>'
        )
        parts.append(
            f'<text x="{x_center:.1f}" y="{margin_top + plot_height + 16:.1f}" text-anchor="middle" class="chart-label" transform="rotate(-25 {x_center:.1f} {margin_top + plot_height + 16:.1f})">{_escape_html(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _workflow_graph_svg(
    top_actions: list[dict[str, object]],
    top_edges: list[dict[str, object]],
    *,
    width: int = 560,
    height: int = 220,
) -> str:
    if not top_actions:
        return '<p class="empty-chart">No workflow graph available.</p>'
    action_rows = top_actions[:8]
    counts = {str(item["action"]): int(item["count"]) for item in action_rows}
    action_rank = {str(item["action"]): index for index, item in enumerate(action_rows)}
    edge_rows = [
        {"src": str(item["src"]), "dst": str(item["dst"]), "count": int(item["count"])}
        for item in top_edges[:12]
        if str(item["src"]) in counts and str(item["dst"]) in counts
    ]
    nodes = [str(item["action"]) for item in action_rows]
    if edge_rows:
        connected = {edge["src"] for edge in edge_rows} | {edge["dst"] for edge in edge_rows}
        nodes = sorted(connected, key=lambda node: (action_rank.get(node, 999), node))
    positions: dict[str, tuple[float, float]] = {}
    x_spacing = width / (len(nodes) + 1)
    for index, node in enumerate(nodes):
        positions[node] = ((index + 1) * x_spacing, 78 if index % 2 == 0 else 150)

    max_edge_count = max((int(item["count"]) for item in edge_rows), default=1)
    parts = [
        f'<svg class="plan-graph-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#94a3b8" /></marker></defs>',
    ]
    for edge in edge_rows:
        if edge["src"] not in positions or edge["dst"] not in positions:
            continue
        x1, y1 = positions[edge["src"]]
        x2, y2 = positions[edge["dst"]]
        width_scale = 1.5 + (4.5 * (int(edge["count"]) / max_edge_count))
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="{width_scale:.1f}" stroke-linecap="round" marker-end="url(#arrow)" opacity="0.88" />'
        )
    max_node_count = max(counts.values()) if counts else 1
    for node in nodes:
        x, y = positions[node]
        radius = 20 + (12 * (counts[node] / max_node_count))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#e0f2fe" stroke="#0f766e" stroke-width="2" />')
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="graph-node-label">{_escape_html(node)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _format_year_span(year_span: list[object]) -> str:
    if len(year_span) == 2:
        return f"{year_span[0]}-{year_span[1]}"
    return "n/a"


def _render_financial_metric_chips(metrics: list[dict[str, str]]) -> str:
    return "".join(
        f'<span class="metric-chip"><span class="metric-label">{_escape_html(metric["label"])}</span>'
        f'<span class="metric-value">{_escape_html(metric["value"])}</span></span>'
        for metric in metrics
    )


def _render_financial_overview(summary: dict[str, object], *, compact: bool = False) -> str:
    if not summary:
        return ""
    latest_year = int(summary.get("latest_year", 0) or 0)
    coverage_years = int(summary.get("coverage_years", 0) or 0)
    ticker = str(summary.get("ticker", "")).strip()
    snapshot_metrics = list(summary.get("snapshot_metrics", []))
    outcome_metrics = list(summary.get("outcome_metrics", []))
    outcome_year = int(summary.get("outcome_year", 0) or 0)
    section_class = "financial financial-compact" if compact else "financial"
    empty_snapshot_html = '<p class="aggregate-empty">No snapshot metrics available.</p>'
    headline = (
        f"{latest_year} snapshot"
        + (f" · {ticker}" if ticker else "")
        + (f" · {coverage_years} years with companyfacts" if coverage_years else "")
    )
    outcome_html = ""
    if outcome_metrics and outcome_year:
        outcome_html = (
            '<div class="financial-subgroup">'
            f'<div class="financial-subtitle">Reward horizon: realized {outcome_year} outcome</div>'
            f'<div class="metric-chips">{_render_financial_metric_chips(outcome_metrics)}</div>'
            "</div>"
        )
    return (
        f'<section class="{section_class}">'
        '<div class="financial-head">'
        "<h3>Financial Context</h3>"
        f'<span class="financial-meta">{_escape_html(headline)}</span>'
        "</div>"
        '<div class="financial-subgroup">'
        '<div class="financial-subtitle">Filing-year snapshot</div>'
        f'<div class="metric-chips">{_render_financial_metric_chips(snapshot_metrics) or empty_snapshot_html}</div>'
        "</div>"
        f"{outcome_html}"
        "</section>"
    )


def _render_year_financial_panel(detail: dict[str, object]) -> str:
    if not detail:
        return ""
    snapshot_metrics = list(detail.get("snapshot_metrics", []))
    outcome_metrics = list(detail.get("outcome_metrics", []))
    outcome_year = int(detail.get("outcome_year", 0) or 0)
    if not snapshot_metrics and not outcome_metrics:
        return ""
    empty_snapshot_html = '<p class="aggregate-empty">No snapshot metrics available.</p>'
    outcome_html = ""
    if outcome_metrics and outcome_year:
        outcome_html = (
            '<div class="financial-subgroup">'
            f'<div class="financial-subtitle">Realized {outcome_year} outcome</div>'
            f'<div class="metric-chips">{_render_financial_metric_chips(outcome_metrics)}</div>'
            "</div>"
        )
    return (
        '<section class="financial year-financial">'
        '<div class="financial-head">'
        "<h3>Financial Context</h3>"
        "</div>"
        '<div class="financial-subgroup">'
        '<div class="financial-subtitle">Filing-year snapshot</div>'
        f'<div class="metric-chips">{_render_financial_metric_chips(snapshot_metrics) or empty_snapshot_html}</div>'
        "</div>"
        f"{outcome_html}"
        "</section>"
    )


def _render_aggregate_action_chips(top_actions: list[dict[str, object]], *, limit: int) -> str:
    return "".join(
        f'<span class="agg-chip">{_escape_html(str(item.get("action", "")))}'
        f'<span class="agg-chip-count">{int(item.get("count", 0))}</span></span>'
        for item in top_actions[:limit]
    )


def _render_aggregate_value_rows(rows: list[dict[str, object]], *, kind: str, limit: int) -> str:
    rendered: list[str] = []
    if kind == "edges":
        for item in rows[:limit]:
            rendered.append(
                f'<div class="agg-row"><span class="agg-flow">{_escape_html(item.get("src", ""))}'
                f' <span class="arrow">→</span> {_escape_html(item.get("dst", ""))}</span>'
                f'<span class="agg-count">{int(item.get("count", 0))}</span></div>'
            )
    else:
        for item in rows[:limit]:
            rendered.append(
                f'<div class="agg-row"><span class="agg-flow">{_escape_html(item.get("sequence", ""))}</span>'
                f'<span class="agg-count">{int(item.get("count", 0))}</span></div>'
            )
    return "".join(rendered)


def _render_aggregate_plan_card(aggregate: dict[str, object], *, inspect_href: str = "") -> str:
    top_actions = list(aggregate.get("top_actions", []))
    top_edges = list(aggregate.get("top_edges", []))
    top_sequences = list(aggregate.get("top_sequences", []))
    if not top_actions:
        return '<section class="aggregate"><h3>Aggregate Plan</h3><p class="aggregate-empty">No aggregate plan data available.</p></section>'
    inspect_button = ""
    if inspect_href:
        inspect_button = f'<a class="pill detail" href="{_escape_html(inspect_href)}">Compare Before/After</a>'
    empty_text = '<p class="aggregate-empty">None</p>'
    return (
        '<section class="aggregate">'
        '<div class="aggregate-head">'
        "<div>"
        "<h3>Aggregate Plan</h3>"
        f'<span class="aggregate-meta">{_escape_html(_format_year_span(list(aggregate.get("year_span", []))))} · '
        f'{int(aggregate.get("year_count", 0))} fiscal years · {int(aggregate.get("plan_count", 0)):,} plans</span>'
        "</div>"
        f"{inspect_button}"
        "</div>"
        f'<div class="plan-graph">{_workflow_graph_svg(top_actions, top_edges)}</div>'
        f'<div class="agg-chips">{_render_aggregate_action_chips(top_actions, limit=6)}</div>'
        '<div class="agg-grid">'
        '<div class="agg-block"><div class="agg-title">Top edges</div>'
        f'{_render_aggregate_value_rows(top_edges, kind="edges", limit=4) or empty_text}'
        "</div>"
        '<div class="agg-block"><div class="agg-title">Top sequences</div>'
        f'{_render_aggregate_value_rows(top_sequences, kind="sequences", limit=3) or empty_text}'
        "</div>"
        "</div>"
        "</section>"
    )


def _top_counter(counter: Counter, *, limit: int = 8, formatter=None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item, count in counter.most_common(limit):
        if formatter is None:
            rows.append({"item": item, "count": int(count)})
        else:
            rows.append(formatter(item, int(count)))
    return rows


def _workset_payload(index_row: dict[str, object], workset_dir: Path) -> dict[str, object]:
    rankings_payload = _load_json(workset_dir / "rocket_rankings.json")
    candidates_payload = _load_json(workset_dir / "candidate_workflows.json")
    artifacts_payload = _load_json(workset_dir / "basket_artifacts.json")
    chunks_payload = _load_json(workset_dir / "filing_chunks.json")
    psr_payload = _load_json(workset_dir / "psr_models.json")
    report_text = (workset_dir / "workflow_report.md").read_text(encoding="utf-8") if (workset_dir / "workflow_report.md").exists() else ""

    filings = list(index_row.get("filings") or [])
    rankings = list(rankings_payload.get("rankings") or [])
    candidates = list(candidates_payload.get("candidates") or [])
    artifacts = list(artifacts_payload.get("artifacts") or [])
    chunks = list(chunks_payload.get("chunks") or [])

    stage_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()
    sequence_counter: Counter[tuple[str, ...]] = Counter()
    section_counter: Counter[str] = Counter()
    chunk_counter: Counter[str] = Counter()
    filing_section_counts: list[tuple[str, float]] = []

    for artifact in artifacts:
        section_titles = [str(item) for item in artifact.get("section_titles") or [] if str(item).strip()]
        for section_title in section_titles:
            section_counter[section_title] += 1
        filing_section_counts.append((str(artifact.get("filing_title") or "filing"), float(len(section_titles))))

    for chunk in chunks:
        chunk_counter[str(chunk.get("filing_title") or "filing")] += 1

    for candidate in candidates:
        stages = [str(stage) for stage in candidate.get("workflow_stages") or [] if str(stage).strip()]
        stage_counter.update(stages)
        for left, right in zip(stages, stages[1:]):
            edge_counter[(left, right)] += 1
        if stages:
            sequence_counter[tuple(stages)] += 1

    aggregate_plan = {
        "top_actions": _top_counter(stage_counter, formatter=lambda action, count: {"action": str(action), "count": count}),
        "top_edges": _top_counter(edge_counter, formatter=lambda edge, count: {"src": str(edge[0]), "dst": str(edge[1]), "count": count}),
        "top_sequences": _top_counter(sequence_counter, limit=5, formatter=lambda sequence, count: {"sequence": _sequence_string(list(sequence)), "count": count}),
    }
    top_candidate = rankings[0] if rankings else None
    top_filing_chunks = chunk_counter.most_common(6)
    top_section_rows = section_counter.most_common(8)
    top_rankings = rankings[:8]

    if rankings:
        stage_counter = Counter()
        edge_counter = Counter()
        sequence_counter = Counter()
        for ranking in rankings:
            stages = [str(stage) for stage in ranking.get("workflow_stages") or [] if str(stage).strip()]
            stage_counter.update(stages)
            for left, right in zip(stages, stages[1:]):
                edge_counter[(left, right)] += 1
            if stages:
                sequence_counter[tuple(stages)] += 1
        aggregate_plan = {
            "top_actions": _top_counter(stage_counter, formatter=lambda action, count: {"action": str(action), "count": count}),
            "top_edges": _top_counter(edge_counter, formatter=lambda edge, count: {"src": str(edge[0]), "dst": str(edge[1]), "count": count}),
            "top_sequences": _top_counter(sequence_counter, limit=5, formatter=lambda sequence, count: {"sequence": _sequence_string(list(sequence)), "count": count}),
        }

    changed_count = sum(1 for ranking in rankings if bool(ranking.get("changed", False)))
    mean_score_gain = sum(float(ranking.get("score_gain") or 0.0) for ranking in rankings) / max(len(rankings), 1)
    reward_backend = str(rankings_payload.get("reward_backend") or (top_candidate.get("reward_backend") if top_candidate else "unknown"))
    top_evidence_spans = list(top_candidate.get("evidence_spans") or [])[:6] if top_candidate else []

    return {
        "workset_name": str(index_row.get("workset_name") or workset_dir.name),
        "company": str(index_row.get("company") or ""),
        "filing_year": str(index_row.get("filing_year") or ""),
        "form_type": str(index_row.get("form_type") or ""),
        "filing_count": int(index_row.get("filing_count") or len(filings)),
        "filings": filings,
        "chunk_count": len(chunks),
        "candidate_count": len(candidates),
        "ranking_count": len(rankings),
        "changed_count": changed_count,
        "mean_score_gain": mean_score_gain,
        "psr_status": str(psr_payload.get("status") or "missing"),
        "reward_backend": reward_backend,
        "latent_states": [str(item) for item in psr_payload.get("latent_states") or []],
        "top_candidate": top_candidate,
        "top_rankings": top_rankings,
        "top_evidence_spans": top_evidence_spans,
        "top_sections": [{"section": title, "count": count} for title, count in top_section_rows],
        "top_filing_chunks": [{"filing": title, "count": count} for title, count in top_filing_chunks],
        "aggregate_plan": aggregate_plan,
        "report_excerpt": "\n".join(line for line in report_text.splitlines()[:8] if line.strip()),
        "filing_section_counts": filing_section_counts[:8],
    }


def _workset_page_html(summary: dict[str, object], *, aggregate_html_name: str) -> str:
    top_candidate = summary["top_candidate"]
    ranking_cards = []
    for ranking in summary["top_rankings"]:
        gain = float(ranking.get("score_gain") or 0.0)
        base_flow = _sequence_string([str(item) for item in ranking.get("base_workflow_stages") or []])
        selected_flow = _sequence_string([str(item) for item in ranking.get("workflow_stages") or []])
        ranking_cards.append(
            f"""
      <article class="ranking-card">
        <div class="ranking-head">
          <h3>Rank {_escape_html(ranking['rank'])}: {_escape_html(ranking['candidate_id'])}</h3>
          <span class="pill">score {_escape_html(_format_score(float(ranking['score'])))}</span>
        </div>
        <p class="meta-line">{_escape_html(ranking['filing_title'])}</p>
        <p class="meta-line">{_escape_html(str(ranking.get('selected_source') or 'basket'))} · {_escape_html(str(ranking.get('selected_label') or 'base'))} · gain {_escape_html(_format_score(gain))}</p>
        <p class="meta-line">base { _escape_html(base_flow) }</p>
        <p class="flow">selected {_escape_html(selected_flow)}</p>
      </article>"""
        )
    if not ranking_cards:
        ranking_cards.append('<p class="empty-chart">No ROCKET rankings available.</p>')

    filing_cards = []
    for filing in summary["filings"][:8]:
        filing_cards.append(
            f"""
      <article class="filing-card">
        <h3>{_escape_html(filing.get('title', 'filing'))}</h3>
        <p class="meta-line">{_escape_html(filing.get('filing_date', ''))} · {_escape_html(filing.get('form_type', ''))}</p>
        <p class="meta-line">ticker {_escape_html(filing.get('ticker', ''))} · cik {_escape_html(filing.get('cik', ''))}</p>
      </article>"""
        )
    if not filing_cards:
        filing_cards.append('<p class="empty-chart">No filing rows available.</p>')

    stage_items = [(str(item["action"]), float(item["count"])) for item in summary["aggregate_plan"]["top_actions"]]
    section_items = [(str(item["section"])[:28], float(item["count"])) for item in summary["top_sections"]]
    chunk_items = [(str(item["filing"])[:28], float(item["count"])) for item in summary["top_filing_chunks"]]
    excerpt = summary["report_excerpt"] or "No workflow report excerpt available."
    evidence_cards = []
    for span in summary["top_evidence_spans"]:
        evidence_cards.append(
            f"""
      <article class="filing-card">
        <h3>{_escape_html(str(span.get('action') or 'evidence'))}</h3>
        <p class="meta-line">{_escape_html(str(span.get('section_title') or 'section'))} · {_escape_html(str(span.get('match_basis') or 'grounded'))}</p>
        <p class="meta-line">{_escape_html(str(span.get('snippet') or ''))}</p>
      </article>"""
        )
    if not evidence_cards:
        evidence_cards.append('<p class="empty-chart">No grounded evidence spans available.</p>')
    top_candidate_line = (
        _sequence_string([str(item) for item in top_candidate.get("workflow_stages")])
        if top_candidate
        else "No ranked candidate yet."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE BASKET/ROCKET Workset · {_escape_html(summary['workset_name'])}</title>
  <style>
    :root {{
      --bg: #f6f3ed; --panel: #fffdfa; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca;
      --accent: #0f766e; --accent2: #d97706; --accent3: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .wrap {{ max-width: 1260px; margin: 0 auto; padding: 28px 20px 56px; }}
    .hero {{ background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(15,118,110,0.92)); color: white; padding: 28px; border-radius: 22px; margin-bottom: 22px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 30px; }} .hero p {{ margin: 0; color: rgba(255,255,255,0.84); max-width: 80ch; }}
    a.nav {{ color: white; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0 24px; }}
    .stat, .section, .ranking-card, .filing-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 28px rgba(20,33,61,0.05); }}
    .stat {{ padding: 16px 18px; }} .section {{ padding: 18px; margin-bottom: 18px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }} .value {{ margin-top: 7px; font-size: 28px; font-weight: 700; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .chart-card {{ border: 1px solid var(--line); border-radius: 16px; background: white; padding: 14px; }}
    .chart-card h3 {{ margin: 0 0 10px; font-size: 15px; }} .chart-label, .chart-value {{ font-size: 11px; fill: #334155; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .ranking-grid, .filing-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .ranking-card, .filing-card {{ padding: 14px; }}
    .ranking-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
    .ranking-head h3, .filing-card h3 {{ margin: 0; font-size: 16px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 12px; background: #dcfce7; color: #166534; font-weight: 700; }}
    .meta-line {{ margin: 10px 0 0; color: var(--muted); }}
    .flow {{ margin: 12px 0 0; color: #334155; line-height: 1.45; font-weight: 600; }}
    pre.report {{ margin: 0; white-space: pre-wrap; line-height: 1.45; color: #334155; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .empty-chart {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>BAFFLE BASKET/ROCKET Workset · {_escape_html(summary['company'])} {_escape_html(summary['form_type'])} {_escape_html(summary['filing_year'])}</h1>
      <p>SEC filings grouped into one company/year/form workset, then scaffolded through chunking, BASKET artifact construction, ROCKET reranking, and PSR synthesis.</p>
      <p><a class="nav" href="index.html">Back to workset index</a> · <a class="nav" href="{_escape_html(aggregate_html_name)}">Aggregate plans</a></p>
    </section>
    <section class="stats">
      <div class="stat"><div class="label">Filings</div><div class="value">{summary['filing_count']}</div></div>
      <div class="stat"><div class="label">Chunks</div><div class="value">{summary['chunk_count']}</div></div>
      <div class="stat"><div class="label">Candidates</div><div class="value">{summary['candidate_count']}</div></div>
      <div class="stat"><div class="label">Rankings</div><div class="value">{summary['ranking_count']}</div></div>
      <div class="stat"><div class="label">Changed</div><div class="value">{summary['changed_count']}</div></div>
      <div class="stat"><div class="label">Mean Gain</div><div class="value">{_escape_html(_format_score(float(summary['mean_score_gain'])))}</div></div>
      <div class="stat"><div class="label">PSR Status</div><div class="value">{_escape_html(summary['psr_status'])}</div></div>
      <div class="stat"><div class="label">Reward Backend</div><div class="value">{_escape_html(summary['reward_backend'])}</div></div>
      <div class="stat"><div class="label">Top Candidate</div><div class="value">{_escape_html(top_candidate['rank']) if top_candidate else '-'}</div></div>
    </section>
    <section class="section">
      <h2>Workset Overview</h2>
      <div class="section-grid">
        <div class="chart-card">
          <h3>Workflow stage frequency</h3>
          {_inline_bar_chart(stage_items, bar_color="#0f766e")}
        </div>
        <div class="chart-card">
          <h3>Section titles surfaced by BASKET</h3>
          {_inline_bar_chart(section_items, bar_color="#1d4ed8")}
        </div>
        <div class="chart-card">
          <h3>Chunk counts by filing</h3>
          {_inline_bar_chart(chunk_items, bar_color="#d97706")}
        </div>
      </div>
    </section>
    <section class="section">
      <h2>Top ROCKET candidate</h2>
      <p class="flow">{_escape_html(top_candidate_line)}</p>
      <p class="meta-line">Base workflow: {_escape_html(_sequence_string([str(item) for item in top_candidate.get('base_workflow_stages') or []])) if top_candidate else 'none'}</p>
      <p class="meta-line">Source: {_escape_html(str(top_candidate.get('selected_source') or '')) if top_candidate else 'none'} · Label: {_escape_html(str(top_candidate.get('selected_label') or '')) if top_candidate else 'none'} · Candidate source: {_escape_html(str(top_candidate.get('candidate_source') or '')) if top_candidate else 'none'} · Gain: {_escape_html(_format_score(float(top_candidate.get('score_gain') or 0.0))) if top_candidate else '0.000'}</p>
      <p class="meta-line">Latent PSR states: {_escape_html(", ".join(summary['latent_states']) if summary['latent_states'] else 'none')}</p>
    </section>
    <section class="section">
      <h2>Grounded evidence spans</h2>
      <div class="filing-grid">
        {''.join(evidence_cards)}
      </div>
    </section>
    <section class="section">
      <h2>ROCKET rankings</h2>
      <div class="ranking-grid">
        {''.join(ranking_cards)}
      </div>
    </section>
    <section class="section">
      <h2>Staged filings</h2>
      <div class="filing-grid">
        {''.join(filing_cards)}
      </div>
    </section>
    <section class="section">
      <h2>Workflow report excerpt</h2>
      <pre class="report">{_escape_html(excerpt)}</pre>
    </section>
  </div>
</body>
</html>"""


def _aggregate_page_html(summary: dict[str, object], *, workset_html_name: str) -> str:
    aggregate = summary["aggregate_plan"]
    stage_rows = aggregate["top_actions"]
    sequence_rows = aggregate["top_sequences"]
    stage_items = [(str(item["action"]), float(item["count"])) for item in stage_rows]
    sequence_cards = []
    for row in sequence_rows:
        sequence_cards.append(
            f'<div class="agg-row"><span class="agg-flow">{_escape_html(row["sequence"])}</span><span class="agg-count">{row["count"]}</span></div>'
        )
    if not sequence_cards:
        sequence_cards.append('<p class="empty-chart">No candidate sequences available.</p>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE Aggregate Plans · {_escape_html(summary['workset_name'])}</title>
  <style>
    :root {{
      --bg: #f7f6f2; --panel: #fffdf8; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca;
      --accent: #1d4ed8; --accent2: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 56px; }}
    .hero {{ background: linear-gradient(135deg, rgba(20,33,61,0.98), rgba(29,78,216,0.92)); color: white; padding: 28px; border-radius: 24px; margin-bottom: 20px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 32px; }} .hero p {{ margin: 0; color: rgba(255,255,255,0.84); }}
    a.nav {{ color: white; }}
    .section {{ background: rgba(255,253,248,0.94); border: 1px solid var(--line); border-radius: 22px; padding: 18px; margin-bottom: 18px; box-shadow: 0 12px 34px rgba(20,33,61,0.05); }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
    .section h2 {{ margin: 0; font-size: 22px; }} .section-meta {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .plan-graph {{ border: 1px solid #dbe3ee; border-radius: 18px; background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,1)); padding: 10px; min-height: 160px; }}
    .plan-graph-svg {{ width: 100%; height: auto; display: block; }} .graph-node-label {{ font-size: 10px; font-weight: 700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .agg-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .agg-block {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px; background: white; }}
    .agg-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
    .agg-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px solid #f1f5f9; }}
    .agg-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .agg-flow {{ color: #1f2937; line-height: 1.35; font-size: 13px; }}
    .agg-count {{ color: #0f766e; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .empty-chart {{ color: var(--muted); font-style: italic; }}
    .chart-card {{ border: 1px solid var(--line); border-radius: 16px; background: white; padding: 14px; margin-top: 16px; }}
    .chart-card h3 {{ margin: 0 0 10px; font-size: 15px; }} .chart-label, .chart-value {{ font-size: 11px; fill: #334155; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>BAFFLE Aggregate Plans · {_escape_html(summary['company'])} {_escape_html(summary['form_type'])} {_escape_html(summary['filing_year'])}</h1>
      <p>Aggregate plan structure induced by the current candidate workflows for this BAFFLE workset.</p>
      <p><a class="nav" href="index.html">Back to workset index</a> · <a class="nav" href="{_escape_html(workset_html_name)}">Workset detail</a></p>
    </section>
    <section class="section">
      <div class="section-head">
        <h2>Aggregate plan graph</h2>
        <span class="section-meta">{summary['candidate_count']} candidates · {summary['ranking_count']} rankings</span>
      </div>
      <div class="plan-graph">{_workflow_graph_svg(aggregate['top_actions'], aggregate['top_edges'], width=760, height=240)}</div>
      <div class="agg-grid">
        <div class="agg-block">
          <div class="agg-title">Top workflow stages</div>
          {''.join(f'<div class="agg-row"><span>{_escape_html(item["action"])}</span><span class="agg-count">{item["count"]}</span></div>' for item in stage_rows)}
        </div>
        <div class="agg-block">
          <div class="agg-title">Top stage sequences</div>
          {''.join(sequence_cards)}
        </div>
      </div>
      <div class="chart-card">
        <h3>Workflow stage frequency</h3>
        {_inline_bar_chart(stage_items, bar_color="#1d4ed8")}
      </div>
    </section>
  </div>
</body>
</html>"""


def _index_html(company_rows: list[dict[str, object]], workset_rows: list[dict[str, object]]) -> str:
    company_cards = []
    for row in company_rows:
        financial_html = _render_financial_overview(dict(row.get("financial_summary") or {}), compact=True)
        aggregate_html = _render_aggregate_plan_card(
            dict(row.get("aggregate_plan", {})),
            inspect_href=str(row.get("aggregate_detail_html", "")),
        )
        company_cards.append(
            f"""
      <article class="card">
        <div class="head">
          <h2>{_escape_html(row['company'])}</h2>
          <a class="pill" href="{_escape_html(row['html'])}">Open Visualizer</a>
          <a class="pill alt" href="{_escape_html(row['aggregate_html'])}">Aggregate</a>
        </div>
        <div class="stats">
          <div class="stat"><span class="label">Rows</span><span class="value">{int(row['n_rows']):,}</span></div>
          <div class="stat"><span class="label">Changed</span><span class="value">{int(row['n_changed']):,}</span></div>
          <div class="stat"><span class="label">Changed Rate</span><span class="value">{_escape_html(_format_percent(float(row['changed_rate'])))}</span></div>
          <div class="stat"><span class="label">Mean Gain</span><span class="value">{_escape_html(_format_score(float(row['mean_score_gain'])))}</span></div>
        </div>
        <p class="mix"><strong>Selected sources:</strong> {_escape_html(row['source_mix'])}</p>
        <p class="mix"><strong>Top inserted actions:</strong> {_escape_html(row['inserted_mix'])}</p>
        {financial_html}
        {aggregate_html}
      </article>"""
        )

    workset_cards = []
    for row in workset_rows:
        workset_cards.append(
            f"""
      <article class="card">
        <div class="head">
          <h2>{_escape_html(row['company'])} {_escape_html(row['form_type'])} {_escape_html(row['filing_year'])}</h2>
          <a class="pill" href="{_escape_html(row['html'])}">Workset</a>
          <a class="pill alt" href="{_escape_html(row['aggregate_html'])}">Aggregate</a>
          <a class="pill year" href="{_escape_html(row['year_html'])}">Year View</a>
        </div>
        <div class="stats">
          <div class="stat"><span class="label">Filings</span><span class="value">{row['filing_count']}</span></div>
          <div class="stat"><span class="label">Chunks</span><span class="value">{row['chunk_count']}</span></div>
          <div class="stat"><span class="label">Candidates</span><span class="value">{row['candidate_count']}</span></div>
          <div class="stat"><span class="label">PSR</span><span class="value">{_escape_html(row['psr_status'])}</span></div>
        </div>
        <p class="mix"><strong>Role:</strong> {_escape_html(row['semantic_role'])}</p>
        <p class="mix"><strong>Top candidate:</strong> {_escape_html(row['top_candidate'])}</p>
        <div class="aggregate">
          <h3>Aggregate workflow graph</h3>
          <div class="plan-graph">{_workflow_graph_svg(row['aggregate_plan']['top_actions'], row['aggregate_plan']['top_edges'])}</div>
        </div>
      </article>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE BASKET/ROCKET Visualizers</title>
  <style>
    :root {{
      --bg: #f7f6f2; --panel: #fffdf8; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca; --accent: #0f766e; --accent2: #1d4ed8; --accent3: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29,78,216,0.09), transparent 28rem),
        radial-gradient(circle at top right, rgba(15,118,110,0.09), transparent 22rem),
        var(--bg);
    }}
    .wrap {{ max-width: 1260px; margin: 0 auto; padding: 30px 20px 54px; }}
    .hero {{ background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(15,118,110,0.92)); color: white; padding: 28px; border-radius: 22px; box-shadow: 0 20px 50px rgba(20,33,61,0.18); margin-bottom: 22px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 30px; line-height: 1.1; }} .hero p {{ margin: 0; color: rgba(255,255,255,0.82); max-width: 80ch; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px; box-shadow: 0 12px 34px rgba(20,33,61,0.05); }}
    .head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }} .head h2 {{ margin: 0; font-size: 22px; flex: 1 1 auto; }}
    .pill {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 8px 12px; text-decoration: none; color: white; background: var(--accent); font-weight: 700; font-size: 13px; }}
    .pill.alt {{ background: var(--accent2); }}
    .pill.year {{ background: var(--accent3); }}
    .pill.detail {{ background: var(--accent); }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 16px; padding: 10px 12px; background: white; }}
    .label {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
    .value {{ font-size: 22px; font-weight: 700; }} .mix {{ margin: 10px 0 0; color: #334155; line-height: 1.4; }}
    .financial {{
      margin-top: 14px; border: 1px solid #dbe3ee; border-radius: 16px; background: rgba(255,255,255,0.92); padding: 12px;
    }}
    .financial-compact {{ margin-bottom: 0; }}
    .financial-head {{ display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
    .financial h3 {{ margin: 0; font-size: 15px; }}
    .financial-meta, .financial-subtitle {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .financial-subgroup + .financial-subgroup {{ margin-top: 10px; }}
    .financial-subtitle {{ margin-bottom: 8px; }}
    .metric-chips, .agg-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric-chip, .agg-chip {{
      display: inline-flex; align-items: baseline; gap: 8px; border-radius: 999px; border: 1px solid #dbe3ee; background: #f8fafc; padding: 6px 10px;
    }}
    .metric-label {{ color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }}
    .metric-value {{ color: #0f172a; font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .aggregate {{ margin-top: 16px; border-top: 1px solid #e7e2d7; padding-top: 16px; }} .aggregate h3 {{ margin: 0 0 10px; font-size: 17px; }}
    .aggregate-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 12px; }}
    .aggregate-meta {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }}
    .plan-graph {{ border: 1px solid #dbe3ee; border-radius: 16px; background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,1)); padding: 8px; min-height: 120px; }}
    .plan-graph-svg {{ width: 100%; height: auto; display: block; }} .graph-node-label {{ font-size: 10px; font-weight: 700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .agg-chip-count {{ border-radius: 999px; background: #e2e8f0; padding: 2px 7px; color: #334155; font-size: 11px; }}
    .agg-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .agg-block {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px; background: white; }}
    .agg-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
    .agg-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 5px 0; border-bottom: 1px solid #f1f5f9; }}
    .agg-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .agg-flow {{ color: #1f2937; line-height: 1.35; font-size: 13px; }}
    .arrow {{ color: #64748b; font-weight: 700; }}
    .agg-count {{ color: #0f766e; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>BAFFLE BASKET/ROCKET Visualizers</h1>
      <p>Visualization surfaces generated from SEC-backed company/year/form worksets, including company pages that now track much more closely to the older ROCKET company visualizers.</p>
    </section>
    <section class="hero" style="background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(29,78,216,0.92)); margin-bottom: 18px;">
      <h1>Company ROCKET Views</h1>
      <p>These pages aggregate selected and base workflow edits across all BAFFLE worksets for each company, with financial context and before/after aggregate comparisons in the style of the earlier `rocket_company_viz_*` outputs.</p>
    </section>
    <section class="grid">
      {''.join(company_cards) if company_cards else '<p>No company-level visualizations available.</p>'}
    </section>
    <section class="hero" style="background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(15,118,110,0.92)); margin-top: 22px; margin-bottom: 18px;">
      <h1>Workset Views</h1>
      <p>These pages keep the BAFFLE-specific company/year/form drilldown and the 10-K anchor plus 8-K event-patch structure.</p>
    </section>
    <section class="grid">
      {''.join(workset_cards) if workset_cards else '<p>No workset visualizations available.</p>'}
    </section>
  </div>
</body>
</html>"""


def _year_context_payload(workset_rows: list[dict[str, object]]) -> dict[str, object]:
    anchor_rows = [row for row in workset_rows if row.get("semantic_role") == "annual_anchor"]
    event_rows = [row for row in workset_rows if row.get("semantic_role") == "event_patch"]
    quarterly_rows = [row for row in workset_rows if row.get("semantic_role") == "quarterly_update"]
    event_item_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()
    event_timeline = []
    for row in event_rows:
        for filing in row.get("filings") or []:
            filing_date = str(filing.get("filing_date") or "")
            codes = [str(code) for code in filing.get("event_item_codes") or [] if str(code).strip()]
            event_item_counter.update(codes)
            event_timeline.append(
                {
                    "title": str(filing.get("title") or ""),
                    "filing_date": filing_date,
                    "item_codes": codes,
                    "form_type": str(filing.get("form_type") or ""),
                    "workset_name": str(row.get("workset_name") or ""),
                    "html": str(row.get("html") or ""),
                }
            )
        for action in (row.get("aggregate_plan") or {}).get("top_actions") or []:
            stage_counter[str(action.get("action") or "")] += int(action.get("count") or 0)
    for row in anchor_rows + quarterly_rows:
        for action in (row.get("aggregate_plan") or {}).get("top_actions") or []:
            stage_counter[str(action.get("action") or "")] += int(action.get("count") or 0)

    event_timeline.sort(key=lambda item: (item["filing_date"], item["title"]))
    anchor = anchor_rows[0] if anchor_rows else None
    return {
        "company": str(workset_rows[0].get("company") or "") if workset_rows else "",
        "anchor_year": str(workset_rows[0].get("filing_year") or "") if workset_rows else "",
        "annual_anchor_count": len(anchor_rows),
        "event_patch_count": len(event_rows),
        "quarterly_update_count": len(quarterly_rows),
        "anchor_workset": anchor,
        "event_worksets": event_rows,
        "quarterly_worksets": quarterly_rows,
        "event_timeline": event_timeline,
        "top_event_items": [{"item_code": code, "count": count} for code, count in event_item_counter.most_common(8)],
        "aggregate_plan": {
            "top_actions": _top_counter(stage_counter, formatter=lambda action, count: {"action": str(action), "count": count}),
            "top_edges": [],
            "top_sequences": [],
        },
    }


def _year_context_page_html(summary: dict[str, object]) -> str:
    anchor = summary["anchor_workset"]
    timeline_cards = []
    for event in summary["event_timeline"][:12]:
        timeline_cards.append(
            f"""
      <article class="event-card">
        <div class="event-head">
          <h3>{_escape_html(event['filing_date'])}</h3>
          <a class="pill alt" href="{_escape_html(event['html'])}">Workset</a>
        </div>
        <p class="meta-line">{_escape_html(event['title'])}</p>
        <p class="flow">Items: {_escape_html(', '.join(event['item_codes']) if event['item_codes'] else 'none extracted')}</p>
      </article>"""
        )
    if not timeline_cards:
        timeline_cards.append('<p class="empty-chart">No 8-K event patches are attached to this year yet.</p>')

    top_event_items = [(str(item["item_code"]), float(item["count"])) for item in summary["top_event_items"]]
    stage_items = [(str(item["action"]), float(item["count"])) for item in summary["aggregate_plan"]["top_actions"]]
    anchor_line = (
        f"{anchor['form_type']} · {anchor['filing_count']} filing(s) · top candidate: {anchor['top_candidate']}"
        if anchor
        else "No annual 10-K anchor is present for this year context."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE Year Backbone · {_escape_html(summary['company'])} {_escape_html(summary['anchor_year'])}</title>
  <style>
    :root {{
      --bg: #f6f3ed; --panel: #fffdfa; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca;
      --accent: #1d4ed8; --accent2: #d97706; --accent3: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .wrap {{ max-width: 1260px; margin: 0 auto; padding: 28px 20px 56px; }}
    .hero {{ background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(29,78,216,0.92)); color: white; padding: 28px; border-radius: 22px; margin-bottom: 22px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 30px; }} .hero p {{ margin: 0; color: rgba(255,255,255,0.84); max-width: 80ch; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0 24px; }}
    .stat, .section, .event-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 28px rgba(20,33,61,0.05); }}
    .stat {{ padding: 16px 18px; }} .section {{ padding: 18px; margin-bottom: 18px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }} .value {{ margin-top: 7px; font-size: 28px; font-weight: 700; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .chart-card {{ border: 1px solid var(--line); border-radius: 16px; background: white; padding: 14px; }}
    .chart-card h3 {{ margin: 0 0 10px; font-size: 15px; }} .chart-label, .chart-value {{ font-size: 11px; fill: #334155; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .timeline-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .event-card {{ padding: 14px; }}
    .event-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
    .event-head h3 {{ margin: 0; font-size: 16px; }}
    .pill {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 8px 12px; text-decoration: none; color: white; background: var(--accent3); font-weight: 700; font-size: 13px; }}
    .pill.alt {{ background: var(--accent2); }}
    .meta-line {{ margin: 10px 0 0; color: var(--muted); }}
    .flow {{ margin: 12px 0 0; color: #334155; line-height: 1.45; font-weight: 600; }}
    .empty-chart {{ color: var(--muted); font-style: italic; }}
    a.nav {{ color: white; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>BAFFLE Year Backbone · {_escape_html(summary['company'])} {_escape_html(summary['anchor_year'])}</h1>
      <p>This view treats the 10-K as the yearly baseline anchor and the 8-Ks as event patches attached to that annual state.</p>
      <p><a class="nav" href="index.html">Back to visualization index</a></p>
    </section>
    <section class="stats">
      <div class="stat"><div class="label">Annual Anchors</div><div class="value">{summary['annual_anchor_count']}</div></div>
      <div class="stat"><div class="label">Event Patches</div><div class="value">{summary['event_patch_count']}</div></div>
      <div class="stat"><div class="label">Quarterly Updates</div><div class="value">{summary['quarterly_update_count']}</div></div>
    </section>
    <section class="section">
      <h2>Backbone summary</h2>
      <p class="flow">{_escape_html(anchor_line)}</p>
    </section>
    <section class="section">
      <h2>Year context overview</h2>
      <div class="section-grid">
        <div class="chart-card">
          <h3>Event item codes</h3>
          {_inline_bar_chart(top_event_items, bar_color="#d97706")}
        </div>
        <div class="chart-card">
          <h3>Combined workflow stages</h3>
          {_inline_bar_chart(stage_items, bar_color="#1d4ed8")}
        </div>
      </div>
    </section>
    <section class="section">
      <h2>8-K event timeline</h2>
      <div class="timeline-grid">
        {''.join(timeline_cards)}
      </div>
    </section>
  </div>
</body>
</html>"""


def _aggregate_payload_from_rows(rows: list[dict[str, object]], *, action_key: str) -> dict[str, object]:
    action_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()
    sequence_counter: Counter[tuple[str, ...]] = Counter()
    years = sorted({int(row["year"]) for row in rows if str(row.get("year", "")).isdigit()})
    for row in rows:
        actions = [str(action) for action in row.get(action_key) or [] if str(action).strip()]
        action_counter.update(actions)
        for left, right in zip(actions, actions[1:]):
            edge_counter[(left, right)] += 1
        if actions:
            sequence_counter[tuple(actions)] += 1
    return {
        "statement_count": len(rows),
        "plan_count": len(rows),
        "year_count": len(years),
        "year_span": [years[0], years[-1]] if years else [],
        "top_actions": _top_counter(action_counter, formatter=lambda action, count: {"action": str(action), "count": count}),
        "top_edges": _top_counter(edge_counter, formatter=lambda edge, count: {"src": str(edge[0]), "dst": str(edge[1]), "count": count}),
        "top_sequences": _top_counter(sequence_counter, limit=5, formatter=lambda sequence, count: {"sequence": _sequence_string(list(sequence)), "count": count}),
    }


def _company_payload(
    company: str,
    rows: list[dict[str, object]],
    *,
    ticker: str = "",
    reward_backend: str = "",
    financial_summary: dict[str, object] | None = None,
    financial_by_year: dict[int, dict[str, object]] | None = None,
) -> dict[str, object]:
    changed_rows = [row for row in rows if bool(row.get("changed", False))]
    changed_by_year: Counter[int] = Counter()
    selected_labels: Counter[str] = Counter()
    selected_sources: Counter[str] = Counter()
    inserted_actions: Counter[str] = Counter()
    rows_by_year: dict[int, list[dict[str, object]]] = {}

    for row in rows:
        year = int(row["year"])
        rows_by_year.setdefault(year, []).append(row)
        if bool(row.get("changed", False)):
            changed_by_year[year] += 1
            label = str(row.get("selected_label") or "")
            source = str(row.get("selected_source") or "")
            if label:
                selected_labels[label] += 1
            if source:
                selected_sources[source] += 1
            base_set = set(str(action) for action in row.get("base_actions") or [])
            for action in row.get("selected_actions") or []:
                action_str = str(action)
                if action_str and action_str not in base_set:
                    inserted_actions[action_str] += 1

    top_examples = sorted(changed_rows, key=lambda row: (-float(row.get("score_gain") or 0.0), str(row.get("candidate_id") or "")))[:8]
    selected_by_year = []
    base_by_year = []
    for year in sorted(rows_by_year, reverse=True):
        year_rows = rows_by_year[year]
        selected_by_year.append(
            {
                "year": year,
                "statement_count": len(year_rows),
                "changed_count": sum(1 for row in year_rows if bool(row.get("changed", False))),
                "aggregate_plan": _aggregate_payload_from_rows(year_rows, action_key="selected_actions"),
            }
        )
        base_by_year.append(
            {
                "year": year,
                "statement_count": len(year_rows),
                "changed_count": sum(1 for row in year_rows if bool(row.get("changed", False))),
                "aggregate_plan": _aggregate_payload_from_rows(year_rows, action_key="base_actions"),
            }
        )

    return {
        "company": company,
        "ticker": ticker,
        "reward_backend": reward_backend,
        "n_rows": len(rows),
        "n_changed": len(changed_rows),
        "n_unique_selected_plans": len({tuple(str(action) for action in row.get("selected_actions") or []) for row in rows if row.get("selected_actions")}),
        "changed_rate": float(len(changed_rows)) / float(max(len(rows), 1)),
        "mean_score_gain": sum(float(row.get("score_gain") or 0.0) for row in rows) / max(len(rows), 1),
        "max_score_gain": max((float(row.get("score_gain") or 0.0) for row in rows), default=0.0),
        "changed_by_company": {company: len(changed_rows)},
        "changed_by_year": {str(year): int(count) for year, count in sorted(changed_by_year.items())},
        "selected_labels": dict(selected_labels.most_common()),
        "selected_sources": dict(selected_sources.most_common()),
        "inserted_actions": dict(inserted_actions.most_common()),
        "top_examples": top_examples,
        "top_changed_examples": top_examples,
        "aggregate_plan_selected": _aggregate_payload_from_rows(rows, action_key="selected_actions"),
        "aggregate_plan_base": _aggregate_payload_from_rows(rows, action_key="base_actions"),
        "aggregate_plan_by_year": selected_by_year,
        "aggregate_plan_selected_by_year": selected_by_year,
        "aggregate_plan_base_by_year": base_by_year,
        "financial_summary": financial_summary or {},
        "financial_by_year": financial_by_year or {},
    }


def _company_reranking_page_html(summary: dict[str, object]) -> str:
    company_slug = _slugify(str(summary["company"]))
    changed_by_year_items = [(year, float(count)) for year, count in summary["changed_by_year"].items()]
    selected_label_items = [(label, float(count)) for label, count in summary["selected_labels"].items()]
    selected_source_items = [(label, float(count)) for label, count in summary["selected_sources"].items()]
    inserted_items = [(action, float(count)) for action, count in summary["inserted_actions"].items()]
    example_cards = []
    for example in summary["top_examples"]:
        example_cards.append(
            f"""
      <article class="example-card">
        <div class="example-head">
          <h3>{_escape_html(example['candidate_id'])}</h3>
          <span class="gain-pill">gain {_escape_html(_format_score(float(example['score_gain'])))}</span>
        </div>
        <p class="meta-line">year {_escape_html(example['year'])} · form {_escape_html(example['form_type'])} · source {_escape_html(example['selected_source'])} · label {_escape_html(example['selected_label'])}</p>
        <p class="statement-snippet">{_escape_html(example['filing_title'])}</p>
        <div class="plan-grid">
          <section class="plan-panel">
            <div class="plan-title">Base workflow</div>
            <div class="plan-flow">{_escape_html(_sequence_string([str(item) for item in example['base_actions']]))}</div>
          </section>
          <section class="plan-panel selected">
            <div class="plan-title">Selected workflow</div>
            <div class="plan-flow">{_escape_html(_sequence_string([str(item) for item in example['selected_actions']]))}</div>
          </section>
        </div>
      </article>"""
        )
    if not example_cards:
        example_cards.append('<p class="empty-chart">No changed examples for this company.</p>')

    reward_backend = str(summary.get("reward_backend") or "unknown")
    financial_html = _render_financial_overview(dict(summary.get("financial_summary") or {}))
    aggregate_html = _render_aggregate_plan_card(
        dict(summary.get("aggregate_plan_selected") or {}),
        inspect_href=f"rocket_aggregate_plans___{company_slug}.html",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROCKET Reranking Visualizer · {_escape_html(summary['company'])}</title>
  <style>
    :root {{
      --bg: #f7f6f2; --panel: #fffdf8; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca;
      --accent: #1d4ed8; --teal: #0f766e; --warm: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29,78,216,0.09), transparent 28rem),
        radial-gradient(circle at top right, rgba(15,118,110,0.09), transparent 22rem),
        var(--bg);
    }}
    .wrap {{ max-width: 1220px; margin: 0 auto; padding: 30px 20px 54px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(20,33,61,0.97), rgba(29,78,216,0.92));
      color: white;
      padding: 28px;
      border-radius: 22px;
      box-shadow: 0 20px 50px rgba(20,33,61,0.18);
      margin-bottom: 22px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,0.82); max-width: 76ch; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0 26px; }}
    .stat, .section, .example-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 28px rgba(20,33,61,0.05); }}
    .stat {{ padding: 16px 18px; }} .section {{ padding: 18px; margin-bottom: 18px; }} .example-card {{ padding: 16px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }} .value {{ margin-top: 7px; font-size: 28px; font-weight: 700; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .chart-card {{ border: 1px solid var(--line); border-radius: 16px; background: white; padding: 14px; }}
    .chart-card h3 {{ margin: 0 0 10px; font-size: 15px; }} .chart-label, .chart-value {{ font-size: 11px; fill: #334155; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .financial {{
      border: 1px solid #dbe3ee;
      border-radius: 18px;
      background: rgba(255,255,255,0.92);
      padding: 14px;
      margin-bottom: 14px;
    }}
    .financial-head {{ display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
    .financial h3 {{ margin: 0; font-size: 16px; }}
    .financial-meta, .financial-subtitle {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .financial-subgroup + .financial-subgroup {{ margin-top: 12px; }}
    .financial-subtitle {{ margin-bottom: 8px; }}
    .metric-chips, .agg-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric-chip, .agg-chip {{
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      border-radius: 999px;
      border: 1px solid #dbe3ee;
      background: #f8fafc;
      padding: 6px 10px;
      font-weight: 700;
    }}
    .metric-label {{ color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .metric-value {{ color: #0f172a; font-size: 12px; font-variant-numeric: tabular-nums; }}
    .aggregate {{ margin-top: 16px; border-top: 1px solid #e7e2d7; padding-top: 16px; }}
    .aggregate-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 12px; }}
    .aggregate h3 {{ margin: 0; font-size: 17px; }}
    .aggregate-meta {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }}
    .plan-graph {{ border: 1px solid #dbe3ee; border-radius: 16px; background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,1)); padding: 8px; min-height: 120px; }}
    .plan-graph-svg {{ width: 100%; height: auto; display: block; }}
    .graph-node-label {{ font-size: 10px; font-weight: 700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .agg-chip-count {{ border-radius: 999px; background: #e2e8f0; padding: 2px 7px; color: #334155; font-size: 11px; }}
    .agg-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .agg-block {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px; background: white; }}
    .agg-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
    .agg-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 5px 0; border-bottom: 1px solid #f1f5f9; }}
    .agg-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .agg-flow {{ color: #1f2937; line-height: 1.35; font-size: 13px; }}
    .arrow {{ color: #64748b; font-weight: 700; }}
    .agg-count {{ color: #0f766e; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 8px 12px;
      text-decoration: none;
      color: white;
      background: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }}
    .pill.detail {{ background: var(--teal); }}
    .examples-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 12px; }} .examples-head p {{ margin: 0; color: var(--muted); }}
    .example-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }} .example-head h3 {{ margin: 0; font-size: 16px; }}
    .gain-pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 12px; background: #dbeafe; color: #1d4ed8; font-weight: 700; }}
    .meta-line {{ margin: 10px 0 0; color: var(--muted); }} .statement-snippet {{ margin: 12px 0 14px; color: #334155; line-height: 1.45; }}
    .plan-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }} .plan-panel {{ border: 1px solid var(--line); border-radius: 16px; padding: 14px; background: #fcfcfb; }}
    .plan-panel.selected {{ border-color: #bfdbfe; background: linear-gradient(180deg, #f8fbff, #ffffff); }} .plan-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px; }}
    .plan-flow {{ color: #1f2937; line-height: 1.45; font-weight: 600; }}
    .empty-chart {{ color: var(--muted); font-style: italic; }}
    a.nav {{ color: white; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>ROCKET Reranking Visualizer · {_escape_html(summary['company'])}</h1>
      <p>Company-specific visualizer for the SEC-backed FF2 run, using the same panel-driven financial objective family as the earlier ROCKET company views whenever the legacy reward path is available.</p>
      <p><a class="nav" href="index.html">Back to visualization index</a> · <a class="nav" href="rocket_aggregate_plans___{_escape_html(company_slug)}.html">Aggregate plans</a></p>
    </section>
    <section class="stats">
      <div class="stat"><div class="label">Workflows</div><div class="value">{summary['n_rows']}</div></div>
      <div class="stat"><div class="label">Changed</div><div class="value">{summary['n_changed']}</div></div>
      <div class="stat"><div class="label">Changed Rate</div><div class="value">{_escape_html(_format_percent(float(summary['changed_rate'])))}</div></div>
      <div class="stat"><div class="label">Mean Gain</div><div class="value">{_escape_html(_format_score(float(summary['mean_score_gain'])))}</div></div>
      <div class="stat"><div class="label">Unique Selected Plans</div><div class="value">{summary['n_unique_selected_plans']}</div></div>
      <div class="stat"><div class="label">Max Gain</div><div class="value">{_escape_html(_format_score(float(summary['max_score_gain'])))}</div></div>
      <div class="stat"><div class="label">Ticker</div><div class="value">{_escape_html(summary.get('ticker') or '-')}</div></div>
      <div class="stat"><div class="label">Reward Backend</div><div class="value">{_escape_html(reward_backend)}</div></div>
    </section>
    <section class="section">
      <h2>Reranking Overview</h2>
      {financial_html}
      {aggregate_html}
      <div class="section-grid">
        <div class="chart-card">
          <h3>Changed workflows by year</h3>
          {_inline_bar_chart(changed_by_year_items, bar_color="#1d4ed8")}
        </div>
        <div class="chart-card">
          <h3>Selected labels on changed rows</h3>
          {_inline_bar_chart(selected_label_items[:8], bar_color="#0f766e")}
        </div>
        <div class="chart-card">
          <h3>Selected sources on changed rows</h3>
          {_inline_bar_chart(selected_source_items[:8], bar_color="#2563eb")}
        </div>
        <div class="chart-card">
          <h3>Inserted actions on changed rows</h3>
          {_inline_bar_chart(inserted_items[:8], bar_color="#f59e0b")}
        </div>
      </div>
    </section>
    <section class="section">
      <div class="examples-head">
        <h2>Top changed examples</h2>
        <p>Ordered by score gain.</p>
      </div>
      {''.join(example_cards)}
    </section>
  </div>
</body>
</html>"""


def _company_aggregate_page_html(summary: dict[str, object]) -> str:
    company_slug = _slugify(str(summary["company"]))
    selected = summary["aggregate_plan_selected"]
    base = summary["aggregate_plan_base"]
    financial_summary = dict(summary.get("financial_summary") or {})
    financial_by_year = {
        int(year): dict(payload)
        for year, payload in (summary.get("financial_by_year") or {}).items()
        if str(year).isdigit()
    }

    def year_cards_html(year_rows: list[dict[str, object]]) -> str:
        cards = []
        for year_summary in year_rows:
            aggregate = dict(year_summary.get("aggregate_plan") or {})
            year = int(year_summary.get("year") or 0)
            cards.append(
                f"""
      <article class="year-card">
        <div class="year-head">
          <h2>{_escape_html(year)}</h2>
          <span class="year-meta">{int(year_summary.get('statement_count', 0)):,} workflows · {int(year_summary.get('changed_count', 0)):,} changed</span>
        </div>
        {_render_year_financial_panel(financial_by_year.get(year, {}))}
        <div class="plan-graph large">{_workflow_graph_svg(aggregate.get('top_actions', []), aggregate.get('top_edges', []), width=760, height=240)}</div>
        <div class="agg-chips">{_render_aggregate_action_chips(list(aggregate.get('top_actions', [])), limit=8)}</div>
        <div class="agg-grid detail-grid">
          <div class="agg-block">
            <div class="agg-title">Top edges</div>
            {_render_aggregate_value_rows(list(aggregate.get('top_edges', [])), kind="edges", limit=6) or '<p class="aggregate-empty">None</p>'}
          </div>
          <div class="agg-block">
            <div class="agg-title">Top sequences</div>
            {_render_aggregate_value_rows(list(aggregate.get('top_sequences', [])), kind="sequences", limit=5) or '<p class="aggregate-empty">None</p>'}
          </div>
        </div>
      </article>"""
            )
        return "".join(cards) if cards else '<p class="aggregate-empty">No yearly aggregate plans available.</p>'

    def mode_panel_html(
        *,
        mode_id: str,
        mode_label: str,
        aggregate: dict[str, object],
        year_rows: list[dict[str, object]],
        default_visible: bool,
    ) -> str:
        hidden_attr = "" if default_visible else " hidden"
        empty_text = '<p class="aggregate-empty">None</p>'
        return (
            f'<section class="mode-panel" data-mode-panel="{mode_id}"{hidden_attr}>'
            '<section class="section">'
            '<div class="section-head">'
            f'<h2>{_escape_html(mode_label)}</h2>'
            '<span class="section-meta">Company-wide aggregate</span>'
            '</div>'
            f'{_render_financial_overview(financial_summary)}'
            '<div class="stats compact">'
            f'<div class="stat"><span class="label">Year Span</span><span class="value">{_escape_html(_format_year_span(list(aggregate.get("year_span", []))))}</span></div>'
            f'<div class="stat"><span class="label">Fiscal Years</span><span class="value">{int(aggregate.get("year_count", 0))}</span></div>'
            f'<div class="stat"><span class="label">Plans</span><span class="value">{int(aggregate.get("plan_count", 0)):,}</span></div>'
            f'<div class="stat"><span class="label">Statements</span><span class="value">{int(aggregate.get("statement_count", 0)):,}</span></div>'
            '</div>'
            f'<div class="plan-graph large">{_workflow_graph_svg(list(aggregate.get("top_actions", [])), list(aggregate.get("top_edges", [])), width=760, height=240)}</div>'
            f'<div class="agg-chips">{_render_aggregate_action_chips(list(aggregate.get("top_actions", [])), limit=8)}</div>'
            '<div class="agg-grid detail-grid">'
            f'<div class="agg-block"><div class="agg-title">Top edges</div>{_render_aggregate_value_rows(list(aggregate.get("top_edges", [])), kind="edges", limit=8) or empty_text}</div>'
            f'<div class="agg-block"><div class="agg-title">Top sequences</div>{_render_aggregate_value_rows(list(aggregate.get("top_sequences", [])), kind="sequences", limit=6) or empty_text}</div>'
            '</div>'
            '</section>'
            '<section class="section">'
            '<div class="section-head">'
            '<h2>Year-By-Year Aggregate Plans</h2>'
            '<span class="section-meta">Latest year first</span>'
            '</div>'
            f'<div class="year-grid">{year_cards_html(year_rows)}</div>'
            '</section>'
            '</section>'
        )

    selected_year_rows = list(summary.get("aggregate_plan_selected_by_year") or [])
    base_year_rows = list(summary.get("aggregate_plan_base_by_year") or [])
    selected_year_span = _format_year_span(list(selected.get("year_span", [])))
    base_year_span = _format_year_span(list(base.get("year_span", [])))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROCKET Aggregate Plans · {_escape_html(summary['company'])}</title>
  <style>
    :root {{
      --bg: #f6f3ed; --panel: #fffdfa; --panel-strong: #ffffff; --ink: #14213d; --muted: #5f6c7b; --line: #d9d5ca; --accent: #1d4ed8; --accent2: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(29,78,216,0.09), transparent 28rem),
        radial-gradient(circle at top right, rgba(15,118,110,0.10), transparent 24rem),
        var(--bg);
    }}
    .wrap {{ max-width: 1380px; margin: 0 auto; padding: 28px 20px 52px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(20,33,61,0.98), rgba(29,78,216,0.92));
      color: white;
      padding: 28px;
      border-radius: 24px;
      box-shadow: 0 24px 55px rgba(20,33,61,0.20);
      margin-bottom: 20px;
    }}
    .hero-top {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 32px; }} .hero p {{ margin: 0; color: rgba(255,255,255,0.84); max-width: 82ch; }}
    .nav {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{
      display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 9px 14px;
      text-decoration: none; color: white; background: var(--accent); font-weight: 700; font-size: 13px; border: 0;
    }}
    .pill.alt {{ background: rgba(255,255,255,0.18); }}
    .section, .year-card {{ background: rgba(255,253,248,0.92); border: 1px solid var(--line); border-radius: 22px; padding: 18px; margin-bottom: 18px; box-shadow: 0 12px 34px rgba(20,33,61,0.05); }}
    .section-head, .year-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
    .section h2, .year-head h2 {{ margin: 0; font-size: 22px; }} .section-meta, .year-meta {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }}
    .stats.compact {{ margin-top: 0; margin-bottom: 14px; }}
    .stat {{ border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; background: var(--panel-strong); }}
    .label {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .plan-graph {{ border: 1px solid #dbe3ee; border-radius: 18px; background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,1)); padding: 10px; min-height: 160px; }}
    .plan-graph.large {{ min-height: 220px; }}
    .plan-graph-svg {{ width: 100%; height: auto; display: block; }} .graph-node-label {{ font-size: 10px; font-weight: 700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .agg-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .agg-chip {{
      display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; border: 1px solid #cbd5e1;
      background: #f8fafc; padding: 7px 11px; font-size: 12px; font-weight: 700; color: #0f172a;
    }}
    .agg-chip-count {{ border-radius: 999px; background: #e2e8f0; padding: 2px 7px; color: #334155; font-size: 11px; }}
    .agg-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .detail-grid {{ margin-top: 14px; }}
    .agg-block {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px; background: var(--panel-strong); }}
    .agg-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }}
    .agg-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 6px 0; border-bottom: 1px solid #f1f5f9; }}
    .agg-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .agg-flow {{ color: #1f2937; line-height: 1.35; font-size: 13px; }}
    .arrow {{ color: #64748b; font-weight: 700; }}
    .agg-count {{ color: #0f766e; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .year-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .financial {{
      border: 1px solid #dbe3ee; border-radius: 18px; background: rgba(255,255,255,0.92); padding: 14px; margin-bottom: 14px;
    }}
    .year-financial {{ margin-bottom: 12px; }}
    .financial-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
    .financial h3 {{ margin: 0; font-size: 16px; }}
    .financial-meta, .financial-subtitle {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .financial-subtitle {{ margin-bottom: 8px; }}
    .financial-subgroup + .financial-subgroup {{ margin-top: 12px; }}
    .metric-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .metric-chip {{
      display: inline-flex; align-items: baseline; gap: 8px; border-radius: 999px; border: 1px solid #dbe3ee; background: #f8fafc; padding: 6px 10px;
    }}
    .metric-label {{ color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }}
    .metric-value {{ color: #0f172a; font-size: 13px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .toggle-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    .toggle-btn {{
      appearance: none; border: 1px solid rgba(255,255,255,0.28); background: rgba(255,255,255,0.12); color: white;
      border-radius: 999px; padding: 10px 14px; font: inherit; font-size: 13px; font-weight: 700; cursor: pointer;
    }}
    .toggle-btn.active {{ background: white; color: var(--accent); border-color: white; }}
    .toggle-caption {{ margin-top: 10px; color: rgba(255,255,255,0.82); font-size: 13px; max-width: 80ch; }}
    .aggregate-empty {{ margin: 0; color: var(--muted); font-style: italic; }}
    @media (max-width: 820px) {{
      .wrap {{ padding: 18px 14px 36px; }}
      .hero h1 {{ font-size: 26px; }}
      .agg-grid {{ grid-template-columns: 1fr; }}
      .year-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>ROCKET Aggregate Plans · {_escape_html(summary['company'])}</h1>
          <p>Expanded aggregate-plan inspection for {_escape_html(summary['company'])}. Use the toggle to compare the baseline extracted workflows before ROCKET against the post-ROCKET selected plans, both company-wide and year by year.</p>
        </div>
        <div class="nav">
          <a class="pill alt" href="index.html">Back To Index</a>
          <a class="pill" href="rocket_reranking_visualizer___{_escape_html(company_slug)}.html">Open Company Visualizer</a>
        </div>
      </div>
      <div class="toggle-bar">
        <button type="button" class="toggle-btn active" data-mode-button="selected">After ROCKET</button>
        <button type="button" class="toggle-btn" data-mode-button="base">Before ROCKET</button>
      </div>
      <p class="toggle-caption">Current coverage: after ROCKET spans {_escape_html(selected_year_span)} across {int(selected.get('plan_count', 0)):,} plans; before ROCKET spans {_escape_html(base_year_span)} across {int(base.get('plan_count', 0)):,} plans.</p>
    </section>
    {mode_panel_html(mode_id="selected", mode_label="After ROCKET", aggregate=selected, year_rows=selected_year_rows, default_visible=True)}
    {mode_panel_html(mode_id="base", mode_label="Before ROCKET", aggregate=base, year_rows=base_year_rows, default_visible=False)}
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll('[data-mode-button]'));
    const panels = Array.from(document.querySelectorAll('[data-mode-panel]'));
    function setMode(mode) {{
      buttons.forEach((button) => {{
        button.classList.toggle('active', button.dataset.modeButton === mode);
      }});
      panels.forEach((panel) => {{
        panel.hidden = panel.dataset.modePanel !== mode;
      }});
    }}
    buttons.forEach((button) => {{
      button.addEventListener('click', () => setMode(button.dataset.modeButton));
    }});
    setMode('selected');
  </script>
</body>
</html>"""


def generate_basket_rocket_visualizations(batch_outdir: str | Path) -> BasketRocketVisualizationResult:
    """Write index and per-workset HTML visualizations from workflow batch artifacts."""

    batch_path = Path(batch_outdir).resolve()
    workset_index = list(json.loads((batch_path / "workset_index.json").read_text(encoding="utf-8")))
    panel_rows = _load_csv_rows(_DEFAULT_FINANCIAL_PANEL_PATH) if _DEFAULT_FINANCIAL_PANEL_PATH.exists() else []
    financial_panel_index = _build_financial_panel_index(panel_rows)
    viz_dir = batch_path / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    workset_pages: list[tuple[str, str]] = []
    aggregate_pages: list[tuple[str, str]] = []
    year_context_pages: list[tuple[str, str]] = []
    company_reranking_pages: list[tuple[str, str]] = []
    company_aggregate_pages: list[tuple[str, str]] = []
    company_rows_by_name: dict[str, list[dict[str, object]]] = {}
    company_meta_by_name: dict[str, dict[str, str]] = {}
    for index_row in workset_index:
        workset_name = str(index_row.get("workset_name") or "")
        workset_dir = batch_path / workset_name
        summary = _workset_payload(index_row, workset_dir)
        rankings_payload = _load_json(workset_dir / "rocket_rankings.json")
        rankings = list(rankings_payload.get("rankings") or [])
        filings = list(index_row.get("filings") or [])
        ticker = next((str(filing.get("ticker") or "").strip() for filing in filings if str(filing.get("ticker") or "").strip()), "")
        reward_backend = str(rankings_payload.get("reward_backend") or "")
        company_meta_by_name.setdefault(summary["company"], {"ticker": ticker, "reward_backend": reward_backend})
        workset_slug = _slugify(workset_name)
        workset_html_name = f"basket_workset_visualizer___{workset_slug}.html"
        workset_summary_name = f"basket_workset_visualizer___{workset_slug}_summary.json"
        aggregate_html_name = f"basket_aggregate_plans___{workset_slug}.html"

        _write_json(viz_dir / workset_summary_name, summary)
        _write_text(viz_dir / workset_html_name, _workset_page_html(summary, aggregate_html_name=aggregate_html_name))
        _write_text(viz_dir / aggregate_html_name, _aggregate_page_html(summary, workset_html_name=workset_html_name))

        top_candidate = summary["top_candidate"]
        index_rows.append(
            {
                "workset_name": workset_name,
                "company": summary["company"],
                "filing_year": summary["filing_year"],
                "form_type": summary["form_type"],
                "filing_count": summary["filing_count"],
                "chunk_count": summary["chunk_count"],
                "candidate_count": summary["candidate_count"],
                "ranking_count": summary["ranking_count"],
                "psr_status": summary["psr_status"],
                "semantic_role": str(index_row.get("semantic_role") or ""),
                "top_candidate": _sequence_string([str(item) for item in top_candidate.get("workflow_stages")]) if top_candidate else "none",
                "html": workset_html_name,
                "summary": workset_summary_name,
                "aggregate_html": aggregate_html_name,
                "year_key": f"{summary['company']}::{summary['filing_year']}",
                "aggregate_plan": summary["aggregate_plan"],
            }
        )
        workset_pages.append((workset_name, str(viz_dir / workset_html_name)))
        aggregate_pages.append((workset_name, str(viz_dir / aggregate_html_name)))
        for ranking in rankings:
            company_rows_by_name.setdefault(summary["company"], []).append(
                {
                    "candidate_id": str(ranking.get("candidate_id") or ""),
                    "filing_title": str(ranking.get("filing_title") or ""),
                    "year": int(summary["filing_year"]) if str(summary["filing_year"]).isdigit() else 0,
                    "form_type": str(summary["form_type"]),
                    "semantic_role": str(index_row.get("semantic_role") or ""),
                    "workset_name": workset_name,
                    "base_actions": [str(item) for item in ranking.get("base_workflow_stages") or [] if str(item).strip()],
                    "selected_actions": [str(item) for item in ranking.get("workflow_stages") or [] if str(item).strip()],
                    "selected_label": str(ranking.get("selected_label") or ""),
                    "selected_source": str(ranking.get("selected_source") or ""),
                    "score_gain": float(ranking.get("score_gain") or 0.0),
                    "selected_score": float(ranking.get("selected_score") or 0.0),
                    "base_score": float(ranking.get("base_score") or 0.0),
                    "changed": bool(ranking.get("changed", False)),
                    "ticker": ticker,
                    "reward_backend": str(ranking.get("reward_backend") or reward_backend),
                }
            )

    year_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in index_rows:
        year_groups.setdefault((str(row["company"]), str(row["filing_year"])), []).append(row)
    year_html_by_key: dict[tuple[str, str], str] = {}
    for (company, filing_year), rows in sorted(year_groups.items()):
        summary = _year_context_payload(rows)
        year_slug = _slugify(f"{company}_{filing_year}")
        year_html_name = f"basket_year_backbone___{year_slug}.html"
        _write_text(viz_dir / year_html_name, _year_context_page_html(summary))
        year_html_by_key[(company, filing_year)] = year_html_name
        year_context_pages.append((f"{company}::{filing_year}", str(viz_dir / year_html_name)))

    for row in index_rows:
        row["year_html"] = year_html_by_key[(str(row["company"]), str(row["filing_year"]))]

    company_index_rows: list[dict[str, object]] = []
    for company, rows in sorted(company_rows_by_name.items()):
        company_meta = company_meta_by_name.get(company, {})
        ticker = str(company_meta.get("ticker") or "")
        reward_backend = str(company_meta.get("reward_backend") or "")
        financial_summary = _summarize_company_financial(company, financial_panel_index, ticker=ticker)
        financial_by_year = _summarize_company_financial_by_year(company, financial_panel_index, ticker=ticker)
        summary = _company_payload(
            company,
            rows,
            ticker=ticker,
            reward_backend=reward_backend,
            financial_summary=financial_summary,
            financial_by_year=financial_by_year,
        )
        company_slug = _slugify(company)
        reranking_html_name = f"rocket_reranking_visualizer___{company_slug}.html"
        reranking_summary_name = f"rocket_reranking_visualizer___{company_slug}_summary.json"
        aggregate_html_name = f"rocket_aggregate_plans___{company_slug}.html"
        _write_json(viz_dir / reranking_summary_name, summary)
        _write_text(viz_dir / reranking_html_name, _company_reranking_page_html(summary))
        _write_text(viz_dir / aggregate_html_name, _company_aggregate_page_html(summary))
        source_mix = ", ".join(f"{key}={value}" for key, value in list(summary["selected_sources"].items())[:4]) or "none"
        inserted_mix = ", ".join(f"{key}={value}" for key, value in list(summary["inserted_actions"].items())[:4]) or "none"
        company_index_rows.append(
            {
                "company": company,
                "html": reranking_html_name,
                "summary": reranking_summary_name,
                "evidence_html": "",
                "evidence_summary": "",
                "aggregate_html": aggregate_html_name,
                "aggregate_detail_html": aggregate_html_name,
                "n_rows": summary["n_rows"],
                "n_changed": summary["n_changed"],
                "changed_rate": summary["changed_rate"],
                "mean_score_gain": summary["mean_score_gain"],
                "source_mix": source_mix,
                "inserted_mix": inserted_mix,
                "financial_summary": financial_summary,
                "aggregate_plan": summary["aggregate_plan_selected"],
            }
        )
        company_reranking_pages.append((company, str(viz_dir / reranking_html_name)))
        company_aggregate_pages.append((company, str(viz_dir / aggregate_html_name)))

    summary_path = viz_dir / "index_summary.json"
    company_summary_path = viz_dir / "company_index_summary.json"
    index_path = viz_dir / "index.html"
    _write_json(summary_path, index_rows)
    _write_json(company_summary_path, company_index_rows)
    _write_text(index_path, _index_html(company_index_rows, index_rows))
    return BasketRocketVisualizationResult(
        index_path=index_path,
        summary_path=summary_path,
        company_summary_path=company_summary_path,
        workset_pages=tuple(workset_pages),
        aggregate_pages=tuple(aggregate_pages),
        year_context_pages=tuple(year_context_pages),
        company_reranking_pages=tuple(company_reranking_pages),
        company_aggregate_pages=tuple(company_aggregate_pages),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BAFFLE BASKET/ROCKET HTML visualizations from a workflow_batches directory.")
    parser.add_argument("--batch-outdir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_basket_rocket_visualizations(args.batch_outdir)
    print(
        json.dumps(
            {
                "index_path": str(result.index_path),
                "summary_path": str(result.summary_path),
                "company_summary_path": str(result.company_summary_path),
                "workset_pages": len(result.workset_pages),
                "company_pages": len(result.company_reranking_pages),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
