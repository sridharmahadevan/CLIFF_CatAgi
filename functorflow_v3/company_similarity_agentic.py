"""Natural-language company similarity runner built on temporal diffusion outputs."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .democritus_query_agentic import _resolve_sec_user_agent
from .repo_layout import repo_root, resolve_brand_panel_root
from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


def _slugify(value: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "company"


def _normalize_alias(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9& ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _safe_read_json(path: Path) -> dict[str, object]:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _resolve_brand_workspace_root() -> Path:
    candidate = resolve_brand_panel_root()
    if (candidate / "configs" / "company_batch_registry.csv").exists():
        return candidate
    sibling = resolve_brand_panel_root()
    if (sibling / "configs" / "company_batch_registry.csv").exists():
        return sibling.resolve()
    return candidate


def _portable_output_path(raw_path: str, *, brand_root: Path, fallback: Path) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return fallback.resolve()
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    if "outputs" in parts:
        outputs_index = parts.index("outputs")
        translated = brand_root / Path(*parts[outputs_index:])
        return translated.resolve()
    return fallback.resolve()


@dataclass(frozen=True)
class CompanySimilarityQueryPlan:
    query: str
    company_a: str
    company_b: str
    company_a_slug: str
    company_b_slug: str
    intent: str = "company_similarity"


@dataclass(frozen=True)
class _CompanyRecord:
    brand: str
    slug: str
    aliases: tuple[str, ...]
    ticker: str = ""
    index_url: str = ""
    outdir: Path | None = None
    existing_combined_dir: Path | None = None


@dataclass(frozen=True)
class CompanySimilarityRunResult:
    query_plan: CompanySimilarityQueryPlan
    route_outdir: Path
    analysis_dir: Path
    summary_path: Path
    artifact_path: Path | None
    company_a_manifest_path: Path | None
    company_b_manifest_path: Path | None


def looks_like_company_similarity_query(query: str) -> bool:
    normalized = _normalize_alias(query)
    if not normalized:
        return False
    similarity_markers = (
        "similar",
        "similarity",
        "compare",
        "comparison",
        "versus",
        " vs ",
    )
    if not any(marker in normalized for marker in similarity_markers):
        return False
    return " to " in normalized or " and " in normalized or " vs " in normalized or " versus " in normalized


def _load_company_registry(brand_root: Path) -> dict[str, _CompanyRecord]:
    records: dict[str, _CompanyRecord] = {}
    registry_path = brand_root / "configs" / "company_batch_registry.csv"
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                brand = str(row.get("brand", "")).strip()
                if not brand:
                    continue
                slug = _slugify(brand)
                aliases = [_normalize_alias(brand), _normalize_alias(slug.replace("_", " "))]
                alias_field = str(row.get("company_aliases", "")).strip()
                if alias_field:
                    aliases.extend(_normalize_alias(item.replace("_", " ")) for item in alias_field.split(","))
                ticker = str(row.get("edgar_ticker", "")).strip().lower()
                if len(ticker) >= 3:
                    aliases.append(ticker)
                outdir_raw = str(row.get("outdir", "")).strip()
                default_outdir = (brand_root / "outputs" / slug).resolve()
                outdir = _portable_output_path(outdir_raw, brand_root=brand_root, fallback=default_outdir)
                combined_raw = str(row.get("existing_combined_dir", "")).strip()
                default_combined = default_outdir / f"runs_{slug}_financial_filings" / f"atlas_{slug}_financial_combined"
                combined_dir = (
                    _portable_output_path(combined_raw, brand_root=brand_root, fallback=default_combined)
                    if combined_raw
                    else None
                )
                records[slug] = _CompanyRecord(
                    brand=brand,
                    slug=slug,
                    aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                    ticker=ticker,
                    index_url=str(row.get("index_url", "")).strip(),
                    outdir=outdir,
                    existing_combined_dir=combined_dir if combined_dir else None,
                )
    outputs_root = brand_root / "outputs"
    if outputs_root.exists():
        for manifest_path in sorted(outputs_root.glob("*/add_company_analysis_manifest.json")):
            payload = _safe_read_json(manifest_path)
            brand = str(payload.get("brand", "")).strip() or manifest_path.parent.name.replace("_", " ").title()
            slug = str(payload.get("brand_slug", "")).strip() or _slugify(brand)
            existing = records.get(slug)
            aliases = list(existing.aliases) if existing else []
            aliases.extend((_normalize_alias(brand), _normalize_alias(slug.replace("_", " ")), _normalize_alias(manifest_path.parent.name.replace("_", " "))))
            if existing and existing.ticker and len(existing.ticker) >= 3:
                aliases.append(existing.ticker)
            records[slug] = _CompanyRecord(
                brand=brand,
                slug=slug,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                ticker=existing.ticker if existing else "",
                index_url=existing.index_url if existing else "",
                outdir=manifest_path.parent.resolve(),
                existing_combined_dir=existing.existing_combined_dir if existing else None,
            )
    preset_path = brand_root / "configs" / "company_universe_presets.json"
    if preset_path.exists():
        payload = _safe_read_json(preset_path)
        presets = payload.get("presets", {})
        if isinstance(presets, dict):
            for preset in presets.values():
                companies = preset.get("companies", [])
                if not isinstance(companies, list):
                    continue
                for item in companies:
                    if not isinstance(item, dict):
                        continue
                    brand = str(item.get("brand", "")).strip()
                    if not brand:
                        continue
                    slug = _slugify(brand)
                    ticker = str(item.get("ticker", "")).strip().lower()
                    existing = records.get(slug)
                    aliases = list(existing.aliases) if existing else []
                    aliases.extend(
                        (
                            _normalize_alias(brand),
                            _normalize_alias(slug.replace("_", " ")),
                        )
                    )
                    if ticker and len(ticker) >= 1:
                        aliases.append(ticker)
                    records[slug] = _CompanyRecord(
                        brand=existing.brand if existing else brand,
                        slug=slug,
                        aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                        ticker=existing.ticker if existing and existing.ticker else ticker,
                        index_url=existing.index_url if existing else "",
                        outdir=existing.outdir if existing else (brand_root / "outputs" / slug).resolve(),
                        existing_combined_dir=existing.existing_combined_dir if existing else None,
                    )
    return records


def _extract_companies_from_query(query: str, records: dict[str, _CompanyRecord]) -> tuple[_CompanyRecord, _CompanyRecord]:
    normalized = f" {_normalize_alias(query)} "
    matches: list[tuple[int, int, _CompanyRecord]] = []
    for record in records.values():
        best_position: int | None = None
        best_length = -1
        for alias in record.aliases:
            alias = alias.strip()
            if not alias or len(alias) < 3:
                continue
            pattern = f" {alias} "
            position = normalized.find(pattern)
            if position < 0:
                continue
            if best_position is None or position < best_position or (position == best_position and len(alias) > best_length):
                best_position = position
                best_length = len(alias)
        if best_position is not None:
            matches.append((best_position, best_length, record))
    matches.sort(key=lambda item: (item[0], -item[1], item[2].brand.lower()))
    ordered: list[_CompanyRecord] = []
    seen = set()
    for _, _, record in matches:
        if record.slug in seen:
            continue
        seen.add(record.slug)
        ordered.append(record)
    if len(ordered) < 2:
        raise ValueError(
            "Could not confidently identify two companies in the similarity query. "
            "Try phrasing it like 'How similar is Adobe to Nike?' or 'Compare Adobe and Nike.'"
        )
    return ordered[0], ordered[1]


def interpret_company_similarity_query(query: str) -> CompanySimilarityQueryPlan:
    brand_root = _resolve_brand_workspace_root()
    records = _load_company_registry(brand_root)
    company_a, company_b = _extract_companies_from_query(query, records)
    return CompanySimilarityQueryPlan(
        query=" ".join(query.split()),
        company_a=company_a.brand,
        company_b=company_b.brand,
        company_a_slug=company_a.slug,
        company_b_slug=company_b.slug,
    )


def _find_company_record(plan: CompanySimilarityQueryPlan, brand_root: Path, slug: str) -> _CompanyRecord:
    records = _load_company_registry(brand_root)
    record = records.get(slug)
    if record is not None:
        return record
    brand = plan.company_a if slug == plan.company_a_slug else plan.company_b
    return _CompanyRecord(
        brand=brand,
        slug=slug,
        aliases=(_normalize_alias(brand), _normalize_alias(slug.replace("_", " "))),
        outdir=(brand_root / "outputs" / slug).resolve(),
    )


def _combined_dir_candidates(record: _CompanyRecord) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if record.existing_combined_dir is not None:
        candidates.append(record.existing_combined_dir)
    if record.outdir is not None:
        candidates.append(record.outdir / f"runs_{record.slug}_financial_filings" / f"atlas_{record.slug}_financial_combined")
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _manifest_path(record: _CompanyRecord) -> Path:
    base = record.outdir or (_resolve_brand_workspace_root() / "outputs" / record.slug)
    return base / "add_company_analysis_manifest.json"


def _existing_local_filing_manifest(record: _CompanyRecord) -> Path | None:
    if record.outdir is None:
        return None
    filings_outdir = record.outdir / f"runs_{record.slug}_financial_filings"
    candidates = (
        filings_outdir / f"{record.slug}_sec_edgar_manifest.csv",
        filings_outdir / f"{record.slug}_sec_edgar_manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _existing_combined_dir(record: _CompanyRecord) -> Path | None:
    for candidate in _combined_dir_candidates(record):
        if candidate.exists():
            return candidate
    manifest_path = _manifest_path(record)
    if manifest_path.exists():
        payload = _safe_read_json(manifest_path)
        raw = str(payload.get("combined_dir", "")).strip()
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate.resolve()
    return None


def _run_command(command: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    try:
        subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_tail = "\n".join((exc.stderr or "").splitlines()[-20:]).strip()
        stdout_tail = "\n".join((exc.stdout or "").splitlines()[-20:]).strip()
        details = stderr_tail or stdout_tail or "No stdout/stderr was captured."
        raise RuntimeError(
            "Company similarity backend command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"CWD: {cwd}\n"
            f"Exit code: {exc.returncode}\n"
            f"Last output:\n{details}"
        ) from exc


def _python_has_modules(python_executable: str, modules: tuple[str, ...], *, cwd: Path | None = None) -> bool:
    probe = [
        python_executable,
        "-c",
        "import importlib.util, sys; "
        + "; ".join(
            f"assert importlib.util.find_spec({module!r}) is not None, {module!r}" for module in modules
        )
        + "; print(sys.executable)",
    ]
    try:
        subprocess.run(
            probe,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        )
        return True
    except Exception:
        return False


def _select_python_for_brand_pipeline() -> str:
    env_override = os.environ.get("CLIFF_BRAND_PIPELINE_PYTHON", "").strip()
    workspace_root = repo_root().parents[0]
    brand_root = resolve_brand_panel_root()
    required_modules = (
        "brand_democritus_block_denoise",
        "pandas",
        "pyarrow",
        "matplotlib",
        "tqdm",
        "umap",
    )
    candidates = [
        env_override,
        sys.executable,
        str((brand_root / ".venv" / "bin" / "python")),
        str((repo_root() / ".venv" / "bin" / "python")),
        "/opt/homebrew/bin/python3",
        "python3",
    ]
    for candidate in dict.fromkeys(item for item in candidates if item):
        if candidate != "python3" and not Path(candidate).exists():
            continue
        if _python_has_modules(candidate, required_modules, cwd=workspace_root):
            return candidate
    raise RuntimeError(
        "Could not find a Python interpreter with the required brand diffusion pipeline dependencies "
        "(brand_democritus_block_denoise, pandas, pyarrow, matplotlib, tqdm, umap). "
        "If needed, set CLIFF_BRAND_PIPELINE_PYTHON to the correct interpreter."
    )


def _ensure_company_analysis(
    *,
    record: _CompanyRecord,
    sec_user_agent: str,
    workspace_root: Path,
    python_executable: str,
) -> tuple[Path, Path | None]:
    combined_dir = _existing_combined_dir(record)
    manifest_path = _manifest_path(record)
    if combined_dir is not None:
        return combined_dir, manifest_path if manifest_path.exists() else None
    if record.outdir is None:
        raise ValueError(f"No output directory is configured for {record.brand}.")
    if not record.index_url and not record.ticker:
        raise ValueError(
            f"No retrieval metadata is configured for {record.brand}. "
            "Add an index URL or EDGAR ticker to the company registry before requesting this comparison."
        )
    existing_filing_manifest = _existing_local_filing_manifest(record)
    command = [
        python_executable,
        "-m",
        "brand_democritus_block_denoise.add_company_analysis",
        "--brand",
        record.brand,
        "--company-aliases",
        ",".join(dict.fromkeys(alias.replace(" ", "_") for alias in record.aliases if alias)),
        "--outdir",
        str(record.outdir),
        "--jobs",
        "1",
        "--llm-jobs",
        "1",
        "--filings-profile",
        "fast",
        "--basis-mode",
        "motif",
        "--block-size",
        "3",
        "--epochs",
        "3",
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]
    if existing_filing_manifest is not None:
        command.extend(["--manifest", str(existing_filing_manifest)])
    elif record.index_url:
        command.extend(["--index-url", record.index_url])
    elif record.ticker:
        resolved_sec_user_agent = _resolve_sec_user_agent(sec_user_agent)
        command.extend(["--edgar-ticker", record.ticker])
        command.extend(["--sec-user-agent", resolved_sec_user_agent])
    _run_command(command, cwd=workspace_root)
    combined_dir = _existing_combined_dir(record)
    if combined_dir is None:
        raise FileNotFoundError(f"Expected combined atlas output for {record.brand} after company analysis completed.")
    return combined_dir, manifest_path if manifest_path.exists() else None


def _write_html_report(
    *,
    output_path: Path,
    analysis_dir: Path,
    plan: CompanySimilarityQueryPlan,
    summary_markdown: str,
    manifest: dict[str, object],
) -> Path:
    artifact_path = output_path
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(plan.query, route_name="company_similarity"),
    )
    dashboard_png = analysis_dir / "cross_company_functors_dashboard.png"
    summary_html = html.escape(summary_markdown)
    relative_analysis_dir = analysis_dir.relative_to(artifact_path.parent)
    image_markup = (
        f'<figure><img src="{html.escape((relative_analysis_dir / "cross_company_functors_dashboard.png").as_posix())}" alt="Cross-company similarity dashboard" /></figure>'
        if dashboard_png.exists()
        else "<p class=\"muted\">Visualization image not available yet.</p>"
    )
    mean_cosine = manifest.get("mean_yearly_cosine_similarity", "n/a")
    mean_js = manifest.get("mean_yearly_js_divergence", "n/a")
    mean_defect = manifest.get("mean_relative_naturality_defect", "n/a")
    artifact_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(plan.company_a)} vs {html.escape(plan.company_b)} similarity</title>
    <style>
      :root {{
        --ink: #1f2320;
        --muted: #5d685f;
        --paper: #f6f1e6;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #9a5a12;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top, rgba(154, 90, 18, 0.10), transparent 26%),
          linear-gradient(180deg, #fcf8f1 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 1120px;
        margin: 40px auto;
        padding: 0 18px 48px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 18px 48px rgba(42, 31, 12, 0.10);
        margin-bottom: 20px;
      }}
      .eyebrow {{
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--accent);
      }}
      h1 {{
        margin: 0 0 12px 0;
        font-size: clamp(30px, 4vw, 50px);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .metric {{
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px;
      }}
      .metric strong {{
        display: block;
        font-size: 24px;
        margin-top: 4px;
      }}
      .muted {{
        color: var(--muted);
      }}
      .textbook-list {{
        padding-left: 20px;
        display: grid;
        gap: 10px;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
      img {{
        width: 100%;
        border-radius: 18px;
        border: 1px solid var(--line);
      }}
      pre {{
        white-space: pre-wrap;
        background: #fbf7ef;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
        color: var(--muted);
        line-height: 1.5;
      }}
      a {{
        color: #7d4306;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Company Similarity</p>
        <h1>{html.escape(plan.company_a)} vs {html.escape(plan.company_b)}</h1>
        <p class="muted">CLIFF resolved the query as a cross-company temporal diffusion comparison over yearly 10-K causal atlases.</p>
        <div class="metrics">
          <div class="metric"><span class="muted">Mean yearly cosine</span><strong>{html.escape(str(mean_cosine))}</strong></div>
          <div class="metric"><span class="muted">Mean yearly JS divergence</span><strong>{html.escape(str(mean_js))}</strong></div>
          <div class="metric"><span class="muted">Mean relative naturality defect</span><strong>{html.escape(str(mean_defect))}</strong></div>
        </div>
      </section>
      <section class="card">
        {textbook_html}
      </section>
      <section class="card">
        <p class="eyebrow">Dashboard</p>
        {image_markup}
      </section>
      <section class="card">
        <p class="eyebrow">Summary</p>
        <pre>{summary_html}</pre>
        <p class="muted">
          Raw outputs:
          <a href="{html.escape((relative_analysis_dir / 'cross_company_functors_summary.md').as_posix())}">summary markdown</a>,
          <a href="{html.escape((relative_analysis_dir / 'cross_company_year_metrics.csv').as_posix())}">year metrics</a>,
          <a href="{html.escape((relative_analysis_dir / 'cross_company_top_shared_edges.csv').as_posix())}">shared edges</a>.
        </p>
      </section>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return artifact_path


class CompanySimilarityAgenticRunner:
    def __init__(self, query: str, outdir: Path, *, sec_user_agent: str = "") -> None:
        self.query = " ".join(query.split())
        self.outdir = outdir.resolve()
        self.sec_user_agent = sec_user_agent
        if not self.query:
            raise ValueError("A non-empty company similarity query is required.")

    def run(self) -> CompanySimilarityRunResult:
        self.outdir.mkdir(parents=True, exist_ok=True)
        brand_root = _resolve_brand_workspace_root()
        workspace_root = repo_root().parents[0]
        pipeline_python = _select_python_for_brand_pipeline()
        plan = interpret_company_similarity_query(self.query)
        record_a = _find_company_record(plan, brand_root, plan.company_a_slug)
        record_b = _find_company_record(plan, brand_root, plan.company_b_slug)
        combined_a, manifest_a = _ensure_company_analysis(
            record=record_a,
            sec_user_agent=self.sec_user_agent,
            workspace_root=workspace_root,
            python_executable=pipeline_python,
        )
        combined_b, manifest_b = _ensure_company_analysis(
            record=record_b,
            sec_user_agent=self.sec_user_agent,
            workspace_root=workspace_root,
            python_executable=pipeline_python,
        )

        analysis_dir = self.outdir / f"{plan.company_a_slug}_vs_{plan.company_b_slug}_functors"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.outdir / "company_similarity_summary.json"

        _run_command(
            [
                pipeline_python,
                "-m",
                "brand_democritus_block_denoise.cross_company_functors",
                "--company-a-dir",
                str(combined_a),
                "--company-b-dir",
                str(combined_b),
                "--company-a-name",
                plan.company_a_slug,
                "--company-b-name",
                plan.company_b_slug,
                "--outdir",
                str(analysis_dir),
            ],
            cwd=workspace_root,
        )
        _run_command(
            [
                pipeline_python,
                "-m",
                "brand_democritus_block_denoise.visualize_cross_company_functors",
                "--analysis-dir",
                str(analysis_dir),
                "--outdir",
                str(analysis_dir),
            ],
            cwd=workspace_root,
        )

        manifest = _safe_read_json(analysis_dir / "cross_company_functors_manifest.json")
        year_metrics_path = analysis_dir / "cross_company_year_metrics.csv"
        naturality_path = analysis_dir / "naturality_defects.csv"
        if year_metrics_path.exists():
            with year_metrics_path.open("r", encoding="utf-8", newline="") as handle:
                year_rows = [dict(row) for row in csv.DictReader(handle)]
            cosine_values = [float(row["cosine_similarity"]) for row in year_rows if str(row.get("cosine_similarity", "")).strip()]
            js_values = [float(row["js_divergence"]) for row in year_rows if str(row.get("js_divergence", "")).strip()]
            manifest["mean_yearly_cosine_similarity"] = (
                round(sum(cosine_values) / len(cosine_values), 4) if cosine_values else math.nan
            )
            manifest["mean_yearly_js_divergence"] = (
                round(sum(js_values) / len(js_values), 4) if js_values else math.nan
            )
        if naturality_path.exists():
            with naturality_path.open("r", encoding="utf-8", newline="") as handle:
                defect_rows = [dict(row) for row in csv.DictReader(handle)]
            defect_values = [
                float(row["naturality_defect_rel"])
                for row in defect_rows
                if str(row.get("naturality_defect_rel", "")).strip()
            ]
            manifest["mean_relative_naturality_defect"] = (
                round(sum(defect_values) / len(defect_values), 4) if defect_values else math.nan
            )
        summary_markdown = (analysis_dir / "cross_company_functors_summary.md").read_text(encoding="utf-8")
        artifact_path = _write_html_report(
            output_path=self.outdir / "company_similarity_dashboard.html",
            analysis_dir=analysis_dir,
            plan=plan,
            summary_markdown=summary_markdown,
            manifest=manifest,
        )
        payload = {
            "query_plan": asdict(plan),
            "analysis_dir": str(analysis_dir),
            "artifact_path": str(artifact_path),
            "company_a_manifest_path": str(manifest_a) if manifest_a else None,
            "company_b_manifest_path": str(manifest_b) if manifest_b else None,
            "cross_company_manifest_path": str(analysis_dir / "cross_company_functors_manifest.json"),
            "summary_markdown_path": str(analysis_dir / "cross_company_functors_summary.md"),
            "dashboard_png_path": str(analysis_dir / "cross_company_functors_dashboard.png"),
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return CompanySimilarityRunResult(
            query_plan=plan,
            route_outdir=self.outdir,
            analysis_dir=analysis_dir,
            summary_path=summary_path,
            artifact_path=artifact_path,
            company_a_manifest_path=manifest_a,
            company_b_manifest_path=manifest_b,
        )
