"""Corpus-level post-processing for multi-filing BASKET/ROCKET runs."""

from __future__ import annotations

import html
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


@dataclass(frozen=True)
class BasketRocketCorpusSynthesisResult:
    """Materialized cross-filing synthesis artifacts."""

    summary_path: Path
    dashboard_path: Path


@dataclass(frozen=True)
class FilingWorksetSynthesis:
    workset_name: str
    company: str
    filing_year: str
    form_type: str
    semantic_role: str
    top_candidate: str
    aggregate_plan: str
    selected_source: str
    score_gain: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_basket_rocket_corpus_synthesis(
    *,
    query: str,
    batch_outdir: Path,
    workset_index_path: Path,
    visualization_summary_path: Path | None = None,
    company_summary_path: Path | None = None,
) -> BasketRocketCorpusSynthesisResult:
    synthesis_dir = batch_outdir / "corpus_synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = synthesis_dir / "basket_rocket_corpus_synthesis.json"
    dashboard_path = synthesis_dir / "basket_rocket_corpus_synthesis.html"

    workset_index = _load_json_list(workset_index_path)
    workset_rows = _load_json_list(visualization_summary_path) if visualization_summary_path else []
    company_rows = _load_json_list(company_summary_path) if company_summary_path else []

    rows_by_workset = {str(row.get("workset_name") or ""): row for row in workset_rows}
    synthesized_worksets: list[FilingWorksetSynthesis] = []
    for item in workset_index:
        workset_name = str(item.get("workset_name") or "")
        row = rows_by_workset.get(workset_name, {})
        synthesized_worksets.append(
            FilingWorksetSynthesis(
                workset_name=workset_name,
                company=str(item.get("company") or ""),
                filing_year=str(item.get("filing_year") or ""),
                form_type=str(item.get("form_type") or ""),
                semantic_role=str(item.get("semantic_role") or ""),
                top_candidate=str(row.get("top_candidate") or "none"),
                aggregate_plan=str(row.get("aggregate_plan") or "none"),
                selected_source=str(row.get("selected_source") or ""),
                score_gain=float(row.get("mean_score_gain") or row.get("score_gain") or 0.0),
            )
        )

    payload = {
        "query": query,
        "n_worksets": len(workset_index),
        "n_companies": len({str(item.get("company") or "") for item in workset_index if str(item.get("company") or "")}),
        "n_filings": sum(len(item.get("filings") or []) for item in workset_index),
        "worksets": [item.as_dict() for item in synthesized_worksets],
        "company_rows": company_rows,
        "workset_index_path": str(workset_index_path),
        "visualization_summary_path": str(visualization_summary_path) if visualization_summary_path else None,
        "company_summary_path": str(company_summary_path) if company_summary_path else None,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dashboard_path.write_text(_render_dashboard_html(payload, dashboard_path=dashboard_path, batch_outdir=batch_outdir), encoding="utf-8")
    return BasketRocketCorpusSynthesisResult(summary_path=summary_path, dashboard_path=dashboard_path)


def _load_json_list(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return []


def _relative_href(target: Path, *, start: Path) -> str:
    if not target.exists():
        return ""
    return os.path.relpath(target.resolve(), start=start.resolve())


def _render_workset_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    return (
        '<article class="card">'
        f'<div class="meta">{esc(item.get("form_type"))} · {esc(str(item.get("semantic_role") or "").replace("_", " "))}</div>'
        f'<h3>{esc(item.get("company"))} · {esc(item.get("filing_year"))}</h3>'
        f'<p><strong>Recovered workflow:</strong> {esc(item.get("top_candidate") or "none")}</p>'
        f'<p><strong>Aggregate plan:</strong> {esc(item.get("aggregate_plan") or "none")}</p>'
        f'<p class="trace">workset={esc(item.get("workset_name"))} · score gain={esc(item.get("score_gain"))}</p>'
        "</article>"
    )


def _render_company_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    return (
        '<article class="card">'
        f'<div class="meta">company aggregate</div>'
        f'<h3>{esc(item.get("company"))}</h3>'
        f'<p><strong>Changed rate:</strong> {esc(item.get("changed_rate"))}</p>'
        f'<p><strong>Mean score gain:</strong> {esc(item.get("mean_score_gain"))}</p>'
        f'<p class="trace">{esc(item.get("aggregate_plan") or "aggregate plan pending")}</p>'
        "</article>"
    )


def _render_dashboard_html(payload: dict[str, object], *, dashboard_path: Path, batch_outdir: Path) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    workset_markup = "".join(_render_workset_card(item) for item in payload.get("worksets") or []) or '<div class="empty">No workset synthesis available yet.</div>'
    company_markup = "".join(_render_company_card(item) for item in payload.get("company_rows") or []) or '<div class="empty">No company-level aggregate synthesis available yet.</div>'
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(str(payload.get("query") or ""), route_name="basket_rocket_sec"),
    )
    live_gui_href = _relative_href(batch_outdir / "basket_rocket_gui.html", start=dashboard_path.parent)
    viz_href = _relative_href(batch_outdir / "visualizations" / "index.html", start=dashboard_path.parent)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BASKET/ROCKET Corpus Synthesis</title>
    <style>
      :root {{ --ink:#16211f; --muted:#5a6661; --paper:#f3ead7; --card:#fffdf8; --line:#d0c0a0; --accent:#0f6d63; }}
      * {{ box-sizing:border-box; }}
      body {{ margin:0; font-family:Georgia,"Iowan Old Style",serif; color:var(--ink); background:linear-gradient(180deg,#faf6ef 0%,var(--paper) 100%); }}
      main {{ width:min(1220px, calc(100vw - 32px)); margin:28px auto 48px; display:grid; gap:18px; }}
      .panel {{ background:rgba(255,252,246,0.96); border:1px solid var(--line); border-radius:26px; padding:24px; }}
      .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
      .cards {{ display:grid; gap:12px; }}
      .card {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:#fffdfa; }}
      h1,h2,h3,p {{ margin:0; }}
      .eyebrow,.meta,.trace {{ color:var(--muted); }}
      .eyebrow {{ text-transform:uppercase; letter-spacing:0.14em; font-size:12px; margin-bottom:10px; color:var(--accent); }}
      .meta,.trace {{ font-size:0.94rem; line-height:1.5; }}
      .links {{ margin-top:14px; display:flex; gap:12px; flex-wrap:wrap; }}
      a {{ color:var(--accent); text-decoration:none; font-weight:700; }}
      a:hover {{ text-decoration:underline; }}
      .chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
      .chip {{ border-radius:999px; padding:8px 12px; background:#edf3f1; color:#184a43; }}
      .textbook-list {{ padding-left:20px; display:grid; gap:10px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      .empty {{ color:var(--muted); line-height:1.6; }}
      @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">CLIFF Workflow Synthesis</p>
        <h1>{esc(payload.get("query") or "BASKET/ROCKET corpus synthesis")}</h1>
        <p class="trace">The conscious layer triggered a second pass that glues individual filing workflows into a cross-filing operational view. This surface summarizes the recovered workflow backbones across worksets and company-level aggregates.</p>
        <div class="chips">
          <span class="chip">{esc(payload.get("n_filings") or 0)} filings analyzed</span>
          <span class="chip">{esc(payload.get("n_worksets") or 0)} worksets synthesized</span>
          <span class="chip">{esc(payload.get("n_companies") or 0)} companies represented</span>
        </div>
        <div class="links">
          {f'<a href="{esc(live_gui_href)}" target="_blank" rel="noreferrer">Open live BASKET/ROCKET GUI</a>' if live_gui_href else ''}
          {f'<a href="{esc(viz_href)}" target="_blank" rel="noreferrer">Open visualization suite</a>' if viz_href else ''}
        </div>
      </section>
      <section class="panel">
        {textbook_html}
      </section>
      <section class="grid">
        <section class="panel"><p class="eyebrow">Workset Gluing</p><div class="cards">{workset_markup}</div></section>
        <section class="panel"><p class="eyebrow">Company Aggregates</p><div class="cards">{company_markup}</div></section>
      </section>
    </main>
  </body>
</html>"""
