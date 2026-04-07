"""Corpus-level post-processing for product feedback runs."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .product_feedback_agentic import ProductFeedbackRunResult
from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


@dataclass(frozen=True)
class ProductFeedbackCorpusSynthesisResult:
    """Materialized cross-review synthesis artifacts."""

    summary_path: Path
    dashboard_path: Path


def build_product_feedback_corpus_synthesis(
    *,
    query: str,
    outdir: Path,
    feedback_result: ProductFeedbackRunResult | None = None,
    analysis_outdir: Path | None = None,
) -> ProductFeedbackCorpusSynthesisResult:
    synthesis_dir = outdir / "corpus_synthesis"
    synthesis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = synthesis_dir / "product_feedback_corpus_synthesis.json"
    dashboard_path = synthesis_dir / "product_feedback_corpus_synthesis.html"

    base_outdir = feedback_result.dashboard_path.parent if feedback_result is not None else Path(analysis_outdir or outdir).resolve()
    if feedback_result is None:
        feedback_result = ProductFeedbackRunResult(
            records=(),
            normalized_feedback_path=base_outdir / "normalized_feedback.jsonl",
            usage_workflows_path=base_outdir / "usage_workflows.json",
            aspect_summary_path=base_outdir / "aspect_summary.json",
            outcome_summary_path=base_outdir / "outcome_summary.json",
            causal_hypotheses_path=base_outdir / "causal_hypotheses.json",
            success_scorecard_path=base_outdir / "product_success_scorecard.json",
            ablation_comparison_path=base_outdir / "ablation_comparison.json",
            report_path=base_outdir / "product_feedback_report.md",
            dashboard_path=base_outdir / "product_feedback_dashboard.html",
            dashboard_summary_path=base_outdir / "product_feedback_dashboard_summary.json",
        )

    scorecard = json.loads(feedback_result.success_scorecard_path.read_text(encoding="utf-8"))
    aspects = json.loads(feedback_result.aspect_summary_path.read_text(encoding="utf-8"))
    outcomes = json.loads(feedback_result.outcome_summary_path.read_text(encoding="utf-8"))
    workflows = json.loads(feedback_result.usage_workflows_path.read_text(encoding="utf-8"))
    hypotheses = json.loads(feedback_result.causal_hypotheses_path.read_text(encoding="utf-8"))

    payload = {
        "query": query,
        "product_name": str(scorecard.get("product_name") or ""),
        "brand_name": str(scorecard.get("brand_name") or ""),
        "verdict": str(scorecard.get("verdict") or "unknown"),
        "overall_score": float(scorecard.get("overall_score") or 0.0),
        "return_warning_recommended": bool(scorecard.get("return_warning_recommended")),
        "feedback_count": int(outcomes.get("feedback_count") or 0),
        "top_positive_aspects": list(scorecard.get("top_positive_aspects") or []),
        "top_negative_aspects": list(scorecard.get("top_negative_aspects") or []),
        "top_return_risk_aspects": list(scorecard.get("top_return_risk_aspects") or []),
        "usage_workflows": list((workflows.get("workflow_summaries") or workflows.get("usage_workflows") or [])),
        "causal_hypotheses": list(hypotheses.get("hypotheses") or hypotheses.get("drivers") or []),
        "dashboard_path": str(feedback_result.dashboard_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dashboard_path.write_text(_render_dashboard_html(payload, dashboard_path=dashboard_path, feedback_result=feedback_result), encoding="utf-8")
    return ProductFeedbackCorpusSynthesisResult(summary_path=summary_path, dashboard_path=dashboard_path)


def _relative_href(target: Path, *, start: Path) -> str:
    if not target.exists():
        return ""
    return os.path.relpath(target.resolve(), start=start.resolve())


def _render_list(items: list[object], *, empty: str) -> str:
    if not items:
        return f'<div class="empty">{html.escape(empty)}</div>'
    return "".join(f'<li>{html.escape(str(item))}</li>' for item in items[:8])


def _render_dashboard_html(
    payload: dict[str, object],
    *,
    dashboard_path: Path,
    feedback_result: ProductFeedbackRunResult,
) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    dashboard_href = _relative_href(feedback_result.dashboard_path, start=dashboard_path.parent)
    report_href = _relative_href(feedback_result.report_path, start=dashboard_path.parent)
    workflows = [item.get("summary") if isinstance(item, dict) else item for item in payload.get("usage_workflows") or []]
    hypotheses = [item.get("statement") if isinstance(item, dict) else item for item in payload.get("causal_hypotheses") or []]
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(str(payload.get("query") or ""), route_name="product_feedback"),
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Product Feedback Corpus Synthesis</title>
    <style>
      :root {{ --ink:#16211f; --muted:#5a6661; --paper:#f3ead7; --card:#fffdf8; --line:#d0c0a0; --accent:#0f6d63; }}
      * {{ box-sizing:border-box; }}
      body {{ margin:0; font-family:Georgia,"Iowan Old Style",serif; color:var(--ink); background:linear-gradient(180deg,#faf6ef 0%,var(--paper) 100%); }}
      main {{ width:min(1180px, calc(100vw - 32px)); margin:28px auto 48px; display:grid; gap:18px; }}
      .panel {{ background:rgba(255,252,246,0.96); border:1px solid var(--line); border-radius:26px; padding:24px; }}
      .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
      h1,h2,p,ul {{ margin:0; }}
      .eyebrow {{ text-transform:uppercase; letter-spacing:0.14em; font-size:12px; margin-bottom:10px; color:var(--accent); }}
      .trace,.empty {{ color:var(--muted); line-height:1.6; }}
      .chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
      .chip {{ border-radius:999px; padding:8px 12px; background:#edf3f1; color:#184a43; }}
      ul {{ padding-left:20px; display:grid; gap:8px; }}
      .textbook-list {{ padding-left:20px; display:grid; gap:10px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      a {{ color:var(--accent); text-decoration:none; font-weight:700; }}
      a:hover {{ text-decoration:underline; }}
      .links {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:12px; }}
      @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">CLIFF Review Synthesis</p>
        <h1>{esc(payload.get("product_name") or payload.get("query") or "Product feedback synthesis")}</h1>
        <p class="trace">The conscious layer triggered a second pass that treats the retrieved reviews as one evidence corpus. This page summarizes the glued verdict, major strengths, risk factors, and usage patterns across the analyzed review set.</p>
        <div class="chips">
          <span class="chip">verdict: {esc(payload.get("verdict"))}</span>
          <span class="chip">overall score: {esc(payload.get("overall_score"))}</span>
          <span class="chip">{esc(payload.get("feedback_count") or 0)} reviews synthesized</span>
          <span class="chip">return warning: {esc(payload.get("return_warning_recommended"))}</span>
        </div>
        <div class="links">
          {f'<a href="{esc(dashboard_href)}" target="_blank" rel="noreferrer">Open product feedback dashboard</a>' if dashboard_href else ''}
          {f'<a href="{esc(report_href)}" target="_blank" rel="noreferrer">Open product feedback report</a>' if report_href else ''}
        </div>
      </section>
      <section class="panel">
        {textbook_html}
      </section>
      <section class="grid">
        <section class="panel"><p class="eyebrow">Positive Aspects</p><ul>{_render_list(list(payload.get("top_positive_aspects") or []), empty="No positive aspects surfaced yet.")}</ul></section>
        <section class="panel"><p class="eyebrow">Negative Aspects</p><ul>{_render_list(list(payload.get("top_negative_aspects") or []), empty="No negative aspects surfaced yet.")}</ul></section>
      </section>
      <section class="grid">
        <section class="panel"><p class="eyebrow">Return Risk</p><ul>{_render_list(list(payload.get("top_return_risk_aspects") or []), empty="No return-risk aspects surfaced yet.")}</ul></section>
        <section class="panel"><p class="eyebrow">Usage Workflows</p><ul>{_render_list(list(workflows or []), empty="No usage workflows surfaced yet.")}</ul></section>
      </section>
      <section class="panel"><p class="eyebrow">Causal Hypotheses</p><ul>{_render_list(list(hypotheses or []), empty="No causal hypotheses surfaced yet.")}</ul></section>
    </main>
  </body>
</html>"""
