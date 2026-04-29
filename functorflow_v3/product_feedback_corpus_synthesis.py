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
    topos_psr_path = feedback_result.topos_psr_path or (base_outdir / "topos_psr_hankel.json")
    topos_psr = (
        json.loads(topos_psr_path.read_text(encoding="utf-8"))
        if topos_psr_path.exists()
        else {}
    )

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
        "report_path": str(feedback_result.report_path),
        "topos_summary": dict(topos_psr.get("summary") or {}),
        "topos_path": str(topos_psr_path) if topos_psr_path.exists() else "",
        "topos_view_path": str(topos_psr_path.with_name("topos_psr_bundle.html")) if topos_psr_path.with_name("topos_psr_bundle.html").exists() else "",
        "review_episodes_path": str(feedback_result.review_episodes_path) if feedback_result.review_episodes_path else "",
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


def _short_action(action: object) -> str:
    text = " ".join(str(action or "").strip().split())
    return text[:1].upper() + text[1:] if text else ""


def _primary_repair(payload: dict[str, object]) -> dict[str, object]:
    hypotheses = [
        dict(item)
        for item in list(payload.get("causal_hypotheses") or [])
        if isinstance(item, dict)
    ]
    risk_aspects = [str(item) for item in list(payload.get("top_return_risk_aspects") or [])]
    negative_aspects = [str(item) for item in list(payload.get("top_negative_aspects") or [])]
    positive_aspects = [str(item) for item in list(payload.get("top_positive_aspects") or [])]
    topos_summary = dict(payload.get("topos_summary") or {})
    restriction_total = int(topos_summary.get("n_restriction_checks") or 0)
    restriction_ok = int(topos_summary.get("n_compatible_restrictions") or 0)
    restriction_tense = max(0, restriction_total - restriction_ok)

    reducing = [
        item
        for item in hypotheses
        if str(item.get("relation") or "").upper() == "REDUCES"
    ]
    candidates = reducing or hypotheses
    candidates.sort(
        key=lambda item: (
            -float(item.get("confidence") or 0.0),
            -int(item.get("support_count") or 0),
            str(item.get("src") or ""),
        )
    )
    best = candidates[0] if candidates else {}
    best_action = _short_action(best.get("recommended_action"))
    if not best_action:
        if bool(payload.get("return_warning_recommended")) or risk_aspects:
            best_action = "Offer a return-safe correction path and collect the missing reason before refunding."
        elif positive_aspects:
            best_action = f"Preserve the satisfied local state around {positive_aspects[0]} and monitor weaker contexts."
        else:
            best_action = "Collect a clearer return reason before choosing refund, replacement, or product-page correction."

    if reducing:
        problem = str(best.get("src") or (negative_aspects[0] if negative_aspects else "local experience friction"))
        explanation = (
            f"The strongest repair signal is {problem}. Prometheus treats it like a return-reason state: "
            "find the local obstruction, then recommend the correction that should move that state toward satisfaction."
        )
    elif negative_aspects:
        problem = f"{negative_aspects[0]} concern"
        explanation = (
            f"The corpus is mostly successful, but {negative_aspects[0]} is the clearest local weakness. "
            "The correction should preserve the positive experience while tightening that local state."
        )
    else:
        problem = "mostly satisfied experience"
        explanation = (
            "The retrieved corpus glues into a mostly satisfied product experience. "
            "The best correction is conservative: reinforce the conditions under which the product works."
        )

    return {
        "problem": problem,
        "action": best_action,
        "confidence": float(best.get("confidence") or payload.get("overall_score") or 0.0),
        "support": int(best.get("support_count") or payload.get("feedback_count") or 0),
        "tense_restrictions": restriction_tense,
        "top_positive": positive_aspects[0] if positive_aspects else "",
        "top_negative": negative_aspects[0] if negative_aspects else "",
        "explanation": explanation,
    }


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
    topos_view_href = _relative_href(Path(str(payload.get("topos_view_path") or "")), start=dashboard_path.parent) if payload.get("topos_view_path") else ""
    raw_topos_href = _relative_href(Path(str(payload.get("topos_path") or "")), start=dashboard_path.parent) if payload.get("topos_path") else ""
    review_episodes_href = _relative_href(Path(str(payload.get("review_episodes_path") or "")), start=dashboard_path.parent) if payload.get("review_episodes_path") else ""
    workflows = [item.get("summary") if isinstance(item, dict) else item for item in payload.get("usage_workflows") or []]
    hypotheses = []
    for item in payload.get("causal_hypotheses") or []:
        if isinstance(item, dict):
            statement = item.get("statement")
            if not statement:
                src = str(item.get("src") or "local state")
                relation = str(item.get("relation") or "AFFECTS").lower()
                dst = str(item.get("dst") or "satisfaction")
                action = _short_action(item.get("recommended_action"))
                statement = f"{src} {relation} {dst}"
                if action:
                    statement = f"{statement}; correction: {action}"
            hypotheses.append(statement)
        else:
            hypotheses.append(item)
    topos_summary = dict(payload.get("topos_summary") or {})
    topos_review_count = int(topos_summary.get("n_review_records") or topos_summary.get("n_episodes") or 0)
    topos_context_views = int(topos_summary.get("n_context_projected_views") or 0)
    repair = _primary_repair(payload)
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
      :root {{ --ink:#172026; --muted:#5f6b73; --paper:#f4f6f1; --card:#fff; --line:#dce1dc; --accent:#226b5f; --good:#1f7a52; --warn:#a46416; }}
      * {{ box-sizing:border-box; }}
      body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--paper); }}
      main {{ width:min(1180px, calc(100vw - 32px)); margin:28px auto 48px; display:grid; gap:18px; }}
      .panel {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:24px; }}
      .hero {{ display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:18px; align-items:stretch; }}
      .decision {{ background:#17322e; color:#fff; border-color:#17322e; display:grid; gap:12px; align-content:start; }}
      .decision .trace,.decision .eyebrow {{ color:#d8eee7; }}
      .decision strong {{ font-size:28px; line-height:1.15; }}
      .story {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
      .story-card {{ border:1px solid #e3e8e2; border-radius:8px; padding:16px; background:#fbfcf8; display:grid; gap:10px; align-content:start; }}
      .story-card strong {{ font-size:20px; line-height:1.2; }}
      .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
      h1,h2,p,ul {{ margin:0; }}
      h1 {{ max-width:850px; font-size:clamp(32px,4vw,56px); line-height:1.03; letter-spacing:0; }}
      .eyebrow {{ text-transform:uppercase; font-size:12px; font-weight:800; margin-bottom:10px; color:var(--accent); }}
      .trace,.empty {{ color:var(--muted); line-height:1.6; }}
      .chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
      .chip {{ border:1px solid #ccd5d0; border-radius:999px; padding:8px 12px; background:#f7faf6; color:#3d4b45; font-weight:700; }}
      .chip.good {{ border-color:#b7d9c4; background:#eaf6ee; color:var(--good); }}
      .chip.warn {{ border-color:#ead1aa; background:#fff6e9; color:var(--warn); }}
      ul {{ padding-left:20px; display:grid; gap:8px; }}
      .textbook-list {{ padding-left:20px; display:grid; gap:10px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      a {{ color:var(--accent); text-decoration:none; font-weight:700; }}
      a:hover {{ text-decoration:underline; }}
      .links {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:12px; }}
      .drilldown {{ border-style:dashed; }}
      @media (max-width:920px) {{ .hero,.story,.grid {{ grid-template-columns:1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel hero">
        <div>
        <p class="eyebrow">Product Experience Repair</p>
        <h1>{esc(payload.get("product_name") or payload.get("query") or "Product feedback synthesis")}</h1>
        <p class="trace">Prometheus treats the retrieved reviews as a customer-experience corpus. Instead of opening with every PSR diagnostic, it first asks what local experience state the customer is in and what correction would repair it.</p>
        <div class="chips">
          <span class="chip">verdict: {esc(payload.get("verdict"))}</span>
          <span class="chip">overall score: {esc(payload.get("overall_score"))}</span>
          <span class="chip">{esc(payload.get("feedback_count") or 0)} reviews synthesized</span>
          <span class="chip {'warn' if payload.get("return_warning_recommended") else 'good'}">return warning: {esc(payload.get("return_warning_recommended"))}</span>
        </div>
        </div>
        <section class="panel decision">
          <p class="eyebrow">Recommended Correction</p>
          <strong>{esc(repair.get("action"))}</strong>
          <p class="trace"><strong>{esc(repair.get("problem"))}.</strong> {esc(repair.get("explanation"))}</p>
          <p class="trace">Confidence {esc(f"{float(repair.get('confidence') or 0.0):.2f}")}; support {esc(repair.get("support"))} review signals.</p>
        </section>
      </section>
      <section class="panel">
        <p class="eyebrow">Experience Repair Model</p>
        <div class="story">
          <article class="story-card">
            <span class="chip">1. Local experience</span>
            <strong>{esc(repair.get("problem"))}</strong>
            <p class="trace">Return reasons and review snippets become local states such as comfort, assembly, fit, durability, use, and decision.</p>
          </article>
          <article class="story-card">
            <span class="chip">2. Topos check</span>
            <strong>{esc(repair.get("tense_restrictions"))} tense restriction(s)</strong>
            <p class="trace">The system checks whether those local states glue into one stable product experience or expose a useful obstruction.</p>
          </article>
          <article class="story-card">
            <span class="chip">3. Correction</span>
            <strong>{esc(repair.get("action"))}</strong>
            <p class="trace">Counterfactual reasoning asks which correction would turn the negative local state into a better-supported satisfaction state.</p>
          </article>
        </div>
      </section>
      <section class="panel drilldown">
        <p class="eyebrow">Technical Drill-Downs</p>
        <p class="trace">Detailed dashboards and raw data remain available for inspection, but they are no longer the first thing the viewer has to parse.</p>
        <div class="links">
          {f'<a href="{esc(dashboard_href)}" target="_blank" rel="noreferrer">Open product feedback dashboard</a>' if dashboard_href else ''}
          {f'<a href="{esc(report_href)}" target="_blank" rel="noreferrer">Open product feedback report</a>' if report_href else ''}
          {f'<a href="{esc(topos_view_href or raw_topos_href)}" target="_blank" rel="noreferrer">Open topos PSR bundle</a>' if (topos_view_href or raw_topos_href) else ''}
          {f'<a href="{esc(raw_topos_href)}" target="_blank" rel="noreferrer">Raw PSR JSON</a>' if raw_topos_href else ''}
          {f'<a href="{esc(review_episodes_href)}" target="_blank" rel="noreferrer">Open review episodes</a>' if review_episodes_href else ''}
        </div>
      </section>
      <section class="panel">
        <p class="eyebrow">Topos PSR Summary</p>
        <p class="trace">The presheaf-valued predictive state layer is retained as supporting machinery underneath the correction recommendation.</p>
        <div class="chips">
          <span class="chip">reviews used: {esc(topos_review_count)}</span>
          <span class="chip">context views: {esc(topos_context_views)}</span>
          <span class="chip">contexts: {esc(int(topos_summary.get("n_contexts", 0)))}</span>
          <span class="chip">mean rank: {esc(topos_summary.get("mean_rank", 0.0))}</span>
          <span class="chip">restriction checks: {esc(f"{int(topos_summary.get('n_compatible_restrictions', 0))}/{int(topos_summary.get('n_restriction_checks', 0))}")}</span>
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
