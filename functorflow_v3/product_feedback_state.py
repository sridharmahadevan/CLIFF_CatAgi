"""Persistent Prometheus-style state for product-feedback world models."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


def _read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _query_context(query: str, *, domain: str, context_ids: list[object]) -> str:
    lowered = query.lower()
    available = {str(item) for item in context_ids}
    ordered_rules = (
        ("taste", ("taste", "tasty", "flavor", "flavour", "texture", "sweet", "bitter", "aroma")),
        ("sit", ("comfortable", "comfort", "sofa", "couch", "seat", "sitting")),
        ("assemble", ("assemble", "assembly", "setup", "instructions")),
        ("fit", ("fit", "size", "sizing", "tight", "loose")),
        ("durability", ("durable", "durability", "quality", "long run", "long-term")),
        ("decision", ("worth", "value", "recommend", "return")),
    )
    for context_id, tokens in ordered_rules:
        if context_id in available and any(token in lowered for token in tokens):
            return context_id
    if domain in {"food", "beverage"} and "taste" in available:
        return "taste"
    if domain == "furniture" and "sit" in available:
        return "sit"
    return "use" if "use" in available else ("review" if "review" in available else (sorted(available)[0] if available else "review"))


def _context_support(topos_psr: dict[str, object], context_id: str) -> dict[str, object]:
    for row in list(topos_psr.get("local_hankel_family") or []):
        if not isinstance(row, dict) or str(row.get("context_id") or "") != context_id:
            continue
        tests = list(row.get("tests") or [])
        histories = list(row.get("histories") or [])
        svd = dict(row.get("svd") or {})
        return {
            "context_id": context_id,
            "episode_views": int(row.get("n_episode_views") or 0),
            "history_count": len(histories),
            "test_count": len(tests),
            "rank": int(svd.get("rank") or 0),
            "top_tests": [
                {
                    "signature": str(dict(test).get("signature") or ""),
                    "kind": str(dict(test).get("test_kind") or "test"),
                    "support": int(dict(test).get("support") or 0),
                }
                for test in tests[:8]
                if isinstance(test, dict)
            ],
        }
    return {"context_id": context_id, "episode_views": 0, "history_count": 0, "test_count": 0, "rank": 0, "top_tests": []}


def _restriction_support(topos_psr: dict[str, object], context_id: str) -> dict[str, object]:
    relevant = [
        dict(row)
        for row in list(topos_psr.get("restriction_diagnostics") or [])
        if isinstance(row, dict)
        and (str(row.get("source_context") or "") == context_id or str(row.get("target_context") or "") == context_id)
    ]
    if not relevant:
        return {"checks": 0, "compatible": 0, "max_gap": 0.0}
    return {
        "checks": len(relevant),
        "compatible": sum(1 for row in relevant if bool(row.get("compatible"))),
        "max_gap": max(float(row.get("max_abs_gap") or 0.0) for row in relevant),
    }


def _counterfactual_diagnostics(counterfactuals: dict[str, object], query_context: str) -> dict[str, object]:
    summary = dict(counterfactuals.get("summary") or {})
    supported = [dict(row) for row in list(counterfactuals.get("counterfactuals") or []) if isinstance(row, dict)]
    exploratory = [
        dict(row) for row in list(counterfactuals.get("exploratory_counterfactuals") or []) if isinstance(row, dict)
    ]
    query_supported = [row for row in supported if str(row.get("aspect") or "") == query_context]
    query_exploratory = [row for row in exploratory if str(row.get("aspect") or "") == query_context]
    best_probe = (query_supported or supported or query_exploratory or exploratory or [{}])[0]
    return {
        "supported_count": int(summary.get("counterfactual_count") or len(supported)),
        "strict_candidate_count": int(summary.get("candidate_count") or 0),
        "exploratory_count": int(summary.get("exploratory_count") or len(exploratory)),
        "query_context_supported_count": len(query_supported),
        "query_context_exploratory_count": len(query_exploratory),
        "best_probe": {
            "aspect": str(best_probe.get("aspect") or ""),
            "evidence_status": str(best_probe.get("evidence_status") or ""),
            "support_tier": str(best_probe.get("support_tier") or ""),
            "estimated_satisfaction_gain": best_probe.get("estimated_satisfaction_gain"),
            "title": str(best_probe.get("title") or ""),
        },
    }


def _confidence_and_actions(
    *,
    query: str,
    query_context: str,
    target_documents: int,
    context_support: dict[str, object],
    restriction_support: dict[str, object],
    counterfactual_support: dict[str, object],
) -> tuple[str, list[str], list[dict[str, object]]]:
    reasons: list[str] = []
    actions: list[dict[str, object]] = []
    episode_views = int(context_support.get("episode_views") or 0)
    test_count = int(context_support.get("test_count") or 0)
    rank = int(context_support.get("rank") or 0)
    checks = int(restriction_support.get("checks") or 0)
    compatible = int(restriction_support.get("compatible") or 0)
    max_gap = float(restriction_support.get("max_gap") or 0.0)
    if episode_views < max(5, target_documents):
        reasons.append(f"{query_context} context has only {episode_views} supporting view(s)")
    if test_count < 4:
        reasons.append(f"{query_context} Hankel slice has only {test_count} test column(s)")
    if rank <= 1 and test_count >= 3:
        reasons.append(f"{query_context} Hankel rank is low enough to be fragile")
    if checks and compatible < checks:
        reasons.append(f"{query_context} restriction checks glue only {compatible}/{checks}")
    if max_gap >= 0.2:
        reasons.append(f"{query_context} max restriction gap is {_fmt(max_gap)}")
    supported_repairs = int(counterfactual_support.get("supported_count") or 0)
    strict_candidates = int(counterfactual_support.get("strict_candidate_count") or 0)
    exploratory_repairs = int(counterfactual_support.get("exploratory_count") or 0)
    query_supported_repairs = int(counterfactual_support.get("query_context_supported_count") or 0)
    query_exploratory_repairs = int(counterfactual_support.get("query_context_exploratory_count") or 0)
    if supported_repairs == 0 and exploratory_repairs > 0:
        reasons.append("j-do repair layer has only low-confidence exploratory probes")
    elif query_supported_repairs == 0 and query_exploratory_repairs > 0:
        reasons.append(f"{query_context} j-do repair support is exploratory only")
    confidence = "stable" if not reasons else ("thin" if episode_views < 4 or test_count < 3 else "provisional")
    if supported_repairs == 0 and exploratory_repairs > 0 and episode_views >= max(5, target_documents) and test_count >= 4:
        actions.append(
            {
                "action": "counterfactual_probe",
                "label": f"Probe {query_context} repair evidence",
                "rationale": (
                    "The descriptive topos state is stable, but the local j-do layer has no strict supported "
                    f"repairs ({strict_candidates} strict candidate(s), {exploratory_repairs} exploratory probe(s))."
                ),
                "config_patch": {
                    "route": "product_feedback",
                    "product_target_docs": max(target_documents * 2, 12),
                    "product_max_docs": max(target_documents * 5, 40),
                    "analysis_question": (
                        "Search specifically for observed negative local states or tense overlap evidence "
                        f"that can support `{query_context}` j-do repair probes for: {query}"
                    ),
                },
            }
        )
    elif confidence != "stable":
        deeper_target = max(target_documents * 3, 15)
        actions.append(
            {
                "action": "product_feedback_deeper_probe",
                "label": f"Run deeper {query_context}-focused probe",
                "rationale": "; ".join(reasons[:3]) or "The query-relevant local state is still provisional.",
                "config_patch": {
                    "route": "product_feedback",
                    "product_target_docs": deeper_target,
                    "product_max_docs": max(deeper_target * 3, 40),
                    "analysis_question": f"Deepen the product-feedback analysis for: {query}",
                },
            }
        )
    elif supported_repairs == 0 and exploratory_repairs == 0:
        actions.append(
            {
                "action": "accept_answer",
                "label": "Accept current answer",
                "rationale": (
                    f"The {query_context} local state is stable and no counterfactual repair opportunity was isolated."
                ),
                "config_patch": {},
            }
        )
    else:
        actions.append(
            {
                "action": "accept_answer",
                "label": "Accept current answer",
                "rationale": (
                    f"The {query_context} local state has adequate support, compatible restrictions, "
                    "and supported repair semantics where applicable."
                ),
                "config_patch": {},
            }
        )
    return confidence, reasons, actions


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _render_state_html(state: dict[str, object]) -> str:
    diagnostics = dict(state.get("diagnostics") or {})
    query_context = str(diagnostics.get("query_context") or "review")
    support = dict(diagnostics.get("query_context_support") or {})
    restrictions = dict(diagnostics.get("query_context_restrictions") or {})
    counterfactual_support = dict(diagnostics.get("counterfactual_support") or {})
    actions = [dict(item) for item in list(state.get("recommended_actions") or []) if isinstance(item, dict)]
    reason_items = "".join(f"<li>{escape(str(item))}</li>" for item in list(state.get("reasons") or []))
    action_cards = "".join(
        '<article class="action">'
        f'<h3>{escape(str(action.get("label") or action.get("action") or "Next action"))}</h3>'
        f'<p>{escape(str(action.get("rationale") or ""))}</p>'
        f'<pre>{escape(json.dumps(action.get("config_patch") or {}, indent=2))}</pre>'
        "</article>"
        for action in actions
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Prometheus Product Feedback State</title>
    <style>
      body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f8f3ea; }}
      main {{ width:min(1080px, calc(100vw - 32px)); margin:30px auto 48px; display:grid; gap:18px; }}
      section, .action {{ background:#fff; border:1px solid #d9d1c5; border-radius:8px; padding:22px; }}
      h1,h2,h3,p {{ margin:0; }}
      h1 {{ font-size:clamp(30px,4vw,48px); line-height:1.05; }}
      .eyebrow {{ color:#9a4b2c; font-size:12px; font-weight:800; text-transform:uppercase; margin-bottom:10px; }}
      .trace {{ color:#5b6870; line-height:1.6; margin-top:12px; }}
      .metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }}
      .metric {{ border:1px solid #e4ddd2; border-radius:8px; padding:14px; background:#fdfbf8; display:grid; gap:10px; }}
      .metric span {{ color:#5b6870; font-size:12px; font-weight:800; text-transform:uppercase; }}
      .metric strong {{ font-size:24px; }}
      .actions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
      pre {{ white-space:pre-wrap; overflow:auto; background:#f4f0e8; padding:12px; border-radius:8px; }}
      ul {{ margin:0; padding-left:20px; display:grid; gap:8px; }}
      @media (max-width:900px) {{ .metrics,.actions {{ grid-template-columns:1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section>
        <p class="eyebrow">Persistent Product Feedback State</p>
        <h1>{escape(str(state.get("query") or "Product feedback query"))}</h1>
        <p class="trace">Prometheus persisted the query-relevant topos state so future actions can be chosen from diagnostics, not from a cold rerun.</p>
      </section>
      <section>
        <p class="eyebrow">Decision State</p>
        <div class="metrics">
          <div class="metric"><span>Confidence</span><strong>{escape(str(state.get("confidence") or "unknown"))}</strong></div>
          <div class="metric"><span>Query context</span><strong>{escape(query_context)}</strong></div>
          <div class="metric"><span>Context views</span><strong>{escape(_fmt(support.get("episode_views")))}</strong></div>
          <div class="metric"><span>Tests</span><strong>{escape(_fmt(support.get("test_count")))}</strong></div>
          <div class="metric"><span>Restrictions</span><strong>{escape(_fmt(restrictions.get("compatible")))}/{escape(_fmt(restrictions.get("checks")))}</strong></div>
        </div>
      </section>
      <section>
        <p class="eyebrow">Counterfactual State</p>
        <div class="metrics">
          <div class="metric"><span>Supported repairs</span><strong>{escape(_fmt(counterfactual_support.get("supported_count")))}</strong></div>
          <div class="metric"><span>Strict candidates</span><strong>{escape(_fmt(counterfactual_support.get("strict_candidate_count")))}</strong></div>
          <div class="metric"><span>Exploratory probes</span><strong>{escape(_fmt(counterfactual_support.get("exploratory_count")))}</strong></div>
          <div class="metric"><span>Context repairs</span><strong>{escape(_fmt(counterfactual_support.get("query_context_supported_count")))}</strong></div>
          <div class="metric"><span>Context exploratory</span><strong>{escape(_fmt(counterfactual_support.get("query_context_exploratory_count")))}</strong></div>
        </div>
      </section>
      <section>
        <p class="eyebrow">Reasons</p>
        <ul>{reason_items or "<li>No blocking state issues were detected.</li>"}</ul>
      </section>
      <section>
        <p class="eyebrow">Recommended Actions</p>
        <div class="actions">{action_cards}</div>
      </section>
    </main>
  </body>
</html>
"""


def _summary_metrics(state: dict[str, object]) -> dict[str, object]:
    diagnostics = dict(state.get("diagnostics") or {})
    summary = dict(diagnostics.get("topos_summary") or {})
    restrictions = dict(diagnostics.get("query_context_restrictions") or {})
    support = dict(diagnostics.get("query_context_support") or {})
    counterfactuals = dict(diagnostics.get("counterfactual_support") or {})
    return {
        "reviews_used": int(summary.get("n_review_records") or summary.get("n_episodes") or 0),
        "context_views": int(summary.get("n_context_projected_views") or 0),
        "local_contexts": int(summary.get("n_contexts") or 0),
        "mean_rank": float(summary.get("mean_rank") or 0.0),
        "restriction_compatible": int(summary.get("n_compatible_restrictions") or 0),
        "restriction_checks": int(summary.get("n_restriction_checks") or 0),
        "query_context": str(diagnostics.get("query_context") or ""),
        "query_context_views": int(support.get("episode_views") or 0),
        "query_context_checks": int(restrictions.get("checks") or 0),
        "query_context_compatible": int(restrictions.get("compatible") or 0),
        "supported_repairs": int(counterfactuals.get("supported_count") or 0),
        "strict_repair_candidates": int(counterfactuals.get("strict_candidate_count") or 0),
        "exploratory_repairs": int(counterfactuals.get("exploratory_count") or 0),
        "query_context_supported_repairs": int(counterfactuals.get("query_context_supported_count") or 0),
        "query_context_exploratory_repairs": int(counterfactuals.get("query_context_exploratory_count") or 0),
    }


def _load_topos_from_state(state: dict[str, object]) -> dict[str, object]:
    source = dict(state.get("source_artifacts") or {})
    return _read_json(Path(str(source.get("topos_psr_path") or ""))) if source.get("topos_psr_path") else {}


def _incompatible_restrictions(topos_psr: dict[str, object]) -> list[dict[str, object]]:
    rows = [dict(row) for row in list(topos_psr.get("restriction_diagnostics") or []) if isinstance(row, dict)]
    return [row for row in rows if not bool(row.get("compatible"))]


def _transition_policy(parent: dict[str, object], current: dict[str, object]) -> tuple[str, list[str], list[dict[str, object]]]:
    before = _summary_metrics(parent)
    after = _summary_metrics(current)
    reasons: list[str] = []
    actions: list[dict[str, object]] = []
    rank_delta = abs(float(after["mean_rank"]) - float(before["mean_rank"]))
    contexts_changed = int(after["local_contexts"]) != int(before["local_contexts"])
    compatibility_changed = (
        int(after["restriction_compatible"]) != int(before["restriction_compatible"])
        or int(after["restriction_checks"]) != int(before["restriction_checks"])
    )
    query_context_compatibility_changed = (
        int(after["query_context_compatible"]) != int(before["query_context_compatible"])
        or int(after["query_context_checks"]) != int(before["query_context_checks"])
    )
    reviews_added = int(after["reviews_used"]) - int(before["reviews_used"])
    counterfactual_remains_exploratory = (
        int(before["supported_repairs"]) == 0
        and int(after["supported_repairs"]) == 0
        and int(after["exploratory_repairs"]) > 0
    )
    query_context_not_improved = int(after["query_context_views"]) <= int(before["query_context_views"])
    strict_candidates_not_improved = int(after["strict_repair_candidates"]) <= int(before["strict_repair_candidates"])
    if counterfactual_remains_exploratory and query_context_not_improved and strict_candidates_not_improved:
        assessment = "counterfactual_probe_stalled"
        reasons.append(
            "The follow-up did not produce strict j-do repair candidates; counterfactual support remains exploratory."
        )
        reasons.append(
            f"Query-context support did not improve ({before['query_context_views']} -> {after['query_context_views']} view(s))."
        )
        actions.append(
            {
                "action": "accept_descriptive_answer",
                "label": "Accept descriptive answer; stop repair probing",
                "rationale": (
                    "The descriptive topos state can be used, but repeated product-feedback probes are not "
                    "finding observed negative local states or GB-supported repair evidence."
                ),
                "config_patch": {},
            }
        )
        return assessment, reasons, actions
    query_context_has_residual_tension = int(after["query_context_compatible"]) < int(after["query_context_checks"])
    if query_context_has_residual_tension and query_context_not_improved and not query_context_compatibility_changed:
        assessment = "gluing_probe_stalled"
        reasons.append(
            "The follow-up did not improve query-context support or resolve the query-context restriction tension."
        )
        reasons.append(
            f"{after['query_context']} restrictions remain at {after['query_context_compatible']}/{after['query_context_checks']} "
            f"with support {before['query_context_views']} -> {after['query_context_views']} view(s)."
        )
        current_topos = _load_topos_from_state(current)
        residuals = [
            row
            for row in _incompatible_restrictions(current_topos)
            if str(row.get("source_context") or "") == str(after["query_context"])
            or str(row.get("target_context") or "") == str(after["query_context"])
        ] or _incompatible_restrictions(current_topos)
        target = residuals[0] if residuals else {}
        source_context = str(target.get("source_context") or after["query_context"] or "source")
        target_context = str(target.get("target_context") or "target")
        actions.append(
            {
                "action": "accept_with_gluing_caveat",
                "label": "Accept answer with gluing caveat",
                "rationale": (
                    "Repeated broad product-feedback probes are not resolving the incompatible local restriction; "
                    "treat the answer as descriptive and inspect the residual edge manually."
                ),
                "target": {
                    "source_context": source_context,
                    "target_context": target_context,
                    "max_abs_gap": target.get("max_abs_gap", 0.0),
                },
                "config_patch": {},
            }
        )
        return assessment, reasons, actions
    if reviews_added > 0 and not contexts_changed and rank_delta <= 0.25 and not compatibility_changed:
        if int(after["restriction_compatible"]) < int(after["restriction_checks"]):
            assessment = "stable_with_residual_gluing_tension"
            reasons.append(
                "The deeper probe added evidence, but context coverage, rank, and restriction compatibility remained stable."
            )
            reasons.append("At least one restriction is still incompatible, so more broad retrieval is unlikely to be the best next move.")
            current_topos = _load_topos_from_state(current)
            residuals = _incompatible_restrictions(current_topos)
            target = residuals[0] if residuals else {}
            source_context = str(target.get("source_context") or "source")
            target_context = str(target.get("target_context") or "target")
            actions.append(
                {
                    "action": "inspect_residual_restriction",
                    "label": "Inspect incompatible restriction",
                    "rationale": f"Restriction compatibility stayed at {after['restriction_compatible']}/{after['restriction_checks']} after adding {reviews_added} review(s).",
                    "target": {
                        "source_context": source_context,
                        "target_context": target_context,
                        "max_abs_gap": target.get("max_abs_gap", 0.0),
                    },
                    "config_patch": {
                        "route": "product_feedback",
                        "analysis_question": (
                            "Investigate the residual gluing tension between "
                            f"{source_context} and {target_context} for: {current.get('query')}"
                        ),
                    },
                }
            )
            return assessment, reasons, actions
        assessment = "stable_answer"
        reasons.append("The deeper probe added evidence without changing the world-model shape materially.")
        actions.append(
            {
                "action": "accept_answer",
                "label": "Accept stable answer",
                "rationale": "The product-feedback world state appears stable under the deeper probe.",
                "config_patch": {},
            }
        )
        return assessment, reasons, actions
    assessment = "state_changed"
    if contexts_changed:
        reasons.append("The deeper probe changed the local context cover.")
    if rank_delta > 0.25:
        reasons.append(f"Mean local rank shifted by {_fmt(rank_delta)}.")
    if compatibility_changed:
        reasons.append("Restriction compatibility changed across runs.")
    if not reasons:
        reasons.append("The deeper probe changed the persisted state enough to merit inspection.")
    actions.append(
        {
            "action": "inspect_transition",
            "label": "Inspect changed world state",
            "rationale": "; ".join(reasons),
            "config_patch": {},
        }
    )
    return assessment, reasons, actions


def _render_transition_html(transition: dict[str, object]) -> str:
    before = dict(transition.get("parent_metrics") or {})
    after = dict(transition.get("current_metrics") or {})
    actions = [dict(item) for item in list(transition.get("recommended_actions") or []) if isinstance(item, dict)]
    reason_items = "".join(f"<li>{escape(str(item))}</li>" for item in list(transition.get("reasons") or []))
    action_items = "".join(
        '<article class="action">'
        f'<h3>{escape(str(action.get("label") or action.get("action") or "Next action"))}</h3>'
        f'<p>{escape(str(action.get("rationale") or ""))}</p>'
        f'<pre>{escape(json.dumps(action.get("target") or action.get("config_patch") or {}, indent=2))}</pre>'
        "</article>"
        for action in actions
    )
    rows = "".join(
        "<tr>"
        f"<th>{escape(label)}</th>"
        f"<td>{escape(_fmt(before.get(key)))}</td>"
        f"<td>{escape(_fmt(after.get(key)))}</td>"
        "</tr>"
        for key, label in (
            ("reviews_used", "Reviews used"),
            ("context_views", "Context views"),
            ("local_contexts", "Local contexts"),
            ("mean_rank", "Mean rank"),
            ("restriction_compatible", "Compatible restrictions"),
            ("restriction_checks", "Restriction checks"),
            ("query_context_views", "Query-context views"),
            ("supported_repairs", "Supported repairs"),
            ("strict_repair_candidates", "Strict repair candidates"),
            ("exploratory_repairs", "Exploratory repairs"),
        )
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Prometheus Product Feedback State Transition</title>
    <style>
      body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f8f3ea; }}
      main {{ width:min(1080px, calc(100vw - 32px)); margin:30px auto 48px; display:grid; gap:18px; }}
      section,.action {{ background:#fff; border:1px solid #d9d1c5; border-radius:8px; padding:22px; }}
      h1,h2,h3,p {{ margin:0; }}
      h1 {{ font-size:clamp(30px,4vw,48px); line-height:1.05; }}
      .eyebrow {{ color:#9a4b2c; font-size:12px; font-weight:800; text-transform:uppercase; margin-bottom:10px; }}
      .trace {{ color:#5b6870; line-height:1.6; margin-top:12px; }}
      table {{ width:100%; border-collapse:collapse; }}
      th,td {{ padding:10px 12px; border-bottom:1px solid #ece6dc; text-align:left; }}
      th {{ color:#40505a; }}
      .actions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
      pre {{ white-space:pre-wrap; overflow:auto; background:#f4f0e8; padding:12px; border-radius:8px; }}
      @media (max-width:900px) {{ .actions {{ grid-template-columns:1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section>
        <p class="eyebrow">World State Transition</p>
        <h1>{escape(str(transition.get("assessment") or "state transition"))}</h1>
        <p class="trace">Prometheus compared the parent product-feedback world state with this deeper follow-up run.</p>
      </section>
      <section>
        <p class="eyebrow">Metric Delta</p>
        <table><thead><tr><th>Metric</th><th>Parent</th><th>Current</th></tr></thead><tbody>{rows}</tbody></table>
      </section>
      <section>
        <p class="eyebrow">Reasons</p>
        <ul>{reason_items}</ul>
      </section>
      <section>
        <p class="eyebrow">Recommended Policy</p>
        <div class="actions">{action_items}</div>
      </section>
    </main>
  </body>
</html>
"""


def materialize_product_feedback_state_transition(
    *,
    parent_state_path: Path,
    current_state_path: Path,
    outdir: Path,
) -> dict[str, Path]:
    """Compare parent/current product-feedback world states and persist the transition."""

    parent = _read_json(parent_state_path)
    current = _read_json(current_state_path)
    state_dir = outdir / "prometheus_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    assessment, reasons, actions = _transition_policy(parent, current)
    transition = {
        "schema_version": 1,
        "state_kind": "product_feedback_state_transition",
        "assessment": assessment,
        "query": current.get("query") or parent.get("query") or "",
        "parent_state_path": str(parent_state_path),
        "current_state_path": str(current_state_path),
        "parent_metrics": _summary_metrics(parent),
        "current_metrics": _summary_metrics(current),
        "reasons": reasons,
        "recommended_actions": actions,
    }
    json_path = state_dir / "product_feedback_state_transition.json"
    html_path = state_dir / "product_feedback_state_transition.html"
    json_path.write_text(json.dumps(transition, indent=2), encoding="utf-8")
    html_path.write_text(_render_transition_html(transition), encoding="utf-8")
    return {"json_path": json_path, "html_path": html_path}


def materialize_product_feedback_world_state(
    *,
    query: str,
    query_plan: object,
    summary_path: Path,
    product_feedback_result: object | None,
    outdir: Path,
) -> dict[str, Path]:
    """Persist a small decision-state artifact derived from the product-feedback topos bundle."""

    state_dir = outdir / "prometheus_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    query_plan_dict = dict(getattr(query_plan, "__dict__", {}) or {})
    feedback = getattr(product_feedback_result, "product_feedback_result", None) or product_feedback_result
    topos_path = getattr(feedback, "topos_psr_path", None)
    scorecard_path = getattr(feedback, "success_scorecard_path", None)
    counterfactual_path = getattr(feedback, "prometheus_counterfactuals_path", None)
    topos_psr = _read_json(topos_path)
    scorecard = _read_json(scorecard_path)
    counterfactuals = _read_json(counterfactual_path)
    summary = dict(topos_psr.get("summary") or {})
    context_ids = list(summary.get("context_ids") or [])
    query_context = _query_context(str(query), domain=str(summary.get("domain") or ""), context_ids=context_ids)
    support = _context_support(topos_psr, query_context)
    restriction = _restriction_support(topos_psr, query_context)
    counterfactual_support = _counterfactual_diagnostics(counterfactuals, query_context)
    target_documents = int(query_plan_dict.get("target_documents") or 5)
    confidence, reasons, actions = _confidence_and_actions(
        query=str(query),
        query_context=query_context,
        target_documents=target_documents,
        context_support=support,
        restriction_support=restriction,
        counterfactual_support=counterfactual_support,
    )
    state = {
        "schema_version": 1,
        "state_kind": "product_feedback_world_state",
        "query": str(query),
        "query_plan": query_plan_dict,
        "confidence": confidence,
        "reasons": reasons,
        "diagnostics": {
            "query_context": query_context,
            "query_context_support": support,
            "query_context_restrictions": restriction,
            "counterfactual_support": counterfactual_support,
            "topos_summary": summary,
            "scorecard_summary": {
                "verdict": scorecard.get("verdict"),
                "overall_score": scorecard.get("overall_score"),
                "top_positive_aspects": list(scorecard.get("top_positive_aspects") or []),
                "top_negative_aspects": list(scorecard.get("top_negative_aspects") or []),
                "top_return_risk_aspects": list(scorecard.get("top_return_risk_aspects") or []),
            },
        },
        "recommended_actions": actions,
        "source_artifacts": {
            "summary_path": str(summary_path),
            "topos_psr_path": str(topos_path) if topos_path else "",
            "scorecard_path": str(scorecard_path) if scorecard_path else "",
            "counterfactual_path": str(counterfactual_path) if counterfactual_path else "",
        },
    }
    json_path = state_dir / "product_feedback_world_state.json"
    html_path = state_dir / "product_feedback_world_state.html"
    json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    html_path.write_text(_render_state_html(state), encoding="utf-8")
    return {"json_path": json_path, "html_path": html_path}
