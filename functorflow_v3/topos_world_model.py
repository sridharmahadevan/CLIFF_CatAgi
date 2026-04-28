"""Route-level Topos World Model artifact helpers for CLIFF."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from html import escape
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rel(path: Path | None, *, start: Path) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except ValueError:
        return str(path.resolve())


def _load_json(path_text: object) -> dict[str, object]:
    text = str(path_text or "").strip()
    if not text:
        return {}
    path = Path(text).expanduser()
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _psr_view_path(psr_path: Path | None) -> Path | None:
    if psr_path is None or not psr_path.exists():
        return psr_path
    html_path = psr_path.with_name("topos_psr_bundle.html")
    return html_path if html_path.exists() else psr_path


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value is None or value == "":
        return "not available"
    return str(value)


def _metric(label: str, value: object, note: str = "") -> str:
    note_markup = f'<span>{escape(note)}</span>' if note else ""
    return (
        '<div class="metric">'
        f'<p>{escape(label)}</p>'
        f'<strong>{escape(_format_value(value))}</strong>'
        f"{note_markup}"
        "</div>"
    )


def _render_link(href: str, label: str) -> str:
    if not href:
        return ""
    return f'<a href="{escape(href)}" target="_blank" rel="noreferrer">{escape(label)}</a>'


def _render_restrictions(rows: list[object]) -> str:
    clean_rows = [_as_dict(row) for row in rows[:6] if isinstance(row, dict)]
    if not clean_rows:
        return '<p class="empty">No restriction diagnostics were published with this run.</p>'
    rendered = []
    for row in clean_rows:
        source = row.get("source_context") or row.get("source") or "source"
        target = row.get("target_context") or row.get("target") or "target"
        compatible = bool(row.get("compatible"))
        status = "compatible" if compatible else "needs review"
        gap = row.get("mean_abs_gap", row.get("max_abs_gap", ""))
        rendered.append(
            '<div class="restriction-row">'
            f'<span>{escape(str(source))} -> {escape(str(target))}</span>'
            f'<strong class="status {"ok" if compatible else "warn"}">{escape(status)}</strong>'
            f'<em>gap {escape(_format_value(gap))}</em>'
            "</div>"
        )
    return "\n".join(rendered)


def _render_query_plan(query_plan: dict[str, object]) -> str:
    fields = (
        ("Product", query_plan.get("product_name")),
        ("Retrieval query", query_plan.get("retrieval_query")),
        ("Normalized query", query_plan.get("normalized_query")),
        ("Target documents", query_plan.get("target_documents")),
    )
    rows = [
        f'<div><span>{escape(label)}</span><strong>{escape(_format_value(value))}</strong></div>'
        for label, value in fields
        if value not in (None, "")
    ]
    if not rows:
        return '<p class="empty">No route plan details were published with this run.</p>'
    return "\n".join(rows)


def _render_html(payload: dict[str, object], *, dashboard_path: Path) -> str:
    esc = escape
    base_href = _rel(Path(str(payload.get("base_artifact_path") or "")), start=dashboard_path.parent) if payload.get("base_artifact_path") else ""
    psr_path = Path(str(payload.get("psr_path") or "")) if payload.get("psr_path") else None
    psr_href = _rel(_psr_view_path(psr_path), start=dashboard_path.parent) if psr_path else ""
    raw_psr_href = _rel(psr_path, start=dashboard_path.parent) if psr_path else ""
    summary_href = _rel(Path(str(payload.get("summary_path") or "")), start=dashboard_path.parent) if payload.get("summary_path") else ""
    route = str(payload.get("route_name") or "CLIFF")
    query = str(payload.get("query") or "Topos World Model")
    model_family = str(payload.get("model_family") or "topos_world_model")
    extra = _as_dict(payload.get("extra"))
    route_decision = _as_dict(extra.get("route_decision"))
    query_plan = _as_dict(extra.get("query_plan"))
    psr_payload = _load_json(payload.get("psr_path"))
    psr_summary = _as_dict(psr_payload.get("summary"))
    restrictions = _as_list(psr_payload.get("restriction_diagnostics"))
    local_hankels = _as_list(psr_payload.get("local_hankel_family"))
    review_count = int(psr_summary.get("n_review_records") or psr_summary.get("n_episodes") or 0) if psr_summary else "not available"
    context_view_count = int(psr_summary.get("n_context_projected_views") or 0) if psr_summary else 0
    if psr_summary and context_view_count <= 0:
        context_view_count = sum(
            int(_as_dict(row).get("n_episode_views") or 0)
            for row in local_hankels
            if isinstance(row, dict)
        )
    compatibility = (
        f"{int(psr_summary.get('n_compatible_restrictions', 0))}/"
        f"{int(psr_summary.get('n_restriction_checks', 0))}"
        if psr_summary
        else "not available"
    )
    route_label = route.replace("_", " ")
    links = "\n".join(
        item
        for item in (
            _render_link(base_href, "Route dashboard"),
            _render_link(psr_href, "PSR bundle"),
            _render_link(raw_psr_href, "Raw PSR JSON") if raw_psr_href and raw_psr_href != psr_href else "",
            _render_link(summary_href, "Route summary"),
        )
        if item
    )
    metrics = "\n".join(
        (
            _metric("Route", route_label),
            _metric("Model family", model_family.replace("_", " ")),
            _metric("Reviews used", review_count),
            _metric("Context views", context_view_count if psr_summary else "not available"),
            _metric("Local contexts", psr_summary.get("n_contexts", "not available")),
            _metric("Mean local rank", psr_summary.get("mean_rank", "not available")),
            _metric("Restriction compatibility", compatibility),
        )
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(query)} - Topos World Model</title>
    <style>
      :root {{
        --ink: #172026;
        --muted: #5e6b73;
        --paper: #fbf7ef;
        --panel: #ffffff;
        --line: #d8d1c5;
        --green: #17634f;
        --blue: #245c83;
        --rust: #9a4b2c;
        --soft-green: #e8f1ed;
        --soft-blue: #e8f0f5;
        --soft-rust: #f5ebe5;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--paper);
        color: var(--ink);
      }}
      main {{
        width: min(1180px, calc(100vw - 32px));
        margin: 30px auto 44px;
        display: grid;
        gap: 18px;
      }}
      section {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 24px;
      }}
      h1, h2, h3, p {{ margin: 0; }}
      h1 {{ max-width: 920px; font-size: clamp(30px, 4vw, 52px); line-height: 1.02; font-weight: 780; }}
      h2 {{ font-size: 20px; line-height: 1.2; }}
      h3 {{ font-size: 15px; }}
      .eyebrow {{
        margin-bottom: 10px;
        color: var(--rust);
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
      }}
      .hero {{
        min-height: 330px;
        display: grid;
        align-content: end;
        gap: 18px;
        background:
          linear-gradient(120deg, rgba(255,255,255,0.96), rgba(255,255,255,0.72)),
          radial-gradient(circle at 88% 14%, rgba(36,92,131,0.18), transparent 30%),
          linear-gradient(140deg, #fdfbf7, #eef5f1 54%, #f6eee8);
      }}
      .trace {{ max-width: 850px; color: var(--muted); line-height: 1.62; }}
      .links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
      a {{
        display: inline-flex;
        align-items: center;
        min-height: 38px;
        border: 1px solid #b9d0c7;
        border-radius: 999px;
        padding: 8px 12px;
        color: var(--green);
        background: var(--soft-green);
        font-weight: 800;
        text-decoration: none;
      }}
      a:hover {{ text-decoration: underline; }}
      .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }}
      .metric {{
        min-height: 112px;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        display: grid;
        align-content: space-between;
        gap: 10px;
        background: #fdfbf8;
      }}
      .metric p, .plan span {{ color: var(--muted); font-size: 12px; font-weight: 750; text-transform: uppercase; }}
      .metric strong {{ font-size: 22px; line-height: 1.1; overflow-wrap: anywhere; }}
      .metric span {{ color: var(--muted); font-size: 13px; }}
      .world-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
      .stage {{
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        padding: 16px;
        background: var(--soft-blue);
      }}
      .stage:nth-child(2) {{ border-left-color: var(--green); background: var(--soft-green); }}
      .stage:nth-child(3) {{ border-left-color: var(--rust); background: var(--soft-rust); }}
      .stage p {{ margin-top: 8px; color: #40505a; line-height: 1.55; }}
      .split {{ display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 16px; }}
      .plan {{ display: grid; gap: 10px; }}
      .plan div {{
        display: grid;
        grid-template-columns: 150px minmax(0, 1fr);
        gap: 12px;
        align-items: start;
        border-bottom: 1px solid #ece6dc;
        padding-bottom: 10px;
      }}
      .plan strong {{ overflow-wrap: anywhere; }}
      .restriction-list {{ display: grid; gap: 8px; }}
      .restriction-row {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) 116px 86px;
        gap: 10px;
        align-items: center;
        border: 1px solid #e2ddd3;
        border-radius: 8px;
        padding: 10px 12px;
      }}
      .restriction-row span {{ overflow-wrap: anywhere; }}
      .restriction-row em {{ color: var(--muted); font-style: normal; text-align: right; }}
      .status {{ font-size: 13px; text-transform: uppercase; }}
      .status.ok {{ color: var(--green); }}
      .status.warn {{ color: var(--rust); }}
      details {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fffdfa;
      }}
      summary {{ cursor: pointer; padding: 14px 16px; font-weight: 800; }}
      pre {{
        margin: 0;
        max-height: 420px;
        overflow: auto;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        background: #f4f0e8;
        border-top: 1px solid var(--line);
        padding: 14px;
      }}
      .empty {{ color: var(--muted); line-height: 1.55; }}
      @media (max-width: 980px) {{
        .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .world-grid, .split {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 560px) {{
        main {{ width: min(100vw - 22px, 1180px); margin-top: 12px; }}
        section {{ padding: 18px; }}
        .metrics {{ grid-template-columns: 1fr; }}
        .plan div, .restriction-row {{ grid-template-columns: 1fr; }}
        .restriction-row em {{ text-align: left; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <p class="eyebrow">CLIFF Topos World Model</p>
        <h1>{esc(query)}</h1>
        <p class="trace">A route-level world-model artifact for the {esc(route_label)} run. It keeps the original product-feedback answer available, then foregrounds the local predictive states, restriction checks, and gluing evidence that make the result inspectable.</p>
        <div class="links">{links}</div>
      </section>
      <section>
        <p class="eyebrow">Run State</p>
        <div class="metrics">{metrics}</div>
      </section>
      <section>
        <p class="eyebrow">World Model Construction</p>
        <div class="world-grid">
          <div class="stage">
            <h3>1. Evidence Base</h3>
            <p>The product-feedback route gathers review evidence and records the retrieval plan that shaped the local corpus.</p>
          </div>
          <div class="stage">
            <h3>2. Local PSR Family</h3>
            <p>{esc(str(len(local_hankels)))} local Hankel slice(s) are available for the contexts induced by aspects, outcomes, and usage workflows.</p>
          </div>
          <div class="stage">
            <h3>3. Gluing State</h3>
            <p>Restriction diagnostics compare overlapping local states so downstream inspection can see where the model coheres and where it needs review.</p>
          </div>
        </div>
      </section>
      <section class="split">
        <div>
          <p class="eyebrow">Route Plan</p>
          <div class="plan">{_render_query_plan(query_plan)}</div>
          {f'<p class="trace" style="margin-top:14px;">Router rationale: {esc(str(route_decision.get("rationale") or ""))}</p>' if route_decision.get("rationale") else ""}
        </div>
        <div>
          <p class="eyebrow">Restriction Diagnostics</p>
          <div class="restriction-list">{_render_restrictions(restrictions)}</div>
        </div>
      </section>
      <section>
        <details>
          <summary>Raw artifact payload</summary>
          <pre>{esc(json.dumps(payload, indent=2))}</pre>
        </details>
      </section>
    </main>
  </body>
</html>
"""


def materialize_topos_world_model(
    *,
    query: str,
    route_name: str,
    route_outdir: Path,
    base_artifact_path: Path | None,
    summary_path: Path | None,
    psr_path: Path | None = None,
    model_family: str = "topos_world_model",
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a route-level Topos World Model JSON plus a small dashboard."""

    outdir = route_outdir / "topos_world_model"
    outdir.mkdir(parents=True, exist_ok=True)
    dashboard_path = outdir / "topos_world_model.html"
    payload = {
        "query": query,
        "route_name": route_name,
        "model_family": model_family,
        "base_artifact_path": str(base_artifact_path) if base_artifact_path else "",
        "summary_path": str(summary_path) if summary_path else "",
        "psr_path": str(psr_path) if psr_path else "",
        "summary": {
            "has_base_dashboard": bool(base_artifact_path and base_artifact_path.exists()),
            "has_psr_bundle": bool(psr_path and psr_path.exists()),
            "route": route_name,
        },
        "extra": extra or {},
    }
    _write_json(outdir / "topos_world_model.json", payload)
    dashboard_path.write_text(_render_html(payload, dashboard_path=dashboard_path), encoding="utf-8")
    return dashboard_path


def asdict_safe(value: object) -> dict[str, object]:
    try:
        return dict(asdict(value))
    except Exception:
        return {}
