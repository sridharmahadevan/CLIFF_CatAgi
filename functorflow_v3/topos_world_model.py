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


def _render_html(payload: dict[str, object], *, dashboard_path: Path) -> str:
    esc = escape
    base_href = _rel(Path(str(payload.get("base_artifact_path") or "")), start=dashboard_path.parent) if payload.get("base_artifact_path") else ""
    psr_href = _rel(Path(str(payload.get("psr_path") or "")), start=dashboard_path.parent) if payload.get("psr_path") else ""
    summary_href = _rel(Path(str(payload.get("summary_path") or "")), start=dashboard_path.parent) if payload.get("summary_path") else ""
    route = str(payload.get("route_name") or "CLIFF")
    query = str(payload.get("query") or "Topos World Model")
    model_family = str(payload.get("model_family") or "topos_world_model")
    link_markup = "\n".join(
        item
        for item in (
            f'<a href="{esc(base_href)}" target="_blank" rel="noreferrer">Open route dashboard</a>' if base_href else "",
            f'<a href="{esc(psr_href)}" target="_blank" rel="noreferrer">Open PSR bundle</a>' if psr_href else "",
            f'<a href="{esc(summary_href)}" target="_blank" rel="noreferrer">Open route summary</a>' if summary_href else "",
        )
        if item
    )
    summary = dict(payload.get("summary") or {})
    chips = "\n".join(
        f'<span class="chip">{esc(str(key).replace("_", " "))}: {esc(str(value))}</span>'
        for key, value in summary.items()
        if value not in (None, "")
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{esc(query)} - Topos World Model</title>
    <style>
      body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f5ef; color: #18222c; }}
      main {{ width: min(1080px, calc(100vw - 32px)); margin: 32px auto; display: grid; gap: 16px; }}
      section {{ background: rgba(255,255,255,.92); border: 1px solid #d7d0c2; border-radius: 8px; padding: 22px; }}
      .eyebrow {{ margin: 0 0 8px; text-transform: uppercase; letter-spacing: .12em; color: #7c3f1d; font-size: 12px; }}
      h1, p {{ margin: 0; }}
      h1 {{ font-size: 32px; line-height: 1.12; }}
      .trace {{ margin-top: 12px; color: #596877; line-height: 1.55; }}
      .chips, .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
      .chip {{ border: 1px solid #d7d0c2; background: #f0eadf; border-radius: 999px; padding: 8px 11px; font-size: 13px; }}
      a {{ color: #17634f; font-weight: 700; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f0e8; padding: 14px; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <main>
      <section>
        <p class="eyebrow">CLIFF Topos World Model</p>
        <h1>{esc(query)}</h1>
        <p class="trace">Built from the existing {esc(route)} route output using the {esc(model_family)} view. This preserves the original answer while exposing the local-state/gluing artifact for downstream inspection.</p>
        <div class="chips">{chips}</div>
        <div class="links">{link_markup}</div>
      </section>
      <section>
        <p class="eyebrow">World Model Payload</p>
        <pre>{esc(json.dumps(payload, indent=2))}</pre>
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
