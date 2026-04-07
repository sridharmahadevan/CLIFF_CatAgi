"""Static HTML dashboard for BAFFLE product-feedback runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ProductFeedbackVisualizationResult:
    """Generated dashboard assets for one product-feedback run."""

    dashboard_path: Path
    summary_path: Path


def bootstrap_product_feedback_dashboard(
    run_outdir: str | Path,
    *,
    product_name: str = "",
    brand_name: str = "",
    analysis_question: str = "",
    run_status: str = "starting",
    run_status_note: str = "",
    feedback_count: int = 0,
) -> ProductFeedbackVisualizationResult:
    """Write a lightweight placeholder dashboard before full analysis completes."""

    run_path = Path(run_outdir).resolve()
    _write_json(
        run_path / "product_success_scorecard.json",
        {
            "product_name": product_name,
            "brand_name": brand_name,
            "analysis_question": analysis_question,
            "overall_score": None,
            "verdict": "unknown",
            "return_warning_recommended": False,
            "warning_text": "",
            "top_return_risk_aspects": [],
            "top_negative_aspects": [],
            "top_positive_aspects": [],
            "hypothesis_count": 0,
            "workflow_motif_count": 0,
            "run_status": run_status,
            "run_status_note": run_status_note,
        },
    )
    _write_json(
        run_path / "outcome_summary.json",
        {
            "feedback_count": feedback_count,
            "average_rating": None,
            "positive_share": 0.0,
            "negative_share": 0.0,
            "return_risk_rate": 0.0,
            "recommendation_rate": 0.0,
        },
    )
    return generate_product_feedback_dashboard(run_path)


def _escape_html(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_product_visual_asset(run_path: Path) -> dict[str, object]:
    direct_payload = _load_json(run_path / "product_visual_asset.json")
    if direct_payload:
        return direct_payload
    manifest_payload = _load_json(run_path / "feedback_manifest.json")
    nested = manifest_payload.get("product_visual_asset")
    return dict(nested) if isinstance(nested, dict) else {}


def _format_percent(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _format_rating(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.2f}"


def _verdict_tone(verdict: str) -> str:
    return {
        "successful": "success",
        "mixed_positive": "mixed",
        "at_risk": "risk",
        "unsuccessful": "danger",
    }.get(verdict, "neutral")


def _tone_chip(label: str, tone: str) -> str:
    return f'<span class="tone-chip {tone}">{_escape_html(label)}</span>'


def _bar_row(label: str, value: float, *, tone: str = "neutral") -> str:
    width = max(0.0, min(100.0, 100.0 * value))
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{_escape_html(label)}</div>'
        '<div class="bar-track">'
        f'<div class="bar-fill {tone}" style="width:{width:.1f}%"></div>'
        "</div>"
        f'<div class="bar-value">{_escape_html(_format_percent(value))}</div>'
        "</div>"
    )


def _metric_card(label: str, value: object, *, tone: str = "neutral") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{_escape_html(label)}</div>'
        f'<div class="metric-value {tone}">{_escape_html(value)}</div>'
        "</div>"
    )


def _aspect_rows(aspect_summary: dict[str, dict[str, object]]) -> str:
    if not aspect_summary:
        return '<tr><td colspan="6" class="empty">No aspect data available.</td></tr>'
    rows = []
    ranked = sorted(
        aspect_summary.items(),
        key=lambda item: (-int(item[1].get("mentions", 0)), item[0]),
    )
    for aspect, stats in ranked:
        rows.append(
            "<tr>"
            f"<td>{_escape_html(aspect)}</td>"
            f"<td>{int(stats.get('mentions', 0))}</td>"
            f"<td>{int(stats.get('positive_mentions', 0))}</td>"
            f"<td>{int(stats.get('negative_mentions', 0))}</td>"
            f"<td>{int(stats.get('return_risk_mentions', 0))}</td>"
            f"<td>{int(stats.get('recommendation_mentions', 0))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _hypothesis_rows(hypotheses: list[dict[str, object]]) -> str:
    if not hypotheses:
        return '<tr><td colspan="5" class="empty">No hypotheses generated.</td></tr>'
    rows = []
    for item in hypotheses[:10]:
        tone = "risk" if str(item.get("relation")) == "INCREASES" and "return" in str(item.get("dst", "")).lower() else "neutral"
        rows.append(
            "<tr>"
            f"<td>{_escape_html(item.get('src', ''))}</td>"
            f"<td>{_tone_chip(str(item.get('relation', '')), tone)}</td>"
            f"<td>{_escape_html(item.get('dst', ''))}</td>"
            f"<td>{int(item.get('support_count', 0))}</td>"
            f"<td>{float(item.get('confidence', 0.0)):.2f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _ablation_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return '<tr><td colspan="8" class="empty">No ablation rows available.</td></tr>'
    html_rows = []
    for row in rows:
        verdict = str(row.get("verdict") or "unknown")
        html_rows.append(
            "<tr>"
            f"<td>{_escape_html(row.get('label', ''))}</td>"
            f"<td>{_tone_chip(verdict.replace('_', ' '), _verdict_tone(verdict))}</td>"
            f"<td>{float(row.get('overall_score', 0.0)):.3f}</td>"
            f"<td>{float(row.get('risk_adjustment_vs_baseline', 0.0)):+.3f}</td>"
            f"<td>{'yes' if row.get('return_warning_recommended') else 'no'}</td>"
            f"<td>{_escape_html(row.get('top_driver', ''))}</td>"
            f"<td>{int(row.get('hypothesis_count', 0))}</td>"
            f"<td>{_escape_html(_format_percent(row.get('evidence_coverage')))}</td>"
            "</tr>"
        )
    return "".join(html_rows)


def _workflow_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return '<tr><td colspan="3" class="empty">No workflow motifs available.</td></tr>'
    html_rows = []
    for row in rows[:8]:
        stages = " -> ".join(str(stage) for stage in row.get("workflow_stages") or [])
        html_rows.append(
            "<tr>"
            f"<td>{_escape_html(stages or 'none')}</td>"
            f"<td>{int(row.get('count', 0))}</td>"
            f"<td>{len(list(row.get('workflow_stages') or []))}</td>"
            "</tr>"
        )
    return "".join(html_rows)


_EVIDENCE_NOISE_PATTERNS = (
    "affiliate",
    "commission",
    "shop",
    "search",
    "sign out",
    "log in",
    "create an account",
    "reset your password",
    "return policy",
    "shipping",
    "free shipping",
    "secure checkout",
    "table of contents",
    "skip to content",
    "cookie",
    "privacy policy",
    "disclosure",
    "buy now",
    "check price",
    "learn more",
    "limited-time deals",
    "add to cart",
    "recommended use",
    "ideal for",
    "pros cons",
    "skill level",
    "tools required",
    "approximate assembly time",
)

_WORKFLOW_STAGE_HINTS = {
    "research": ("compare", "compared", "review", "looked at", "showroom"),
    "order": ("order", "ordered", "purchased", "bought"),
    "deliver": ("arrived", "delivery", "shipped", "boxes"),
    "unbox": ("unbox", "unboxed", "opened", "pieces arrived"),
    "assemble": ("assemble", "assembly", "clamp", "setup", "set up"),
    "configure": ("configure", "configuration", "layout", "orientation", "rearranged"),
    "wear": ("wear", "wore", "upper", "slip on", "fit"),
    "run": ("run", "running", "miles", "trainer", "ride"),
    "sit": ("sit", "sitting", "seat depth", "loung", "movie"),
    "clean": ("clean", "spot-treat", "maintenance", "upkeep"),
    "wash": ("wash", "washed", "washable", "line dry", "covers"),
    "reconfigure": ("reconfigure", "modular", "swap", "change the layout"),
    "return": ("return", "returned", "send it back", "didn't fit", "did not fit"),
    "recommend": ("recommend", "worth it", "favorite", "would buy"),
}


def _split_sentences(text: str) -> list[str]:
    collapsed = " ".join(str(text).split())
    if not collapsed:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=[A-Z][a-z]+\s[A-Z])", collapsed)
    return [part.strip(" -") for part in parts if part.strip(" -")]


def _is_noisy_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(lowered) < 30:
        return True
    if any(pattern in lowered for pattern in _EVIDENCE_NOISE_PATTERNS):
        return True
    if lowered.count("menu") >= 2 or lowered.count("review") >= 4:
        return True
    return False


def _workflow_summary(workflow_row: dict[str, object] | None, event: dict[str, object]) -> str:
    if not workflow_row:
        return "workflow not extracted"
    stages = list(dict(workflow_row.get("selected_workflow") or {}).get("workflow_stages") or [])
    cleaned = [str(stage).strip() for stage in stages if str(stage).strip()]
    if not cleaned:
        return "workflow not extracted"
    summary = " -> ".join(cleaned)
    if event.get("return_risk_signal") and "return" not in cleaned:
        summary += " | return-risk evidence"
    return summary


def _sentence_score(
    sentence: str,
    *,
    workflow_tokens: set[str],
    priority_terms: set[str],
    return_risk_signal: bool,
    recommendation_signal: bool,
) -> tuple[int, int, int, int]:
    lowered = sentence.lower()
    stage_hits = 0
    for token in workflow_tokens:
        if token in lowered:
            stage_hits += 2
        for hint in _WORKFLOW_STAGE_HINTS.get(token, ()):
            if hint in lowered:
                stage_hits += 2
    priority_hits = sum(1 for term in priority_terms if term and term in lowered)
    risk_hits = 0
    if return_risk_signal:
        risk_hits += sum(3 for term in ("return", "returned", "tight", "narrow", "didn't fit", "did not fit", "assembly", "difficult") if term in lowered)
        if "recommend" in lowered and "return" not in lowered:
            risk_hits -= 2
    if recommendation_signal:
        risk_hits += sum(1 for term in ("comfortable", "worth", "washable", "durable", "recommend") if term in lowered)
    return (stage_hits, priority_hits, risk_hits, len(sentence))


def _supporting_excerpt(event: dict[str, object], workflow_row: dict[str, object] | None) -> str:
    text = str(event.get("text") or "")
    sentences = [sentence for sentence in _split_sentences(text) if not _is_noisy_sentence(sentence)]
    if not sentences:
        fallback = " ".join(text.split())
        return fallback[:280] + ("..." if len(fallback) > 280 else "")

    workflow_tokens = {str(stage).lower() for stage in list(dict(workflow_row.get("selected_workflow") or {}).get("workflow_stages") or [])} if workflow_row else set()
    aspect_tokens = {str(item).lower().replace("_", " ") for item in list(event.get("aspects") or [])}
    priority_terms = workflow_tokens | aspect_tokens
    if event.get("return_risk_signal"):
        priority_terms |= {"return", "returned", "tight", "narrow", "assembly", "difficult"}
    if event.get("recommendation_signal"):
        priority_terms |= {"recommend", "comfortable", "worth", "washable", "durable"}

    ranked = sorted(
        sentences,
        key=lambda sentence: _sentence_score(
            sentence,
            workflow_tokens=workflow_tokens,
            priority_terms=priority_terms,
            return_risk_signal=bool(event.get("return_risk_signal")),
            recommendation_signal=bool(event.get("recommendation_signal")),
        ),
        reverse=True,
    )
    excerpt = " ".join(ranked[:2])
    return excerpt[:280] + ("..." if len(excerpt) > 280 else "")


def _evidence_cards(events: list[dict[str, object]], usage_workflow_payload: dict[str, object]) -> str:
    if not events:
        return '<div class="empty-card">No normalized feedback events available.</div>'
    workflow_index = {
        str(row.get("feedback_id") or ""): row
        for row in list(usage_workflow_payload.get("workflows") or [])
    }
    ranked = sorted(
        events,
        key=lambda event: (
            -int(bool(event.get("return_risk_signal"))),
            -int(bool(event.get("recommendation_signal"))),
            -abs(float(event.get("sentiment_score", 0.0))),
        ),
    )
    cards = []
    for event in ranked[:6]:
        sentiment = str(event.get("sentiment", "mixed"))
        source_reference = str(event.get("source_reference") or "").strip()
        source_markup = ""
        if source_reference:
            if source_reference.startswith(("http://", "https://")):
                source_markup = (
                    '<div class="evidence-source">'
                    f'<a href="{_escape_html(source_reference)}" target="_blank" rel="noopener noreferrer">Open source</a>'
                    "</div>"
                )
            else:
                source_markup = f'<div class="evidence-source">Source: {_escape_html(source_reference)}</div>'
        tone = {
            "positive": "success",
            "negative": "danger",
            "mixed": "mixed",
        }.get(sentiment, "neutral")
        workflow_row = workflow_index.get(str(event.get("feedback_id") or ""))
        excerpt = _supporting_excerpt(event, workflow_row)
        workflow_summary = _workflow_summary(workflow_row, event)
        cards.append(
            '<article class="evidence-card">'
            f'<div class="evidence-topline">{_escape_html(event.get("title") or event.get("feedback_id") or "feedback")}</div>'
            f'<div class="evidence-meta">{_tone_chip(sentiment, tone)} {(" " + _tone_chip("return risk", "risk")) if event.get("return_risk_signal") else ""}</div>'
            f'<div class="key-label">Workflow</div>'
            f'<p>{_escape_html(workflow_summary)}</p>'
            f'<div class="key-label">Supporting Evidence</div>'
            f'<p>{_escape_html(excerpt)}</p>'
            f'<div class="evidence-tags">{", ".join(_escape_html(item) for item in list(event.get("aspects") or [])) or "no aspects"}</div>'
            f"{source_markup}"
            "</article>"
        )
    return "".join(cards)


def _product_visual_markup(product_visual_asset: dict[str, object]) -> str:
    image_url = str(product_visual_asset.get("image_url") or "").strip()
    if not image_url:
        return ""
    image_alt = str(product_visual_asset.get("image_alt") or "product visual").strip() or "product visual"
    source_reference = str(product_visual_asset.get("source_reference") or "").strip()
    source_note = (
        f'<div class="hero-visual-note">Visual source: {_escape_html(source_reference)}</div>'
        if source_reference
        else ""
    )
    return (
        '<aside class="hero-visual">'
        '<div class="hero-visual-frame">'
        f'<img src="{_escape_html(image_url)}" alt="{_escape_html(image_alt)}" loading="eager">'
        "</div>"
        f"{source_note}"
        "</aside>"
    )


def _dashboard_html(
    *,
    scorecard: dict[str, object],
    outcome: dict[str, object],
    aspect_payload: dict[str, object],
    hypothesis_payload: dict[str, object],
    usage_workflow_payload: dict[str, object],
    ablation_payload: dict[str, object],
    events: list[dict[str, object]],
    product_visual_asset: dict[str, object],
) -> str:
    product_name = str(scorecard.get("product_name") or "Product")
    brand_name = str(scorecard.get("brand_name") or "").strip()
    title = f"{brand_name} {product_name}".strip()
    verdict = str(scorecard.get("verdict") or "unknown")
    verdict_tone = _verdict_tone(verdict)
    run_status = str(scorecard.get("run_status") or "").strip()
    run_status_note = str(scorecard.get("run_status_note") or "").strip()
    aspect_summary = dict(aspect_payload.get("aspect_summary") or {})
    hypotheses = list(hypothesis_payload.get("hypotheses") or [])
    workflow_motifs = list(usage_workflow_payload.get("top_workflow_motifs") or [])
    ablation_rows = list(ablation_payload.get("rows") or [])
    ablation_takeaways = list(ablation_payload.get("takeaways") or [])
    top_negative = ", ".join(str(item) for item in scorecard.get("top_negative_aspects") or []) or "none detected"
    top_positive = ", ".join(str(item) for item in scorecard.get("top_positive_aspects") or []) or "none detected"
    top_return = ", ".join(str(item) for item in scorecard.get("top_return_risk_aspects") or []) or "none detected"
    product_visual_markup = _product_visual_markup(product_visual_asset)

    warning_banner = ""
    if scorecard.get("return_warning_recommended"):
        warning_banner = (
            '<section class="warning-banner">'
            f"<strong>Warning candidate:</strong> {_escape_html(scorecard.get('warning_text') or '')}"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLIFF Product Feedback Dashboard · {_escape_html(title)}</title>
  <style>
    :root {{
      --ink: #153243;
      --muted: #58727f;
      --paper: #f5efe6;
      --panel: #fffdf9;
      --line: #d8cfc0;
      --accent: #0f766e;
      --accent-soft: #d6f0ec;
      --risk: #b42318;
      --risk-soft: #fde7e4;
      --mixed: #b7791f;
      --mixed-soft: #fff0d6;
      --success: #0f766e;
      --success-soft: #d6f0ec;
      --shadow: 0 16px 40px rgba(21, 50, 67, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 26%),
        linear-gradient(180deg, #f7f1e8 0%, #efe6d8 100%);
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(250,244,235,0.94));
      border: 1px solid rgba(216, 207, 192, 0.9);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -60px -90px auto;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(15,118,110,0.12), transparent 70%);
      pointer-events: none;
    }}
    .hero-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.9fr);
      gap: 24px;
      align-items: center;
      position: relative;
      z-index: 1;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{
      font-size: clamp(32px, 4vw, 54px);
      line-height: 0.98;
      margin: 0 0 12px;
      max-width: 800px;
    }}
    .hero p {{
      margin: 0;
      max-width: 760px;
      line-height: 1.55;
      color: var(--muted);
      font-size: 17px;
    }}
    .hero-meta {{
      margin-top: 18px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .hero-visual {{
      justify-self: end;
      width: 100%;
      max-width: 360px;
    }}
    .hero-visual-frame {{
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,241,232,0.98));
      border: 1px solid rgba(216, 207, 192, 0.95);
      border-radius: 24px;
      padding: 14px;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 18px 44px rgba(21, 50, 67, 0.12);
    }}
    .hero-visual img {{
      display: block;
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      mix-blend-mode: multiply;
    }}
    .hero-visual-note {{
      margin-top: 10px;
      font-size: 12px;
      line-height: 1.4;
      color: var(--muted);
      word-break: break-word;
    }}
    .tone-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      border: 1px solid transparent;
      background: #eef2f5;
      color: var(--ink);
    }}
    .tone-chip.success {{ background: var(--success-soft); color: var(--success); border-color: rgba(15,118,110,0.18); }}
    .tone-chip.risk, .metric-value.risk {{ background: var(--risk-soft); color: var(--risk); border-color: rgba(180,35,24,0.15); }}
    .tone-chip.mixed {{ background: var(--mixed-soft); color: var(--mixed); border-color: rgba(183,121,31,0.15); }}
    .tone-chip.neutral {{ background: #eef2f5; color: var(--ink); }}
    .warning-banner {{
      margin-top: 18px;
      background: linear-gradient(90deg, rgba(180,35,24,0.10), rgba(180,35,24,0.04));
      border: 1px solid rgba(180,35,24,0.18);
      color: var(--risk);
      padding: 14px 16px;
      border-radius: 18px;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
      margin-top: 22px;
    }}
    .panel {{
      background: rgba(255, 253, 249, 0.96);
      border: 1px solid rgba(216, 207, 192, 0.95);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    .panel h3 {{
      margin: 0 0 12px;
      font-size: 16px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      min-height: 108px;
    }}
    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .metric-value {{
      margin-top: 10px;
      font-size: 34px;
      line-height: 1;
      font-weight: 700;
      color: var(--ink);
    }}
    .metric-value.success {{ color: var(--success); }}
    .metric-value.mixed {{ color: var(--mixed); }}
    .metric-value.danger {{ color: var(--risk); }}
    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-7 {{ grid-column: span 7; }}
    .span-6 {{ grid-column: span 6; }}
    .span-5 {{ grid-column: span 5; }}
    .span-4 {{ grid-column: span 4; }}
    .bar-stack {{ display: grid; gap: 12px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 150px 1fr 72px;
      gap: 10px;
      align-items: center;
    }}
    .bar-label, .bar-value {{ font-size: 14px; color: var(--muted); }}
    .bar-track {{
      width: 100%;
      background: #eee5d8;
      border-radius: 999px;
      height: 13px;
      overflow: hidden;
      border: 1px solid rgba(88,114,127,0.08);
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: #91a8b0;
    }}
    .bar-fill.success {{ background: linear-gradient(90deg, #0f766e, #38b2ac); }}
    .bar-fill.risk {{ background: linear-gradient(90deg, #b42318, #ef4444); }}
    .bar-fill.mixed {{ background: linear-gradient(90deg, #b7791f, #f59e0b); }}
    .key-list {{
      display: grid;
      gap: 12px;
    }}
    .key-item {{
      padding: 14px 16px;
      border-radius: 18px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .key-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(216, 207, 192, 0.7);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.08em;
    }}
    td.empty {{
      color: var(--muted);
      text-align: center;
      padding: 24px 8px;
    }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .evidence-card, .empty-card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .evidence-topline {{
      font-size: 16px;
      margin-bottom: 8px;
    }}
    .evidence-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .evidence-card p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 14px;
    }}
    .evidence-tags {{
      font-size: 13px;
      color: var(--ink);
    }}
    .evidence-source {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-word;
    }}
    .evidence-source a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .evidence-source a:hover {{
      text-decoration: underline;
    }}
    .footnote {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    @media (max-width: 980px) {{
      .hero-layout {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .span-8, .span-7, .span-6, .span-5, .span-4 {{ grid-column: span 12; }}
      .evidence-grid {{ grid-template-columns: 1fr; }}
      .hero-visual {{ justify-self: start; max-width: 320px; }}
    }}
    @media (max-width: 680px) {{
      .shell {{ padding: 18px 14px 40px; }}
      .hero, .panel {{ border-radius: 18px; padding: 18px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-layout">
        <div class="hero-copy">
          <div class="eyebrow">CLIFF Product Feedback Dashboard</div>
          <h1>{_escape_html(title)}</h1>
          <p>{_escape_html(scorecard.get("analysis_question") or "")}</p>
          <div class="hero-meta">
            {_tone_chip(verdict.replace("_", " "), verdict_tone)}
            {_tone_chip(run_status.replace("_", " "), "neutral") if run_status else ""}
            {_tone_chip(f"{int(outcome.get('feedback_count', 0))} feedback sources", "neutral")}
            {_tone_chip(f"{int(scorecard.get('hypothesis_count', 0))} hypotheses", "neutral")}
          </div>
          <div class="footnote">CLIFF is a research prototype. This analysis is exploratory and should not be treated as a product endorsement, product criticism, or purchasing advice. AI-based analyses can make mistakes.</div>
          {f'<div class="footnote">{_escape_html(run_status_note)}</div>' if run_status_note else ""}
          {warning_banner}
        </div>
        {product_visual_markup}
      </div>
    </section>

    <section class="grid">
      <div class="panel span-12">
        <h2>Outcome Snapshot</h2>
        <div class="metrics">
          {_metric_card("Overall Score", scorecard.get("overall_score"), tone=verdict_tone)}
          {_metric_card("Average Rating", _format_rating(outcome.get("average_rating")))}
          {_metric_card("Return Risk Rate", _format_percent(outcome.get("return_risk_rate")), tone="danger" if float(outcome.get("return_risk_rate", 0.0)) >= 0.25 else "mixed")}
          {_metric_card("Recommendation Rate", _format_percent(outcome.get("recommendation_rate")), tone="success")}
        </div>
      </div>

      <div class="panel span-12">
        <h2>Ablation Comparison</h2>
        <table>
          <thead>
            <tr>
              <th>Mode</th>
              <th>Verdict</th>
              <th>Score</th>
              <th>Risk-adjustment</th>
              <th>Warning</th>
              <th>Top driver</th>
              <th>Hypotheses</th>
              <th>Evidence coverage</th>
            </tr>
          </thead>
          <tbody>
            {_ablation_rows(ablation_rows)}
          </tbody>
        </table>
        <div class="footnote">
          {" ".join(_escape_html(item) for item in ablation_takeaways) if ablation_takeaways else "Use this table to compare the flattened baseline against the structured scaffold and any optional Democritus/KET rows. Positive risk-adjustment means the method became more conservative than the prompt-like baseline."}
        </div>
      </div>

      <div class="panel span-7">
        <h2>Signal Mix</h2>
        <div class="bar-stack">
          {_bar_row("Positive share", float(outcome.get("positive_share", 0.0)), tone="success")}
          {_bar_row("Negative share", float(outcome.get("negative_share", 0.0)), tone="risk")}
          {_bar_row("Return risk rate", float(outcome.get("return_risk_rate", 0.0)), tone="mixed")}
          {_bar_row("Recommendation rate", float(outcome.get("recommendation_rate", 0.0)), tone="success")}
        </div>
      </div>

      <div class="panel span-5">
        <h2>Driver Summary</h2>
        <div class="key-list">
          <div class="key-item">
            <div class="key-label">Top Positive Aspects</div>
            <div>{_escape_html(top_positive)}</div>
          </div>
          <div class="key-item">
            <div class="key-label">Top Negative Aspects</div>
            <div>{_escape_html(top_negative)}</div>
          </div>
          <div class="key-item">
            <div class="key-label">Top Return-Risk Aspects</div>
            <div>{_escape_html(top_return)}</div>
          </div>
        </div>
      </div>

      <div class="panel span-6">
        <h2>Aspect Table</h2>
        <table>
          <thead>
            <tr>
              <th>Aspect</th>
              <th>Mentions</th>
              <th>Positive</th>
              <th>Negative</th>
              <th>Return risk</th>
              <th>Recommend</th>
            </tr>
          </thead>
          <tbody>
            {_aspect_rows(aspect_summary)}
          </tbody>
        </table>
      </div>

      <div class="panel span-6">
        <h2>Causal Hypotheses</h2>
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Relation</th>
              <th>Destination</th>
              <th>Support</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {_hypothesis_rows(hypotheses)}
          </tbody>
        </table>
      </div>

      <div class="panel span-6">
        <h2>Usage Workflows</h2>
        <table>
          <thead>
            <tr>
              <th>Workflow motif</th>
              <th>Count</th>
              <th>Stages</th>
            </tr>
          </thead>
          <tbody>
            {_workflow_rows(workflow_motifs)}
          </tbody>
        </table>
        <div class="footnote">
          Product success often depends on what customers actually do with the product: assemble it, sit on it, run in it, wash it, reconfigure it, or return it.
        </div>
      </div>

      <div class="panel span-12">
        <h2>Evidence Preview</h2>
        <div class="evidence-grid">
          {_evidence_cards(events, usage_workflow_payload)}
        </div>
        <div class="footnote">
          This dashboard visualizes causal-style hypotheses inferred from product feedback. It is best used as a decision-support surface, not as identified causal truth.
        </div>
      </div>
    </section>
  </div>
</body>
</html>"""


def generate_product_feedback_dashboard(run_outdir: str | Path) -> ProductFeedbackVisualizationResult:
    """Generate a static HTML dashboard from a product-feedback run directory."""

    run_path = Path(run_outdir).resolve()
    scorecard = _load_json(run_path / "product_success_scorecard.json")
    outcome = _load_json(run_path / "outcome_summary.json")
    aspect_payload = _load_json(run_path / "aspect_summary.json")
    hypothesis_payload = _load_json(run_path / "causal_hypotheses.json")
    usage_workflow_payload = _load_json(run_path / "usage_workflows.json")
    ablation_payload = _load_json(run_path / "ablation_comparison.json")
    events = _load_jsonl(run_path / "normalized_feedback.jsonl")
    product_visual_asset = _load_product_visual_asset(run_path)

    dashboard_path = run_path / "product_feedback_dashboard.html"
    summary_path = run_path / "product_feedback_dashboard_summary.json"

    _write_text(
        dashboard_path,
        _dashboard_html(
            scorecard=scorecard,
            outcome=outcome,
            aspect_payload=aspect_payload,
            hypothesis_payload=hypothesis_payload,
            usage_workflow_payload=usage_workflow_payload,
            ablation_payload=ablation_payload,
            events=events,
            product_visual_asset=product_visual_asset,
        ),
    )
    _write_json(
        summary_path,
        {
            "dashboard_path": str(dashboard_path),
            "product_name": str(scorecard.get("product_name") or ""),
            "brand_name": str(scorecard.get("brand_name") or ""),
            "verdict": str(scorecard.get("verdict") or ""),
            "overall_score": scorecard.get("overall_score"),
            "feedback_count": outcome.get("feedback_count"),
            "hypothesis_count": len(list(hypothesis_payload.get("hypotheses") or [])),
            "workflow_motif_count": len(list(usage_workflow_payload.get("top_workflow_motifs") or [])),
            "ablation_row_count": len(list(ablation_payload.get("rows") or [])),
            "product_visual_available": bool(product_visual_asset.get("image_url")),
            "product_visual_asset_path": (
                str(run_path / "product_visual_asset.json") if (run_path / "product_visual_asset.json").exists() else None
            ),
        },
    )
    return ProductFeedbackVisualizationResult(
        dashboard_path=dashboard_path,
        summary_path=summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a BAFFLE product-feedback dashboard from a run directory.")
    parser.add_argument("run_outdir")
    args = parser.parse_args()
    result = generate_product_feedback_dashboard(args.run_outdir)
    print(f"[BAFFLE product_feedback_visualizations] dashboard: {result.dashboard_path}")


if __name__ == "__main__":
    main()
