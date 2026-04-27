"""Optional bridge from CLIFF product feedback runs into Prometheus v1."""

from __future__ import annotations

from pathlib import Path
import sys

from .repo_layout import resolve_prometheus_root


ASPECT_TO_PROMETHEUS = {
    "fit": ("performance", "performance#sizing/fit"),
    "comfort": ("performance", "performance#comfort"),
    "support": ("performance", "performance#support/stability"),
    "stability": ("performance", "performance#support/stability"),
    "durability": ("performance", "performance#durability"),
    "quality": ("quality", "quality"),
    "style": ("appearance", "appearance#form"),
    "appearance": ("appearance", "appearance#general"),
    "material": ("appearance", "appearance#material"),
    "price": ("cost/value", "cost/value"),
    "value": ("cost/value", "cost/value"),
    "ease_of_use": ("contextofuse", "contextofuse#use case"),
    "assembly": ("contextofuse", "contextofuse#use case"),
    "taste": ("food", "food"),
    "flavor": ("food", "food"),
    "service": ("service", "service"),
}


ACTIVITY_TERMS = (
    "assemble",
    "assembly",
    "setup",
    "set up",
    "move",
    "moving",
    "lift",
    "heavy",
    "sit",
    "sat",
    "sleep",
    "run",
    "running",
    "walk",
    "walking",
    "drive",
    "driving",
    "taste",
    "eat",
)

FAILURE_TERMS = (
    "flat",
    "uncomfortable",
    "issue",
    "problem",
    "drawback",
    "break",
    "broke",
    "wearing down",
    "stain",
    "heavy",
    "hard to",
    "difficult",
)

SATISFACTION_TERMS = (
    "favorite",
    "recommend",
    "worth",
    "verdict",
    "loved",
    "love",
    "returned",
    "return",
    "outgrow",
    "pitch it",
    "dealbreaker",
)


def prometheus_available() -> bool:
    return (resolve_prometheus_root() / "src" / "prometheus").exists()


def _ensure_prometheus_importable() -> None:
    source_root = resolve_prometheus_root() / "src"
    if not (source_root / "prometheus").exists():
        raise FileNotFoundError(f"Prometheus_v1 source tree not found at {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def _normalized_sentiment(value: object) -> str:
    sentiment = str(value or "neutral").strip().lower()
    if sentiment == "mixed":
        return "neutral"
    if sentiment not in {"positive", "negative", "neutral"}:
        return "neutral"
    return sentiment


def _text_polarity(text: str, *, default: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("not ", "issue", "problem", "drawback", "uncomfortable", "heavy", "difficult", "hard to", "returned", "return")):
        return "negative"
    if any(token in lowered for token in ("favorite", "recommend", "comfortable", "worth", "love", "loved", "great")):
        return "positive"
    return default


def _append_unique_spec(
    specs: list[tuple[str, str, str, str]],
    seen: set[tuple[str, str, str]],
    context: str,
    relation: str,
    sentiment: str,
    aspect: str,
) -> None:
    key = (context, relation, sentiment)
    if key in seen:
        return
    seen.add(key)
    specs.append((context, relation, sentiment, aspect))


def _event_specs(row: dict[str, object]) -> list[tuple[str, str, str, str]]:
    aspect_polarities = dict(row.get("aspect_polarities") or {})
    specs: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    text = str(row.get("text") or "")
    lowered = text.lower()
    overall_sentiment = _normalized_sentiment(row.get("sentiment"))
    for aspect, polarity in sorted(aspect_polarities.items()):
        normalized = str(aspect).strip().lower()
        sentiment = _normalized_sentiment(polarity)
        context, relation = ASPECT_TO_PROMETHEUS.get(normalized, ("target", "target"))
        _append_unique_spec(specs, seen, context, relation, sentiment, normalized)

    if any(token in lowered for token in ACTIVITY_TERMS):
        _append_unique_spec(
            specs,
            seen,
            "contextofuse",
            "contextofuse#use case",
            _text_polarity(text, default=overall_sentiment),
            "activity_use",
        )
    if any(token in lowered for token in FAILURE_TERMS):
        _append_unique_spec(
            specs,
            seen,
            "performance",
            "performance#durability",
            "negative",
            "failure_mode",
        )
    if any(token in lowered for token in SATISFACTION_TERMS) or overall_sentiment != "neutral":
        _append_unique_spec(
            specs,
            seen,
            "general",
            "general",
            overall_sentiment,
            "satisfaction",
        )

    if specs:
        return specs

    return [("general", "general", overall_sentiment, "general")]


def build_prometheus_world_model_from_feedback(
    normalized_events: list[dict[str, object]],
    *,
    product_name: str,
    brand_name: str = "",
    outdir: str | Path,
) -> dict[str, object]:
    """Build a Prometheus v1 Topos World Model from CLIFF normalized feedback."""

    _ensure_prometheus_importable()
    from prometheus.artifacts import CausalEpisode, CausalEvent
    from prometheus.pipeline import build_world_model_from_episodes, write_markdown_report, write_world_model

    episodes = []
    for index, row in enumerate(normalized_events, start=1):
        feedback_id = str(row.get("feedback_id") or f"feedback_{index:04d}")
        text = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip()
        events = []
        for event_index, (context, relation, sentiment, aspect) in enumerate(_event_specs(row)):
            events.append(
                CausalEvent(
                    event_id=f"{feedback_id}:{event_index}",
                    action="mention:direct",
                    observation=f"{context}|{relation}|{sentiment}",
                    subject=product_name,
                    relation=relation,
                    object=sentiment,
                    context=context,
                    evidence=text[:500],
                )
            )
        episodes.append(
            CausalEpisode(
                episode_id=feedback_id,
                events=tuple(events),
                source_uri=str(row.get("source_reference") or ""),
                metadata={
                    "split": "cliff_gui",
                    "line_number": index,
                    "review_text": text,
                    "title": title,
                    "source": str(row.get("source") or ""),
                    "aspects": [aspect for _, _, _, aspect in _event_specs(row)],
                },
            )
        )

    label = " ".join(part for part in (brand_name.strip(), product_name.strip()) if part).strip()
    if not label:
        label = "CLIFF Product Feedback"
    model = build_world_model_from_episodes(tuple(episodes), label=f"{label} Prometheus TWM")
    output = Path(outdir).expanduser().resolve()
    artifact_path = write_world_model(model, output)
    report_path = write_markdown_report(model, output)
    summary = {
        "status": "ok",
        "artifact": str(artifact_path),
        "report": str(report_path),
        "summary": model.summary,
    }
    (output / "prometheus_twm_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
