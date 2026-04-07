"""Agentic product-feedback scaffold for brand and launch analysis."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .agentic_workflows import (
    AgentSpec,
    AgenticWorkflow,
    ArtifactSpec,
    AttentionSpec,
    DiffusionSpec,
    build_agentic_workflow,
)
from .product_feedback_visualizations import bootstrap_product_feedback_dashboard, generate_product_feedback_dashboard

_POSITIVE_TERMS = (
    "comfortable",
    "great",
    "excellent",
    "love",
    "stylish",
    "good value",
    "worth it",
    "fits well",
    "true to size",
    "durable",
    "supportive",
    "easy to slip on",
    "easy to wear",
    "recommend",
)

_NEGATIVE_TERMS = (
    "uncomfortable",
    "tight",
    "too tight",
    "too small",
    "too big",
    "too loose",
    "pinches",
    "blister",
    "hurts",
    "returned",
    "returning",
    "send it back",
    "slips off",
    "falls apart",
    "cheap",
    "expensive",
    "not worth",
    "bad fit",
    "poor quality",
)

_RETURN_SIGNAL_TERMS = (
    "returned",
    "returning",
    "often returned",
    "send it back",
    "sent it back",
    "did not fit",
    "didn't fit",
    "too tight",
    "too small",
    "too loose",
    "size runs small",
    "size runs large",
    "narrow",
    "heel slip",
    "slips off",
)

_PROMPT_BASELINE_DRIVER_FALLBACK = "sentiment summary"
_OPTIONAL_ABLATION_FILES = (
    ("democritus", "Democritus layer", "democritus_ablation.json"),
    ("ket", "KET layer", "ket_ablation.json"),
)

_ASPECT_LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "fit": {
        "positive": ("fits well", "true to size", "perfect fit", "good fit"),
        "negative": (
            "tight fit",
            "too tight",
            "too small",
            "too loose",
            "size runs small",
            "size runs large",
            "too narrow",
            "heel slip",
            "slips off",
            "didn't fit",
            "did not fit",
            "bad fit",
        ),
    },
    "comfort": {
        "positive": ("comfortable", "cushioned", "supportive", "soft", "feels great"),
        "negative": ("uncomfortable", "hurts", "pain", "blister", "stiff", "no support"),
    },
    "durability": {
        "positive": ("durable", "holds up", "well made", "solid quality"),
        "negative": ("falls apart", "wore out", "coming apart", "poor quality", "cheaply made"),
    },
    "style": {
        "positive": ("stylish", "looks great", "cute", "good looking", "nice design"),
        "negative": ("ugly", "looks cheap", "weird looking", "bulky"),
    },
    "traction": {
        "positive": ("good grip", "great traction", "non slip", "stable"),
        "negative": ("slippery", "no grip", "slides", "unstable"),
    },
    "value": {
        "positive": ("worth it", "good value", "great price", "worth the price"),
        "negative": ("too expensive", "overpriced", "not worth", "poor value"),
    },
    "seat_depth": {
        "positive": ("great seat depth", "deep seating", "comfortable depth"),
        "negative": ("too deep", "too shallow", "seat depth regret", "awkward depth"),
    },
    "cushion_stability": {
        "positive": ("cushions stay in place", "doesn't shift", "minimal shifting"),
        "negative": ("cushions shifting", "covers shifting", "slides around", "won't stay in place"),
    },
    "assembly": {
        "positive": ("easy assembly", "easy to assemble", "simple setup"),
        "negative": ("hard to assemble", "difficult assembly", "tedious assembly", "dreaded assembly"),
    },
    "ease_of_use": {
        "positive": ("easy to slip on", "easy to put on", "convenient", "easy to wear"),
        "negative": ("hard to put on", "hard to slip on", "takes effort"),
    },
}

_USAGE_ACTION_ORDER = (
    "research",
    "order",
    "deliver",
    "unbox",
    "assemble",
    "configure",
    "wear",
    "run",
    "drive",
    "charge",
    "sit",
    "clean",
    "wash",
    "reconfigure",
    "return",
    "recommend",
)

_USAGE_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "research": ("review", "research", "looked at", "compared", "showroom", "tested out"),
    "order": ("ordered", "placed our order", "buy", "purchased", "purchase"),
    "deliver": ("arrived", "delivery", "shipping", "shipped"),
    "unbox": ("unbox", "unboxed", "opened the boxes", "pieces arrived"),
    "assemble": ("assemble", "assembly", "put it together", "set up", "setup"),
    "configure": ("configuration", "arrangement", "layout", "orientation", "deep seats", "fit"),
    "wear": ("wear", "wore", "upper comfort", "true to size", "slip on", "midfoot"),
    "run": ("run", "miles", "training", "long run", "tempo", "daily trainer"),
    "drive": ("drive", "driving", "steering", "ride", "handling", "commute", "road trip", "autopilot"),
    "charge": ("charge", "charging", "supercharger", "charger", "battery", "range"),
    "sit": ("sit", "seating", "loung", "movie", "watch", "nap", "couch"),
    "clean": ("clean", "pet hair", "spot treat", "upkeep", "maintenance"),
    "wash": ("wash", "washed", "washable", "line dry", "covers"),
    "reconfigure": ("rearrange", "reconfigure", "modular", "swap covers", "change the layout", "guest bed"),
    "return": ("return", "returned", "send it back", "returns", "not for you"),
    "recommend": ("recommend", "worth it", "good option", "would buy", "favorite"),
}

_USAGE_MACRO_TEMPLATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "shoe": (
        ("wear", "run", "return"),
        ("wear", "run", "recommend"),
        ("wear", "run", "clean", "recommend"),
    ),
    "vehicle": (
        ("research", "order", "deliver", "configure", "drive", "recommend"),
        ("research", "order", "deliver", "configure", "drive", "charge", "recommend"),
        ("research", "drive", "charge", "recommend"),
    ),
    "sofa": (
        ("order", "deliver", "assemble", "sit", "recommend"),
        ("order", "assemble", "configure", "sit", "wash", "recommend"),
        ("configure", "sit", "reconfigure", "clean", "recommend"),
    ),
    "generic": (
        ("research", "order", "configure", "recommend"),
    ),
}

_USAGE_ALLOWED_ACTIONS: dict[str, tuple[str, ...]] = {
    "shoe": ("research", "order", "deliver", "unbox", "configure", "wear", "run", "clean", "return", "recommend"),
    "vehicle": ("research", "order", "deliver", "configure", "drive", "charge", "clean", "return", "recommend"),
    "sofa": ("research", "order", "deliver", "unbox", "assemble", "configure", "sit", "clean", "wash", "reconfigure", "return", "recommend"),
    "generic": _USAGE_ACTION_ORDER,
}

_SUPPORTED_QUESTIONS = (
    "Was the product broadly successful, mixed, or at risk based on available feedback?",
    "Which product aspects most strongly appear to drive return risk?",
    "Which positive attributes seem to support satisfaction and recommendation?",
    "What evidence speaks to long-run comfort, durability, or maintenance burden?",
    "Should the product page surface a warning such as often returned due to tight fit?",
    "Which issues should the product team investigate first?",
)


def _slugify(name: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "product_feedback"


def _read_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError(f"Expected a JSON list in {path}")
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(dict(json.loads(line)))
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported manifest format for {path}; expected .json, .jsonl, or .csv")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _product_visual_asset_from_records(
    records: list[dict[str, object]],
    *,
    product_name: str,
    brand_name: str,
) -> dict[str, object] | None:
    label = " ".join(part for part in (brand_name.strip(), product_name.strip()) if part).strip() or "product"
    for record in records:
        image_url = str(record.get("image_url") or record.get("image") or record.get("thumbnail_url") or "").strip()
        if not image_url:
            continue
        return {
            "image_url": image_url,
            "image_alt": str(record.get("image_alt") or f"{label} visual"),
            "source_reference": str(
                record.get("image_source_reference")
                or record.get("source_reference")
                or record.get("url")
                or record.get("review_url")
                or ""
            ).strip(),
            "source_type": "feedback_manifest",
        }
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "returned", "flagged"}


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _article_rating_candidates(text: str) -> list[tuple[float, float, float]]:
    lowered = text.lower()
    candidates: list[tuple[float, float, float]] = []
    pattern_specs: tuple[tuple[str, float, float | None], ...] = (
        (
            r"(?:our verdict|final verdict|overall score|expert score|user score|review score|score)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(100|10|5)",
            1.0,
            None,
        ),
        (
            r"(?:our verdict|final verdict|overall score|expert score|user score|review score|score)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(%)",
            1.0,
            None,
        ),
        (
            r"(?:overall|verdict|score)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(100|10|5)",
            0.9,
            None,
        ),
        (
            r"(?:overall|verdict|score)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(%)",
            0.9,
            None,
        ),
        (
            r"([0-9]+(?:\.[0-9]+)?)\s*/\s*(100|10|5)\s*(?:overall|verdict|score)",
            0.85,
            None,
        ),
        (
            r"([0-9]+(?:\.[0-9]+)?)\s*(%)\s*(?:overall|verdict|score)",
            0.85,
            None,
        ),
        (
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:expert score|user score|overall score|review score|score)",
            0.82,
            10.0,
        ),
        (
            r"([0-9]+(?:\.[0-9]+)?)\s*out of\s*(100|10|5)\s*stars?",
            0.8,
            None,
        ),
        (
            r"(?:overall score|review score|final verdict|our verdict)[^0-9]{0,80}([0-9]+(?:\.[0-9]+)?)\s*/\s*(100|10|5)",
            0.78,
            None,
        ),
    )
    for pattern, confidence, fallback_scale in pattern_specs:
        for match in re.finditer(pattern, lowered):
            raw = _coerce_float(match.group(1))
            if raw is None:
                continue
            if fallback_scale is not None:
                scale = fallback_scale
            else:
                scale_token = match.group(2)
                scale = 100.0 if scale_token == "%" else float(scale_token)
            if scale <= 0 or raw < 0 or raw > scale:
                continue
            candidates.append((raw, scale, confidence))

    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*(5|10|100)", lowered):
        start = max(0, match.start() - 40)
        end = min(len(lowered), match.end() + 40)
        window = lowered[start:end]
        if not any(token in window for token in ("score", "verdict", "overall", "review")):
            continue
        raw = _coerce_float(match.group(1))
        scale = _coerce_float(match.group(2))
        if raw is None or scale is None or scale <= 0 or raw < 0 or raw > scale:
            continue
        candidates.append((raw, scale, 0.8))
    return candidates


def _extract_article_rating_from_text(title: str, text: str) -> tuple[float | None, float | None]:
    haystack = _clean_text(f"{title} {text}")
    candidates = _article_rating_candidates(haystack)
    if not candidates:
        return None, None
    raw, scale, _confidence = max(candidates, key=lambda item: (item[2], item[1], item[0]))
    return raw, scale


def _extract_rating_fields(record: dict[str, object]) -> tuple[float | None, float | None]:
    raw_rating = None
    for key in (
        "rating",
        "stars",
        "star_rating",
        "user_rating",
        "expert_rating",
        "overall_rating",
        "review_rating",
        "verdict_rating",
    ):
        raw_rating = _coerce_float(record.get(key))
        if raw_rating is not None:
            break

    rating_scale = None
    for key in ("rating_scale", "max_rating", "rating_out_of", "rating_max", "stars_out_of"):
        rating_scale = _coerce_float(record.get(key))
        if rating_scale is not None:
            break

    if rating_scale is None:
        rating_unit = str(record.get("rating_unit") or record.get("rating_format") or "").strip().lower()
        if rating_unit in {"percent", "percentage", "%"}:
            rating_scale = 100.0

    if raw_rating is None:
        title = str(record.get("title") or "").strip()
        text = str(record.get("text") or record.get("review_text") or record.get("body") or record.get("content") or "").strip()
        raw_rating, rating_scale = _extract_article_rating_from_text(title, text)

    return raw_rating, rating_scale


def _normalize_rating(raw_rating: float | None, rating_scale: float | None) -> float | None:
    if raw_rating is None:
        return None
    if rating_scale is not None and rating_scale > 0:
        normalized = 5.0 * (raw_rating / rating_scale)
        return round(_clamp(normalized, 0.0, 5.0), 3)
    if 0.0 <= raw_rating <= 5.0:
        return round(raw_rating, 3)
    if 5.0 < raw_rating <= 10.0:
        return round(0.5 * raw_rating, 3)
    if 10.0 < raw_rating <= 100.0:
        return round(5.0 * (raw_rating / 100.0), 3)
    return round(_clamp(raw_rating, 0.0, 5.0), 3)


def _extract_text(record: dict[str, object]) -> str:
    parts = [
        str(record.get("title") or "").strip(),
        str(record.get("summary") or record.get("abstract") or "").strip(),
        str(record.get("text") or record.get("review_text") or record.get("body") or record.get("content") or "").strip(),
    ]
    return _clean_text(" ".join(part for part in parts if part))


def _aspect_matches(text: str) -> dict[str, str]:
    lowered = text.lower()
    matches: dict[str, str] = {}
    for aspect, lexicon in _ASPECT_LEXICONS.items():
        positive_hits = sum(1 for phrase in lexicon["positive"] if phrase in lowered)
        negative_hits = sum(1 for phrase in lexicon["negative"] if phrase in lowered)
        if positive_hits == 0 and negative_hits == 0:
            continue
        if negative_hits > positive_hits:
            matches[aspect] = "negative"
        elif positive_hits > negative_hits:
            matches[aspect] = "positive"
        else:
            matches[aspect] = "mixed"
    return matches


def _sentiment_score(text: str, rating: float | None) -> float:
    lowered = text.lower()
    score = 0.0
    if rating is not None:
        score += max(-1.0, min(1.0, (rating - 3.0) / 2.0))
    score += 0.25 * sum(1 for term in _POSITIVE_TERMS if term in lowered)
    score -= 0.25 * sum(1 for term in _NEGATIVE_TERMS if term in lowered)
    return max(-1.0, min(1.0, score))


def _sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    return "mixed"


def _has_return_signal(text: str, returned: bool, aspect_matches: dict[str, str]) -> bool:
    if returned:
        return True
    lowered = text.lower()
    if "return policy" in lowered or "refund policy" in lowered or "30-day return window" in lowered:
        return False
    if any(term in lowered for term in _RETURN_SIGNAL_TERMS):
        return True
    return aspect_matches.get("fit") == "negative" and any(
        term in lowered for term in ("tight", "too small", "too loose", "too narrow")
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _append_unique(actions: list[str], *new_actions: str) -> None:
    for action in new_actions:
        value = str(action).strip()
        if value and value not in actions:
            actions.append(value)


def _ordered_actions(actions: list[str]) -> list[str]:
    seen = set(actions)
    ordered = [action for action in _USAGE_ACTION_ORDER if action in seen]
    return ordered


def _ordered_unique_actions(actions: list[str]) -> list[str]:
    return _ordered_actions(list(dict.fromkeys(str(action).strip() for action in actions if str(action).strip())))


def _insert_action(actions: list[str], action: str, *, before: tuple[str, ...] = ("recommend", "return")) -> list[str]:
    if action in actions:
        return list(actions)
    updated = list(actions)
    for index, existing in enumerate(updated):
        if existing in before:
            updated.insert(index, action)
            return _ordered_unique_actions(updated)
    updated.append(action)
    return _ordered_unique_actions(updated)


def _linear_edges(actions: list[str]) -> list[tuple[str, str]]:
    return [(actions[index], actions[index + 1]) for index in range(len(actions) - 1)]


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _product_usage_family(product_name: str, brand_name: str, texts: list[str]) -> str:
    haystack = " ".join([product_name, brand_name, *texts]).lower()
    if any(
        token in haystack
        for token in (
            "tesla",
            "model 3",
            "miata",
            "vehicle",
            "car",
            "sedan",
            "drive",
            "driving",
            "steering",
            "charging",
            "charger",
            "battery",
            "autopilot",
        )
    ):
        return "vehicle"
    if any(token in haystack for token in ("pegasus", "shoe", "runner", "running", "sneaker")):
        return "shoe"
    if any(token in haystack for token in ("sactional", "sofa", "couch", "sectional", "seat")):
        return "sofa"
    return "generic"


def _top_aspects(aspect_summary: dict[str, dict[str, object]], *, key: str, limit: int = 3) -> list[str]:
    ranked = sorted(
        aspect_summary.items(),
        key=lambda item: (-int(item[1].get(key, 0)), item[0]),
    )
    return [name for name, stats in ranked if int(stats.get(key, 0)) > 0][:limit]


def _derive_verdict(overall_score: float, success_threshold: float) -> str:
    if overall_score >= max(success_threshold + 0.15, 0.75):
        return "successful"
    if overall_score >= success_threshold:
        return "mixed_positive"
    if overall_score >= 0.40:
        return "at_risk"
    return "unsuccessful"


def _normalize_ablation_row(row: dict[str, object], *, fallback_mode: str, fallback_label: str) -> dict[str, object]:
    return {
        "mode": str(row.get("mode") or fallback_mode),
        "label": str(row.get("label") or fallback_label),
        "verdict": str(row.get("verdict") or "unknown"),
        "overall_score": round(float(row.get("overall_score", 0.0)), 3),
        "risk_adjustment_vs_baseline": round(float(row.get("risk_adjustment_vs_baseline", 0.0)), 3),
        "return_warning_recommended": bool(row.get("return_warning_recommended")),
        "top_driver": str(row.get("top_driver") or "none detected"),
        "hypothesis_count": int(row.get("hypothesis_count", 0)),
        "avg_hypothesis_confidence": round(float(row.get("avg_hypothesis_confidence", 0.0)), 3),
        "evidence_coverage": round(float(row.get("evidence_coverage", 0.0)), 3),
        "notes": str(row.get("notes") or ""),
    }


@dataclass(frozen=True)
class ProductFeedbackAgenticConfig:
    """Configuration for product-feedback analysis."""

    manifest_path: Path
    outdir: Path
    product_name: str
    brand_name: str = ""
    analysis_question: str = ""
    success_threshold: float = 0.60
    return_warning_threshold: float = 0.18

    def resolved(self) -> "ProductFeedbackAgenticConfig":
        question = self.analysis_question.strip()
        if not question:
            question = (
                f"Was {self.product_name} successful, and which customer-feedback mechanisms "
                "appear to drive returns or satisfaction?"
            )
        return ProductFeedbackAgenticConfig(
            manifest_path=self.manifest_path.resolve(),
            outdir=self.outdir.resolve(),
            product_name=self.product_name.strip(),
            brand_name=self.brand_name.strip(),
            analysis_question=question,
            success_threshold=self.success_threshold,
            return_warning_threshold=self.return_warning_threshold,
        )


@dataclass(frozen=True)
class ProductFeedbackAgentRecord:
    """Execution record for one product-feedback agent."""

    agent_name: str
    frontier_index: int
    status: str
    started_at: float
    ended_at: float
    outputs: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ProductFeedbackRunResult:
    """Result bundle for the product-feedback scaffold."""

    records: tuple[ProductFeedbackAgentRecord, ...]
    normalized_feedback_path: Path
    usage_workflows_path: Path
    aspect_summary_path: Path
    outcome_summary_path: Path
    causal_hypotheses_path: Path
    success_scorecard_path: Path
    ablation_comparison_path: Path
    report_path: Path
    dashboard_path: Path
    dashboard_summary_path: Path


def build_product_feedback_agentic_workflow() -> AgenticWorkflow:
    """Build a Democritus-style workflow for product feedback and return-risk analysis."""

    return build_agentic_workflow(
        name="ProductFeedbackAgenticWorkflow",
        artifacts=(
            ArtifactSpec("feedback_corpus", "feedback_document_bundle", persistent=True),
            ArtifactSpec("feedback_manifest", "feedback_manifest", persistent=True),
            ArtifactSpec("normalized_feedback", "normalized_feedback_events", persistent=True),
            ArtifactSpec("usage_workflows", "product_usage_workflow_bundle", persistent=True),
            ArtifactSpec("aspect_state", "product_aspect_state", persistent=True),
            ArtifactSpec("outcome_state", "behavioral_outcome_state", persistent=True),
            ArtifactSpec("causal_hypotheses", "causal_hypothesis_set", persistent=True),
            ArtifactSpec("success_scorecard", "product_success_scorecard", persistent=True),
            ArtifactSpec("ablation_comparison", "product_feedback_ablation_comparison", persistent=True),
            ArtifactSpec("feedback_report", "product_feedback_report", persistent=True),
        ),
        agents=(
            AgentSpec(
                name="feedback_collection_agent",
                role="collect_product_feedback",
                produces=("feedback_corpus", "feedback_manifest"),
                capabilities=("manifest_ingestion", "feedback_provenance_tracking"),
            ),
            AgentSpec(
                name="feedback_normalization_agent",
                role="normalize_feedback_events",
                consumes=("feedback_corpus",),
                produces=("normalized_feedback",),
                attention_from=("feedback_collection_agent",),
                capabilities=("record_normalization", "event_cleanup", "channel_alignment"),
            ),
            AgentSpec(
                name="usage_workflow_agent",
                role="extract_product_usage_workflows",
                consumes=("normalized_feedback",),
                produces=("usage_workflows",),
                attention_from=("feedback_normalization_agent",),
                capabilities=("workflow_stage_extraction", "usage_workflow_scoring"),
            ),
            AgentSpec(
                name="aspect_grounding_agent",
                role="ground_feedback_to_product_aspects",
                consumes=("normalized_feedback",),
                produces=("aspect_state",),
                attention_from=("feedback_normalization_agent",),
                capabilities=("aspect_mapping", "issue_clustering"),
            ),
            AgentSpec(
                name="outcome_signal_agent",
                role="aggregate_behavioral_outcomes",
                consumes=("normalized_feedback",),
                produces=("outcome_state",),
                attention_from=("feedback_normalization_agent",),
                capabilities=("return_risk_aggregation", "satisfaction_estimation"),
            ),
            AgentSpec(
                name="causal_hypothesis_agent",
                role="infer_feedback_driven_causal_hypotheses",
                consumes=("aspect_state", "outcome_state", "usage_workflows"),
                produces=("causal_hypotheses",),
                attention_from=("aspect_grounding_agent", "outcome_signal_agent", "usage_workflow_agent"),
                capabilities=("hypothesis_gluing", "return_driver_analysis"),
            ),
            AgentSpec(
                name="success_scoring_agent",
                role="score_product_success_state",
                consumes=("causal_hypotheses", "outcome_state"),
                produces=("success_scorecard",),
                attention_from=("causal_hypothesis_agent", "outcome_signal_agent"),
                capabilities=("launch_scoring", "warning_generation"),
            ),
            AgentSpec(
                name="executive_summary_agent",
                role="summarize_product_feedback_state",
                consumes=("success_scorecard", "causal_hypotheses", "ablation_comparison"),
                produces=("feedback_report",),
                attention_from=("success_scoring_agent", "causal_hypothesis_agent", "ablation_comparison_agent"),
                capabilities=("report_generation", "decision_support"),
            ),
            AgentSpec(
                name="ablation_comparison_agent",
                role="compare_prompt_like_and_structured_modes",
                consumes=("success_scorecard", "causal_hypotheses", "aspect_state", "outcome_state"),
                produces=("ablation_comparison",),
                attention_from=("success_scoring_agent", "causal_hypothesis_agent", "aspect_grounding_agent", "outcome_signal_agent"),
                capabilities=("baseline_comparison", "quantitative_ablation_summary"),
            ),
        ),
        attentions=(
            AttentionSpec(
                name="normalization_attention",
                target_agent="feedback_normalization_agent",
                input_artifacts=("feedback_corpus", "feedback_manifest"),
                source_agents=("feedback_collection_agent",),
            ),
            AttentionSpec(
                name="causal_hypothesis_attention",
                target_agent="causal_hypothesis_agent",
                input_artifacts=("aspect_state", "outcome_state", "feedback_manifest"),
                source_agents=("aspect_grounding_agent", "outcome_signal_agent"),
            ),
            AttentionSpec(
                name="feedback_summary_attention",
                target_agent="executive_summary_agent",
                input_artifacts=("success_scorecard", "causal_hypotheses", "usage_workflows", "ablation_comparison", "feedback_manifest"),
                source_agents=("success_scoring_agent", "causal_hypothesis_agent", "ablation_comparison_agent"),
            ),
            AttentionSpec(
                name="ablation_comparison_attention",
                target_agent="ablation_comparison_agent",
                input_artifacts=("success_scorecard", "causal_hypotheses", "aspect_state", "outcome_state", "feedback_manifest"),
                source_agents=("success_scoring_agent", "causal_hypothesis_agent", "aspect_grounding_agent", "outcome_signal_agent"),
            ),
            AttentionSpec(
                name="usage_workflow_attention",
                target_agent="usage_workflow_agent",
                input_artifacts=("normalized_feedback", "feedback_manifest"),
                source_agents=("feedback_normalization_agent",),
            ),
        ),
        diffusions=(
            DiffusionSpec(
                name="workflow_causal_diffusion",
                target_agent="causal_hypothesis_agent",
                input_artifacts=("usage_workflows", "aspect_state", "outcome_state"),
            ),
            DiffusionSpec(
                name="feedback_causal_diffusion",
                target_agent="causal_hypothesis_agent",
                input_artifacts=("aspect_state", "outcome_state", "usage_workflows"),
            ),
            DiffusionSpec(
                name="ablation_summary_diffusion",
                target_agent="ablation_comparison_agent",
                input_artifacts=("success_scorecard", "causal_hypotheses", "aspect_state", "outcome_state"),
            ),
            DiffusionSpec(
                name="success_summary_diffusion",
                target_agent="executive_summary_agent",
                input_artifacts=("success_scorecard", "causal_hypotheses", "ablation_comparison"),
            ),
        ),
        metadata={"family": "Democritus", "semantic_role": "agentic_product_feedback_scaffold"},
    )


class ProductFeedbackAgenticRunner:
    """Run the local product-feedback scaffold end to end."""

    def __init__(self, config: ProductFeedbackAgenticConfig) -> None:
        self.config = config.resolved()
        self.workflow = build_product_feedback_agentic_workflow()
        self.outdir = self.config.outdir
        self.summary_path = self.outdir / "agent_run_summary.json"
        self._state: dict[str, object] = {}
        self._handlers = {
            "feedback_collection_agent": self._run_feedback_collection_agent,
            "feedback_normalization_agent": self._run_feedback_normalization_agent,
            "usage_workflow_agent": self._run_usage_workflow_agent,
            "aspect_grounding_agent": self._run_aspect_grounding_agent,
            "outcome_signal_agent": self._run_outcome_signal_agent,
            "causal_hypothesis_agent": self._run_causal_hypothesis_agent,
            "success_scoring_agent": self._run_success_scoring_agent,
            "ablation_comparison_agent": self._run_ablation_comparison_agent,
            "executive_summary_agent": self._run_executive_summary_agent,
        }

    def plan(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(agent.name for agent in frontier) for frontier in self.workflow.parallel_frontiers())

    def run_agent(self, agent_name: str) -> tuple[str, ...]:
        return self._handlers[agent_name]()

    def run(self) -> ProductFeedbackRunResult:
        self.outdir.mkdir(parents=True, exist_ok=True)
        bootstrap_product_feedback_dashboard(
            self.outdir,
            product_name=self.config.product_name,
            brand_name=self.config.brand_name,
            analysis_question=self.config.analysis_question,
            run_status="running",
            run_status_note="Collecting and structuring product feedback evidence.",
            feedback_count=len(self._raw_records()),
        )
        records: list[ProductFeedbackAgentRecord] = []
        for frontier_index, frontier in enumerate(self.workflow.parallel_frontiers()):
            for agent in frontier:
                started_at = time.time()
                outputs = self.run_agent(agent.name)
                ended_at = time.time()
                records.append(
                    ProductFeedbackAgentRecord(
                        agent_name=agent.name,
                        frontier_index=frontier_index,
                        status="ok",
                        started_at=started_at,
                        ended_at=ended_at,
                        outputs=outputs,
                    )
                )
        ordered = tuple(records)
        self.summary_path.write_text(json.dumps([asdict(record) for record in ordered], indent=2), encoding="utf-8")
        dashboard_result = generate_product_feedback_dashboard(self.outdir)
        return ProductFeedbackRunResult(
            records=ordered,
            normalized_feedback_path=self.outdir / "normalized_feedback.jsonl",
            usage_workflows_path=self.outdir / "usage_workflows.json",
            aspect_summary_path=self.outdir / "aspect_summary.json",
            outcome_summary_path=self.outdir / "outcome_summary.json",
            causal_hypotheses_path=self.outdir / "causal_hypotheses.json",
            success_scorecard_path=self.outdir / "product_success_scorecard.json",
            ablation_comparison_path=self.outdir / "ablation_comparison.json",
            report_path=self.outdir / "product_feedback_report.md",
            dashboard_path=dashboard_result.dashboard_path,
            dashboard_summary_path=dashboard_result.summary_path,
        )

    def _raw_records(self) -> list[dict[str, object]]:
        cached = self._state.get("raw_records")
        if isinstance(cached, list):
            return cached
        records = _read_records(self.config.manifest_path)
        self._state["raw_records"] = records
        return records

    def _normalized_events(self) -> list[dict[str, object]]:
        cached = self._state.get("normalized_events")
        if isinstance(cached, list):
            return cached
        path = self.outdir / "normalized_feedback.jsonl"
        rows = [
            dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._state["normalized_events"] = rows
        return rows

    def _json_state(self, key: str, filename: str) -> dict[str, object]:
        cached = self._state.get(key)
        if isinstance(cached, dict):
            return cached
        payload = dict(json.loads((self.outdir / filename).read_text(encoding="utf-8")))
        self._state[key] = payload
        return payload

    def _run_feedback_collection_agent(self) -> tuple[str, ...]:
        records = self._raw_records()
        corpus_path = self.outdir / "feedback_corpus.json"
        manifest_path = self.outdir / "feedback_manifest.json"
        product_visual_asset = _product_visual_asset_from_records(
            records,
            product_name=self.config.product_name,
            brand_name=self.config.brand_name,
        )
        _write_json(corpus_path, records)
        _write_json(
            manifest_path,
            {
                "product_name": self.config.product_name,
                "brand_name": self.config.brand_name,
                "analysis_question": self.config.analysis_question,
                "source_manifest_path": str(self.config.manifest_path),
                "feedback_count": len(records),
                "supported_questions": list(_SUPPORTED_QUESTIONS),
                "semantic_target": "product_success_and_return_risk",
                "product_visual_asset": product_visual_asset,
            },
        )
        if product_visual_asset:
            _write_json(self.outdir / "product_visual_asset.json", product_visual_asset)
        return (str(corpus_path), str(manifest_path))

    def _run_feedback_normalization_agent(self) -> tuple[str, ...]:
        rows = []
        for index, record in enumerate(self._raw_records(), start=1):
            text = _extract_text(record)
            if not text:
                continue
            raw_rating, rating_scale = _extract_rating_fields(record)
            rating = _normalize_rating(raw_rating, rating_scale)
            returned = _coerce_bool(record.get("returned") or record.get("is_returned") or record.get("return_flag"))
            aspects = _aspect_matches(text)
            sentiment_score = _sentiment_score(text, rating)
            sentiment = _sentiment_label(sentiment_score)
            return_risk_signal = _has_return_signal(text, returned, aspects)
            recommendation_signal = sentiment == "positive" and any(
                aspects.get(aspect) == "positive" for aspect in ("comfort", "style", "fit", "ease_of_use")
            )
            rows.append(
                {
                    "feedback_id": str(record.get("id") or record.get("review_id") or f"feedback_{index:04d}"),
                    "source": str(record.get("source") or record.get("channel") or "unknown"),
                    "title": str(record.get("title") or "").strip(),
                    "text": text,
                    "rating": rating,
                    "raw_rating": raw_rating,
                    "rating_scale": rating_scale,
                    "returned": returned,
                    "sentiment": sentiment,
                    "sentiment_score": round(sentiment_score, 3),
                    "aspects": list(aspects),
                    "aspect_polarities": aspects,
                    "return_risk_signal": return_risk_signal,
                    "recommendation_signal": recommendation_signal,
                }
            )
        normalized_path = self.outdir / "normalized_feedback.jsonl"
        _write_jsonl(normalized_path, rows)
        self._state["normalized_events"] = rows
        return (str(normalized_path),)

    def _usage_workflow_base_actions(self, text: str, *, family: str) -> list[str]:
        lowered = text.lower()
        actions: list[str] = []
        for action, keywords in _USAGE_ACTION_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                _append_unique(actions, action)
        if family == "shoe":
            if any(token in lowered for token in ("easy run", "tempo", "daily trainer", "miles", "training")):
                _append_unique(actions, "wear", "run")
        if family == "vehicle":
            if any(token in lowered for token in ("drive", "driving", "steering", "ride", "handling", "commute")):
                _append_unique(actions, "drive")
            if any(token in lowered for token in ("charge", "charging", "supercharger", "charger", "battery", "range")):
                _append_unique(actions, "charge")
        if family == "sofa":
            if any(token in lowered for token in ("movie", "watch", "nap", "family room", "sit")):
                _append_unique(actions, "sit")
            if any(token in lowered for token in ("covers", "washable", "line dry")):
                _append_unique(actions, "wash")
        if not actions:
            _append_unique(actions, "configure")
        if any(token in lowered for token in ("recommend", "worth it", "favorite", "good option")):
            _append_unique(actions, "recommend")
        if any(token in lowered for token in ("return", "returned", "send it back")):
            _append_unique(actions, "return")
        allowed = set(_USAGE_ALLOWED_ACTIONS.get(family, _USAGE_ACTION_ORDER))
        return _ordered_unique_actions([action for action in actions if action in allowed])

    def _usage_workflow_variants(self, base_actions: list[str], *, family: str, text: str) -> list[dict[str, object]]:
        variants = [{"label": "base", "source": "basket", "workflow_stages": list(base_actions)}]
        current = list(base_actions)
        lowered = text.lower()
        if family == "shoe":
            if "wear" not in current:
                variants.append({"label": "add_wear", "source": "local_insert", "workflow_stages": _insert_action(current, "wear", before=("run", "recommend", "return"))})
            if "run" not in current and any(token in lowered for token in ("run", "trainer", "miles", "tempo", "training")):
                variants.append({"label": "add_run", "source": "local_insert", "workflow_stages": _insert_action(current, "run", before=("recommend", "return"))})
            if "return" not in current and any(token in lowered for token in ("too tight", "not for you", "returned", "harsh", "firm")):
                variants.append({"label": "add_return", "source": "local_insert", "workflow_stages": _insert_action(current, "return")})
        if family == "vehicle":
            if "drive" not in current and any(token in lowered for token in ("drive", "driving", "steering", "ride", "handling", "commute")):
                variants.append({"label": "add_drive", "source": "local_insert", "workflow_stages": _insert_action(current, "drive", before=("charge", "recommend", "return"))})
            if "charge" not in current and any(token in lowered for token in ("charge", "charging", "supercharger", "charger", "battery", "range")):
                variants.append({"label": "add_charge", "source": "local_insert", "workflow_stages": _insert_action(current, "charge", before=("recommend", "return"))})
            if "return" not in current and any(token in lowered for token in ("return", "returned", "gave it back", "trade in")):
                variants.append({"label": "add_return", "source": "local_insert", "workflow_stages": _insert_action(current, "return")})
        if family == "sofa":
            if "assemble" not in current and any(token in lowered for token in ("assembly", "assemble", "put it together")):
                variants.append({"label": "add_assemble", "source": "local_insert", "workflow_stages": _insert_action(current, "assemble", before=("configure", "sit", "wash", "recommend"))})
            if "sit" not in current and any(token in lowered for token in ("sit", "movie", "nap", "watch", "loung")):
                variants.append({"label": "add_sit", "source": "local_insert", "workflow_stages": _insert_action(current, "sit", before=("clean", "wash", "recommend"))})
            if "wash" not in current and any(token in lowered for token in ("wash", "covers", "line dry")):
                variants.append({"label": "add_wash", "source": "local_insert", "workflow_stages": _insert_action(current, "wash", before=("recommend",))})
        for template in _USAGE_MACRO_TEMPLATES.get(family, _USAGE_MACRO_TEMPLATES["generic"]):
            merged = _ordered_unique_actions(list(current) + list(template))
            variants.append({"label": "usage_macro", "source": "macro_completion", "workflow_stages": merged})
        deduped = []
        seen = set()
        for variant in variants:
            actions = tuple(_ordered_unique_actions([str(action) for action in variant["workflow_stages"]]))
            key = (str(variant["label"]), actions)
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"label": str(variant["label"]), "source": str(variant["source"]), "workflow_stages": list(actions)})
        return deduped

    def _score_usage_workflow_variant(self, actions: list[str], *, family: str, text: str) -> dict[str, float]:
        lowered = text.lower()
        local_hits = sum(
            1 for action in actions if any(keyword in lowered for keyword in _USAGE_ACTION_KEYWORDS.get(action, ()))
        )
        local = _safe_ratio(local_hits, max(len(actions), 1))
        struct = _safe_ratio(
            sum(1 for left, right in zip(actions, actions[1:]) if _USAGE_ACTION_ORDER.index(left) < _USAGE_ACTION_ORDER.index(right)),
            max(len(actions) - 1, 1),
        )
        macro = max(
            (
                _safe_ratio(sum(1 for action in template if action in actions), max(len(template), 1))
                for template in _USAGE_MACRO_TEMPLATES.get(family, _USAGE_MACRO_TEMPLATES["generic"])
            ),
            default=0.0,
        )
        terminal = 1.0 if actions and actions[-1] in {"recommend", "return"} else 0.5
        simp = _clamp_score(1.0 - (abs(len(actions) - 4) / 4.0))
        text_score = _clamp_score((_safe_ratio(len(text.split()), 350.0)) + 0.35 * local)
        total = 0.32 * local + 0.20 * struct + 0.20 * macro + 0.14 * terminal + 0.08 * simp + 0.06 * text_score
        return {
            "local": round(local, 6),
            "struct": round(struct, 6),
            "macro": round(macro, 6),
            "terminal": round(terminal, 6),
            "simp": round(simp, 6),
            "text": round(text_score, 6),
            "total": round(total, 6),
        }

    def _run_usage_workflow_agent(self) -> tuple[str, ...]:
        events = self._normalized_events()
        family = _product_usage_family(
            self.config.product_name,
            self.config.brand_name,
            [str(event.get("text") or "")[:500] for event in events[:3]],
        )
        workflow_rows = []
        motif_counter: Counter[tuple[str, ...]] = Counter()
        stage_counter: Counter[str] = Counter()
        for event in events:
            text = str(event.get("text") or "")
            base_actions = self._usage_workflow_base_actions(text, family=family)
            variants = self._usage_workflow_variants(base_actions, family=family, text=text)
            ranked_variants = []
            for variant in variants:
                actions = list(variant["workflow_stages"])
                scores = self._score_usage_workflow_variant(actions, family=family, text=text)
                ranked_variants.append(
                    {
                        "label": variant["label"],
                        "source": variant["source"],
                        "workflow_stages": actions,
                        "scores": scores,
                        "edges": [list(edge) for edge in _linear_edges(actions)],
                    }
                )
            ranked_variants.sort(
                key=lambda item: (float(item["scores"]["total"]), len(item["workflow_stages"])),
                reverse=True,
            )
            selected = ranked_variants[0] if ranked_variants else {"workflow_stages": base_actions, "scores": {"total": 0.0}, "edges": []}
            motif_counter.update([tuple(selected["workflow_stages"])])
            for action in selected["workflow_stages"]:
                stage_counter[action] += 1
            workflow_rows.append(
                {
                    "feedback_id": str(event.get("feedback_id") or ""),
                    "title": str(event.get("title") or ""),
                    "usage_family": family,
                    "selected_workflow": selected,
                    "base_workflow_stages": list(base_actions),
                    "workflow_variants": ranked_variants[:5],
                    "return_risk_signal": bool(event.get("return_risk_signal")),
                    "aspect_polarities": dict(event.get("aspect_polarities") or {}),
                }
            )
        top_motifs = [
            {"workflow_stages": list(actions), "count": count}
            for actions, count in motif_counter.most_common(6)
        ]
        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "usage_family": family,
            "workflow_count": len(workflow_rows),
            "top_workflow_motifs": top_motifs,
            "stage_frequency": dict(sorted(stage_counter.items(), key=lambda item: (-item[1], item[0]))),
            "workflows": workflow_rows,
        }
        path = self.outdir / "usage_workflows.json"
        _write_json(path, payload)
        self._state["usage_workflows"] = payload
        return (str(path),)

    def _run_aspect_grounding_agent(self) -> tuple[str, ...]:
        summary: dict[str, dict[str, object]] = {}
        for event in self._normalized_events():
            for aspect, polarity in dict(event.get("aspect_polarities") or {}).items():
                stats = summary.setdefault(
                    aspect,
                    {
                        "mentions": 0,
                        "positive_mentions": 0,
                        "negative_mentions": 0,
                        "mixed_mentions": 0,
                        "return_risk_mentions": 0,
                        "recommendation_mentions": 0,
                    },
                )
                stats["mentions"] = int(stats["mentions"]) + 1
                stats[f"{polarity}_mentions"] = int(stats.get(f"{polarity}_mentions", 0)) + 1
                if bool(event.get("return_risk_signal")) and polarity == "negative":
                    stats["return_risk_mentions"] = int(stats["return_risk_mentions"]) + 1
                if bool(event.get("recommendation_signal")) and polarity == "positive":
                    stats["recommendation_mentions"] = int(stats["recommendation_mentions"]) + 1
        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "aspect_summary": summary,
            "top_negative_aspects": _top_aspects(summary, key="negative_mentions"),
            "top_return_risk_aspects": _top_aspects(summary, key="return_risk_mentions"),
            "top_positive_aspects": _top_aspects(summary, key="positive_mentions"),
        }
        path = self.outdir / "aspect_summary.json"
        _write_json(path, payload)
        self._state["aspect_summary"] = payload
        return (str(path),)

    def _run_outcome_signal_agent(self) -> tuple[str, ...]:
        events = self._normalized_events()
        ratings = [float(event["rating"]) for event in events if event.get("rating") is not None]
        feedback_count = len(events)
        positive_count = sum(1 for event in events if event.get("sentiment") == "positive")
        negative_count = sum(1 for event in events if event.get("sentiment") == "negative")
        mixed_count = feedback_count - positive_count - negative_count
        return_risk_count = sum(1 for event in events if bool(event.get("return_risk_signal")))
        returned_count = sum(1 for event in events if bool(event.get("returned")))
        recommendation_count = sum(1 for event in events if bool(event.get("recommendation_signal")))
        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "feedback_count": feedback_count,
            "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "mixed_count": mixed_count,
            "return_risk_count": return_risk_count,
            "returned_count": returned_count,
            "recommendation_count": recommendation_count,
            "positive_share": round(positive_count / feedback_count, 3) if feedback_count else 0.0,
            "negative_share": round(negative_count / feedback_count, 3) if feedback_count else 0.0,
            "return_risk_rate": round(return_risk_count / feedback_count, 3) if feedback_count else 0.0,
            "recommendation_rate": round(recommendation_count / feedback_count, 3) if feedback_count else 0.0,
        }
        path = self.outdir / "outcome_summary.json"
        _write_json(path, payload)
        self._state["outcome_summary"] = payload
        return (str(path),)

    def _run_causal_hypothesis_agent(self) -> tuple[str, ...]:
        aspect_summary = dict(self._json_state("aspect_summary", "aspect_summary.json").get("aspect_summary") or {})
        workflow_payload = self._json_state("usage_workflows", "usage_workflows.json")
        top_workflow_motifs = list(workflow_payload.get("top_workflow_motifs") or [])
        hypotheses = []
        for aspect, stats in sorted(aspect_summary.items()):
            negative_mentions = int(stats.get("negative_mentions", 0))
            positive_mentions = int(stats.get("positive_mentions", 0))
            return_risk_mentions = int(stats.get("return_risk_mentions", 0))
            if return_risk_mentions > 0:
                action = "surface a fit warning and review sizing specifications" if aspect == "fit" else "investigate the issue in customer feedback and return reasons"
                src = "tight or inconsistent fit perception" if aspect == "fit" else f"negative {aspect} perception"
                hypotheses.append(
                    {
                        "src": src,
                        "relation": "INCREASES",
                        "dst": "return risk",
                        "support_count": return_risk_mentions,
                        "confidence": round(_clamp(0.45 + 0.12 * return_risk_mentions), 3),
                        "recommended_action": action,
                    }
                )
            if negative_mentions >= 1:
                hypotheses.append(
                    {
                        "src": f"negative {aspect} perception",
                        "relation": "REDUCES",
                        "dst": "product satisfaction",
                        "support_count": negative_mentions,
                        "confidence": round(_clamp(0.35 + 0.08 * negative_mentions), 3),
                        "recommended_action": f"inspect {aspect} complaints in detail",
                    }
                )
            if positive_mentions >= 1:
                hypotheses.append(
                    {
                        "src": f"positive {aspect} perception",
                        "relation": "INCREASES",
                        "dst": "recommendation intent",
                        "support_count": positive_mentions,
                        "confidence": round(_clamp(0.35 + 0.08 * positive_mentions), 3),
                        "recommended_action": f"reinforce {aspect} in product messaging",
                    }
                )
        for motif in top_workflow_motifs[:4]:
            stages = [str(stage) for stage in motif.get("workflow_stages") or [] if str(stage).strip()]
            if not stages:
                continue
            support_count = int(motif.get("count", 0))
            if "assemble" in stages:
                hypotheses.append(
                    {
                        "src": "assembly workflow friction",
                        "relation": "REDUCES",
                        "dst": "product satisfaction",
                        "support_count": support_count,
                        "confidence": round(_clamp(0.32 + 0.07 * support_count), 3),
                        "recommended_action": "reduce setup burden or improve setup guidance",
                    }
                )
            if "run" in stages and "return" in stages:
                hypotheses.append(
                    {
                        "src": "run-time usage friction",
                        "relation": "INCREASES",
                        "dst": "return risk",
                        "support_count": support_count,
                        "confidence": round(_clamp(0.36 + 0.08 * support_count), 3),
                        "recommended_action": "investigate fit and ride complaints during actual runs",
                    }
                )
            if "sit" in stages and "recommend" in stages:
                hypotheses.append(
                    {
                        "src": "comfortable seated use",
                        "relation": "INCREASES",
                        "dst": "recommendation intent",
                        "support_count": support_count,
                        "confidence": round(_clamp(0.34 + 0.08 * support_count), 3),
                        "recommended_action": "reinforce comfort-in-use messaging for living-room scenarios",
                    }
                )
        deduped: dict[tuple[str, str, str, str], dict[str, object]] = {}
        for item in hypotheses:
            key = (
                str(item["src"]),
                str(item["relation"]),
                str(item["dst"]),
                str(item["recommended_action"]),
            )
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = dict(item)
                continue
            existing["support_count"] = int(existing.get("support_count", 0)) + int(item.get("support_count", 0))
            existing["confidence"] = round(
                _clamp(max(float(existing.get("confidence", 0.0)), float(item.get("confidence", 0.0)))),
                3,
            )
        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "hypotheses": sorted(
                list(deduped.values()),
                key=lambda item: (-int(item["support_count"]), -float(item["confidence"]), str(item["src"])),
            ),
        }
        path = self.outdir / "causal_hypotheses.json"
        _write_json(path, payload)
        self._state["causal_hypotheses"] = payload
        return (str(path),)

    def _run_success_scoring_agent(self) -> tuple[str, ...]:
        aspect_payload = self._json_state("aspect_summary", "aspect_summary.json")
        outcome = self._json_state("outcome_summary", "outcome_summary.json")
        hypotheses = self._json_state("causal_hypotheses", "causal_hypotheses.json")
        usage_workflows = self._json_state("usage_workflows", "usage_workflows.json")
        average_rating = outcome.get("average_rating")
        rating_component = 0.5 if average_rating is None else _clamp((float(average_rating) - 1.0) / 4.0)
        positive_share = float(outcome.get("positive_share", 0.0))
        negative_share = float(outcome.get("negative_share", 0.0))
        sentiment_component = _clamp((positive_share - negative_share + 1.0) / 2.0)
        return_component = _clamp(1.0 - float(outcome.get("return_risk_rate", 0.0)))
        overall_score = round(0.40 * rating_component + 0.25 * sentiment_component + 0.35 * return_component, 3)

        if overall_score >= max(self.config.success_threshold + 0.15, 0.75):
            verdict = "successful"
        elif overall_score >= self.config.success_threshold:
            verdict = "mixed_positive"
        elif overall_score >= 0.40:
            verdict = "at_risk"
        else:
            verdict = "unsuccessful"

        top_return_risk_aspects = list(aspect_payload.get("top_return_risk_aspects") or [])
        top_negative_aspects = list(aspect_payload.get("top_negative_aspects") or [])
        top_positive_aspects = list(aspect_payload.get("top_positive_aspects") or [])
        fit_return_pressure = 0
        aspect_summary = dict(aspect_payload.get("aspect_summary") or {})
        if "fit" in aspect_summary:
            fit_return_pressure = int(dict(aspect_summary["fit"]).get("return_risk_mentions", 0))
        return_warning_recommended = bool(
            fit_return_pressure >= 2 or (
                "fit" in top_return_risk_aspects and float(outcome.get("return_risk_rate", 0.0)) >= self.config.return_warning_threshold
            )
        )
        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "analysis_question": self.config.analysis_question,
            "overall_score": overall_score,
            "verdict": verdict,
            "run_status": "complete",
            "run_status_note": "",
            "return_warning_recommended": return_warning_recommended,
            "warning_text": "Often returned due to fit issues" if return_warning_recommended else "",
            "top_return_risk_aspects": top_return_risk_aspects,
            "top_negative_aspects": top_negative_aspects,
            "top_positive_aspects": top_positive_aspects,
            "supported_questions": list(_SUPPORTED_QUESTIONS),
            "limitations": [
                "This scaffold produces causal-style hypotheses from feedback patterns, not identified causal truth.",
                "Behavioral telemetry such as return reasons, conversion, and sell-through would strengthen the model.",
            ],
            "hypothesis_count": len(list(hypotheses.get("hypotheses") or [])),
            "workflow_motif_count": len(list(usage_workflows.get("top_workflow_motifs") or [])),
        }
        path = self.outdir / "product_success_scorecard.json"
        _write_json(path, payload)
        self._state["success_scorecard"] = payload
        return (str(path),)

    def _prompt_like_driver(self, events: list[dict[str, object]]) -> str:
        counts: Counter[str] = Counter()
        for event in events:
            lowered = str(event.get("text") or "").lower()
            event_weight = 2 if str(event.get("sentiment")) == "negative" else 1
            for aspect, lexicon in _ASPECT_LEXICONS.items():
                phrases = tuple(lexicon["positive"]) + tuple(lexicon["negative"])
                if any(phrase in lowered for phrase in phrases):
                    counts[aspect] += event_weight
        if not counts:
            return _PROMPT_BASELINE_DRIVER_FALLBACK
        return counts.most_common(1)[0][0]

    def _prompt_like_ablation_row(self, events: list[dict[str, object]]) -> dict[str, object]:
        feedback_count = len(events)
        ratings = [float(event["rating"]) for event in events if event.get("rating") is not None]
        positive_count = sum(1 for event in events if str(event.get("sentiment")) == "positive")
        negative_count = sum(1 for event in events if str(event.get("sentiment")) == "negative")
        explicit_fit_return_count = 0
        covered_events = 0
        for event in events:
            lowered = str(event.get("text") or "").lower()
            explicit_fit_return = any(term in lowered for term in _RETURN_SIGNAL_TERMS) and any(
                token in lowered for token in ("fit", "size", "tight", "small", "large", "loose", "narrow", "slip")
            )
            explicit_driver = any(
                phrase in lowered
                for lexicon in _ASPECT_LEXICONS.values()
                for phrase in tuple(lexicon["positive"]) + tuple(lexicon["negative"])
            )
            if explicit_fit_return:
                explicit_fit_return_count += 1
            if event.get("rating") is not None or abs(float(event.get("sentiment_score", 0.0))) >= 0.30 or explicit_driver:
                covered_events += 1

        rating_component = 0.5 if not ratings else _clamp(((sum(ratings) / len(ratings)) - 1.0) / 4.0)
        sentiment_component = _clamp((_safe_ratio(positive_count - negative_count, max(feedback_count, 1)) + 1.0) / 2.0)
        overall_score = round(0.55 * rating_component + 0.45 * sentiment_component, 3)
        fit_return_rate = _safe_ratio(explicit_fit_return_count, max(feedback_count, 1))
        return_warning = bool(
            explicit_fit_return_count >= 2 or fit_return_rate >= self.config.return_warning_threshold
        )
        return _normalize_ablation_row(
            {
                "mode": "prompt_like_baseline",
                "label": "Prompt-like baseline",
                "verdict": _derive_verdict(overall_score, self.config.success_threshold),
                "overall_score": overall_score,
                "return_warning_recommended": return_warning,
                "top_driver": self._prompt_like_driver(events),
                "hypothesis_count": 0,
                "avg_hypothesis_confidence": 0.0,
                "evidence_coverage": round(_safe_ratio(covered_events, max(feedback_count, 1)), 3),
                "notes": (
                    "Flattens the review corpus into rating, sentiment, and explicit fit-return cues "
                    "without structured hypothesis generation."
                ),
            },
            fallback_mode="prompt_like_baseline",
            fallback_label="Prompt-like baseline",
        )

    def _structured_ablation_row(
        self,
        *,
        scorecard: dict[str, object],
        outcome: dict[str, object],
        hypotheses: list[dict[str, object]],
        events: list[dict[str, object]],
    ) -> dict[str, object]:
        covered_events = sum(
            1
            for event in events
            if list(event.get("aspects") or [])
            or bool(event.get("return_risk_signal"))
            or bool(event.get("recommendation_signal"))
        )
        avg_confidence = _safe_ratio(
            sum(float(item.get("confidence", 0.0)) for item in hypotheses),
            max(len(hypotheses), 1),
        )
        top_driver = next(
            iter(
                list(scorecard.get("top_return_risk_aspects") or [])
                or list(scorecard.get("top_negative_aspects") or [])
                or list(scorecard.get("top_positive_aspects") or [])
                or ["none detected"]
            )
        )
        return _normalize_ablation_row(
            {
                "mode": "ff2_structured_scaffold",
                "label": "BAFFLE structured scaffold",
                "verdict": scorecard.get("verdict", "unknown"),
                "overall_score": scorecard.get("overall_score", 0.0),
                "return_warning_recommended": scorecard.get("return_warning_recommended", False),
                "top_driver": top_driver,
                "hypothesis_count": len(hypotheses),
                "avg_hypothesis_confidence": round(avg_confidence, 3),
                "evidence_coverage": round(_safe_ratio(covered_events, max(len(events), 1)), 3),
                "notes": (
                    "Uses normalized events, aspect grounding, outcome aggregation, and causal-style "
                    "hypothesis generation before scoring the product state."
                ),
            },
            fallback_mode="ff2_structured_scaffold",
            fallback_label="BAFFLE structured scaffold",
        )

    def _load_optional_ablation_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for mode, label, filename in _OPTIONAL_ABLATION_FILES:
            path = self.outdir / filename
            if not path.exists():
                continue
            payload = dict(json.loads(path.read_text(encoding="utf-8")))
            rows.append(_normalize_ablation_row(payload, fallback_mode=mode, fallback_label=label))
        return rows

    def _takeaways_for_ablation_rows(self, rows: list[dict[str, object]]) -> list[str]:
        if len(rows) < 2:
            return []
        baseline = rows[0]
        takeaways = []
        for row in rows[1:]:
            label = str(row["label"])
            score_delta = float(row["overall_score"]) - float(baseline["overall_score"])
            risk_adjustment = float(row.get("risk_adjustment_vs_baseline", 0.0))
            takeaways.append(f"{label} score delta vs baseline: {score_delta:+.3f}.")
            if abs(risk_adjustment) >= 0.001:
                if risk_adjustment > 0:
                    takeaways.append(
                        f"{label} is more risk-adjusted than the prompt-like baseline by {risk_adjustment:.3f} score points."
                    )
                else:
                    takeaways.append(
                        f"{label} is less risk-adjusted than the prompt-like baseline by {abs(risk_adjustment):.3f} score points."
                    )
            if bool(row["return_warning_recommended"]) != bool(baseline["return_warning_recommended"]):
                takeaways.append(
                    f"{label} changed the return-warning decision from "
                    f"{'on' if baseline['return_warning_recommended'] else 'off'} to "
                    f"{'on' if row['return_warning_recommended'] else 'off'}."
                )
            if str(row["top_driver"]) != str(baseline["top_driver"]):
                takeaways.append(
                    f"{label} shifted the top driver from `{baseline['top_driver']}` to `{row['top_driver']}`."
                )
            if int(row["hypothesis_count"]) > int(baseline["hypothesis_count"]):
                takeaways.append(
                    f"{label} added {int(row['hypothesis_count']) - int(baseline['hypothesis_count'])} "
                    "structured hypotheses over the baseline."
                )
        return takeaways[:6]

    def _run_ablation_comparison_agent(self) -> tuple[str, ...]:
        events = self._normalized_events()
        scorecard = self._json_state("success_scorecard", "product_success_scorecard.json")
        outcome = self._json_state("outcome_summary", "outcome_summary.json")
        hypotheses_payload = self._json_state("causal_hypotheses", "causal_hypotheses.json")
        hypotheses = list(hypotheses_payload.get("hypotheses") or [])

        rows = [
            self._prompt_like_ablation_row(events),
            self._structured_ablation_row(
                scorecard=scorecard,
                outcome=outcome,
                hypotheses=hypotheses,
                events=events,
            ),
            *self._load_optional_ablation_rows(),
        ]
        baseline_score = float(rows[0]["overall_score"]) if rows else 0.0
        for row in rows[1:]:
            row["score_delta_vs_baseline"] = round(float(row["overall_score"]) - baseline_score, 3)
            row["risk_adjustment_vs_baseline"] = round(baseline_score - float(row["overall_score"]), 3)
        if rows:
            rows[0]["score_delta_vs_baseline"] = 0.0
            rows[0]["risk_adjustment_vs_baseline"] = 0.0

        payload = {
            "product_name": self.config.product_name,
            "brand_name": self.config.brand_name,
            "analysis_question": self.config.analysis_question,
            "rows": rows,
            "takeaways": self._takeaways_for_ablation_rows(rows),
            "notes": [
                "The prompt-like baseline is a deterministic lexical summary proxy, not a live LLM prompt run.",
                "Positive risk-adjustment values mean the method is more conservative than the prompt-like baseline.",
                "Optional Democritus/KET rows appear automatically if matching ablation JSON files are written into the run directory.",
            ],
        }
        path = self.outdir / "ablation_comparison.json"
        _write_json(path, payload)
        self._state["ablation_comparison"] = payload
        return (str(path),)

    def _run_executive_summary_agent(self) -> tuple[str, ...]:
        scorecard = self._json_state("success_scorecard", "product_success_scorecard.json")
        outcome = self._json_state("outcome_summary", "outcome_summary.json")
        hypotheses = list(self._json_state("causal_hypotheses", "causal_hypotheses.json").get("hypotheses") or [])
        usage_workflows = self._json_state("usage_workflows", "usage_workflows.json")
        ablation_payload = self._json_state("ablation_comparison", "ablation_comparison.json")
        ablation_rows = list(ablation_payload.get("rows") or [])
        ablation_takeaways = list(ablation_payload.get("takeaways") or [])
        top_workflow_motifs = list(usage_workflows.get("top_workflow_motifs") or [])

        title = self.config.product_name
        if self.config.brand_name:
            title = f"{self.config.brand_name} {title}"

        lines = [
            f"# Product Feedback Report: {title}",
            "",
            "## Core Question",
            self.config.analysis_question,
            "",
            "## Current Verdict",
            f"- verdict: `{scorecard['verdict']}`",
            f"- overall score: `{scorecard['overall_score']}`",
            f"- average rating: `{outcome.get('average_rating')}`",
            f"- return risk rate: `{outcome.get('return_risk_rate')}`",
            f"- positive share: `{outcome.get('positive_share')}`",
            f"- negative share: `{outcome.get('negative_share')}`",
        ]
        if ablation_rows:
            lines.extend(["", "## Ablation Comparison"])
            for row in ablation_rows:
                lines.append(
                    f"- {row['label']}: verdict=`{row['verdict']}`, score=`{row['overall_score']}`, "
                    f"risk_adjustment=`{row.get('risk_adjustment_vs_baseline', 0.0):+.3f}`, "
                    f"warning=`{'yes' if row['return_warning_recommended'] else 'no'}`, "
                    f"top driver=`{row['top_driver']}`, hypotheses=`{row['hypothesis_count']}`, "
                    f"evidence coverage=`{row['evidence_coverage']}`"
                )
            if ablation_takeaways:
                lines.append("")
                lines.append("## Quantitative Takeaways")
                lines.extend(f"- {item}" for item in ablation_takeaways)
        lines.extend(["", "## What This Scaffold Can Answer"])
        lines.extend(f"- {question}" for question in _SUPPORTED_QUESTIONS)
        lines.extend(
            [
                "",
                "## Usage Workflows",
            ]
        )
        if top_workflow_motifs:
            for motif in top_workflow_motifs[:5]:
                lines.append(
                    f"- workflow motif: {' -> '.join(str(stage) for stage in motif.get('workflow_stages') or [])} "
                    f"(count={motif.get('count', 0)})"
                )
        else:
            lines.append("- No workflow motifs were extracted from the available feedback.")
        lines.extend(
            [
                "",
                "## Main Drivers",
                f"- top return-risk aspects: {', '.join(scorecard.get('top_return_risk_aspects') or ['none detected'])}",
                f"- top negative aspects: {', '.join(scorecard.get('top_negative_aspects') or ['none detected'])}",
                f"- top positive aspects: {', '.join(scorecard.get('top_positive_aspects') or ['none detected'])}",
            ]
        )
        if scorecard.get("return_warning_recommended"):
            lines.append(f"- customer-facing warning candidate: {scorecard['warning_text']}")
        lines.extend(["", "## Causal Hypotheses"])
        if hypotheses:
            for item in hypotheses[:5]:
                lines.append(
                    f"- `{item['src']}` {item['relation'].lower()} `{item['dst']}` "
                    f"(support={item['support_count']}, confidence={item['confidence']})"
                )
        else:
            lines.append("- No strong hypotheses were generated from the available feedback.")
        lines.extend(
            [
                "",
                "## Limits",
                "- This is a causal-hypothesis interface over customer feedback, not a substitute for experiments or telemetry.",
                "- The highest-value next step is to join these hypotheses with return reasons, conversion, and sell-through data.",
                "",
            ]
        )
        path = self.outdir / "product_feedback_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return (str(path),)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BAFFLE product-feedback agentic scaffold.")
    parser.add_argument("--manifest", required=True, help="Path to a .json, .jsonl, or .csv feedback manifest")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--brand-name", default="")
    parser.add_argument("--analysis-question", default="")
    parser.add_argument("--success-threshold", type=float, default=0.60)
    parser.add_argument("--return-warning-threshold", type=float, default=0.18)
    args = parser.parse_args()

    runner = ProductFeedbackAgenticRunner(
        ProductFeedbackAgenticConfig(
            manifest_path=Path(args.manifest),
            outdir=Path(args.outdir),
            product_name=args.product_name,
            brand_name=args.brand_name,
            analysis_question=args.analysis_question,
            success_threshold=args.success_threshold,
            return_warning_threshold=args.return_warning_threshold,
        )
    )
    result = runner.run()
    print(f"[BAFFLE product_feedback_agentic] scorecard: {result.success_scorecard_path}")
    print(f"[BAFFLE product_feedback_agentic] report: {result.report_path}")
    print(f"[BAFFLE product_feedback_agentic] dashboard: {result.dashboard_path}")


if __name__ == "__main__":
    main()
