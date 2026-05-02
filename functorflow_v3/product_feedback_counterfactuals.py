"""Prometheus-style counterfactual repair probes for product feedback."""

from __future__ import annotations

from collections import Counter
import re


COUNTERFACTUAL_BUILDER_VERSION = 17


NEGATIVE_ASPECT_CUES: dict[str, tuple[str, ...]] = {
    "fit": (
        "too tight",
        "too small",
        "too big",
        "too loose",
        "too narrow",
        "heel slip",
        "heel slips",
        "slipping",
        "slips",
        "hot spot",
        "hotspot",
        "rubbing",
        "rubbed",
        "irritation",
        "blister",
        "true to size issue",
        "sizing issue",
        "didn't fit",
        "did not fit",
    ),
    "comfort": (
        "uncomfortable",
        "not comfortable",
        "hurt",
        "hurts",
        "pain",
        "painful",
        "sore",
        "too stiff",
        "stiff upper",
        "harsh",
        "firm ride",
        "too firm",
        "bottomed out",
        "not cushioned",
        "lack of cushion",
        "leg fatigue",
        "foot fatigue",
        "numb",
        "no support",
        "not supportive",
        "tight space",
        "not much headroom",
        "headroom",
        "legroom",
        "bump their knees",
    ),
    "traction": (
        "slippery",
        "no grip",
        "poor grip",
        "lost grip",
        "slides",
        "sliding",
        "unstable",
        "not stable",
        "wet pavement",
        "cornering",
    ),
    "durability": (
        "wore out",
        "wears out",
        "falling apart",
        "fell apart",
        "coming apart",
        "tread wear",
        "outsole wear",
        "upper tore",
        "ripped",
        "poor quality",
    ),
    "value": (
        "overpriced",
        "too expensive",
        "not worth",
        "waste of money",
        "poor value",
        "would not buy again",
    ),
    "taste": (
        "waxy",
        "chalky",
        "too bitter",
        "too sweet",
        "bland",
        "not tasty",
        "artificial taste",
        "flat flavor",
        "flat flavour",
        "nothing stands out",
        "underwhelming",
        "muted flavor",
        "muted flavour",
    ),
}

NEGATIVE_CONTEXT_TERMS = (
    "bad",
    "issue",
    "issues",
    "problem",
    "problems",
    "complaint",
    "complaints",
    "drawback",
    "drawbacks",
    "weakness",
    "weaknesses",
    "con",
    "cons",
    "but",
    "however",
    "unfortunately",
    "not",
    "no",
    "never",
    "too",
    "less",
    "lacks",
    "lack",
    "return",
    "returned",
    "send back",
    "sent back",
    "hurt",
    "hurts",
    "pain",
    "painful",
    "sore",
    "irritation",
    "slip",
    "slips",
    "slipping",
    "unstable",
    "slippery",
    "wore out",
    "falling apart",
    "not worth",
    "nothing stands out",
    "underwhelming",
)

POSITIVE_CONTEXT_TERMS = (
    "good",
    "great",
    "excellent",
    "smooth",
    "fun",
    "forgiving",
    "comfortable",
    "stable",
    "planted",
    "trustworthy",
    "locks in",
    "locked in",
    "secure",
    "breathable",
    "soft",
    "flexible",
    "low leg fatigue",
    "don't feel beaten up",
    "doesn't feel beaten up",
    "less tippy",
    "pros",
    "what we like",
    "performance fit",
    "wider toe box",
    "true to size",
    "solid lockdown",
    "hard for kids to get hurt",
    "would be hard for kids to get hurt",
    "provides enough stability",
    "reliable, strong feel",
    "reliable strong feel",
    "impressive results",
)

FILTER_CONTEXT_TERMS = (
    "select size",
    "filters",
    "show more",
    "terrain",
    "pace",
    "brand",
    "pronation",
    "heel to toe drop",
    "width / fit",
    "audience score",
    "release date",
    "condition",
    "midsole softness",
    "outsole durability",
    "toebox durability",
    "arch support neutral",
    "best running shoes",
    "best of",
    "continue reading below",
    "advertisement",
    "mailbag",
    "podcast",
    "first impressions",
    "we rank our top",
)

EXPLICIT_NEGATIVE_CUES = (
    "too tight",
    "too small",
    "too big",
    "too loose",
    "too narrow",
    "heel slip",
    "heel slips",
    "slipping",
    "uncomfortable",
    "not comfortable",
    "hurt",
    "hurts",
    "pain",
    "painful",
    "sore",
    "harsh",
    "firm ride",
    "too firm",
    "bottomed out",
    "not cushioned",
    "no support",
    "not supportive",
    "tight space",
    "not much headroom",
    "bump their knees",
    "slippery",
    "no grip",
    "poor grip",
    "lost grip",
    "unstable",
    "not stable",
    "wore out",
    "falling apart",
    "poor quality",
    "overpriced",
    "too expensive",
    "not worth",
    "waxy",
    "chalky",
    "too bitter",
    "bland",
    "not tasty",
    "nothing stands out",
    "underwhelming",
)

NEGATIVE_STANCE_TERMS = (
    "bad",
    "complaint",
    "complaints",
    "con",
    "cons",
    "drawback",
    "drawbacks",
    "issue",
    "issues",
    "problem",
    "problems",
    "unfortunately",
    "what we don't like",
    "what we do not like",
    "not for",
    "should look elsewhere",
    "i wish",
    "wish it",
    "i would like",
    "could improve",
    "needs more",
    "lacks",
    "lack of",
    "too ",
    "not ",
    "no ",
    "never",
    "return",
    "returned",
    "send it back",
    "sent it back",
    "hurt",
    "hurts",
    "pain",
    "painful",
    "sore",
    "slip",
    "slips",
    "slipping",
    "unstable",
    "slippery",
    "wore out",
    "not worth",
    "tight space",
    "not much headroom",
    "bump their knees",
)


REPAIR_RULES: dict[str, dict[str, str]] = {
    "fit": {
        "repair": "clearer sizing guidance, better fit calibration, and stronger retention where the product slips or feels loose",
        "rationale": "Fit failures are often local causes of returns and low comfort.",
    },
    "comfort": {
        "repair": "softer pressure points, improved support, and more forgiving materials in the contact surfaces",
        "rationale": "Comfort is the main satisfaction test for seating, shoes, and long-duration use.",
    },
    "durability": {
        "repair": "stronger materials, reinforced high-wear components, and clearer durability expectations",
        "rationale": "Durability failures convert initial satisfaction into value and return-risk complaints.",
    },
    "style": {
        "repair": "more appealing visual options and a design language closer to the customer's stated use context",
        "rationale": "Style repairs can recover positive sentiment when the functional experience is acceptable.",
    },
    "traction": {
        "repair": "more stable contact surfaces, better grip, and activity-specific slip resistance",
        "rationale": "Slip and instability complaints are actionable safety and use-case failures.",
    },
    "value": {
        "repair": "a better price-to-quality match, longer useful life, or clearer value proposition",
        "rationale": "Value complaints usually summarize a mismatch between cost and experienced benefit.",
    },
    "seat_depth": {
        "repair": "adjustable seat depth, clearer configuration guidance, and cushion geometry matched to more body types",
        "rationale": "Seat-depth mismatch is a local sofa comfort failure that can dominate the whole review.",
    },
    "cushion_stability": {
        "repair": "stronger cushion anchoring, less slippery covers, and modular connectors that reduce shifting during use",
        "rationale": "Cushion movement turns a comfortable design into repeated maintenance friction.",
    },
    "assembly": {
        "repair": "fewer assembly steps, clearer labels, better tolerances, and setup guidance that prevents misalignment",
        "rationale": "Assembly friction can sour the product before normal use begins.",
    },
    "ease_of_use": {
        "repair": "simpler everyday interactions, fewer effortful steps, and clearer affordances",
        "rationale": "Ease-of-use repairs target repeated friction in ordinary workflows.",
    },
    "taste": {
        "repair": "better ingredient balance, fresher preparation, and more consistent flavor quality",
        "rationale": "Taste is the direct satisfaction state for food products.",
    },
}

DOMAIN_ALLOWED_ASPECTS = {
    "food": {"taste", "value", "durability"},
    "shoe": {"fit", "comfort", "durability", "style", "traction", "value"},
    "sofa": {"comfort", "durability", "style", "value", "seat_depth", "cushion_stability", "assembly", "ease_of_use"},
    "vehicle": {"comfort", "ease_of_use", "durability", "value"},
    "generic": set(REPAIR_RULES),
}

FOOD_TERMS = (
    "chocolate",
    "bar",
    "bars",
    "cocoa",
    "cacao",
    "taste",
    "tasty",
    "flavor",
    "flavour",
    "aroma",
    "eat",
    "eating",
    "dessert",
    "amedei",
    "porcelana",
)

SHOE_TERMS = ("shoe", "shoes", "runner", "running", "sneaker", "sneakers", "saucony", "endorphin")
SOFA_TERMS = ("sofa", "couch", "sectional", "sactional", "seat", "lovesac")
VEHICLE_TERMS = (
    "tesla",
    "telsa",
    "model 3",
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "sedan",
    "drive",
    "driving",
    "driver",
    "steering",
    "autopilot",
    "charging",
    "charger",
    "battery",
)

ASPECT_TWM_CONTEXTS: dict[str, tuple[str, ...]] = {
    "fit": ("chart:fit", "sizing/fit", "fit"),
    "comfort": ("chart:comfort", "comfort"),
    "durability": ("chart:failure_mode", "durability", "quality"),
    "style": ("chart:appearance", "appearance", "style"),
    "traction": ("chart:activity_use", "support/stability", "traction", "stability"),
    "value": ("chart:return_satisfaction", "cost/value", "value", "satisfaction"),
    "seat_depth": ("chart:comfort", "seat_depth", "seat depth", "comfort"),
    "cushion_stability": ("chart:comfort", "chart:failure_mode", "cushion", "stability"),
    "assembly": ("chart:activity_use", "contextofuse", "assembly", "setup"),
    "ease_of_use": ("chart:activity_use", "contextofuse", "ease_of_use", "use case"),
    "taste": ("food", "taste"),
}

SATISFACTION_TWM_CONTEXTS = ("chart:return_satisfaction", "return_satisfaction", "satisfaction", "cost/value")

ASPECT_TOPOS_CONTEXTS: dict[str, tuple[str, ...]] = {
    "assembly": ("assembly", "assemble", "setup"),
    "ease_of_use": ("ease_of_use", "use", "assemble", "setup"),
    "comfort": ("comfort", "sit"),
    "cushion_stability": ("cushion_stability", "sit", "durability"),
    "seat_depth": ("seat_depth", "sit", "comfort"),
    "durability": ("durability",),
    "fit": ("fit",),
    "style": ("style", "appearance"),
    "taste": ("taste",),
    "traction": ("traction", "use"),
    "value": ("value", "decision", "post_purchase"),
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_question_label(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.endswith("?") or lowered.startswith(("how ", "what ", "why ", "which ", "should ", "is "))


def _clean_product_label(product_name: str, brand_name: str) -> str:
    product = product_name.strip()
    brand = brand_name.strip()
    if _is_question_label(product):
        return brand or "the product"
    label = f"{brand} {product}".strip() or product
    return label if not _is_question_label(label) else "the product"


def _product_domain(product_name: str, brand_name: str) -> str:
    lowered = f"{brand_name} {product_name}".lower()
    if any(term in lowered for term in FOOD_TERMS):
        return "food"
    if any(term in lowered for term in SHOE_TERMS):
        return "shoe"
    if any(term in lowered for term in SOFA_TERMS):
        return "sofa"
    if any(term in lowered for term in VEHICLE_TERMS):
        return "vehicle"
    return "generic"


def _vehicle_comfort_repair_rule(
    *,
    aspect: str,
    evidence: str,
    event: dict[str, object],
    product_domain: str,
    base_rule: dict[str, str],
) -> dict[str, str]:
    if product_domain != "vehicle" or aspect != "comfort":
        return base_rule

    text = " ".join(
        str(part or "")
        for part in (
            event.get("title"),
            evidence,
            event.get("text"),
        )
    ).lower()
    if any(
        term in text
        for term in (
            "steering wheel",
            "apply pressure",
            "every 30 seconds",
            "interior camera",
            "driver attention",
            "detecting inattention",
            "autopilot",
            "prove you were still in control",
        )
    ):
        return {
            "repair": (
                "a less intrusive driver-attention check that uses camera monitoring cleanly, "
                "reduces steering-wheel pressure prompts, and keeps assisted-driving supervision predictable on long journeys"
            ),
            "rationale": (
                "The local comfort complaint is stress from the assisted-driving attention-confirmation loop, "
                "not seat padding or contact materials."
            ),
        }
    if any(
        term in text
        for term in (
            "rear seat",
            "back seat",
            "taller passengers",
            "bump their knees",
            "not much headroom",
            "headroom",
            "legroom",
            "tight space",
            "knee room",
        )
    ):
        return {
            "repair": (
                "rear-seat packaging with more knee room and headroom, a less knees-up seating posture, "
                "and clearer expectations for adult passenger comfort"
            ),
            "rationale": (
                "The local comfort complaint is passenger-space geometry in the rear cabin rather than cushion softness."
            ),
        }
    if any(term in text for term in ("ride", "suspension", "road noise", "wind noise", "cabin noise", "rattles")):
        return {
            "repair": (
                "more compliant suspension tuning, better cabin noise isolation, and vibration damping tuned for rough roads"
            ),
            "rationale": "The local comfort complaint is ride harshness or cabin disturbance during driving.",
        }
    return base_rule


def _event_negative_aspects(event: dict[str, object]) -> list[str]:
    text = str(event.get("text") or "")
    aspect_polarities = dict(event.get("aspect_polarities") or {})
    aspects = [
        str(aspect)
        for aspect, polarity in aspect_polarities.items()
        if str(polarity) == "negative" and _aspect_has_local_negative_support(text, str(aspect), event)
    ]
    for inferred in _infer_negative_aspects_from_text(text):
        if inferred not in aspects:
            aspects.append(inferred)
    if aspects:
        return aspects
    return []


def _event_exploratory_aspects(event: dict[str, object], allowed_aspects: set[str], product_domain: str) -> list[str]:
    text = str(event.get("text") or "")
    combined_text = f"{event.get('title') or ''} {text}"
    lowered = combined_text.lower()
    if _looks_like_filter_context(lowered) or _looks_like_navigation_context(lowered):
        return []

    aspects: list[str] = []
    aspect_polarities = dict(event.get("aspect_polarities") or {})
    for aspect, polarity in aspect_polarities.items():
        aspect_name = str(aspect)
        support_text = text if aspect_name in {"style", "value"} else combined_text
        if (
            str(polarity) == "negative"
            and aspect_name in allowed_aspects
            and _has_exploratory_local_support(support_text, aspect_name)
        ):
            aspects.append(aspect_name)

    if product_domain == "food" and "taste" in allowed_aspects:
        taste_opportunity_terms = (
            "nothing stands out",
            "nothing that stands out",
            "underwhelming",
            "bland",
            "flat flavor",
            "flat flavour",
            "muted flavor",
            "muted flavour",
            "too bitter",
            "too sweet",
            "waxy",
            "chalky",
        )
        if any(term in lowered for term in taste_opportunity_terms):
            aspects.append("taste")

    if "value" in allowed_aspects and _has_exploratory_local_support(text, "value"):
        aspects.append("value")

    if product_domain == "sofa" and "comfort" in allowed_aspects:
        sofa_opportunity_terms = (
            "not comfortable",
            "uncomfortable",
            "too firm",
            "too soft",
            "stiff",
            "sagging",
            "sags",
            "seat depth",
            "back support",
        )
        if any(term in lowered for term in sofa_opportunity_terms) and _has_negative_stance(lowered):
            aspects.append("comfort")

    return list(dict.fromkeys(aspects))


def _query_conditioned_aspects(product_name: str, product_domain: str, allowed_aspects: set[str]) -> list[str]:
    lowered = product_name.lower()
    if not _is_question_label(product_name):
        return []

    aspects: list[str] = []
    if "comfort" in allowed_aspects and any(
        term in lowered for term in ("comfortable", "comfort", "comfy", "sit", "sitting", "run with", "running")
    ):
        aspects.append("comfort")
    if "taste" in allowed_aspects and any(term in lowered for term in ("tasty", "taste", "eat", "eating", "flavor", "flavour")):
        aspects.append("taste")
    if "fit" in allowed_aspects and any(term in lowered for term in ("fit", "size", "sizing", "wear", "run with", "running")):
        aspects.append("fit")
    if "traction" in allowed_aspects and any(term in lowered for term in ("slip", "traction", "grip")):
        aspects.append("traction")
    if product_domain == "sofa" and "comfort" in allowed_aspects and any(term in lowered for term in ("sofa", "couch", "sectional")):
        aspects.append("comfort")
    return list(dict.fromkeys(aspects))


def _has_exploratory_local_support(text: str, aspect: str) -> bool:
    lowered = text.lower()
    if _looks_like_filter_context(lowered) or _looks_like_navigation_context(lowered):
        return False
    if _looks_like_exploratory_noise(lowered, aspect):
        return False
    if _looks_like_positive_section(lowered) or _looks_like_comparative_aside(lowered):
        return False

    weak_terms: dict[str, tuple[str, ...]] = {
        "fit": ("fit", "size", "sizing", "toe box", "toebox", "heel", "narrow", "wide", "loose", "tight"),
        "comfort": (
            "comfort",
            "comfortable",
            "uncomfortable",
            "cushion",
            "cushions",
            "seat",
            "support",
            "stiff",
            "firm",
            "soft",
            "sagging",
            "recline",
        ),
        "traction": ("traction", "grip", "slip", "slippery", "stable", "stability", "unstable"),
        "durability": ("durability", "durable", "wear", "wore", "tore", "quality", "last", "lasting"),
        "value": (
            "too expensive",
            "overpriced",
            "not worth",
            "poor value",
            "waste of money",
            "price felt hefty",
            "pricey",
            "costs too much",
        ),
        "style": (
            "not my style",
            "ugly",
            "dated",
            "unappealing",
            "limited color",
            "limited fabric",
            "wish there were more colors",
            "wish there were more fabrics",
        ),
        "seat_depth": ("seat depth", "too deep", "too shallow"),
        "cushion_stability": ("cushion", "cushions", "shifting", "slide", "sliding", "move around"),
        "assembly": ("assembly", "assemble", "setup", "set up", "hard to put together", "difficult to assemble"),
        "ease_of_use": ("hard to use", "difficult to use", "clunky", "frustrating", "effort"),
        "taste": ("taste", "flavor", "flavour", "aroma", "bland", "bitter", "waxy", "chalky"),
    }
    terms = weak_terms.get(aspect, ())
    if not any(term in lowered for term in terms):
        return False
    if aspect == "value":
        return any(
            term in lowered
            for term in (
                "too expensive",
                "overpriced",
                "not worth",
                "poor value",
                "waste of money",
                "price felt hefty",
                "pricey",
                "costs too much",
            )
        )
    if aspect == "style":
        return any(term in lowered for term in NEGATIVE_CONTEXT_TERMS)
    return _has_negative_stance(lowered)


def _looks_like_exploratory_noise(lowered_text: str, aspect: str) -> bool:
    if aspect == "style" and any(
        term in lowered_text
        for term in (
            "you like this style",
            "if you like this style",
            "old couch issues",
            "crate and barrel",
            "doesn't look too shabby",
            "doesn’t look too shabby",
        )
    ):
        return True
    if aspect == "value":
        if re.search(r"\bis\s+it\s+worth\s+the\s+hefty\s+price\s*tag\b", lowered_text):
            return True
        if re.search(r"\bvalue\s*:\s*[abcdf][+-]?\b", lowered_text):
            return True
        header_terms = (
            "skip to main content",
            "skip to header navigation",
            "share on facebook",
            "google preferred",
            "view all",
            "save to wishlist",
        )
        if sum(1 for term in header_terms if term in lowered_text) >= 2:
            return True
    return False


def _aspect_has_local_negative_support(text: str, aspect: str, event: dict[str, object]) -> bool:
    lowered = text.lower()
    if _looks_like_filter_context(lowered):
        return False
    if aspect == "comfort" and _has_vehicle_comfort_complaint(lowered):
        return True
    if aspect == "taste" and _has_taste_disappointment_support(lowered):
        return True
    if _looks_like_mixed_positive_limitation(lowered):
        return True
    if any(_has_negative_cue_context(text, cue) for cue in NEGATIVE_ASPECT_CUES.get(aspect, ())):
        return True
    return False


def _infer_negative_aspects_from_text(text: str) -> list[str]:
    aspects: list[str] = []
    for aspect, cues in NEGATIVE_ASPECT_CUES.items():
        if any(_has_negative_cue_context(text, cue) for cue in cues):
            aspects.append(aspect)
    lowered = text.lower()
    if "return" in lowered or "send it back" in lowered or "sent it back" in lowered:
        if "fit" not in aspects and any(token in lowered for token in ("size", "tight", "loose", "narrow", "slip", "heel")):
            aspects.append("fit")
        if "comfort" not in aspects and any(token in lowered for token in ("comfort", "hurt", "pain", "sore", "stiff")):
            aspects.append("comfort")
    if "taste" not in aspects and _has_taste_disappointment_support(lowered):
        aspects.append("taste")
    return aspects


def _has_taste_disappointment_support(lowered_text: str) -> bool:
    if not any(term in lowered_text for term in ("taste", "tasty", "flavor", "flavour", "aroma", "chocolate")):
        return False
    disappointment_terms = (
        "big disappointment",
        "disappointment",
        "disappointing",
        "disappointed",
        "letdown",
        "let down",
    )
    contrast_terms = ("unfortunately", "but", "however", "hoped", "expected", "waited for")
    if any(term in lowered_text for term in disappointment_terms) and any(term in lowered_text for term in contrast_terms):
        return True
    if any(term in lowered_text for term in ("shocked", "shock")) and any(
        term in lowered_text for term in ("disappointment", "disappointed", "unfortunately")
    ):
        return True
    return False


def _has_negative_cue_context(text: str, cue: str, *, window: int = 90) -> bool:
    lowered = text.lower()
    if _looks_like_filter_context(lowered):
        return False
    cue_lower = cue.lower()
    for index in _cue_indices(lowered, cue_lower):
        snippet_start = max(0, index - window)
        snippet_end = min(len(lowered), index + len(cue_lower) + window)
        snippet = lowered[snippet_start:snippet_end]
        if _looks_like_positive_section(snippet):
            start = index + len(cue_lower)
            continue
        if cue_lower in {"hurt", "hurts", "pain", "painful"} and _looks_like_safety_context(snippet):
            continue
        if cue_lower in EXPLICIT_NEGATIVE_CUES:
            if _has_negative_stance(snippet) and not _has_positive_context(snippet):
                return True
        elif _has_negative_stance(snippet) and any(term in snippet for term in NEGATIVE_CONTEXT_TERMS) and not _has_positive_context(snippet):
            return True
    return False


def _cue_indices(lowered_text: str, cue_lower: str) -> list[int]:
    pattern = r"(?<![a-z0-9])" + re.escape(cue_lower) + r"(?![a-z0-9])"
    return [match.start() for match in re.finditer(pattern, lowered_text)]


def _looks_like_filter_context(lowered_text: str) -> bool:
    hits = sum(1 for term in FILTER_CONTEXT_TERMS if term in lowered_text)
    numeric_range = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|ac)?\s*-\s*\d+(?:\.\d+)?", lowered_text))
    parenthetical_counts = len(re.findall(r"\(\d+\)", lowered_text))
    return hits >= 3 or (hits >= 2 and (numeric_range or parenthetical_counts >= 3))


def _looks_like_navigation_context(lowered_text: str) -> bool:
    navigation_hits = sum(
        1
        for term in (
            "advertisement",
            "anatomy of",
            "apparel",
            "best of",
            "continue reading below",
            "daily trainers",
            "events videos podcasts",
            "best running shoes",
            "first impressions",
            "lifestyle trail",
            "most popular brands",
            "new balance nike asics hoka",
            "review (2025)",
            "road best of",
            "podcast",
            "mailbag",
            "we rank our top",
            "shoe finder",
            "shop the shoe",
            "tempo race day",
            "technical non-technical",
            "wide foot track/xc",
        )
        if term in lowered_text
    )
    return navigation_hits >= 2


def _repair_evidence_is_supported(snippet: str, aspect: str) -> bool:
    lowered = snippet.lower()
    if _looks_like_filter_context(lowered) or _looks_like_navigation_context(lowered):
        return False
    if _looks_like_comparative_aside(lowered):
        return False
    if aspect == "comfort" and _has_vehicle_comfort_complaint(lowered):
        return True
    if aspect == "taste" and _has_taste_disappointment_support(lowered):
        return True
    if _looks_like_mixed_positive_limitation(lowered):
        return True
    if _looks_like_positive_section(lowered) and not any(term in lowered for term in ("cons :", "cons:", "what we don't like", "what we do not like")):
        return False
    return any(_has_negative_cue_context(snippet, cue, window=130) for cue in NEGATIVE_ASPECT_CUES.get(aspect, ()))


def _has_vehicle_comfort_complaint(lowered_text: str) -> bool:
    attention_terms = (
        "apply pressure",
        "steering wheel",
        "every 30 seconds",
        "interior camera",
        "driver attention",
        "detecting inattention",
        "prove you were still in control",
    )
    if any(term in lowered_text for term in attention_terms) and any(
        term in lowered_text for term in ("uncomfortable", "stressful", "stress", "annoying", "intrusive")
    ):
        return True
    rear_space_terms = (
        "rear seat",
        "back seat",
        "taller passengers",
        "bump their knees",
        "not much headroom",
        "tight space",
        "legroom",
        "headroom",
    )
    if any(term in lowered_text for term in rear_space_terms) and any(
        term in lowered_text for term in ("sore point", "tight", "not much", "bump", "cramped", "limited")
    ):
        return True
    return False


def _evidence_support_tier(snippet: str, aspect: str) -> str:
    lowered = snippet.lower()
    if _looks_like_mixed_positive_limitation(lowered):
        return "exploratory"
    if aspect == "taste" and _has_taste_disappointment_support(lowered):
        return "moderate"
    strong_markers = (
        "cons ",
        "cons:",
        "complaint",
        "drawback",
        "blister",
        "blistering",
        "rubbing",
        "irritation",
        "lace bite",
        "uncomfortable",
        "not comfortable",
        "too tight",
        "too narrow",
        "heel slip",
        "too bitter",
        "waxy",
        "chalky",
        "disappointment",
        "disappointing",
        "disappointed",
        "not worth",
    )
    moderate_markers = (
        "major problem",
        "too stiff",
        "not ideal",
        "should look elsewhere",
        "problem",
        "lacks",
        "lack of",
    )
    if any(marker in lowered for marker in strong_markers):
        return "strong"
    if any(marker in lowered for marker in moderate_markers):
        return "moderate"
    return "exploratory"


def _looks_like_mixed_positive_limitation(lowered_text: str) -> bool:
    return bool(re.search(r"\bnot\s+(?:be\s+)?ideal\b", lowered_text)) and any(
        term in lowered_text
        for term in (
            "provides enough stability",
            "reliable, strong feel",
            "reliable strong feel",
            "impressive results",
        )
    )


def _looks_like_comparative_aside(lowered_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:better than|far better than|rather than)\b.{0,90}\b(?:shoe|product|option)\b.{0,120}\b(?:uncomfortable|not worth|too expensive|falling flat)",
            lowered_text,
        )
    )


def _looks_like_positive_section(snippet: str) -> bool:
    return any(term in snippet for term in ("what we like", "pros :", "pros:", "shop the shoe")) and not any(
        term in snippet for term in ("cons ", "cons :", "cons:", "what we don't like", "what we do not like")
    )


def _looks_like_safety_context(snippet: str) -> bool:
    return any(
        term in snippet
        for term in (
            "hard for kids to get hurt",
            "would be hard for kids to get hurt",
            "unlikely to get hurt",
            "safe for kids",
        )
    )


def _has_negative_stance(snippet: str) -> bool:
    return any(term in snippet for term in NEGATIVE_STANCE_TERMS)


def _has_positive_context(snippet: str) -> bool:
    if any(term in snippet for term in ("cons ", "cons :", "cons:", "what we don't like", "what we do not like")):
        return False
    return any(_contains_phrase(snippet, term) for term in POSITIVE_CONTEXT_TERMS)


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _cue_for_aspect(text: str, aspect: str) -> str:
    lowered = text.lower()
    for cue in NEGATIVE_ASPECT_CUES.get(aspect, ()):
        if _cue_indices(lowered, cue.lower()):
            return cue
    return aspect


def _focused_evidence(text: str, aspect: str, *, window: int = 190) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    cue = _cue_for_aspect(cleaned, aspect)
    index = lowered.find(cue.lower())
    if index < 0:
        return cleaned[: 2 * window].strip()
    start = max(0, index - window)
    end = min(len(cleaned), index + len(cue) + window)
    snippet = cleaned[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(cleaned):
        snippet += " ..."
    return snippet


def _topos_status(aspect: str, topos_psr: dict[str, object]) -> str:
    aliases = ASPECT_TOPOS_CONTEXTS.get(aspect, (aspect,))
    restrictions = list(topos_psr.get("restriction_diagnostics") or [])
    for row in restrictions:
        target = str(row.get("target_context") or "")
        if _context_matches(target, aliases):
            return "overlap_stable" if row.get("compatible") else "context_tense"
    contexts = {
        str(row.get("context_id") or "")
        for row in list(topos_psr.get("local_hankel_family") or [])
    }
    return "local_psr" if any(_context_matches(context, aliases) for context in contexts) else "local_only"


def _context_matches(context: str, targets: tuple[str, ...]) -> bool:
    lowered = context.lower()
    compact = lowered.replace("_", "").replace("-", "").replace("/", "")
    for target in targets:
        target_lower = target.lower()
        target_compact = target_lower.replace("_", "").replace("-", "").replace("/", "")
        if target_lower in lowered or target_compact in compact:
            return True
    return False


def _prometheus_twm_aspect_support(aspect: str, prometheus_twm: dict[str, object]) -> dict[str, object] | None:
    targets = ASPECT_TWM_CONTEXTS.get(aspect, (aspect,))
    diagnostics = list(prometheus_twm.get("gluing_diagnostics") or [])
    best: dict[str, object] | None = None
    best_score = -1.0
    for row in diagnostics:
        source = str(row.get("source_context") or "")
        target = str(row.get("target_context") or "")
        if not (_context_matches(source, targets) or _context_matches(target, targets)):
            continue
        weighted_loss = _safe_float(row.get("weighted_glue_loss"))
        glue_loss = _safe_float(row.get("glue_loss"))
        sections = int(_safe_float(row.get("overlap_sections"), 0.0))
        confidence = _safe_float(row.get("overlap_confidence"), 0.0)
        compatible = bool(row.get("compatible"))
        touches_satisfaction = _context_matches(source, SATISFACTION_TWM_CONTEXTS) or _context_matches(
            target, SATISFACTION_TWM_CONTEXTS
        )
        tense = (not compatible) or weighted_loss >= 0.075
        if sections <= 0 or not tense:
            continue
        if sections < 2 and not touches_satisfaction:
            continue
        score = weighted_loss + 0.04 * sections + (0.08 if touches_satisfaction else 0.0) + 0.02 * confidence
        if score > best_score:
            best_score = score
            best = {
                "source_context": source,
                "target_context": target,
                "overlap_sections": sections,
                "weighted_glue_loss": weighted_loss,
                "glue_loss": glue_loss,
                "overlap_confidence": confidence,
                "compatible": compatible,
                "touches_satisfaction": touches_satisfaction,
                "topos_status": "context_tense" if not compatible else "overlap_tense",
            }
    return best


def _counterfactual_sentence(product_label: str, aspect: str, repair: str, score: float) -> str:
    subject = product_label or "the product"
    return (
        f"If {subject} had been redesigned with {repair}, then this negative `{aspect}` "
        f"state would be replaced by a positive local state, with an estimated "
        f"satisfaction gain of {score:.3f} under the CLIFF/Prometheus repair probe."
    )


def _query_conditioned_sentence(product_label: str, aspect: str, repair: str, score: float) -> str:
    subject = product_label or "the product"
    return (
        f"If {subject} were evaluated under a local j-do repair for `{aspect}` using {repair}, "
        f"Prometheus would treat this as a query-conditioned design probe with an estimated "
        f"satisfaction gain of {score:.3f}; no supported negative local state was isolated in the retrieved evidence."
    )


def _gluing_supported_sentence(product_label: str, aspect: str, repair: str, score: float) -> str:
    subject = product_label or "the product"
    return (
        f"If {subject} were repaired for `{aspect}` using {repair}, Prometheus GB would treat the "
        f"repair as gluing-supported with an estimated satisfaction gain of {score:.3f}; the support "
        f"comes from a tense overlap in the sheaf model rather than a directly isolated complaint."
    )


def _repair_candidate(
    event: dict[str, object],
    *,
    aspect: str,
    rule: dict[str, str],
    evidence: str,
    support_tier: str,
    score: float,
    product_label: str,
    topos_payload: dict[str, object],
    evidence_status: str = "supported",
) -> dict[str, object]:
    return {
        "feedback_id": str(event.get("feedback_id") or ""),
        "title": str(event.get("title") or event.get("feedback_id") or "feedback"),
        "aspect": aspect,
        "observed_sentiment": str(event.get("sentiment") or "negative"),
        "observed_evidence": evidence,
        "repair": rule["repair"],
        "rationale": rule["rationale"],
        "support_tier": support_tier,
        "evidence_status": evidence_status,
        "score": round(score, 3),
        "estimated_satisfaction_gain": round(score, 3),
        "topos_status": _topos_status(aspect, topos_payload),
        "counterfactual": _counterfactual_sentence(product_label, aspect, rule["repair"], score),
        "semantics": "prometheus_local_repair_probe_v1",
    }


def _gluing_supported_candidate(
    event: dict[str, object],
    *,
    aspect: str,
    rule: dict[str, str],
    product_label: str,
    product_name: str,
    support: dict[str, object],
) -> dict[str, object]:
    source = str(support.get("source_context") or "local chart")
    target = str(support.get("target_context") or "neighboring chart")
    weighted_loss = _safe_float(support.get("weighted_glue_loss"))
    confidence = _safe_float(support.get("overlap_confidence"))
    sections = int(_safe_float(support.get("overlap_sections"), 0.0))
    score = min(0.55, 0.06 + min(0.20, weighted_loss) + (0.04 if support.get("touches_satisfaction") else 0.0))
    title = str(event.get("title") or event.get("feedback_id") or "retrieved feedback")
    evidence = (
        f"No directly supported negative `{aspect}` state was isolated from the retrieved snippets for "
        f"`{product_name}`. Prometheus GB did find a tense overlap between `{source}` and `{target}` "
        f"over {sections} shared section(s), with confidence {confidence:.3f} and weighted glue loss "
        f"{weighted_loss:.4f}. Nearest retrieved source: {title}."
    )
    return {
        "feedback_id": str(event.get("feedback_id") or "gluing_supported"),
        "title": title,
        "aspect": aspect,
        "observed_sentiment": "gluing_supported",
        "observed_evidence": evidence,
        "repair": rule["repair"],
        "rationale": rule["rationale"],
        "support_tier": "gb-tense",
        "evidence_status": "gluing_supported",
        "score": round(score, 3),
        "estimated_satisfaction_gain": round(score, 3),
        "topos_status": str(support.get("topos_status") or "context_tense"),
        "counterfactual": _gluing_supported_sentence(product_label, aspect, rule["repair"], score),
        "semantics": "prometheus_gluing_supported_repair_probe_v1",
        "gluing_support": dict(support),
    }


def _query_conditioned_candidate(
    event: dict[str, object],
    *,
    aspect: str,
    rule: dict[str, str],
    product_label: str,
    product_name: str,
    topos_payload: dict[str, object],
) -> dict[str, object]:
    score = 0.05
    title = str(event.get("title") or event.get("feedback_id") or "retrieved feedback")
    evidence = (
        f"No supported negative `{aspect}` state was isolated from the retrieved snippets for "
        f"`{product_name}`. This fallback is conditioned on the user's query and should be read as "
        f"a low-confidence design probe, not an observed complaint. Nearest retrieved source: {title}."
    )
    return {
        "feedback_id": str(event.get("feedback_id") or "query_conditioned"),
        "title": title,
        "aspect": aspect,
        "observed_sentiment": "query_conditioned",
        "observed_evidence": evidence,
        "repair": rule["repair"],
        "rationale": rule["rationale"],
        "support_tier": "query-conditioned",
        "evidence_status": "query_conditioned",
        "score": score,
        "estimated_satisfaction_gain": score,
        "topos_status": _topos_status(aspect, topos_payload),
        "counterfactual": _query_conditioned_sentence(product_label, aspect, rule["repair"], score),
        "semantics": "prometheus_query_conditioned_repair_probe_v1",
    }


def _select_repair_rows(
    rows: list[dict[str, object]],
    *,
    limit: int,
    max_per_aspect: int,
) -> tuple[list[dict[str, object]], Counter[str]]:
    selected: list[dict[str, object]] = []
    aspect_counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for candidate in rows:
        key = (str(candidate["feedback_id"]), str(candidate["aspect"]))
        if key in seen:
            continue
        aspect = str(candidate["aspect"])
        if aspect_counts[aspect] >= max_per_aspect:
            continue
        selected.append(candidate)
        seen.add(key)
        aspect_counts[aspect] += 1
        if len(selected) >= limit:
            break
    return selected, aspect_counts


def build_product_feedback_counterfactuals(
    events: list[dict[str, object]],
    *,
    topos_psr: dict[str, object] | None = None,
    prometheus_twm: dict[str, object] | None = None,
    product_name: str = "",
    brand_name: str = "",
    limit: int = 12,
    max_per_aspect: int = 3,
) -> dict[str, object]:
    """Build Prometheus-style local repair counterfactuals for dashboard display."""

    topos_payload = dict(topos_psr or {})
    prometheus_twm_payload = dict(prometheus_twm or {})
    product_label = _clean_product_label(product_name, brand_name)
    product_domain = _product_domain(product_name, brand_name)
    allowed_aspects = DOMAIN_ALLOWED_ASPECTS.get(product_domain, DOMAIN_ALLOWED_ASPECTS["generic"])
    candidates: list[dict[str, object]] = []
    exploratory_candidates: list[dict[str, object]] = []

    for event in events:
        negative_aspects = _event_negative_aspects(event)
        if not negative_aspects:
            continue
        sentiment_score = _safe_float(event.get("sentiment_score"))
        rating = event.get("rating")
        rating_pressure = 0.0
        if rating not in (None, ""):
            rating_pressure = max(0.0, (3.0 - _safe_float(rating, 3.0)) / 2.0)
        return_pressure = 0.25 if event.get("return_risk_signal") or event.get("returned") else 0.0

        for aspect in negative_aspects:
            if aspect not in allowed_aspects:
                continue
            rule = REPAIR_RULES.get(aspect)
            if rule is None:
                continue
            polarity_pressure = max(0.0, -sentiment_score)
            score = min(0.95, 0.10 + 0.35 * polarity_pressure + 0.20 * rating_pressure + return_pressure)
            evidence = _focused_evidence(str(event.get("text") or ""), aspect)
            if not _repair_evidence_is_supported(evidence, aspect):
                continue
            rule = _vehicle_comfort_repair_rule(
                aspect=aspect,
                evidence=evidence,
                event=event,
                product_domain=product_domain,
                base_rule=rule,
            )
            support_tier = _evidence_support_tier(evidence, aspect)
            candidates.append(
                _repair_candidate(
                    event,
                    aspect=aspect,
                    rule=rule,
                    evidence=evidence,
                    support_tier=support_tier,
                    score=score,
                    product_label=product_label,
                    topos_payload=topos_payload,
                )
            )

    if events and prometheus_twm_payload:
        fallback_event = next(
            (
                event
                for event in events
                if not _looks_like_filter_context(str(event.get("text") or "").lower())
                and not _looks_like_navigation_context(str(event.get("text") or "").lower())
            ),
            events[0],
        )
        candidate_aspects = {str(row.get("aspect") or "") for row in candidates}
        for aspect in _query_conditioned_aspects(product_name, product_domain, allowed_aspects):
            if aspect in candidate_aspects:
                continue
            rule = REPAIR_RULES.get(aspect)
            if rule is None:
                continue
            support = _prometheus_twm_aspect_support(aspect, prometheus_twm_payload)
            if support is None:
                continue
            candidates.append(
                _gluing_supported_candidate(
                    fallback_event,
                    aspect=aspect,
                    rule=rule,
                    product_label=product_label,
                    product_name=product_name,
                    support=support,
                )
            )

    candidates.sort(key=lambda row: (-float(row["score"]), str(row["feedback_id"]), str(row["aspect"])))
    selected, aspect_counts = _select_repair_rows(candidates, limit=limit, max_per_aspect=max_per_aspect)
    selected_keys = {(str(row["feedback_id"]), str(row["aspect"])) for row in selected}

    for event in events:
        exploratory_aspects = _event_exploratory_aspects(event, allowed_aspects, product_domain)
        if not exploratory_aspects:
            continue
        sentiment_score = _safe_float(event.get("sentiment_score"))
        rating = event.get("rating")
        rating_pressure = 0.0
        if rating not in (None, ""):
            rating_pressure = max(0.0, (3.0 - _safe_float(rating, 3.0)) / 2.0)
        return_pressure = 0.25 if event.get("return_risk_signal") or event.get("returned") else 0.0
        for aspect in exploratory_aspects:
            key = (str(event.get("feedback_id") or ""), aspect)
            if key in selected_keys:
                continue
            rule = REPAIR_RULES.get(aspect)
            if rule is None:
                continue
            evidence_source = f"{event.get('title') or ''}. {event.get('text') or ''}".strip()
            evidence = _focused_evidence(evidence_source, aspect)
            lowered_evidence = evidence.lower()
            if _looks_like_filter_context(lowered_evidence) or _looks_like_navigation_context(lowered_evidence):
                continue
            if _looks_like_positive_section(lowered_evidence) or _looks_like_comparative_aside(lowered_evidence):
                continue
            polarity_pressure = max(0.0, -sentiment_score)
            score = min(0.95, 0.08 + 0.25 * polarity_pressure + 0.15 * rating_pressure + return_pressure)
            rule = _vehicle_comfort_repair_rule(
                aspect=aspect,
                evidence=evidence,
                event=event,
                product_domain=product_domain,
                base_rule=rule,
            )
            exploratory_candidates.append(
                _repair_candidate(
                    event,
                    aspect=aspect,
                    rule=rule,
                    evidence=evidence,
                    support_tier="exploratory",
                    evidence_status="domain_plausible",
                    score=score,
                    product_label=product_label,
                    topos_payload=topos_payload,
                )
            )

    exploratory_candidates.sort(key=lambda row: (-float(row["score"]), str(row["feedback_id"]), str(row["aspect"])))
    exploratory_selected, exploratory_aspect_counts = _select_repair_rows(
        exploratory_candidates,
        limit=max(4, limit // 2),
        max_per_aspect=max(1, max_per_aspect),
    )
    if not selected and not exploratory_selected and events:
        fallback_rows: list[dict[str, object]] = []
        fallback_event = next(
            (
                event
                for event in events
                if not _looks_like_filter_context(str(event.get("text") or "").lower())
                and not _looks_like_navigation_context(str(event.get("text") or "").lower())
            ),
            events[0],
        )
        for aspect in _query_conditioned_aspects(product_name, product_domain, allowed_aspects):
            rule = REPAIR_RULES.get(aspect)
            if rule is None:
                continue
            fallback_rows.append(
                _query_conditioned_candidate(
                    fallback_event,
                    aspect=aspect,
                    rule=rule,
                    product_label=product_label,
                    product_name=product_name,
                    topos_payload=topos_payload,
                )
            )
        exploratory_selected, exploratory_aspect_counts = _select_repair_rows(
            fallback_rows,
            limit=2,
            max_per_aspect=1,
        )

    return {
        "schema_version": "cliff.prometheus_counterfactuals.v1",
        "summary": {
            "builder_version": COUNTERFACTUAL_BUILDER_VERSION,
            "counterfactual_count": len(selected),
            "candidate_count": len(candidates),
            "exploratory_count": len(exploratory_selected),
            "exploratory_candidate_count": len(exploratory_candidates),
            "aspect_counts": dict(aspect_counts),
            "exploratory_aspect_counts": dict(exploratory_aspect_counts),
            "support_tier_counts": dict(Counter(str(row.get("support_tier") or "unknown") for row in selected)),
            "exploratory_support_tier_counts": dict(
                Counter(str(row.get("support_tier") or "unknown") for row in exploratory_selected)
            ),
            "semantics": (
                "Prometheus-style local j-do repair probes over negative product-feedback states, "
                "with GB tense-overlap support for queried aspects when direct complaint evidence is absent."
            ),
        },
        "counterfactuals": selected,
        "exploratory_counterfactuals": exploratory_selected,
    }
