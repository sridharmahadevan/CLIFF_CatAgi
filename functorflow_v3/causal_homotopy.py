"""Shared normalization helpers for homotopy-localized causal claims."""

from __future__ import annotations

import re

_LEADING_PHRASE_PREFIXES = (
    "the use of ",
    "use of ",
    "treatment with ",
    "treatment using ",
    "administration of ",
    "exposure to ",
    "an increase in ",
    "increase in ",
    "a rise in ",
    "rise in ",
    "the effect of ",
    "effect of ",
)

_CLAUSE_MARKERS = (
    ", which ",
    " which ",
    " by ",
    " due to ",
    " through ",
    " allowing ",
    " because ",
    " while ",
    " when ",
)

_PHRASE_REWRITES: tuple[tuple[str, str], ...] = (
    (r"\bglucagon[\s-]+like[\s-]+peptide[\s-]+1\b", "glp1"),
    (r"\bglp[\s-]*1\b", "glp1"),
    (r"\bglp1ras?\b", "glp1 receptor agonist"),
    (r"\bglp1 receptor agonists\b", "glp1 receptor agonist"),
    (r"\bglp1 medicines\b", "glp1 receptor agonist"),
    (r"\bglp1 drugs\b", "glp1 receptor agonist"),
    (r"\bindividuals\b", "people"),
    (r"\bpatients\b", "people"),
    (r"\bsubjects\b", "people"),
    (r"\bpersons\b", "people"),
    (r"\bmoving\b", "move"),
    (r"\bmoves\b", "move"),
)

_RELATION_REWRITES = {
    "leads_to": "causes",
    "leads to": "causes",
    "drives": "causes",
    "results_in": "causes",
    "results in": "causes",
    "influences": "affects",
}

_POSITIVE_RELATIONS = {
    "causes",
    "affects",
    "increases",
    "supports",
}

_NEGATIVE_RELATIONS = {
    "reduces",
    "decreases",
    "prevents",
    "blocks",
    "inhibits",
}


def normalize_relation(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    normalized = normalized.replace("-", "_")
    return _RELATION_REWRITES.get(normalized, normalized)


def relation_polarity(value: str) -> str:
    normalized = normalize_relation(value)
    if normalized in _POSITIVE_RELATIONS:
        return "positive"
    if normalized in _NEGATIVE_RELATIONS:
        return "negative"
    return normalized


def normalize_claim_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for marker in _CLAUSE_MARKERS:
        if marker in text:
            prefix, _, _ = text.partition(marker)
            if prefix.strip():
                text = prefix.strip()
                break
    text = text.replace("_", " ")
    text = re.sub(r"[\-/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in _PHRASE_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    changed = True
    while changed and text:
        changed = False
        for prefix in _LEADING_PHRASE_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    text = re.sub(r"^(?:the|a|an)\s+", "", text).strip()
    return re.sub(r"\s+", " ", text).strip()
