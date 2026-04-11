"""Query-driven corpus acquisition layer for agentic Democritus."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import textwrap
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from html import escape, unescape
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import unquote, urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from .democritus_batch_agentic import (
    DemocritusBatchAgenticRunner,
    DemocritusBatchConfig,
    DemocritusBatchRecord,
    DemocritusBatchRunResult,
)
from .dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
from .evidence_convergence import (
    EvidenceConvergenceAdapter,
    EvidenceConvergencePolicy,
    EvidenceConvergenceTracker,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "benefits",
    "can",
    "drinking",
    "find",
    "filing",
    "filings",
    "for",
    "from",
    "have",
    "help",
    "id",
    "in",
    "interested",
    "is",
    "it",
    "jointly",
    "know",
    "learn",
    "like",
    "me",
    "n",
    "of",
    "on",
    "or",
    "please",
    "report",
    "reports",
    "show",
    "support",
    "supports",
    "synthesize",
    "synthesizes",
    "synthesized",
    "synthesis",
    "studies",
    "study",
    "that",
    "the",
    "these",
    "those",
    "they",
    "their",
    "tell",
    "to",
    "topic",
    "topics",
    "focus",
    "avoid",
    "recent",
    "joint",
    "understand",
    "want",
    "what",
    "with",
    "would",
    "analyze",
    "analyzed",
    "analysis",
    "article",
    "articles",
    "at",
    "document",
    "documents",
    "page",
    "pages",
    "url",
}

_GENERIC_RETRIEVAL_TOKENS = {
    "benefit",
    "benefits",
    "burden",
    "consumption",
    "diet",
    "drinking",
    "effect",
    "effects",
    "exposure",
    "health",
    "impact",
    "impacts",
    "lifestyle",
    "management",
    "mechanism",
    "mechanisms",
    "outcome",
    "outcomes",
    "patient",
    "patients",
    "population",
    "populations",
    "prevention",
    "protective",
    "response",
    "responses",
    "review",
    "reviews",
    "risk",
    "risks",
    "safety",
    "therapy",
    "treatment",
    "treatments",
}

_AMBIGUOUS_QUERY_TERMS: tuple[dict[str, object], ...] = (
    {
        "term": "inflation",
        "reason": (
            "The term 'inflation' is ambiguous here. In research corpora it can refer to "
            "economic inflation, cosmological inflation, device or balloon inflation, and other contexts."
        ),
        "disambiguating_tokens": (
            "economics",
            "economic",
            "macroeconomic",
            "macro",
            "prices",
            "price",
            "cpi",
            "monetary",
            "fed",
            "reserve",
            "wages",
            "wage",
            "unemployment",
            "recession",
            "cosmic",
            "cosmology",
            "cosmological",
            "universe",
            "primordial",
            "balloon",
            "tire",
            "tyre",
            "cuff",
            "catheter",
            "medical",
        ),
        "disambiguating_phrases": (
            "inflation rate",
            "consumer prices",
            "price level",
            "cost of living",
            "purchasing power",
            "early universe",
            "big bang",
            "tire inflation",
            "balloon inflation",
            "cuff inflation",
        ),
        "suggested_queries": (
            "Analyze 20 recent studies on economic inflation and synthesize what they jointly support",
            "Analyze 20 recent studies on cosmic inflation in cosmology and synthesize what they jointly support",
            "Analyze 20 recent studies on balloon or device inflation and synthesize what they jointly support",
        ),
    },
)

_EXPLANATION_QUERY_PREFIXES: tuple[str, ...] = (
    "explain ",
    "what is ",
    "what are ",
    "how does ",
    "how do ",
    "help me understand ",
    "teach me ",
    "walk me through ",
)

_EVIDENCE_ACQUISITION_TOKENS = {
    "study",
    "studies",
    "paper",
    "papers",
    "article",
    "articles",
    "document",
    "documents",
    "filing",
    "filings",
    "report",
    "reports",
    "pdf",
    "corpus",
    "evidence",
    "source",
    "sources",
    "dataset",
    "datasets",
}

_BROAD_CLIMATE_QUERY_TERMS = {
    "climate",
    "change",
    "global",
    "warming",
}

_CLIMATE_BREADTH_FACETS: tuple[str, ...] = (
    "ecosystems",
    "agriculture",
    "energy",
    "water resources",
    "biodiversity",
    "adaptation policy",
    "public health",
    "extreme weather",
)

_TOPIC_EQUIVALENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "effect",
    "effects",
    "for",
    "impact",
    "impacts",
    "in",
    "of",
    "on",
    "the",
    "to",
}

_TOPIC_PHRASE_REWRITES: tuple[tuple[str, str], ...] = (
    (r"\bclimate change\b", "climate"),
    (r"\bpublic health\b", "health"),
    (r"\bheat waves\b", "heat wave"),
    (r"\bheat-related\b", "heat related"),
    (r"\bagenda-setting\b", "agenda setting"),
)

_TOPIC_TOKEN_REWRITES = {
    "adaptative": "adaptation",
    "adaptive": "adaptation",
    "adaptation": "adaptation",
    "agenda": "agenda",
    "deaths": "death",
    "death": "death",
    "diseases": "disease",
    "gaps": "gap",
    "governances": "governance",
    "illness": "illness",
    "illnesses": "illness",
    "impacts": "impact",
    "morbidity": "illness",
    "mortalities": "death",
    "mortality": "death",
    "policies": "policy",
    "strategies": "strategy",
}

_TOPIC_GENERIC_SURFACE_TOKENS = _GENERIC_RETRIEVAL_TOKENS | {
    "article",
    "articles",
    "artefact",
    "artefacts",
    "artifact",
    "artifacts",
    "author",
    "authors",
    "data",
    "document",
    "documents",
    "paper",
    "papers",
    "study",
    "studies",
}

_TOPIC_CONTEXT_EXCLUDED_TOKENS = _TOPIC_GENERIC_SURFACE_TOKENS | {
    "abstract",
    "analysis",
    "artefact",
    "artefacts",
    "artifact",
    "artifacts",
    "author",
    "authors",
    "background",
    "biorxiv",
    "conclusion",
    "copyright",
    "discussion",
    "finding",
    "findings",
    "implication",
    "implications",
    "introduction",
    "journal",
    "manuscript",
    "method",
    "methods",
    "ncbi",
    "overlook",
    "preprint",
    "result",
    "results",
    "review",
    "reviews",
    "section",
    "source",
    "supplementary",
    "trend",
    "trends",
    "university",
    "varying",
}

_SEC_COMPANY_QUERY_STOPWORDS = {
    "10",
    "8",
    "k",
    "q",
    "annual",
    "quarterly",
    "current",
    "recent",
    "latest",
    "new",
    "sec",
    "edgar",
    "filing",
    "filings",
    "form",
    "forms",
    "report",
    "reports",
    "company",
    "companies",
    "workflow",
    "workflows",
    "extract",
    "extracts",
    "extraction",
    "extractions",
    "analyze",
    "analysis",
    "review",
    "reviews",
    "document",
    "documents",
}

_DJIA_COMPANY_TARGETS: tuple[tuple[str, ...], ...] = (
    ("3m", "mmm"),
    ("amazon", "amzn"),
    ("american express", "axp"),
    ("amgen", "amgn"),
    ("apple", "aapl"),
    ("boeing", "ba"),
    ("caterpillar", "cat"),
    ("chevron", "cvx"),
    ("cisco", "csco"),
    ("coca cola", "ko"),
    ("disney", "dis"),
    ("goldman sachs", "gs"),
    ("home depot", "hd"),
    ("honeywell", "hon"),
    ("ibm", "international business machines"),
    ("johnson johnson", "jnj"),
    ("jpmorgan", "jpm"),
    ("mcdonald s", "mcd"),
    ("merck", "mrk"),
    ("microsoft", "msft"),
    ("nike", "nke"),
    ("nvidia", "nvda"),
    ("procter gamble", "pg"),
    ("salesforce", "crm"),
    ("sherwin williams", "shw"),
    ("travelers", "trv"),
    ("unitedhealth", "unh"),
    ("verizon", "vz"),
    ("visa", "v"),
    ("walmart", "wmt"),
)

_SEC_COMPANY_GROUPS: tuple[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]], ...] = (
    (("djia",), _DJIA_COMPANY_TARGETS),
    (("dow jones industrial average",), _DJIA_COMPANY_TARGETS),
    (("dow companies",), _DJIA_COMPANY_TARGETS),
    (("dow 30",), _DJIA_COMPANY_TARGETS),
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_COUNT_FILLER_PATTERN = r"(?:\s+(?:recent|latest|new|top|best|peer[- ]reviewed|open[- ]access|additional|relevant|high[- ]quality))*"
_URL_PATTERN = re.compile(r"https?://[^\s<>'\"\])]+")
_COMMON_ABSOLUTE_PATH_PREFIXES = (
    "Users/",
    "private/",
    "Volumes/",
    "Applications/",
    "Library/",
    "System/",
    "opt/",
    "etc/",
    "var/",
    "tmp/",
)
_DIRECT_PDF_TOKEN_PATTERN = re.compile(
    r"(?:file://)?(?:~|\.{1,2}/|/|Users/|private/|Volumes/|Applications/|Library/|System/|opt/|etc/|var/|tmp/)[^\s`\"']+\.pdf",
    re.IGNORECASE,
)
_DIRECT_LOCAL_TOKEN_PATTERN = re.compile(
    r"(?:file://)?(?:~|\.{1,2}/|/|Users/|private/|Volumes/|Applications/|Library/|System/|opt/|etc/|var/|tmp/)[^\s`\"']+",
    re.IGNORECASE,
)
_MAX_NON_PDF_REMOTE_BYTES = 2 * 1024 * 1024
_HEAVY_NEWS_HOST_LIMITS = {
    "washingtonpost.com": 768 * 1024,
}
_NON_PDF_STREAM_CHUNK_BYTES = 64 * 1024
_HEAVY_NEWS_HOST_TIMEOUTS = {
    "washingtonpost.com": (8.0, 20.0, 40.0),
}


def _slugify(name: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "document"


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


def _read_topic_lines(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    lines = [
        " ".join(line.split()).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(line for line in lines if line)


def _read_document_guide(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _path_to_file_uri(path_value: object) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    try:
        return Path(raw).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _checkpoint_query_focus_terms(query: str) -> tuple[str, ...]:
    retrieval_query = _derive_retrieval_query(query)
    ordered = list(_tokenize(retrieval_query))
    focus_terms = [token for token in ordered if token not in _GENERIC_RETRIEVAL_TOKENS]
    if not focus_terms:
        focus_terms = ordered
    return tuple(dict.fromkeys(focus_terms[:8]))


def _normalized_topics(values: tuple[str, ...] | list[str] | object) -> tuple[str, ...]:
    topics: list[str] = []
    for value in tuple(values or ()):
        normalized = " ".join(str(value or "").split()).strip()
        if normalized:
            topics.append(normalized)
    return tuple(dict.fromkeys(topics))


def _topic_signature(value: str) -> tuple[str, ...]:
    text = " ".join(str(value or "").lower().split()).strip()
    if not text:
        return ()
    for pattern, replacement in _TOPIC_PHRASE_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ()
    tokens: list[str] = []
    for raw_token in text.split():
        token = _TOPIC_TOKEN_REWRITES.get(raw_token, raw_token)
        if len(token) > 4 and token.endswith("s") and token not in {"analysis"}:
            token = token[:-1]
        if token and token not in _TOPIC_EQUIVALENCE_STOPWORDS:
            tokens.append(token)
    return tuple(sorted(dict.fromkeys(tokens)))


def _topic_display_rank(value: str) -> tuple[int, int]:
    text = " ".join(str(value or "").split()).strip()
    return (len(_topic_signature(text)), -len(text))


def _topic_is_low_quality_surface(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip().lower()
    if not text:
        return True
    trailing_bad_tokens = {
        "attributable",
        "by",
        "find",
        "for",
        "from",
        "in",
        "of",
        "on",
        "over",
        "source",
        "to",
        "under",
        "varying",
        "with",
    }
    raw_tokens = text.split()
    if raw_tokens and raw_tokens[-1] in trailing_bad_tokens:
        return True
    if any(
        token in {
            "artefact",
            "artefacts",
            "artifact",
            "artifacts",
            "center",
            "find",
            "ncbi",
            "source",
            "university",
        }
        for token in raw_tokens
    ):
        return True
    normalized_surface_tokens: list[str] = []
    for raw_token in raw_tokens:
        token = _TOPIC_TOKEN_REWRITES.get(raw_token, raw_token)
        if len(token) > 4 and token.endswith("s") and token not in {"analysis"}:
            token = token[:-1]
        if token:
            normalized_surface_tokens.append(token)
    content_tokens = [
        token
        for token in normalized_surface_tokens
        if token not in _TOPIC_EQUIVALENCE_STOPWORDS
    ]
    non_generic_tokens = [
        token
        for token in content_tokens
        if token not in _TOPIC_GENERIC_SURFACE_TOKENS
    ]
    if (
        len(content_tokens) <= 2
        and len(non_generic_tokens) <= 1
        and any(token in _TOPIC_GENERIC_SURFACE_TOKENS for token in normalized_surface_tokens)
    ):
        return True
    if any(token in {"alert", "alerts", "implication", "overlook", "trend", "varying"} for token in normalized_surface_tokens):
        return True
    if any(len(token) >= 5 and token.endswith(("ndez", "tions")) for token in normalized_surface_tokens):
        return True
    return not _topic_signature(text)


def _topic_has_truncated_token(topic: str, *, support_tokens: set[str]) -> bool:
    signature = _topic_signature(topic)
    if not signature or not support_tokens:
        return False
    for token in signature:
        if token in support_tokens or len(token) < 5:
            continue
        if any(
            support.startswith(token) and len(support) >= len(token) + 2
            for support in support_tokens
        ):
            return True
    return False


def _topic_is_subsumed_fragment(topic: str, *, other_topics: tuple[str, ...]) -> bool:
    signature = set(_topic_signature(topic))
    if not signature:
        return True
    topic_text = " ".join(str(topic or "").split()).strip().lower()
    for other in other_topics:
        other_text = " ".join(str(other or "").split()).strip().lower()
        if other_text == topic_text:
            continue
        other_signature = set(_topic_signature(other))
        if not other_signature:
            continue
        if signature < other_signature and (
            topic_text in other_text
            or len(other_signature) - len(signature) <= 2
        ):
            return True
    return False


def _fallback_topic_phrases_from_context(
    *,
    title: str,
    guide_summary: str,
    causal_gestalt: str,
    limit: int = 6,
) -> tuple[str, ...]:
    context_parts = [
        " ".join(str(title or "").split()).strip(),
        " ".join(str(guide_summary or "").split()).strip(),
        " ".join(str(causal_gestalt or "").split()).strip(),
    ]
    context_text = " ".join(part for part in context_parts if part).strip().lower()
    if not context_text:
        return ()
    raw_tokens = re.findall(r"[a-z][a-z\-]{2,}", context_text)
    normalized_tokens: list[str] = []
    for raw_token in raw_tokens:
        token = _TOPIC_TOKEN_REWRITES.get(raw_token, raw_token)
        if len(token) > 4 and token.endswith("s") and token not in {"analysis"}:
            token = token[:-1]
        if (
            token
            and len(token) >= 4
            and token not in _STOPWORDS
            and token not in _TOPIC_CONTEXT_EXCLUDED_TOKENS
        ):
            normalized_tokens.append(token)
    if not normalized_tokens:
        return ()
    counts: Counter[str] = Counter()
    for size in (3, 2):
        for index in range(len(normalized_tokens) - size + 1):
            phrase = " ".join(normalized_tokens[index : index + size])
            if _topic_is_low_quality_surface(phrase):
                continue
            counts[phrase] += 1
    ordered = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), -len(item[0].split()), item[0]),
    )
    fallback_topics: list[str] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for phrase, _count in ordered:
        signature = _topic_signature(phrase)
        if not signature or signature in seen_signatures:
            continue
        fallback_topics.append(phrase)
        seen_signatures.add(signature)
        if len(fallback_topics) >= max(1, int(limit)):
            break
    return tuple(fallback_topics)


def _topic_support_tokens(*parts: str) -> set[str]:
    supported: set[str] = set()
    for part in parts:
        raw_tokens = re.findall(r"[a-z][a-z\-]{2,}", " ".join(str(part or "").split()).strip().lower())
        for raw_token in raw_tokens:
            token = _TOPIC_TOKEN_REWRITES.get(raw_token, raw_token)
            if len(token) > 4 and token.endswith("s") and token not in {"analysis"}:
                token = token[:-1]
            if (
                token
                and len(token) >= 4
                and token not in _STOPWORDS
                and token not in _TOPIC_CONTEXT_EXCLUDED_TOKENS
            ):
                supported.add(token)
    return supported


def _prepare_document_topics(
    raw_topics: tuple[str, ...] | list[str] | object,
    *,
    title: str,
    guide_summary: str,
    causal_gestalt: str,
    limit: int = 8,
) -> tuple[str, ...]:
    support_tokens = _topic_support_tokens(title, guide_summary, causal_gestalt)
    candidate_topics: list[str] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for topic in _normalized_topics(raw_topics):
        if _topic_is_low_quality_surface(topic):
            continue
        signature = _topic_signature(topic) or (topic.lower(),)
        if support_tokens and not any(token in support_tokens for token in signature):
            continue
        if _topic_has_truncated_token(topic, support_tokens=support_tokens):
            continue
        if signature in seen_signatures:
            continue
        candidate_topics.append(topic)
        seen_signatures.add(signature)
    for topic in _fallback_topic_phrases_from_context(
        title=title,
        guide_summary=guide_summary,
        causal_gestalt=causal_gestalt,
        limit=limit,
    ):
        signature = _topic_signature(topic) or (topic.lower(),)
        if signature in seen_signatures:
            continue
        candidate_topics.append(topic)
        seen_signatures.add(signature)
    cleaned_topics: list[str] = []
    for topic in candidate_topics:
        if _topic_is_subsumed_fragment(topic, other_topics=tuple(candidate_topics)):
            continue
        cleaned_topics.append(topic)
        if len(cleaned_topics) >= max(1, int(limit)):
            break
    return tuple(cleaned_topics)


def _merge_fragmented_topic_records(
    grouped: dict[tuple[str, ...], dict[str, object]],
) -> dict[tuple[str, ...], dict[str, object]]:
    merged = dict(grouped)
    signatures = list(merged.keys())
    removed: set[tuple[str, ...]] = set()
    for signature in signatures:
        if signature in removed or signature not in merged:
            continue
        record = merged[signature]
        runs = set(record.get("document_runs") or ())
        topic = str(record.get("topic") or "")
        for other_signature in signatures:
            if (
                other_signature == signature
                or other_signature in removed
                or other_signature not in merged
            ):
                continue
            other_record = merged[other_signature]
            other_runs = set(other_record.get("document_runs") or ())
            other_topic = str(other_record.get("topic") or "")
            if not set(signature) < set(other_signature):
                continue
            if runs and other_runs and not runs.issubset(other_runs):
                continue
            if not (
                topic.lower() in other_topic.lower()
                or len(other_signature) - len(signature) <= 2
            ):
                continue
            title_values = list(other_record.get("representative_titles") or [])
            for item in list(record.get("representative_titles") or []):
                if item not in title_values:
                    title_values.append(item)
            other_record["representative_titles"] = title_values[:3]
            cast_runs = other_record.get("document_runs")
            if isinstance(cast_runs, set):
                cast_runs.update(runs)
            removed.add(signature)
            break
    for signature in removed:
        merged.pop(signature, None)
    return merged


def _collapse_topic_equivalence_classes(
    documents_payload: list[dict[str, object]],
    *,
    limit: int = 16,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for document in documents_payload:
        run_name = str(document.get("run_name") or "")
        title = " ".join(str(document.get("title") or "").split()).strip()
        for raw_topic in list(document.get("topics") or []):
            topic = " ".join(str(raw_topic or "").split()).strip()
            if not topic or _topic_is_low_quality_surface(topic):
                continue
            signature = _topic_signature(topic) or (topic.lower(),)
            record = grouped.setdefault(
                signature,
                {
                    "topic": topic,
                    "document_runs": set(),
                    "aliases": [],
                    "representative_titles": [],
                },
            )
            current_topic = str(record.get("topic") or "")
            if _topic_display_rank(topic) > _topic_display_rank(current_topic):
                record["topic"] = topic
            alias_values = list(record.get("aliases") or [])
            if topic not in alias_values:
                alias_values.append(topic)
            record["aliases"] = alias_values
            if run_name:
                cast_runs = record.get("document_runs")
                if isinstance(cast_runs, set):
                    cast_runs.add(run_name)
            title_values = list(record.get("representative_titles") or [])
            if title and title not in title_values:
                title_values.append(title)
            record["representative_titles"] = title_values[:3]
    grouped = _merge_fragmented_topic_records(grouped)
    collapsed = [
        {
            "topic": str(item.get("topic") or ""),
            "document_count": len(item.get("document_runs") or ()),
            "aliases": list(item.get("aliases") or []),
            "equivalence_class_size": len(list(item.get("aliases") or [])),
            "representative_titles": list(item.get("representative_titles") or []),
        }
        for item in grouped.values()
    ]
    collapsed.sort(
        key=lambda item: (
            -int(item.get("document_count") or 0),
            -int(item.get("equivalence_class_size") or 0),
            str(item.get("topic") or "").lower(),
        )
    )
    return collapsed[:limit]


def _scholarly_query_variants(plan: "QueryPlan") -> tuple[str, ...]:
    base_query = " ".join(str(plan.retrieval_query or plan.query or "").split()).strip()
    if not base_query:
        return ()
    variants = [base_query]
    lowered = base_query.lower()
    focus_terms = _checkpoint_query_focus_terms(base_query)
    non_climate_focus = [
        term for term in focus_terms
        if term not in _BROAD_CLIMATE_QUERY_TERMS
    ]
    if (
        any(phrase in lowered for phrase in ("climate change", "global warming", "climate crisis"))
        and not non_climate_focus
    ):
        for facet in _CLIMATE_BREADTH_FACETS[:6]:
            variants.append(f"{base_query} {facet}")
    return tuple(dict.fromkeys(variants))


def _document_context_text(document: "DiscoveredDocument") -> str:
    metadata_text = " ".join(
        str(value or "").strip()
        for value in dict(document.metadata or {}).values()
        if str(value or "").strip()
    )
    return " ".join(
        part
        for part in (
            str(document.title or "").strip(),
            str(document.abstract or "").strip(),
            metadata_text,
        )
        if part
    ).strip()


def _document_local_topic_profile(
    document: "DiscoveredDocument",
    *,
    query_focus_terms: tuple[str, ...],
) -> dict[str, object]:
    focus_term_set = {term for term in query_focus_terms if term}
    title_tokens = tuple(
        token
        for token in _tokenize(str(document.title or ""))
        if token not in focus_term_set and token not in _GENERIC_RETRIEVAL_TOKENS and len(token) >= 4
    )
    context_tokens = tuple(
        token
        for token in _tokenize(_document_context_text(document))
        if token not in focus_term_set and token not in _GENERIC_RETRIEVAL_TOKENS and len(token) >= 4
    )
    weighted = Counter()
    for token in context_tokens:
        weighted[token] += 1
    for token in title_tokens:
        weighted[token] += 2
    local_topics = tuple(token for token, _count in weighted.most_common(3))
    full_token_set = set(_tokenize(_document_context_text(document)))
    matched_query_terms = tuple(term for term in query_focus_terms if term in full_token_set)
    dominant_local_topic = (
        local_topics[0]
        if local_topics
        else (matched_query_terms[0] if matched_query_terms else str(document.retrieval_backend or "document"))
    )
    return {
        "local_topics": local_topics,
        "dominant_local_topic": dominant_local_topic,
        "matched_query_terms": matched_query_terms,
    }


def _annotate_document_with_local_topics(
    document: "DiscoveredDocument",
    *,
    profile: dict[str, object],
) -> "DiscoveredDocument":
    metadata = dict(document.metadata or {})
    metadata["dominant_local_topic"] = str(profile.get("dominant_local_topic") or "")
    metadata["local_topics"] = "|".join(str(item) for item in tuple(profile.get("local_topics") or ()) if str(item))
    metadata["matched_query_terms"] = "|".join(
        str(item) for item in tuple(profile.get("matched_query_terms") or ()) if str(item)
    )
    return replace(document, metadata=metadata)


def _rebalance_discovered_documents(
    plan: "QueryPlan",
    documents: tuple["DiscoveredDocument", ...],
    *,
    component_cap: int,
    score_floor_ratio: float,
    prior_selected: tuple["DiscoveredDocument", ...] = (),
) -> tuple["DiscoveredDocument", ...]:
    if len(documents) <= 2:
        return documents
    if plan.direct_document_paths or plan.direct_document_directories or plan.direct_document_urls:
        return documents
    if plan.sec_company_targets:
        return documents

    query_focus_terms = _checkpoint_query_focus_terms(plan.retrieval_query or plan.query)
    if not query_focus_terms:
        return documents

    profiles = [
        _document_local_topic_profile(document, query_focus_terms=query_focus_terms)
        for document in documents
    ]
    prior_component_counts: Counter[str] = Counter()
    for prior_document in prior_selected:
        prior_profile = _document_local_topic_profile(prior_document, query_focus_terms=query_focus_terms)
        prior_component = str(prior_profile.get("dominant_local_topic") or "").strip()
        if prior_component:
            prior_component_counts[prior_component] += 1

    best_score = max((float(document.score) for document in documents), default=0.0)
    score_floor = max(1.0, best_score * max(0.0, float(score_floor_ratio)))
    remaining = list(range(len(documents)))
    ordered: list[DiscoveredDocument] = []
    selected_component_counts: Counter[str] = Counter(prior_component_counts)

    while remaining:
        best_index = min(
            remaining,
            key=lambda index: _diversity_rank_key(
                document=documents[index],
                profile=profiles[index],
                selected_component_counts=selected_component_counts,
                component_cap=component_cap,
                score_floor=score_floor,
                original_index=index,
            ),
        )
        profile = profiles[best_index]
        annotated = _annotate_document_with_local_topics(documents[best_index], profile=profile)
        ordered.append(annotated)
        dominant_component = str(profile.get("dominant_local_topic") or "").strip()
        if dominant_component:
            selected_component_counts[dominant_component] += 1
        remaining.remove(best_index)

    return tuple(ordered)


def _diversity_rank_key(
    *,
    document: "DiscoveredDocument",
    profile: dict[str, object],
    selected_component_counts: Counter[str],
    component_cap: int,
    score_floor: float,
    original_index: int,
) -> tuple[int, int, int, int, float, int]:
    dominant_component = str(profile.get("dominant_local_topic") or "").strip()
    component_count = int(selected_component_counts.get(dominant_component, 0))
    query_overlap = len(tuple(profile.get("matched_query_terms") or ()))
    is_below_floor = float(document.score) < score_floor
    has_seen_component = 1 if component_count > 0 else 0
    exceeds_cap = 1 if component_count >= max(1, int(component_cap)) else 0
    return (
        has_seen_component if not is_below_floor else 1,
        exceeds_cap if not is_below_floor else 1,
        component_count if not is_below_floor else max(component_count, 1),
        -query_overlap,
        -float(document.score),
        original_index,
    )


def _summarize_retrieval_components(
    documents: tuple["DiscoveredDocument", ...],
    *,
    query: str,
) -> list[dict[str, object]]:
    query_focus_terms = _checkpoint_query_focus_terms(query)
    grouped: dict[str, dict[str, object]] = {}
    for document in documents:
        profile = _document_local_topic_profile(document, query_focus_terms=query_focus_terms)
        dominant_component = str(profile.get("dominant_local_topic") or "").strip()
        if not dominant_component:
            continue
        record = grouped.setdefault(
            dominant_component,
            {
                "topic": dominant_component,
                "document_count": 0,
                "matched_query_terms": list(profile.get("matched_query_terms") or ()),
                "representative_titles": [],
            },
        )
        record["document_count"] = int(record.get("document_count") or 0) + 1
        titles = list(record.get("representative_titles") or [])
        title = " ".join(str(document.title or "").split()).strip()
        if title and title not in titles:
            titles.append(title)
        record["representative_titles"] = titles[:3]
    components = sorted(
        grouped.values(),
        key=lambda item: (-int(item.get("document_count") or 0), str(item.get("topic") or "")),
    )
    return components[:8]


def _topic_alignment_diagnostics(
    *,
    query: str,
    documents_payload: list[dict[str, object]],
    top_topics: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    normalized_query = " ".join(str(query or "").lower().split())
    focus_terms = _checkpoint_query_focus_terms(query)
    topic_context: dict[str, list[str]] = {}
    for document in documents_payload:
        context_parts = [
            str(document.get("title") or ""),
            str(document.get("guide_summary") or ""),
            str(document.get("causal_gestalt") or ""),
        ]
        context_text = " ".join(part for part in context_parts if part).strip()
        for topic in list(document.get("topics") or []):
            normalized_topic = " ".join(str(topic or "").split()).strip()
            if not normalized_topic:
                continue
            topic_context.setdefault(normalized_topic, []).append(context_text)
    diagnostics: list[dict[str, object]] = []
    suspicious_topic_count = 0
    aligned_topic_count = 0
    alignment_total = 0.0
    for item in top_topics:
        topic = " ".join(str(dict(item).get("topic") or "").split()).strip()
        if not topic:
            continue
        aliases = [
            " ".join(str(alias or "").split()).strip()
            for alias in list(dict(item).get("aliases") or [])
            if " ".join(str(alias or "").split()).strip()
        ]
        alias_topics = tuple(dict.fromkeys([topic] + aliases))
        context_text = " ".join(
            entry
            for alias_topic in alias_topics
            for entry in topic_context.get(alias_topic, [])
        )
        context_tokens = set(
            _tokenize(
                " ".join(
                    part
                    for part in (
                        " ".join(alias_topics),
                        context_text,
                    )
                    if part
                )
            )
        )
        matched_query_terms = tuple(term for term in focus_terms if term in context_tokens)
        exact_phrase_match = any(alias_topic.lower() in normalized_query for alias_topic in alias_topics if alias_topic)
        alignment_score = float(len(matched_query_terms))
        if exact_phrase_match:
            alignment_score += 1.0
        if focus_terms:
            alignment_score = min(1.0, alignment_score / float(len(focus_terms)))
        is_suspicious = bool(focus_terms) and not exact_phrase_match and not matched_query_terms
        if matched_query_terms or exact_phrase_match:
            aligned_topic_count += 1
        if is_suspicious:
            suspicious_topic_count += 1
        alignment_total += alignment_score
        diagnostics.append(
            {
                "topic": topic,
                "document_count": int(dict(item).get("document_count") or 0),
                "aliases": list(alias_topics),
                "matched_query_terms": list(matched_query_terms),
                "alignment_score": round(alignment_score, 3),
                "exact_phrase_match": exact_phrase_match,
                "is_suspicious_drift": is_suspicious,
            }
        )
    total_topics = len(diagnostics)
    drift_metrics = {
        "total_topic_count": total_topics,
        "aligned_topic_count": aligned_topic_count,
        "suspicious_topic_count": suspicious_topic_count,
        "aligned_topic_ratio": round(aligned_topic_count / total_topics, 3) if total_topics else 0.0,
        "mean_alignment_score": round(alignment_total / total_topics, 3) if total_topics else 0.0,
        "synthesis_readiness_proxy": round(alignment_total / total_topics, 3) if total_topics else 0.0,
    }
    return diagnostics, {
        "query_focus_terms": list(focus_terms),
        "topic_alignment": diagnostics,
        "suspicious_topics": [item for item in diagnostics if bool(item.get("is_suspicious_drift"))],
        "drift_metrics": drift_metrics,
    }


def _render_democritus_topic_checkpoint_html(payload: dict[str, object]) -> str:
    query = escape(str(payload.get("query") or "Democritus interactive checkpoint"))
    stage_label = escape(str(payload.get("stage_label") or "Topic checkpoint"))
    summary_text = escape(str(payload.get("summary_text") or ""))
    n_documents = int(payload.get("n_documents") or 0)
    top_topics = list(payload.get("top_topics") or [])
    documents = list(payload.get("documents") or [])
    suspicious_topics = list(payload.get("suspicious_topics") or [])
    query_focus_terms = list(payload.get("query_focus_terms") or [])
    retrieval_components = list(payload.get("retrieval_components") or [])
    topic_chips = "".join(
        (
            f'<span class="chip" title="{escape("Aliases: " + " | ".join(str(alias) for alias in list(item.get("aliases") or [])[:4]))}">'
            f'{escape(str(item.get("topic") or ""))} · {escape(str(item.get("document_count") or 0))} docs'
            + (
                f' · {escape(str(int(item.get("equivalence_class_size") or 0)))} variants'
                if int(item.get("equivalence_class_size") or 0) > 1
                else ""
            )
            + "</span>"
        )
        for item in top_topics[:16]
    ) or '<span class="chip">No recurring topics detected yet</span>'
    suspicious_chips = "".join(
        f'<span class="chip drift">{escape(str(item.get("topic") or ""))}</span>'
        for item in suspicious_topics[:8]
    ) or '<span class="chip">No obvious drift topics were detected from query alignment.</span>'
    focus_term_chips = "".join(
        f'<span class="chip focus">{escape(str(term))}</span>'
        for term in query_focus_terms[:8]
    ) or '<span class="chip">No strong query anchors were extracted.</span>'
    retrieval_component_cards = "".join(
        (
            '<article class="component-card">'
            f'<div class="doc-meta">{escape(str(item.get("document_count") or 0))} retrieved doc(s)</div>'
            f'<h3 class="doc-title">{escape(str(item.get("topic") or ""))}</h3>'
            + (
                f'<p class="trace">Query overlap: {escape(" | ".join(str(term) for term in list(item.get("matched_query_terms") or [])[:3]))}</p>'
                if list(item.get("matched_query_terms") or [])
                else '<p class="trace">Query overlap: local retrieval component only</p>'
            )
            + (
                f'<p class="guide">Representative titles: {escape(" | ".join(str(title) for title in list(item.get("representative_titles") or [])[:3]))}</p>'
                if list(item.get("representative_titles") or [])
                else ""
            )
            + "</article>"
        )
        for item in retrieval_components
    ) or '<div class="empty">No retrieval-component summary was recorded for this checkpoint.</div>'
    document_cards = "".join(
        (
            '<article class="doc-card">'
            f'<div class="doc-meta">{escape(str(item.get("run_name") or ""))}</div>'
            f'<h3 class="doc-title" title="{escape(str(item.get("title") or ""))}">{escape(str(item.get("title") or ""))}</h3>'
            + (
                f'<div class="doc-actions"><a href="{escape(_path_to_file_uri(item.get("pdf_path")))}" target="_blank" rel="noopener">Inspect PDF</a></div>'
                if _path_to_file_uri(item.get("pdf_path"))
                else ""
            )
            + (
                f'<p class="guide">{escape(str(item.get("guide_summary") or ""))}</p>'
                if str(item.get("guide_summary") or "").strip()
                else ""
            )
            + (
                f'<p class="guide"><strong>Causal gestalt:</strong> {escape(str(item.get("causal_gestalt") or ""))}</p>'
                if str(item.get("causal_gestalt") or "").strip()
                else ""
            )
            + '<div class="topic-list">'
            + "".join(f'<span class="topic-pill">{escape(topic)}</span>' for topic in list(item.get("topics") or [])[:12])
            + "</div>"
            "</article>"
        )
        for item in documents
    ) or '<div class="empty">Root topics have not been materialized yet.</div>'
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Democritus Interactive Checkpoint</title>
    <style>
      :root {{
        --ink: #18222d;
        --muted: #5b6874;
        --paper: #f6f1e8;
        --card: rgba(255,255,255,0.9);
        --line: #d7ccb8;
        --accent: #93451e;
        --green: #1f6a56;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(147,69,30,0.12), transparent 24%),
          linear-gradient(180deg, #fbf7f0 0%, var(--paper) 100%);
      }}
      main {{ width: min(1280px, calc(100vw - 32px)); margin: 32px auto 48px; display: grid; gap: 18px; }}
      .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; box-shadow: 0 24px 60px rgba(30,25,18,0.08); }}
      .eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      .hero-grid {{ display: grid; gap: 18px; grid-template-columns: 1.4fr 1fr; }}
      h1, h2, h3, p {{ margin: 0; }}
      .trace {{ color: var(--muted); line-height: 1.6; }}
      .chip-row, .topic-list {{ display: flex; flex-wrap: wrap; gap: 10px; min-width: 0; }}
      .chip, .topic-pill {{ border-radius: 999px; padding: 8px 12px; background: #efe7d9; font-size: 0.92rem; color: #64492b; }}
      .chip.focus {{ background: #e8f4ee; color: #204d41; }}
      .chip.drift {{ background: #f8ede0; color: #8b4a1f; }}
      .topic-pill {{ background: #f5efe4; max-width: 100%; overflow-wrap: anywhere; }}
      .doc-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); align-items: start; }}
      .doc-card, .component-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 18px; background: #fffdf9; display: grid; gap: 12px; min-width: 0; align-content: start; overflow: hidden; }}
      .doc-meta {{ color: var(--muted); font-size: 0.9rem; }}
      .doc-title {{
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.25;
        min-width: 0;
      }}
      .doc-actions {{ min-width: 0; }}
      .guide {{ color: var(--ink); line-height: 1.65; min-width: 0; overflow-wrap: anywhere; max-width: 68ch; }}
      .callout {{ background: #f8ede0; }}
      .empty {{ color: var(--muted); line-height: 1.6; }}
      a {{ color: var(--green); text-decoration: none; font-weight: 700; }}
      a:hover {{ text-decoration: underline; }}
      @media (max-width: 920px) {{ .hero-grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel hero-grid">
        <div>
          <p class="eyebrow">Democritus Interactive Mode</p>
          <h1>{query}</h1>
          <p class="trace">Democritus paused at the <strong>{stage_label}</strong>. This preview is meant to be inspected before launching the deeper causal extraction and corpus synthesis stages.</p>
        </div>
        <div class="panel callout">
          <p class="eyebrow">Next Step</p>
          <p class="trace">{summary_text}</p>
          <p class="trace" style="margin-top:12px;">Use the session list's <strong>Go deeper</strong> button to continue from this checkpoint into claim extraction, manifold scoring, and cross-document synthesis.</p>
        </div>
      </section>
      <section class="panel">
        <p class="eyebrow">Atlas Drift Signal</p>
        <p class="trace">Democritus now treats the atlas pass as an anti-drift checkpoint before deeper extraction. Query-global anchors and suspicious retrieved-local topics are surfaced here so you can tighten the corpus early.</p>
        <div class="chip-row" style="margin-top:14px;">{focus_term_chips}</div>
        <div class="chip-row" style="margin-top:12px;">{suspicious_chips}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Retrieved Local Components</p>
        <p class="trace">These cards summarize the dominant local retrieval components before Democritus expands them into root-topic cards. If a broad query gets trapped in one basin, it should become visible here immediately.</p>
        <div class="doc-grid" style="margin-top:12px;">{retrieval_component_cards}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Corpus Topic Atlas</p>
        <p class="trace">{n_documents} documents reached the root-topic frontier. These recurring themes are the first shared causal surface Democritus recovered.</p>
        <div class="chip-row" style="margin-top:14px;">{topic_chips}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Per-Document Topics</p>
        <div class="doc-grid" style="margin-top:12px;">{document_cards}</div>
      </section>
    </main>
  </body>
</html>"""


def _build_democritus_topic_checkpoint(
    *,
    query: str,
    base_query: str,
    selected_topics: tuple[str, ...] = (),
    rejected_topics: tuple[str, ...] = (),
    retrieval_refinement: str = "",
    outdir: Path,
    batch_runner: DemocritusBatchAgenticRunner,
    selected_documents: tuple["DiscoveredDocument", ...] = (),
) -> tuple[Path, Path]:
    checkpoint_dir = outdir / "interactive_checkpoint"
    manifest_path = checkpoint_dir / "democritus_topic_checkpoint.json"
    dashboard_path = checkpoint_dir / "democritus_topic_checkpoint.html"
    documents_payload: list[dict[str, object]] = []
    topic_counter: Counter[str] = Counter()
    for document in batch_runner._documents_snapshot():
        topics_path = document.outdir / "configs" / "root_topics.txt"
        guide_path = document.outdir / "configs" / "document_topic_guide.json"
        guide_payload = _read_document_guide(guide_path)
        title = document.pdf_path.stem.replace("_", " ")
        summary_text = " ".join(str(guide_payload.get("summary") or "").split()).strip()
        raw_text = " ".join(str(guide_payload.get("raw") or "").split()).strip()
        guide_summary = summary_text or raw_text
        causal_gestalt = " ".join(str(guide_payload.get("causal_gestalt") or "").split()).strip()
        topics = _prepare_document_topics(
            _read_topic_lines(topics_path),
            title=title,
            guide_summary=summary_text,
            causal_gestalt=causal_gestalt,
        )
        if guide_summary and len(guide_summary) > 280:
            guide_summary = guide_summary[:277].rstrip() + "..."
        if causal_gestalt and len(causal_gestalt) > 280:
            causal_gestalt = causal_gestalt[:277].rstrip() + "..."
        for topic in topics:
            topic_counter[topic] += 1
        documents_payload.append(
            {
                "run_name": document.run_name,
                "title": title,
                "pdf_path": str(document.pdf_path),
                "topic_count": len(topics),
                "topics": topics,
                "guide_summary": guide_summary,
                "causal_gestalt": causal_gestalt,
            }
        )
    top_topics = _collapse_topic_equivalence_classes(documents_payload, limit=16)
    focus_query = _retrieval_source_query(
        base_query=base_query,
        selected_topics=selected_topics,
        retrieval_refinement=retrieval_refinement,
        fallback_query=query,
    )
    alignment_payload, drift_payload = _topic_alignment_diagnostics(
        query=focus_query,
        documents_payload=documents_payload,
        top_topics=top_topics,
    )
    recurring = ", ".join(item["topic"] for item in top_topics[:4]) if top_topics else "no recurring topics yet"
    suspicious_topics = list(drift_payload.get("suspicious_topics") or [])
    drift_metrics = dict(drift_payload.get("drift_metrics") or {})
    suspicious_count = int(drift_metrics.get("suspicious_topic_count") or 0)
    retrieval_components = _summarize_retrieval_components(selected_documents, query=focus_query)
    payload = {
        "query": query,
        "base_query": base_query,
        "stage_id": "root_topics",
        "stage_phase": "atlas_pass",
        "stage_label": "Atlas Drift Checkpoint",
        "n_documents": len(documents_payload),
        "top_topics": top_topics,
        "selected_topics": list(selected_topics),
        "rejected_topics": list(rejected_topics),
        "retrieval_refinement": retrieval_refinement,
        "topic_alignment": alignment_payload,
        "query_focus_terms": list(drift_payload.get("query_focus_terms") or []),
        "suspicious_topics": suspicious_topics,
        "retrieval_components": retrieval_components,
        "drift_metrics": drift_metrics,
        "documents": documents_payload,
        "summary_text": (
            "Democritus has finished the atlas pass and built an anti-drift topic surface around "
            f"{recurring}. "
            + (
                f"{suspicious_count} topic{'s look' if suspicious_count != 1 else ' looks'} weakly aligned to the query and may indicate corpus drift. "
                if suspicious_count
                else "No obvious off-scope atlas topics were detected from query alignment. "
            )
            + "Use the atlas to tighten retrieval before continuing into causal question generation, statement extraction, and cross-document synthesis."
        ),
        "recommended_next_action": "review_atlas_drift",
    }
    _write_json(manifest_path, payload)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(_render_democritus_topic_checkpoint_html(payload), encoding="utf-8")
    return manifest_path, dashboard_path


@dataclass(frozen=True)
class QueryClarificationRequest:
    """Clarification checkpoint emitted before retrieval when the query is ambiguous."""

    summary_text: str
    reason: str
    suggested_queries: tuple[str, ...]
    ambiguous_term: str = ""


def _query_clarification_request(
    query: str,
    retrieval_query: str,
    *,
    has_direct_document_input: bool,
) -> QueryClarificationRequest | None:
    if has_direct_document_input:
        return None
    normalized_query = " ".join(_strip_urls(query.lower()).split())
    retrieval_tokens = set(_tokenize(retrieval_query))
    query_tokens = set(_tokenize(normalized_query))
    for spec in _AMBIGUOUS_QUERY_TERMS:
        term = str(spec.get("term") or "").strip().lower()
        if not term or (term not in retrieval_tokens and term not in query_tokens):
            continue
        disambiguating_phrases = tuple(str(item).lower() for item in tuple(spec.get("disambiguating_phrases") or ()))
        if any(phrase and phrase in normalized_query for phrase in disambiguating_phrases):
            continue
        disambiguating_tokens = {
            str(item).strip().lower()
            for item in tuple(spec.get("disambiguating_tokens") or ())
            if str(item).strip()
        }
        if (query_tokens - {term}) & disambiguating_tokens:
            continue
        return QueryClarificationRequest(
            summary_text=(
                f"The term '{term}' needs a domain-specific meaning before Democritus can search for a reliable corpus."
            ),
            reason=str(spec.get("reason") or "").strip() or f"The term {term!r} needs clarification.",
            suggested_queries=tuple(
                str(item).strip()
                for item in tuple(spec.get("suggested_queries") or ())
                if str(item).strip()
            ),
            ambiguous_term=term,
        )
    return _query_intent_clarification_request(
        query,
        retrieval_query,
        has_direct_document_input=has_direct_document_input,
    )


def _query_intent_clarification_request(
    query: str,
    retrieval_query: str,
    *,
    has_direct_document_input: bool,
) -> QueryClarificationRequest | None:
    if has_direct_document_input:
        return None
    normalized_query = " ".join(_strip_urls(str(query).lower()).split())
    if not _looks_like_explanation_query(normalized_query):
        return None
    if _looks_like_evidence_acquisition_query(normalized_query):
        return None
    focus = _clarification_focus_text(query, retrieval_query=retrieval_query)
    return QueryClarificationRequest(
        summary_text=(
            "This looks like an explanation or teaching request, not a scoped evidence-acquisition query."
        ),
        reason=(
            "Democritus is strongest when the request names a document set or asks for studies, papers, filings, "
            "or other evidence-backed sources. Pure explanation prompts can trigger a long retrieval run and still "
            "return poor matches."
        ),
        suggested_queries=(
            f"Explain {focus} and point me to the best course demo or textbook section",
            f"Find 10 studies, papers, or documents on {focus} and synthesize what they jointly support",
            f"Analyze the PDF or document set on {focus} that I provide explicitly",
        ),
    )


def _looks_like_explanation_query(normalized_query: str) -> bool:
    return normalized_query.startswith(_EXPLANATION_QUERY_PREFIXES) or any(
        phrase in normalized_query for phrase in (" explain ", " teach me ", " understand ")
    )


def _looks_like_evidence_acquisition_query(normalized_query: str) -> bool:
    query_tokens = set(_tokenize(normalized_query))
    return bool(query_tokens & _EVIDENCE_ACQUISITION_TOKENS)


def _clarification_focus_text(query: str, *, retrieval_query: str) -> str:
    collapsed = " ".join(str(query or "").split()).strip().rstrip("?")
    lowered = collapsed.lower()
    for prefix in _EXPLANATION_QUERY_PREFIXES:
        if lowered.startswith(prefix):
            focus = collapsed[len(prefix) :].strip()
            return focus or (retrieval_query or collapsed)
    return retrieval_query or collapsed or "this topic"


def _render_query_clarification_html(payload: dict[str, object]) -> str:
    query = escape(str(payload.get("query") or "Democritus query clarification"))
    stage_label = escape(str(payload.get("stage_label") or "Query Clarification"))
    ambiguous_term = escape(str(payload.get("ambiguous_term") or "term"))
    summary_text = escape(str(payload.get("summary_text") or "This query needs clarification before retrieval can begin."))
    reason = escape(str(payload.get("reason") or "This query needs clarification before retrieval can begin."))
    suggestions = tuple(str(item).strip() for item in tuple(payload.get("suggested_queries") or ()) if str(item).strip())
    suggestion_markup = "".join(
        (
            '<article class="suggestion-card">'
            f"<p class=\"suggestion-label\">Suggested rewrite</p>"
            f"<pre>{escape(item)}</pre>"
            f"<button type=\"button\" onclick=\"copySuggestion(this)\" data-query=\"{escape(item)}\">Copy query</button>"
            "</article>"
        )
        for item in suggestions
    ) or '<div class="empty">No suggested rewrites were generated.</div>'
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Democritus Query Clarification</title>
    <style>
      :root {{
        --ink: #18222d;
        --muted: #59646f;
        --paper: #f6f1e8;
        --card: rgba(255,255,255,0.92);
        --line: #d7ccb8;
        --accent: #93451e;
        --accent-soft: #f6e5d7;
        --green: #1f6a56;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(147,69,30,0.12), transparent 24%),
          linear-gradient(180deg, #fbf7f0 0%, var(--paper) 100%);
      }}
      main {{ width: min(1080px, calc(100vw - 32px)); margin: 32px auto 48px; display: grid; gap: 18px; }}
      .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; box-shadow: 0 24px 60px rgba(30,25,18,0.08); }}
      .eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      h1, h2, p, pre {{ margin: 0; }}
      .trace {{ color: var(--muted); line-height: 1.6; }}
      .callout {{ background: var(--accent-soft); }}
      .suggestion-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }}
      .suggestion-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 18px; background: #fffdf9; display: grid; gap: 12px; }}
      .suggestion-label {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); }}
      pre {{
        white-space: pre-wrap;
        word-break: break-word;
        padding: 14px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: #fbf6ef;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
        line-height: 1.55;
      }}
      button {{
        border: 0;
        border-radius: 999px;
        padding: 10px 14px;
        background: var(--green);
        color: white;
        font-weight: 700;
        cursor: pointer;
      }}
      button:hover {{ filter: brightness(0.96); }}
      code {{
        background: rgba(0,0,0,0.04);
        padding: 2px 5px;
        border-radius: 6px;
      }}
      .empty {{ color: var(--muted); line-height: 1.6; }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">Democritus Clarification Checkpoint</p>
        <h1>{query}</h1>
        <p class="trace">Democritus paused at the <strong>{stage_label}</strong> before retrieval. {("The term <code>" + ambiguous_term + "</code> needs a domain-specific meaning so the corpus search does not drift into unrelated literature.") if ambiguous_term else summary_text}</p>
      </section>
      <section class="panel callout">
        <p class="eyebrow">Why CLIFF Paused</p>
        <p class="trace">{reason}</p>
        <p class="trace" style="margin-top:12px;">Copy one of the rewritten queries below, paste it back into the CLIFF session prompt, and rerun.</p>
      </section>
      <section class="panel">
        <p class="eyebrow">Suggested Queries</p>
        <div class="suggestion-grid" style="margin-top:12px;">{suggestion_markup}</div>
      </section>
    </main>
    <script>
      function copySuggestion(button) {{
        var query = button.getAttribute("data-query") || "";
        if (!query) {{
          return;
        }}
        navigator.clipboard.writeText(query).then(function () {{
          button.textContent = "Copied";
          window.setTimeout(function () {{
            button.textContent = "Copy query";
          }}, 1200);
        }}).catch(function () {{
          button.textContent = "Copy failed";
          window.setTimeout(function () {{
            button.textContent = "Copy query";
          }}, 1200);
        }});
      }}
    </script>
  </body>
</html>"""


def _build_query_clarification_checkpoint(
    *,
    query: str,
    outdir: Path,
    clarification_request: QueryClarificationRequest,
) -> tuple[Path, Path]:
    checkpoint_dir = outdir / "query_clarification"
    manifest_path = checkpoint_dir / "democritus_query_clarification.json"
    dashboard_path = checkpoint_dir / "democritus_query_clarification.html"
    payload = {
        "query": query,
        "stage_id": "query_clarification",
        "stage_label": "Query Clarification",
        "summary_text": clarification_request.summary_text,
        "ambiguous_term": clarification_request.ambiguous_term,
        "reason": clarification_request.reason,
        "suggested_queries": list(clarification_request.suggested_queries),
        "recommended_next_action": "resubmit_query",
    }
    _write_json(manifest_path, payload)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(_render_query_clarification_html(payload), encoding="utf-8")
    return manifest_path, dashboard_path


def _looks_like_pdf(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or "pdf" in lowered.split("?")[0]


def _infer_document_format(url: str | None) -> str:
    if not url:
        return "unknown"
    lowered = url.lower()
    if _looks_like_pdf(lowered):
        return "pdf"
    if lowered.endswith(".htm") or lowered.endswith(".html"):
        return "html"
    if lowered.endswith(".txt"):
        return "txt"
    return "unknown"


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.lower().split())
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized)
        if token not in _STOPWORDS and len(token) > 1
    )


def _ordered_query_tokens(plan: "QueryPlan") -> tuple[str, ...]:
    query = plan.retrieval_query or plan.normalized_query or plan.query
    return _tokenize(query)


def _split_query_tokens(plan: "QueryPlan") -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered_tokens = _ordered_query_tokens(plan)
    if not ordered_tokens:
        ordered_tokens = tuple(token for token in plan.keyword_tokens if token)
    anchors: list[str] = []
    generic: list[str] = []
    seen: set[str] = set()
    for token in ordered_tokens:
        if token in seen:
            continue
        seen.add(token)
        if token in _GENERIC_RETRIEVAL_TOKENS:
            generic.append(token)
        else:
            anchors.append(token)
    if anchors:
        return tuple(anchors), tuple(generic)
    return tuple(generic), ()


def _best_query_phrase_match(plan: "QueryPlan", primary_text: str, full_text: str) -> tuple[float, str | None]:
    ordered_anchors, _ = _split_query_tokens(plan)
    if len(ordered_anchors) < 2:
        return 0.0, None
    best_phrase: str | None = None
    best_score = 0.0
    max_window = min(3, len(ordered_anchors))
    for window in range(max_window, 1, -1):
        for index in range(len(ordered_anchors) - window + 1):
            phrase = " ".join(ordered_anchors[index : index + window])
            if phrase in primary_text:
                return 5.0 + float(window - 2), f"title_phrase:{phrase}"
            if best_phrase is None and phrase in full_text:
                best_phrase = phrase
                best_score = 2.0 + float(window - 2) * 0.5
    if best_phrase is None:
        return 0.0, None
    return best_score, f"text_phrase:{best_phrase}"


def _match_score(plan: "QueryPlan", *texts: str) -> tuple[float, tuple[str, ...]]:
    primary_text = " ".join(str(texts[0] or "").lower().split()) if texts else ""
    secondary_text = " ".join(" ".join(str(text or "") for text in texts[1:]).lower().split())
    haystack = " ".join(part for part in (primary_text, secondary_text) if part)
    primary_tokens = set(_tokenize(primary_text))
    secondary_tokens = set(_tokenize(secondary_text))
    anchor_tokens, generic_tokens = _split_query_tokens(plan)
    evidence: list[str] = []
    score = 0.0
    if plan.normalized_query and plan.normalized_query in haystack:
        score += 6.0
        evidence.append("exact_query")
    phrase_bonus, phrase_evidence = _best_query_phrase_match(plan, primary_text, haystack)
    score += phrase_bonus
    if phrase_evidence:
        evidence.append(phrase_evidence)
    title_anchor_matches = tuple(token for token in anchor_tokens if token in primary_tokens)
    text_anchor_matches = tuple(token for token in anchor_tokens if token in secondary_tokens)
    if anchor_tokens and not title_anchor_matches and not text_anchor_matches and "exact_query" not in evidence:
        return 0.0, ()
    title_generic_matches = tuple(token for token in generic_tokens if token in primary_tokens)
    text_generic_matches = tuple(token for token in generic_tokens if token in secondary_tokens)
    score += 3.0 * float(len(title_anchor_matches))
    score += 0.85 * float(len(text_anchor_matches))
    score += 0.35 * float(len(title_generic_matches))
    score += 0.10 * float(len(text_generic_matches))
    matched_anchor_count = len(set(title_anchor_matches) | set(text_anchor_matches))
    if len(anchor_tokens) >= 2 and len(title_anchor_matches) >= 2:
        score += 2.0
        evidence.append("title_anchor_pair")
    elif len(anchor_tokens) >= 2 and matched_anchor_count >= 2:
        score += 0.75
        evidence.append("anchor_pair")
    elif len(anchor_tokens) >= 2 and matched_anchor_count == 1 and not title_anchor_matches:
        score *= 0.5
    if anchor_tokens and not title_anchor_matches and matched_anchor_count < min(2, len(anchor_tokens)):
        score *= 0.65
    evidence.extend(f"title:{token}" for token in title_anchor_matches)
    evidence.extend(f"text:{token}" for token in text_anchor_matches if token not in title_anchor_matches)
    evidence.extend(title_anchor_matches)
    evidence.extend(token for token in text_anchor_matches if token not in title_anchor_matches)
    evidence.extend(f"generic_title:{token}" for token in title_generic_matches)
    evidence.extend(f"generic_text:{token}" for token in text_generic_matches if token not in title_generic_matches)
    deduped_evidence = tuple(dict.fromkeys(evidence))
    return max(score, 0.0), deduped_evidence


def _scholarly_citation_bonus(score: float, evidence: tuple[str, ...], citations: object, *, divisor: float) -> float:
    if score < 3.0:
        return 0.0
    if not any(
        item == "exact_query"
        or item == "title_anchor_pair"
        or item.startswith("title_phrase:")
        or item.startswith("title:")
        for item in evidence
    ):
        return 0.0
    try:
        citation_count = float(citations or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(citation_count / divisor, 1.5)


def _sec_company_match_score(plan: "QueryPlan", *texts: str) -> tuple[float, tuple[str, ...]]:
    haystack = " ".join(texts).lower()
    company_tokens = tuple(
        token
        for token in plan.keyword_tokens
        if not str(token).isdigit() and token not in _SEC_COMPANY_QUERY_STOPWORDS and len(token) >= 3
    )
    if not company_tokens:
        return _match_score(plan, *texts)
    matched = sorted({token for token in company_tokens if token in haystack})
    return float(len(matched)), tuple(matched)


def _company_text_tokens(*texts: str) -> set[str]:
    return set(_tokenize(" ".join(texts)))


def _alias_tokens(alias: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _tokenize(alias)
        if token not in _SEC_COMPANY_QUERY_STOPWORDS
    )


def _sec_company_target_match_score(plan: "QueryPlan", *texts: str) -> tuple[float, tuple[str, ...]]:
    if not plan.sec_company_targets:
        return 0.0, ()
    company_tokens = _company_text_tokens(*texts)
    best_score = 0.0
    best_evidence: tuple[str, ...] = ()
    for target in plan.sec_company_targets:
        matched_aliases: list[str] = []
        matched_token_count = 0
        for alias in target:
            alias_token_values = _alias_tokens(alias)
            if alias_token_values and set(alias_token_values).issubset(company_tokens):
                matched_aliases.append(alias)
                matched_token_count = max(matched_token_count, len(alias_token_values))
        if matched_aliases:
            score = 100.0 + float(matched_token_count)
            evidence = tuple(sorted(set(matched_aliases)))
            if score > best_score:
                best_score = score
                best_evidence = evidence
    return best_score, best_evidence


@dataclass(frozen=True)
class QueryPlan:
    """Interpreted query intent for corpus acquisition."""

    query: str
    normalized_query: str
    keyword_tokens: tuple[str, ...]
    target_documents: int
    base_query: str = ""
    requested_forms: tuple[str, ...] = ()
    retrieval_query: str = ""
    selected_topics: tuple[str, ...] = ()
    rejected_topics: tuple[str, ...] = ()
    retrieval_refinement: str = ""
    sec_company_targets: tuple[tuple[str, ...], ...] = ()
    sec_cohort_mode: str = "ranked"
    direct_document_urls: tuple[str, ...] = ()
    direct_document_paths: tuple[str, ...] = ()
    direct_document_directories: tuple[str, ...] = ()
    clarification_request: QueryClarificationRequest | None = None


def _derive_retrieval_query(query: str) -> str:
    normalized = " ".join(query.lower().split())
    had_url = normalized != _strip_urls(normalized)
    normalized = _strip_urls(normalized)
    stripped = normalized
    for pattern in (
        r"^i(?:'d|\swould)?\s+like\s+to\s+know\s+",
        r"^i\s+want\s+to\s+know\s+",
        r"^i(?:'m|\sam)\s+interested\s+in\s+",
        r"^tell\s+me\s+about\s+",
        r"^help\s+me\s+understand\s+",
        r"^what\s+do\s+we\s+know\s+about\s+",
        r"^what\s+are\s+the\s+",
        r"^what\s+is\s+the\s+",
        rf"^(?:analyze|summarize|review|find)\s+(?:\d+|{'|'.join(_NUMBER_WORDS)}){_COUNT_FILLER_PATTERN}\s+(?:study|studies|paper|papers|article|articles|document|documents)\s+(?:of|on|about)\s+",
        rf"^(?:analyze|summarize|review|find)\s+{_COUNT_FILLER_PATTERN}\s*(?:study|studies|paper|papers|article|articles|document|documents)\s+(?:of|on|about)\s+",
        r"^(?:analyze|summarize|review|find)\s+",
    ):
        stripped = re.sub(pattern, "", stripped)
    for pattern in (
        r"\s+and\s+synthesize\s+what\s+they\s+jointly\s+support\b.*$",
        r"\s+and\s+summari[sz]e\b.*$",
        r"\s+and\s+extract\b.*$",
    ):
        stripped = re.sub(pattern, "", stripped)
    retrieval_tokens = _tokenize(stripped)
    if retrieval_tokens:
        return " ".join(retrieval_tokens)
    fallback_tokens = _tokenize(normalized)
    if fallback_tokens:
        return " ".join(fallback_tokens)
    if had_url:
        return ""
    return normalized


def _compose_structured_democritus_query(
    *,
    base_query: str,
    selected_topics: tuple[str, ...],
    rejected_topics: tuple[str, ...],
    retrieval_refinement: str,
) -> str:
    parts = [" ".join(str(base_query or "").split()).strip()]
    if selected_topics:
        parts.append("focus on topics: " + "; ".join(selected_topics))
    if rejected_topics:
        parts.append("avoid topics: " + "; ".join(rejected_topics))
    refinement = " ".join(str(retrieval_refinement or "").split()).strip()
    if refinement:
        parts.append(refinement)
    return " ".join(part for part in parts if part).strip()


def _retrieval_source_query(
    *,
    base_query: str,
    selected_topics: tuple[str, ...],
    retrieval_refinement: str,
    fallback_query: str,
) -> str:
    parts = [" ".join(str(base_query or "").split()).strip()]
    if selected_topics:
        parts.append(" ".join(selected_topics))
    refinement = " ".join(str(retrieval_refinement or "").split()).strip()
    if refinement:
        parts.append(refinement)
    composed = " ".join(part for part in parts if part).strip()
    return composed or " ".join(str(fallback_query or "").split()).strip()


def _strip_urls(text: str) -> str:
    return " ".join(_URL_PATTERN.sub(" ", str(text)).split())


def _extract_direct_document_urls(query: str) -> tuple[str, ...]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_PATTERN.findall(str(query or "")):
        candidate = match.rstrip(".,;:!?)]}\"'")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return tuple(urls)


def _extract_direct_document_paths(query: str, *, explicit_path: Path | None = None) -> tuple[str, ...]:
    candidates = _direct_document_path_candidates(query, explicit_path=explicit_path)
    resolved_paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = _normalize_direct_document_path(candidate)
        if resolved is None:
            continue
        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        resolved_paths.append(resolved_str)
    return tuple(resolved_paths)


def _direct_document_path_candidates(query: str, *, explicit_path: Path | None = None) -> tuple[str, ...]:
    candidates: list[str] = []
    if explicit_path is not None:
        candidates.append(str(explicit_path))
    text = str(query or "")
    for pattern in (
        r"`([^`]+\.pdf)`",
        r'"([^"]+\.pdf)"',
        r"'([^']+\.pdf)'",
    ):
        candidates.extend(match.group(1) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    candidates.extend(_DIRECT_PDF_TOKEN_PATTERN.findall(text))
    return tuple(dict.fromkeys(candidate for candidate in candidates if str(candidate or "").strip()))


def _extract_direct_document_directories(
    query: str,
    *,
    explicit_dir: Path | None = None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if explicit_dir is not None:
        candidates.append(str(explicit_dir))
    text = str(query or "")
    for pattern in (
        r"`([^`]+)`",
        r'"([^"]+)"',
        r"'([^']+)'",
    ):
        candidates.extend(match.group(1) for match in re.finditer(pattern, text))
    candidates.extend(_DIRECT_LOCAL_TOKEN_PATTERN.findall(text))

    resolved_dirs: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = _normalize_direct_document_directory(candidate)
        if resolved is None:
            continue
        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        resolved_dirs.append(resolved_str)
    return tuple(resolved_dirs)


def _normalize_direct_document_path(candidate: str) -> Path | None:
    raw = str(candidate or "").strip().strip("\"'`")
    if not raw:
        return None
    if raw.startswith("file://"):
        raw = unquote(urlparse(raw).path)
    for prefix in _COMMON_ABSOLUTE_PATH_PREFIXES:
        if raw.startswith(prefix):
            raw = "/" + raw
            break
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.suffix.lower() != ".pdf":
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def _normalize_direct_document_directory(candidate: str) -> Path | None:
    raw = str(candidate or "").strip().strip("\"'`")
    if not raw:
        return None
    if raw.startswith("file://"):
        raw = unquote(urlparse(raw).path)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists() or not path.is_dir():
        return None
    return path


def _iter_pdf_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(path.resolve() for path in directory.rglob("*.pdf") if path.is_file()))


def _direct_directory_pdf_paths(directories: tuple[str, ...]) -> tuple[str, ...]:
    pdf_paths: list[str] = []
    seen: set[str] = set()
    for directory_str in directories:
        for path in _iter_pdf_files(Path(directory_str)):
            path_str = str(path)
            if path_str in seen:
                continue
            seen.add(path_str)
            pdf_paths.append(path_str)
    return tuple(pdf_paths)


def _has_direct_document_input(
    query: str,
    *,
    explicit_path: Path | None = None,
    explicit_dir: Path | None = None,
) -> bool:
    return bool(
        _extract_direct_document_paths(query, explicit_path=explicit_path)
        or _extract_direct_document_directories(query, explicit_dir=explicit_dir)
        or _extract_direct_document_urls(query)
    )


def _looks_like_html(path_or_url: str) -> bool:
    lowered = str(path_or_url or "").lower()
    return lowered.endswith(".html") or lowered.endswith(".htm") or "html" in lowered


def _payload_looks_like_html(payload: str) -> bool:
    snippet = payload[:1000].lower()
    return any(marker in snippet for marker in ("<!doctype html", "<html", "<body", "<article", "<main"))


def _extract_html_region(html_text: str) -> str:
    candidates = []
    for pattern in (
        r"(?is)<article\b[^>]*>(.*?)</article>",
        r"(?is)<main\b[^>]*>(.*?)</main>",
        r"(?is)<body\b[^>]*>(.*?)</body>",
    ):
        candidates.extend(re.findall(pattern, html_text))
    if not candidates:
        return html_text
    return max(candidates, key=len)


def _strip_html(html_text: str) -> str:
    text = _extract_html_region(html_text)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?is)<(header|footer|nav|aside|form)\b[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    filtered = []
    for line in lines:
        lowered = line.lower()
        if not line:
            continue
        if lowered in {"skip to content", "search", "menu"}:
            continue
        if lowered.startswith("home »") or lowered.startswith("home >"):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def _extract_document_text(payload: str, *, source_hint: str) -> str:
    if _looks_like_html(source_hint) or _payload_looks_like_html(payload):
        return _strip_html(payload)
    return " ".join(payload.split())


def _extract_html_title(payload: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", payload)
    if not match:
        return ""
    text = re.sub(r"(?s)<[^>]+>", " ", match.group(1))
    return " ".join(unescape(text).split())


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    tail = unquote(Path(parsed.path).name).strip()
    if tail:
        stem = Path(tail).stem
        return _pretty_url_title(stem)
    if parsed.netloc:
        return _pretty_url_title(parsed.netloc)
    return "document"


def _pretty_url_title(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", value or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "document"


def _extract_sec_company_targets(query: str) -> tuple[tuple[str, ...], ...]:
    normalized = " ".join(str(query).lower().split())
    for phrases, targets in _SEC_COMPANY_GROUPS:
        if any(phrase in normalized for phrase in phrases):
            return targets

    stripped = re.sub(r"\b(?:10-k|10-q|8-k)\b", " ", normalized)
    segments = re.split(r"\b(?:and|or)\b|,|&", stripped)
    targets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for segment in segments:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", segment)
            if token not in _STOPWORDS and token not in _SEC_COMPANY_QUERY_STOPWORDS and len(token) >= 2
        ]
        if not tokens:
            continue
        alias = " ".join(tokens)
        target = (alias,)
        if target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return tuple(targets)


def _infer_sec_cohort_mode(query: str, sec_company_targets: tuple[tuple[str, ...], ...]) -> str:
    normalized = " ".join(str(query).lower().split())
    if len(sec_company_targets) > 1 and any(
        phrase in normalized
        for phrase in (
            "djia",
            "dow jones industrial average",
            "dow companies",
            "dow 30",
            "companies",
        )
    ):
        return "latest_per_company"
    return "ranked"


def _document_identity(document: "DiscoveredDocument") -> str:
    return (
        document.identifier
        or document.download_url
        or document.url
        or document.source_path
        or document.title
    ).lower()


def infer_requested_result_count(query: str, *, nouns: tuple[str, ...]) -> int | None:
    normalized = " ".join(str(query).lower().split())
    if not normalized or not nouns:
        return None
    noun_pattern = "|".join(re.escape(noun.lower()) for noun in nouns)
    match = re.search(
        rf"\b(?P<count>\d+|{'|'.join(_NUMBER_WORDS)})\b{_COUNT_FILLER_PATTERN}\s+(?P<noun>{noun_pattern})\b",
        normalized,
    )
    if not match:
        return None
    raw_count = match.group("count")
    if raw_count.isdigit():
        return max(1, int(raw_count))
    return _NUMBER_WORDS.get(raw_count)


_BROWSER_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_DEFAULT_RETRIEVAL_USER_AGENT = "FunctorFlow_v2/0.1 (agentic retrieval; local use)"


def _looks_like_placeholder_user_agent(value: str) -> bool:
    normalized = " ".join(str(value).split()).strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    return (
        lowered.startswith("functorflow_v3/")
        or lowered.startswith("functorflow_v2/")
        or lowered.startswith("functorflow v3/")
        or lowered.startswith("functorflow v2/")
    )


def _resolve_sec_user_agent(user_agent: str) -> str:
    normalized = " ".join(str(user_agent).split()).strip()
    if normalized and not _looks_like_placeholder_user_agent(normalized):
        return normalized

    for env_name in ("FF3_SEC_USER_AGENT", "FF2_SEC_USER_AGENT", "SEC_USER_AGENT", "SEC_IDENTITY"):
        env_value = " ".join(os.environ.get(env_name, "").split()).strip()
        if env_value:
            return env_value

    contact_name = " ".join(os.environ.get("SEC_CONTACT_NAME", "").split()).strip()
    contact_email = " ".join(os.environ.get("SEC_CONTACT_EMAIL", "").split()).strip()
    if contact_name and contact_email:
        return f"{contact_name} {contact_email}"

    raise ValueError(
        "SEC retrieval requires an identifying User-Agent that includes contact information. "
        "Pass --retrieval-user-agent 'Your Name your_email@example.com' or set "
        "FF3_SEC_USER_AGENT, FF2_SEC_USER_AGENT, SEC_USER_AGENT, SEC_IDENTITY, or SEC_CONTACT_NAME plus "
        "SEC_CONTACT_EMAIL."
    )


@dataclass(frozen=True)
class DiscoveredDocument:
    """Metadata for one candidate document returned by a search provider."""

    title: str
    score: float
    retrieval_backend: str = "unknown"
    source_path: str | None = None
    download_url: str | None = None
    url: str | None = None
    abstract: str = ""
    year: str = ""
    identifier: str = ""
    document_format: str = "pdf"
    evidence: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquiredCorpusDocument:
    """A document copied or downloaded into the working Democritus corpus."""

    title: str
    acquired_pdf_path: str
    source_path: str
    score: float
    run_name_hint: str
    retrieval_backend: str


@dataclass(frozen=True)
class DemocritusEvidenceSnapshot:
    """Retrieval-side snapshot used for Democritus convergence checks."""

    keyword_support: tuple[str, ...]
    top_context_terms: tuple[str, ...]
    average_score: float
    evidence_count: int


class DemocritusConvergenceAdapter(EvidenceConvergenceAdapter[DemocritusEvidenceSnapshot]):
    """Convergence semantics for retrieval evidence in query-driven Democritus."""

    def similarity(
        self,
        previous: DemocritusEvidenceSnapshot,
        current: DemocritusEvidenceSnapshot,
        *,
        policy: EvidenceConvergencePolicy,
    ) -> float:
        del policy
        keyword_score = self._tuple_overlap(previous.keyword_support, current.keyword_support)
        term_score = self._tuple_overlap(previous.top_context_terms, current.top_context_terms)
        score_delta = abs(previous.average_score - current.average_score)
        score_score = max(0.0, 1.0 - min(score_delta / 2.0, 1.0))
        return round(0.45 * keyword_score + 0.45 * term_score + 0.10 * score_score, 3)

    def describe(self, snapshot: DemocritusEvidenceSnapshot) -> str:
        keywords = ", ".join(snapshot.keyword_support) or "no matched keywords"
        terms = ", ".join(snapshot.top_context_terms[:4]) or "no stable context terms yet"
        return (
            f"matched_keywords=[{keywords}], context_terms=[{terms}], "
            f"avg_score={snapshot.average_score:.3f}"
        )

    @staticmethod
    def _tuple_overlap(left_values: tuple[str, ...], right_values: tuple[str, ...]) -> float:
        if not left_values and not right_values:
            return 1.0
        left = set(left_values)
        right = set(right_values)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


@dataclass(frozen=True)
class DemocritusQueryAgenticConfig:
    """Configuration for query-driven Democritus corpus acquisition."""

    query: str
    outdir: Path
    execution_mode: str = "quick"
    target_documents: int = 10
    base_query: str = ""
    selected_topics: tuple[str, ...] = ()
    rejected_topics: tuple[str, ...] = ()
    retrieval_refinement: str = ""
    input_pdf_path: Path | None = None
    input_pdf_dir: Path | None = None
    manifest_path: Path | None = None
    source_pdf_root: Path | None = None
    retrieval_backend: str = "auto"
    retrieval_user_agent: str = "FunctorFlow_v2/0.1 (agentic retrieval; local use)"
    retrieval_timeout_seconds: float = 20.0
    corpus_name: str = "query_corpus"
    max_docs: int = 0
    retrieval_topic_component_cap: int = 2
    retrieval_diversity_score_floor_ratio: float = 0.45
    consensus_enabled: bool = True
    consensus_batch_size: int = 1
    consensus_similarity_threshold: float = 0.9
    consensus_required_stable_passes: int = 1
    max_workers: int = 8
    agent_concurrency_limits: tuple[tuple[str, int], ...] = ()
    include_phase2: bool = True
    auto_topics_from_pdf: bool = True
    root_topic_strategy: str = "summary_guided"
    depth_limit: int = 3
    max_total_topics: int = 100
    statements_per_question: int = 2
    statement_batch_size: int = 16
    statement_max_tokens: int = 192
    manifold_mode: str = "full"
    topk: int = 200
    radii: str = "1,2,3"
    maxnodes: str = "10,20,30,40,60"
    lambda_edge: float = 0.25
    topk_models: int = 5
    topk_claims: int = 30
    alpha: float = 1.0
    tier1: float = 0.60
    tier2: float = 0.30
    anchors: str = ""
    title: str = ""
    dedupe_focus: bool = False
    require_anchor_in_focus: bool = False
    focus_blacklist_regex: str = ""
    render_topk_pngs: bool = True
    assets_dir: str = "assets"
    png_dpi: int = 200
    write_deep_dive: bool = False
    deep_dive_max_bullets: int = 8
    intra_document_shards: int = 1
    discovery_only: bool = False
    dry_run: bool = False
    enable_corpus_synthesis: bool = True
    sec_form_types: tuple[str, ...] = ("10-K", "10-Q")
    sec_company_limit: int = 3

    def resolved(self) -> "DemocritusQueryAgenticConfig":
        normalized_mode = str(self.execution_mode).strip().lower()
        if normalized_mode == "deep":
            execution_mode = "deep"
        elif normalized_mode == "interactive":
            execution_mode = "interactive"
        else:
            execution_mode = "quick"
        target_documents = max(1, int(self.target_documents))
        if execution_mode == "quick":
            target_documents = min(target_documents, 3)
        max_docs = int(self.max_docs)
        if execution_mode == "quick":
            quick_max_docs = target_documents + 2
            max_docs = min(max_docs, quick_max_docs) if max_docs > 0 else quick_max_docs
        elif execution_mode == "interactive":
            interactive_max_docs = target_documents + 2
            max_docs = min(max_docs, interactive_max_docs) if max_docs > 0 else interactive_max_docs
        depth_limit = max(1, int(self.depth_limit))
        max_total_topics = max(10, int(self.max_total_topics))
        statements_per_question = max(1, int(self.statements_per_question))
        statement_batch_size = max(1, int(self.statement_batch_size))
        statement_max_tokens = max(48, int(self.statement_max_tokens))
        topk = max(1, int(self.topk))
        topk_models = max(1, int(self.topk_models))
        topk_claims = max(1, int(self.topk_claims))
        root_topic_strategy = str(self.root_topic_strategy)
        intra_document_shards = max(1, int(self.intra_document_shards))
        direct_document_input = _has_direct_document_input(
            self.query,
            explicit_path=self.input_pdf_path,
            explicit_dir=self.input_pdf_dir,
        )
        if execution_mode == "quick" and not direct_document_input:
            if root_topic_strategy in {"v0_openai", "summary_guided"}:
                root_topic_strategy = "heuristic"
            depth_limit = min(depth_limit, 2)
            max_total_topics = min(max_total_topics, 40)
            statements_per_question = 1
            statement_batch_size = max(statement_batch_size, 32)
            statement_max_tokens = min(statement_max_tokens, 72)
            intra_document_shards = max(2, intra_document_shards)
        include_phase2 = self.include_phase2
        if execution_mode in {"quick", "interactive"}:
            include_phase2 = False
        return DemocritusQueryAgenticConfig(
            query=self.query.strip(),
            outdir=self.outdir.resolve(),
            execution_mode=execution_mode,
            target_documents=target_documents,
            base_query=" ".join(str(self.base_query or "").split()).strip(),
            selected_topics=_normalized_topics(self.selected_topics),
            rejected_topics=tuple(
                topic
                for topic in _normalized_topics(self.rejected_topics)
                if topic not in set(_normalized_topics(self.selected_topics))
            ),
            retrieval_refinement=" ".join(str(self.retrieval_refinement or "").split()).strip(),
            input_pdf_path=self.input_pdf_path.resolve() if self.input_pdf_path else None,
            input_pdf_dir=self.input_pdf_dir.resolve() if self.input_pdf_dir else None,
            manifest_path=self.manifest_path.resolve() if self.manifest_path else None,
            source_pdf_root=self.source_pdf_root.resolve() if self.source_pdf_root else None,
            retrieval_backend=self.retrieval_backend,
            retrieval_user_agent=self.retrieval_user_agent,
            retrieval_timeout_seconds=self.retrieval_timeout_seconds,
            corpus_name=self.corpus_name,
            max_docs=max_docs,
            retrieval_topic_component_cap=max(1, int(self.retrieval_topic_component_cap)),
            retrieval_diversity_score_floor_ratio=min(1.0, max(0.0, float(self.retrieval_diversity_score_floor_ratio))),
            consensus_enabled=self.consensus_enabled,
            consensus_batch_size=self.consensus_batch_size,
            consensus_similarity_threshold=self.consensus_similarity_threshold,
            consensus_required_stable_passes=self.consensus_required_stable_passes,
            max_workers=self.max_workers,
            agent_concurrency_limits=tuple(self.agent_concurrency_limits),
            include_phase2=include_phase2,
            auto_topics_from_pdf=self.auto_topics_from_pdf,
            root_topic_strategy=root_topic_strategy,
            depth_limit=depth_limit,
            max_total_topics=max_total_topics,
            statements_per_question=statements_per_question,
            statement_batch_size=statement_batch_size,
            statement_max_tokens=statement_max_tokens,
            manifold_mode=self.manifold_mode,
            topk=topk,
            radii=self.radii,
            maxnodes=self.maxnodes,
            lambda_edge=float(self.lambda_edge),
            topk_models=topk_models,
            topk_claims=topk_claims,
            alpha=float(self.alpha),
            tier1=float(self.tier1),
            tier2=float(self.tier2),
            anchors=self.anchors,
            title=self.title,
            dedupe_focus=bool(self.dedupe_focus),
            require_anchor_in_focus=bool(self.require_anchor_in_focus),
            focus_blacklist_regex=self.focus_blacklist_regex,
            render_topk_pngs=bool(self.render_topk_pngs),
            assets_dir=self.assets_dir,
            png_dpi=max(72, int(self.png_dpi)),
            write_deep_dive=bool(self.write_deep_dive),
            deep_dive_max_bullets=max(1, int(self.deep_dive_max_bullets)),
            intra_document_shards=intra_document_shards,
            discovery_only=self.discovery_only,
            dry_run=self.dry_run,
            enable_corpus_synthesis=self.enable_corpus_synthesis,
            sec_form_types=tuple(self.sec_form_types),
            sec_company_limit=self.sec_company_limit,
        )


@dataclass(frozen=True)
class DemocritusQueryRunResult:
    """Result bundle for query-driven corpus acquisition plus optional batch execution."""

    query_plan: QueryPlan
    selected_documents: tuple[DiscoveredDocument, ...]
    acquired_documents: tuple[AcquiredCorpusDocument, ...]
    batch_records: tuple[DemocritusBatchRecord, ...]
    pdf_dir: Path
    batch_outdir: Path
    summary_path: Path
    analysis_iterations: int = 0
    consensus_reached: bool = False
    convergence_assessment: dict[str, object] | None = None
    csql_sqlite_path: Path | None = None
    csql_summary_path: Path | None = None
    corpus_synthesis_summary_path: Path | None = None
    corpus_synthesis_dashboard_path: Path | None = None
    checkpoint_manifest_path: Path | None = None
    checkpoint_dashboard_path: Path | None = None
    clarification_manifest_path: Path | None = None
    clarification_dashboard_path: Path | None = None


class RetrievalBackend(Protocol):
    """Retrieval backend interface."""

    backend_name: str

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        """Return ranked candidate documents for a query plan."""


class HttpJsonRetrievalBackend:
    """Base class for JSON-over-HTTP retrieval backends."""

    backend_name = "http_json"

    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def _fetch_json(self, url: str, params: dict[str, object] | None = None) -> object:
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(full_url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _fetch_text(
        self,
        url: str,
        params: dict[str, object] | None = None,
        *,
        accept: str = "text/plain",
    ) -> str:
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(full_url, headers={"User-Agent": self.user_agent, "Accept": accept})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8")


class EuropePMCOARetrievalBackend(HttpJsonRetrievalBackend):
    """Europe PMC open-access retrieval with PMC OA PDF resolution."""

    backend_name = "europe_pmc"

    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        super().__init__(user_agent=user_agent, timeout_seconds=timeout_seconds)
        self._oa_link_cache: dict[str, tuple[str | None, str]] = {}

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        query = f"({plan.retrieval_query or plan.query}) OPEN_ACCESS:y"
        payload = self._fetch_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {
                "query": query,
                "format": "json",
                "pageSize": max(limit, plan.target_documents),
                "resultType": "core",
            },
        )
        items = list((((payload or {}).get("resultList") or {}).get("result")) or [])
        candidates: list[DiscoveredDocument] = []
        for item in items:
            if str(item.get("isOpenAccess") or "").upper() != "Y":
                continue
            title = str(item.get("title") or "untitled_paper").strip()
            abstract = str(item.get("abstractText") or "")
            journal = str(item.get("journalTitle") or "")
            score, evidence = _match_score(plan, title, abstract, journal)
            score += _scholarly_citation_bonus(
                score,
                evidence,
                item.get("citedByCount"),
                divisor=250.0,
            )
            if score <= 0.0:
                continue
            pmcid = self._normalize_pmcid(str(item.get("pmcid") or ""))
            download_url: str | None = None
            document_format = "unknown"
            if pmcid:
                download_url, document_format = self._resolve_oa_pdf(pmcid)
            if document_format != "pdf":
                continue
            doi = str(item.get("doi") or "").strip()
            source = str(item.get("source") or "MED").strip()
            article_id = str(item.get("id") or "").strip()
            source_url = None
            if pmcid:
                source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
            elif doi:
                source_url = f"https://doi.org/{doi}"
            elif article_id:
                source_url = f"https://europepmc.org/article/{source}/{article_id}"
            candidates.append(
                DiscoveredDocument(
                    title=title,
                    score=score,
                    retrieval_backend=self.backend_name,
                    download_url=download_url,
                    url=source_url,
                    abstract=abstract,
                    year=str(item.get("pubYear") or ""),
                    identifier=pmcid or doi or article_id,
                    document_format=document_format,
                    evidence=evidence,
                    metadata={
                        "pmcid": pmcid,
                        "doi": doi,
                        "source": source,
                        "journal": journal,
                    },
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])

    def _resolve_oa_pdf(self, pmcid: str) -> tuple[str | None, str]:
        cached = self._oa_link_cache.get(pmcid)
        if cached is not None:
            return cached
        xml_text = self._fetch_text(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            {"id": pmcid},
            accept="application/xml,text/xml;q=0.9,*/*;q=0.8",
        )
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            resolved = (None, "unknown")
            self._oa_link_cache[pmcid] = resolved
            return resolved
        download_url: str | None = None
        document_format = "unknown"
        for record in root.findall(".//record"):
            for link in record.findall("./link"):
                fmt = str(link.attrib.get("format") or "").lower()
                href = str(link.attrib.get("href") or "").strip()
                if not href:
                    continue
                if fmt == "pdf":
                    download_url = self._normalize_oa_href(href)
                    document_format = "pdf"
                    break
            if download_url:
                break
        resolved = (download_url, document_format)
        self._oa_link_cache[pmcid] = resolved
        return resolved

    @staticmethod
    def _normalize_pmcid(value: str) -> str:
        raw = value.strip()
        if not raw:
            return ""
        return raw if raw.upper().startswith("PMC") else f"PMC{raw}"

    @staticmethod
    def _normalize_oa_href(href: str) -> str:
        if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            return "https://ftp.ncbi.nlm.nih.gov/" + href[len("ftp://ftp.ncbi.nlm.nih.gov/") :]
        if href.startswith("ftp://"):
            return "https://" + href[len("ftp://") :]
        return href


class ManifestRetrievalBackend:
    """Search local manifest metadata using token-overlap heuristics."""

    backend_name = "manifest"

    def __init__(self, manifest_path: Path, source_pdf_root: Path | None = None) -> None:
        self.manifest_path = manifest_path
        self.source_pdf_root = source_pdf_root
        self._records = _read_records(manifest_path)

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        candidates: list[DiscoveredDocument] = []
        for record in self._records:
            title = str(record.get("title") or record.get("name") or "").strip()
            abstract = str(record.get("abstract") or record.get("summary") or "").strip()
            keywords = str(record.get("keywords") or record.get("tags") or "").strip()
            score, evidence = _match_score(plan, title, abstract, keywords)
            if "study" in (title + " " + abstract + " " + keywords).lower():
                score += 0.5
            if score <= 0.0:
                continue
            source_path = self._resolve_source_path(record)
            download_url = str(record.get("download_url") or record.get("pdf_url") or "") or None
            url = str(record.get("url") or download_url or "") or None
            candidates.append(
                DiscoveredDocument(
                    title=title or (Path(source_path).stem if source_path else "untitled_document"),
                    score=score,
                    retrieval_backend=self.backend_name,
                    source_path=str(source_path) if source_path else None,
                    download_url=download_url,
                    url=url,
                    abstract=abstract,
                    year=str(record.get("year") or ""),
                    identifier=str(record.get("doi") or record.get("id") or ""),
                    document_format=str(record.get("document_format") or _infer_document_format(download_url or url)),
                    evidence=evidence,
                    metadata={
                        key: str(value)
                        for key, value in record.items()
                        if key
                        not in {
                            "title",
                            "name",
                            "abstract",
                            "summary",
                            "keywords",
                            "tags",
                            "url",
                            "download_url",
                            "pdf_url",
                            "document_format",
                        }
                        and value not in (None, "")
                    },
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])

    def _resolve_source_path(self, record: dict[str, object]) -> Path | None:
        for key in ("pdf_path", "path", "local_path", "file_path", "source_path"):
            raw = record.get(key)
            if not raw:
                continue
            candidate = Path(str(raw))
            if candidate.is_absolute():
                return candidate
            if self.source_pdf_root:
                return self.source_pdf_root / candidate
            return (self.manifest_path.parent / candidate).resolve()
        return None


class FilesystemRetrievalBackend:
    """Search PDF filenames directly when no manifest is available."""

    backend_name = "filesystem"

    def __init__(self, source_pdf_root: Path) -> None:
        self.source_pdf_root = source_pdf_root

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        candidates: list[DiscoveredDocument] = []
        for path in sorted(self.source_pdf_root.rglob("*.pdf")):
            title = path.stem.replace("_", " ").replace("-", " ")
            score, evidence = _match_score(plan, title)
            if score <= 0.0:
                continue
            candidates.append(
                DiscoveredDocument(
                    title=path.stem,
                    score=score,
                    retrieval_backend=self.backend_name,
                    source_path=str(path),
                    document_format="pdf",
                    evidence=evidence,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])


class DirectFileRetrievalBackend:
    """Treat local PDF paths as a fixed one-off Democritus corpus."""

    backend_name = "direct_file"

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        candidates: list[DiscoveredDocument] = []
        for path_str in plan.direct_document_paths[:limit]:
            path = Path(path_str)
            candidates.append(
                DiscoveredDocument(
                    title=path.stem.replace("_", " ").replace("-", " "),
                    score=1000.0,
                    retrieval_backend=self.backend_name,
                    source_path=str(path),
                    identifier=str(path),
                    document_format="pdf",
                    evidence=("direct_file",),
                )
            )
        return tuple(candidates)


class DirectDirectoryRetrievalBackend:
    """Treat local PDF directories as a fixed one-off Democritus corpus."""

    backend_name = "direct_directory"

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        candidates: list[DiscoveredDocument] = []
        for path_str in _direct_directory_pdf_paths(plan.direct_document_directories)[:limit]:
            path = Path(path_str)
            candidates.append(
                DiscoveredDocument(
                    title=path.stem.replace("_", " ").replace("-", " "),
                    score=1000.0,
                    retrieval_backend=self.backend_name,
                    source_path=str(path),
                    identifier=str(path),
                    document_format="pdf",
                    evidence=("direct_directory",),
                )
            )
        return tuple(candidates)


class DirectURLRetrievalBackend:
    """Treat URLs embedded in the query as a fixed one-off Democritus corpus."""

    backend_name = "direct_url"

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        candidates: list[DiscoveredDocument] = []
        for url in plan.direct_document_urls[:limit]:
            document_format = _infer_document_format(url)
            candidates.append(
                DiscoveredDocument(
                    title=_title_from_url(url),
                    score=1000.0,
                    retrieval_backend=self.backend_name,
                    download_url=url if document_format == "pdf" else None,
                    url=url,
                    identifier=url,
                    document_format=document_format,
                    evidence=("direct_url",),
                )
            )
        return tuple(candidates)


class CrossrefRetrievalBackend(HttpJsonRetrievalBackend):
    """Crossref metadata retrieval with optional PDF link extraction."""

    backend_name = "crossref"

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        retrieval_query = plan.retrieval_query or plan.normalized_query or plan.query
        payload = self._fetch_json(
            "https://api.crossref.org/works",
            {
                "query.bibliographic": retrieval_query,
                "rows": max(limit, plan.target_documents),
                "select": "DOI,title,abstract,published-print,published-online,URL,link,type,is-referenced-by-count",
            },
        )
        items = list(((payload or {}).get("message") or {}).get("items") or [])
        candidates: list[DiscoveredDocument] = []
        for item in items:
            title_values = item.get("title") or []
            title = title_values[0].strip() if title_values else "untitled_work"
            abstract = str(item.get("abstract") or "")
            score, evidence = _match_score(plan, title, abstract)
            score += _scholarly_citation_bonus(
                score,
                evidence,
                item.get("is-referenced-by-count"),
                divisor=500.0,
            )
            if score <= 0.0:
                continue
            links = item.get("link") or []
            download_url = None
            for link in links:
                link_url = str(link.get("URL") or "")
                content_type = str(link.get("content-type") or "")
                if "pdf" in content_type.lower() or _looks_like_pdf(link_url):
                    download_url = link_url
                    break
            year = self._extract_crossref_year(item)
            candidates.append(
                DiscoveredDocument(
                    title=title,
                    score=score,
                    retrieval_backend=self.backend_name,
                    download_url=download_url,
                    url=str(item.get("URL") or "") or None,
                    abstract=abstract,
                    year=year,
                    identifier=str(item.get("DOI") or ""),
                    document_format=_infer_document_format(download_url or str(item.get("URL") or "")),
                    evidence=evidence,
                    metadata={"type": str(item.get("type") or "")},
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])

    @staticmethod
    def _extract_crossref_year(item: dict[str, object]) -> str:
        for key in ("published-print", "published-online"):
            date_parts = (((item.get(key) or {}).get("date-parts") or [[]])[0] if isinstance(item.get(key), dict) else [])
            if date_parts:
                return str(date_parts[0])
        return ""


class SemanticScholarRetrievalBackend(HttpJsonRetrievalBackend):
    """Semantic Scholar paper search with open-access PDF extraction."""

    backend_name = "semantic_scholar"

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        retrieval_query = plan.retrieval_query or plan.normalized_query or plan.query
        payload = self._fetch_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {
                "query": retrieval_query,
                "limit": max(limit, plan.target_documents),
                "fields": "title,abstract,year,url,venue,citationCount,openAccessPdf",
            },
        )
        items = list((payload or {}).get("data") or [])
        candidates: list[DiscoveredDocument] = []
        for item in items:
            title = str(item.get("title") or "untitled_paper")
            abstract = str(item.get("abstract") or "")
            score, evidence = _match_score(plan, title, abstract, str(item.get("venue") or ""))
            score += _scholarly_citation_bonus(
                score,
                evidence,
                item.get("citationCount"),
                divisor=250.0,
            )
            if score <= 0.0:
                continue
            open_access_pdf = item.get("openAccessPdf") or {}
            download_url = str(open_access_pdf.get("url") or "") or None
            candidates.append(
                DiscoveredDocument(
                    title=title,
                    score=score,
                    retrieval_backend=self.backend_name,
                    download_url=download_url,
                    url=str(item.get("url") or "") or None,
                    abstract=abstract,
                    year=str(item.get("year") or ""),
                    identifier=str(item.get("paperId") or ""),
                    document_format=_infer_document_format(download_url or str(item.get("url") or "")),
                    evidence=evidence,
                    metadata={"venue": str(item.get("venue") or "")},
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])


class ScholarlyRetrievalBackend:
    """Aggregate scholarly retrieval across multiple paper backends."""

    backend_name = "scholarly"

    def __init__(self, backends: tuple[RetrievalBackend, ...]) -> None:
        self.backends = backends

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        merged: dict[str, DiscoveredDocument] = {}
        backend_limit = min(max(limit, plan.target_documents * 4, 20), 100)
        variant_queries = _scholarly_query_variants(plan)
        per_variant_limit = min(max(limit, plan.target_documents, 8), backend_limit)
        backend_failures: list[str] = []
        for backend in self.backends:
            backend_had_success = False
            backend_variant_failures: list[str] = []
            for variant_query in variant_queries:
                variant_plan = plan
                if variant_query != (plan.retrieval_query or plan.query):
                    variant_plan = replace(
                        plan,
                        retrieval_query=variant_query,
                        normalized_query=variant_query,
                        keyword_tokens=_tokenize(variant_query),
                    )
                try:
                    results = backend.search(variant_plan, limit=per_variant_limit)
                    backend_had_success = True
                except Exception as exc:
                    backend_variant_failures.append(f"{variant_query}: {exc}")
                    continue
                for item in results:
                    metadata = dict(item.metadata or {})
                    metadata.setdefault("retrieval_query_variant", variant_query)
                    annotated = replace(item, metadata=metadata)
                    key = (annotated.identifier or annotated.download_url or annotated.url or annotated.title).lower()
                    existing = merged.get(key)
                    if existing is None or self._rank_key(annotated) > self._rank_key(existing):
                        merged[key] = annotated
            if not backend_had_success and backend_variant_failures:
                backend_failures.append(
                    f"{getattr(backend, 'backend_name', type(backend).__name__)}: "
                    + "; ".join(backend_variant_failures[:3])
                )
        ranked = sorted(merged.values(), key=self._sort_key, reverse=True)
        if not ranked and backend_failures:
            raise RuntimeError("All scholarly backends failed: " + "; ".join(backend_failures))
        return tuple(ranked[:limit])

    @staticmethod
    def _rank_key(item: DiscoveredDocument) -> tuple[int, float]:
        return (
            1 if item.document_format == "pdf" else 0,
            1 if item.retrieval_backend == "europe_pmc" else 0,
            float(item.score),
        )

    @staticmethod
    def _sort_key(item: DiscoveredDocument) -> tuple[int, float]:
        return (
            1 if item.document_format == "pdf" else 0,
            1 if item.retrieval_backend == "europe_pmc" else 0,
            float(item.score),
        )


class SECFilingRetrievalBackend(HttpJsonRetrievalBackend):
    """SEC EDGAR retrieval backend for recent company filings."""

    backend_name = "sec"

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float,
        form_types: tuple[str, ...],
        company_limit: int,
    ) -> None:
        super().__init__(user_agent=_resolve_sec_user_agent(user_agent), timeout_seconds=timeout_seconds)
        self.form_types = tuple(form_types)
        self.company_limit = max(1, company_limit)

    def search(self, plan: QueryPlan, *, limit: int) -> tuple[DiscoveredDocument, ...]:
        requested_forms = plan.requested_forms or self.form_types
        companies = self._search_companies(plan)
        candidates: list[DiscoveredDocument] = []
        company_limit = max(self.company_limit, len(plan.sec_company_targets)) if plan.sec_company_targets else self.company_limit
        for company in companies[: company_limit]:
            submissions = self._fetch_json(company["submissions_url"])
            recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
            accession_numbers = list(recent.get("accessionNumber") or [])
            forms = list(recent.get("form") or [])
            filing_dates = list(recent.get("filingDate") or [])
            primary_documents = list(recent.get("primaryDocument") or [])
            for accession, form, filing_date, primary_document in zip(
                accession_numbers,
                forms,
                filing_dates,
                primary_documents,
            ):
                if requested_forms and str(form) not in requested_forms:
                    continue
                accession_no_dash = str(accession).replace("-", "")
                primary_document = str(primary_document)
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{company['cik_nopad']}/{accession_no_dash}/{primary_document}"
                )
                title = f"{company['title']} {form} {filing_date}"
                score, evidence = _match_score(plan, title, company["title"], company["ticker"])
                score += company["score"]
                candidates.append(
                    DiscoveredDocument(
                        title=title,
                        score=score,
                        retrieval_backend=self.backend_name,
                        download_url=filing_url,
                        url=filing_url,
                        year=str(filing_date).split("-", 1)[0],
                        identifier=str(accession),
                        document_format=_infer_document_format(filing_url),
                        evidence=evidence + (str(form),),
                        metadata={
                            "ticker": company["ticker"],
                            "company": company["title"],
                            "cik": company["cik"],
                            "form": str(form),
                            "filing_date": str(filing_date),
                        },
                    )
                )
                if len(candidates) >= limit * 2:
                    break
                if plan.sec_cohort_mode == "latest_per_company":
                    break
            if len(candidates) >= limit * 2:
                break
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])

    def _search_companies(self, plan: QueryPlan) -> list[dict[str, object]]:
        payload = self._fetch_json("https://www.sec.gov/files/company_tickers.json")
        companies = []
        values = payload.values() if isinstance(payload, dict) else []
        for item in values:
            title = str(item.get("title") or "")
            ticker = str(item.get("ticker") or "")
            if plan.sec_company_targets:
                score, evidence = _sec_company_target_match_score(plan, title, ticker)
                if score <= 0.0:
                    continue
            else:
                score, evidence = _sec_company_match_score(plan, title, ticker)
                if score <= 0.0:
                    continue
            cik_str = str(item.get("cik_str") or "")
            cik = cik_str.zfill(10)
            companies.append(
                {
                    "title": title,
                    "ticker": ticker,
                    "cik": cik,
                    "cik_nopad": str(int(cik_str)) if cik_str else "",
                    "score": score,
                    "evidence": evidence,
                    "submissions_url": f"https://data.sec.gov/submissions/CIK{cik}.json",
                }
            )
        companies.sort(key=lambda item: (-float(item["score"]), str(item["title"]).lower()))
        return companies


class DemocritusQueryAgenticRunner:
    """Query-first ingress layer that acquires a corpus before running Democritus."""

    def __init__(self, config: DemocritusQueryAgenticConfig) -> None:
        self.config = config.resolved()
        if not self.config.query:
            raise ValueError("A non-empty acquisition query is required.")
        self.logs_dir = self.config.outdir / "query_agent_logs"
        self.query_plan_path = self.config.outdir / "query_plan.json"
        self.discovered_path = self.config.outdir / "discovered_documents.json"
        self.selected_path = self.config.outdir / "selected_documents.json"
        self.corpus_manifest_path = self.config.outdir / "acquired_corpus_manifest.json"
        self.summary_path = self.config.outdir / "query_run_summary.json"
        self.pdf_dir = self.config.outdir / "acquired_pdfs"
        self.batch_outdir = self.config.outdir / "democritus_runs"

    def run(self) -> DemocritusQueryRunResult:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        plan = self._run_query_interpretation_agent()
        batch_runner: DemocritusBatchAgenticRunner | None = None
        batch_result: DemocritusBatchRunResult | None = None
        checkpoint_manifest_path: Path | None = None
        checkpoint_dashboard_path: Path | None = None
        clarification_manifest_path: Path | None = None
        clarification_dashboard_path: Path | None = None
        selected_documents: tuple[DiscoveredDocument, ...] = ()
        acquired: tuple[AcquiredCorpusDocument, ...] = ()
        analysis_iterations = 0
        consensus_reached = False
        convergence_assessment = None
        if plan.clarification_request is not None:
            clarification_manifest_path, clarification_dashboard_path = _build_query_clarification_checkpoint(
                query=plan.query,
                outdir=self.config.outdir,
                clarification_request=plan.clarification_request,
            )
        elif not self.config.discovery_only and not self.config.dry_run:
            batch_runner = self._build_batch_runner(streaming=True)
        if batch_runner is not None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_future = executor.submit(batch_runner.run_with_artifacts)
                try:
                    (
                        selected_documents,
                        acquired,
                        analysis_iterations,
                        consensus_reached,
                        convergence_assessment,
                    ) = self._run_corpus_materialization_agent(
                        plan,
                        on_document_acquired=lambda item: batch_runner.register_document(
                            Path(item.acquired_pdf_path)
                        ),
                    )
                finally:
                    batch_runner.close_document_stream()
                batch_result = batch_future.result()
        elif plan.clarification_request is None:
            if self.config.discovery_only:
                discovered = self._run_document_discovery_agent(plan)
                selected_documents = tuple(discovered[: plan.target_documents])
            else:
                (
                    selected_documents,
                    acquired,
                    analysis_iterations,
                    consensus_reached,
                    convergence_assessment,
                ) = self._run_corpus_materialization_agent(
                    plan,
                )
                batch_runner = self._build_batch_runner(streaming=False)
                batch_result = batch_runner.run_with_artifacts()
        batch_records: tuple[DemocritusBatchRecord, ...] = ()
        if batch_result is not None:
            batch_records = batch_result.records
        if self.config.execution_mode == "interactive" and batch_runner is not None:
            checkpoint_manifest_path, checkpoint_dashboard_path = _build_democritus_topic_checkpoint(
                query=plan.query,
                base_query=plan.base_query,
                selected_topics=plan.selected_topics,
                rejected_topics=plan.rejected_topics,
                retrieval_refinement=plan.retrieval_refinement,
                outdir=self.config.outdir,
                batch_runner=batch_runner,
                selected_documents=selected_documents,
            )
        result = DemocritusQueryRunResult(
            query_plan=plan,
            selected_documents=selected_documents,
            acquired_documents=acquired,
            batch_records=batch_records,
            pdf_dir=self.pdf_dir,
            batch_outdir=self.batch_outdir,
            summary_path=self.summary_path,
            analysis_iterations=analysis_iterations,
            consensus_reached=consensus_reached,
            convergence_assessment=convergence_assessment,
            csql_sqlite_path=batch_result.csql_bundle.sqlite_path if batch_result and batch_result.csql_bundle else None,
            csql_summary_path=batch_result.csql_bundle.summary_path if batch_result and batch_result.csql_bundle else None,
            corpus_synthesis_summary_path=(
                batch_result.corpus_synthesis.summary_path
                if batch_result and batch_result.corpus_synthesis
                else None
            ),
            corpus_synthesis_dashboard_path=(
                batch_result.corpus_synthesis.dashboard_path
                if batch_result and batch_result.corpus_synthesis
                else None
            ),
            checkpoint_manifest_path=checkpoint_manifest_path,
            checkpoint_dashboard_path=checkpoint_dashboard_path,
            clarification_manifest_path=clarification_manifest_path,
            clarification_dashboard_path=clarification_dashboard_path,
        )
        self._write_run_summary(result)
        return result

    def _write_run_summary(self, result: DemocritusQueryRunResult) -> None:
        _write_json(
            self.summary_path,
            {
                "query_plan": asdict(result.query_plan),
                "execution_mode": self.config.execution_mode,
                "selected_documents": [asdict(item) for item in result.selected_documents],
                "acquired_documents": [asdict(item) for item in result.acquired_documents],
                "pdf_dir": str(result.pdf_dir),
                "batch_outdir": str(result.batch_outdir),
                "batch_records": len(result.batch_records),
                "analysis_iterations": result.analysis_iterations,
                "consensus_reached": result.consensus_reached,
                "convergence_assessment": result.convergence_assessment,
                "discovery_only": self.config.discovery_only,
                "retrieval_backend": self._backend_name(),
                "csql_sqlite_path": str(result.csql_sqlite_path) if result.csql_sqlite_path else None,
                "csql_summary_path": str(result.csql_summary_path) if result.csql_summary_path else None,
                "corpus_synthesis_summary_path": (
                    str(result.corpus_synthesis_summary_path)
                    if result.corpus_synthesis_summary_path
                    else None
                ),
                "corpus_synthesis_dashboard_path": (
                    str(result.corpus_synthesis_dashboard_path)
                    if result.corpus_synthesis_dashboard_path
                    else None
                ),
                "checkpoint_manifest_path": (
                    str(result.checkpoint_manifest_path)
                    if result.checkpoint_manifest_path
                    else None
                ),
                "checkpoint_dashboard_path": (
                    str(result.checkpoint_dashboard_path)
                    if result.checkpoint_dashboard_path
                    else None
                ),
                "clarification_manifest_path": (
                    str(result.clarification_manifest_path)
                    if result.clarification_manifest_path
                    else None
                ),
                "clarification_dashboard_path": (
                    str(result.clarification_dashboard_path)
                    if result.clarification_dashboard_path
                    else None
                ),
            },
        )

    def _build_batch_runner(self, *, streaming: bool) -> DemocritusBatchAgenticRunner:
        return DemocritusBatchAgenticRunner(
            DemocritusBatchConfig(
                pdf_dir=self.pdf_dir,
                outdir=self.batch_outdir,
                request_query=self.config.query,
                max_docs=self.config.max_docs,
                max_workers=self.config.max_workers,
                agent_concurrency_limits=self.config.agent_concurrency_limits,
                include_phase2=self.config.include_phase2,
                auto_topics_from_pdf=self.config.auto_topics_from_pdf,
                root_topic_strategy=self.config.root_topic_strategy,
                depth_limit=self.config.depth_limit,
                max_total_topics=self.config.max_total_topics,
                statements_per_question=self.config.statements_per_question,
                statement_batch_size=self.config.statement_batch_size,
                statement_max_tokens=self.config.statement_max_tokens,
                manifold_mode=self.config.manifold_mode,
                topk=self.config.topk,
                radii=self.config.radii,
                maxnodes=self.config.maxnodes,
                lambda_edge=self.config.lambda_edge,
                topk_models=self.config.topk_models,
                topk_claims=self.config.topk_claims,
                alpha=self.config.alpha,
                tier1=self.config.tier1,
                tier2=self.config.tier2,
                anchors=self.config.anchors,
                title=self.config.title,
                dedupe_focus=self.config.dedupe_focus,
                require_anchor_in_focus=self.config.require_anchor_in_focus,
                focus_blacklist_regex=self.config.focus_blacklist_regex,
                render_topk_pngs=self.config.render_topk_pngs,
                assets_dir=self.config.assets_dir,
                png_dpi=self.config.png_dpi,
                write_deep_dive=self.config.write_deep_dive,
                deep_dive_max_bullets=self.config.deep_dive_max_bullets,
                intra_document_shards=self.config.intra_document_shards,
                enable_corpus_synthesis=(
                    False if self.config.execution_mode == "interactive" else self.config.enable_corpus_synthesis
                ),
                stop_after_frontier_index=(1 if self.config.execution_mode == "interactive" else None),
                discover_existing_documents=not streaming,
                allow_incremental_admission=streaming,
                dry_run=self.config.dry_run,
            )
        )

    def _log(self, agent_name: str, lines: list[str]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"{agent_name}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_query_interpretation_agent(self) -> QueryPlan:
        base_query = " ".join(str(self.config.base_query or self.config.query).split()).strip()
        selected_topics = _normalized_topics(self.config.selected_topics)
        rejected_topics = tuple(
            topic for topic in _normalized_topics(self.config.rejected_topics) if topic not in set(selected_topics)
        )
        retrieval_refinement = " ".join(str(self.config.retrieval_refinement or "").split()).strip()
        display_query = _compose_structured_democritus_query(
            base_query=base_query,
            selected_topics=selected_topics,
            rejected_topics=rejected_topics,
            retrieval_refinement=retrieval_refinement,
        ) or self.config.query
        retrieval_source = _retrieval_source_query(
            base_query=base_query,
            selected_topics=selected_topics,
            retrieval_refinement=retrieval_refinement,
            fallback_query=self.config.query,
        )
        raw_normalized = " ".join(display_query.lower().split())
        requested_forms = tuple(sorted(set(re.findall(r"\b(?:10-k|10-q|8-k)\b", raw_normalized))))
        retrieval_query = _derive_retrieval_query(retrieval_source)
        sec_company_targets = _extract_sec_company_targets(base_query)
        sec_cohort_mode = _infer_sec_cohort_mode(base_query, sec_company_targets)
        direct_document_urls = _extract_direct_document_urls(base_query)
        raw_direct_document_path_candidates = _direct_document_path_candidates(
            base_query,
            explicit_path=self.config.input_pdf_path,
        )
        direct_document_paths = _extract_direct_document_paths(
            base_query,
            explicit_path=self.config.input_pdf_path,
        )
        direct_document_directories = _extract_direct_document_directories(
            base_query,
            explicit_dir=self.config.input_pdf_dir,
        )
        if raw_direct_document_path_candidates and not direct_document_paths:
            joined_candidates = ", ".join(raw_direct_document_path_candidates[:3])
            raise ValueError(
                "Could not resolve the requested local PDF path. "
                f"Tried: {joined_candidates}. Use an absolute path like `/Users/.../file.pdf` "
                "or provide the uploaded file path explicitly."
            )
        clarification_request = _query_clarification_request(
            base_query,
            retrieval_query,
            has_direct_document_input=bool(
                direct_document_paths
                or direct_document_directories
                or direct_document_urls
            ),
        )
        explicit_target_documents = infer_requested_result_count(
            base_query,
            nouns=("filing", "filings", "report", "reports", "document", "documents"),
        )
        if direct_document_paths:
            target_documents = len(direct_document_paths)
        elif direct_document_directories:
            target_documents = len(_direct_directory_pdf_paths(direct_document_directories))
        elif direct_document_urls:
            target_documents = len(direct_document_urls)
        elif explicit_target_documents is not None:
            target_documents = explicit_target_documents
        elif sec_cohort_mode == "latest_per_company":
            target_documents = len(sec_company_targets)
        else:
            target_documents = max(1, self.config.target_documents)
        if (
            self.config.execution_mode == "quick"
            and sec_cohort_mode != "latest_per_company"
            and not (direct_document_paths or direct_document_directories or direct_document_urls)
        ):
            target_documents = min(target_documents, 3)
        plan = QueryPlan(
            query=display_query,
            base_query=base_query,
            retrieval_query=retrieval_query,
            normalized_query=retrieval_query,
            keyword_tokens=_tokenize(retrieval_query),
            target_documents=target_documents,
            selected_topics=selected_topics,
            rejected_topics=rejected_topics,
            retrieval_refinement=retrieval_refinement,
            requested_forms=tuple(form.upper() for form in requested_forms),
            sec_company_targets=sec_company_targets,
            sec_cohort_mode=sec_cohort_mode,
            direct_document_urls=direct_document_urls,
            direct_document_paths=direct_document_paths,
            direct_document_directories=direct_document_directories,
            clarification_request=clarification_request,
        )
        _write_json(self.query_plan_path, asdict(plan))
        self._log(
            "query_interpretation_agent",
            [
                f"[QUERY] {display_query}",
                f"[BASE_QUERY] {base_query}",
                f"[RETRIEVAL_QUERY] {plan.retrieval_query}",
                f"[NORMALIZED] {plan.normalized_query}",
                f"[KEYWORDS] {', '.join(plan.keyword_tokens) if plan.keyword_tokens else '(none)'}",
                f"[SELECTED_TOPICS] {' | '.join(plan.selected_topics) if plan.selected_topics else '(none)'}",
                f"[REJECTED_TOPICS] {' | '.join(plan.rejected_topics) if plan.rejected_topics else '(none)'}",
                f"[RETRIEVAL_REFINEMENT] {plan.retrieval_refinement or '(none)'}",
                f"[EXECUTION_MODE] {self.config.execution_mode}",
                f"[TARGET_DOCUMENTS] {plan.target_documents}",
                f"[REQUESTED_FORMS] {', '.join(plan.requested_forms) if plan.requested_forms else '(none)'}",
                f"[SEC_COMPANY_TARGETS] {' | '.join('/'.join(target) for target in plan.sec_company_targets) if plan.sec_company_targets else '(none)'}",
                f"[SEC_COHORT_MODE] {plan.sec_cohort_mode}",
                f"[DIRECT_DOCUMENT_URLS] {' | '.join(plan.direct_document_urls) if plan.direct_document_urls else '(none)'}",
                f"[DIRECT_DOCUMENT_PATHS] {' | '.join(plan.direct_document_paths) if plan.direct_document_paths else '(none)'}",
                f"[DIRECT_DOCUMENT_DIRECTORIES] {' | '.join(plan.direct_document_directories) if plan.direct_document_directories else '(none)'}",
                (
                    "[CLARIFICATION] "
                    + (
                        f"term={plan.clarification_request.ambiguous_term} "
                        f"reason={plan.clarification_request.reason}"
                        if plan.clarification_request is not None
                        else "(none)"
                    )
                ),
                f"[RETRIEVAL_BACKEND] {self._backend_name()}",
            ],
        )
        return plan

    def _backend_name(self) -> str:
        if self.config.retrieval_backend == "auto":
            if _extract_direct_document_paths(self.config.query, explicit_path=self.config.input_pdf_path):
                return "direct_file"
            if _extract_direct_document_directories(self.config.query, explicit_dir=self.config.input_pdf_dir):
                return "direct_directory"
            if _extract_direct_document_urls(self.config.query):
                return "direct_url"
            if self.config.manifest_path:
                return "manifest"
            if self.config.source_pdf_root:
                return "filesystem"
            return "scholarly"
        return self.config.retrieval_backend

    def _provider(self) -> RetrievalBackend:
        backend_name = self._backend_name()
        if backend_name == "manifest":
            if not self.config.manifest_path:
                raise ValueError("Manifest backend requires `manifest_path`.")
            return ManifestRetrievalBackend(
                manifest_path=self.config.manifest_path,
                source_pdf_root=self.config.source_pdf_root,
            )
        if backend_name == "filesystem":
            if not self.config.source_pdf_root:
                raise ValueError("Filesystem backend requires `source_pdf_root`.")
            return FilesystemRetrievalBackend(self.config.source_pdf_root)
        if backend_name == "direct_file":
            return DirectFileRetrievalBackend()
        if backend_name == "direct_directory":
            return DirectDirectoryRetrievalBackend()
        if backend_name == "direct_url":
            return DirectURLRetrievalBackend()
        if backend_name == "crossref":
            return CrossrefRetrievalBackend(
                user_agent=self.config.retrieval_user_agent,
                timeout_seconds=self.config.retrieval_timeout_seconds,
            )
        if backend_name == "semantic_scholar":
            return SemanticScholarRetrievalBackend(
                user_agent=self.config.retrieval_user_agent,
                timeout_seconds=self.config.retrieval_timeout_seconds,
            )
        if backend_name == "europe_pmc":
            return EuropePMCOARetrievalBackend(
                user_agent=self.config.retrieval_user_agent,
                timeout_seconds=self.config.retrieval_timeout_seconds,
            )
        if backend_name == "scholarly":
            return ScholarlyRetrievalBackend(
                (
                    EuropePMCOARetrievalBackend(
                        user_agent=self.config.retrieval_user_agent,
                        timeout_seconds=self.config.retrieval_timeout_seconds,
                    ),
                    SemanticScholarRetrievalBackend(
                        user_agent=self.config.retrieval_user_agent,
                        timeout_seconds=self.config.retrieval_timeout_seconds,
                    ),
                    CrossrefRetrievalBackend(
                        user_agent=self.config.retrieval_user_agent,
                        timeout_seconds=self.config.retrieval_timeout_seconds,
                    ),
                )
            )
        if backend_name == "sec":
            return SECFilingRetrievalBackend(
                user_agent=self.config.retrieval_user_agent,
                timeout_seconds=self.config.retrieval_timeout_seconds,
                form_types=self.config.sec_form_types,
                company_limit=self.config.sec_company_limit,
            )
        raise ValueError(f"Unknown retrieval backend: {backend_name}")

    def _discovery_limit(self, plan: QueryPlan) -> int:
        backend_name = self._backend_name()
        if backend_name in {"scholarly", "semantic_scholar", "crossref"}:
            return max(plan.target_documents * 12, 30)
        if backend_name == "sec":
            return max(plan.target_documents * 6, 20)
        return max(1, plan.target_documents * 3)

    def _discovery_batch_size(self, plan: QueryPlan) -> int:
        backend_name = self._backend_name()
        if backend_name in {"scholarly", "semantic_scholar", "crossref"}:
            return max(plan.target_documents * 2, 25)
        if backend_name == "sec":
            return max(plan.target_documents * 2, 10)
        return max(plan.target_documents, 10)

    def _search_documents(
        self,
        provider: RetrievalBackend,
        plan: QueryPlan,
        *,
        limit: int,
    ) -> tuple[DiscoveredDocument, ...]:
        try:
            return provider.search(plan, limit=limit)
        except Exception as exc:
            raise RuntimeError(
                f"Document discovery failed for backend {getattr(provider, 'backend_name', self._backend_name())!r} "
                f"with retrieval query {plan.retrieval_query!r}: {exc}"
            ) from exc

    def _run_document_discovery_agent(self, plan: QueryPlan) -> tuple[DiscoveredDocument, ...]:
        provider = self._provider()
        discovery_limit = self._discovery_limit(plan)
        discovered = self._search_documents(provider, plan, limit=discovery_limit)
        discovered = _rebalance_discovered_documents(
            plan,
            discovered,
            component_cap=self.config.retrieval_topic_component_cap,
            score_floor_ratio=self.config.retrieval_diversity_score_floor_ratio,
        )
        if not discovered:
            raise FileNotFoundError(
                f"No candidate documents matched query: {self.config.query} "
                f"(retrieval query={plan.retrieval_query!r}, backend={getattr(provider, 'backend_name', self._backend_name())!r})"
            )
        _write_json(self.discovered_path, [asdict(item) for item in discovered])
        self._log(
            "document_discovery_agent",
            [
                f"[QUERY] {plan.query}",
                f"[RETRIEVAL_QUERY] {plan.retrieval_query}",
                f"[BACKEND] {getattr(provider, 'backend_name', self._backend_name())}",
                f"[DISCOVERED] {len(discovered)}",
                f"[TARGET] {plan.target_documents}",
                *[
                    f"[RETRIEVAL_COMPONENT {component}] {count}"
                    for component, count in Counter(
                        str(item.metadata.get('dominant_local_topic') or '')
                        for item in discovered[: max(plan.target_documents, 8)]
                        if str(item.metadata.get('dominant_local_topic') or '').strip()
                    ).most_common(5)
                ],
                *[
                    f"[DOC {index:02d}] score={item.score:.3f} backend={item.retrieval_backend} "
                    f"format={item.document_format} local_topic={item.metadata.get('dominant_local_topic') or 'n/a'} "
                    f"title={item.title}"
                    for index, item in enumerate(discovered[: plan.target_documents], start=1)
                ],
            ],
        )
        return discovered

    def _run_corpus_materialization_agent(
        self,
        plan: QueryPlan,
        *,
        on_document_acquired: Callable[[AcquiredCorpusDocument], object] | None = None,
    ) -> tuple[
        tuple[DiscoveredDocument, ...],
        tuple[AcquiredCorpusDocument, ...],
        int,
        bool,
        dict[str, object] | None,
    ]:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        selected: list[DiscoveredDocument] = []
        acquired: list[AcquiredCorpusDocument] = []
        analysis_iterations = 0
        consensus_reached = False
        convergence_assessment: dict[str, object] | None = None
        provider = self._provider()
        log_lines = [
            f"[CORPUS] {self.config.corpus_name}",
            f"[TARGET_DIR] {self.pdf_dir}",
            "[MODE] materialize_pdf_documents",
        ]
        failures: list[str] = []
        max_documents = max(0, int(self.config.max_docs))
        batch_size = max(1, int(self.config.consensus_batch_size))
        convergence_tracker = self._build_convergence_tracker(plan, max_documents=max_documents)
        discovery_batch_size = self._discovery_batch_size(plan)
        discovered_documents: list[DiscoveredDocument] = []
        seen_document_keys: set[str] = set()
        candidate_index = 0
        requested_limit = 0
        discovery_round = 0
        discovery_exhausted = False

        while True:
            while candidate_index >= len(discovered_documents) and not discovery_exhausted:
                discovery_round += 1
                next_limit = requested_limit + discovery_batch_size if requested_limit else max(
                    plan.target_documents,
                    discovery_batch_size,
                )
                if max_documents > 0:
                    next_limit = min(next_limit, max_documents)
                if next_limit <= requested_limit:
                    discovery_exhausted = True
                    break
                batch = self._search_documents(provider, plan, limit=next_limit)
                previous_count = len(discovered_documents)
                requested_limit = next_limit
                for document in batch:
                    key = _document_identity(document)
                    if key in seen_document_keys:
                        continue
                    seen_document_keys.add(key)
                    discovered_documents.append(document)
                    if max_documents > 0 and len(discovered_documents) >= max_documents:
                        break
                if len(discovered_documents) > previous_count:
                    reordered_pending = _rebalance_discovered_documents(
                        plan,
                        tuple(discovered_documents[previous_count:]),
                        component_cap=self.config.retrieval_topic_component_cap,
                        score_floor_ratio=self.config.retrieval_diversity_score_floor_ratio,
                        prior_selected=tuple(selected),
                    )
                    discovered_documents[previous_count:] = list(reordered_pending)
                _write_json(self.discovered_path, [asdict(item) for item in discovered_documents])
                log_lines.append(
                    f"[DISCOVERY] round={discovery_round} backend={getattr(provider, 'backend_name', self._backend_name())} "
                    f"requested_limit={requested_limit} total_candidates={len(discovered_documents)} "
                    f"retrieval_query={plan.retrieval_query}"
                )
                component_preview = Counter(
                    str(item.metadata.get("dominant_local_topic") or "")
                    for item in discovered_documents[previous_count: min(len(discovered_documents), previous_count + 8)]
                    if str(item.metadata.get("dominant_local_topic") or "").strip()
                )
                if component_preview:
                    log_lines.append(
                        "[DISCOVERY_COMPONENTS] "
                        + ", ".join(f"{component}:{count}" for component, count in component_preview.most_common(4))
                    )
                if len(discovered_documents) == previous_count:
                    discovery_exhausted = True
                    log_lines.append("[DISCOVERY] no new candidates returned; treating retrieval as exhausted")
                elif len(batch) < requested_limit:
                    discovery_exhausted = True
                    log_lines.append(
                        f"[DISCOVERY] provider returned only {len(batch)} candidates for limit={requested_limit}; "
                        "treating retrieval as exhausted"
                    )
                elif max_documents > 0 and len(discovered_documents) >= max_documents:
                    discovery_exhausted = True
                    log_lines.append(
                        f"[DISCOVERY] reached configured max_docs={max_documents}; stopping further retrieval"
                    )

            if candidate_index >= len(discovered_documents):
                break

            document = discovered_documents[candidate_index]
            candidate_index += 1
            target_name = f"{len(acquired) + 1:04d}_{_slugify(document.title)}.pdf"
            target_path = self.pdf_dir / target_name
            if document.source_path:
                source_path = Path(document.source_path)
                if not source_path.exists():
                    failures.append(f"{document.title}: missing local source {source_path}")
                    log_lines.append(f"[SKIP] missing local source for {document.title}: {source_path}")
                    continue
                shutil.copy2(source_path, target_path)
                try:
                    self._validate_materialized_pdf(target_path)
                except Exception as exc:
                    if target_path.exists():
                        target_path.unlink()
                    failures.append(f"{document.title}: {exc}")
                    log_lines.append(
                        f"[SKIP] rank={candidate_index} copied file invalid for {document.title}: {exc}"
                    )
                    continue
                selected.append(document)
                materialized = AcquiredCorpusDocument(
                    title=document.title,
                    acquired_pdf_path=str(target_path),
                    source_path=str(source_path),
                    score=document.score,
                    run_name_hint=target_path.stem,
                    retrieval_backend=document.retrieval_backend,
                )
                acquired.append(materialized)
                self._persist_materialized_corpus(selected, acquired)
                if on_document_acquired is not None:
                    on_document_acquired(materialized)
                log_lines.append(f"[COPY] rank={candidate_index} {source_path} -> {target_path}")
                stop_now, assessment_ran, reason = self._assess_convergence_if_ready(
                    plan,
                    selected,
                    acquired_count=len(acquired),
                    search_exhausted=discovery_exhausted and candidate_index >= len(discovered_documents),
                    batch_size=batch_size,
                    tracker=convergence_tracker,
                )
                if assessment_ran:
                    analysis_iterations += 1
                    convergence_assessment = convergence_tracker.last_assessment().as_dict()
                if reason:
                    log_lines.append(f"[CONVERGENCE] {reason}")
                if stop_now:
                    consensus_reached = (
                        convergence_tracker.last_assessment().stop_trigger == "stability"
                    )
                    break
                continue
            if document.download_url and document.document_format == "pdf":
                try:
                    self._download_file(document.download_url, target_path, referer=document.url)
                    self._validate_materialized_pdf(target_path)
                except Exception as exc:
                    if target_path.exists():
                        target_path.unlink()
                    if document.url and document.url != document.download_url:
                        try:
                            source_reference = self._materialize_remote_document_as_pdf(
                                replace(document, download_url=None),
                                target_path=target_path,
                                source_url_override=document.url,
                            )
                        except Exception as fallback_exc:
                            if target_path.exists():
                                target_path.unlink()
                            failures.append(
                                f"{document.title}: {exc}; fallback via landing page failed: {fallback_exc}"
                            )
                            log_lines.append(
                                f"[SKIP] rank={candidate_index} download failed for {document.title}: {exc}; "
                                f"landing-page fallback failed: {fallback_exc}"
                            )
                            continue
                        selected.append(document)
                        materialized = AcquiredCorpusDocument(
                            title=document.title,
                            acquired_pdf_path=str(target_path),
                            source_path=source_reference,
                            score=document.score,
                            run_name_hint=target_path.stem,
                            retrieval_backend=document.retrieval_backend,
                        )
                        acquired.append(materialized)
                        self._persist_materialized_corpus(selected, acquired)
                        if on_document_acquired is not None:
                            on_document_acquired(materialized)
                        log_lines.append(
                            f"[MATERIALIZE_FALLBACK] rank={candidate_index} {document.url} -> {target_path} "
                            f"after direct PDF download failed: {exc}"
                        )
                        stop_now, assessment_ran, reason = self._assess_convergence_if_ready(
                            plan,
                            selected,
                            acquired_count=len(acquired),
                            search_exhausted=discovery_exhausted and candidate_index >= len(discovered_documents),
                            batch_size=batch_size,
                            tracker=convergence_tracker,
                        )
                        if assessment_ran:
                            analysis_iterations += 1
                            convergence_assessment = convergence_tracker.last_assessment().as_dict()
                        if reason:
                            log_lines.append(f"[CONVERGENCE] {reason}")
                        if stop_now:
                            consensus_reached = (
                                convergence_tracker.last_assessment().stop_trigger == "stability"
                            )
                            break
                        continue
                    failures.append(f"{document.title}: {exc}")
                    log_lines.append(
                        f"[SKIP] rank={candidate_index} download failed for {document.title}: {exc}"
                    )
                    continue
                selected.append(document)
                materialized = AcquiredCorpusDocument(
                    title=document.title,
                    acquired_pdf_path=str(target_path),
                    source_path=document.download_url,
                    score=document.score,
                    run_name_hint=target_path.stem,
                    retrieval_backend=document.retrieval_backend,
                )
                acquired.append(materialized)
                self._persist_materialized_corpus(selected, acquired)
                if on_document_acquired is not None:
                    on_document_acquired(materialized)
                log_lines.append(f"[DOWNLOAD] rank={candidate_index} {document.download_url} -> {target_path}")
                stop_now, assessment_ran, reason = self._assess_convergence_if_ready(
                    plan,
                    selected,
                    acquired_count=len(acquired),
                    search_exhausted=discovery_exhausted and candidate_index >= len(discovered_documents),
                    batch_size=batch_size,
                    tracker=convergence_tracker,
                )
                if assessment_ran:
                    analysis_iterations += 1
                    convergence_assessment = convergence_tracker.last_assessment().as_dict()
                if reason:
                    log_lines.append(f"[CONVERGENCE] {reason}")
                if stop_now:
                    consensus_reached = (
                        convergence_tracker.last_assessment().stop_trigger == "stability"
                    )
                    break
                continue
            if document.url or document.download_url:
                source_reference = document.url or document.download_url or ""
                try:
                    source_reference = self._materialize_remote_document_as_pdf(
                        document,
                        target_path=target_path,
                    )
                except Exception as exc:
                    if target_path.exists():
                        target_path.unlink()
                    failures.append(f"{document.title}: {exc}")
                    log_lines.append(
                        f"[SKIP] rank={candidate_index} remote document materialization failed for {document.title}: {exc}"
                    )
                    continue
                selected.append(document)
                materialized = AcquiredCorpusDocument(
                    title=document.title,
                    acquired_pdf_path=str(target_path),
                    source_path=source_reference,
                    score=document.score,
                    run_name_hint=target_path.stem,
                    retrieval_backend=document.retrieval_backend,
                )
                acquired.append(materialized)
                self._persist_materialized_corpus(selected, acquired)
                if on_document_acquired is not None:
                    on_document_acquired(materialized)
                log_lines.append(f"[MATERIALIZE] rank={candidate_index} {source_reference} -> {target_path}")
                stop_now, assessment_ran, reason = self._assess_convergence_if_ready(
                    plan,
                    selected,
                    acquired_count=len(acquired),
                    search_exhausted=discovery_exhausted and candidate_index >= len(discovered_documents),
                    batch_size=batch_size,
                    tracker=convergence_tracker,
                )
                if assessment_ran:
                    analysis_iterations += 1
                    convergence_assessment = convergence_tracker.last_assessment().as_dict()
                if reason:
                    log_lines.append(f"[CONVERGENCE] {reason}")
                if stop_now:
                    consensus_reached = (
                        convergence_tracker.last_assessment().stop_trigger == "stability"
                    )
                    break
                continue
            failures.append(
                f"{document.title}: unsupported document format {document.document_format!r} "
                f"from backend {document.retrieval_backend!r}"
            )
            log_lines.append(
                f"[SKIP] rank={candidate_index} unsupported format={document.document_format} "
                f"backend={document.retrieval_backend} title={document.title}"
            )
        if not acquired:
            self._log("corpus_materialization_agent", log_lines + ["[ERROR] no materializable PDF documents found"])
            failure_preview = "; ".join(failures[:3]) if failures else "no candidates were materializable"
            raise RuntimeError(
                "Failed to materialize any PDF documents for Democritus. "
                f"First issues: {failure_preview}"
            )
        if not consensus_reached and not discovery_exhausted and max_documents == 0:
            log_lines.append("[WARN] retrieval loop exited before declaring convergence or exhaustion")
        if len(acquired) < self.config.target_documents:
            log_lines.append(
                f"[WARN] acquired {len(acquired)} documents out of requested {self.config.target_documents}"
            )
        self._persist_materialized_corpus(selected, acquired)
        self._log("corpus_materialization_agent", log_lines)
        return tuple(selected), tuple(acquired), analysis_iterations, consensus_reached, convergence_assessment

    def _assess_convergence_if_ready(
        self,
        plan: QueryPlan,
        selected: list[DiscoveredDocument],
        *,
        acquired_count: int,
        search_exhausted: bool,
        batch_size: int,
        tracker: EvidenceConvergenceTracker[DemocritusEvidenceSnapshot],
    ) -> tuple[bool, bool, str]:
        if not self.config.consensus_enabled:
            return acquired_count >= self.config.target_documents, False, ""
        if acquired_count < self.config.target_documents:
            return False, False, ""
        is_batch_boundary = (acquired_count - self.config.target_documents) % batch_size == 0
        is_last_candidate = search_exhausted
        if not is_batch_boundary and not is_last_candidate:
            return False, False, ""
        snapshot = self._democritus_convergence_snapshot(plan, tuple(selected))
        assessment = tracker.assess(snapshot, evidence_count=acquired_count)
        return assessment.stop, True, assessment.reason

    def _build_convergence_tracker(
        self,
        plan: QueryPlan,
        *,
        max_documents: int,
    ) -> EvidenceConvergenceTracker[DemocritusEvidenceSnapshot]:
        del plan
        return EvidenceConvergenceTracker(
            policy=EvidenceConvergencePolicy(
                min_evidence=max(1, self.config.target_documents),
                stability_threshold=float(self.config.consensus_similarity_threshold),
                required_stable_passes=max(1, self.config.consensus_required_stable_passes),
                max_evidence=max(0, max_documents),
            ),
            adapter=DemocritusConvergenceAdapter(),
        )

    def _max_documents_to_consider(self, discovered_count: int) -> int:
        if self.config.max_docs > 0:
            return min(self.config.max_docs, discovered_count)
        return discovered_count

    def _democritus_convergence_snapshot(
        self,
        plan: QueryPlan,
        selected_documents: tuple[DiscoveredDocument, ...],
    ) -> DemocritusEvidenceSnapshot:
        corpus_tokens: Counter[str] = Counter()
        matched_keywords = set()
        total_score = 0.0
        for document in selected_documents:
            total_score += float(document.score)
            text = " ".join(
                value
                for value in (
                    document.title,
                    document.abstract,
                    " ".join(document.evidence),
                    document.year,
                )
                if value
            ).lower()
            tokens = _tokenize(text)
            for token in tokens:
                if token in plan.keyword_tokens:
                    matched_keywords.add(token)
                    continue
                corpus_tokens[token] += 1
        top_terms = tuple(
            token
            for token, _ in corpus_tokens.most_common(6)
            if token not in plan.keyword_tokens
        )
        average_score = total_score / len(selected_documents) if selected_documents else 0.0
        return DemocritusEvidenceSnapshot(
            keyword_support=tuple(sorted(matched_keywords)),
            top_context_terms=top_terms,
            average_score=round(average_score, 3),
            evidence_count=len(selected_documents),
        )

    def _persist_materialized_corpus(
        self,
        selected: list[DiscoveredDocument],
        acquired: list[AcquiredCorpusDocument],
    ) -> None:
        _write_json(self.selected_path, [asdict(item) for item in selected])
        _write_json(self.corpus_manifest_path, [asdict(item) for item in acquired])

    def _download_file(self, url: str, target_path: Path, *, referer: str | None = None) -> None:
        candidate_urls = [url]
        if url.startswith("http://"):
            candidate_urls.insert(0, "https://" + url[len("http://") :])
        last_error: Exception | None = None
        for candidate_url in dict.fromkeys(candidate_urls):
            headers = self._download_request_headers(referer=referer)
            for timeout_seconds in self._request_timeouts_for_url(candidate_url, kind="download"):
                request = Request(candidate_url, headers=headers)
                try:
                    with urlopen(request, timeout=timeout_seconds) as response:
                        content = response.read()
                except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError) as exc:
                    last_error = exc
                    continue
                target_path.write_bytes(content)
                try:
                    self._validate_pdf_file(target_path)
                except Exception as exc:
                    if target_path.exists():
                        target_path.unlink()
                    last_error = exc
                    continue
                return
        if last_error is None:
            raise RuntimeError(f"Could not download {url}")
        raise last_error

    def _validate_materialized_pdf(self, path: Path) -> None:
        self._validate_pdf_file(path)
        if not self.config.auto_topics_from_pdf:
            return
        text = self._extract_pdf_text_for_validation(path)
        if text is None:
            return
        if not text.strip():
            raise RuntimeError(
                f"PDF {path.name!r} does not contain extractable text for Democritus topic discovery"
            )

    @staticmethod
    def _extract_pdf_text_for_validation(path: Path, *, max_chars: int = 12000) -> str | None:
        text = ""
        backend_available = False
        try:
            import fitz  # PyMuPDF
            backend_available = True

            doc = fitz.open(str(path))
            try:
                for index in range(min(5, doc.page_count)):
                    text += doc.load_page(index).get_text("text") + "\n"
                    if len(text) >= max_chars:
                        break
            finally:
                doc.close()
        except Exception:
            try:
                from pypdf import PdfReader
                backend_available = True

                reader = PdfReader(str(path))
                for index in range(min(5, len(reader.pages))):
                    text += (reader.pages[index].extract_text() or "") + "\n"
                    if len(text) >= max_chars:
                        break
            except Exception as exc:
                if not backend_available:
                    return None
                raise RuntimeError(f"Could not extract text from PDF {path!r}") from exc
        return text[:max_chars]

    def _download_request_headers(self, *, referer: str | None = None) -> dict[str, str]:
        user_agent = self.config.retrieval_user_agent.strip()
        if not user_agent or user_agent.startswith("FunctorFlow_v2/"):
            user_agent = _BROWSER_DOWNLOAD_USER_AGENT
        headers = {
            "User-Agent": user_agent,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _browser_like_request_headers(self, *, referer: str | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": _BROWSER_DOWNLOAD_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
            "Upgrade-Insecure-Requests": "1",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _materialize_remote_document_as_pdf(
        self,
        document: DiscoveredDocument,
        *,
        target_path: Path,
        source_url_override: str | None = None,
    ) -> str:
        source_url = source_url_override or document.download_url or document.url
        if not source_url:
            raise ValueError(f"Document {document.title!r} does not define a URL to materialize.")
        payload, content_type, source_reference = self._fetch_url_payload(
            source_url,
            referer=document.url if (source_url == document.download_url and document.download_url and document.url) else None,
        )
        if self._payload_is_pdf(payload, content_type=content_type, source_reference=source_reference):
            target_path.write_bytes(payload)
            self._validate_materialized_pdf(target_path)
            return source_reference
        decoded_payload = self._decode_remote_payload(payload, content_type=content_type)
        extracted_text = _extract_document_text(decoded_payload, source_hint=source_reference)
        if not extracted_text.strip():
            raise RuntimeError(f"Document {source_reference!r} did not yield extractable article text")
        document_title = _extract_html_title(decoded_payload) or document.title or _title_from_url(source_reference)
        self._write_extractable_text_pdf(
            target_path,
            title=document_title,
            text=extracted_text,
            source_reference=source_reference,
        )
        self._validate_materialized_pdf(target_path)
        return source_reference

    def _fetch_url_payload(
        self,
        url: str,
        *,
        referer: str | None = None,
    ) -> tuple[bytes, str, str]:
        request_specs = self._direct_document_request_specs(url, referer=referer)
        last_error: Exception | None = None
        for headers in request_specs:
            for timeout_seconds in self._request_timeouts_for_url(url, kind="direct_url"):
                request = Request(url, headers=headers)
                try:
                    with urlopen(request, timeout=timeout_seconds) as response:
                        content_type = response.headers.get("Content-Type", "")
                        source_reference = str(response.geturl() or url)
                        payload = self._read_remote_payload(
                            response,
                            content_type=content_type,
                            source_reference=source_reference,
                        )
                        return payload, content_type, source_reference
                except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError) as exc:
                    last_error = exc
                    continue
        if last_error is None:
            raise RuntimeError(f"Unable to fetch {url}")
        raise last_error

    def _direct_document_request_specs(self, url: str, *, referer: str | None = None) -> tuple[dict[str, str], ...]:
        browser_headers = self._browser_like_request_headers(referer=referer or url)
        hostname = urlparse(str(url or "")).netloc.lower()
        specs: list[dict[str, str]] = [self._download_request_headers(referer=referer)]
        if self._is_heavy_news_host(hostname):
            ranged_headers = dict(browser_headers)
            ranged_headers["Range"] = f"bytes=0-{self._non_pdf_remote_byte_limit(url) - 1}"
            specs.append(ranged_headers)
        specs.append(browser_headers)
        return tuple(specs)

    def _request_timeouts(self, *, kind: str) -> tuple[float, ...]:
        base_timeout = max(5.0, float(self.config.retrieval_timeout_seconds))
        if kind == "direct_url":
            candidates = (
                base_timeout,
                max(base_timeout * 2.0, 45.0),
                max(base_timeout * 4.0, 90.0),
            )
        elif kind == "download":
            candidates = (
                base_timeout,
                max(base_timeout * 2.0, 60.0),
            )
        else:
            candidates = (base_timeout,)
        ordered: list[float] = []
        for value in candidates:
            if value not in ordered:
                ordered.append(value)
        return tuple(ordered)

    def _request_timeouts_for_url(self, url: str, *, kind: str) -> tuple[float, ...]:
        if kind == "direct_url":
            hostname = urlparse(str(url or "")).netloc.lower()
            if self._is_heavy_news_host(hostname):
                for host_suffix, values in _HEAVY_NEWS_HOST_TIMEOUTS.items():
                    if hostname == host_suffix or hostname.endswith(f".{host_suffix}"):
                        return values
        return self._request_timeouts(kind=kind)

    def _read_remote_payload(
        self,
        response,
        *,
        content_type: str,
        source_reference: str,
    ) -> bytes:
        if self._should_read_remote_payload_fully(content_type=content_type, source_reference=source_reference):
            return response.read()
        return self._read_non_pdf_payload_chunked(
            response,
            source_reference=source_reference,
        )

    def _read_non_pdf_payload_chunked(self, response, *, source_reference: str) -> bytes:
        max_bytes = self._non_pdf_remote_byte_limit(source_reference)
        payload = bytearray()
        while len(payload) < max_bytes:
            remaining = max_bytes - len(payload)
            chunk = response.read(min(_NON_PDF_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            if self._has_enough_non_pdf_payload(bytes(payload), source_reference=source_reference):
                break
        return bytes(payload)

    @staticmethod
    def _non_pdf_remote_byte_limit(source_reference: str) -> int:
        hostname = urlparse(str(source_reference or "")).netloc.lower()
        for host_suffix, limit in _HEAVY_NEWS_HOST_LIMITS.items():
            if hostname == host_suffix or hostname.endswith(f".{host_suffix}"):
                return limit
        return _MAX_NON_PDF_REMOTE_BYTES

    @staticmethod
    def _is_heavy_news_host(hostname: str) -> bool:
        normalized = str(hostname or "").lower()
        return any(
            normalized == host_suffix or normalized.endswith(f".{host_suffix}")
            for host_suffix in _HEAVY_NEWS_HOST_LIMITS
        )

    @staticmethod
    def _has_enough_non_pdf_payload(payload: bytes, *, source_reference: str) -> bool:
        lowered = payload[: 256 * 1024].lower()
        if b"</article>" in lowered:
            return True
        if b"<article" in lowered and lowered.count(b"<p") >= 6:
            return True
        if b"</main>" in lowered and lowered.count(b"<p") >= 8:
            return True
        if b"application/ld+json" in lowered and b"articlebody" in lowered:
            return True
        hostname = urlparse(str(source_reference or "")).netloc.lower()
        if "washingtonpost.com" in hostname and (lowered.count(b"<p") >= 6 or b"articlebody" in lowered):
            return True
        return False

    @staticmethod
    def _should_read_remote_payload_fully(*, content_type: str, source_reference: str) -> bool:
        lowered_content_type = str(content_type or "").lower()
        if "application/pdf" in lowered_content_type:
            return True
        if "application/octet-stream" in lowered_content_type and _looks_like_pdf(source_reference):
            return True
        return False

    @staticmethod
    def _payload_is_pdf(payload: bytes, *, content_type: str, source_reference: str) -> bool:
        if payload.lstrip().startswith(b"%PDF-"):
            return True
        lowered_content_type = str(content_type or "").lower()
        if "application/pdf" in lowered_content_type:
            return True
        return _looks_like_pdf(source_reference)

    @staticmethod
    def _decode_remote_payload(payload: bytes, *, content_type: str) -> str:
        charset_match = re.search(r"charset=([^\s;]+)", str(content_type or ""), re.IGNORECASE)
        charset = charset_match.group(1).strip("\"'") if charset_match else "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _write_extractable_text_pdf(
        target_path: Path,
        *,
        title: str,
        text: str,
        source_reference: str,
    ) -> None:
        try:
            import fitz  # PyMuPDF
        except ModuleNotFoundError:
            try:
                DemocritusQueryAgenticRunner._write_extractable_text_pdf_with_matplotlib(
                    target_path,
                    title=title,
                    text=text,
                    source_reference=source_reference,
                )
            except ModuleNotFoundError:
                DemocritusQueryAgenticRunner._write_extractable_text_pdf_basic(
                    target_path,
                    title=title,
                    text=text,
                    source_reference=source_reference,
                )
            return

        page_width, page_height = fitz.paper_size("letter")
        left_margin = 54
        right_margin = 54
        top_margin = 54
        bottom_margin = 54
        title_font_size = 16
        source_font_size = 8.5
        body_font_size = 10.5
        body_line_height = body_font_size * 1.45
        title_text, source_line, wrapped_body_lines = DemocritusQueryAgenticRunner._prepare_text_pdf_layout(
            title=title,
            text=text,
            source_reference=source_reference,
        )

        available_height = page_height - top_margin - bottom_margin - (title_font_size * 2.4) - (source_font_size * 2.0)
        lines_per_page = max(1, int(available_height // body_line_height))

        document = fitz.open()
        try:
            line_index = 0
            while line_index < len(wrapped_body_lines) or document.page_count == 0:
                page = document.new_page(width=page_width, height=page_height)
                page.insert_text(
                    (left_margin, top_margin),
                    title_text,
                    fontsize=title_font_size,
                    fontname="helv",
                )
                page.insert_text(
                    (left_margin, top_margin + (title_font_size * 1.9)),
                    source_line,
                    fontsize=source_font_size,
                    fontname="helv",
                )
                chunk = wrapped_body_lines[line_index : line_index + lines_per_page]
                line_index += len(chunk)
                body_rect = fitz.Rect(
                    left_margin,
                    top_margin + (title_font_size * 3.0),
                    page_width - right_margin,
                    page_height - bottom_margin,
                )
                page.insert_textbox(
                    body_rect,
                    "\n".join(chunk),
                    fontsize=body_font_size,
                    fontname="helv",
                    lineheight=1.25,
                )
            document.set_metadata(
                {
                    "title": title_text,
                    "subject": source_reference[:2000],
                }
            )
            document.save(str(target_path))
        finally:
            document.close()

    @staticmethod
    def _prepare_text_pdf_layout(
        *,
        title: str,
        text: str,
        source_reference: str,
    ) -> tuple[str, str, list[str]]:
        max_chars_per_line = 92
        source_line = f"Source: {source_reference}".strip()
        wrapped_body_lines: list[str] = []
        for paragraph in re.split(r"\n{2,}", text):
            normalized = " ".join(paragraph.split())
            if not normalized:
                wrapped_body_lines.append("")
                continue
            wrapped_body_lines.extend(
                textwrap.wrap(
                    normalized,
                    width=max_chars_per_line,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [normalized]
            )
            wrapped_body_lines.append("")
        if wrapped_body_lines and not wrapped_body_lines[-1]:
            wrapped_body_lines.pop()
        title_text = " ".join(str(title or "Document").split()) or "Document"
        return title_text, source_line, wrapped_body_lines

    @staticmethod
    def _write_extractable_text_pdf_with_matplotlib(
        target_path: Path,
        *,
        title: str,
        text: str,
        source_reference: str,
    ) -> None:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        title_text, source_line, wrapped_body_lines = DemocritusQueryAgenticRunner._prepare_text_pdf_layout(
            title=title,
            text=text,
            source_reference=source_reference,
        )
        lines_per_page = 58
        with PdfPages(str(target_path)) as pdf:
            line_index = 0
            while line_index < len(wrapped_body_lines) or line_index == 0:
                figure = plt.figure(figsize=(8.5, 11.0))
                figure.patch.set_facecolor("white")
                figure.text(0.08, 0.96, title_text, fontsize=15, va="top", ha="left")
                figure.text(0.08, 0.925, source_line, fontsize=8, va="top", ha="left")
                chunk = wrapped_body_lines[line_index : line_index + lines_per_page]
                line_index += len(chunk)
                figure.text(
                    0.08,
                    0.89,
                    "\n".join(chunk),
                    fontsize=10,
                    va="top",
                    ha="left",
                    family="monospace",
                )
                pdf.savefig(figure)
                plt.close(figure)

    @staticmethod
    def _write_extractable_text_pdf_basic(
        target_path: Path,
        *,
        title: str,
        text: str,
        source_reference: str,
    ) -> None:
        title_text, source_line, wrapped_body_lines = DemocritusQueryAgenticRunner._prepare_text_pdf_layout(
            title=title,
            text=text,
            source_reference=source_reference,
        )
        lines_per_page = 48
        page_chunks: list[list[str]] = []
        if not wrapped_body_lines:
            page_chunks.append([])
        else:
            for index in range(0, len(wrapped_body_lines), lines_per_page):
                page_chunks.append(wrapped_body_lines[index : index + lines_per_page])

        objects: list[bytes] = []

        def add_object(payload: str | bytes) -> int:
            if isinstance(payload, str):
                payload_bytes = payload.encode("latin-1", errors="replace")
            else:
                payload_bytes = payload
            objects.append(payload_bytes)
            return len(objects)

        font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        page_ids: list[int] = []
        content_ids: list[int] = []
        pages_id = len(objects) + (2 * len(page_chunks)) + 1

        for chunk in page_chunks:
            content_stream = DemocritusQueryAgenticRunner._basic_pdf_content_stream(
                title_text=title_text,
                source_line=source_line,
                body_lines=chunk,
            )
            content_id = add_object(
                f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
                + content_stream
                + b"\nendstream"
            )
            content_ids.append(content_id)
            page_id = add_object(
                (
                    "<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                    "/Resources << /ProcSet [/PDF /Text] /Font << /F1 {font_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).format(pages_id=pages_id, font_id=font_id)
            )
            page_ids.append(page_id)

        add_object(
            "<< /Type /Pages /Kids ["
            + " ".join(f"{page_id} 0 R" for page_id in page_ids)
            + f"] /Count {len(page_ids)} >>"
        )
        catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        buffer = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, payload in enumerate(objects, start=1):
            offsets.append(len(buffer))
            buffer.extend(f"{index} 0 obj\n".encode("latin-1"))
            buffer.extend(payload)
            buffer.extend(b"\nendobj\n")
        xref_offset = len(buffer)
        buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        buffer.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            buffer.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
        buffer.extend(
            (
                f"trailer << /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("latin-1")
        )
        target_path.write_bytes(bytes(buffer))

    @staticmethod
    def _basic_pdf_content_stream(
        *,
        title_text: str,
        source_line: str,
        body_lines: list[str],
    ) -> bytes:
        commands = [
            "BT",
            "/F1 16 Tf",
            "1 0 0 1 54 738 Tm",
            f"({DemocritusQueryAgenticRunner._escape_pdf_text(title_text)}) Tj",
            "/F1 8 Tf",
            "1 0 0 1 54 718 Tm",
            f"({DemocritusQueryAgenticRunner._escape_pdf_text(source_line)}) Tj",
            "/F1 10 Tf",
            "1 0 0 1 54 694 Tm",
        ]
        for line in body_lines:
            safe_line = DemocritusQueryAgenticRunner._escape_pdf_text(line)
            commands.append(f"({safe_line}) Tj")
            commands.append("0 -13 Td")
        commands.append("ET")
        return "\n".join(commands).encode("latin-1", errors="replace")

    @staticmethod
    def _escape_pdf_text(value: str) -> str:
        ascii_value = str(value or "").encode("latin-1", errors="replace").decode("latin-1")
        return ascii_value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    @staticmethod
    def _validate_pdf_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        if not payload:
            raise RuntimeError(f"downloaded file {path.name!r} is empty")
        if not payload.lstrip().startswith(b"%PDF-"):
            raise RuntimeError(f"downloaded file {path.name!r} is not a valid PDF payload")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire a query-driven corpus, then run agentic Democritus.")
    parser.add_argument(
        "--query",
        default="",
        help="Natural-language corpus acquisition request. If omitted, a local dashboard will open to collect it.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target-docs", type=int, default=None)
    parser.add_argument("--input-pdf", default="")
    parser.add_argument("--input-pdf-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--source-pdf-root", default="")
    parser.add_argument(
        "--retrieval-backend",
        default="auto",
        choices=["auto", "manifest", "filesystem", "direct_file", "direct_directory", "direct_url", "crossref", "semantic_scholar", "europe_pmc", "scholarly", "sec"],
    )
    parser.add_argument(
        "--retrieval-user-agent",
        default=_DEFAULT_RETRIEVAL_USER_AGENT,
        help=(
            "Retrieval identity string. For SEC EDGAR this should include contact info, "
            "for example 'Your Name your_email@example.com'."
        ),
    )
    parser.add_argument("--retrieval-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--sec-form", action="append", default=[], help="SEC form to request, e.g. 10-K. May be repeated.")
    parser.add_argument("--sec-company-limit", type=int, default=3)
    parser.add_argument("--corpus-name", default="query_corpus")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--consensus-batch-size", type=int, default=1)
    parser.add_argument("--consensus-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--consensus-stable-passes", type=int, default=1)
    parser.add_argument("--disable-consensus", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--agent-limit",
        action="append",
        default=[],
        help="Per-agent concurrency limit in the form agent_name=limit. May be repeated.",
    )
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument(
        "--root-topic-strategy",
        default="summary_guided",
        choices=["summary_guided", "v0_openai", "heuristic"],
    )
    parser.add_argument("--no-auto-topics", action="store_true")
    parser.add_argument("--depth-limit", type=int, default=3)
    parser.add_argument("--max-total-topics", type=int, default=100)
    parser.add_argument("--statements-per-question", type=int, default=2)
    parser.add_argument("--statement-batch-size", type=int, default=16)
    parser.add_argument("--statement-max-tokens", type=int, default=192)
    parser.add_argument("--manifold-mode", default="full", choices=["full", "lite", "moe"])
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--radii", default="1,2,3")
    parser.add_argument("--maxnodes", default="10,20,30,40,60")
    parser.add_argument("--lambda-edge", type=float, default=0.25)
    parser.add_argument("--topk-models", type=int, default=5)
    parser.add_argument("--topk-claims", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tier1", type=float, default=0.60)
    parser.add_argument("--tier2", type=float, default=0.30)
    parser.add_argument("--anchors", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--dedupe-focus", action="store_true")
    parser.add_argument("--require-anchor-in-focus", action="store_true")
    parser.add_argument("--focus-blacklist-regex", default="")
    parser.add_argument(
        "--render-topk-pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--png-dpi", type=int, default=200)
    parser.add_argument("--write-deep-dive", action="store_true")
    parser.add_argument("--deep-dive-max-bullets", type=int, default=8)
    parser.add_argument("--intra-document-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_agent_limit_args(values: list[str]) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --agent-limit value {value!r}; expected agent_name=limit.")
        agent_name, raw_limit = value.split("=", 1)
        agent_name = agent_name.strip()
        if not agent_name:
            raise ValueError(f"Invalid --agent-limit value {value!r}; missing agent name.")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"Invalid --agent-limit value {value!r}; limit must be an integer.") from exc
        parsed.append((agent_name, limit))
    return tuple(parsed)


def _resolve_query_for_main(args: argparse.Namespace) -> str:
    query = " ".join(str(args.query or "").split()).strip()
    if query:
        return query
    artifact_path = (
        Path(args.outdir).resolve()
        / "democritus_runs"
        / "corpus_synthesis"
        / "democritus_corpus_synthesis.html"
    )
    with DashboardQueryLauncher(
        DashboardQueryLauncherConfig(
            title="Democritus Query Dashboard",
            subtitle=(
                "Describe the corpus you want BAFFLE to retrieve, analyze document by document, "
                "and then synthesize across the full corpus."
            ),
            query_label="Corpus acquisition query",
            query_placeholder=(
                "Analyze 10 studies on the benefits of drinking red wine and synthesize the results\n"
                "or\n"
                "Analyze the PDF at /absolute/path/to/document.pdf\n"
                "or\n"
                "Analyze the PDFs in /absolute/path/to/folder\n"
                "or\n"
                "Analyze recent 10-K filings for Nvidia and AMD\n"
                "or\n"
                "Analyze the document at https://example.org/article"
            ),
            submit_label="Launch Democritus Run",
            waiting_message=(
                "The query has been captured. BAFFLE will retrieve the corpus, analyze each document, and open the synthesized Democritus result when the run finishes."
            ),
            artifact_path=artifact_path,
        )
    ) as launcher:
        return launcher.wait_for_submission()


def _open_dashboard_artifact(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=1, autoraise=True)
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    query = _resolve_query_for_main(args)
    inferred_target = infer_requested_result_count(
        query,
        nouns=("study", "studies", "paper", "papers", "article", "articles", "document", "documents"),
    )
    target_documents = int(args.target_docs) if args.target_docs is not None else (inferred_target or 10)
    max_docs = int(args.max_docs) if args.max_docs is not None else (inferred_target or 0)
    runner = DemocritusQueryAgenticRunner(
        DemocritusQueryAgenticConfig(
            query=query,
            outdir=Path(args.outdir),
            target_documents=target_documents,
            input_pdf_path=Path(args.input_pdf) if args.input_pdf else None,
            input_pdf_dir=Path(args.input_pdf_dir) if args.input_pdf_dir else None,
            manifest_path=Path(args.manifest) if args.manifest else None,
            source_pdf_root=Path(args.source_pdf_root) if args.source_pdf_root else None,
            retrieval_backend=args.retrieval_backend,
            retrieval_user_agent=args.retrieval_user_agent,
            retrieval_timeout_seconds=args.retrieval_timeout_seconds,
            discovery_only=args.discovery_only,
            sec_form_types=tuple(args.sec_form) if args.sec_form else ("10-K", "10-Q"),
            sec_company_limit=args.sec_company_limit,
            corpus_name=args.corpus_name,
            max_docs=max_docs,
            consensus_enabled=not args.disable_consensus,
            consensus_batch_size=args.consensus_batch_size,
            consensus_similarity_threshold=args.consensus_similarity_threshold,
            consensus_required_stable_passes=args.consensus_stable_passes,
            max_workers=args.max_workers,
            agent_concurrency_limits=_parse_agent_limit_args(args.agent_limit),
            include_phase2=not args.skip_phase2,
            auto_topics_from_pdf=not args.no_auto_topics,
            root_topic_strategy=args.root_topic_strategy,
            depth_limit=args.depth_limit,
            max_total_topics=args.max_total_topics,
            statements_per_question=args.statements_per_question,
            statement_batch_size=args.statement_batch_size,
            statement_max_tokens=args.statement_max_tokens,
            manifold_mode=args.manifold_mode,
            topk=args.topk,
            radii=args.radii,
            maxnodes=args.maxnodes,
            lambda_edge=args.lambda_edge,
            topk_models=args.topk_models,
            topk_claims=args.topk_claims,
            alpha=args.alpha,
            tier1=args.tier1,
            tier2=args.tier2,
            anchors=args.anchors,
            title=args.title,
            dedupe_focus=args.dedupe_focus,
            require_anchor_in_focus=args.require_anchor_in_focus,
            focus_blacklist_regex=args.focus_blacklist_regex,
            render_topk_pngs=args.render_topk_pngs,
            assets_dir=args.assets_dir,
            png_dpi=args.png_dpi,
            write_deep_dive=args.write_deep_dive,
            deep_dive_max_bullets=args.deep_dive_max_bullets,
            intra_document_shards=args.intra_document_shards,
            dry_run=args.dry_run,
        )
    )
    result = runner.run()
    artifact_path = result.checkpoint_dashboard_path or result.corpus_synthesis_dashboard_path
    if artifact_path is None or not artifact_path.exists():
        gui_path = result.batch_outdir / "democritus_gui.html"
        artifact_path = gui_path if gui_path.exists() else result.batch_outdir / "dashboard.html"
    _open_dashboard_artifact(artifact_path)
    print(
        json.dumps(
            {
                "query": result.query_plan.query,
                "retrieval_backend": runner._backend_name(),
                "selected_documents": len(result.selected_documents),
                "acquired_documents": len(result.acquired_documents),
                "pdf_dir": str(result.pdf_dir),
                "batch_outdir": str(result.batch_outdir),
                "batch_records": len(result.batch_records),
                "summary_path": str(result.summary_path),
                "corpus_synthesis_dashboard_path": (
                    str(result.corpus_synthesis_dashboard_path)
                    if result.corpus_synthesis_dashboard_path
                    else None
                ),
                "checkpoint_dashboard_path": (
                    str(result.checkpoint_dashboard_path)
                    if result.checkpoint_dashboard_path
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
