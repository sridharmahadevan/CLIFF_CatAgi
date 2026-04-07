"""Shared textbook backstop recommendations for CLIFF routes."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .repo_layout import resolve_book_pdf_path


@dataclass(frozen=True)
class TextbookSection:
    section_id: str
    title: str
    description: str
    start_page: int
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class TextbookBackstop:
    route_name: str
    query: str
    book_pdf_path: Path
    sections: tuple[TextbookSection, ...]
    rationale: str


def _normalize(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace("gt+db", "gt db")
    lowered = lowered.replace("gt-full", "gt full")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _contains_phrase(haystack: str, needle: str) -> bool:
    normalized_needle = _normalize(needle)
    if not haystack or not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {haystack} "


_SECTIONS: tuple[TextbookSection, ...] = (
    TextbookSection(
        section_id="kan_extension_transformers",
        title="Kan Extension and Topological Coend Transformers",
        description="Core KET chapter covering Kan-style aggregation and transformer constructions.",
        start_page=143,
        aliases=("kan extension transformers", "kan extension transformer", "ket"),
        keywords=("kan extension", "ket", "transformer"),
    ),
    TextbookSection(
        section_id="structured_language_modeling",
        title="Structured Language Modeling",
        description="Connects the model constructions to concrete sequence and language tasks.",
        start_page=179,
        aliases=("structured language modeling", "language modeling"),
        keywords=("language", "sequence", "modeling"),
    ),
    TextbookSection(
        section_id="manifold_learning_gt",
        title="Manifold Learning with Geometric Transformers",
        description="Relates geometric transformer machinery to learned manifolds and embeddings.",
        start_page=195,
        aliases=("manifold learning with geometric transformers", "manifold learning"),
        keywords=("manifold", "embedding", "geometry"),
    ),
    TextbookSection(
        section_id="causality_from_language",
        title="Causality from Language",
        description="Introduces the causal and language-grounded viewpoint used across Democritus-style synthesis.",
        start_page=273,
        aliases=("causality from language", "causality"),
        keywords=("causal", "causality", "language"),
    ),
    TextbookSection(
        section_id="temporal_diffusion",
        title="Temporal Diffusion over Causal Trajectories",
        description="Closest textbook anchor for cross-trajectory and cross-entity similarity through temporal causal structure.",
        start_page=287,
        aliases=("temporal diffusion over causal trajectories", "temporal diffusion"),
        keywords=("temporal diffusion", "trajectory", "similarity"),
    ),
    TextbookSection(
        section_id="agentic_systems_ket",
        title="Building Agentic Systems using Kan Extension Transformers",
        description="Covers KET-backed agentic workflows and composed reasoning systems.",
        start_page=307,
        aliases=("building agentic systems using kan extension transformers", "agentic systems"),
        keywords=("agentic", "workflow", "kan extension"),
    ),
    TextbookSection(
        section_id="topos_causal_models",
        title="Topos Causal Models",
        description="Connects structural semantics, gluing, and causal organization.",
        start_page=321,
        aliases=("topos causal models", "topos"),
        keywords=("topos", "causal model", "gluing"),
    ),
    TextbookSection(
        section_id="judo_calculus",
        title="Judo Calculus",
        description="Useful for intervention, conditioning, and counterfactual-style distinctions.",
        start_page=335,
        aliases=("judo calculus", "kan do"),
        keywords=("intervention", "conditioning", "do calculus"),
    ),
    TextbookSection(
        section_id="csql",
        title="CSQL: Mapping Documents into Topos Causal Model Databases",
        description="Best textbook bridge from document corpora into the structured causal database view used by Democritus.",
        start_page=371,
        aliases=("csql", "mapping documents into topos causal model databases"),
        keywords=("documents", "database", "csql", "corpus"),
    ),
    TextbookSection(
        section_id="universal_rl",
        title="Universal Reinforcement Learning",
        description="Relevant when feedback, behavior, and iterative decision signals are treated as structured evidence.",
        start_page=455,
        aliases=("universal reinforcement learning", "reinforcement learning"),
        keywords=("behavior", "policy", "feedback"),
    ),
    TextbookSection(
        section_id="consciousness",
        title="Consciousness",
        description="Connects the overall CLIFF-style conscious layer metaphor to the broader textbook framing.",
        start_page=493,
        aliases=("consciousness",),
        keywords=("consciousness", "interface", "conscious layer"),
    ),
    TextbookSection(
        section_id="code_companion",
        title="Code Companion",
        description="Points readers back from the textbook into the runnable code ecosystem.",
        start_page=521,
        aliases=("code companion",),
        keywords=("code", "companion", "implementation"),
    ),
)

_SECTIONS_BY_ID = {section.section_id: section for section in _SECTIONS}

_ROUTE_DEFAULTS: dict[str, tuple[str, ...]] = {
    "company_similarity": ("temporal_diffusion", "manifold_learning_gt", "causality_from_language"),
    "democritus": ("csql", "causality_from_language", "topos_causal_models"),
    "product_feedback": ("causality_from_language", "topos_causal_models", "judo_calculus"),
    "basket_rocket_sec": ("agentic_systems_ket", "kan_extension_transformers", "code_companion"),
    "culinary_tour": ("agentic_systems_ket", "consciousness", "code_companion"),
}

_ROUTE_RATIONALES: dict[str, str] = {
    "company_similarity": "These chapters explain the temporal, geometric, and causal ideas behind the cross-company diffusion comparison.",
    "democritus": "These chapters help connect multi-document synthesis to causal language structure, database-style gluing, and topos organization.",
    "product_feedback": "These chapters give the closest textbook framing for structured evidence, causal hypotheses, and intervention-style reasoning over feedback.",
    "basket_rocket_sec": "These chapters connect filing workflows back to KET-style agentic systems and the code-oriented companion sections.",
    "culinary_tour": "These chapters are the closest textbook bridge from the culinary itinerary to CLIFF's conscious layer, agentic coordination, and runnable code ecosystem.",
}


def _default_book_pdf_path() -> Path:
    return resolve_book_pdf_path()


def recommend_textbook_backstop(query: str, *, route_name: str) -> TextbookBackstop:
    normalized = _normalize(query)
    score_by_id: dict[str, int] = {section.section_id: 0 for section in _SECTIONS}

    for section_id in _ROUTE_DEFAULTS.get(route_name, ()):
        score_by_id[section_id] += 8

    for section in _SECTIONS:
        for alias in section.aliases:
            if _contains_phrase(normalized, alias):
                score_by_id[section.section_id] += 5
        for keyword in section.keywords:
            if _contains_phrase(normalized, keyword):
                score_by_id[section.section_id] += 2

    if route_name == "company_similarity" and any(token in normalized for token in ("similar", "similarity", "compare", "versus", " vs ")):
        score_by_id["temporal_diffusion"] += 4
        score_by_id["manifold_learning_gt"] += 2
    if route_name == "democritus" and any(token in normalized for token in ("study", "studies", "paper", "claim", "synthesize", "joint claims", "corpus")):
        score_by_id["csql"] += 4
        score_by_id["causality_from_language"] += 3
    if route_name == "product_feedback" and any(token in normalized for token in ("review", "reviews", "comfort", "easy", "drive", "assembly", "feedback")):
        score_by_id["causality_from_language"] += 3
        score_by_id["topos_causal_models"] += 2
    if route_name == "basket_rocket_sec" and any(token in normalized for token in ("10 k", "10 q", "workflow", "filing", "filings", "sec")):
        score_by_id["agentic_systems_ket"] += 3
        score_by_id["code_companion"] += 2
    if route_name == "culinary_tour" and any(token in normalized for token in ("tour", "itinerary", "travel", "trip", "meal", "restaurant", "culinary", "food")):
        score_by_id["consciousness"] += 3
        score_by_id["agentic_systems_ket"] += 2

    ranked = sorted(
        (section for section in _SECTIONS if score_by_id.get(section.section_id, 0) > 0),
        key=lambda section: (-score_by_id[section.section_id], section.start_page, section.title),
    )
    selected = tuple(ranked[:3])
    rationale = _ROUTE_RATIONALES.get(route_name, "These sections provide the closest textbook framing for this CLIFF route.")
    return TextbookBackstop(
        route_name=route_name,
        query=" ".join(query.split()),
        book_pdf_path=_default_book_pdf_path(),
        sections=selected,
        rationale=rationale,
    )


def render_textbook_backstop_html(backstop: TextbookBackstop, *, heading: str = "Read This in the Book") -> str:
    if not backstop.sections:
        return ""

    def esc(value: object) -> str:
        return html.escape(str(value))

    items = "".join(
        (
            "<li>"
            f"<strong>{esc(section.title)}</strong> "
            f"<span class=\"mono\">page {esc(section.start_page)}</span><br />"
            f"{esc(section.description)}"
            "</li>"
        )
        for section in backstop.sections
    )
    return (
        f"<p class=\"eyebrow\">{esc(heading)}</p>"
        f"<p class=\"trace\">{esc(backstop.rationale)}</p>"
        f"<p class=\"trace\">Textbook source: <code>{esc(backstop.book_pdf_path.name)}</code></p>"
        f"<ul class=\"textbook-list\">{items}</ul>"
    )
