"""Corpus-level post-processing for multi-document Democritus runs."""

from __future__ import annotations

import html
import itertools
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .causal_homotopy import (
    normalize_claim_text as _normalize_claim_text,
    normalize_relation as _normalize_relation,
    relation_polarity as _relation_polarity,
)
from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


@dataclass(frozen=True)
class DemocritusCorpusSynthesisResult:
    """Materialized corpus-level synthesis artifacts."""

    summary_path: Path
    dashboard_path: Path


@dataclass(frozen=True)
class CorpusClaim:
    subj: str
    rel: str
    obj: str
    domain: str
    statement: str
    document_support: int
    claim_count: int
    support_ratio: float
    truth_value: str
    supporting_runs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusRelationGroup:
    subj: str
    obj: str
    domain: str
    relation_class: str
    variants: tuple[CorpusClaim, ...]
    domain_aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "subj": self.subj,
            "obj": self.obj,
            "domain": self.domain,
            "domain_aliases": list(self.domain_aliases),
            "relation_class": self.relation_class,
            "variants": [item.as_dict() for item in self.variants],
        }


@dataclass(frozen=True)
class DiagnosticCorpusClaim:
    subj: str
    rel: str
    obj: str
    domain: str
    statement: str
    canonical_subj: str
    canonical_rel: str
    canonical_obj: str
    document_support: int
    claim_count: int
    support_ratio: float
    truth_value: str
    supporting_runs: tuple[str, ...]
    surface_form_count: int
    exact_document_support_max: int
    surface_forms: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HomotopyClaimClass:
    subj: str
    rel: str
    obj: str
    domain: str
    statement: str
    canonical_subj: str
    canonical_rel: str
    canonical_obj: str
    document_support: int
    claim_count: int
    support_ratio: float
    truth_value: str
    supporting_runs: tuple[str, ...]
    surface_form_count: int
    surface_forms: tuple[str, ...]
    domain_aliases: tuple[str, ...]
    variant_count: int
    simplex_vertices: int
    simplex_edges: int
    simplex_triangles: int
    open_horns: int
    connected_components: int
    horn_fill_ratio: float
    coherence_state: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeGluingClaim:
    canonical_subj: str
    canonical_obj: str
    regime_variant_count: int
    regime_count: int
    canonical_relation_count: int
    polarity_count: int
    total_document_support: int
    max_regime_support: int
    regimes: tuple[str, ...]
    canonical_relations: tuple[str, ...]
    gluing_state: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_DISPLAY_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "into",
    "of",
    "on",
    "that",
    "the",
    "their",
    "this",
    "to",
    "which",
}

_SURFACE_TOKEN_REWRITES: tuple[tuple[str, str], ...] = (
    (r"\bglucagon[\s-]+like[\s-]+peptide[\s-]+1\b", "glp1"),
    (r"\bglp[\s-]*1\b", "glp1"),
    (r"\bglp1 medicines\b", "glp1 receptor agonist"),
    (r"\bglp1 drugs\b", "glp1 receptor agonist"),
    (r"\bglp1 receptor agonists\b", "glp1 receptor agonist"),
)


def build_democritus_corpus_synthesis(
    *,
    query: str,
    batch_outdir: Path,
    csql_sqlite_path: Path,
) -> DemocritusCorpusSynthesisResult:
    """Build a corpus-level synthesis page from the batch CSQL bundle."""

    corpus_dir = batch_outdir / "corpus_synthesis"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_dir / "democritus_corpus_synthesis.json"
    dashboard_path = corpus_dir / "democritus_corpus_synthesis.html"

    with sqlite3.connect(str(csql_sqlite_path)) as connection:
        total_documents = _scalar(connection, "SELECT COUNT(*) FROM documents")
        claims = _load_claims(connection, total_documents=total_documents)
        equivalence_classes, disagreements = _load_relation_groups(connection, total_documents=total_documents)
        diagnostic_claims = _load_diagnostic_claims(connection, total_documents=total_documents)
        homotopy_classes = _load_homotopy_claim_classes(
            connection,
            claims=claims,
            total_documents=total_documents,
        )
        regime_gluing_claims = _load_regime_gluing_claims(connection)
        contested_keys = {
            (item.subj, item.obj, item.domain)
            for item in disagreements
        }
        strong_claims = [
            item for item in claims
            if item.truth_value in {"entailed", "strong_support"}
            and (item.subj, item.obj, item.domain) not in contested_keys
        ]
        weak_claims = [
            item for item in claims
            if item.truth_value in {"provisional_support", "weak_support"}
            and (item.subj, item.obj, item.domain) not in contested_keys
        ]
        strong_claims = _coalesce_display_claims(strong_claims, total_documents=total_documents)
        weak_claims = _coalesce_display_claims(weak_claims, total_documents=total_documents)
        study_cards = _load_study_cards(connection, batch_outdir=batch_outdir)

    homotopy_summary = {
        "class_count": len(homotopy_classes),
        "within_document_class_count": sum(1 for item in homotopy_classes if item.document_support <= 1),
        "cross_document_class_count": sum(1 for item in homotopy_classes if item.document_support > 1),
        "coherent_count": sum(1 for item in homotopy_classes if item.coherence_state == "coherent"),
        "partially_glued_count": sum(1 for item in homotopy_classes if item.coherence_state == "partially_glued"),
        "disconnected_count": sum(1 for item in homotopy_classes if item.coherence_state == "disconnected"),
    }
    regime_gluing_summary = {
        "surface_count": len(regime_gluing_claims),
        "obstructed_count": sum(1 for item in regime_gluing_claims if item.gluing_state == "obstructed"),
        "regime_sensitive_count": sum(1 for item in regime_gluing_claims if item.gluing_state == "regime_sensitive"),
        "multi_regime_glued_count": sum(1 for item in regime_gluing_claims if item.gluing_state == "multi_regime_glued"),
    }

    payload = {
        "query": query,
        "csql_sqlite_path": str(csql_sqlite_path),
        "n_documents": total_documents,
        "strongly_supported": [item.as_dict() for item in strong_claims[:12]],
        "weakly_supported": [item.as_dict() for item in weak_claims[:12]],
        "diagnostic_supported": [item.as_dict() for item in diagnostic_claims[:12]],
        "equivalence_classes": [item.as_dict() for item in equivalence_classes[:8]],
        "disagreements": [item.as_dict() for item in disagreements[:8]],
        "homotopy_summary": homotopy_summary,
        "homotopy_classes": [item.as_dict() for item in homotopy_classes[:8]],
        "regime_gluing_summary": regime_gluing_summary,
        "regime_gluing_claims": [item.as_dict() for item in regime_gluing_claims[:8]],
        "study_cards": study_cards,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dashboard_path.write_text(
        _render_dashboard_html(payload, dashboard_path=dashboard_path, batch_outdir=batch_outdir),
        encoding="utf-8",
    )
    return DemocritusCorpusSynthesisResult(
        summary_path=summary_path,
        dashboard_path=dashboard_path,
    )


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0] or 0) if row else 0


def _support_truth_value(*, document_support: int, total_documents: int) -> str:
    if total_documents <= 0:
        return "weak_support"
    ratio = document_support / total_documents
    if document_support == total_documents and total_documents >= 2:
        return "entailed"
    if document_support >= max(3, (total_documents + 1) // 2):
        return "strong_support"
    if ratio >= 0.4 or document_support >= 2:
        return "provisional_support"
    return "weak_support"


def _split_runs(raw_runs: str) -> tuple[str, ...]:
    runs = [item.strip() for item in str(raw_runs or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(runs))


def _decode_json_array(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple)):
        values = raw_value
    else:
        text = str(raw_value or "").strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
        values = parsed if isinstance(parsed, list) else [parsed]
    decoded = [str(item).strip() for item in values if str(item).strip()]
    return tuple(dict.fromkeys(decoded))


def _load_diagnostic_claims(
    connection: sqlite3.Connection,
    *,
    total_documents: int,
) -> list[DiagnosticCorpusClaim]:
    rows = connection.execute(
        """
        SELECT
            canonical_subj,
            canonical_rel,
            canonical_obj,
            canonical_domain,
            statement,
            document_support,
            claim_count,
            supporting_runs_json,
            surface_form_count,
            exact_document_support_max,
            surface_forms_json
        FROM homotopy_localized_claims
        ORDER BY document_support DESC, surface_form_count DESC, claim_count DESC, canonical_subj, canonical_obj
        """
    ).fetchall()
    diagnostic_claims: list[DiagnosticCorpusClaim] = []
    for row in rows:
        canonical_subj = str(row[0] or "")
        canonical_rel = str(row[1] or "")
        canonical_obj = str(row[2] or "")
        document_support = int(row[5] or 0)
        if document_support < 2:
            continue
        surface_form_count = int(row[8] or 0)
        exact_document_support_max = int(row[9] or 0)
        if document_support <= exact_document_support_max and surface_form_count <= 1:
            continue
        supporting_runs = _decode_json_array(row[7])
        surface_forms = _decode_json_array(row[10])
        diagnostic_claims.append(
            DiagnosticCorpusClaim(
                subj=canonical_subj,
                rel=canonical_rel,
                obj=canonical_obj,
                domain=str(row[3] or ""),
                statement=str(row[4] or ""),
                canonical_subj=canonical_subj,
                canonical_rel=canonical_rel,
                canonical_obj=canonical_obj,
                document_support=document_support,
                claim_count=int(row[6] or 0),
                support_ratio=round(document_support / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(
                    document_support=document_support,
                    total_documents=total_documents,
                ),
                supporting_runs=tuple(supporting_runs),
                surface_form_count=surface_form_count,
                exact_document_support_max=exact_document_support_max,
                surface_forms=surface_forms[:4],
            )
        )
    diagnostic_claims.sort(
        key=lambda item: (
            -item.document_support,
            -item.surface_form_count,
            -item.claim_count,
            item.subj.lower(),
            item.obj.lower(),
        )
    )
    return diagnostic_claims


def _variant_surface_form(item: CorpusClaim) -> str:
    surface = str(item.statement or "").strip()
    if surface:
        return " ".join(surface.split())
    return " ".join(f"{item.subj} {item.rel} {item.obj}".split()).strip()


def _variant_signature(item: CorpusClaim) -> tuple[str, ...]:
    tokens: list[str] = []
    tokens.extend(_token_signature(item.subj))
    tokens.extend(_token_signature(item.obj))
    tokens.extend(_token_signature(item.domain))
    tokens.extend(_token_signature(_variant_surface_form(item)))
    relation = _normalize_relation(item.rel)
    if relation:
        tokens.append(relation)
    return tuple(sorted(dict.fromkeys(token for token in tokens if token)))


def _surface_signature(value: str) -> tuple[str, ...]:
    text = str(value or "").strip().lower()
    if not text:
        return ()
    text = text.replace("_", " ")
    for pattern, replacement in _SURFACE_TOKEN_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[\-/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ()
    return tuple(
        sorted(
            dict.fromkeys(
                token
                for token in text.split()
                if token and token not in _DISPLAY_TOKEN_STOPWORDS
            )
        )
    )


def _variant_similarity(left: CorpusClaim, right: CorpusClaim) -> float:
    subj_score = _jaccard_similarity(_token_signature(left.subj), _token_signature(right.subj))
    obj_score = _jaccard_similarity(_token_signature(left.obj), _token_signature(right.obj))
    domain_score = _jaccard_similarity(_token_signature(left.domain), _token_signature(right.domain))
    statement_score = _jaccard_similarity(_surface_signature(_variant_surface_form(left)), _surface_signature(_variant_surface_form(right)))
    relation_score = 0.0
    if _normalize_relation(left.rel) == _normalize_relation(right.rel):
        relation_score = 1.0
    elif _relation_polarity(left.rel) == _relation_polarity(right.rel):
        relation_score = 0.6
    return (
        (0.18 * subj_score)
        + (0.18 * obj_score)
        + (0.24 * domain_score)
        + (0.30 * statement_score)
        + (0.10 * relation_score)
    )


def _should_link_homotopy_variants(left: CorpusClaim, right: CorpusClaim) -> bool:
    if _relation_polarity(left.rel) != _relation_polarity(right.rel):
        return False
    subj_score = _jaccard_similarity(_token_signature(left.subj), _token_signature(right.subj))
    obj_score = _jaccard_similarity(_token_signature(left.obj), _token_signature(right.obj))
    if subj_score < 0.65 or obj_score < 0.65:
        return False
    domain_score = _jaccard_similarity(_token_signature(left.domain), _token_signature(right.domain))
    statement_score = _jaccard_similarity(_surface_signature(_variant_surface_form(left)), _surface_signature(_variant_surface_form(right)))
    combined_score = _variant_similarity(left, right)
    if statement_score >= 0.76:
        return True
    if statement_score >= 0.62 and domain_score >= 0.25:
        return True
    if statement_score >= 0.52 and domain_score >= 0.55:
        return True
    return combined_score >= 0.64 and statement_score >= 0.48 and domain_score >= 0.40


def _homotopy_graph_metrics(items: list[CorpusClaim]) -> tuple[int, int, int, int, float, str]:
    vertex_count = len(items)
    if vertex_count <= 1:
        return 0, 0, 0, 1, 1.0, "coherent"

    adjacency: list[set[int]] = [set() for _ in items]
    edge_count = 0
    for left_index, right_index in itertools.combinations(range(vertex_count), 2):
        if not _should_link_homotopy_variants(items[left_index], items[right_index]):
            continue
        adjacency[left_index].add(right_index)
        adjacency[right_index].add(left_index)
        edge_count += 1

    connected_components = 0
    seen: set[int] = set()
    for start_index in range(vertex_count):
        if start_index in seen:
            continue
        connected_components += 1
        stack = [start_index]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(sorted(adjacency[current] - seen))

    triangle_count = 0
    open_horns = 0
    for left_index, middle_index, right_index in itertools.combinations(range(vertex_count), 3):
        edges_present = sum(
            (
                int(middle_index in adjacency[left_index]),
                int(right_index in adjacency[left_index]),
                int(right_index in adjacency[middle_index]),
            )
        )
        if edges_present == 3:
            triangle_count += 1
        elif edges_present == 2:
            open_horns += 1

    denominator = triangle_count + open_horns
    horn_fill_ratio = round((triangle_count / denominator), 3) if denominator else 1.0
    if connected_components == 1 and horn_fill_ratio >= 0.75:
        coherence_state = "coherent"
    elif connected_components == 1:
        coherence_state = "partially_glued"
    else:
        coherence_state = "disconnected"
    return edge_count, triangle_count, open_horns, connected_components, horn_fill_ratio, coherence_state


def _load_homotopy_claim_classes(
    connection: sqlite3.Connection,
    *,
    claims: list[CorpusClaim],
    total_documents: int,
) -> list[HomotopyClaimClass]:
    grouped: dict[tuple[str, str, str], list[CorpusClaim]] = {}
    for item in claims:
        key = (
            _normalize_claim_text(item.subj),
            _normalize_relation(item.rel),
            _normalize_claim_text(item.obj),
        )
        if not all(key):
            continue
        grouped.setdefault(key, []).append(item)

    localized_rows = connection.execute(
        """
        SELECT
            canonical_subj,
            canonical_rel,
            canonical_obj,
            canonical_domain,
            statement,
            document_support,
            claim_count,
            supporting_runs_json,
            surface_form_count,
            surface_forms_json,
            domain_aliases_json,
            variant_count
        FROM homotopy_localized_claims
        ORDER BY document_support DESC, surface_form_count DESC, claim_count DESC, canonical_subj, canonical_obj
        """
    ).fetchall()

    homotopy_classes: list[HomotopyClaimClass] = []
    for row in localized_rows:
        canonical_subj = str(row[0] or "")
        canonical_rel = str(row[1] or "")
        canonical_obj = str(row[2] or "")
        items = grouped.get((canonical_subj, canonical_rel, canonical_obj), [])
        surface_forms = _decode_json_array(row[9])
        domain_aliases = _decode_json_array(row[10])
        if not items and len(surface_forms) <= 1 and len(domain_aliases) <= 1:
            continue
        simplex_edges, simplex_triangles, open_horns, connected_components, horn_fill_ratio, coherence_state = _homotopy_graph_metrics(items)
        supporting_runs = _decode_json_array(row[7])
        document_support = int(row[5] or 0)
        representative = max(
            items,
            key=lambda item: (
                int(item.document_support),
                int(item.claim_count),
                _domain_rank(item.domain),
                len(item.statement),
            ),
            default=CorpusClaim(
                subj=canonical_subj,
                rel=canonical_rel,
                obj=canonical_obj,
                domain=str(row[3] or ""),
                statement=str(row[4] or ""),
                document_support=document_support,
                claim_count=int(row[6] or 0),
                support_ratio=round(document_support / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(document_support=document_support, total_documents=total_documents),
                supporting_runs=supporting_runs,
            ),
        )
        homotopy_classes.append(
            HomotopyClaimClass(
                subj=representative.subj,
                rel=representative.rel,
                obj=representative.obj,
                domain=max(domain_aliases, key=_domain_rank, default=str(row[3] or representative.domain)),
                statement=str(row[4] or representative.statement),
                canonical_subj=canonical_subj,
                canonical_rel=canonical_rel,
                canonical_obj=canonical_obj,
                document_support=document_support,
                claim_count=int(row[6] or 0),
                support_ratio=round(document_support / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(
                    document_support=document_support,
                    total_documents=total_documents,
                ),
                supporting_runs=tuple(supporting_runs),
                surface_form_count=int(row[8] or len(surface_forms)),
                surface_forms=surface_forms[:6],
                domain_aliases=domain_aliases[:6],
                variant_count=int(row[11] or len(items)),
                simplex_vertices=max(len(items), int(row[8] or 0)),
                simplex_edges=simplex_edges,
                simplex_triangles=simplex_triangles,
                open_horns=open_horns,
                connected_components=connected_components,
                horn_fill_ratio=horn_fill_ratio,
                coherence_state=coherence_state,
            )
        )

    homotopy_classes.sort(
        key=lambda item: (
            -item.document_support,
            -item.surface_form_count,
            -item.claim_count,
            item.subj.lower(),
            item.obj.lower(),
        )
    )
    return homotopy_classes


def _load_regime_gluing_claims(connection: sqlite3.Connection) -> list[RegimeGluingClaim]:
    rows = connection.execute(
        """
        SELECT
            canonical_subj,
            canonical_obj,
            regime_variant_count,
            regime_count,
            canonical_relation_count,
            polarity_count,
            total_document_support,
            max_regime_support,
            regimes_json,
            canonical_relations_json,
            gluing_state
        FROM regime_gluing_surfaces
        ORDER BY
            CASE gluing_state
                WHEN 'obstructed' THEN 0
                WHEN 'regime_sensitive' THEN 1
                WHEN 'multi_regime_glued' THEN 2
                ELSE 3
            END,
            total_document_support DESC,
            canonical_subj,
            canonical_obj
        """
    ).fetchall()
    claims: list[RegimeGluingClaim] = []
    for row in rows:
        claims.append(
            RegimeGluingClaim(
                canonical_subj=str(row[0] or ""),
                canonical_obj=str(row[1] or ""),
                regime_variant_count=int(row[2] or 0),
                regime_count=int(row[3] or 0),
                canonical_relation_count=int(row[4] or 0),
                polarity_count=int(row[5] or 0),
                total_document_support=int(row[6] or 0),
                max_regime_support=int(row[7] or 0),
                regimes=_decode_json_array(row[8]),
                canonical_relations=_decode_json_array(row[9]),
                gluing_state=str(row[10] or "single_regime"),
            )
        )
    return claims


def _load_claims(connection: sqlite3.Connection, *, total_documents: int) -> list[CorpusClaim]:
    rows = connection.execute(
        """
        SELECT
            subj.name AS subj,
            rel.name AS rel,
            obj.name AS obj,
            dom.name AS domain,
            MIN(c.statement) AS statement,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT d.run_name) AS document_support,
            GROUP_CONCAT(DISTINCT d.run_name) AS supporting_runs
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        JOIN documents d ON d.document_id = c.document_id
        GROUP BY subj.name, rel.name, obj.name, dom.name
        ORDER BY document_support DESC, claim_count DESC, subj.name, rel.name, obj.name
        """
    ).fetchall()
    claims: list[CorpusClaim] = []
    for row in rows:
        document_support = int(row[6] or 0)
        claims.append(
            CorpusClaim(
                subj=str(row[0] or ""),
                rel=str(row[1] or ""),
                obj=str(row[2] or ""),
                domain=str(row[3] or ""),
                statement=str(row[4] or ""),
                claim_count=int(row[5] or 0),
                document_support=document_support,
                support_ratio=round(document_support / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(
                    document_support=document_support,
                    total_documents=total_documents,
                ),
                supporting_runs=_split_runs(str(row[7] or "")),
            )
        )
    return claims


def _token_signature(value: str) -> tuple[str, ...]:
    normalized = _normalize_claim_text(value)
    if not normalized:
        return ()
    tokens = [
        token
        for token in normalized.split()
        if token and token not in _DISPLAY_TOKEN_STOPWORDS
    ]
    if not tokens:
        return ()
    return tuple(sorted(dict.fromkeys(tokens)))


def _domain_rank(value: str) -> tuple[int, int]:
    text = str(value or "").strip()
    lowered = text.lower()
    is_placeholder = lowered.startswith("no relevant content on ")
    return (0 if is_placeholder else 1, len(text))


def _coalesce_display_claims(
    claims: list[CorpusClaim],
    *,
    total_documents: int,
    include_domain: bool = True,
    relation_key_mode: str = "polarity",
) -> list[CorpusClaim]:
    if len(claims) <= 1:
        return claims

    grouped: dict[tuple[tuple[str, ...], str, tuple[str, ...], str], list[CorpusClaim]] = {}
    for item in claims:
        domain_key = ""
        if include_domain and total_documents > 1:
            domain_key = _normalize_claim_text(item.domain)
        relation_key = _relation_polarity(item.rel)
        if relation_key_mode == "normalized":
            relation_key = _normalize_relation(item.rel)
        key = (
            _token_signature(item.subj),
            relation_key,
            _token_signature(item.obj),
            domain_key,
        )
        if not all(key[:3]):
            key = (
                (item.subj.strip().lower(),),
                relation_key,
                (item.obj.strip().lower(),),
                domain_key,
            )
        grouped.setdefault(key, []).append(item)

    collapsed: list[CorpusClaim] = []
    for items in grouped.values():
        if len(items) == 1:
            collapsed.append(items[0])
            continue
        representative = max(
            items,
            key=lambda item: (
                int(item.document_support),
                int(item.claim_count),
                _domain_rank(item.domain),
                len(item.statement),
            ),
        )
        supporting_runs: list[str] = []
        seen_runs: set[str] = set()
        for item in items:
            for run_name in item.supporting_runs:
                if run_name in seen_runs:
                    continue
                seen_runs.add(run_name)
                supporting_runs.append(run_name)
        best_domain = max((item.domain for item in items), key=_domain_rank, default=representative.domain)
        collapsed.append(
            CorpusClaim(
                subj=representative.subj,
                rel=representative.rel,
                obj=representative.obj,
                domain=best_domain,
                statement=representative.statement,
                claim_count=sum(int(item.claim_count) for item in items),
                document_support=len(supporting_runs),
                support_ratio=round(len(supporting_runs) / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(
                    document_support=len(supporting_runs),
                    total_documents=total_documents,
                ),
                supporting_runs=tuple(supporting_runs),
            )
        )

    collapsed.sort(
        key=lambda item: (
            -item.document_support,
            -item.claim_count,
            item.subj.lower(),
            item.rel.lower(),
            item.obj.lower(),
        )
    )
    return collapsed


def _jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _relation_group_polarity_key(item: CorpusRelationGroup) -> str:
    polarities = sorted({_relation_polarity(variant.rel) for variant in item.variants})
    return "|".join(polarities) or "unknown"


def _relation_group_signature(item: CorpusRelationGroup) -> tuple[str, ...]:
    representative = max(
        item.variants,
        key=lambda variant: (
            int(variant.document_support),
            int(variant.claim_count),
            len(variant.statement),
        ),
    )
    tokens: list[str] = []
    tokens.extend(_token_signature(item.domain))
    tokens.extend(_token_signature(representative.statement or representative.obj))
    tokens.extend(_normalize_relation(variant.rel) for variant in item.variants if str(variant.rel or "").strip())
    return tuple(sorted(dict.fromkeys(token for token in tokens if token)))


def _merge_relation_group_component(
    items: list[CorpusRelationGroup],
    *,
    total_documents: int,
) -> CorpusRelationGroup:
    representative = max(
        items,
        key=lambda item: (
            max((int(variant.document_support) for variant in item.variants), default=0),
            sum(int(variant.claim_count) for variant in item.variants),
            _domain_rank(item.domain),
        ),
    )
    domain_aliases = tuple(
        dict.fromkeys(
            domain
            for item in items
            for domain in ((item.domain,) + tuple(item.domain_aliases or ()))
            if str(domain or "").strip()
        )
    )
    domain_support: dict[str, int] = {}
    for item in items:
        item_support = max((int(variant.document_support) for variant in item.variants), default=0)
        for domain in ((item.domain,) + tuple(item.domain_aliases or ())):
            if not str(domain or "").strip():
                continue
            domain_support[str(domain)] = domain_support.get(str(domain), 0) + item_support
    best_domain = max(
        domain_support,
        key=lambda domain: (domain_support[domain], _domain_rank(domain)),
        default=representative.domain,
    )
    merged_variants = tuple(
        _coalesce_display_claims(
            [variant for item in items for variant in item.variants],
            total_documents=total_documents,
            include_domain=False,
            relation_key_mode="normalized",
        )
    )
    return CorpusRelationGroup(
        subj=representative.subj,
        obj=representative.obj,
        domain=best_domain,
        domain_aliases=domain_aliases,
        relation_class=representative.relation_class,
        variants=merged_variants,
    )


def _merge_equivalence_groups_knn(
    groups: list[CorpusRelationGroup],
    *,
    total_documents: int,
    k: int = 3,
    min_similarity: float = 0.33,
    mean_similarity_gate: float = 0.36,
) -> list[CorpusRelationGroup]:
    if len(groups) <= 1:
        return groups

    merged: list[CorpusRelationGroup] = []
    buckets: dict[tuple[tuple[str, ...], tuple[str, ...], str], list[CorpusRelationGroup]] = {}
    for item in groups:
        key = (
            _token_signature(item.subj) or (item.subj.strip().lower(),),
            _token_signature(item.obj) or (item.obj.strip().lower(),),
            _relation_group_polarity_key(item),
        )
        buckets.setdefault(key, []).append(item)

    for bucket_items in buckets.values():
        if len(bucket_items) <= 1:
            merged.extend(bucket_items)
            continue

        signatures = [_relation_group_signature(item) for item in bucket_items]
        neighbors: list[list[int]] = [[] for _ in bucket_items]
        for left_index, left_item in enumerate(bucket_items):
            scored_neighbors: list[tuple[float, int]] = []
            for right_index, right_item in enumerate(bucket_items):
                if left_index == right_index:
                    continue
                similarity = _jaccard_similarity(signatures[left_index], signatures[right_index])
                if similarity < min_similarity:
                    continue
                right_support = max((int(variant.document_support) for variant in right_item.variants), default=0)
                score = similarity * max(right_support, 1)
                scored_neighbors.append((score, right_index))
            scored_neighbors.sort(reverse=True)
            chosen = scored_neighbors[: max(k, 1)]
            if not chosen:
                continue
            mean_similarity = sum(
                _jaccard_similarity(signatures[left_index], signatures[right_index])
                for _score, right_index in chosen
            ) / len(chosen)
            if mean_similarity < mean_similarity_gate:
                continue
            neighbors[left_index] = [right_index for _score, right_index in chosen]

        adjacency: list[set[int]] = [set() for _ in bucket_items]
        for left_index, right_indexes in enumerate(neighbors):
            for right_index in right_indexes:
                if left_index in neighbors[right_index]:
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)

        seen: set[int] = set()
        for start_index in range(len(bucket_items)):
            if start_index in seen:
                continue
            stack = [start_index]
            component_indexes: list[int] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component_indexes.append(current)
                stack.extend(sorted(adjacency[current] - seen))
            component = [bucket_items[index] for index in sorted(component_indexes)]
            if len(component) == 1:
                merged.append(component[0])
            else:
                merged.append(
                    _merge_relation_group_component(
                        component,
                        total_documents=total_documents,
                    )
                )
    return merged


def _load_relation_groups(
    connection: sqlite3.Connection,
    *,
    total_documents: int,
) -> tuple[list[CorpusRelationGroup], list[CorpusRelationGroup]]:
    rows = connection.execute(
        """
        SELECT
            subj.name AS subj,
            obj.name AS obj,
            dom.name AS domain,
            rel.name AS rel,
            MIN(c.statement) AS statement,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT d.run_name) AS document_support,
            GROUP_CONCAT(DISTINCT d.run_name) AS supporting_runs
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        JOIN documents d ON d.document_id = c.document_id
        GROUP BY subj.name, obj.name, dom.name, rel.name
        ORDER BY subj.name, obj.name, dom.name, document_support DESC, claim_count DESC
        """
    ).fetchall()
    grouped: dict[tuple[str, str, str], list[CorpusClaim]] = {}
    for row in rows:
        key = (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""))
        document_support = int(row[6] or 0)
        grouped.setdefault(key, []).append(
            CorpusClaim(
                subj=key[0],
                rel=str(row[3] or ""),
                obj=key[1],
                domain=key[2],
                statement=str(row[4] or ""),
                claim_count=int(row[5] or 0),
                document_support=document_support,
                support_ratio=round(document_support / total_documents, 3) if total_documents else 0.0,
                truth_value=_support_truth_value(
                    document_support=document_support,
                    total_documents=total_documents,
                ),
                supporting_runs=_split_runs(str(row[7] or "")),
            )
        )
    equivalence_classes: list[CorpusRelationGroup] = []
    disagreements: list[CorpusRelationGroup] = []
    for key, items in grouped.items():
        if len(items) <= 1:
            continue
        polarities = {_relation_polarity(item.rel) for item in items}
        relation_class = "equivalence_class"
        target = equivalence_classes
        if "positive" in polarities and "negative" in polarities:
            relation_class = "disagreement"
            target = disagreements
        target.append(
            CorpusRelationGroup(
                subj=key[0],
                obj=key[1],
                domain=key[2],
                domain_aliases=(key[2],),
                relation_class=relation_class,
                variants=tuple(items),
            )
        )
    equivalence_classes = _merge_equivalence_groups_knn(
        equivalence_classes,
        total_documents=total_documents,
    )
    equivalence_classes = [item for item in equivalence_classes if len(item.variants) > 1]
    equivalence_classes.sort(
        key=lambda item: (
            -max(variant.document_support for variant in item.variants),
            item.subj.lower(),
            item.obj.lower(),
            item.domain.lower(),
        )
    )
    disagreements.sort(
        key=lambda item: (
            -max(variant.document_support for variant in item.variants),
            item.subj.lower(),
            item.obj.lower(),
            item.domain.lower(),
        )
    )
    return equivalence_classes, disagreements


def _load_study_cards(connection: sqlite3.Connection, *, batch_outdir: Path) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT run_name, pdf_path
        FROM documents
        ORDER BY run_name
        """
    ).fetchall()
    cards: list[dict[str, object]] = []
    for run_name, pdf_path in rows:
        run_name_str = str(run_name or "")
        pdf_path_obj = Path(str(pdf_path or ""))
        summary_viewer = batch_outdir / run_name_str / "reports" / f"{run_name_str}_executive_summary.html"
        credibility_viewer = batch_outdir / run_name_str / "reports" / f"{run_name_str}_credibility_report.html"
        cards.append(
            {
                "run_name": run_name_str,
                "title": pdf_path_obj.stem.replace("_", " ") if pdf_path_obj.name else run_name_str,
                "summary_href": _relative_href(summary_viewer, start=batch_outdir / "corpus_synthesis"),
                "credibility_href": _relative_href(credibility_viewer, start=batch_outdir / "corpus_synthesis"),
            }
        )
    return cards


def _relative_href(target: Path, *, start: Path) -> str:
    if not target.exists():
        return ""
    return os.path.relpath(target.resolve(), start=start.resolve())


def _render_claim_card(claim: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    runs = ", ".join(claim.get("supporting_runs") or [])
    return (
        '<article class="claim-card">'
        f'<div class="claim-meta">{esc(claim["truth_value"]).replace("_", " ")} · '
        f'{esc(claim["document_support"])} study(s)</div>'
        f'<h3>{esc(claim["subj"])} {esc(claim["rel"])} {esc(claim["obj"])}</h3>'
        f'<p>{esc(claim["statement"] or claim["domain"])}</p>'
        f'<p class="trace">Domain: {esc(claim["domain"])} · Support ratio: {esc(claim["support_ratio"])}'
        + (f' · Runs: {esc(runs)}' if runs else "")
        + "</p>"
        "</article>"
    )


def _representative_variant(item: dict[str, object]) -> dict[str, object]:
    variants = list(item.get("variants") or [])
    if not variants:
        return {}
    return max(
        variants,
        key=lambda variant: (
            int(variant.get("document_support") or 0),
            int(variant.get("claim_count") or 0),
            len(str(variant.get("statement") or "")),
        ),
    )


def _truncate_text(value: object, *, maxlen: int = 140) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 3].rstrip() + "..."


def _render_equivalence_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    variants = list(item.get("variants") or [])
    representative = _representative_variant(item)
    variant_count = len(variants)
    max_support = max((int(variant.get("document_support") or 0) for variant in variants), default=0)
    domain_aliases = [
        str(domain).strip()
        for domain in (item.get("domain_aliases") or [])
        if str(domain).strip()
    ]
    alternate_domains = [domain for domain in domain_aliases if domain != str(item.get("domain") or "")]
    relation_variants = ", ".join(
        str(variant.get("rel") or "")
        for variant in variants
        if str(variant.get("rel") or "").strip()
    )
    representative_statement = _truncate_text(representative.get("statement") or representative.get("obj") or "")
    variant_rows = "".join(
        (
            '<div class="variant-row">'
            f'<strong>{esc(variant.get("rel") or "")}</strong>'
            f'<span>{esc(_truncate_text(variant.get("statement") or variant.get("obj") or "", maxlen=110))}</span>'
            f'<span class="variant-meta">{esc(variant.get("document_support") or 0)} study(s)</span>'
            "</div>"
        )
        for variant in variants
    )
    return (
        '<article class="equivalence-card">'
        f'<div class="claim-meta">{esc(variant_count)} same-direction variant(s) · up to {esc(max_support)} study(s)</div>'
        f'<h3>{esc(item["subj"])} -> {esc(item["obj"])}</h3>'
        + (
            f'<p><strong>Backbone claim:</strong> {esc(representative_statement)}</p>'
            if representative_statement
            else ""
        )
        + (
            f'<p class="trace">Domain: {esc(item["domain"])} · Relation family: {esc(relation_variants)}</p>'
            if relation_variants
            else f'<p class="trace">Domain: {esc(item["domain"])}</p>'
        )
        + (
            f'<p class="trace">Also seen under: {esc(" | ".join(alternate_domains[:3]))}</p>'
            if alternate_domains
            else ""
        )
        + variant_rows
        + "</article>"
    )


def _render_disagreement_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    variants = "".join(
        (
            '<div class="variant-row">'
            f'<strong>{esc(variant["rel"])}</strong> '
            f'<span>{esc(variant["statement"] or variant["obj"])}</span>'
            f'<span class="variant-meta">{esc(variant["document_support"])} study(s)</span>'
            "</div>"
        )
        for variant in item.get("variants") or []
    )
    return (
        '<article class="disagreement-card">'
        f'<h3>{esc(item["subj"])} ↔ {esc(item["obj"])}</h3>'
        f'<p class="trace">Domain: {esc(item["domain"])}</p>'
        f"{variants}"
        "</article>"
    )


def _render_diagnostic_claim_card(claim: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    runs = ", ".join(claim.get("supporting_runs") or [])
    surface_forms = claim.get("surface_forms") or []
    surface_preview = " | ".join(str(item) for item in surface_forms[:3])
    return (
        '<article class="claim-card">'
        f'<div class="claim-meta">diagnostic {esc(claim["truth_value"]).replace("_", " ")} · '
        f'{esc(claim["document_support"])} study(s) · {esc(claim["surface_form_count"])} surface form(s)</div>'
        f'<h3>{esc(claim["subj"])} {esc(claim["rel"])} {esc(claim["obj"])}</h3>'
        f'<p>{esc(claim["statement"] or claim["domain"])}</p>'
        f'<p class="trace">Canonical key: {esc(claim["canonical_subj"])} · {esc(claim["canonical_rel"])} · '
        f'{esc(claim["canonical_obj"])}'
        + (f" · Surface forms: {esc(surface_preview)}" if surface_preview else "")
        + (f" · Runs: {esc(runs)}" if runs else "")
        + f' · Best exact support: {esc(claim["exact_document_support_max"])} study(s)</p>'
        "</article>"
    )


def _render_homotopy_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    surface_preview = " | ".join(str(entry) for entry in (item.get("surface_forms") or [])[:3])
    domain_aliases = [
        str(entry).strip()
        for entry in (item.get("domain_aliases") or [])
        if str(entry).strip()
    ]
    alternate_domains = [entry for entry in domain_aliases if entry != str(item.get("domain") or "")]
    return (
        '<article class="homotopy-card">'
        f'<div class="claim-meta">{esc(item.get("coherence_state") or "coherent").replace("_", " ")} · '
        f'{esc(item.get("surface_form_count") or 0)} surface form(s) · '
        f'{esc(item.get("document_support") or 0)} study(s)</div>'
        f'<h3>{esc(item.get("subj") or "")} {esc(item.get("rel") or "")} {esc(item.get("obj") or "")}</h3>'
        + (f'<p>{esc(item.get("statement") or item.get("domain") or "")}</p>' if item.get("statement") or item.get("domain") else "")
        + (
            f'<p class="trace">Canonical key: {esc(item.get("canonical_subj") or "")} · '
            f'{esc(item.get("canonical_rel") or "")} · {esc(item.get("canonical_obj") or "")}</p>'
        )
        + (
            f'<p class="trace">Vertices: {esc(item.get("simplex_vertices") or 0)} · '
            f'Edges: {esc(item.get("simplex_edges") or 0)} · '
            f'Filled triangles: {esc(item.get("simplex_triangles") or 0)} · '
            f'Open horns: {esc(item.get("open_horns") or 0)} · '
            f'Horn-fill ratio: {esc(item.get("horn_fill_ratio") or 0)}</p>'
        )
        + (f'<p class="trace">Surface forms: {esc(surface_preview)}</p>' if surface_preview else "")
        + (
            f'<p class="trace">Also seen under: {esc(" | ".join(alternate_domains[:3]))}</p>'
            if alternate_domains
            else ""
        )
        + "</article>"
    )


def _render_regime_gluing_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    regimes = " | ".join(str(entry) for entry in (item.get("regimes") or [])[:4])
    relations = " | ".join(str(entry) for entry in (item.get("canonical_relations") or [])[:4])
    gluing_state = str(item.get("gluing_state") or "single_regime").replace("_", " ")
    return (
        '<article class="regime-card">'
        f'<div class="claim-meta">{esc(gluing_state)} · {esc(item.get("regime_count") or 0)} regime(s) · '
        f'{esc(item.get("total_document_support") or 0)} total supporting study hits</div>'
        f'<h3>{esc(item.get("canonical_subj") or "")} -> {esc(item.get("canonical_obj") or "")}</h3>'
        + (f'<p class="trace">Regimes: {esc(regimes)}</p>' if regimes else "")
        + (f'<p class="trace">Relation family: {esc(relations)}</p>' if relations else "")
        + (
            f'<p class="trace">Regime variants: {esc(item.get("regime_variant_count") or 0)} · '
            f'Polarity count: {esc(item.get("polarity_count") or 0)} · '
            f'Max single-regime support: {esc(item.get("max_regime_support") or 0)}</p>'
        )
        + "</article>"
    )


def _render_study_card(item: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    links = []
    if item.get("summary_href"):
        links.append(f'<a href="{esc(item["summary_href"])}" target="_blank" rel="noreferrer">Executive summary</a>')
    if item.get("credibility_href"):
        links.append(f'<a href="{esc(item["credibility_href"])}" target="_blank" rel="noreferrer">Credibility report</a>')
    link_markup = " · ".join(links) if links else "Artifacts pending"
    return (
        '<article class="study-card">'
        f'<div class="claim-meta">{esc(item["run_name"])}</div>'
        f'<h3>{esc(item["title"])}</h3>'
        f'<p class="trace">{link_markup}</p>'
        "</article>"
    )


def _render_dashboard_html(
    payload: dict[str, object],
    *,
    dashboard_path: Path,
    batch_outdir: Path,
) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    n_documents = int(payload.get("n_documents") or 0)
    single_document_mode = n_documents <= 1
    equivalence_empty_text = (
        "No internal causal equivalence classes were detected in the current document graph."
        if single_document_mode
        else "No cross-study causal equivalence classes were detected in the current corpus graph."
    )
    equivalence_label = "Internal Causal Equivalence Classes" if single_document_mode else "Causal Equivalence Classes"
    equivalence_chip = "equivalence classes"
    disagreement_empty_text = (
        "No internal polarity conflicts were detected in the current document graph."
        if single_document_mode
        else "No cross-study disagreements were detected in the current corpus graph."
    )
    disagreement_label = "Polarity Conflicts" if single_document_mode else "Disagreements"
    disagreement_chip = "polarity conflicts" if single_document_mode else "disagreement surfaces"
    hero_trace = (
        "The conscious layer commissioned a corpus-level agent after Democritus finished. "
        "For a single recovered document, the synthesized claims are classified by internal support tiers, "
        "diagnostic near-matches, causal equivalence classes, and true polarity conflicts."
        if single_document_mode
        else
        "The conscious layer commissioned a corpus-level agent after Democritus finished. "
        "Instead of a Boolean true/false verdict, the synthesized claims are classified by cross-study support, "
        "partial support, causal equivalence classes, and disagreements across the recovered corpus graph."
    )
    classifier_trace = (
        "`provisional_support` and `weak_support` describe how often the same edge reappeared inside this single document graph. "
        "In a single-document run, the causal-equivalence cards below do not mean two studies disagree; they mean Democritus "
        "produced multiple nearby same-direction relations, such as `causes` versus `increases`, for the same subject/object pair. "
        "The conflict cards below are reserved for genuinely opposed polarity, such as `increases` versus `reduces`."
        if single_document_mode
        else
        "`entailed` means every recovered study supported the same edge. `strong_support` means a majority-level cross-study pattern. "
        "`provisional_support` and `weak_support` mark partial but incomplete agreement. Causal-equivalence cards show where the corpus "
        "found multiple same-direction relation variants for the same backbone claim. Disagreement cards are reserved for opposed polarity."
    )
    equivalence_trace = (
        "These cards collapse homotopically equivalent same-direction relations onto one backbone claim so wording variants do not read like separate final conclusions."
    )
    disagreement_trace = (
        "These cards are reserved for genuine polarity conflicts, where the recovered corpus supports opposed directions for the same subject/object backbone."
    )
    homotopy_trace = (
        "This section localizes causally equivalent mentions by a normalized subject-relation-object key, then measures whether their paraphrase family forms a coherent simplicial patch. Filled triangles indicate multiway agreement; open horns flag wording drift or regime-sensitive gluing failures. The chips below separate within-document paraphrase families from cross-document homotopy classes."
    )
    regime_trace = (
        "These cards read directly from the CSQL bundle's regime-gluing view. They distinguish claims that glue cleanly across multiple canonical regimes from claims whose relation family or polarity changes across regimes, which is where descent starts to fail."
    )

    strong_markup = "".join(_render_claim_card(item) for item in payload.get("strongly_supported") or []) or (
        '<div class="empty">No strongly supported cross-study claims were recovered yet.</div>'
    )
    weak_markup = "".join(_render_claim_card(item) for item in payload.get("weakly_supported") or []) or (
        '<div class="empty">No weakly supported claims were classified.</div>'
    )
    diagnostic_markup = "".join(_render_diagnostic_claim_card(item) for item in payload.get("diagnostic_supported") or []) or (
        '<div class="empty">No additional claims were recovered by the relaxed diagnostic gluing pass.</div>'
    )
    equivalence_markup = "".join(_render_equivalence_card(item) for item in payload.get("equivalence_classes") or []) or (
        f'<div class="empty">{esc(equivalence_empty_text)}</div>'
    )
    disagreement_markup = "".join(_render_disagreement_card(item) for item in payload.get("disagreements") or []) or (
        f'<div class="empty">{esc(disagreement_empty_text)}</div>'
    )
    homotopy_markup = "".join(_render_homotopy_card(item) for item in payload.get("homotopy_classes") or []) or (
        '<div class="empty">No non-trivial homotopy-localized claim classes were detected yet.</div>'
    )
    regime_markup = "".join(_render_regime_gluing_card(item) for item in payload.get("regime_gluing_claims") or []) or (
        '<div class="empty">No cross-regime gluing surfaces were detected yet.</div>'
    )
    study_markup = "".join(_render_study_card(item) for item in payload.get("study_cards") or []) or (
        '<div class="empty">Study cards will appear once document-level artifacts are available.</div>'
    )
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(str(payload.get("query") or ""), route_name="democritus"),
    )
    democritus_gui_href = _relative_href(batch_outdir / "democritus_gui.html", start=dashboard_path.parent)
    csql_summary_href = _relative_href(Path(str(payload.get("csql_sqlite_path") or "")).with_name("democritus_csql_summary.json"), start=dashboard_path.parent)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Democritus Corpus Synthesis</title>
    <style>
      :root {{
        --ink: #15222d;
        --muted: #5a6a77;
        --paper: #f6f1e7;
        --card: rgba(255,255,255,0.88);
        --line: #d8ccb5;
        --accent: #8a3e1d;
        --green: #1d6a52;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(138,62,29,0.12), transparent 24%),
          linear-gradient(180deg, #fbf7ef 0%, var(--paper) 100%);
      }}
      main {{ width: min(1220px, calc(100vw - 32px)); margin: 32px auto 48px; display: grid; gap: 18px; }}
      .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; box-shadow: 0 24px 60px rgba(30,25,18,0.08); }}
      .eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      .hero-grid, .section-grid {{ display: grid; gap: 18px; }}
      .hero-grid {{ grid-template-columns: 1.5fr 1fr; }}
      .section-grid {{ grid-template-columns: 1fr 1fr; }}
      h1, h2, h3, p {{ margin: 0; }}
      .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
      .chip {{ border-radius: 999px; padding: 8px 12px; background: #efe7d9; font-size: 0.92rem; color: #64492b; }}
      .link-row {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 12px; }}
      .link-row a, .trace a {{ color: var(--green); text-decoration: none; font-weight: 700; }}
      .link-row a:hover, .trace a:hover {{ text-decoration: underline; }}
      .claim-grid, .study-grid {{ display: grid; gap: 12px; }}
      .claim-card, .equivalence-card, .disagreement-card, .homotopy-card, .regime-card, .study-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 16px; background: #fffdf9; }}
      .equivalence-card {{ background: #fcf8ef; }}
      .homotopy-card {{ background: #f7f4ff; }}
      .regime-card {{ background: #eef7f2; }}
      .claim-meta, .trace, .variant-meta {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
      .textbook-list {{ padding-left: 20px; display: grid; gap: 10px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      .variant-row {{ display: flex; gap: 10px; align-items: baseline; justify-content: space-between; padding-top: 10px; }}
      .empty {{ color: var(--muted); line-height: 1.6; }}
      @media (max-width: 920px) {{ .hero-grid, .section-grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel hero-grid">
        <div>
          <p class="eyebrow">CLIFF Topos Synthesis</p>
          <h1>{esc(payload.get("query") or "Democritus corpus synthesis")}</h1>
          <p class="trace">{esc(hero_trace)}</p>
          <div class="chip-row">
            <span class="chip">{esc(n_documents)} {"document" if n_documents == 1 else "documents"} glued into one corpus object</span>
            <span class="chip">{esc(len(payload.get("strongly_supported") or []))} strongly supported claims</span>
            <span class="chip">{esc(len(payload.get("weakly_supported") or []))} weak or provisional claims</span>
            <span class="chip">{esc(len(payload.get("diagnostic_supported") or []))} normalized diagnostic claims</span>
            <span class="chip">{esc(len(payload.get("equivalence_classes") or []))} {esc(equivalence_chip)}</span>
            <span class="chip">{esc(len(payload.get("disagreements") or []))} {esc(disagreement_chip)}</span>
            <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("class_count") or 0))} homotopy-localized claim classes</span>
            <span class="chip">{esc(int((payload.get("regime_gluing_summary") or {}).get("surface_count") or 0))} regime-gluing surfaces</span>
          </div>
          <div class="link-row">
            {f'<a href="{esc(democritus_gui_href)}" target="_blank" rel="noreferrer">Open Democritus batch GUI</a>' if democritus_gui_href else ''}
            {f'<a href="{esc(csql_summary_href)}" target="_blank" rel="noreferrer">Open CSQL summary JSON</a>' if csql_summary_href else ''}
          </div>
        </div>
        <div class="panel" style="padding:18px; background:#f8ede0;">
          <p class="eyebrow">Subobject Classifier</p>
          <p class="trace">{esc(classifier_trace)}</p>
        </div>
      </section>
      <section class="panel">
        {textbook_html}
      </section>
      <section class="section-grid">
        <section class="panel">
          <p class="eyebrow">Strongly Supported</p>
          <div class="claim-grid">{strong_markup}</div>
        </section>
        <section class="panel">
          <p class="eyebrow">Weakly Supported</p>
          <div class="claim-grid">{weak_markup}</div>
        </section>
      </section>
      <section class="panel">
        <p class="eyebrow">Normalized Diagnostic Support</p>
        <p class="trace">This relaxed pass ignores per-paper topic domains and merges lightweight language variants so we can see claims that almost glued but were split by wording drift. Treat these as diagnostic evidence, not the primary strict verdict.</p>
        <div class="claim-grid" style="margin-top:12px;">{diagnostic_markup}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Homotopy Localization</p>
        <p class="trace">{esc(homotopy_trace)}</p>
        <div class="chip-row">
          <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("cross_document_class_count") or 0))} cross-document classes</span>
          <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("within_document_class_count") or 0))} within-document families</span>
          <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("coherent_count") or 0))} coherent classes</span>
          <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("partially_glued_count") or 0))} partially glued classes</span>
          <span class="chip">{esc(int((payload.get("homotopy_summary") or {}).get("disconnected_count") or 0))} disconnected classes</span>
        </div>
        <div class="claim-grid" style="margin-top:12px;">{homotopy_markup}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Regime Gluing</p>
        <p class="trace">{esc(regime_trace)}</p>
        <div class="chip-row">
          <span class="chip">{esc(int((payload.get("regime_gluing_summary") or {}).get("multi_regime_glued_count") or 0))} multi-regime glued</span>
          <span class="chip">{esc(int((payload.get("regime_gluing_summary") or {}).get("regime_sensitive_count") or 0))} regime-sensitive</span>
          <span class="chip">{esc(int((payload.get("regime_gluing_summary") or {}).get("obstructed_count") or 0))} obstructed</span>
        </div>
        <div class="claim-grid" style="margin-top:12px;">{regime_markup}</div>
      </section>
      <section class="section-grid">
        <section class="panel">
          <p class="eyebrow">{esc(equivalence_label)}</p>
          <p class="trace" style="margin-bottom:12px;">{esc(equivalence_trace)}</p>
          <div class="claim-grid">{equivalence_markup}</div>
        </section>
        <section class="panel">
          <p class="eyebrow">{esc(disagreement_label)}</p>
          <p class="trace" style="margin-bottom:12px;">{esc(disagreement_trace)}</p>
          <div class="claim-grid">{disagreement_markup}</div>
        </section>
      </section>
      <section class="panel">
        <p class="eyebrow">Study Artifacts</p>
        <div class="study-grid">{study_markup}</div>
      </section>
    </main>
  </body>
</html>"""
