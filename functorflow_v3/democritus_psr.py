"""Causal PSR construction for Democritus document collections.

This module treats each Democritus run/document as a partially observed causal
episode. Relational triples supply the observable causal event stream; local
contexts then give presheaf-style views over the same corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .causal_homotopy import normalize_claim_text, normalize_relation


@dataclass(frozen=True)
class DemocritusPSRSource:
    """One source document worth of Democritus triples."""

    run_name: str
    triples_path: Path
    pdf_path: str = ""


@dataclass(frozen=True)
class DemocritusContextSpec:
    """A local context for interpreting causal histories and tests."""

    context_id: str
    label: str
    domains: frozenset[str]
    parents: tuple[str, ...] = ()
    description: str = ""


def _slug(value: object, *, fallback: str = "unknown", maxlen: int = 64) -> str:
    lowered = re.sub(r"\s+", " ", str(value or "").strip().lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return (cleaned or fallback)[:maxlen]


def _label(value: object, *, fallback: str = "Unknown") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text or fallback


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _relation_token(rel: object) -> str:
    token = _slug(rel, fallback="rel")
    aliases = {
        "leads_to": "causes",
        "leadsto": "causes",
        "increase": "increases",
        "decrease": "reduces",
    }
    return aliases.get(token, token)


def _compact_context_matrix(matrix: dict[str, object]) -> dict[str, object]:
    """Keep the main TWM bundle inspectable without dropping PSR semantics."""

    entries = [dict(row) for row in list(matrix.get("entries") or [])]
    strongest_entries = sorted(
        entries,
        key=lambda row: (
            float(row.get("probability") or 0.0),
            int(row.get("matches") or 0),
            int(row.get("history_support") or 0),
        ),
        reverse=True,
    )[:24]
    return {
        "context_id": matrix.get("context_id"),
        "context_label": matrix.get("context_label"),
        "parents": list(matrix.get("parents") or []),
        "description": matrix.get("description"),
        "n_episode_views": int(matrix.get("n_episode_views") or 0),
        "n_histories": len(list(matrix.get("histories") or [])),
        "n_tests": len(list(matrix.get("tests") or [])),
        "rank": int(matrix.get("rank") or 0),
        "top_histories": list(matrix.get("histories") or [])[:8],
        "top_tests": list(matrix.get("tests") or [])[:12],
        "strongest_test_cells": strongest_entries,
    }


def _claim_test_witnesses(
    *,
    episodes: list[dict[str, object]],
    matrix_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    event_index: dict[tuple[str, str], dict[str, object]] = {}
    for episode in episodes:
        for raw_event in list(episode.get("events") or []):
            event = dict(raw_event)
            action = str(event.get("action") or "")
            observation = str(event.get("observation") or "")
            event_index[(action, observation)] = {
                "canonical_subj": normalize_claim_text(str(event.get("subject") or "")),
                "canonical_rel": normalize_relation(str(event.get("relation") or "")),
                "canonical_obj": normalize_claim_text(str(event.get("object") or "")),
                "canonical_domain": normalize_claim_text(str(event.get("domain") or "")),
                "domain": str(event.get("domain") or ""),
                "action": action,
                "observation": observation,
                "edge_signature": str(event.get("edge_signature") or ""),
                "statement": str(event.get("statement") or ""),
            }

    witnesses: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for context_id, matrix in matrix_lookup.items():
        for raw_entry in list(matrix.get("entries") or []):
            entry = dict(raw_entry)
            probability = float(entry.get("probability") or 0.0)
            if probability <= 0.0:
                continue
            history_tokens = str(entry.get("history_signature") or "").split()
            test_tokens = str(entry.get("test_signature") or "").split()
            for action in history_tokens:
                for observation in test_tokens:
                    event = event_index.get((action, observation))
                    if event is None:
                        continue
                    key = (
                        str(event["canonical_subj"]),
                        str(event["canonical_rel"]),
                        str(event["canonical_obj"]),
                        context_id,
                    )
                    current = witnesses.get(key)
                    score = (
                        probability,
                        int(entry.get("matches") or 0),
                        int(entry.get("history_support") or 0),
                    )
                    previous_score = (
                        float(current.get("test_probability") or 0.0),
                        int(current.get("matches") or 0),
                        int(current.get("history_support") or 0),
                    ) if current else (-1.0, -1, -1)
                    if score <= previous_score:
                        continue
                    witnesses[key] = {
                        **event,
                        "context_id": context_id,
                        "context_label": matrix.get("context_label"),
                        "history_signature": entry.get("history_signature"),
                        "test_signature": entry.get("test_signature"),
                        "test_probability": round(probability, 6),
                        "matches": int(entry.get("matches") or 0),
                        "history_support": int(entry.get("history_support") or 0),
                        "semantics": "psr_test_cell_witness_v1",
                    }
    return sorted(
        witnesses.values(),
        key=lambda row: (
            -float(row.get("test_probability") or 0.0),
            -int(row.get("matches") or 0),
            str(row.get("canonical_subj") or ""),
            str(row.get("canonical_obj") or ""),
        ),
    )


def _causal_event(row: dict[str, object], *, index: int) -> dict[str, object]:
    subj = _label(row.get("subj") or row.get("src") or row.get("source"), fallback=f"source {index}")
    obj = _label(row.get("obj") or row.get("dst") or row.get("target"), fallback=f"target {index}")
    rel = _relation_token(row.get("rel") or row.get("relation") or "affects")
    domain = _label(row.get("domain") or row.get("topic") or row.get("context"), fallback="general")
    topic_path = tuple(_label(part) for part in (row.get("path") or ()) if _label(part))
    return {
        "event_id": f"event_{index:04d}",
        "subject": subj,
        "relation": rel,
        "object": obj,
        "domain": domain,
        "domain_id": _slug(domain, fallback="general"),
        "topic_path": list(topic_path),
        "topic_id": _slug(topic_path[-1] if topic_path else domain, fallback="general"),
        "action": f"intervene:{_slug(subj, fallback='source')}",
        "observation": f"{rel}:{_slug(obj, fallback='target')}",
        "edge_signature": f"{_slug(subj, fallback='source')}->{rel}->{_slug(obj, fallback='target')}",
        "statement": str(row.get("statement") or ""),
    }


def build_democritus_psr_episodes(sources: Iterable[DemocritusPSRSource]) -> list[dict[str, object]]:
    """Collapse Democritus triples into one causal episode per source document."""

    episodes: list[dict[str, object]] = []
    for source in sources:
        rows = _read_jsonl(source.triples_path)
        events = [_causal_event(row, index=index) for index, row in enumerate(rows, start=1)]
        if not events:
            continue
        domains = sorted({str(event["domain_id"]) for event in events})
        topics = sorted({str(event["topic_id"]) for event in events})
        action_observation_stream: list[str] = []
        for event in events:
            action_observation_stream.append(str(event["action"]))
            action_observation_stream.append(str(event["observation"]))
        episodes.append(
            {
                "episode_id": source.run_name,
                "run_name": source.run_name,
                "pdf_path": source.pdf_path,
                "triples_path": str(source.triples_path),
                "domains": domains,
                "topics": topics,
                "events": events,
                "stream": action_observation_stream,
            }
        )
    return episodes


def _context_specs(episodes: Iterable[dict[str, object]], *, max_domain_contexts: int) -> tuple[DemocritusContextSpec, ...]:
    domain_counts: Counter[str] = Counter()
    domain_labels: dict[str, str] = {}
    for episode in episodes:
        for event in list(episode.get("events") or []):
            row = dict(event)
            domain_id = str(row.get("domain_id") or "general")
            domain_counts[domain_id] += 1
            domain_labels.setdefault(domain_id, _label(row.get("domain"), fallback=domain_id))
    specs = [
        DemocritusContextSpec(
            context_id="corpus",
            label="Corpus",
            domains=frozenset({"*"}),
            description="Global Democritus causal-PSR context over all documents.",
        )
    ]
    for domain_id, _ in domain_counts.most_common(max(0, max_domain_contexts)):
        specs.append(
            DemocritusContextSpec(
                context_id=f"domain::{domain_id}",
                label=domain_labels.get(domain_id, domain_id),
                domains=frozenset({domain_id}),
                parents=("corpus",),
                description="Local domain view of the Democritus causal event stream.",
            )
        )
    return tuple(specs)


def _stream_for_context(episode: dict[str, object], context: DemocritusContextSpec) -> tuple[str, ...]:
    if "*" in context.domains:
        return tuple(str(item) for item in episode.get("stream") or ())
    stream: list[str] = []
    for event in list(episode.get("events") or []):
        row = dict(event)
        if str(row.get("domain_id") or "") not in context.domains:
            continue
        stream.append(str(row.get("action") or ""))
        stream.append(str(row.get("observation") or ""))
    return tuple(item for item in stream if item)


def _iter_prefixes(sequence: tuple[str, ...], *, max_history_length: int) -> Iterable[tuple[str, ...]]:
    yield ()
    for size in range(1, min(len(sequence), max_history_length) + 1):
        yield sequence[:size]


def _iter_tests(sequence: tuple[str, ...], *, max_test_length: int) -> Iterable[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    for start in range(len(sequence)):
        for size in range(1, max_test_length + 1):
            motif = sequence[start : start + size]
            if len(motif) != size or motif in seen:
                continue
            seen.add(motif)
            yield motif


def _contains_subsequence(sequence: tuple[str, ...], motif: tuple[str, ...], *, start_index: int = 0) -> bool:
    if not motif:
        return True
    for index in range(max(0, start_index), len(sequence) - len(motif) + 1):
        if sequence[index : index + len(motif)] == motif:
            return True
    return False


def _rank(matrix: list[list[float]], *, tolerance: float = 1e-9) -> int:
    rows = [list(row) for row in matrix if row]
    if not rows:
        return 0
    n_rows = len(rows)
    n_cols = len(rows[0])
    rank = 0
    pivot_row = 0
    for col in range(n_cols):
        pivot = None
        for row_index in range(pivot_row, n_rows):
            if abs(rows[row_index][col]) > tolerance:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][col]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for row_index in range(n_rows):
            if row_index == pivot_row:
                continue
            factor = rows[row_index][col]
            if abs(factor) <= tolerance:
                continue
            rows[row_index] = [current - factor * pivot_current for current, pivot_current in zip(rows[row_index], rows[pivot_row])]
        rank += 1
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    return rank


def _matrix_for_context(
    *,
    context: DemocritusContextSpec,
    episode_streams: list[tuple[str, ...]],
    max_history_length: int,
    max_test_length: int,
    min_support: int,
    max_histories: int,
    max_tests: int,
) -> dict[str, object] | None:
    nonempty_streams = [stream for stream in episode_streams if stream]
    if not nonempty_streams:
        return None
    history_support: Counter[tuple[str, ...]] = Counter()
    test_support: Counter[tuple[str, ...]] = Counter()
    for stream in nonempty_streams:
        history_support.update(set(_iter_prefixes(stream, max_history_length=max_history_length)))
        test_support.update(set(_iter_tests(stream, max_test_length=max_test_length)))
    histories = [
        history
        for history, count in sorted(history_support.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        if history == () or count >= min_support
    ][:max_histories]
    tests = [
        test
        for test, count in sorted(test_support.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        if count >= min_support
    ][:max_tests]
    matrix: list[list[float]] = []
    entries: list[dict[str, object]] = []
    for history in histories:
        matching = [stream for stream in nonempty_streams if not history or stream[: len(history)] == history]
        denominator = len(matching)
        row_values: list[float] = []
        for test in tests:
            numerator = sum(1 for stream in matching if _contains_subsequence(stream, test, start_index=len(history))) if denominator else 0
            probability = round(float(numerator) / float(denominator), 6) if denominator else 0.0
            row_values.append(probability)
            entries.append(
                {
                    "history_signature": " ".join(history) if history else "epsilon",
                    "test_signature": " ".join(test),
                    "probability": probability,
                    "matches": int(numerator),
                    "history_support": int(denominator),
                }
            )
        matrix.append(row_values)
    return {
        "context_id": context.context_id,
        "context_label": context.label,
        "parents": list(context.parents),
        "description": context.description,
        "n_episode_views": len(nonempty_streams),
        "histories": [
            {"signature": " ".join(history) if history else "epsilon", "tokens": list(history), "support": int(history_support[history])}
            for history in histories
        ],
        "tests": [
            {"signature": " ".join(test), "tokens": list(test), "support": int(test_support[test])}
            for test in tests
        ],
        "matrix": matrix,
        "entries": entries,
        "rank": _rank(matrix),
    }


def build_democritus_topos_psr_bundle(
    sources: Iterable[DemocritusPSRSource],
    *,
    corpus_label: str = "Democritus corpus",
    max_history_length: int = 4,
    max_test_length: int = 3,
    min_support: int = 1,
    max_histories_per_context: int = 16,
    max_tests_per_context: int = 24,
    max_domain_contexts: int = 8,
    glue_tolerance: float = 0.20,
) -> dict[str, object]:
    """Build a presheaf-style causal PSR bundle from Democritus triples."""

    source_tuple = tuple(sources)
    episodes = build_democritus_psr_episodes(source_tuple)
    contexts = _context_specs(episodes, max_domain_contexts=max_domain_contexts)
    local_hankel_family: list[dict[str, object]] = []
    matrix_lookup: dict[str, dict[str, object]] = {}
    for context in contexts:
        streams = [_stream_for_context(episode, context) for episode in episodes]
        matrix = _matrix_for_context(
            context=context,
            episode_streams=streams,
            max_history_length=max_history_length,
            max_test_length=max_test_length,
            min_support=min_support,
            max_histories=max_histories_per_context,
            max_tests=max_tests_per_context,
        )
        if matrix is None:
            continue
        local_hankel_family.append(matrix)
        matrix_lookup[context.context_id] = matrix

    restriction_diagnostics: list[dict[str, object]] = []
    for context in contexts:
        child = matrix_lookup.get(context.context_id)
        if child is None:
            continue
        for parent_id in context.parents:
            parent = matrix_lookup.get(parent_id)
            if parent is None:
                continue
            child_entries = {
                (str(row["history_signature"]), str(row["test_signature"])): float(row["probability"])
                for row in list(child.get("entries") or [])
            }
            parent_entries = {
                (str(row["history_signature"]), str(row["test_signature"])): float(row["probability"])
                for row in list(parent.get("entries") or [])
            }
            shared = sorted(set(child_entries) & set(parent_entries))
            gaps = [abs(child_entries[key] - parent_entries[key]) for key in shared]
            max_gap = max(gaps) if gaps else 0.0
            mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
            restriction_diagnostics.append(
                {
                    "source_context": parent_id,
                    "target_context": context.context_id,
                    "shared_cells": len(shared),
                    "mean_abs_gap": round(mean_gap, 6),
                    "max_abs_gap": round(max_gap, 6),
                    "compatible": bool(max_gap <= glue_tolerance),
                    "restriction_rule": "compare shared history/test probabilities after restricting the corpus stream to the local domain",
                }
            )

    ranks = [int(row.get("rank", 0)) for row in local_hankel_family]
    claim_test_witnesses = _claim_test_witnesses(
        episodes=episodes,
        matrix_lookup=matrix_lookup,
    )
    episode_index = [
        {
            "episode_id": episode.get("episode_id"),
            "run_name": episode.get("run_name"),
            "pdf_path": episode.get("pdf_path"),
            "triples_path": episode.get("triples_path"),
            "event_count": len(list(episode.get("events") or [])),
            "domain_count": len(list(episode.get("domains") or [])),
            "topic_count": len(list(episode.get("topics") or [])),
        }
        for episode in episodes
    ]
    return {
        "bundle_type": "democritus_causal_psr",
        "corpus_label": corpus_label,
        "episode_index": episode_index,
        "contexts": [
            {
                "context_id": context.context_id,
                "label": context.label,
                "parents": list(context.parents),
                "description": context.description,
            }
            for context in contexts
            if context.context_id in matrix_lookup
        ],
        "local_hankel_family": [_compact_context_matrix(matrix) for matrix in local_hankel_family],
        "restriction_diagnostics": restriction_diagnostics,
        "claim_test_witnesses": claim_test_witnesses[:256],
        "summary": {
            "corpus_label": corpus_label,
            "n_sources": len(source_tuple),
            "n_episodes": len(episodes),
            "n_contexts": len(local_hankel_family),
            "n_events": sum(len(list(episode.get("events") or [])) for episode in episodes),
            "n_psr_test_witnesses": len(claim_test_witnesses),
            "n_restriction_checks": len(restriction_diagnostics),
            "n_compatible_restrictions": sum(1 for row in restriction_diagnostics if bool(row.get("compatible"))),
            "mean_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
            "max_rank": max(ranks) if ranks else 0,
            "config": {
                "max_history_length": max_history_length,
                "max_test_length": max_test_length,
                "min_support": min_support,
                "max_histories_per_context": max_histories_per_context,
                "max_tests_per_context": max_tests_per_context,
                "max_domain_contexts": max_domain_contexts,
                "glue_tolerance": glue_tolerance,
                "artifact_layout": "compact_hankel_json_plus_episode_jsonl_sidecar",
            },
        },
    }


def write_democritus_topos_psr_bundle(
    *,
    batch_outdir: Path,
    sources: Iterable[DemocritusPSRSource],
    corpus_label: str = "Democritus corpus",
) -> Path | None:
    """Materialize the Democritus causal PSR bundle under a batch output dir."""

    source_tuple = tuple(sources)
    if not source_tuple:
        return None
    bundle = build_democritus_topos_psr_bundle(source_tuple, corpus_label=corpus_label)
    if int(dict(bundle.get("summary") or {}).get("n_episodes", 0)) <= 0:
        return None
    psr_dir = batch_outdir / "topos_psr"
    psr_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = psr_dir / "democritus_topos_psr_hankel.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    episodes_path = psr_dir / "democritus_psr_episodes.jsonl"
    episodes = build_democritus_psr_episodes(source_tuple)
    with episodes_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")
    return bundle_path
