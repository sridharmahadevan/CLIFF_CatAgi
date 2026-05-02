"""Topos-style Observer Operator Models over token streams.

Unlike the product-review PSR construction, this module does not require an
action parser. It treats tokens as observations and builds local Hankel tables
and observation-update operators on a simple textual cover.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
import json
import math
import re
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    np = None  # type: ignore[assignment]


TOPOS_OOM_VERSION = 1


@dataclass(frozen=True)
class OOMContextSpec:
    context_id: str
    label: str
    parents: tuple[str, ...] = ()
    description: str = ""


_DEFAULT_CONTEXTS: tuple[OOMContextSpec, ...] = (
    OOMContextSpec(
        context_id="corpus",
        label="Corpus",
        parents=(),
        description="Global observation context over every token stream.",
    ),
    OOMContextSpec(
        context_id="document",
        label="Document",
        parents=("corpus",),
        description="Document-level observation context.",
    ),
    OOMContextSpec(
        context_id="paragraph",
        label="Paragraph",
        parents=("document",),
        description="Paragraph-level local observation context.",
    ),
    OOMContextSpec(
        context_id="sentence",
        label="Sentence",
        parents=("paragraph",),
        description="Sentence-level local observation context.",
    ),
)


def _tokenize(text: str, *, lowercase: bool = True) -> tuple[str, ...]:
    value = text.lower() if lowercase else text
    return tuple(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", value))


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _normalize_documents(documents: Iterable[object] | object) -> list[dict[str, object]]:
    if isinstance(documents, str):
        raw_documents = [documents]
    elif isinstance(documents, dict):
        raw_documents = [documents]
    else:
        raw_documents = list(documents)

    normalized: list[dict[str, object]] = []
    for index, item in enumerate(raw_documents, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("body") or "")
            doc_id = str(item.get("document_id") or item.get("id") or f"document_{index:04d}")
            title = str(item.get("title") or doc_id)
            metadata = dict(item.get("metadata") or {})
        else:
            text = str(item)
            doc_id = f"document_{index:04d}"
            title = doc_id
            metadata = {}
        normalized.append(
            {
                "document_id": doc_id,
                "title": title,
                "text": text,
                "metadata": metadata,
            }
        )
    return normalized


def _document_token_sequences(documents: Iterable[object] | object, *, lowercase: bool = True) -> list[tuple[str, ...]]:
    return [
        _tokenize(str(document.get("text") or ""), lowercase=lowercase)
        for document in _normalize_documents(documents)
        if str(document.get("text") or "").strip()
    ]


def _context_views(documents: list[dict[str, object]], *, lowercase: bool) -> dict[str, list[dict[str, object]]]:
    views: dict[str, list[dict[str, object]]] = {context.context_id: [] for context in _DEFAULT_CONTEXTS}
    for doc_index, document in enumerate(documents, start=1):
        doc_id = str(document["document_id"])
        text = str(document["text"])
        doc_tokens = _tokenize(text, lowercase=lowercase)
        if doc_tokens:
            views["corpus"].append({"view_id": f"corpus::{doc_id}", "document_id": doc_id, "tokens": doc_tokens})
            views["document"].append({"view_id": f"document::{doc_id}", "document_id": doc_id, "tokens": doc_tokens})
        for paragraph_index, paragraph in enumerate(_split_paragraphs(text), start=1):
            paragraph_tokens = _tokenize(paragraph, lowercase=lowercase)
            if paragraph_tokens:
                paragraph_id = f"{doc_id}::p{paragraph_index:03d}"
                views["paragraph"].append(
                    {"view_id": f"paragraph::{paragraph_id}", "document_id": doc_id, "tokens": paragraph_tokens}
                )
            for sentence_index, sentence in enumerate(_split_sentences(paragraph), start=1):
                sentence_tokens = _tokenize(sentence, lowercase=lowercase)
                if sentence_tokens:
                    views["sentence"].append(
                        {
                            "view_id": f"sentence::{doc_id}::p{paragraph_index:03d}::s{sentence_index:03d}",
                            "document_id": doc_id,
                            "tokens": sentence_tokens,
                        }
                    )
    return views


def _ngrams(tokens: tuple[str, ...], min_length: int, max_length: int) -> Iterable[tuple[str, ...]]:
    if max_length <= 0:
        return
    upper = min(max_length, len(tokens))
    for length in range(min_length, upper + 1):
        for start in range(0, len(tokens) - length + 1):
            yield tokens[start : start + length]


def _history_occurrence_starts(tokens: tuple[str, ...], history: tuple[str, ...]) -> list[int]:
    if not history:
        return list(range(0, len(tokens) + 1))
    length = len(history)
    return [start for start in range(0, len(tokens) - length + 1) if tokens[start : start + length] == history]


def _test_follows(tokens: tuple[str, ...], *, history: tuple[str, ...], start: int, test: tuple[str, ...]) -> bool:
    offset = start + len(history)
    return bool(test) and tokens[offset : offset + len(test)] == test


def _signature(tokens: tuple[str, ...]) -> str:
    return " ".join(tokens) if tokens else "epsilon"


def _mapped_sequences(
    sequences: list[tuple[str, ...]],
    *,
    vocabulary: set[str],
) -> list[tuple[str, ...]]:
    return [tuple(token if token in vocabulary else "<unk>" for token in sequence) for sequence in sequences]


def _history_for_position(sequence: tuple[str, ...], index: int, order: int) -> tuple[str, ...]:
    if order <= 0:
        return ()
    start = max(0, index - order)
    return sequence[start:index]


def evaluate_topos_oom_perplexity(
    train_documents: Iterable[object] | object,
    eval_documents: Iterable[object] | object,
    *,
    lowercase: bool = True,
    max_history_length: int = 5,
    alpha: float = 0.1,
    min_token_count: int = 1,
    max_eval_tokens: int | None = None,
) -> dict[str, object]:
    """Evaluate next-observation prediction for empirical OOM histories.

    The curve is intentionally transparent: order k estimates P(x_t | x_{t-k:t})
    from observation histories only, with additive smoothing and an <unk> bucket.
    """

    train_sequences = _document_token_sequences(train_documents, lowercase=lowercase)
    eval_sequences_raw = _document_token_sequences(eval_documents, lowercase=lowercase)
    token_counts: Counter[str] = Counter()
    for sequence in train_sequences:
        token_counts.update(sequence)
    vocabulary = {token for token, count in token_counts.items() if count >= max(1, int(min_token_count))}
    vocabulary.add("<unk>")
    train_sequences = _mapped_sequences(train_sequences, vocabulary=vocabulary)
    eval_sequences = _mapped_sequences(eval_sequences_raw, vocabulary=vocabulary)
    max_order = max(0, int(max_history_length))
    smoothing = max(1e-12, float(alpha))
    vocab_size = max(1, len(vocabulary))

    curve: list[dict[str, object]] = []
    for order in range(0, max_order + 1):
        history_counts: Counter[tuple[str, ...]] = Counter()
        next_counts: Counter[tuple[tuple[str, ...], str]] = Counter()
        for sequence in train_sequences:
            for index, token in enumerate(sequence):
                history = _history_for_position(sequence, index, order)
                history_counts[history] += 1
                next_counts[(history, token)] += 1

        log_loss = 0.0
        evaluated = 0
        unknown_targets = 0
        unseen_histories = 0
        for sequence in eval_sequences:
            for index, token in enumerate(sequence):
                if max_eval_tokens is not None and evaluated >= max_eval_tokens:
                    break
                history = _history_for_position(sequence, index, order)
                denominator = history_counts.get(history, 0)
                if denominator <= 0:
                    unseen_histories += 1
                if token == "<unk>":
                    unknown_targets += 1
                probability = (next_counts.get((history, token), 0) + smoothing) / (
                    denominator + smoothing * vocab_size
                )
                log_loss -= math.log(max(probability, 1e-300))
                evaluated += 1
            if max_eval_tokens is not None and evaluated >= max_eval_tokens:
                break
        average_nll = log_loss / evaluated if evaluated else 0.0
        perplexity = math.exp(average_nll) if evaluated else 0.0
        curve.append(
            {
                "history_length": order,
                "eval_tokens": evaluated,
                "negative_log_likelihood": round(average_nll, 6),
                "perplexity": round(perplexity, 6),
                "unseen_history_rate": round(unseen_histories / evaluated, 6) if evaluated else 0.0,
                "unknown_target_rate": round(unknown_targets / evaluated, 6) if evaluated else 0.0,
                "history_count": len(history_counts),
            }
        )

    best = min(curve, key=lambda row: float(row.get("perplexity") or float("inf"))) if curve else {}
    return {
        "semantics": "topos_oom_next_observation_perplexity_v1",
        "train_documents": len(train_sequences),
        "eval_documents": len(eval_sequences),
        "train_tokens": sum(len(sequence) for sequence in train_sequences),
        "eval_tokens_available": sum(len(sequence) for sequence in eval_sequences),
        "vocabulary_size": vocab_size,
        "alpha": smoothing,
        "min_token_count": max(1, int(min_token_count)),
        "max_eval_tokens": max_eval_tokens,
        "curve": curve,
        "best": best,
    }


def _matrix_rank_fallback(matrix: list[list[float]], *, tolerance: float = 1e-9) -> int:
    rows = [list(map(float, row)) for row in matrix if row]
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


def _svd_summary(matrix: list[list[float]]) -> dict[str, object]:
    if not matrix or not matrix[0]:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    if np is None:
        return {"rank": _matrix_rank_fallback(matrix), "singular_values": [], "energy_captured_top3": 0.0}
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    _, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    if singular_values.size == 0:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    tolerance = max(1e-9, float(singular_values[0]) * 1e-6)
    rank = int(sum(float(value) > tolerance for value in singular_values))
    squared = singular_values ** 2
    total_energy = float(np.sum(squared))
    top_energy = float(np.sum(squared[:3])) if total_energy > 0.0 else 0.0
    return {
        "rank": rank,
        "singular_values": [round(float(value), 6) for value in singular_values[:10]],
        "energy_captured_top3": round(top_energy / total_energy, 6) if total_energy > 0.0 else 0.0,
    }


def _history_rows(
    views: list[dict[str, object]],
    *,
    context_id: str,
    max_history_length: int,
    min_support: int,
    max_histories: int,
) -> list[dict[str, object]]:
    support: Counter[tuple[str, ...]] = Counter()
    support[()] = len(views)
    for view in views:
        tokens = tuple(str(token) for token in view["tokens"])
        support.update(_ngrams(tokens, 1, max_history_length))
    ranked = [
        (history, count)
        for history, count in sorted(support.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        if history == () or count >= min_support
    ]
    rows: list[dict[str, object]] = []
    for history, count in ranked[:max_histories]:
        signature = _signature(history)
        rows.append(
            {
                "history_id": f"{context_id}::history::{signature}",
                "signature": signature,
                "observations": list(history),
                "support": int(count),
            }
        )
    return rows


def _test_rows(
    views: list[dict[str, object]],
    *,
    context_id: str,
    max_test_length: int,
    min_support: int,
    max_tests: int,
) -> list[dict[str, object]]:
    support: Counter[tuple[str, ...]] = Counter()
    for view in views:
        tokens = tuple(str(token) for token in view["tokens"])
        support.update(_ngrams(tokens, 1, max_test_length))
    rows: list[dict[str, object]] = []
    for test, count in sorted(support.items(), key=lambda item: (-item[1], len(item[0]), item[0])):
        if count < min_support:
            continue
        signature = _signature(test)
        rows.append(
            {
                "test_id": f"{context_id}::test::{signature}",
                "signature": signature,
                "observations": list(test),
                "support": int(count),
            }
        )
        if len(rows) >= max_tests:
            break
    return rows


def _history_denominator(views: list[dict[str, object]], history: tuple[str, ...]) -> int:
    return sum(len(_history_occurrence_starts(tuple(str(token) for token in view["tokens"]), history)) for view in views)


def _local_hankel_matrix(
    context: OOMContextSpec,
    views: list[dict[str, object]],
    *,
    max_history_length: int,
    max_test_length: int,
    min_support: int,
    max_histories: int,
    max_tests: int,
    max_operator_observations: int,
) -> dict[str, object] | None:
    if not views:
        return None
    histories = _history_rows(
        views,
        context_id=context.context_id,
        max_history_length=max_history_length,
        min_support=min_support,
        max_histories=max_histories,
    )
    tests = _test_rows(
        views,
        context_id=context.context_id,
        max_test_length=max_test_length,
        min_support=min_support,
        max_tests=max_tests,
    )
    if not histories or not tests:
        return None

    history_tokens = [tuple(str(token) for token in row.get("observations") or ()) for row in histories]
    test_tokens = [tuple(str(token) for token in row.get("observations") or ()) for row in tests]
    matrix: list[list[float]] = []
    entries: list[dict[str, object]] = []
    for history, history_row in zip(history_tokens, histories):
        denominator = _history_denominator(views, history)
        row_values: list[float] = []
        for test, test_row in zip(test_tokens, tests):
            numerator = 0
            for view in views:
                tokens = tuple(str(token) for token in view["tokens"])
                for start in _history_occurrence_starts(tokens, history):
                    if _test_follows(tokens, history=history, start=start, test=test):
                        numerator += 1
            probability = round(float(numerator) / float(denominator), 6) if denominator else 0.0
            row_values.append(probability)
            entries.append(
                {
                    "history_signature": history_row["signature"],
                    "test_signature": test_row["signature"],
                    "probability": probability,
                    "matches": int(numerator),
                    "history_support": int(denominator),
                }
            )
        matrix.append(row_values)

    operators = _observation_operators(
        views,
        histories=histories,
        max_history_length=max_history_length,
        max_operator_observations=max_operator_observations,
    )
    return {
        "context_id": context.context_id,
        "context_label": context.label,
        "description": context.description,
        "parents": list(context.parents),
        "n_observation_views": len(views),
        "token_count": sum(len(tuple(view["tokens"])) for view in views),
        "histories": histories,
        "tests": tests,
        "matrix": matrix,
        "entries": entries,
        "svd": _svd_summary(matrix),
        "observation_operators": operators,
    }


def _observation_operators(
    views: list[dict[str, object]],
    *,
    histories: list[dict[str, object]],
    max_history_length: int,
    max_operator_observations: int,
) -> list[dict[str, object]]:
    observation_support: Counter[str] = Counter()
    for view in views:
        observation_support.update(str(token) for token in view["tokens"])
    history_tokens = [tuple(str(token) for token in row.get("observations") or ()) for row in histories]
    history_index = {_signature(history): index for index, history in enumerate(history_tokens)}
    operators: list[dict[str, object]] = []
    for observation, support in sorted(observation_support.items(), key=lambda item: (-item[1], item[0]))[:max_operator_observations]:
        matrix = [[0.0 for _ in histories] for _ in histories]
        entries: list[dict[str, object]] = []
        for row_index, history in enumerate(history_tokens):
            denominator = _history_denominator(views, history)
            if denominator <= 0:
                continue
            transitions: Counter[str] = Counter()
            for view in views:
                tokens = tuple(str(token) for token in view["tokens"])
                for start in _history_occurrence_starts(tokens, history):
                    next_index = start + len(history)
                    if next_index >= len(tokens) or tokens[next_index] != observation:
                        continue
                    next_history = (*history, observation)[-max_history_length:] if max_history_length > 0 else ()
                    next_signature = _signature(next_history)
                    if next_signature in history_index:
                        transitions[next_signature] += 1
            for next_signature, count in transitions.items():
                col_index = history_index[next_signature]
                probability = round(float(count) / float(denominator), 6)
                matrix[row_index][col_index] = probability
                entries.append(
                    {
                        "source_history": _signature(history),
                        "target_history": next_signature,
                        "observation": observation,
                        "probability": probability,
                        "matches": int(count),
                        "history_support": int(denominator),
                    }
                )
        operators.append(
            {
                "observation": observation,
                "support": int(support),
                "matrix": matrix,
                "entries": entries,
            }
        )
    return operators


def _restriction_diagnostics(
    context_specs: tuple[OOMContextSpec, ...],
    matrix_lookup: dict[str, dict[str, object]],
    *,
    glue_tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in context_specs:
        child = matrix_lookup.get(context.context_id)
        if child is None:
            continue
        child_entries = {
            (str(row["history_signature"]), str(row["test_signature"])): float(row["probability"])
            for row in list(child.get("entries") or [])
        }
        for parent_id in context.parents:
            parent = matrix_lookup.get(parent_id)
            if parent is None:
                continue
            parent_entries = {
                (str(row["history_signature"]), str(row["test_signature"])): float(row["probability"])
                for row in list(parent.get("entries") or [])
            }
            shared = sorted(set(child_entries) & set(parent_entries))
            gaps = [abs(child_entries[key] - parent_entries[key]) for key in shared]
            max_gap = max(gaps) if gaps else 0.0
            mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
            rows.append(
                {
                    "source_context": parent_id,
                    "target_context": context.context_id,
                    "shared_cells": len(shared),
                    "shared_histories": sorted({history for history, _test in shared}),
                    "shared_tests": sorted({test for _history, test in shared}),
                    "mean_abs_gap": round(mean_gap, 6),
                    "max_abs_gap": round(max_gap, 6),
                    "compatible": bool(max_gap <= glue_tolerance),
                    "restriction_rule": "restrict observation histories and tests to signatures visible in the child chart",
                }
            )
    return rows


def build_topos_oom_bundle(
    documents: Iterable[object] | object,
    *,
    corpus_label: str = "Token corpus",
    lowercase: bool = True,
    max_history_length: int = 3,
    max_test_length: int = 3,
    min_support: int = 1,
    max_histories_per_context: int = 16,
    max_tests_per_context: int = 24,
    max_operator_observations: int = 8,
    glue_tolerance: float = 0.20,
) -> dict[str, object]:
    """Build a sheaf-valued OOM bundle from arbitrary text-like inputs."""

    normalized_documents = _normalize_documents(documents)
    views_by_context = _context_views(normalized_documents, lowercase=lowercase)
    local_hankel_family: list[dict[str, object]] = []
    matrix_lookup: dict[str, dict[str, object]] = {}
    for context in _DEFAULT_CONTEXTS:
        matrix = _local_hankel_matrix(
            context,
            views_by_context.get(context.context_id, []),
            max_history_length=max_history_length,
            max_test_length=max_test_length,
            min_support=min_support,
            max_histories=max_histories_per_context,
            max_tests=max_tests_per_context,
            max_operator_observations=max_operator_observations,
        )
        if matrix is None:
            continue
        local_hankel_family.append(matrix)
        matrix_lookup[context.context_id] = matrix

    restriction_rows = _restriction_diagnostics(_DEFAULT_CONTEXTS, matrix_lookup, glue_tolerance=glue_tolerance)
    ranks = [int(dict(row.get("svd") or {}).get("rank") or 0) for row in local_hankel_family]
    context_view_count = sum(int(row.get("n_observation_views") or 0) for row in local_hankel_family)
    token_count = sum(len(_tokenize(str(document.get("text") or ""), lowercase=lowercase)) for document in normalized_documents)
    summary = {
        "bundle_type": "topos_observer_operator_model",
        "builder_version": TOPOS_OOM_VERSION,
        "corpus_label": corpus_label,
        "n_documents": len(normalized_documents),
        "n_tokens": token_count,
        "n_contexts": len(local_hankel_family),
        "context_ids": [str(row["context_id"]) for row in local_hankel_family],
        "n_context_projected_views": context_view_count,
        "n_restriction_checks": len(restriction_rows),
        "n_compatible_restrictions": sum(1 for row in restriction_rows if bool(row.get("compatible"))),
        "mean_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
        "max_rank": max(ranks) if ranks else 0,
        "config": {
            "lowercase": lowercase,
            "max_history_length": max_history_length,
            "max_test_length": max_test_length,
            "min_support": min_support,
            "max_histories_per_context": max_histories_per_context,
            "max_tests_per_context": max_tests_per_context,
            "max_operator_observations": max_operator_observations,
            "glue_tolerance": glue_tolerance,
        },
    }
    return {
        "bundle_type": "topos_observer_operator_model",
        "documents": [
            {
                "document_id": document["document_id"],
                "title": document["title"],
                "token_count": len(_tokenize(str(document.get("text") or ""), lowercase=lowercase)),
                "metadata": document["metadata"],
            }
            for document in normalized_documents
        ],
        "contexts": [
            {
                "context_id": context.context_id,
                "label": context.label,
                "parents": list(context.parents),
                "description": context.description,
            }
            for context in _DEFAULT_CONTEXTS
            if context.context_id in matrix_lookup
        ],
        "local_hankel_family": local_hankel_family,
        "restriction_diagnostics": restriction_rows,
        "summary": summary,
    }


def write_topos_oom_bundle(
    documents: Iterable[object] | object,
    *,
    outdir: Path,
    corpus_label: str = "Token corpus",
    json_name: str = "topos_oom_hankel.json",
    html_name: str = "topos_oom_bundle.html",
    **kwargs: object,
) -> dict[str, Path]:
    """Materialize a Topos OOM JSON bundle and readable HTML companion."""

    outdir.mkdir(parents=True, exist_ok=True)
    bundle = build_topos_oom_bundle(documents, corpus_label=corpus_label, **kwargs)
    json_path = outdir / json_name
    html_path = outdir / html_name
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(render_topos_oom_bundle_html(bundle, raw_json_href=json_path.name), encoding="utf-8")
    return {"topos_oom_path": json_path, "topos_oom_html_path": html_path}


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value is None or value == "":
        return "not available"
    return str(value)


def _metric(label: str, value: object) -> str:
    return f'<div class="metric"><span>{escape(label)}</span><strong>{escape(_format_value(value))}</strong></div>'


def _render_context_rows(bundle: dict[str, object]) -> str:
    rows: list[str] = []
    for row in list(bundle.get("local_hankel_family") or []):
        payload = dict(row)
        svd = dict(payload.get("svd") or {})
        singular_values = ", ".join(str(value) for value in list(svd.get("singular_values") or [])[:4])
        operators = ", ".join(str(op.get("observation")) for op in list(payload.get("observation_operators") or [])[:6])
        rows.append(
            "<tr>"
            f"<td><strong>{escape(str(payload.get('context_label') or payload.get('context_id')))}</strong><br><span>{escape(str(payload.get('context_id') or ''))}</span></td>"
            f"<td>{escape(str(payload.get('n_observation_views') or 0))}</td>"
            f"<td>{escape(str(len(list(payload.get('histories') or []))))}</td>"
            f"<td>{escape(str(len(list(payload.get('tests') or []))))}</td>"
            f"<td>{escape(str(svd.get('rank') or 0))}</td>"
            f"<td>{escape(singular_values)}</td>"
            f"<td>{escape(operators)}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="7" class="empty">No local OOM contexts were published.</td></tr>'
    return "\n".join(rows)


def _render_restriction_rows(bundle: dict[str, object]) -> str:
    rows: list[str] = []
    for row in list(bundle.get("restriction_diagnostics") or []):
        payload = dict(row)
        rows.append(
            "<tr>"
            f"<td>{escape(str(payload.get('source_context') or ''))} -> {escape(str(payload.get('target_context') or ''))}</td>"
            f"<td>{escape(str(payload.get('shared_cells') or 0))}</td>"
            f"<td>{escape(_format_value(payload.get('mean_abs_gap')))}</td>"
            f"<td>{escape(_format_value(payload.get('max_abs_gap')))}</td>"
            f"<td>{escape('compatible' if payload.get('compatible') else 'tense')}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="5" class="empty">No restriction checks were available.</td></tr>'
    return "\n".join(rows)


def render_topos_oom_bundle_html(bundle: dict[str, object], *, raw_json_href: str = "topos_oom_hankel.json") -> str:
    """Render a compact human-readable Topos OOM companion page."""

    summary = dict(bundle.get("summary") or {})
    metrics = "\n".join(
        [
            _metric("Documents", summary.get("n_documents", 0)),
            _metric("Tokens", summary.get("n_tokens", 0)),
            _metric("Contexts", summary.get("n_contexts", 0)),
            _metric("Context views", summary.get("n_context_projected_views", 0)),
            _metric("Mean rank", summary.get("mean_rank", 0)),
            _metric("Compatibility", f"{summary.get('n_compatible_restrictions', 0)}/{summary.get('n_restriction_checks', 0)}"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(summary.get("corpus_label") or "Topos OOM Bundle"))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    .trace {{ color: #52606d; max-width: 920px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; background: #f8fafc; }}
    .metric span {{ display: block; color: #627d98; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: #627d98; text-transform: uppercase; letter-spacing: .03em; }}
    td span, .empty {{ color: #829ab1; }}
    a {{ color: #0b69a3; }}
  </style>
</head>
<body>
  <h1>Topos OOM Bundle</h1>
  <p class="trace">A readable companion for a sheaf-valued Observer Operator Model. The bundle learns hidden predictive state from observation tokens only, without action parsing.</p>
  <p><a href="{escape(raw_json_href)}">Raw JSON bundle</a></p>
  <section class="metrics">{metrics}</section>
  <h2>Local Observation Contexts</h2>
  <table>
    <thead><tr><th>Context</th><th>Views</th><th>Histories</th><th>Tests</th><th>Rank</th><th>Singular values</th><th>Operators</th></tr></thead>
    <tbody>{_render_context_rows(bundle)}</tbody>
  </table>
  <h2>Restriction Diagnostics</h2>
  <table>
    <thead><tr><th>Restriction</th><th>Shared cells</th><th>Mean gap</th><th>Max gap</th><th>Status</th></tr></thead>
    <tbody>{_render_restriction_rows(bundle)}</tbody>
  </table>
</body>
</html>
"""
