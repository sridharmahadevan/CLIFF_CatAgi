"""Decision metrics shared across Democritus query checkpoints and batch telemetry."""

from __future__ import annotations

import math
import re
from typing import Mapping


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _safe_ratio(numerator: object, denominator: object) -> float:
    denom = _safe_float(denominator)
    if denom <= 0:
        return 0.0
    return _safe_float(numerator) / denom


def _exp_support_score(count: object, scale: float) -> float:
    return 1.0 - math.exp(-max(_safe_float(count), 0.0) / max(scale, 1e-6))


def _normalize_topic(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _action_label(value: str) -> str:
    return value.replace("_", " ").strip() or "continue"


def compute_checkpoint_decision_state(
    *,
    drift_metrics: Mapping[str, object] | None,
    top_topics: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
    documents_payload: tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
) -> dict[str, object]:
    metrics = dict(drift_metrics or {})
    total_topic_count = max(
        1,
        _safe_int(metrics.get("total_topic_count")) or len(list(top_topics)),
    )
    suspicious_topic_count = _safe_int(metrics.get("suspicious_topic_count"))
    aligned_topic_ratio = _clamp01(metrics.get("aligned_topic_ratio"))
    readiness_score = _clamp01(
        metrics.get("synthesis_readiness_proxy") or metrics.get("mean_alignment_score")
    )
    recurring_topics = [
        dict(item)
        for item in top_topics
        if _safe_int(dict(item).get("document_count")) > 1
    ]
    recurring_topic_density = _safe_ratio(len(recurring_topics), total_topic_count)
    recurring_aliases = {
        normalized
        for item in recurring_topics
        for normalized in (
            _normalize_topic(dict(item).get("topic")),
            *(_normalize_topic(alias) for alias in list(dict(item).get("aliases") or [])),
        )
        if normalized
    }
    selected_docs_with_top_topics = 0
    for document in documents_payload:
        document_topics = {
            _normalize_topic(topic)
            for topic in list(dict(document).get("topics") or [])
            if _normalize_topic(topic)
        }
        if document_topics & recurring_aliases:
            selected_docs_with_top_topics += 1
    selected_doc_count = len(list(documents_payload))
    selected_topic_concentration = _safe_ratio(selected_docs_with_top_topics, selected_doc_count)
    drift_penalty = _safe_ratio(suspicious_topic_count, total_topic_count)
    checkpoint_value = _clamp01(
        0.40 * readiness_score
        + 0.25 * aligned_topic_ratio
        + 0.20 * selected_topic_concentration
        + 0.15 * recurring_topic_density
        - 0.30 * drift_penalty
    )
    if checkpoint_value >= 0.55:
        recommended_action = "continue"
    elif checkpoint_value >= 0.35:
        recommended_action = "narrow_retrieval"
    else:
        recommended_action = "stop"
    return {
        "checkpoint_value": round(checkpoint_value, 4),
        "aligned_topic_ratio": round(aligned_topic_ratio, 4),
        "selected_topic_concentration": round(selected_topic_concentration, 4),
        "recurring_topic_density": round(recurring_topic_density, 4),
        "readiness_score": round(readiness_score, 4),
        "drift_penalty": round(drift_penalty, 4),
        "selected_doc_count": selected_doc_count,
        "selected_docs_with_top_topics": selected_docs_with_top_topics,
        "recurring_topic_count": len(recurring_topics),
        "recommended_action": recommended_action,
        "recommended_action_label": _action_label(recommended_action),
    }


def compute_batch_decision_state(
    *,
    corpus_synthesis: Mapping[str, object] | None,
    llm_usage: Mapping[str, object] | None,
    total_documents: int,
    status: str,
    previous_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    synthesis = dict(corpus_synthesis or {})
    usage = dict(llm_usage or {})
    previous = dict(previous_state or {})
    support_summary = dict(synthesis.get("support_summary") or {})
    topic_partition_summary = dict(synthesis.get("topic_partition_summary") or {})
    homotopy_summary = dict(synthesis.get("homotopy_summary") or {})

    strong_support_count = _safe_int(
        support_summary.get("strong_support_count") or len(list(synthesis.get("strongly_supported") or []))
    )
    provisional_support_count = _safe_int(
        support_summary.get("provisional_support_count") or len(list(synthesis.get("weakly_supported") or []))
    )
    diagnostic_support_count = _safe_int(
        support_summary.get("diagnostic_support_count") or len(list(synthesis.get("diagnostic_supported") or []))
    )
    disagreement_count = _safe_int(
        support_summary.get("disagreement_count") or len(list(synthesis.get("disagreements") or []))
    )
    topic_partition_count = max(1, _safe_int(topic_partition_summary.get("partition_count")))
    multi_document_partition_count = _safe_int(topic_partition_summary.get("multi_document_partition_count"))
    cross_document_classes = max(1, _safe_int(homotopy_summary.get("cross_document_class_count")))
    coherent_cross_document_classes = _safe_int(
        homotopy_summary.get("coherent_cross_document_count")
        or min(
            _safe_int(homotopy_summary.get("coherent_count")),
            _safe_int(homotopy_summary.get("cross_document_class_count")),
        )
    )

    partition_glue_score = _clamp01(_safe_ratio(multi_document_partition_count, topic_partition_count))
    homotopy_coherence_score = _clamp01(
        _safe_ratio(coherent_cross_document_classes, cross_document_classes)
    )
    readiness_score = _clamp01(_safe_ratio(_safe_int(synthesis.get("n_documents")), max(total_documents, 1)))
    drift_penalty = _clamp01(
        _safe_ratio(
            max(topic_partition_count - multi_document_partition_count, 0),
            topic_partition_count,
        )
    )
    disagreement_penalty = _clamp01(
        _safe_ratio(
            disagreement_count,
            strong_support_count + provisional_support_count + diagnostic_support_count + disagreement_count,
        )
    )
    information_state = _clamp01(
        0.28 * _exp_support_score(strong_support_count, 3.0)
        + 0.18 * _exp_support_score(provisional_support_count, 6.0)
        + 0.14 * _exp_support_score(diagnostic_support_count, 8.0)
        + 0.18 * partition_glue_score
        + 0.12 * homotopy_coherence_score
        + 0.10 * readiness_score
        - 0.12 * drift_penalty
        - 0.08 * disagreement_penalty
    )

    total_tokens = _safe_int(usage.get("total_tokens"))
    by_agent = [dict(item) for item in list(usage.get("by_agent") or [])]
    statement_tokens = sum(
        _safe_int(item.get("total_tokens"))
        for item in by_agent
        if "statement" in str(item.get("agent_name") or "").lower()
    )
    question_tokens = sum(
        _safe_int(item.get("total_tokens"))
        for item in by_agent
        if "question" in str(item.get("agent_name") or "").lower()
    )
    statement_token_share = _clamp01(_safe_ratio(statement_tokens, total_tokens))
    question_token_share = _clamp01(_safe_ratio(question_tokens, total_tokens))

    previous_information_state = previous.get("information_state")
    previous_total_tokens = _safe_int(previous.get("total_tokens"))
    previous_drift_penalty = _safe_float(previous.get("drift_penalty"))
    previous_partition_glue = _safe_float(previous.get("partition_glue_score"))
    previous_homotopy_coherence = _safe_float(previous.get("homotopy_coherence_score"))
    previous_strong_support = _safe_int(previous.get("strong_support_count"))
    previous_multi_partition = _safe_int(previous.get("multi_document_partition_count"))
    previous_low_efficiency = _safe_int(previous.get("consecutive_low_efficiency"))
    has_prior_progress = previous_information_state is not None and previous_total_tokens > 0

    delta_information_state = (
        information_state - _safe_float(previous_information_state)
        if previous_information_state is not None
        else 0.0
    )
    delta_tokens = total_tokens - previous_total_tokens if previous else total_tokens
    marginal_efficiency: float | None = None
    if previous_information_state is not None and delta_tokens > 0:
        marginal_efficiency = 10_000.0 * delta_information_state / float(delta_tokens)

    drift_increasing = drift_penalty > previous_drift_penalty + 1e-9
    partition_glue_flat = abs(partition_glue_score - previous_partition_glue) < 0.01
    homotopy_flat = abs(homotopy_coherence_score - previous_homotopy_coherence) < 0.01
    strong_support_increased = strong_support_count > previous_strong_support
    partition_glue_increased = multi_document_partition_count > previous_multi_partition

    if marginal_efficiency is None:
        consecutive_low_efficiency = previous_low_efficiency
    elif marginal_efficiency < 0.008:
        consecutive_low_efficiency = previous_low_efficiency + 1
    else:
        consecutive_low_efficiency = 0

    if status == "complete":
        recommended_action = "stop"
    elif strong_support_increased or partition_glue_increased:
        recommended_action = "continue"
    elif (
        has_prior_progress
        and (
            statement_token_share > 0.60
            and delta_information_state < 0.01
            and partition_glue_flat
        )
    ):
        recommended_action = "narrow_retrieval"
    elif (
        has_prior_progress
        and (
            statement_token_share + question_token_share > 0.85
            and partition_glue_flat
            and homotopy_flat
        )
    ):
        recommended_action = "narrow_retrieval"
    elif not has_prior_progress:
        recommended_action = "continue" if information_state > 0.0 or total_tokens == 0 else "continue_cautiously"
    elif marginal_efficiency is None:
        recommended_action = "continue" if information_state > 0.0 or total_tokens == 0 else "continue_cautiously"
    elif marginal_efficiency >= 0.020:
        recommended_action = "continue"
    elif marginal_efficiency >= 0.008:
        recommended_action = "continue_cautiously" if not drift_increasing else "checkpoint"
    elif marginal_efficiency < 0.0 and drift_increasing:
        recommended_action = "narrow_retrieval"
    elif consecutive_low_efficiency >= 2:
        recommended_action = "checkpoint"
    else:
        recommended_action = "continue_cautiously"

    return {
        "information_state": round(information_state, 4),
        "marginal_efficiency": None if marginal_efficiency is None else round(marginal_efficiency, 4),
        "delta_information_state": round(delta_information_state, 4),
        "delta_tokens": delta_tokens,
        "checkpoint_value": None,
        "partition_glue_score": round(partition_glue_score, 4),
        "homotopy_coherence_score": round(homotopy_coherence_score, 4),
        "readiness_score": round(readiness_score, 4),
        "drift_penalty": round(drift_penalty, 4),
        "disagreement_penalty": round(disagreement_penalty, 4),
        "statement_token_share": round(statement_token_share, 4),
        "question_token_share": round(question_token_share, 4),
        "recommended_action": recommended_action,
        "recommended_action_label": _action_label(recommended_action),
        "strong_support_count": strong_support_count,
        "provisional_support_count": provisional_support_count,
        "diagnostic_support_count": diagnostic_support_count,
        "disagreement_count": disagreement_count,
        "multi_document_partition_count": multi_document_partition_count,
        "topic_partition_count": topic_partition_count,
        "cross_document_class_count": _safe_int(homotopy_summary.get("cross_document_class_count")),
        "coherent_cross_document_count": coherent_cross_document_classes,
        "statement_tokens": statement_tokens,
        "question_tokens": question_tokens,
        "total_tokens": total_tokens,
        "consecutive_low_efficiency": consecutive_low_efficiency,
    }
