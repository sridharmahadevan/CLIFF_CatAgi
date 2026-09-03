"""Optional conservative UniCausal front-end for Democritus triples."""

from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence


DEFAULT_SEQUENCE_MODEL = "tanfiona/unicausal-seq-baseline"
DEFAULT_TOKEN_MODEL = "tanfiona/unicausal-tok-baseline"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "causal", "document",
    "documents", "effect", "effects", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "paper", "papers", "study", "studies", "the", "to",
    "what", "which", "with", "analyze", "anchored", "atlas", "collect",
    "democritus", "directly", "discovery", "extract", "focused", "inspect",
    "keep", "lite", "model", "query", "rebuild", "relevant", "run", "triple",
    "triples", "update",
}
_RELATION_CUES = (
    ("contributes to", "contributes_to"),
    ("is associated with", "is_associated_with"),
    ("correlates with", "correlates_with"),
    ("results in", "results_in"),
    ("resulted in", "results_in"),
    ("leads to", "leads_to"),
    ("led to", "leads_to"),
    ("increases", "increases"),
    ("increased", "increases"),
    ("reduces", "reduces"),
    ("reduced", "reduces"),
    ("influences", "influences"),
    ("influenced", "influences"),
    ("affects", "affects"),
    ("affected", "affects"),
    ("causes", "causes"),
    ("caused", "causes"),
)
_PIPELINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class UniCausalDecision:
    causal: bool
    score: float
    label: str
    tags: tuple[dict[str, object], ...]


class UniCausalPredictor(Protocol):
    def predict(self, text: str) -> UniCausalDecision: ...


class TransformersUniCausalPredictor:
    """Lazy, cached wrapper around the public UniCausal sequence/token models."""

    def __init__(
        self,
        *,
        sequence_model: str = DEFAULT_SEQUENCE_MODEL,
        token_model: str = DEFAULT_TOKEN_MODEL,
        device: int = -1,
    ) -> None:
        self.sequence_model = sequence_model
        self.token_model = token_model
        self.device = int(device)

    def predict(self, text: str) -> UniCausalDecision:
        classifier, tagger = _load_pipelines(
            self.sequence_model, self.token_model, self.device
        )
        with _PIPELINE_LOCK:
            classification = classifier(text, truncation=True)[0]
            label = str(classification.get("label") or "").upper()
            score = float(classification.get("score") or 0.0)
            causal = label in {"LABEL_1", "YES", "CAUSAL"}
            tags = tuple(tagger(text)) if causal else ()
        return UniCausalDecision(causal=causal, score=score, label=label, tags=tags)


@lru_cache(maxsize=4)
def _load_pipelines(sequence_model: str, token_model: str, device: int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The UniCausal front-end requires `transformers` and a supported "
            "PyTorch installation. Use the legacy extractor or install the "
            "optional UniCausal dependencies."
        ) from exc
    classifier = pipeline(
        "text-classification",
        model=sequence_model,
        tokenizer=sequence_model,
        device=device,
    )
    tagger = pipeline(
        "token-classification",
        model=token_model,
        tokenizer=token_model,
        aggregation_strategy="simple",
        device=device,
    )
    return classifier, tagger


def _tokens(text: object) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _role_spans(
    text: str, tags: Sequence[dict[str, object]], *, role: str
) -> list[tuple[str, int, int]]:
    if role == "cause":
        begin_labels, inside_labels = {"LABEL_0", "B-C"}, {"LABEL_2", "I-C"}
    elif role == "effect":
        begin_labels, inside_labels = {"LABEL_1", "B-E"}, {"LABEL_3", "I-E"}
    else:
        raise ValueError(f"Unknown UniCausal role: {role}")
    spans: list[tuple[str, int, int]] = []
    active_start: int | None = None
    active_end: int | None = None
    for item in tags:
        label = str(item.get("entity_group") or item.get("entity") or "").upper()
        start, end = int(item["start"]), int(item["end"])
        if label in begin_labels:
            if active_start is not None and active_end is not None:
                spans.append((text[active_start:active_end], active_start, active_end))
            active_start, active_end = start, end
        elif label in inside_labels:
            if active_start is None:
                active_start, active_end = start, end
            else:
                active_end = end
        elif active_start is not None and active_end is not None:
            spans.append((text[active_start:active_end], active_start, active_end))
            active_start = active_end = None
    if active_start is not None and active_end is not None:
        spans.append((text[active_start:active_end], active_start, active_end))
    return spans


def _relation_label(statement: str) -> str:
    lowered = statement.lower()
    for cue, label in _RELATION_CUES:
        if cue in lowered:
            return label
    return "expressed_causal_relation"


def _read_statement_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def extract_unicausal_triples(
    statements_path: Path,
    output_path: Path,
    audit_path: Path,
    *,
    query: str = "",
    min_confidence: float = 0.5,
    predictor: UniCausalPredictor | None = None,
    sequence_model: str = DEFAULT_SEQUENCE_MODEL,
    token_model: str = DEFAULT_TOKEN_MODEL,
    device: int = -1,
    require_nonempty: bool = True,
    require_query_anchor: bool = False,
) -> dict[str, object]:
    """Gate statements with UniCausal and emit span-grounded Democritus triples."""

    predictor = predictor or TransformersUniCausalPredictor(
        sequence_model=sequence_model,
        token_model=token_model,
        device=device,
    )
    threshold = min(1.0, max(0.0, float(min_confidence)))
    query_tokens = _tokens(query)
    statement_rows = _read_statement_rows(statements_path)
    decisions: list[dict[str, object]] = []
    triples: list[dict[str, object]] = []
    statement_count = 0
    accepted_statement_count = 0
    for row_index, row in enumerate(statement_rows):
        topic = str(row.get("topic") or "")
        path = [str(item) for item in list(row.get("path") or [])]
        question = str(row.get("question") or "")
        topic_tokens = _tokens(" ".join([topic, *path]))
        for statement_index, raw_statement in enumerate(list(row.get("statements") or [])):
            statement = str(raw_statement).strip()
            if not statement:
                continue
            statement_count += 1
            decision = predictor.predict(statement)
            causes = _role_spans(statement, decision.tags, role="cause")
            effects = _role_spans(statement, decision.tags, role="effect")
            statement_tokens = _tokens(statement)
            query_overlap = sorted(statement_tokens & query_tokens)
            topic_overlap = sorted(statement_tokens & topic_tokens)
            accepted = (
                decision.causal
                and decision.score >= threshold
                and bool(causes)
                and bool(effects)
                and (not require_query_anchor or not query_tokens or bool(query_overlap))
            )
            decision_record = {
                "row_index": row_index,
                "statement_index": statement_index,
                "topic": topic,
                "statement": statement,
                "accepted": accepted,
                "causal": decision.causal,
                "label": decision.label,
                "score": decision.score,
                "cause_spans": [item[0] for item in causes],
                "effect_spans": [item[0] for item in effects],
                "query_anchor_overlap": query_overlap,
                "topic_anchor_overlap": topic_overlap,
            }
            decisions.append(decision_record)
            if not accepted:
                continue
            accepted_statement_count += 1
            for pair_index, (cause, effect) in enumerate(zip(causes, effects)):
                triples.append(
                    {
                        "topic": topic,
                        "path": path,
                        "question": question,
                        "statement": statement,
                        "subj": cause[0].strip().lower(),
                        "rel": _relation_label(statement),
                        "obj": effect[0].strip().lower(),
                        "domain": path[0] if path else topic,
                        "frontend": "unicausal",
                        "frontend_score": decision.score,
                        "frontend_label": decision.label,
                        "frontend_pair_index": pair_index,
                        "cause_start": cause[1],
                        "cause_end": cause[2],
                        "effect_start": effect[1],
                        "effect_end": effect[2],
                    }
                )

    accepted = [item for item in decisions if item["accepted"]]
    rejected = [item for item in decisions if not item["accepted"]]

    def rate(items: list[dict[str, object]], key: str) -> float:
        return (
            sum(bool(item.get(key)) for item in items) / len(items)
            if items
            else 0.0
        )

    audit = {
        "schema_version": "prometheus.democritus.unicausal_frontend.v1",
        "frontend": "unicausal",
        "sequence_model": sequence_model,
        "token_model": token_model,
        "min_confidence": threshold,
        "require_query_anchor": require_query_anchor,
        "query": query,
        "query_anchor_tokens": sorted(query_tokens),
        "statement_rows": len(statement_rows),
        "statements_seen": statement_count,
        "statements_accepted": accepted_statement_count,
        "statement_retention_rate": (
            accepted_statement_count / statement_count if statement_count else 0.0
        ),
        "triples_emitted": len(triples),
        "accepted_query_anchor_rate": rate(accepted, "query_anchor_overlap"),
        "rejected_query_anchor_rate": rate(rejected, "query_anchor_overlap"),
        "accepted_topic_anchor_rate": rate(accepted, "topic_anchor_overlap"),
        "rejected_topic_anchor_rate": rate(rejected, "topic_anchor_overlap"),
        "decisions": decisions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for triple in triples:
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if require_nonempty and not triples:
        raise RuntimeError(
            "UniCausal rejected every generated statement or found no paired "
            f"cause/effect spans. See {audit_path}."
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the conservative UniCausal front-end on Democritus statements."
    )
    parser.add_argument("--statements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--sequence-model", default=DEFAULT_SEQUENCE_MODEL)
    parser.add_argument("--token-model", default=DEFAULT_TOKEN_MODEL)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--require-query-anchor", action="store_true")
    args = parser.parse_args()
    audit = extract_unicausal_triples(
        args.statements,
        args.output,
        args.audit,
        query=args.query,
        min_confidence=args.min_confidence,
        sequence_model=args.sequence_model,
        token_model=args.token_model,
        device=args.device,
        require_nonempty=not args.allow_empty,
        require_query_anchor=args.require_query_anchor,
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
