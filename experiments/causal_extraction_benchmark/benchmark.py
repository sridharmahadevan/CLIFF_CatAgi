#!/usr/bin/env python3
"""Reproducible causal-language extraction benchmark for Democritus.

The harness consumes UniCausal's grouped CSV format, runs either a transparent
cue baseline, the public UniCausal sequence/token models, or a restricted
OpenAI-compatible verbatim-span extractor, and scores every method against the
same frozen gold examples.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


TAG_RE = re.compile(r"</?(?:ARG0|ARG1|SIG[A-Z0-9]*)>")
TOKEN_RE = re.compile(r"\S+")
CAUSAL_CUES = re.compile(
    r"\b(?:because|because of|due to|caused? by|causes?|led to|leads? to|"
    r"results? in|resulted in|increases?|increased|reduces?|reduced|"
    r"affects?|affected|influences?|influenced|contributes? to|"
    r"as a result of|therefore|thus|hence)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You extract causal relations explicitly expressed in text.
Return JSON only, with this schema:
{"relations":[{"cause_span":"...","effect_span":"...","signal_span":"...","confidence":0.0}]}

Rules:
- Every nonempty span must be copied verbatim from the input.
- Extract only relations explicitly expressed by the input.
- Preserve cause-to-effect direction.
- Return every expressed causal relation, including implicit lexical causation.
- Return {"relations":[]} when no causal relation is expressed.
- Do not infer background mechanisms or rewrite the text.
"""


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Relation:
    cause: Span
    effect: Span
    signal: Span | None = None


@dataclass(frozen=True)
class Example:
    example_id: str
    corpus: str
    text: str
    relations: tuple[Relation, ...]

    @property
    def causal(self) -> bool:
        return bool(self.relations)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marked_to_plain_and_spans(marked: str) -> tuple[str, dict[str, Span]]:
    """Remove UniCausal tags while retaining exact character offsets."""
    output: list[str] = []
    open_spans: dict[str, tuple[int, int]] = {}
    spans: dict[str, Span] = {}
    cursor = 0
    for match in TAG_RE.finditer(marked):
        segment = marked[cursor : match.start()]
        output.append(segment)
        plain_offset = sum(len(part) for part in output)
        token = match.group(0)
        name = token.strip("</>")
        if token.startswith("</"):
            if name in open_spans:
                start, _ = open_spans.pop(name)
                plain = "".join(output)
                spans[name] = Span(plain[start:plain_offset], start, plain_offset)
        else:
            open_spans[name] = (plain_offset, match.end())
        cursor = match.end()
    output.append(marked[cursor:])
    plain = "".join(output)
    return plain, spans


def parse_gold_relation(text: str, marked: str) -> Relation:
    plain, spans = _marked_to_plain_and_spans(marked)
    if plain != text:
        raise ValueError("Marked relation does not reconstruct the benchmark text")
    if "ARG0" not in spans or "ARG1" not in spans:
        raise ValueError("Gold relation lacks ARG0 cause or ARG1 effect")
    signals = [span for name, span in spans.items() if name.startswith("SIG")]
    return Relation(
        cause=spans["ARG0"],
        effect=spans["ARG1"],
        signal=signals[0] if signals else None,
    )


def _gold_marked_relations(row: dict[str, str]) -> list[str]:
    raw = (row.get("causal_text_w_pairs") or "").strip()
    if raw:
        value = ast.literal_eval(raw)
        if not isinstance(value, list):
            raise ValueError("causal_text_w_pairs must decode to a list")
        return [str(item) for item in value]
    if str(row.get("pair_label", "0")) == "1" and row.get("text_w_pairs"):
        return [row["text_w_pairs"]]
    return []


def load_unicausal_grouped(path: Path, limit: int = 0) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row["text"]
            relations = tuple(
                parse_gold_relation(text, marked)
                for marked in _gold_marked_relations(row)
            )
            examples.append(
                Example(
                    example_id=row["index"],
                    corpus=row["corpus"],
                    text=text,
                    relations=relations,
                )
            )
            if limit and len(examples) >= limit:
                break
    return examples


def locate_verbatim(text: str, value: str) -> Span | None:
    value = value.strip()
    if not value:
        return None
    start = text.find(value)
    if start < 0:
        return None
    return Span(value, start, start + len(value))


def parse_prediction_payload(text: str, payload: object) -> tuple[list[Relation], int]:
    """Validate an extractor payload. Invalid/non-verbatim records are counted."""
    invalid = 0
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        return [], 1
    relations: list[Relation] = []
    for record in payload["relations"]:
        if not isinstance(record, dict):
            invalid += 1
            continue
        cause = locate_verbatim(text, str(record.get("cause_span") or ""))
        effect = locate_verbatim(text, str(record.get("effect_span") or ""))
        signal_value = str(record.get("signal_span") or "")
        signal = locate_verbatim(text, signal_value) if signal_value else None
        if cause is None or effect is None or (signal_value and signal is None):
            invalid += 1
            continue
        relations.append(Relation(cause=cause, effect=effect, signal=signal))
    return relations, invalid


def extract_json_object(raw: str) -> object:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def cue_prediction(example: Example) -> dict[str, object]:
    causal = bool(CAUSAL_CUES.search(example.text))
    return {
        "id": example.example_id,
        "causal": causal,
        "relations": [],
        "invalid": 0,
        "runtime_seconds": 0.0,
    }


def json_safe(value: object) -> object:
    """Convert NumPy/PyTorch scalar outputs into ordinary JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def _post_chat_completion(
    *,
    base_url: str,
    model: str,
    text: str,
    api_key: str,
    timeout: float,
) -> tuple[str, dict[str, object]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 200,
        "enable_thinking": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INPUT TEXT:\n{text}"},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return str(content), dict(payload.get("usage") or {})


def run_llm(
    examples: Sequence[Example],
    *,
    output: Path,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    workers: int,
) -> None:
    completed: set[str] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            completed = {json.loads(line)["id"] for line in handle if line.strip()}
    pending = [example for example in examples if example.example_id not in completed]

    def evaluate(example: Example) -> dict[str, object]:
        started = time.perf_counter()
        invalid = 0
        error = ""
        raw = ""
        usage: dict[str, object] = {}
        relations: list[Relation] = []
        try:
            raw, usage = _post_chat_completion(
                base_url=base_url,
                model=model,
                text=example.text,
                api_key=api_key,
                timeout=timeout,
            )
            payload = extract_json_object(raw)
            relations, invalid = parse_prediction_payload(example.text, payload)
        except Exception as exc:  # preserve failures as auditable predictions
            invalid += 1
            error = f"{type(exc).__name__}: {exc}"
        record = prediction_record(
            example,
            relations,
            invalid=invalid,
            runtime_seconds=time.perf_counter() - started,
        )
        record.update({"raw": raw, "usage": usage, "error": error, "model": model})
        return record

    with output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:
        futures = {executor.submit(evaluate, example): example for example in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            example = futures[future]
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {example.example_id}", flush=True)


def run_unicausal(
    examples: Sequence[Example],
    *,
    output: Path,
    sequence_model: str,
    token_model: str,
    device: int,
) -> None:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("The unicausal command requires transformers and torch") from exc

    classifier = pipeline(
        "text-classification", model=sequence_model, tokenizer=sequence_model, device=device
    )
    tagger = pipeline(
        "token-classification",
        model=token_model,
        tokenizer=token_model,
        aggregation_strategy="simple",
        device=device,
    )
    with output.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            started = time.perf_counter()
            classification = classifier(example.text, truncation=True)[0]
            causal = str(classification["label"]).upper() in {"LABEL_1", "YES", "CAUSAL"}
            relations: list[Relation] = []
            invalid = 0
            tags: list[dict[str, object]] = []
            if causal:
                tags = list(tagger(example.text))
                # The public model card specifies:
                # LABEL_0=B-C, LABEL_1=B-E, LABEL_2=I-C,
                # LABEL_3=I-E, LABEL_4=O. Its config retains only the
                # generic LABEL_n names, so reconstruct spans explicitly.
                causes = _unicausal_role_spans(example.text, tags, role="cause")
                effects = _unicausal_role_spans(example.text, tags, role="effect")
                # UniCausal's public token model predicts labeled spans but does not
                # pair multiple causes/effects. Pair in textual order deterministically.
                for cause, effect in zip(causes, effects):
                    relations.append(Relation(cause=cause, effect=effect))
            record = prediction_record(
                example,
                relations,
                causal_override=causal,
                invalid=invalid,
                runtime_seconds=time.perf_counter() - started,
            )
            record.update(
                {
                    "classification": json_safe(classification),
                    "token_predictions": json_safe(tags),
                    "sequence_model": sequence_model,
                    "token_model": token_model,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(examples)}] {example.example_id}", flush=True)


def _unicausal_role_spans(
    text: str, tags: Sequence[dict[str, object]], *, role: str
) -> list[Span]:
    if role == "cause":
        begin_labels, inside_labels = {"LABEL_0", "B-C"}, {"LABEL_2", "I-C"}
    elif role == "effect":
        begin_labels, inside_labels = {"LABEL_1", "B-E"}, {"LABEL_3", "I-E"}
    else:
        raise ValueError(f"Unknown role: {role}")
    spans: list[Span] = []
    active_start: int | None = None
    active_end: int | None = None
    for item in tags:
        label = str(item.get("entity_group") or item.get("entity") or "").upper()
        start, end = int(item["start"]), int(item["end"])
        if label in begin_labels:
            if active_start is not None and active_end is not None:
                spans.append(Span(text[active_start:active_end], active_start, active_end))
            active_start, active_end = start, end
        elif label in inside_labels:
            if active_start is None:
                active_start, active_end = start, end
            else:
                active_end = end
        elif active_start is not None and active_end is not None:
            spans.append(Span(text[active_start:active_end], active_start, active_end))
            active_start = active_end = None
    if active_start is not None and active_end is not None:
        spans.append(Span(text[active_start:active_end], active_start, active_end))
    return spans


def prediction_record(
    example: Example,
    relations: Sequence[Relation],
    *,
    causal_override: bool | None = None,
    invalid: int = 0,
    runtime_seconds: float = 0.0,
) -> dict[str, object]:
    return {
        "id": example.example_id,
        "causal": bool(relations) if causal_override is None else causal_override,
        "relations": [
            {
                "cause_span": relation.cause.text,
                "cause_start": relation.cause.start,
                "cause_end": relation.cause.end,
                "effect_span": relation.effect.text,
                "effect_start": relation.effect.start,
                "effect_end": relation.effect.end,
                "signal_span": relation.signal.text if relation.signal else "",
            }
            for relation in relations
        ],
        "invalid": invalid,
        "runtime_seconds": runtime_seconds,
    }


def write_cue_predictions(examples: Sequence[Example], output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(cue_prediction(example), ensure_ascii=False) + "\n")


def _record_relations(record: dict[str, object]) -> list[Relation]:
    output: list[Relation] = []
    for relation in record.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        output.append(
            Relation(
                cause=Span(
                    str(relation["cause_span"]),
                    int(relation["cause_start"]),
                    int(relation["cause_end"]),
                ),
                effect=Span(
                    str(relation["effect_span"]),
                    int(relation["effect_start"]),
                    int(relation["effect_end"]),
                ),
            )
        )
    return output


def _token_ids(text: str, span: Span) -> set[int]:
    return {
        index
        for index, match in enumerate(TOKEN_RE.finditer(text))
        if match.start() < span.end and span.start < match.end()
    }


def _maximum_matching(
    predicted: Sequence[Relation],
    gold: Sequence[Relation],
    predicate,
) -> int:
    best = 0

    def visit(index: int, used: set[int], count: int) -> None:
        nonlocal best
        if index >= len(predicted):
            best = max(best, count)
            return
        visit(index + 1, used, count)
        for gold_index, target in enumerate(gold):
            if gold_index not in used and predicate(predicted[index], target):
                used.add(gold_index)
                visit(index + 1, used, count + 1)
                used.remove(gold_index)

    visit(0, set(), 0)
    return best


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def score(examples: Sequence[Example], predictions_path: Path) -> dict[str, object]:
    with predictions_path.open(encoding="utf-8") as handle:
        predictions = {row["id"]: row for row in map(json.loads, handle) if row}
    tp = fp = fn = tn = 0
    cause_tp = cause_pred = cause_gold = 0
    effect_tp = effect_pred = effect_gold = 0
    exact_matches = relaxed_matches = predicted_pairs = gold_pairs = 0
    invalid = 0
    runtime = 0.0
    missing = 0
    for example in examples:
        record = predictions.get(example.example_id)
        if record is None:
            missing += 1
            record = {"causal": False, "relations": [], "invalid": 1}
        pred_causal = bool(record.get("causal"))
        if example.causal and pred_causal:
            tp += 1
        elif example.causal:
            fn += 1
        elif pred_causal:
            fp += 1
        else:
            tn += 1
        predicted = _record_relations(record)
        invalid += int(record.get("invalid") or 0)
        runtime += float(record.get("runtime_seconds") or 0.0)
        predicted_pairs += len(predicted)
        gold_pairs += len(example.relations)
        pred_cause_tokens = set().union(
            *(_token_ids(example.text, item.cause) for item in predicted), set()
        )
        gold_cause_tokens = set().union(
            *(_token_ids(example.text, item.cause) for item in example.relations), set()
        )
        pred_effect_tokens = set().union(
            *(_token_ids(example.text, item.effect) for item in predicted), set()
        )
        gold_effect_tokens = set().union(
            *(_token_ids(example.text, item.effect) for item in example.relations), set()
        )
        cause_tp += len(pred_cause_tokens & gold_cause_tokens)
        cause_pred += len(pred_cause_tokens)
        cause_gold += len(gold_cause_tokens)
        effect_tp += len(pred_effect_tokens & gold_effect_tokens)
        effect_pred += len(pred_effect_tokens)
        effect_gold += len(gold_effect_tokens)
        exact_matches += _maximum_matching(
            predicted,
            example.relations,
            lambda left, right: (
                left.cause.start == right.cause.start
                and left.cause.end == right.cause.end
                and left.effect.start == right.effect.start
                and left.effect.end == right.effect.end
            ),
        )
        relaxed_matches += _maximum_matching(
            predicted,
            example.relations,
            lambda left, right: bool(
                _token_ids(example.text, left.cause)
                & _token_ids(example.text, right.cause)
            )
            and bool(
                _token_ids(example.text, left.effect)
                & _token_ids(example.text, right.effect)
            ),
        )
    class_precision = _safe_div(tp, tp + fp)
    class_recall = _safe_div(tp, tp + fn)
    cause_precision = _safe_div(cause_tp, cause_pred)
    cause_recall = _safe_div(cause_tp, cause_gold)
    effect_precision = _safe_div(effect_tp, effect_pred)
    effect_recall = _safe_div(effect_tp, effect_gold)
    return {
        "examples": len(examples),
        "positive_examples": sum(example.causal for example in examples),
        "negative_examples": sum(not example.causal for example in examples),
        "classification": {
            "precision": class_precision,
            "recall": class_recall,
            "f1": _f1(class_precision, class_recall),
            "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "cause_span": {
            "precision": cause_precision,
            "recall": cause_recall,
            "f1": _f1(cause_precision, cause_recall),
        },
        "effect_span": {
            "precision": effect_precision,
            "recall": effect_recall,
            "f1": _f1(effect_precision, effect_recall),
        },
        "span_macro_f1": (
            _f1(cause_precision, cause_recall)
            + _f1(effect_precision, effect_recall)
        )
        / 2,
        "directed_pair": {
            "gold": gold_pairs,
            "predicted": predicted_pairs,
            "exact_matches": exact_matches,
            "exact_f1": _f1(
                _safe_div(exact_matches, predicted_pairs),
                _safe_div(exact_matches, gold_pairs),
            ),
            "relaxed_matches": relaxed_matches,
            "relaxed_f1": _f1(
                _safe_div(relaxed_matches, predicted_pairs),
                _safe_div(relaxed_matches, gold_pairs),
            ),
        },
        "invalid_outputs": invalid,
        "missing_predictions": missing,
        "runtime_seconds": runtime,
        "predictions_sha256": sha256_file(predictions_path),
    }


def write_manifest(dataset: Path, examples: Sequence[Example], output: Path) -> None:
    payload = {
        "dataset_file": dataset.name,
        "dataset_redistributed": False,
        "dataset_sha256": sha256_file(dataset),
        "examples": len(examples),
        "positive_examples": sum(example.causal for example in examples),
        "negative_examples": sum(not example.causal for example in examples),
        "example_ids_sha256": hashlib.sha256(
            "\n".join(example.example_id for example in examples).encode("utf-8")
        ).hexdigest(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)

    cue = subparsers.add_parser("cue")
    cue.add_argument("--output", type=Path, required=True)

    unicausal = subparsers.add_parser("unicausal")
    unicausal.add_argument("--output", type=Path, required=True)
    unicausal.add_argument(
        "--sequence-model", default="tanfiona/unicausal-seq-baseline"
    )
    unicausal.add_argument(
        "--token-model", default="tanfiona/unicausal-tok-baseline"
    )
    unicausal.add_argument("--device", type=int, default=-1)

    llm = subparsers.add_parser("llm")
    llm.add_argument("--output", type=Path, required=True)
    llm.add_argument("--base-url", required=True)
    llm.add_argument("--model", required=True)
    llm.add_argument("--api-key", default="")
    llm.add_argument("--timeout", type=float, default=120.0)
    llm.add_argument("--workers", type=int, default=4)

    scorer = subparsers.add_parser("score")
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_unicausal_grouped(args.dataset, limit=args.limit)
    if args.command == "manifest":
        write_manifest(args.dataset, examples, args.output)
    elif args.command == "cue":
        write_cue_predictions(examples, args.output)
    elif args.command == "unicausal":
        run_unicausal(
            examples,
            output=args.output,
            sequence_model=args.sequence_model,
            token_model=args.token_model,
            device=args.device,
        )
    elif args.command == "llm":
        run_llm(
            examples,
            output=args.output,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            workers=args.workers,
        )
    elif args.command == "score":
        metrics = score(examples, args.predictions)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
