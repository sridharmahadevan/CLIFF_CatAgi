from __future__ import annotations

import json
from pathlib import Path

import pytest

from functorflow_v3.democritus_agentic import DemocritusAgenticConfig
from functorflow_v3.unicausal_frontend import (
    UniCausalDecision,
    extract_unicausal_triples,
)


class _FakePredictor:
    def predict(self, text: str) -> UniCausalDecision:
        if "raises" not in text:
            return UniCausalDecision(False, 0.98, "LABEL_0", ())
        cause = "Exercise"
        effect = "heart rate"
        return UniCausalDecision(
            True,
            0.91,
            "LABEL_1",
            (
                {
                    "entity_group": "LABEL_0",
                    "start": text.index(cause),
                    "end": text.index(cause) + len(cause),
                },
                {
                    "entity_group": "LABEL_1",
                    "start": text.index(effect),
                    "end": text.index(effect) + len(effect),
                },
            ),
        )


def test_unicausal_frontend_filters_and_emits_compatible_triple(tmp_path: Path) -> None:
    statements = tmp_path / "causal_statements.jsonl"
    statements.write_text(
        json.dumps(
            {
                "topic": "Exercise physiology",
                "path": ["Exercise", "Cardiovascular response"],
                "question": "How does exercise affect heart rate?",
                "statements": [
                    "Exercise raises heart rate during physical activity.",
                    "Exercise and heart rate were both measured.",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "relational_triples.jsonl"
    audit = tmp_path / "unicausal_frontend_audit.json"

    summary = extract_unicausal_triples(
        statements,
        output,
        audit,
        query="exercise effects on heart rate",
        min_confidence=0.8,
        predictor=_FakePredictor(),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["subj"] == "exercise"
    assert rows[0]["obj"] == "heart rate"
    assert rows[0]["frontend"] == "unicausal"
    assert summary["statements_seen"] == 2
    assert summary["statements_accepted"] == 1
    assert summary["statement_retention_rate"] == 0.5
    assert json.loads(audit.read_text())["accepted_query_anchor_rate"] == 1.0


def test_unicausal_threshold_can_reject_all_without_failing(tmp_path: Path) -> None:
    statements = tmp_path / "causal_statements.jsonl"
    statements.write_text(
        json.dumps(
            {
                "topic": "Exercise physiology",
                "statements": ["Exercise raises heart rate during physical activity."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "relational_triples.jsonl"
    audit = tmp_path / "unicausal_frontend_audit.json"
    summary = extract_unicausal_triples(
        statements,
        output,
        audit,
        min_confidence=0.95,
        predictor=_FakePredictor(),
        require_nonempty=False,
    )
    assert output.read_text() == ""
    assert summary["statements_accepted"] == 0


def test_query_anchor_gate_rejects_off_topic_causal_statement(tmp_path: Path) -> None:
    statements = tmp_path / "causal_statements.jsonl"
    statements.write_text(
        json.dumps(
            {
                "topic": "Exercise physiology",
                "statements": ["Exercise raises heart rate during physical activity."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "relational_triples.jsonl"
    audit = tmp_path / "unicausal_frontend_audit.json"
    summary = extract_unicausal_triples(
        statements,
        output,
        audit,
        query="penguin breeding habitat",
        predictor=_FakePredictor(),
        require_query_anchor=True,
        require_nonempty=False,
    )
    assert summary["statements_accepted"] == 0
    assert summary["require_query_anchor"] is True


def test_democritus_extractor_configuration_is_normalized() -> None:
    config = DemocritusAgenticConfig(
        outdir=Path("."),
        causal_extractor=" UniCausal ",
        unicausal_min_confidence=1.4,
        unicausal_require_query_anchor=True,
        request_query="  causal   exercise effects ",
    ).resolved()
    assert config.causal_extractor == "unicausal"
    assert config.unicausal_min_confidence == 1.0
    assert config.unicausal_require_query_anchor is True
    assert config.request_query == "causal exercise effects"

    with pytest.raises(ValueError, match="causal_extractor"):
        DemocritusAgenticConfig(
            outdir=Path("."), causal_extractor="unknown"
        ).resolved()
