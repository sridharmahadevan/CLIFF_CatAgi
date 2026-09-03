import json
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    Example,
    Relation,
    Span,
    _unicausal_role_spans,
    _marked_to_plain_and_spans,
    parse_gold_relation,
    parse_prediction_payload,
    score,
    write_manifest,
)


class BenchmarkTests(unittest.TestCase):
    def test_marked_relation_offsets(self):
        text = "Rain caused flooding."
        marked = "<ARG0>Rain</ARG0> caused <ARG1>flooding</ARG1>."
        relation = parse_gold_relation(text, marked)
        self.assertEqual(relation.cause, Span("Rain", 0, 4))
        self.assertEqual(relation.effect, Span("flooding", 12, 20))

    def test_non_verbatim_prediction_is_invalid(self):
        relations, invalid = parse_prediction_payload(
            "Rain caused flooding.",
            {
                "relations": [
                    {
                        "cause_span": "heavy rain",
                        "effect_span": "flooding",
                        "signal_span": "caused",
                    }
                ]
            },
        )
        self.assertEqual(relations, [])
        self.assertEqual(invalid, 1)

    def test_perfect_prediction_scores_one(self):
        text = "Rain caused flooding."
        relation = Relation(Span("Rain", 0, 4), Span("flooding", 12, 20))
        example = Example("one", "fixture", text, (relation,))
        record = {
            "id": "one",
            "causal": True,
            "relations": [
                {
                    "cause_span": "Rain",
                    "cause_start": 0,
                    "cause_end": 4,
                    "effect_span": "flooding",
                    "effect_start": 12,
                    "effect_end": 20,
                    "signal_span": "caused",
                }
            ],
            "invalid": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            metrics = score([example], path)
        self.assertEqual(metrics["classification"]["f1"], 1.0)
        self.assertEqual(metrics["span_macro_f1"], 1.0)
        self.assertEqual(metrics["directed_pair"]["exact_f1"], 1.0)

    def test_unicausal_generic_labels_are_reconstructed(self):
        text = "Rain caused severe flooding."
        tags = [
            {"entity_group": "LABEL_0", "start": 0, "end": 4},
            {"entity_group": "LABEL_4", "start": 5, "end": 11},
            {"entity_group": "LABEL_1", "start": 12, "end": 18},
            {"entity_group": "LABEL_3", "start": 19, "end": 27},
        ]
        self.assertEqual(
            _unicausal_role_spans(text, tags, role="cause"),
            [Span("Rain", 0, 4)],
        )
        self.assertEqual(
            _unicausal_role_spans(text, tags, role="effect"),
            [Span("severe flooding", 12, 27)],
        )

    def test_manifest_does_not_publish_local_dataset_path(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "altlex_test.csv"
            dataset.write_text("index,corpus,text\n", encoding="utf-8")
            output = Path(directory) / "manifest.json"
            write_manifest(dataset, [], output)
            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(manifest["dataset_file"], "altlex_test.csv")
        self.assertFalse(manifest["dataset_redistributed"])
        self.assertNotIn("dataset_path", manifest)


if __name__ == "__main__":
    unittest.main()
