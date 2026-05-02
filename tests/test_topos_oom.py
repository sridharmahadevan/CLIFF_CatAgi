"""Tests for token-observation Topos OOM bundles."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3.topos_oom import (
        build_topos_oom_bundle,
        evaluate_topos_oom_perplexity,
        render_topos_oom_bundle_html,
        write_topos_oom_bundle,
    )
except ModuleNotFoundError:
    from ..functorflow_v3.topos_oom import (
        build_topos_oom_bundle,
        evaluate_topos_oom_perplexity,
        render_topos_oom_bundle_html,
        write_topos_oom_bundle,
    )


class ToposOOMTests(unittest.TestCase):
    def test_builds_observation_only_bundle_from_arbitrary_text(self) -> None:
        bundle = build_topos_oom_bundle(
            [
                {
                    "document_id": "doc-a",
                    "title": "Alpha",
                    "text": "Alpha beta alpha. Beta gamma alpha.\n\nGamma beta alpha.",
                },
                {
                    "document_id": "doc-b",
                    "title": "Beta",
                    "text": "Beta alpha gamma. Alpha beta gamma.",
                },
            ],
            corpus_label="toy token corpus",
            max_history_length=2,
            max_test_length=2,
            min_support=1,
            max_histories_per_context=10,
            max_tests_per_context=12,
            max_operator_observations=4,
        )

        summary = dict(bundle["summary"])
        self.assertEqual(summary["bundle_type"], "topos_observer_operator_model")
        self.assertEqual(summary["n_documents"], 2)
        self.assertIn("corpus", summary["context_ids"])
        self.assertIn("sentence", summary["context_ids"])
        self.assertGreaterEqual(summary["n_restriction_checks"], 1)

        corpus = next(row for row in bundle["local_hankel_family"] if row["context_id"] == "corpus")
        self.assertGreaterEqual(dict(corpus["svd"])["rank"], 1)
        self.assertTrue(corpus["observation_operators"])
        self.assertTrue(all("observation" in row for row in corpus["observation_operators"]))
        self.assertTrue(any(row["signature"] == "epsilon" for row in corpus["histories"]))
        self.assertTrue(any(row["signature"] == "alpha" for row in corpus["tests"]))

    def test_observation_operator_updates_predictive_histories(self) -> None:
        bundle = build_topos_oom_bundle(
            "red blue red blue red green",
            max_history_length=1,
            max_test_length=1,
            max_histories_per_context=6,
            max_tests_per_context=6,
            max_operator_observations=3,
        )

        corpus = next(row for row in bundle["local_hankel_family"] if row["context_id"] == "corpus")
        red_operator = next(row for row in corpus["observation_operators"] if row["observation"] == "red")
        transitions = {
            (entry["source_history"], entry["target_history"]): entry["probability"]
            for entry in red_operator["entries"]
        }
        self.assertGreater(transitions.get(("epsilon", "red"), 0.0), 0.0)

    def test_materializes_json_and_html_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_topos_oom_bundle(
                [{"document_id": "d1", "text": "Tokens predict hidden state. Tokens reveal state."}],
                outdir=Path(tmpdir),
                corpus_label="html corpus",
            )

            json_path = paths["topos_oom_path"]
            html_path = paths["topos_oom_html_path"]
            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["corpus_label"], "html corpus")
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Topos OOM Bundle", html)
            self.assertIn("Observer Operator Model", html)

    def test_html_renderer_links_raw_bundle(self) -> None:
        bundle = build_topos_oom_bundle("one token follows another token")
        html = render_topos_oom_bundle_html(bundle, raw_json_href="raw.json")
        self.assertIn('href="raw.json"', html)
        self.assertIn("Local Observation Contexts", html)

    def test_perplexity_curve_improves_when_history_predicts_tokens(self) -> None:
        payload = evaluate_topos_oom_perplexity(
            [{"text": "red blue red blue red blue"}],
            [{"text": "red blue red blue"}],
            max_history_length=1,
            alpha=0.01,
        )

        curve = {row["history_length"]: row for row in payload["curve"]}
        self.assertEqual(payload["semantics"], "topos_oom_next_observation_perplexity_v1")
        self.assertGreater(curve[0]["perplexity"], curve[1]["perplexity"])
        self.assertEqual(payload["best"]["history_length"], 1)
        self.assertGreater(payload["vocabulary_size"], 1)


if __name__ == "__main__":
    unittest.main()
