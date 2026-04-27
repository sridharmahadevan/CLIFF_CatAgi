"""Tests for Democritus causal PSR construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3.democritus_psr import DemocritusPSRSource, build_democritus_topos_psr_bundle
    from functorflow_v3.democritus_agentic import DemocritusAgentRecord
    from functorflow_v3.democritus_batch_agentic import DemocritusBatchAgenticRunner, DemocritusBatchConfig, DemocritusBatchRecord
except ModuleNotFoundError:
    from ..functorflow_v3.democritus_psr import DemocritusPSRSource, build_democritus_topos_psr_bundle
    from ..functorflow_v3.democritus_agentic import DemocritusAgentRecord
    from ..functorflow_v3.democritus_batch_agentic import DemocritusBatchAgenticRunner, DemocritusBatchConfig, DemocritusBatchRecord


class DemocritusPSRTests(unittest.TestCase):
    def test_builds_corpus_and_domain_hankel_contexts_from_triples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alpha = root / "alpha.jsonl"
            beta = root / "beta.jsonl"
            alpha.write_text(
                "\n".join(
                    [
                        json.dumps({"subj": "Training", "rel": "increases", "obj": "Accuracy", "domain": "ML"}),
                        json.dumps({"subj": "Noise", "rel": "reduces", "obj": "Accuracy", "domain": "ML"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            beta.write_text(
                json.dumps({"subj": "Subsidies", "rel": "affects", "obj": "Adoption", "domain": "Policy"}) + "\n",
                encoding="utf-8",
            )

            bundle = build_democritus_topos_psr_bundle(
                [
                    DemocritusPSRSource(run_name="alpha", triples_path=alpha),
                    DemocritusPSRSource(run_name="beta", triples_path=beta),
                ],
                corpus_label="causal PSR smoke",
            )

        self.assertEqual(bundle["bundle_type"], "democritus_causal_psr")
        self.assertEqual(bundle["summary"]["n_episodes"], 2)
        self.assertEqual(bundle["summary"]["n_events"], 3)
        context_ids = {row["context_id"] for row in bundle["local_hankel_family"]}
        self.assertIn("corpus", context_ids)
        self.assertIn("domain::ml", context_ids)
        self.assertGreaterEqual(bundle["summary"]["n_restriction_checks"], 1)
        corpus = next(row for row in bundle["local_hankel_family"] if row["context_id"] == "corpus")
        self.assertTrue(corpus["top_histories"])
        self.assertTrue(corpus["top_tests"])
        self.assertIn("episode_index", bundle)
        self.assertNotIn("episodes", bundle)
        self.assertGreaterEqual(bundle["summary"]["n_psr_test_witnesses"], 1)
        self.assertTrue(bundle["claim_test_witnesses"])
        self.assertEqual(bundle["claim_test_witnesses"][0]["semantics"], "psr_test_cell_witness_v1")

    def test_batch_runner_materializes_topos_psr_bundle_from_triple_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            triples = root / "relational_triples.jsonl"
            triples.write_text(
                json.dumps({"subj": "Rain", "rel": "increases", "obj": "Crop growth", "domain": "Agriculture"}) + "\n",
                encoding="utf-8",
            )
            runner = DemocritusBatchAgenticRunner(
                DemocritusBatchConfig(
                    pdf_dir=root,
                    outdir=root / "runs",
                    discover_existing_documents=False,
                    request_query="agriculture evidence",
                )
            )
            record = DemocritusBatchRecord(
                run_name="run_alpha",
                pdf_path=str(root / "alpha.pdf"),
                agent_record=DemocritusAgentRecord(
                    agent_name="triple_extraction_agent",
                    frontier_index=0,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples),),
                ),
            )

            bundle_path = runner._build_topos_psr_bundle((record,))

            self.assertIsNotNone(bundle_path)
            assert bundle_path is not None
            self.assertTrue(bundle_path.exists())
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["corpus_label"], "agriculture evidence")
            self.assertEqual(payload["summary"]["n_events"], 1)
            self.assertNotIn("episodes", payload)
            self.assertTrue((bundle_path.parent / "democritus_psr_episodes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
