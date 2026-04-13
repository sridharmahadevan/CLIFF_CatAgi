"""Tests for compact public Democritus example exports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3.democritus_example_export import export_democritus_example
except ModuleNotFoundError:
    from ..functorflow_v3.democritus_example_export import export_democritus_example


class DemocritusExampleExportTests(unittest.TestCase):
    def test_export_writes_compact_bundle_without_absolute_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "saved_run"
            batch_outdir = run_dir / "democritus_runs"
            batch_outdir.mkdir(parents=True)

            run_a = batch_outdir / "run_a"
            (run_a / "reports").mkdir(parents=True)
            (run_a / "viz").mkdir(parents=True)
            (run_a / "reports" / "run_a_executive_summary.md").write_text(
                "\n".join(
                    [
                        "## Tier 1 Claims",
                        "",
                        "**1. (1.00) warming --leads_to--> fish decline**",
                        "> Ocean warming reduces fish population resilience.",
                        "",
                        "**2. (0.95) habitat stress --reduces--> survival**",
                        "> Habitat stress reduces juvenile fish survival.",
                        "",
                        "## Tier 2 Claims",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_a / "viz" / "relational_manifold_2d.png").write_bytes(b"fake-png")

            run_b = batch_outdir / "run_b"
            (run_b / "reports").mkdir(parents=True)

            query_summary = {
                "query_plan": {
                    "query": "Analyze recent studies of fish population decline under ocean warming",
                    "normalized_query": "fish population decline ocean warming",
                    "keyword_tokens": ["fish", "population", "decline", "ocean", "warming"],
                    "target_documents": 2,
                    "retrieval_query": "fish population decline ocean warming",
                    "direct_document_paths": [str(root / "private.pdf")],
                    "direct_document_directories": [str(root / "raw_docs")],
                },
                "execution_mode": "deep",
                "retrieval_backend": "scholarly",
                "batch_outdir": str(batch_outdir),
                "selected_documents": [
                    {
                        "title": "Ocean warming and declining fish resilience",
                        "year": "2025",
                        "score": 9.7,
                        "retrieval_backend": "europe_pmc",
                        "identifier": "PMC123",
                        "url": "https://example.org/study-a",
                        "download_url": "https://example.org/study-a.pdf",
                        "abstract": "Study A abstract.",
                        "evidence": ["title:warming"],
                    },
                    {
                        "title": "Habitat shifts and coastal fish decline",
                        "year": "2024",
                        "score": 8.1,
                        "retrieval_backend": "crossref",
                        "identifier": "10.1000/example",
                        "url": "https://example.org/study-b",
                        "download_url": "",
                        "abstract": "Study B abstract.",
                        "evidence": ["text:fish"],
                    },
                ],
            }
            (run_dir / "query_run_summary.json").write_text(json.dumps(query_summary), encoding="utf-8")

            batch_records = [
                {
                    "run_name": "run_a",
                    "agent_record": {
                        "agent_name": "causal_statement_agent",
                        "status": "ok",
                        "started_at": 10.0,
                        "ended_at": 13.5,
                    },
                },
                {
                    "run_name": "run_b",
                    "agent_record": {
                        "agent_name": "causal_statement_agent",
                        "status": "ok",
                        "started_at": 20.0,
                        "ended_at": 24.0,
                    },
                },
                {
                    "run_name": "run_a",
                    "agent_record": {
                        "agent_name": "credibility_bundle_agent",
                        "status": "ok",
                        "started_at": 30.0,
                        "ended_at": 31.0,
                    },
                },
            ]
            (batch_outdir / "batch_agent_run_summary.json").write_text(json.dumps(batch_records), encoding="utf-8")

            output_dir = root / "examples" / "democritus" / "fish_run"
            manifest = export_democritus_example(
                run_dir,
                output_dir,
                copy_manifold_images=1,
                document_ranks=(2,),
            )

            self.assertEqual(manifest["selected_document_count"], 1)
            self.assertEqual(manifest["source_selected_document_count"], 2)
            self.assertEqual(manifest["included_image_count"], 0)
            self.assertEqual(manifest["document_ranks"], [2])

            readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("Democritus Example Bundle", readme_text)
            self.assertIn("Habitat shifts and coastal fish decline", readme_text)
            self.assertIn("Public release subset", readme_text)
            self.assertNotIn(str(root), readme_text)

            query_plan_payload = json.loads((output_dir / "query_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(query_plan_payload["direct_document_paths"], ["private.pdf"])
            self.assertEqual(query_plan_payload["direct_document_directories"], ["raw_docs"])

            selected_payload = json.loads((output_dir / "selected_documents.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selected_payload), 1)
            self.assertEqual(
                selected_payload[0]["top_tier1_claims"],
                [],
            )
            self.assertIsNone(selected_payload[0]["manifold_image_path"])
            self.assertNotIn(str(root), json.dumps(selected_payload))

            stage_summary = json.loads((output_dir / "batch_stage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stage_summary[0]["agent_name"], "causal_statement_agent")
            self.assertEqual(stage_summary[0]["completed_records"], 1)
            self.assertEqual(stage_summary[0]["document_count"], 1)

            self.assertFalse((output_dir / "documents" / "01_ocean-warming-and-declining-fish-resilience.md").exists())
            self.assertTrue((output_dir / "documents" / "01_habitat-shifts-and-coastal-fish-decline.md").exists())


if __name__ == "__main__":
    unittest.main()
