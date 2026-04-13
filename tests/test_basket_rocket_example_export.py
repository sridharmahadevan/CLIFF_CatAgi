"""Tests for compact public BASKET/ROCKET example exports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3.basket_rocket_example_export import export_basket_rocket_example
except ModuleNotFoundError:
    from ..functorflow_v3.basket_rocket_example_export import export_basket_rocket_example


class BasketRocketExampleExportTests(unittest.TestCase):
    def test_export_writes_compact_bundle_without_absolute_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extractor_dir = root / "extractor"
            reranking_dir = root / "reranking"
            company_viz_dir = root / "viz"
            psr_company_dir = root / "psr"
            for path in (extractor_dir, reranking_dir, company_viz_dir, psr_company_dir):
                path.mkdir(parents=True)

            (extractor_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "source_mode": "raw_pdf",
                        "extractor_mode": "heuristic",
                        "n_statement_files": 599,
                        "n_company_years": 599,
                        "n_statement_rows": 402037,
                        "n_extractions": 56498,
                        "n_steps": 140903,
                        "avg_actions_per_plan": 2.49,
                        "macro_skill_count": 128,
                        "action_vocab_size": 17,
                    }
                ),
                encoding="utf-8",
            )
            (extractor_dir / "plan_block_manifest.json").write_text(
                json.dumps(
                    {
                        "statement_globs": ["**/runs_*_financial_filings/year_*/runs/*/input.pdf"],
                        "paths": {
                            "workflow_extractions": str(root / "private" / "workflow_extractions.jsonl"),
                            "macro_skills": str(root / "private" / "macro_skills.csv"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (reranking_dir / "reranked_summary.json").write_text(
                json.dumps(
                    {
                        "reward_mode": "financial",
                        "financial_targets": ["+revenue_yoy", "+operating_margin"],
                        "financial_horizon": "next_year",
                        "n_rows": 56498,
                        "n_changed": 11197,
                        "changed_rate": 0.198,
                        "mean_score_gain": 0.0037,
                        "changed_mean_score_gain": 0.0190,
                        "selected_sources": {"basket": 45301, "macro_merge": 5063},
                        "selected_labels": {"base": 45301, "panel_macro": 3402},
                        "changed_by_company_top10": {"adobe": 281},
                        "inputs": {
                            "extractions": str(root / "private" / "workflow_extractions.jsonl"),
                            "panel": str(root / "private" / "rocket_company_outcomes.csv"),
                        },
                        "top_changed_examples": [
                            {
                                "statement_id": "adobe:y2008:p0014",
                                "company": "adobe",
                                "year": 2008,
                                "score_gain": 0.1086,
                                "base_actions": ["innovate", "digitize", "optimize"],
                                "selected_actions": ["innovate", "digitize", "optimize", "expand", "realize_revenue"],
                            },
                            {
                                "statement_id": "apple:y2008:p0010",
                                "company": "apple",
                                "year": 2008,
                                "score_gain": 0.1089,
                                "base_actions": ["innovate", "digitize"],
                                "selected_actions": ["innovate", "digitize", "expand"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (company_viz_dir / "index_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "company": "adobe",
                            "html": "rocket_reranking_visualizer___adobe.html",
                            "summary": "rocket_reranking_visualizer___adobe_summary.json",
                            "n_rows": 3545,
                            "n_changed": 281,
                            "changed_rate": 0.0792,
                            "mean_score_gain": 0.0279,
                            "financial_summary": {
                                "coverage_years": 23,
                                "latest_year": 2026,
                                "ticker": "ADBE",
                            },
                            "aggregate_plan": {
                                "year_count": 23,
                                "top_actions": [
                                    {"action": "optimize", "count": 172},
                                    {"action": "expand", "count": 141},
                                ],
                                "top_edges": [
                                    {"src": "optimize", "dst": "realize_revenue", "count": 72},
                                ],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (company_viz_dir / "rocket_reranking_visualizer___adobe_summary.json").write_text(
                json.dumps(
                    {
                        "n_rows": 3545,
                        "n_changed": 281,
                        "changed_rate": 0.0792,
                        "mean_score_gain": 0.0279,
                        "selected_sources": {"macro_merge": 104},
                    }
                ),
                encoding="utf-8",
            )
            (company_viz_dir / "rocket_reranking_visualizer___adobe.html").write_text(
                "<html><body>Visualizer /tmp/private/source.pdf</body></html>",
                encoding="utf-8",
            )
            (company_viz_dir / "rocket_aggregate_plans___adobe.html").write_text(
                '<html><body><a href="index.html">Back</a> /tmp/private/source.pdf</body></html>',
                encoding="utf-8",
            )
            (psr_company_dir / "adobe.html").write_text(
                "<html><body><img src='adobe_timeline.png'/> /tmp/private/source.pdf</body></html>",
                encoding="utf-8",
            )
            (psr_company_dir / "adobe_timeline.png").write_bytes(b"fake-png")

            output_dir = root / "examples" / "basket_rocket" / "adobe_financial_reranking"
            manifest = export_basket_rocket_example(
                company="adobe",
                extractor_dir=extractor_dir,
                reranking_dir=reranking_dir,
                company_viz_dir=company_viz_dir,
                output_dir=output_dir,
                force=True,
                psr_company_dir=psr_company_dir,
            )

            self.assertEqual(manifest["route"], "basket_rocket_sec")
            self.assertEqual(manifest["company"], "adobe")
            self.assertEqual(manifest["company_changed_count"], 281)
            self.assertIn("psr_drilldown.html", manifest["included_visualizations"])
            self.assertIn("timeline.png", manifest["included_images"])

            readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("BASKET/ROCKET Example Bundle", readme_text)
            self.assertIn("basket_rocket_sec", readme_text)
            self.assertNotIn(str(root), readme_text)

            company_payload = json.loads((output_dir / "company_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(company_payload["company"], "adobe")
            self.assertEqual(company_payload["summary_metrics"]["n_changed"], 281)
            self.assertNotIn(str(root), json.dumps(company_payload))

            top_examples = json.loads((output_dir / "top_changed_examples.json").read_text(encoding="utf-8"))
            self.assertEqual(len(top_examples), 1)
            self.assertEqual(top_examples[0]["company"], "adobe")

            company_html = (output_dir / "visualizations" / "company_reranking.html").read_text(encoding="utf-8")
            aggregate_html = (output_dir / "visualizations" / "aggregate_plans.html").read_text(encoding="utf-8")
            psr_html = (output_dir / "visualizations" / "psr_drilldown.html").read_text(encoding="utf-8")
            self.assertNotIn(str(root), company_html)
            self.assertIn('href="company_reranking.html"', aggregate_html)
            self.assertIn("../images/timeline.png", psr_html)
            self.assertTrue((output_dir / "images" / "timeline.png").exists())


if __name__ == "__main__":
    unittest.main()
