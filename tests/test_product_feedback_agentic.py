"""Tests for the product-feedback agentic scaffold."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3 import (
        ProductFeedbackAgenticConfig,
        ProductFeedbackAgenticRunner,
        build_product_feedback_agentic_workflow,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import (
        ProductFeedbackAgenticConfig,
        ProductFeedbackAgenticRunner,
        build_product_feedback_agentic_workflow,
    )


class ProductFeedbackAgenticTests(unittest.TestCase):
    def test_product_usage_family_detects_vehicle_queries(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        family = module._product_usage_family(
            "Tesla Model 3",
            "Tesla",
            ["The steering is easy, charging is convenient, and the car is comfortable on long drives."],
        )

        self.assertEqual(family, "vehicle")

    def test_article_rating_extraction_handles_percent_and_fraction_formats(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "Nike Pegasus 41 Review",
            "OUR VERDICT: 79% - GOOD",
        )
        self.assertEqual((raw, scale), (79.0, 100.0))

        raw, scale = module._extract_article_rating_from_text(
            "Lovesac Sactionals Sofa Review",
            "Product Overview Sofa Overall Score 4.2/5 Pros Cons",
        )
        self.assertEqual((raw, scale), (4.2, 5.0))

        raw, scale = module._extract_article_rating_from_text(
            "Lovesac Sactionals Sofa Review",
            "Product Overview Sofa Overall Score Pros Cons Ideal For Lovesac Sactionals Sofa 4.2/5 Modular layout options",
        )
        self.assertEqual((raw, scale), (4.2, 5.0))

    def test_article_rating_extraction_handles_expert_score_phrase(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "Nike Pegasus 41 review",
            "Nike Pegasus 41 review 7 expert score 7 user's score",
        )
        self.assertEqual((raw, scale), (7.0, 10.0))

    def test_article_rating_extraction_handles_out_of_stars_phrase(self) -> None:
        try:
            from functorflow_v3 import product_feedback_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_agentic as module

        raw, scale = module._extract_article_rating_from_text(
            "My Lovesac Sactionals Review",
            "I give it 5 out of 5 stars for the form, function and comfort and I'd order it all over again!",
        )
        self.assertEqual((raw, scale), (5.0, 5.0))

    def test_workflow_parallel_frontiers_reflect_feedback_pipeline(self) -> None:
        workflow = build_product_feedback_agentic_workflow()

        frontiers = tuple(tuple(agent.name for agent in frontier) for frontier in workflow.parallel_frontiers())

        self.assertEqual(frontiers[0], ("feedback_collection_agent",))
        self.assertEqual(frontiers[1], ("feedback_normalization_agent",))
        self.assertEqual(frontiers[2], ("usage_workflow_agent", "aspect_grounding_agent", "outcome_signal_agent"))
        self.assertEqual(frontiers[3], ("causal_hypothesis_agent",))
        self.assertEqual(frontiers[4], ("success_scoring_agent",))
        self.assertEqual(frontiers[5], ("ablation_comparison_agent",))
        self.assertEqual(frontiers[-1], ("executive_summary_agent",))

    def test_runner_flags_fit_risk_and_recommend_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "r1",
                                "title": "Love the comfort",
                                "text": "Very comfortable and stylish. True to size and easy to slip on.",
                                "rating": 5,
                                "source": "reviews",
                                "source_reference": "https://example.com/review/r1",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r2",
                                "title": "Too tight",
                                "text": "These were too tight in the toe box and I returned them.",
                                "rating": 2,
                                "source": "reviews",
                                "returned": True,
                                "source_reference": "https://example.com/review/r2",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r3",
                                "title": "Runs narrow",
                                "text": "Nice style but the fit runs small and narrow. I had to send it back.",
                                "rating": 1,
                                "source": "qna",
                                "source_reference": "https://example.com/review/r3",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r4",
                                "title": "Convenient for travel",
                                "text": "Easy to slip on and comfortable for airport use.",
                                "rating": 4,
                                "source": "social",
                                "source_reference": "https://example.com/review/r4",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r5",
                                "title": "Not worth the price",
                                "text": "Overpriced and poor quality for the money.",
                                "rating": 2,
                                "source": "reviews",
                                "source_reference": "https://example.com/review/r5",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Slip-On Sneaker",
                    brand_name="Amazon Basics",
                )
            )
            result = runner.run()

            self.assertTrue(result.success_scorecard_path.exists())
            self.assertTrue(result.usage_workflows_path.exists())
            self.assertTrue(result.ablation_comparison_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.dashboard_path.exists())
            self.assertTrue(result.dashboard_summary_path.exists())

            scorecard = json.loads(result.success_scorecard_path.read_text(encoding="utf-8"))
            self.assertEqual(scorecard["verdict"], "at_risk")
            self.assertTrue(scorecard["return_warning_recommended"])
            self.assertIn("fit", scorecard["top_return_risk_aspects"])
            self.assertIn("comfort", scorecard["top_positive_aspects"])

            hypotheses = json.loads(result.causal_hypotheses_path.read_text(encoding="utf-8"))
            hypothesis_sources = {item["src"] for item in hypotheses["hypotheses"]}
            self.assertIn("tight or inconsistent fit perception", hypothesis_sources)
            self.assertIn("run-time usage friction", hypothesis_sources)

            usage_workflows = json.loads(result.usage_workflows_path.read_text(encoding="utf-8"))
            top_motifs = [" -> ".join(row["workflow_stages"]) for row in usage_workflows["top_workflow_motifs"]]
            self.assertTrue(any("wear" in motif or "run" in motif for motif in top_motifs))

            ablation = json.loads(result.ablation_comparison_path.read_text(encoding="utf-8"))
            ablation_labels = [row["label"] for row in ablation["rows"]]
            self.assertEqual(ablation_labels[:2], ["Prompt-like baseline", "BAFFLE structured scaffold"])
            self.assertTrue(any("score delta" in item.lower() for item in ablation["takeaways"]))

            report = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Often returned due to fit issues", report)
            self.assertIn("Which product aspects most strongly appear to drive return risk?", report)
            self.assertIn("Ablation Comparison", report)
            self.assertIn("Quantitative Takeaways", report)
            self.assertIn("Usage Workflows", report)

            dashboard = result.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("CLIFF Product Feedback Dashboard", dashboard)
            self.assertIn("Outcome Snapshot", dashboard)
            self.assertIn("Ablation Comparison", dashboard)
            self.assertIn("Prompt-like baseline", dashboard)
            self.assertIn("Causal Hypotheses", dashboard)
            self.assertIn("Usage Workflows", dashboard)
            self.assertIn("Evidence Preview", dashboard)
            self.assertIn("Open source", dashboard)

    def test_runner_preserves_source_reference_in_normalized_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "r1",
                        "title": "Comfort review",
                        "text": "Comfortable and easy to use.",
                        "rating": 4,
                        "source": "reviews",
                        "source_reference": "https://example.com/review-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Demo Product",
                    brand_name="Demo Brand",
                )
            )
            result = runner.run()

            rows = [
                json.loads(line)
                for line in result.normalized_feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(rows[0]["source_reference"], "https://example.com/review-1")

    def test_runner_builds_vehicle_workflows_without_sofa_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "vehicle_feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "v1",
                                "title": "Great commuter EV",
                                "text": "Easy to drive in traffic, comfortable on long commutes, and charging at home is simple.",
                                "rating": 5,
                                "source": "reviews",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "v2",
                                "title": "Road trip favorite",
                                "text": "The Tesla Model 3 handles well, the steering feels precise, and Supercharger stops are convenient.",
                                "rating": 4,
                                "source": "reviews",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Tesla Model 3",
                    brand_name="Tesla",
                )
            )
            result = runner.run()

            usage_workflows = json.loads(result.usage_workflows_path.read_text(encoding="utf-8"))
            top_motifs = [" -> ".join(row["workflow_stages"]) for row in usage_workflows["top_workflow_motifs"]]
            self.assertTrue(any("drive" in motif for motif in top_motifs))
            self.assertTrue(any("charge" in motif for motif in top_motifs))
            self.assertTrue(all("assemble" not in motif for motif in top_motifs))
            self.assertTrue(all("sit" not in motif for motif in top_motifs))
            self.assertTrue(all("wash" not in motif for motif in top_motifs))

    def test_runner_normalizes_ratings_from_multiple_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            manifest_path = workdir / "feedback.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "r1",
                                "title": "Five-star equivalent",
                                "text": "Comfortable and durable.",
                                "rating": 4,
                                "rating_scale": 5,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r2",
                                "title": "Ten-point equivalent",
                                "text": "Comfortable and supportive.",
                                "rating": 8,
                                "rating_scale": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "r3",
                                "title": "Percent equivalent",
                                "text": "Stylish and worth it.",
                                "rating": 80,
                                "rating_scale": 100,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackAgenticRunner(
                ProductFeedbackAgenticConfig(
                    manifest_path=manifest_path,
                    outdir=workdir / "out",
                    product_name="Scaled Sneaker",
                    brand_name="FF2",
                )
            )
            result = runner.run()

            outcome = json.loads(result.outcome_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(outcome["average_rating"], 4.0)

            normalized_rows = [
                json.loads(line)
                for line in result.normalized_feedback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["rating"] for row in normalized_rows], [4.0, 4.0, 4.0])


if __name__ == "__main__":
    unittest.main()
