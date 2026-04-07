"""Tests for course demo matching and recommendation mode."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from functorflow_v3 import course_demo_agentic as module
except ModuleNotFoundError:
    from ..functorflow_v3 import course_demo_agentic as module


class CourseDemoAgenticTests(unittest.TestCase):
    def test_recommend_course_demos_for_causality(self) -> None:
        topic, demos, rationale = module.recommend_course_demos("What demo should I use for causality?")

        self.assertEqual(topic, "Causality")
        self.assertGreaterEqual(len(demos), 2)
        self.assertEqual(demos[0].demo_id, "causal_discovery_toy")
        self.assertIn("causal", rationale.lower())

    def test_looks_like_course_demo_query_for_recommendation(self) -> None:
        self.assertTrue(module.looks_like_course_demo_query("Which demo should I look at for causality?"))

    def test_match_julia_demo_for_ket(self) -> None:
        demo = module.match_julia_demo("Show me the Julia version of KET")

        self.assertIsNotNone(demo)
        self.assertEqual(demo.demo_id, "julia_ket_block")

    def test_recommend_julia_demos_for_ket(self) -> None:
        topic, demos, rationale = module.recommend_julia_demos("Which Julia demo should I use for KET?")

        self.assertEqual(topic, "Julia KET and Kan Extensions")
        self.assertGreaterEqual(len(demos), 1)
        self.assertIn("Julia", rationale)

    def test_recommend_course_project_ideas_for_ket(self) -> None:
        topic, ideas, starter_demo, book_sections, rationale = module.recommend_course_project_ideas(
            "I would like a project suggestion that applies the Kan Extension Transformer"
        )

        self.assertEqual(topic, "Kan Extension Transformers")
        self.assertGreaterEqual(len(ideas), 2)
        self.assertIsNotNone(starter_demo)
        self.assertEqual(starter_demo.demo_id, "kan_extension_transformer")
        self.assertGreaterEqual(len(book_sections), 2)
        self.assertIn("ket", rationale.lower())

    def test_recommend_course_learning_resources_for_ket(self) -> None:
        topic, demos, book_sections, snippets, rationale = module.recommend_course_learning_resources(
            "I would like to learn about the Kan Extension Transformer"
        )

        self.assertEqual(topic, "Kan Extension Transformers")
        self.assertGreaterEqual(len(demos), 1)
        self.assertGreaterEqual(len(book_sections), 2)
        self.assertGreaterEqual(len(snippets), 2)
        self.assertTrue(any(snippet.language == "python" for snippet in snippets))
        self.assertTrue(any(snippet.language == "julia" for snippet in snippets))
        self.assertIn("ket", rationale.lower())

    def test_recommend_book_sections_for_gt_sudoku_query(self) -> None:
        sections, rationale = module.recommend_book_sections(
            "Explain the Geometric Transformer on the Sudoku problem",
            matched_demo_id="geometric_transformer_sudoku",
        )

        self.assertGreaterEqual(len(sections), 1)
        self.assertEqual(sections[0].section_id, "geometric_transformers")
        self.assertIn("matched demo", rationale.lower())

    def test_product_assembly_query_does_not_false_match_course_demo(self) -> None:
        self.assertFalse(module.looks_like_course_demo_query("How easy is to assemble a Lovesac sectional sofa?"))
        self.assertIsNone(module.match_course_demo("How easy is to assemble a Lovesac sectional sofa?"))

    def test_looks_like_course_demo_query_for_project_request(self) -> None:
        self.assertTrue(
            module.looks_like_course_demo_query(
                "I would like a project suggestion that applies the Kan Extension Transformer"
            )
        )

    def test_looks_like_course_demo_query_for_learning_request(self) -> None:
        self.assertTrue(
            module.looks_like_course_demo_query(
                "I would like to learn about the Kan Extension Transformer"
            )
        )

    def test_explain_how_ket_works_counts_as_learning_request(self) -> None:
        self.assertTrue(module.is_course_learning_query("Explain how the Kan Extension Transformer works"))

    def test_explain_gt_on_sudoku_is_not_learning_query(self) -> None:
        self.assertFalse(module.is_course_learning_query("Explain the Geometric Transformer on the Sudoku problem"))

    def test_course_demo_runner_recommendation_mode_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="What demo should I look at for causality?",
                    outdir=Path(tmpdir),
                    execute_demo=True,
                )
            ).run()
            dashboard_exists = result.dashboard_path.exists()

        self.assertEqual(result.response_mode, "recommendation")
        self.assertEqual(result.execution_status, "recommended")
        self.assertFalse(result.execution_attempted)
        self.assertTrue(dashboard_exists)
        self.assertGreaterEqual(len(result.recommendation_demos), 2)
        self.assertGreaterEqual(len(result.book_recommendations), 1)

    def test_course_demo_runner_project_mode_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="I would like a project suggestion that applies the Kan Extension Transformer",
                    outdir=Path(tmpdir),
                    execute_demo=True,
                )
            ).run()

        self.assertEqual(result.response_mode, "project_ideas")
        self.assertEqual(result.execution_status, "recommended")
        self.assertFalse(result.execution_attempted)
        self.assertEqual(result.project_topic, "Kan Extension Transformers")
        self.assertGreaterEqual(len(result.project_ideas), 2)
        self.assertGreaterEqual(len(result.book_recommendations), 2)

    def test_course_demo_runner_learning_mode_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="I would like to learn about the Kan Extension Transformer",
                    outdir=Path(tmpdir),
                    execute_demo=True,
                )
            ).run()

        self.assertEqual(result.response_mode, "learning_guide")
        self.assertEqual(result.execution_status, "recommended")
        self.assertFalse(result.execution_attempted)
        self.assertEqual(result.recommendation_topic, "Kan Extension Transformers")
        self.assertGreaterEqual(len(result.code_snippets), 2)
        self.assertGreaterEqual(len(result.book_recommendations), 2)

    def test_course_demo_runner_explain_how_ket_works_returns_learning_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="Explain how the Kan Extension Transformer works",
                    outdir=Path(tmpdir),
                    execute_demo=True,
                )
            ).run()

        self.assertEqual(result.response_mode, "learning_guide")
        self.assertGreaterEqual(len(result.code_snippets), 2)

    def test_course_demo_runner_gt_sudoku_prefers_demo_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            notebook_path = workdir / "notebooks" / "week01_sudoku_gt_db.ipynb"
            notebook_path.parent.mkdir(parents=True, exist_ok=True)
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "source": ["print('sudoku demo')\n"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="Explain the Geometric Transformer on the Sudoku problem",
                    outdir=workdir / "out",
                    course_repo_root=workdir,
                    execute_demo=False,
                )
            ).run()

        self.assertEqual(result.response_mode, "demo_run")
        self.assertEqual(result.execution_status, "not_requested")
        self.assertEqual(result.selected_demo.demo_id, "geometric_transformer_sudoku")

    def test_course_demo_runner_julia_recommendation_mode_skips_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = module.CourseDemoAgenticRunner(
                module.CourseDemoAgenticConfig(
                    query="Which Julia demo should I use for KET?",
                    outdir=Path(tmpdir),
                    execute_demo=True,
                )
            ).run()
            dashboard_exists = result.dashboard_path.exists()

        self.assertEqual(result.response_mode, "recommendation")
        self.assertEqual(result.execution_status, "recommended")
        self.assertEqual(result.implementation_language, "julia")
        self.assertFalse(result.execution_attempted)
        self.assertTrue(dashboard_exists)
        self.assertGreaterEqual(len(result.recommendation_julia_demos), 1)
        self.assertGreaterEqual(len(result.book_recommendations), 1)

    def test_execution_output_copy_mentions_julia_for_julia_runs(self) -> None:
        result = module.CourseDemoRunResult(
            query_plan=module.CourseDemoQueryPlan(
                query="Show me the Julia version of KET",
                normalized_query="show me the julia version of ket",
                explanation_focus="Use Julia.",
            ),
            selected_demo=module.match_julia_demo("Show me the Julia version of KET"),
            route_outdir=Path("/tmp"),
            notebook_path=None,
            generated_script_path=None,
            dashboard_path=Path("/tmp/course_demo_dashboard.html"),
            summary_path=Path("/tmp/course_demo_summary.json"),
            response_mode="demo_run",
            execution_attempted=True,
            execution_status="completed",
            stdout_path=Path("/tmp/course_demo_stdout.txt"),
            stderr_path=Path("/tmp/course_demo_stderr.txt"),
            implementation_language="julia",
        )

        copy = module._execution_output_copy(result)
        self.assertIn("Julia FunctorFlow demo", copy)


if __name__ == "__main__":
    unittest.main()
