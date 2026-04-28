"""Tests for product-review retrieval plus feedback analysis."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import (
        ProductFeedbackQueryAgenticConfig,
        ProductFeedbackQueryAgenticRunner,
    )
    from functorflow_v3.product_feedback_agentic import ProductFeedbackRunResult
    from functorflow_v3.product_feedback_query_agentic import (
        DiscoveredReviewDocument,
        ReviewConsensusSnapshot,
        ReviewQueryPlan,
        WebSearchReviewRetrievalBackend,
    )
    from functorflow_v3.product_feedback_state import materialize_product_feedback_state_transition
    from functorflow_v3.product_feedback_state import materialize_product_feedback_world_state
except ModuleNotFoundError:
    from ..functorflow_v3 import (
        ProductFeedbackQueryAgenticConfig,
        ProductFeedbackQueryAgenticRunner,
    )
    from ..functorflow_v3.product_feedback_agentic import ProductFeedbackRunResult
    from ..functorflow_v3.product_feedback_query_agentic import (
        DiscoveredReviewDocument,
        ReviewConsensusSnapshot,
        ReviewQueryPlan,
        WebSearchReviewRetrievalBackend,
    )
    from ..functorflow_v3.product_feedback_state import materialize_product_feedback_state_transition
    from ..functorflow_v3.product_feedback_state import materialize_product_feedback_world_state


class ProductFeedbackQueryAgenticTests(unittest.TestCase):
    def test_world_state_recommends_counterfactual_probe_when_only_exploratory_repairs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            psr_path = root / "topos_psr_hankel.json"
            scorecard_path = root / "scorecard.json"
            counterfactual_path = root / "prometheus_counterfactuals.json"
            psr_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "domain": "food",
                            "context_ids": ["review", "taste"],
                            "n_review_records": 6,
                            "n_context_projected_views": 12,
                            "n_contexts": 2,
                            "n_compatible_restrictions": 1,
                            "n_restriction_checks": 1,
                        },
                        "local_hankel_family": [
                            {
                                "context_id": "taste",
                                "n_episode_views": 6,
                                "histories": [{"signature": "epsilon"} for _ in range(4)],
                                "tests": [{"signature": f"taste_test_{index}"} for index in range(4)],
                                "svd": {"rank": 2},
                            }
                        ],
                        "restriction_diagnostics": [
                            {"source_context": "review", "target_context": "taste", "compatible": True, "max_abs_gap": 0.05}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            scorecard_path.write_text("{}", encoding="utf-8")
            counterfactual_path.write_text(
                json.dumps(
                    {
                        "summary": {"counterfactual_count": 0, "candidate_count": 0, "exploratory_count": 1},
                        "counterfactuals": [],
                        "exploratory_counterfactuals": [
                            {
                                "aspect": "taste",
                                "evidence_status": "domain_plausible",
                                "support_tier": "exploratory",
                                "estimated_satisfaction_gain": 0.08,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths = materialize_product_feedback_world_state(
                query="How tasty are the Amedei chocolates?",
                query_plan=SimpleNamespace(target_documents=5),
                summary_path=root / "summary.json",
                product_feedback_result=SimpleNamespace(
                    topos_psr_path=psr_path,
                    success_scorecard_path=scorecard_path,
                    prometheus_counterfactuals_path=counterfactual_path,
                ),
                outdir=root,
            )

            state = json.loads(paths["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(state["confidence"], "provisional")
            self.assertEqual(state["recommended_actions"][0]["action"], "counterfactual_probe")
            self.assertIn("j-do repair layer", state["reasons"][0])
            self.assertEqual(state["diagnostics"]["counterfactual_support"]["exploratory_count"], 1)
            html = paths["html_path"].read_text(encoding="utf-8")
            self.assertIn("Counterfactual State", html)
            self.assertIn("Probe taste repair evidence", html)

    def test_product_feedback_state_transition_detects_stability_with_residual_gluing_tension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent_psr = root / "parent_psr.json"
            current_psr = root / "current_psr.json"
            parent_psr.write_text(
                json.dumps(
                    {
                        "restriction_diagnostics": [
                            {"source_context": "use", "target_context": "sit", "compatible": False, "max_abs_gap": 0.3}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            current_psr.write_text(
                json.dumps(
                    {
                        "restriction_diagnostics": [
                            {"source_context": "use", "target_context": "sit", "compatible": False, "max_abs_gap": 0.28}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            parent_state = {
                "query": "How comfortable is the Lovesac sectional sofa?",
                "diagnostics": {
                    "query_context": "sit",
                    "topos_summary": {
                        "n_review_records": 6,
                        "n_context_projected_views": 48,
                        "n_contexts": 8,
                        "mean_rank": 5.5,
                        "n_compatible_restrictions": 6,
                        "n_restriction_checks": 7,
                    },
                },
                "source_artifacts": {"topos_psr_path": str(parent_psr)},
            }
            current_state = {
                "query": "How comfortable is the Lovesac sectional sofa?",
                "diagnostics": {
                    "query_context": "sit",
                    "topos_summary": {
                        "n_review_records": 8,
                        "n_context_projected_views": 64,
                        "n_contexts": 8,
                        "mean_rank": 5.625,
                        "n_compatible_restrictions": 6,
                        "n_restriction_checks": 7,
                    },
                },
                "source_artifacts": {"topos_psr_path": str(current_psr)},
            }
            parent_path = root / "parent_state.json"
            current_path = root / "current_state.json"
            parent_path.write_text(json.dumps(parent_state), encoding="utf-8")
            current_path.write_text(json.dumps(current_state), encoding="utf-8")

            paths = materialize_product_feedback_state_transition(
                parent_state_path=parent_path,
                current_state_path=current_path,
                outdir=root,
            )

            transition = json.loads(paths["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(transition["assessment"], "stable_with_residual_gluing_tension")
            self.assertEqual(transition["recommended_actions"][0]["action"], "inspect_residual_restriction")
            self.assertEqual(transition["recommended_actions"][0]["target"]["source_context"], "use")
            html = paths["html_path"].read_text(encoding="utf-8")
            self.assertIn("World State Transition", html)
            self.assertIn("stable_with_residual_gluing_tension", html)

    def test_product_feedback_state_transition_detects_stalled_counterfactual_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent_state = {
                "query": "How tasty are the Amedei chocolates?",
                "diagnostics": {
                    "query_context": "taste",
                    "query_context_support": {"episode_views": 6},
                    "counterfactual_support": {
                        "supported_count": 0,
                        "strict_candidate_count": 0,
                        "exploratory_count": 1,
                        "query_context_supported_count": 0,
                        "query_context_exploratory_count": 1,
                    },
                    "topos_summary": {
                        "n_review_records": 8,
                        "n_context_projected_views": 48,
                        "n_contexts": 8,
                        "mean_rank": 5.0,
                        "n_compatible_restrictions": 7,
                        "n_restriction_checks": 7,
                    },
                },
                "recommended_actions": [{"action": "counterfactual_probe"}],
                "source_artifacts": {},
            }
            current_state = {
                "query": "How tasty are the Amedei chocolates?",
                "diagnostics": {
                    "query_context": "taste",
                    "query_context_support": {"episode_views": 6},
                    "counterfactual_support": {
                        "supported_count": 0,
                        "strict_candidate_count": 0,
                        "exploratory_count": 1,
                        "query_context_supported_count": 0,
                        "query_context_exploratory_count": 1,
                    },
                    "topos_summary": {
                        "n_review_records": 8,
                        "n_context_projected_views": 48,
                        "n_contexts": 8,
                        "mean_rank": 5.0,
                        "n_compatible_restrictions": 7,
                        "n_restriction_checks": 7,
                    },
                },
                "recommended_actions": [{"action": "counterfactual_probe"}],
                "source_artifacts": {},
            }
            parent_path = root / "parent_state.json"
            current_path = root / "current_state.json"
            parent_path.write_text(json.dumps(parent_state), encoding="utf-8")
            current_path.write_text(json.dumps(current_state), encoding="utf-8")

            paths = materialize_product_feedback_state_transition(
                parent_state_path=parent_path,
                current_state_path=current_path,
                outdir=root,
            )

            transition = json.loads(paths["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(transition["assessment"], "counterfactual_probe_stalled")
            self.assertEqual(transition["recommended_actions"][0]["action"], "accept_descriptive_answer")
            self.assertIn("did not produce strict j-do repair candidates", transition["reasons"][0])

    def test_product_feedback_state_transition_detects_stalled_gluing_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent_state = {
                "query": "How comfortable are the Saucony Endorphin running shoes?",
                "diagnostics": {
                    "query_context": "use",
                    "query_context_support": {"episode_views": 10},
                    "query_context_restrictions": {"checks": 4, "compatible": 2, "max_gap": 0.9},
                    "counterfactual_support": {"supported_count": 1, "strict_candidate_count": 1, "exploratory_count": 0},
                    "topos_summary": {
                        "n_review_records": 10,
                        "n_context_projected_views": 80,
                        "n_contexts": 8,
                        "mean_rank": 5.0,
                        "n_compatible_restrictions": 5,
                        "n_restriction_checks": 7,
                    },
                },
                "source_artifacts": {},
            }
            current_state = {
                "query": "How comfortable are the Saucony Endorphin running shoes?",
                "diagnostics": {
                    "query_context": "use",
                    "query_context_support": {"episode_views": 6},
                    "query_context_restrictions": {"checks": 4, "compatible": 2, "max_gap": 0.833},
                    "counterfactual_support": {"supported_count": 1, "strict_candidate_count": 1, "exploratory_count": 0},
                    "topos_summary": {
                        "n_review_records": 10,
                        "n_context_projected_views": 64,
                        "n_contexts": 8,
                        "mean_rank": 5.0,
                        "n_compatible_restrictions": 5,
                        "n_restriction_checks": 7,
                    },
                },
                "source_artifacts": {},
            }
            parent_path = root / "parent_state.json"
            current_path = root / "current_state.json"
            parent_path.write_text(json.dumps(parent_state), encoding="utf-8")
            current_path.write_text(json.dumps(current_state), encoding="utf-8")

            paths = materialize_product_feedback_state_transition(
                parent_state_path=parent_path,
                current_state_path=current_path,
                outdir=root,
            )

            transition = json.loads(paths["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(transition["assessment"], "gluing_probe_stalled")
            self.assertEqual(transition["recommended_actions"][0]["action"], "accept_with_gluing_caveat")
            self.assertIn("did not improve query-context support", transition["reasons"][0])

    def test_query_runner_falls_back_to_example_manifest_after_search_timeout(self) -> None:
        class DiscoveryTimeoutRunner(ProductFeedbackQueryAgenticRunner):
            def _resolve_backend(self):
                class TimeoutBackend:
                    backend_name = "web_search"

                    def search(self, plan: ReviewQueryPlan, *, limit: int):
                        del plan, limit
                        raise socket.timeout("The read operation timed out")

                return TimeoutBackend()

            def _materialize_payload(self, document: DiscoveredReviewDocument):
                source_reference = document.url or document.source_path or "manifest://review"
                title = document.title
                return (
                    (
                        "<html><body><article>"
                        f"<h1>{title}</h1>"
                        "<p>The Lovesac Sactional is comfortable after setup, with some seat-depth tradeoffs.</p>"
                        "</article></body></html>"
                    ),
                    source_reference,
                    ".html",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DiscoveryTimeoutRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How comfortable is the Lovesac sectional sofa?",
                    outdir=Path(tmpdir) / "out",
                    target_documents=2,
                    product_name="Lovesac Sactional",
                    brand_name="Lovesac",
                )
            )

            result = runner.run()

            self.assertGreaterEqual(len(result.selected_documents), 2)
            self.assertTrue(all(item.retrieval_backend == "manifest" for item in result.selected_documents))
            self.assertIsNotNone(result.product_feedback_result)
            self.assertTrue(result.product_feedback_result.dashboard_path.exists())
            discovery_rows = [
                json.loads(line)
                for line in runner.discovery_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(discovery_rows), 2)
            self.assertTrue(all(row["retrieval_backend"] == "manifest" for row in discovery_rows))

    def test_query_runner_tolerates_oserror_timeout_during_materialization(self) -> None:
        class OSErrorTimeoutRunner(ProductFeedbackQueryAgenticRunner):
            def _resolve_backend(self):
                class FakeBackend:
                    backend_name = "web_search"

                    def search(self, plan: ReviewQueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredReviewDocument(
                                title="Stable review",
                                score=4.5,
                                retrieval_backend="web_search",
                                url="https://example.com/stable",
                                abstract="Comfortable and modular.",
                            ),
                            DiscoveredReviewDocument(
                                title="Late timeout review",
                                score=4.0,
                                retrieval_backend="web_search",
                                url="https://example.com/late-timeout",
                                abstract="This one fails later with a wrapped timeout.",
                            ),
                        )

                return FakeBackend()

            def _materialize_payload(self, document: DiscoveredReviewDocument):
                if document.url == "https://example.com/late-timeout":
                    raise OSError("The read operation timed out")
                return (
                    "<html><body><article><p>Comfortable, modular, and easy to clean.</p></article></body></html>",
                    "https://example.com/stable",
                    ".html",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OSErrorTimeoutRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How comfortable is the Lovesac sectional sofa?",
                    outdir=Path(tmpdir) / "out",
                    target_documents=2,
                    max_documents=2,
                    product_name="Sactional",
                    brand_name="Lovesac",
                )
            )

            result = runner.run()

            self.assertEqual(len(result.materialized_documents), 1)
            self.assertEqual(len(runner.materialization_failures), 1)
            self.assertIn("timed out", runner.materialization_failures[0]["error"].lower())
            self.assertIsNotNone(result.product_feedback_result)
            self.assertTrue(result.product_feedback_result.dashboard_path.exists())

    def test_web_search_backend_extracts_review_links_from_search_results(self) -> None:
        class FakeWebSearchBackend(WebSearchReviewRetrievalBackend):
            def _fetch_search_html(self, query: str, *, limit: int) -> str:
                del query, limit
                return """
                <html><body>
                  <div class="result results_links">
                    <div class="result__body">
                      <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Flovesac-review">
                        Lovesac sectional sofa review
                      </a>
                      <a class="result__snippet">A long-term review focused on comfort, cushion depth, and setup burden.</a>
                    </div>
                  </div>
                  <div class="result results_links">
                    <div class="result__body">
                      <a class="result__a" href="https://example.org/lovesac-owners-thoughts">
                        Owners discuss Lovesac comfort
                      </a>
                      <div class="result__snippet">Comfort, durability, and return-risk tradeoffs.</div>
                    </div>
                  </div>
                </body></html>
                """

        backend = FakeWebSearchBackend(user_agent="test-agent", timeout_seconds=5.0)
        results = backend.search(
            ReviewQueryPlan(
                query="How comfortable is the Lovesac sectional sofa?",
                normalized_query="how comfortable is the lovesac sectional sofa?",
                keyword_tokens=("comfortable", "lovesac", "sectional"),
                target_documents=2,
            ),
            limit=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].retrieval_backend, "web_search")
        self.assertEqual(results[0].url, "https://example.com/lovesac-review")
        self.assertIn("comfort", results[0].abstract.lower())

    def test_query_runner_defaults_to_web_search_backend_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ProductFeedbackQueryAgenticRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How comfortable is the Lovesac sectional sofa?",
                    outdir=Path(tmpdir) / "out",
                )
            )

            backend = runner._resolve_backend()

            self.assertIsInstance(backend, WebSearchReviewRetrievalBackend)

    def test_query_interpretation_rewrites_assembly_question_into_retrieval_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ProductFeedbackQueryAgenticRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How easy is to assemble a Lovesac sectional sofa?",
                    outdir=Path(tmpdir) / "out",
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.product_name, "Lovesac sectional sofa")
            self.assertEqual(plan.retrieval_query, "Lovesac sectional sofa assembly reviews")
            self.assertIn("assembly", plan.keyword_tokens)
            self.assertIn("lovesac", plan.keyword_tokens)

    def test_query_interpretation_rewrites_food_taste_question_into_retrieval_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ProductFeedbackQueryAgenticRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How tasty is it to eat the Amedei Porcelana Chocolate Bars?",
                    outdir=Path(tmpdir) / "out",
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.product_name, "Amedei Porcelana Chocolate Bars")
            self.assertEqual(plan.retrieval_query, "Amedei Porcelana Chocolate Bars taste reviews")
            self.assertIn("amedei", plan.keyword_tokens)
            self.assertIn("chocolate", plan.keyword_tokens)

    def test_web_search_query_prefers_retrieval_query_when_available(self) -> None:
        backend = WebSearchReviewRetrievalBackend(user_agent="test-agent", timeout_seconds=5.0)

        search_query = backend._search_query(
            ReviewQueryPlan(
                query="How easy is to assemble a Lovesac sectional sofa?",
                normalized_query="lovesac sectional sofa assembly reviews",
                keyword_tokens=("lovesac", "sectional", "assembly"),
                target_documents=2,
                product_name="Lovesac sectional sofa",
                retrieval_query="Lovesac sectional sofa assembly reviews",
            )
        )

        self.assertEqual(search_query, "Lovesac sectional sofa assembly reviews")

    def test_dashboard_product_label_falls_back_to_query_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ProductFeedbackQueryAgenticRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How comfortable is it to drive the Mazda Miata 3?",
                    outdir=Path(tmpdir) / "out",
                )
            )

            runner._bootstrap_feedback_dashboard(
                status="running",
                note="Testing dashboard labeling.",
                feedback_count=0,
                analysis_question="What evidence speaks to driving comfort?",
            )

            scorecard = json.loads((runner.analysis_outdir / "product_success_scorecard.json").read_text(encoding="utf-8"))
            self.assertEqual(scorecard["product_name"], "How comfortable is it to drive the Mazda Miata 3?")

    def test_query_runner_writes_discovery_manifest_for_discovered_reviews(self) -> None:
        class DiscoveryOnlyRunner(ProductFeedbackQueryAgenticRunner):
            def _resolve_backend(self):
                class FakeBackend:
                    backend_name = "web_search"

                    def search(self, plan: ReviewQueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredReviewDocument(
                                title="Lovesac sectional sofa review",
                                score=4.5,
                                retrieval_backend="web_search",
                                url="https://example.com/lovesac-review",
                                abstract="Comfort and durability review.",
                                evidence=("lovesac", "comfort"),
                            ),
                        )

                return FakeBackend()

            def _materialize_payload(self, document: DiscoveredReviewDocument):
                del document
                return (
                    "<html><body><article><p>Comfortable over the long run.</p></article></body></html>",
                    "https://example.com/lovesac-review",
                    ".html",
                )

            def _run_product_feedback_analysis(self, feedback_manifest: Path, *, analysis_question: str):
                del feedback_manifest, analysis_question
                self.analysis_outdir.mkdir(parents=True, exist_ok=True)
                dashboard_path = self.analysis_outdir / "product_feedback_dashboard.html"
                dashboard_path.write_text("<html><body>dashboard</body></html>", encoding="utf-8")
                summary_path = self.analysis_outdir / "product_feedback_dashboard_summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                report_path = self.analysis_outdir / "product_feedback_report.md"
                report_path.write_text("report", encoding="utf-8")
                for filename in (
                    "normalized_feedback.jsonl",
                    "usage_workflows.json",
                    "aspect_summary.json",
                    "outcome_summary.json",
                    "causal_hypotheses.json",
                    "product_success_scorecard.json",
                    "ablation_comparison.json",
                ):
                    path = self.analysis_outdir / filename
                    if filename.endswith(".jsonl"):
                        path.write_text("", encoding="utf-8")
                    else:
                        path.write_text("{}", encoding="utf-8")
                return ProductFeedbackRunResult(
                    records=(),
                    normalized_feedback_path=self.analysis_outdir / "normalized_feedback.jsonl",
                    usage_workflows_path=self.analysis_outdir / "usage_workflows.json",
                    aspect_summary_path=self.analysis_outdir / "aspect_summary.json",
                    outcome_summary_path=self.analysis_outdir / "outcome_summary.json",
                    causal_hypotheses_path=self.analysis_outdir / "causal_hypotheses.json",
                    success_scorecard_path=self.analysis_outdir / "product_success_scorecard.json",
                    ablation_comparison_path=self.analysis_outdir / "ablation_comparison.json",
                    report_path=report_path,
                    dashboard_path=dashboard_path,
                    dashboard_summary_path=summary_path,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DiscoveryOnlyRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="How comfortable is the Lovesac sectional sofa?",
                    outdir=Path(tmpdir) / "out",
                    target_documents=1,
                )
            )

            result = runner.run()

            discovery_rows = [
                json.loads(line)
                for line in runner.discovery_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(discovery_rows), 1)
            self.assertEqual(discovery_rows[0]["retrieval_backend"], "web_search")
            self.assertEqual(discovery_rows[0]["url"], "https://example.com/lovesac-review")
            self.assertTrue(result.materialized_feedback_manifest_path.exists())

    def test_resolve_query_for_main_uses_cli_query_when_present(self) -> None:
        try:
            from functorflow_v3 import product_feedback_query_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_query_agentic as module

        query = module._resolve_query_for_main(
            SimpleNamespace(query="  How comfortable is the Lovesac sectional sofa?  ", outdir="/tmp/ignored")
        )

        self.assertEqual(query, "How comfortable is the Lovesac sectional sofa?")

    def test_resolve_query_for_main_falls_back_to_dashboard_launcher(self) -> None:
        try:
            from functorflow_v3 import product_feedback_query_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_query_agentic as module

        class FakeLauncher:
            instances: list["FakeLauncher"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def wait_for_submission(self) -> str:
                return "How comfortable is the Lovesac sectional sofa?"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
                query = module._resolve_query_for_main(
                    SimpleNamespace(query="", outdir=tmpdir)
                )

        self.assertEqual(query, "How comfortable is the Lovesac sectional sofa?")
        self.assertEqual(len(FakeLauncher.instances), 1)
        self.assertEqual(
            FakeLauncher.instances[0].config.artifact_path,
            Path(tmpdir).resolve() / "product_feedback_run" / "corpus_synthesis" / "product_feedback_corpus_synthesis.html",
        )

    def test_html_extraction_ignores_policy_noise_for_url_without_html_suffix(self) -> None:
        try:
            from functorflow_v3 import product_feedback_query_agentic as module
        except ImportError:
            from ..functorflow_v3 import product_feedback_query_agentic as module

        payload = """
        <!doctype html>
        <html>
        <body>
          <article>
            <h1>Lovesac review</h1>
            <p>The couch is comfortable over the long run and the washable covers are worth it.</p>
            <p>Assembly was difficult.</p>
          </article>
          <footer>
            <p>Return policy</p>
            <p>30-day return window</p>
          </footer>
        </body>
        </html>
        """

        text = module._extract_article_text(
            payload,
            source_hint="https://example.com/lovesac-review",
        )

        self.assertIn("comfortable over the long run", text)
        self.assertIn("Assembly was difficult", text)
        self.assertNotIn("30-day return window", text)
        self.assertNotIn("Return policy", text)

    def test_manifest_review_retrieval_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            review1 = workdir / "lovesac_review_positive.html"
            review1.write_text(
                """
                <html><body>
                <meta property="og:image" content="https://cdn.example.com/lovesac-sactional.jpg">
                <h1>Lovesac Sactional 5 Year Review</h1>
                <p>We have owned this sofa for more than five years.</p>
                <p>Overall it is very comfortable over the long run and flexible for movie nights.</p>
                <p>The price is high, but washable covers and modular flexibility make it worth it for us.</p>
                </body></html>
                """,
                encoding="utf-8",
            )
            review2 = workdir / "lovesac_review_negative.html"
            review2.write_text(
                """
                <html><body>
                <h1>Lovesac couch regrets</h1>
                <p>The seat depth felt too deep for daily sitting and the cushions kept shifting.</p>
                <p>Assembly was difficult and the sofa felt overpriced for the maintenance burden.</p>
                </body></html>
                """,
                encoding="utf-8",
            )
            review3 = workdir / "other_product.html"
            review3.write_text(
                """
                <html><body>
                <h1>Camping stove review</h1>
                <p>This is unrelated to sofa comfort.</p>
                </body></html>
                """,
                encoding="utf-8",
            )

            manifest_path = workdir / "review_manifest.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "title": "Lovesac Sactional 5 Year Review",
                                "summary": "Long-term comfort, price, and flexibility thoughts.",
                                "product": "Lovesac Sactional",
                                "brand": "Lovesac",
                                "source_path": str(review1),
                            }
                        ),
                        json.dumps(
                            {
                                "title": "Lovesac couch regrets",
                                "summary": "Seat depth, cushion shifting, and assembly issues.",
                                "product": "Lovesac Sactional",
                                "brand": "Lovesac",
                                "source_path": str(review2),
                            }
                        ),
                        json.dumps(
                            {
                                "title": "Camping stove review",
                                "summary": "Outdoor stove test.",
                                "product": "Trail Stove",
                                "brand": "CampCo",
                                "source_path": str(review3),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runner = ProductFeedbackQueryAgenticRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="Lovesac Sactional long term comfort review",
                    outdir=workdir / "out",
                    manifest_path=manifest_path,
                    target_documents=2,
                    product_name="Sactional",
                    brand_name="Lovesac",
                )
            )
            result = runner.run()

            self.assertGreaterEqual(len(result.selected_documents), 2)
            self.assertGreaterEqual(len(result.materialized_documents), 2)
            selected_titles = {item.title for item in result.selected_documents}
            self.assertIn("Lovesac Sactional 5 Year Review", selected_titles)
            self.assertIn("Lovesac couch regrets", selected_titles)
            self.assertTrue(result.materialized_feedback_manifest_path.exists())
            self.assertIsNotNone(result.product_feedback_result)

            materialized_rows = [
                json.loads(line)
                for line in result.materialized_feedback_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all("retrieval_score" in row for row in materialized_rows))
            self.assertTrue(all("score" not in row for row in materialized_rows))

            outcome = json.loads(result.product_feedback_result.outcome_summary_path.read_text(encoding="utf-8"))
            self.assertIsNone(outcome["average_rating"])

            scorecard = json.loads(result.product_feedback_result.success_scorecard_path.read_text(encoding="utf-8"))
            self.assertIn("value", scorecard["top_negative_aspects"])
            self.assertIn("comfort", scorecard["top_positive_aspects"])

            hypotheses = json.loads(result.product_feedback_result.causal_hypotheses_path.read_text(encoding="utf-8"))
            hypothesis_sources = {item["src"] for item in hypotheses["hypotheses"]}
            self.assertIn("negative seat_depth perception", hypothesis_sources)

            report = result.product_feedback_result.report_path.read_text(encoding="utf-8")
            self.assertIn("long-run comfort", report.lower())
            self.assertIn("What evidence speaks to long-run comfort, durability, or maintenance burden?", report)

            product_visual_asset_path = result.product_feedback_result.dashboard_path.parent / "product_visual_asset.json"
            self.assertTrue(product_visual_asset_path.exists())
            product_visual_asset = json.loads(product_visual_asset_path.read_text(encoding="utf-8"))
            self.assertEqual(product_visual_asset["image_url"], "https://cdn.example.com/lovesac-sactional.jpg")

            dashboard_html = result.product_feedback_result.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("hero-visual", dashboard_html)
            self.assertIn("https://cdn.example.com/lovesac-sactional.jpg", dashboard_html)
            self.assertIn("Topos PSR", dashboard_html)

            self.assertIsNotNone(result.product_feedback_result.topos_psr_path)
            self.assertIsNotNone(result.product_feedback_result.review_episodes_path)
            self.assertTrue(result.product_feedback_result.topos_psr_path.exists())
            self.assertTrue(result.product_feedback_result.review_episodes_path.exists())
            topos_psr_view_path = result.product_feedback_result.topos_psr_path.with_name("topos_psr_bundle.html")
            self.assertTrue(topos_psr_view_path.exists())
            topos_psr_view_html = topos_psr_view_path.read_text(encoding="utf-8")
            self.assertIn("Topos PSR Bundle", topos_psr_view_html)
            self.assertIn("Reviews used", topos_psr_view_html)
            self.assertIn("Context views", topos_psr_view_html)
            self.assertIn("Rows are history prefixes; columns are finite tests", topos_psr_view_html)
            self.assertIn("history rows / test columns", topos_psr_view_html)
            self.assertIn("test-legend", topos_psr_view_html)
            self.assertIn("Raw JSON bundle", topos_psr_view_html)
            self.assertIsNotNone(result.persistent_state_path)
            self.assertIsNotNone(result.persistent_state_dashboard_path)
            self.assertTrue(result.persistent_state_path.exists())
            self.assertTrue(result.persistent_state_dashboard_path.exists())
            persistent_state = json.loads(result.persistent_state_path.read_text(encoding="utf-8"))
            self.assertEqual(persistent_state["state_kind"], "product_feedback_world_state")
            self.assertEqual(persistent_state["diagnostics"]["query_context"], "sit")
            self.assertIn("recommended_actions", persistent_state)
            persistent_state_html = result.persistent_state_dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Persistent Product Feedback State", persistent_state_html)
            self.assertIn("Recommended Actions", persistent_state_html)

            corpus_synthesis_html = result.corpus_synthesis_result.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Topos PSR", corpus_synthesis_html)
            self.assertIn("reviews used", corpus_synthesis_html)
            self.assertIn("context views", corpus_synthesis_html)
            self.assertIn("Open topos PSR bundle", corpus_synthesis_html)
            self.assertIn("topos_psr_bundle.html", corpus_synthesis_html)
            self.assertIn("Raw PSR JSON", corpus_synthesis_html)

    def test_query_runner_bootstraps_dashboard_before_search_and_stops_after_consensus(self) -> None:
        class ConsensusRunner(ProductFeedbackQueryAgenticRunner):
            def __init__(self, config: ProductFeedbackQueryAgenticConfig) -> None:
                super().__init__(config)
                self.dashboard_seen_during_search = False
                self.analysis_manifest_sizes: list[int] = []
                self._fake_documents: tuple[DiscoveredReviewDocument, ...] = ()

            def _resolve_backend(self):
                runner = self

                class FakeBackend:
                    backend_name = "manifest"

                    def search(self, plan: ReviewQueryPlan, *, limit: int):
                        del plan
                        runner.dashboard_seen_during_search = (
                            runner.analysis_outdir / "product_feedback_dashboard.html"
                        ).exists()
                        return runner._fake_documents[:limit]

                return FakeBackend()

            def _run_product_feedback_analysis(
                self,
                feedback_manifest: Path,
                *,
                analysis_question: str,
            ) -> ProductFeedbackRunResult:
                del analysis_question
                manifest_lines = [
                    line
                    for line in feedback_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                count = len(manifest_lines)
                self.analysis_manifest_sizes.append(count)
                self.analysis_outdir.mkdir(parents=True, exist_ok=True)
                dashboard_path = self.analysis_outdir / "product_feedback_dashboard.html"
                dashboard_path.write_text("<html><body>live dashboard</body></html>", encoding="utf-8")
                summary_path = self.analysis_outdir / "product_feedback_dashboard_summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                report_path = self.analysis_outdir / "product_feedback_report.md"
                report_path.write_text("report", encoding="utf-8")
                for filename in (
                    "normalized_feedback.jsonl",
                    "usage_workflows.json",
                    "aspect_summary.json",
                    "outcome_summary.json",
                    "causal_hypotheses.json",
                    "product_success_scorecard.json",
                    "ablation_comparison.json",
                ):
                    path = self.analysis_outdir / filename
                    if filename.endswith(".jsonl"):
                        path.write_text("", encoding="utf-8")
                    else:
                        path.write_text("{}", encoding="utf-8")
                return ProductFeedbackRunResult(
                    records=(),
                    normalized_feedback_path=self.analysis_outdir / "normalized_feedback.jsonl",
                    usage_workflows_path=self.analysis_outdir / "usage_workflows.json",
                    aspect_summary_path=self.analysis_outdir / "aspect_summary.json",
                    outcome_summary_path=self.analysis_outdir / "outcome_summary.json",
                    causal_hypotheses_path=self.analysis_outdir / "causal_hypotheses.json",
                    success_scorecard_path=self.analysis_outdir / "product_success_scorecard.json",
                    ablation_comparison_path=self.analysis_outdir / "ablation_comparison.json",
                    report_path=report_path,
                    dashboard_path=dashboard_path,
                    dashboard_summary_path=summary_path,
                )

            def _consensus_snapshot(self, feedback_result: ProductFeedbackRunResult) -> ReviewConsensusSnapshot:
                del feedback_result
                count = self.analysis_manifest_sizes[-1]
                return ReviewConsensusSnapshot(
                    verdict="mixed_positive",
                    overall_score=0.62 if count == 2 else 0.63,
                    return_warning_recommended=False,
                    top_positive_aspects=("comfort", "durability"),
                    top_negative_aspects=("price", "weight"),
                    top_return_risk_aspects=("fit",),
                    feedback_count=count,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            runner = ConsensusRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="Nike Pegasus 41 long run comfort review",
                    outdir=workdir / "out",
                    target_documents=2,
                    max_documents=4,
                    product_name="Pegasus 41",
                    brand_name="Nike",
                )
            )
            documents = []
            for index in range(4):
                review_path = workdir / f"review_{index}.html"
                review_path.write_text(
                    f"<html><body><article><h1>Review {index}</h1><p>Comfortable and durable for daily miles.</p></article></body></html>",
                    encoding="utf-8",
                )
                documents.append(
                    DiscoveredReviewDocument(
                        title=f"Review {index}",
                        score=10.0 - index,
                        retrieval_backend="manifest",
                        source_path=str(review_path),
                    )
                )
            runner._fake_documents = tuple(documents)

            result = runner.run()

            self.assertTrue(runner.dashboard_seen_during_search)
            self.assertEqual(runner.analysis_manifest_sizes, [2, 3])
            self.assertEqual(len(result.selected_documents), 3)
            self.assertEqual(len(result.materialized_documents), 3)
            self.assertEqual(result.analysis_iterations, 2)
            self.assertTrue(result.consensus_reached)
            self.assertEqual(result.convergence_assessment["stop_trigger"], "stability")

    def test_query_runner_marks_budget_stop_without_claiming_consensus(self) -> None:
        class BudgetStopRunner(ProductFeedbackQueryAgenticRunner):
            def __init__(self, config: ProductFeedbackQueryAgenticConfig) -> None:
                super().__init__(config)
                self.analysis_manifest_sizes: list[int] = []
                self._fake_documents: tuple[DiscoveredReviewDocument, ...] = ()

            def _resolve_backend(self):
                runner = self

                class FakeBackend:
                    backend_name = "manifest"

                    def search(self, plan: ReviewQueryPlan, *, limit: int):
                        del plan
                        return runner._fake_documents[:limit]

                return FakeBackend()

            def _run_product_feedback_analysis(
                self,
                feedback_manifest: Path,
                *,
                analysis_question: str,
            ) -> ProductFeedbackRunResult:
                del analysis_question
                manifest_lines = [
                    line
                    for line in feedback_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                count = len(manifest_lines)
                self.analysis_manifest_sizes.append(count)
                self.analysis_outdir.mkdir(parents=True, exist_ok=True)
                dashboard_path = self.analysis_outdir / "product_feedback_dashboard.html"
                dashboard_path.write_text("<html><body>live dashboard</body></html>", encoding="utf-8")
                summary_path = self.analysis_outdir / "product_feedback_dashboard_summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                report_path = self.analysis_outdir / "product_feedback_report.md"
                report_path.write_text("report", encoding="utf-8")
                for filename in (
                    "normalized_feedback.jsonl",
                    "usage_workflows.json",
                    "aspect_summary.json",
                    "outcome_summary.json",
                    "causal_hypotheses.json",
                    "product_success_scorecard.json",
                    "ablation_comparison.json",
                ):
                    path = self.analysis_outdir / filename
                    if filename.endswith(".jsonl"):
                        path.write_text("", encoding="utf-8")
                    else:
                        path.write_text("{}", encoding="utf-8")
                return ProductFeedbackRunResult(
                    records=(),
                    normalized_feedback_path=self.analysis_outdir / "normalized_feedback.jsonl",
                    usage_workflows_path=self.analysis_outdir / "usage_workflows.json",
                    aspect_summary_path=self.analysis_outdir / "aspect_summary.json",
                    outcome_summary_path=self.analysis_outdir / "outcome_summary.json",
                    causal_hypotheses_path=self.analysis_outdir / "causal_hypotheses.json",
                    success_scorecard_path=self.analysis_outdir / "product_success_scorecard.json",
                    ablation_comparison_path=self.analysis_outdir / "ablation_comparison.json",
                    report_path=report_path,
                    dashboard_path=dashboard_path,
                    dashboard_summary_path=summary_path,
                )

            def _consensus_snapshot(self, feedback_result: ProductFeedbackRunResult) -> ReviewConsensusSnapshot:
                del feedback_result
                count = self.analysis_manifest_sizes[-1]
                if count == 2:
                    return ReviewConsensusSnapshot(
                        verdict="mixed_positive",
                        overall_score=0.62,
                        return_warning_recommended=False,
                        top_positive_aspects=("comfort", "durability"),
                        top_negative_aspects=("price", "weight"),
                        top_return_risk_aspects=("fit",),
                        feedback_count=count,
                    )
                return ReviewConsensusSnapshot(
                    verdict="mixed",
                    overall_score=0.48,
                    return_warning_recommended=True,
                    top_positive_aspects=("comfort",),
                    top_negative_aspects=("support", "price"),
                    top_return_risk_aspects=("fit", "break_in"),
                    feedback_count=count,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            runner = BudgetStopRunner(
                ProductFeedbackQueryAgenticConfig(
                    query="Nike Pegasus 41 long run comfort review",
                    outdir=workdir / "out",
                    target_documents=2,
                    max_documents=3,
                    product_name="Pegasus 41",
                    brand_name="Nike",
                )
            )
            documents = []
            for index in range(3):
                review_path = workdir / f"review_{index}.html"
                review_path.write_text(
                    f"<html><body><article><h1>Review {index}</h1><p>Comfortable and durable for daily miles.</p></article></body></html>",
                    encoding="utf-8",
                )
                documents.append(
                    DiscoveredReviewDocument(
                        title=f"Review {index}",
                        score=10.0 - index,
                        retrieval_backend="manifest",
                        source_path=str(review_path),
                    )
                )
            runner._fake_documents = tuple(documents)

            result = runner.run()

            self.assertFalse(result.consensus_reached)
            self.assertEqual(result.analysis_iterations, 2)
            self.assertEqual(result.convergence_assessment["stop_trigger"], "max_evidence")


if __name__ == "__main__":
    unittest.main()
