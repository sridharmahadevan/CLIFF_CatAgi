"""Tests for the top-level FF2 query router."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import query_router_agentic as module
    from functorflow_v3.basket_rocket_sec_agentic import BasketRocketSECRunResult
    from functorflow_v3.democritus_query_agentic import DemocritusQueryRunResult, QueryPlan
    from functorflow_v3.product_feedback_query_agentic import ProductFeedbackQueryRunResult, ReviewQueryPlan
except ModuleNotFoundError:
    from ..functorflow_v3 import query_router_agentic as module
    from ..functorflow_v3.basket_rocket_sec_agentic import BasketRocketSECRunResult
    from ..functorflow_v3.democritus_query_agentic import DemocritusQueryRunResult, QueryPlan
    from ..functorflow_v3.product_feedback_query_agentic import ProductFeedbackQueryRunResult, ReviewQueryPlan


class QueryRouterAgenticTests(unittest.TestCase):
    def test_session_query_outdir_places_runs_inside_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = Path(tmpdir) / "CLIFF_runs_archive"
            archive_root.mkdir()

            outdir = module._session_query_outdir(
                archive_root,
                run_id="run-0004",
                query="Analyze 10 recent studies of climate change",
            )

        self.assertEqual(outdir.parent.resolve(), archive_root.resolve())
        self.assertTrue(outdir.name.startswith("run-0004-"))
        self.assertIn("analyze_10_recent_studies_of_climate_change", outdir.name)

    def test_session_query_outdir_keeps_stem_style_behavior_for_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_root = Path(tmpdir) / "cliff_session1"

            outdir = module._session_query_outdir(
                session_root,
                run_id="run-0004",
                query="Analyze 10 recent studies of climate change",
            )

        self.assertEqual(outdir.parent.resolve(), session_root.parent.resolve())
        self.assertTrue(outdir.name.startswith("cliff_session1-run-0004-"))

    def test_route_ff2_query_selects_sec_runner_for_filing_language(self) -> None:
        decision = module.route_ff2_query("Find me 10 recent AMD 10-K filings")

        self.assertEqual(decision.route_name, "basket_rocket_sec")
        self.assertIn("SEC", decision.rationale)

    def test_route_ff2_query_selects_product_feedback_for_product_question(self) -> None:
        decision = module.route_ff2_query("How comfortable is the Lovesac sectional sofa?")

        self.assertEqual(decision.route_name, "product_feedback")
        self.assertIn("product", decision.rationale.lower())

    def test_route_ff2_query_selects_product_feedback_for_vehicle_usability_question(self) -> None:
        decision = module.route_ff2_query("How easy is it to drive the Mazda Miata 3?")

        self.assertEqual(decision.route_name, "product_feedback")
        self.assertIn("product", decision.rationale.lower())

    def test_route_ff2_query_selects_product_feedback_for_running_shoe_comfort_question(self) -> None:
        decision = module.route_ff2_query("How comfortable is it to run with the Nike Pegasus 41 running shoes?")

        self.assertEqual(decision.route_name, "product_feedback")
        self.assertIn("product", decision.rationale.lower())

    def test_route_ff2_query_selects_product_feedback_for_running_shoe_ease_question(self) -> None:
        decision = module.route_ff2_query("How easy is it to run with the Nike Pegasus 41 running shoes?")

        self.assertEqual(decision.route_name, "product_feedback")
        self.assertIn("product", decision.rationale.lower())

    def test_route_ff2_query_keeps_lovesac_assembly_question_in_product_feedback(self) -> None:
        decision = module.route_ff2_query("How easy is to assemble a Lovesac sectional sofa?")

        self.assertEqual(decision.route_name, "product_feedback")
        self.assertIn("product", decision.rationale.lower())

    def test_route_ff2_query_selects_culinary_tour_for_food_trip_request(self) -> None:
        decision = module.route_ff2_query("Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal")

        self.assertEqual(decision.route_name, "culinary_tour")
        self.assertIn("culinary", decision.rationale.lower())

    def test_route_ff2_query_selects_culinary_tour_for_bare_tour_with_dates(self) -> None:
        decision = module.route_ff2_query("Plan a kimchi tour of Seoul from July 6-10th")

        self.assertEqual(decision.route_name, "culinary_tour")
        self.assertIn("culinary", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_geometric_transformer_sudoku(self) -> None:
        decision = module.route_ff2_query("Explain the Geometric Transformer on the Sudoku problem")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_kan_extension_transformer(self) -> None:
        decision = module.route_ff2_query("Show how the Kan Extension Transformer works on language modeling")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_sheaves(self) -> None:
        decision = module.route_ff2_query("Explain sheaves via covers and gluing")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_diagrammatic_backpropagation(self) -> None:
        decision = module.route_ff2_query("Explain Diagrammatic Backpropagation")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_democritus_manifold(self) -> None:
        decision = module.route_ff2_query("Show the Democritus causal manifold demo")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_selects_course_demo_for_julia_ket(self) -> None:
        decision = module.route_ff2_query("Show me the Julia version of KET")

        self.assertEqual(decision.route_name, "course_demo")
        self.assertIn("course demo", decision.rationale.lower())

    def test_route_ff2_query_defaults_to_democritus_for_study_request(self) -> None:
        decision = module.route_ff2_query("Find me 10 recent studies of the benefits of red wine")

        self.assertEqual(decision.route_name, "democritus")
        self.assertIn("Democritus", decision.rationale)

    def test_route_ff2_query_keeps_semaglutide_trial_synthesis_on_democritus(self) -> None:
        decision = module.route_ff2_query(
            "Analyze 5 recent primary randomized controlled trials of semaglutide for weight loss in adults with obesity, "
            "excluding reviews, meta-analyses, guidance statements, and real-world comparative studies, and synthesize their joint support."
        )

        self.assertEqual(decision.route_name, "democritus")
        self.assertIn("evidence acquisition", decision.rationale.lower())

    def test_route_ff2_query_defaults_to_democritus_for_direct_document_url(self) -> None:
        decision = module.route_ff2_query("Analyze the document at https://example.org/news/story-about-water")

        self.assertEqual(decision.route_name, "democritus")
        self.assertIn("Democritus", decision.rationale)

    def test_route_ff2_query_honors_override(self) -> None:
        decision = module.route_ff2_query(
            "How comfortable is the Lovesac sectional sofa?",
            route_override="democritus",
        )

        self.assertEqual(decision.route_name, "democritus")

    def test_route_ff2_query_respects_wrong_route_feedback_for_product_question(self) -> None:
        decision = module.route_ff2_query(
            "How comfortable is the Lovesac sectional sofa?",
            excluded_routes=("product_feedback",),
        )

        self.assertEqual(decision.route_name, "democritus")
        self.assertIn("User feedback excluded route(s): product_feedback.", decision.rationale)

    def test_route_ff2_query_respects_wrong_route_feedback_for_course_demo_request(self) -> None:
        decision = module.route_ff2_query(
            "Explain the Geometric Transformer on the Sudoku problem",
            excluded_routes=("course_demo",),
        )

        self.assertEqual(decision.route_name, "democritus")
        self.assertIn("User feedback excluded route(s): course_demo.", decision.rationale)

    def test_resolve_query_for_main_uses_dashboard_launcher_when_query_missing(self) -> None:
        class FakeLauncher:
            def __init__(self, config):
                self.config = config

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def wait_for_submission(self) -> str:
                return "Find me 10 recent AMD 10-K filings"

        with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
            query = module._resolve_query_for_main(SimpleNamespace(query="", outdir="/tmp/ff2-router"))

        self.assertEqual(query, "Find me 10 recent AMD 10-K filings")

    def test_resolve_query_for_main_passes_router_artifact_path_to_launcher(self) -> None:
        captured_config = None

        class FakeLauncher:
            def __init__(self, config):
                nonlocal captured_config
                captured_config = config

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def wait_for_submission(self) -> str:
                return "Find me 10 recent studies on GLP-1"

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "selected_route_artifact.html"
            with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
                query = module._resolve_query_for_main(
                    SimpleNamespace(query="", outdir=tmpdir),
                    artifact_path=artifact_path,
                )

        self.assertEqual(query, "Find me 10 recent studies on GLP-1")
        self.assertIsNotNone(captured_config)
        self.assertEqual(captured_config.artifact_path, artifact_path)

    def test_materialize_router_artifact_copies_selected_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            source = workdir / "source_dashboard.html"
            target = workdir / "selected_route_artifact.html"
            source.write_text("<html><body>Democritus dashboard</body></html>", encoding="utf-8")

            module._materialize_router_artifact(source, target)

            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))

    def test_artifact_path_prefers_democritus_clarification_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clarification_path = Path(tmpdir) / "democritus_query_clarification.html"
            clarification_path.write_text("<html>clarification</html>", encoding="utf-8")
            checkpoint_path = Path(tmpdir) / "democritus_topic_checkpoint.html"
            checkpoint_path.write_text("<html>checkpoint</html>", encoding="utf-8")
            plan = QueryPlan(
                query="Analyze studies on inflation",
                normalized_query="inflation",
                keyword_tokens=("inflation",),
                target_documents=20,
            )
            result = module.FF2QueryRouterRunResult(
                route_decision=module.FF2RouteDecision(
                    route_name="democritus",
                    module_name="functorflow_v3.democritus_query_agentic",
                    rationale="test",
                ),
                route_outdir=Path(tmpdir),
                summary_path=Path(tmpdir) / "ff2_query_router_summary.json",
                democritus_result=DemocritusQueryRunResult(
                    query_plan=plan,
                    selected_documents=(),
                    acquired_documents=(),
                    batch_records=(),
                    pdf_dir=Path(tmpdir) / "pdfs",
                    batch_outdir=Path(tmpdir) / "democritus_runs",
                    summary_path=Path(tmpdir) / "query_run_summary.json",
                    checkpoint_dashboard_path=checkpoint_path,
                    clarification_dashboard_path=clarification_path,
                ),
            )

            artifact_path = module._artifact_path_for_result(result)

        self.assertEqual(artifact_path, clarification_path)

    def test_router_dispatches_to_democritus_runner(self) -> None:
        class FakeDemocritusRunner:
            instances: list["FakeDemocritusRunner"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def run(self):
                plan = QueryPlan(
                    query=self.config.query,
                    normalized_query=self.config.query.lower(),
                    keyword_tokens=("red", "wine"),
                    target_documents=self.config.target_documents,
                )
                return DemocritusQueryRunResult(
                    query_plan=plan,
                    selected_documents=(),
                    acquired_documents=(),
                    batch_records=(),
                    pdf_dir=self.config.outdir / "pdfs",
                    batch_outdir=self.config.outdir / "democritus_runs",
                    summary_path=self.config.outdir / "query_run_summary.json",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "DemocritusQueryAgenticRunner", FakeDemocritusRunner):
                result = module.FF2QueryRouter(
                    module.FF2QueryRouterConfig(
                        query="Find me 10 recent studies of the benefits of red wine",
                        outdir=Path(tmpdir),
                    )
                ).run()

        self.assertEqual(result.route_decision.route_name, "democritus")
        self.assertEqual(FakeDemocritusRunner.instances[0].config.outdir, Path(tmpdir).resolve() / "democritus")
        self.assertEqual(FakeDemocritusRunner.instances[0].config.execution_mode, "quick")

    def test_router_passes_deep_mode_to_company_similarity_runner(self) -> None:
        class FakeCompanySimilarityRunner:
            instances: list["FakeCompanySimilarityRunner"] = []

            def __init__(self, query, outdir, *, sec_user_agent="", execution_mode="quick", year_start=None, year_end=None):
                self.query = query
                self.outdir = outdir
                self.sec_user_agent = sec_user_agent
                self.execution_mode = execution_mode
                self.year_start = year_start
                self.year_end = year_end
                type(self).instances.append(self)

            def run(self):
                return module.CompanySimilarityRunResult(
                    query_plan=None,
                    route_outdir=self.outdir,
                    analysis_dir=self.outdir / "analysis",
                    summary_path=self.outdir / "company_similarity_summary.json",
                    artifact_path=self.outdir / "company_similarity_dashboard.html",
                    company_a_manifest_path=None,
                    company_b_manifest_path=None,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "CompanySimilarityAgenticRunner", FakeCompanySimilarityRunner):
                result = module.FF2QueryRouter(
                    module.FF2QueryRouterConfig(
                        query="How similar is Adobe to Nike?",
                        outdir=Path(tmpdir),
                        execution_mode="deep",
                    )
                ).run()

        self.assertEqual(result.route_decision.route_name, "company_similarity")
        self.assertEqual(FakeCompanySimilarityRunner.instances[0].execution_mode, "deep")

    def test_router_dispatches_to_sec_runner(self) -> None:
        class FakeSECRunner:
            instances: list["FakeSECRunner"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def run(self):
                plan = QueryPlan(
                    query=self.config.query,
                    normalized_query=self.config.query.lower(),
                    keyword_tokens=("amd", "10", "k"),
                    target_documents=self.config.target_filings,
                    requested_forms=self.config.sec_form_types,
                )
                return BasketRocketSECRunResult(
                    query_plan=plan,
                    selected_filings=(),
                    materialized_filings=(),
                    batch_records=(),
                    discovery_summary_path=self.config.outdir / "sec_discovery" / "query_run_summary.json",
                    filing_manifest_path=None,
                    company_context_path=None,
                    batch_workset_index_path=None,
                    batch_summary_path=None,
                    batch_live_gui_path=None,
                    batch_visualization_index_path=None,
                    batch_visualization_summary_path=None,
                    summary_path=self.config.outdir / "basket_rocket_run_summary.json",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "BasketRocketSECAgenticRunner", FakeSECRunner):
                result = module.FF2QueryRouter(
                    module.FF2QueryRouterConfig(
                        query="Find me 10 recent AMD 10-K filings",
                        outdir=Path(tmpdir),
                    )
                ).run()

        self.assertEqual(result.route_decision.route_name, "basket_rocket_sec")
        self.assertEqual(FakeSECRunner.instances[0].config.outdir, Path(tmpdir).resolve() / "basket_rocket_sec")

    def test_build_router_from_args_passes_sec_retrieval_user_agent(self) -> None:
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_input_pdf_dir="",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=10,
            democritus_retrieval_backend="auto",
            democritus_max_docs=0,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            culinary_manifest="/tmp/culinary_stop_manifest.jsonl",
            product_target_docs=5,
            product_max_docs=0,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=10,
            sec_retrieval_user_agent="Jane Researcher jane@example.com",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        router = module._build_router_from_args(args, query="Find me 10 recent Adobe 10-K filings")

        self.assertEqual(router.config.sec_retrieval_user_agent, "Jane Researcher jane@example.com")
        self.assertEqual(router.config.culinary_manifest_path, Path("/tmp/culinary_stop_manifest.jsonl").resolve())

    def test_build_router_from_args_uses_requested_study_count_when_not_overridden(self) -> None:
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_input_pdf_dir="",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            product_target_docs=None,
            product_max_docs=None,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=None,
            sec_retrieval_user_agent="",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        router = module._build_router_from_args(
            args,
            query="Give me 5 studies of the health benefits of resveratrol in red wine",
        )

        self.assertEqual(router.config.democritus_target_documents, 5)
        self.assertEqual(router.config.democritus_max_docs, 15)
        self.assertTrue(router.config.democritus_render_topk_pngs)

    def test_build_router_from_args_honors_explicit_democritus_overrides(self) -> None:
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_input_pdf_dir="",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=8,
            democritus_retrieval_backend="auto",
            democritus_max_docs=12,
            democritus_intra_document_shards=1,
            democritus_manifold_mode="moe",
            democritus_topk=144,
            democritus_radii="2,4",
            democritus_maxnodes="12,24",
            democritus_lambda_edge=0.4,
            democritus_topk_models=7,
            democritus_topk_claims=21,
            democritus_alpha=1.3,
            democritus_tier1=0.7,
            democritus_tier2=0.2,
            democritus_anchors="resveratrol, red wine",
            democritus_title="Red Wine Democritus",
            democritus_dedupe_focus=True,
            democritus_require_anchor_in_focus=True,
            democritus_focus_blacklist_regex="^generic$",
            democritus_render_topk_pngs=True,
            democritus_assets_dir="credibility_assets",
            democritus_png_dpi=240,
            democritus_write_deep_dive=True,
            democritus_deep_dive_max_bullets=11,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            product_target_docs=None,
            product_max_docs=None,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=None,
            sec_retrieval_user_agent="",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        router = module._build_router_from_args(
            args,
            query="Give me 5 studies of the health benefits of resveratrol in red wine",
        )

        self.assertEqual(router.config.democritus_target_documents, 8)
        self.assertEqual(router.config.democritus_max_docs, 12)
        self.assertEqual(router.config.democritus_manifold_mode, "moe")
        self.assertEqual(router.config.democritus_topk, 144)
        self.assertEqual(router.config.democritus_radii, "2,4")
        self.assertEqual(router.config.democritus_maxnodes, "12,24")
        self.assertEqual(router.config.democritus_lambda_edge, 0.4)
        self.assertEqual(router.config.democritus_topk_models, 7)
        self.assertEqual(router.config.democritus_topk_claims, 21)
        self.assertEqual(router.config.democritus_alpha, 1.3)
        self.assertEqual(router.config.democritus_tier1, 0.7)
        self.assertEqual(router.config.democritus_tier2, 0.2)
        self.assertEqual(router.config.democritus_anchors, "resveratrol, red wine")
        self.assertEqual(router.config.democritus_title, "Red Wine Democritus")
        self.assertTrue(router.config.democritus_dedupe_focus)
        self.assertTrue(router.config.democritus_require_anchor_in_focus)
        self.assertEqual(router.config.democritus_focus_blacklist_regex, "^generic$")
        self.assertTrue(router.config.democritus_render_topk_pngs)
        self.assertEqual(router.config.democritus_assets_dir, "credibility_assets")
        self.assertEqual(router.config.democritus_png_dpi, 240)
        self.assertTrue(router.config.democritus_write_deep_dive)
        self.assertEqual(router.config.democritus_deep_dive_max_bullets, 11)

    def test_build_router_from_args_with_outdir_uses_override_directory(self) -> None:
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_input_pdf_dir="",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            product_target_docs=None,
            product_max_docs=None,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=None,
            sec_retrieval_user_agent="",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        router = module._build_router_from_args_with_outdir(
            args,
            query="How comfortable are Lovesac sofas?",
            outdir=Path("/tmp/ff2-router-run-0001"),
        )

        self.assertEqual(router.config.outdir, Path("/tmp/ff2-router-run-0001").resolve())

    def test_build_router_from_args_passes_direct_pdf_directory_override(self) -> None:
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_input_pdf="/tmp/uploaded_paper.pdf",
            democritus_input_pdf_dir="/tmp/uploaded_pdfs",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            product_target_docs=None,
            product_max_docs=None,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=None,
            sec_retrieval_user_agent="",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        router = module._build_router_from_args(
            args,
            query="Analyze the PDFs in /tmp/uploaded_pdfs",
        )

        self.assertEqual(router.config.democritus_input_pdf_dir, Path("/tmp/uploaded_pdfs").resolve())

    def test_run_session_query_updates_launcher_and_opens_completed_artifact(self) -> None:
        class FakeLauncher:
            def __init__(self) -> None:
                self.updates: list[dict[str, object]] = []

            def update_session_run(self, run_id: str, **kwargs) -> None:
                payload = {"run_id": run_id}
                payload.update(kwargs)
                self.updates.append(payload)

        class FakeRouter:
            def __init__(self, config) -> None:
                self.config = config

            def run(self):
                route_outdir = self.config.outdir / "product_feedback"
                artifact_path = route_outdir / "product_feedback_run" / "product_feedback_dashboard.html"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text("<html>ok</html>", encoding="utf-8")
                plan = ReviewQueryPlan(
                    query=self.config.query,
                    normalized_query=self.config.query.lower(),
                    keyword_tokens=("lovesac", "sofa"),
                    target_documents=self.config.product_target_documents,
                    product_name="Lovesac sofa",
                )
                result = ProductFeedbackQueryRunResult(
                    query_plan=plan,
                    selected_documents=(),
                    materialized_documents=(),
                    materialized_feedback_manifest_path=self.config.outdir / "materialized_feedback_manifest.jsonl",
                    summary_path=self.config.outdir / "review_query_summary.json",
                    product_feedback_result=SimpleNamespace(dashboard_path=artifact_path),
                )
                return module.FF2QueryRouterRunResult(
                    route_decision=module.FF2RouteDecision(
                        route_name="product_feedback",
                        module_name="functorflow_v3.product_feedback_query_agentic",
                        rationale="test",
                    ),
                    route_outdir=route_outdir,
                    summary_path=self.config.outdir / "ff2_query_router_summary.json",
                    product_feedback_result=result,
                )

        launcher = FakeLauncher()
        opened_paths: list[Path | None] = []
        args = SimpleNamespace(
            outdir="/tmp/ff2-router",
            route="auto",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            product_target_docs=None,
            product_max_docs=None,
            product_name="",
            brand_name="",
            analysis_question="",
            product_discovery_only=False,
            sec_target_filings=None,
            sec_retrieval_user_agent="",
            sec_form=[],
            sec_company_limit=3,
            sec_discovery_only=False,
            sec_dry_run=False,
        )

        with patch.object(module, "_build_router_from_args_with_outdir") as build_router:
            with patch.object(module, "_open_artifact", side_effect=lambda path: opened_paths.append(path)):
                build_router.return_value = FakeRouter(
                    SimpleNamespace(
                        outdir=Path("/tmp/ff2-router-run-0001"),
                        query="How comfortable are Lovesac sofas?",
                        product_target_documents=5,
                    )
                )
                module._run_session_query(
                    launcher,
                    args,
                    run_id="run-0001",
                    query="How comfortable are Lovesac sofas?",
                )

        self.assertEqual(launcher.updates[0]["status"], "routing")
        self.assertEqual(launcher.updates[1]["status"], "running")
        self.assertEqual(launcher.updates[-1]["status"], "complete")
        self.assertEqual(launcher.updates[-1]["route_name"], "product_feedback")
        self.assertTrue(str(launcher.updates[-1]["outdir"]).startswith("/tmp/ff2-router-run-0001"))
        self.assertEqual(opened_paths[0].name, "product_feedback_dashboard.html")

    def test_router_dispatches_to_product_feedback_runner(self) -> None:
        class FakeProductRunner:
            instances: list["FakeProductRunner"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def run(self):
                plan = ReviewQueryPlan(
                    query=self.config.query,
                    normalized_query=self.config.query.lower(),
                    keyword_tokens=("lovesac", "comfortable"),
                    target_documents=self.config.target_documents,
                )
                return ProductFeedbackQueryRunResult(
                    query_plan=plan,
                    selected_documents=(),
                    materialized_documents=(),
                    materialized_feedback_manifest_path=self.config.outdir / "materialized_feedback_manifest.jsonl",
                    summary_path=self.config.outdir / "review_query_summary.json",
                    product_feedback_result=None,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "ProductFeedbackQueryAgenticRunner", FakeProductRunner):
                result = module.FF2QueryRouter(
                    module.FF2QueryRouterConfig(
                        query="How comfortable is the Lovesac sectional sofa?",
                        outdir=Path(tmpdir),
                    )
                ).run()

        self.assertEqual(result.route_decision.route_name, "product_feedback")
        self.assertEqual(FakeProductRunner.instances[0].config.outdir, Path(tmpdir).resolve() / "product_feedback")
        self.assertIsNone(FakeProductRunner.instances[0].config.manifest_path)

    def test_router_dispatches_to_course_demo_runner(self) -> None:
        class FakeCourseRunner:
            instances: list["FakeCourseRunner"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def run(self):
                route_outdir = self.config.outdir
                dashboard_path = route_outdir / "course_demo_dashboard.html"
                dashboard_path.parent.mkdir(parents=True, exist_ok=True)
                dashboard_path.write_text("<html>course demo</html>", encoding="utf-8")
                return SimpleNamespace(
                    dashboard_path=dashboard_path,
                    summary_path=route_outdir / "course_demo_summary.json",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "CourseDemoAgenticRunner", FakeCourseRunner):
                result = module.FF2QueryRouter(
                    module.FF2QueryRouterConfig(
                        query="Explain the Geometric Transformer on the Sudoku problem",
                        outdir=Path(tmpdir),
                    )
                ).run()

        self.assertEqual(result.route_decision.route_name, "course_demo")
        self.assertEqual(FakeCourseRunner.instances[0].config.outdir, Path(tmpdir).resolve() / "course_demo")

    def test_predicted_artifact_path_prefers_sec_live_gui(self) -> None:
        decision = module.route_ff2_query("Find me 10 recent AMD 10-K filings")

        predicted = module._predicted_artifact_path(Path("/tmp/ff2-router"), decision)

        self.assertEqual(
            predicted,
            Path("/tmp/ff2-router").resolve() / "basket_rocket_sec" / "workflow_batches" / "basket_rocket_gui.html",
        )

    def test_predicted_artifact_path_for_course_demo(self) -> None:
        decision = module.route_ff2_query("Explain the Geometric Transformer on the Sudoku problem")

        predicted = module._predicted_artifact_path(Path("/tmp/ff2-router"), decision)

        self.assertEqual(
            predicted,
            Path("/tmp/ff2-router").resolve() / "course_demo" / "course_demo_dashboard.html",
        )

    def test_write_router_error_artifact_includes_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = module._write_router_error_artifact(
                Path(tmpdir),
                title="Launch error",
                message="Missing SEC identity",
                detail="ValueError('Missing SEC identity')",
                hints=("Set FF2_SEC_USER_AGENT",),
            )

            payload = path.read_text(encoding="utf-8")

        self.assertIn("Launch error", payload)
        self.assertIn("Set FF2_SEC_USER_AGENT", payload)


if __name__ == "__main__":
    unittest.main()
