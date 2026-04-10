"""Tests for the CLIFF entrypoint and conscious handoff."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import cliff as module
    from functorflow_v3 import cliff_worker as worker_module
    from functorflow_v3.product_feedback_query_agentic import ProductFeedbackQueryRunResult, ReviewQueryPlan
except ModuleNotFoundError:
    from ..functorflow_v3 import cliff as module
    from ..functorflow_v3 import cliff_worker as worker_module
    from ..functorflow_v3.product_feedback_query_agentic import ProductFeedbackQueryRunResult, ReviewQueryPlan


class CLIFFTests(unittest.TestCase):
    def test_route_cliff_query_routes_sec_language(self) -> None:
        decision = module.route_cliff_query("Find me 10 recent AMD 10-K filings")

        self.assertEqual(decision.route_name, "basket_rocket_sec")

    def test_route_cliff_query_routes_vehicle_usability_to_product_feedback(self) -> None:
        decision = module.route_cliff_query("How easy is it to drive the Mazda Miata 3?")

        self.assertEqual(decision.route_name, "product_feedback")

    def test_route_cliff_query_routes_running_shoe_usability_to_product_feedback(self) -> None:
        decision = module.route_cliff_query("How easy is it to run with the Nike Pegasus 41 running shoes?")

        self.assertEqual(decision.route_name, "product_feedback")

    def test_route_cliff_query_routes_culinary_tour_request(self) -> None:
        decision = module.route_cliff_query("Plan a kimchi culinary tour in Seoul July 6-11 under $50 per meal")

        self.assertEqual(decision.route_name, "culinary_tour")

    def test_route_cliff_query_routes_bare_food_tour_request(self) -> None:
        decision = module.route_cliff_query("Plan a kimchi tour of Seoul from July 6-10th")

        self.assertEqual(decision.route_name, "culinary_tour")

    def test_route_cliff_query_routes_seafood_tour_for_destination_request(self) -> None:
        decision = module.route_cliff_query("Plan a seafood tour for Boston from July 5th-10th")

        self.assertEqual(decision.route_name, "culinary_tour")

    def test_route_cliff_query_routes_course_demo_request(self) -> None:
        decision = module.route_cliff_query("Explain the Geometric Transformer on the Sudoku problem")

        self.assertEqual(decision.route_name, "course_demo")

    def test_route_cliff_query_routes_direct_document_url_to_democritus(self) -> None:
        decision = module.route_cliff_query("Analyze the document at https://example.org/news/story-about-water")

        self.assertEqual(decision.route_name, "democritus")

    def test_route_cliff_query_routes_local_pdf_path_to_democritus(self) -> None:
        decision = module.route_cliff_query("Analyze the PDF at /tmp/uploaded_paper.pdf")

        self.assertEqual(decision.route_name, "democritus")

    def test_route_cliff_query_routes_sheaves_request_to_course_demo(self) -> None:
        decision = module.route_cliff_query("Explain sheaves via covers and gluing")

        self.assertEqual(decision.route_name, "course_demo")

    def test_route_cliff_query_routes_julia_ket_request_to_course_demo(self) -> None:
        decision = module.route_cliff_query("Show me the Julia version of KET")

        self.assertEqual(decision.route_name, "course_demo")

    def test_report_to_cliff_consciousness_selects_completed_report(self) -> None:
        decision = module.route_cliff_query("How comfortable is the Lovesac sectional sofa?")

        report = module.report_to_cliff_consciousness(
            "How comfortable is the Lovesac sectional sofa?",
            decision,
            artifact_path=Path("/tmp/product_feedback_dashboard.html"),
        )

        self.assertEqual(len(report.workspace_state.selected), 1)
        self.assertEqual(report.workspace_state.selected[0].process.source_agent, "product_feedback")
        self.assertEqual(report.workspace_state.deferred, ())

    def test_build_worker_command_includes_query_and_outdir(self) -> None:
        args = argparse.Namespace(
            route="auto",
            democritus_input_pdf="/tmp/uploaded_paper.pdf",
            democritus_input_pdf_dir="/tmp/uploaded_pdfs",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
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
            culinary_manifest="/tmp/culinary_stop_manifest.jsonl",
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
            course_repo_root="",
            course_no_execute=False,
            course_timeout_sec=900,
        )

        command = module._build_worker_command(
            args,
            run_outdir=Path("/tmp/cliff-run-0001"),
            query="How easy is it to drive the Mazda Miata 3?",
        )

        self.assertIn("-m", command)
        self.assertIn("functorflow_v3.cliff_worker", command)
        self.assertIn("How easy is it to drive the Mazda Miata 3?", command)
        self.assertIn("/tmp/cliff-run-0001", command)
        self.assertIn("--democritus-input-pdf", command)
        self.assertIn("/tmp/uploaded_paper.pdf", command)
        self.assertIn("--democritus-input-pdf-dir", command)
        self.assertIn("/tmp/uploaded_pdfs", command)
        self.assertIn("--democritus-manifold-mode", command)
        self.assertIn("moe", command)
        self.assertIn("--democritus-topk", command)
        self.assertIn("144", command)
        self.assertIn("--democritus-anchors", command)
        self.assertIn("resveratrol, red wine", command)
        self.assertIn("--democritus-title", command)
        self.assertIn("Red Wine Democritus", command)
        self.assertIn("--democritus-dedupe-focus", command)
        self.assertIn("--democritus-render-topk-pngs", command)
        self.assertIn("--democritus-write-deep-dive", command)
        self.assertIn("--culinary-manifest", command)
        self.assertIn("/tmp/culinary_stop_manifest.jsonl", command)
        self.assertIn("--course-timeout-sec", command)

    def test_run_cliff_session_query_moves_back_to_conscious_layer(self) -> None:
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
                    target_documents=5,
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
                return module.CLIFFRouterRunResult(
                    route_decision=module.CLIFFRouteDecision(
                        route_name="product_feedback",
                        module_name="functorflow_v3.product_feedback_query_agentic",
                        rationale="test",
                    ),
                    route_outdir=route_outdir,
                    summary_path=self.config.outdir / "ff2_query_router_summary.json",
                    product_feedback_result=result,
                )

        launcher = FakeLauncher()
        args = SimpleNamespace(
            outdir="/tmp/cliff",
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
            culinary_manifest="",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "_build_router_from_args_with_outdir") as build_router:
                with patch.object(module, "_open_artifact") as open_artifact:
                    build_router.return_value = FakeRouter(
                        SimpleNamespace(
                            outdir=Path(tmpdir),
                            query="How comfortable are Lovesac sofas?",
                        )
                    )
                    module._run_cliff_session_query(
                        launcher,
                        args,
                        run_id="run-0001",
                        query="How comfortable are Lovesac sofas?",
                    )

        self.assertEqual(launcher.updates[0]["mind_layer"], "conscious")
        self.assertEqual(launcher.updates[1]["mind_layer"], "unconscious")
        self.assertEqual(launcher.updates[-1]["mind_layer"], "conscious")
        self.assertEqual(launcher.updates[-1]["status"], "complete")
        self.assertIn("won't open automatically", launcher.updates[-1]["note"])
        self.assertIn("second synthesis pass", launcher.updates[-1]["note"])
        open_artifact.assert_not_called()

    def test_run_cliff_session_query_keeps_interactive_democritus_at_checkpoint(self) -> None:
        class FakeLauncher:
            def __init__(self) -> None:
                self.updates: list[dict[str, object]] = []

            def update_session_run(self, run_id: str, **kwargs) -> None:
                payload = {"run_id": run_id}
                payload.update(kwargs)
                self.updates.append(payload)

        checkpoint_path = Path("/tmp/democritus-interactive-checkpoint.html")
        checkpoint_path.write_text("<html>checkpoint</html>", encoding="utf-8")
        self.addCleanup(lambda: checkpoint_path.unlink(missing_ok=True))
        launcher = FakeLauncher()
        args = SimpleNamespace(
            outdir="/tmp/cliff",
            route="auto",
            execution_mode="interactive",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            culinary_manifest="",
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

        fake_result = module.CLIFFRouterRunResult(
            route_decision=module.CLIFFRouteDecision(
                route_name="democritus",
                module_name="functorflow_v3.democritus_query_agentic",
                rationale="test",
            ),
            route_outdir=Path("/tmp/cliff-run") / "democritus",
            summary_path=Path("/tmp/cliff-run") / "ff2_query_router_summary.json",
            democritus_result=SimpleNamespace(
                checkpoint_dashboard_path=checkpoint_path,
                corpus_synthesis_dashboard_path=None,
                batch_outdir=Path("/tmp/cliff-run") / "democritus" / "democritus_runs",
            ),
        )

        with patch.object(module, "_build_router_from_args_with_outdir") as build_router:
            with patch.object(module, "_build_cliff_synthesis_from_first_pass") as build_synthesis:
                build_router.return_value = SimpleNamespace(run=lambda: fake_result)
                module._run_cliff_session_query(
                    launcher,
                    args,
                    run_id="run-0007",
                    query="Analyze 10 recent studies of minimum wage and synthesize their joint support",
                )

        self.assertEqual(launcher.updates[-1]["status"], "complete")
        self.assertEqual(launcher.updates[-1]["artifact_path"], checkpoint_path)
        self.assertNotIn("second synthesis pass", launcher.updates[-1]["note"])
        build_synthesis.assert_not_called()

    def test_run_cliff_session_query_stops_at_democritus_clarification_checkpoint(self) -> None:
        class FakeLauncher:
            def __init__(self) -> None:
                self.updates: list[dict[str, object]] = []

            def update_session_run(self, run_id: str, **kwargs) -> None:
                payload = {"run_id": run_id}
                payload.update(kwargs)
                self.updates.append(payload)

        clarification_path = Path("/tmp/democritus-query-clarification.html")
        clarification_path.write_text("<html>clarification</html>", encoding="utf-8")
        self.addCleanup(lambda: clarification_path.unlink(missing_ok=True))
        launcher = FakeLauncher()
        args = SimpleNamespace(
            outdir="/tmp/cliff",
            route="auto",
            execution_mode="quick",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            culinary_manifest="",
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

        fake_result = module.CLIFFRouterRunResult(
            route_decision=module.CLIFFRouteDecision(
                route_name="democritus",
                module_name="functorflow_v3.democritus_query_agentic",
                rationale="test",
            ),
            route_outdir=Path("/tmp/cliff-run") / "democritus",
            summary_path=Path("/tmp/cliff-run") / "ff2_query_router_summary.json",
            democritus_result=SimpleNamespace(
                query_plan=SimpleNamespace(
                    clarification_request=SimpleNamespace(ambiguous_term="inflation"),
                ),
                clarification_dashboard_path=clarification_path,
                checkpoint_dashboard_path=None,
                corpus_synthesis_dashboard_path=None,
                batch_outdir=Path("/tmp/cliff-run") / "democritus" / "democritus_runs",
            ),
        )

        with patch.object(module, "_build_router_from_args_with_outdir") as build_router:
            with patch.object(module, "_build_cliff_synthesis_from_first_pass") as build_synthesis:
                build_router.return_value = SimpleNamespace(run=lambda: fake_result)
                module._run_cliff_session_query(
                    launcher,
                    args,
                    run_id="run-0008",
                    query="Analyze 20 recent studies on inflation and synthesize their joint support",
                )

        self.assertEqual(launcher.updates[-1]["status"], "complete")
        self.assertEqual(launcher.updates[-1]["artifact_path"], clarification_path)
        self.assertIn("paused before retrieval", launcher.updates[-1]["note"])
        self.assertNotIn("second synthesis pass", launcher.updates[-1]["note"])
        build_synthesis.assert_not_called()

    def test_interactive_company_similarity_is_treated_as_checkpoint(self) -> None:
        decision = module.CLIFFRouteDecision(
            route_name="company_similarity",
            module_name="functorflow_v3.company_similarity_agentic",
            rationale="test",
        )

        self.assertTrue(
            module._should_pause_at_interactive_checkpoint(
                decision,
                execution_mode="interactive",
            )
        )
        self.assertTrue(
            worker_module._should_complete_interactive_checkpoint(
                execution_mode="interactive",
                route_name="company_similarity",
            )
        )

    def test_monitor_cliff_session_worker_re_dispatches_after_first_pass(self) -> None:
        class FakeLauncher:
            def __init__(self) -> None:
                self.updates: list[dict[str, object]] = []

            def update_session_run(self, run_id: str, **kwargs) -> None:
                payload = {"run_id": run_id}
                payload.update(kwargs)
                self.updates.append(payload)

        class FakeProcess:
            def wait(self):
                return 0

        class FakeHandle:
            def close(self) -> None:
                return None

        launcher = FakeLauncher()
        args = argparse.Namespace(
            outdir="/tmp/cliff",
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
            culinary_manifest="",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            run_outdir = Path(tmpdir)
            result_path = run_outdir / module._WORKER_RESULT_FILENAME
            artifact_path = run_outdir / "basket_rocket_sec" / "workflow_batches" / "basket_rocket_gui.html"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("<html>first pass</html>", encoding="utf-8")
            result_path.write_text(
                json.dumps(
                    {
                        "status": "phase1_complete",
                        "artifact_path": str(artifact_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            worker = module._ActiveCLIFFWorker(
                process=FakeProcess(),
                stdout_handle=FakeHandle(),
                stderr_handle=FakeHandle(),
                run_outdir=run_outdir,
                stage="first_pass",
            )
            active_runs = {"run-0003": worker}
            resumed_worker = module._ActiveCLIFFWorker(
                process=FakeProcess(),
                stdout_handle=FakeHandle(),
                stderr_handle=FakeHandle(),
                run_outdir=run_outdir,
                stage="synthesis_pass",
            )
            decision = module.CLIFFRouteDecision(
                route_name="basket_rocket_sec",
                module_name="functorflow_v3.basket_rocket_sec_agentic",
                rationale="test",
            )
            with patch.object(module, "_launch_cliff_worker", return_value=resumed_worker):
                with patch.object(module.threading, "Thread") as thread_ctor:
                    thread_ctor.return_value = SimpleNamespace(start=lambda: None)
                    module._monitor_cliff_session_worker(
                        launcher,
                        args=args,
                        run_id="run-0003",
                        query="Analyze 10 recent 10-K filings from Adobe",
                        decision=decision,
                        active_runs=active_runs,
                        active_runs_lock=module.threading.Lock(),
                    )

        self.assertEqual(launcher.updates[0]["mind_layer"], "conscious")
        self.assertEqual(launcher.updates[0]["status"], "routing")
        self.assertIn("first pass", launcher.updates[0]["note"])
        self.assertEqual(launcher.updates[1]["mind_layer"], "unconscious")
        self.assertEqual(launcher.updates[1]["status"], "running")
        self.assertTrue(str(launcher.updates[1]["artifact_path"]).endswith("basket_rocket_sec/workflow_batches/corpus_synthesis/basket_rocket_corpus_synthesis.html"))

    def test_start_cliff_session_query_seeds_predicted_artifact_for_running_democritus(self) -> None:
        class FakeLauncher:
            def __init__(self) -> None:
                self.updates: list[dict[str, object]] = []

            def update_session_run(self, run_id: str, **kwargs) -> None:
                payload = {"run_id": run_id}
                payload.update(kwargs)
                self.updates.append(payload)

        class FakeProcess:
            def poll(self):
                return None

        launcher = FakeLauncher()
        args = argparse.Namespace(
            outdir="/tmp/cliff",
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
            culinary_manifest="",
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

        with patch.object(module, "route_cliff_query") as route_query:
            with patch.object(module.subprocess, "Popen", return_value=FakeProcess()):
                with patch.object(module.threading, "Thread") as thread_ctor:
                    route_query.return_value = module.CLIFFRouteDecision(
                        route_name="democritus",
                        module_name="functorflow_v3.democritus_query_agentic",
                        rationale="test",
                    )
                    thread_ctor.return_value = SimpleNamespace(start=lambda: None)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        active_runs: dict[str, object] = {}
                        module._start_cliff_session_query(
                            launcher,
                            args,
                            run_id="run-0002",
                            query="Find me 10 studies of red wine",
                            active_runs=active_runs,
                            active_runs_lock=module.threading.Lock(),
                        )
                        for worker in active_runs.values():
                            if getattr(worker, "stdout_handle", None):
                                worker.stdout_handle.close()
                            if getattr(worker, "stderr_handle", None):
                                worker.stderr_handle.close()

        self.assertEqual(launcher.updates[0]["status"], "routing")
        self.assertEqual(launcher.updates[1]["status"], "running")
        self.assertTrue(
            str(launcher.updates[1]["artifact_path"]).endswith(
                "democritus/democritus_runs/democritus_gui.html"
            )
        )

    def test_launch_cliff_worker_sets_unbuffered_python(self) -> None:
        args = argparse.Namespace(
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
            culinary_manifest="",
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
            course_repo_root="",
            course_book_pdf_path="",
            course_no_execute=False,
            course_timeout_sec=900,
        )

        captured: dict[str, object] = {}

        class FakeProcess:
            pass

        def fake_popen(*popen_args, **popen_kwargs):
            captured["args"] = popen_args
            captured["kwargs"] = popen_kwargs
            return FakeProcess()

        with patch.object(module.subprocess, "Popen", side_effect=fake_popen):
            with tempfile.TemporaryDirectory() as tmpdir:
                worker = module._launch_cliff_worker(
                    args,
                    run_outdir=Path(tmpdir),
                    query="How similar is Adobe to Nike?",
                    cycle_stage="first_pass",
                )
                if getattr(worker, "stdout_handle", None):
                    worker.stdout_handle.close()
                if getattr(worker, "stderr_handle", None):
                    worker.stderr_handle.close()

        self.assertEqual(captured["kwargs"]["env"]["PYTHONUNBUFFERED"], "1")

    def test_cliff_worker_completes_interactive_democritus_checkpoint_without_phase2_status(self) -> None:
        checkpoint_path = Path("/tmp/democritus_topic_checkpoint.html")
        checkpoint_path.write_text("<html>checkpoint</html>", encoding="utf-8")
        self.addCleanup(lambda: checkpoint_path.unlink(missing_ok=True))

        args = SimpleNamespace(
            query="Analyze 10 recent studies of minimum wage and synthesize their joint support",
            outdir="/tmp/cliff-worker-interactive",
            cycle_stage="first_pass",
            execution_mode="interactive",
            cliff_defer_final_synthesis=True,
            route="auto",
            democritus_input_pdf="",
            democritus_input_pdf_dir="",
            democritus_manifest="",
            democritus_source_pdf_root="",
            democritus_target_docs=None,
            democritus_retrieval_backend="auto",
            democritus_max_docs=None,
            democritus_intra_document_shards=1,
            democritus_manifold_mode="full",
            democritus_topk=200,
            democritus_radii="1,2,3",
            democritus_maxnodes="10,20,30,40,60",
            democritus_lambda_edge=0.25,
            democritus_topk_models=5,
            democritus_topk_claims=30,
            democritus_alpha=1.0,
            democritus_tier1=0.60,
            democritus_tier2=0.30,
            democritus_anchors="",
            democritus_title="",
            democritus_dedupe_focus=False,
            democritus_require_anchor_in_focus=False,
            democritus_focus_blacklist_regex="",
            democritus_render_topk_pngs=False,
            democritus_assets_dir="assets",
            democritus_png_dpi=200,
            democritus_write_deep_dive=False,
            democritus_deep_dive_max_bullets=8,
            democritus_discovery_only=False,
            democritus_dry_run=False,
            product_manifest="",
            culinary_manifest="",
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
            course_repo_root="",
            course_book_pdf_path="",
            course_no_execute=False,
            course_timeout_sec=900,
        )

        fake_result = module.CLIFFRouterRunResult(
            route_decision=module.CLIFFRouteDecision(
                route_name="democritus",
                module_name="functorflow_v3.democritus_query_agentic",
                rationale="test",
            ),
            route_outdir=Path("/tmp/cliff-worker-interactive") / "democritus",
            summary_path=Path("/tmp/cliff-worker-interactive") / "ff2_query_router_summary.json",
            democritus_result=SimpleNamespace(
                checkpoint_dashboard_path=checkpoint_path,
                corpus_synthesis_dashboard_path=None,
                batch_outdir=Path("/tmp/cliff-worker-interactive") / "democritus" / "democritus_runs",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            args.outdir = str(outdir)
            with patch.object(worker_module, "_parse_args", return_value=args):
                with patch.object(worker_module, "route_cliff_query", return_value=fake_result.route_decision):
                    with patch.object(worker_module, "_build_router_from_args_with_outdir", return_value=SimpleNamespace(run=lambda: fake_result)):
                        worker_module.main()

            payload = json.loads((outdir / module._WORKER_RESULT_FILENAME).read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["artifact_path"], str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
