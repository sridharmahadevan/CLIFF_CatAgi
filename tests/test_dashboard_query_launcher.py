"""Tests for the dashboard query launcher."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import dashboard_query_launcher as module
    from functorflow_v3.dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
except ModuleNotFoundError:
    from ..functorflow_v3 import dashboard_query_launcher as module
    from ..functorflow_v3.dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig


class DashboardQueryLauncherTests(unittest.TestCase):
    def test_session_mode_queues_submissions_and_exposes_run_state(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="BAFFLE Query Router",
                subtitle="Test session",
                query_label="BAFFLE query",
                query_placeholder="Find me 5 studies of GLP-1",
                submit_label="Route Query",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("Find me 5 studies of GLP-1")
        submission = launcher.wait_for_next_submission(timeout=0.01)

        self.assertEqual(submission, (run_id, "Find me 5 studies of GLP-1", "quick"))

        launcher.update_session_run(
            run_id,
            status="running",
            route_name="democritus",
            note="Running in the background.",
            outdir=Path("/tmp/ff2-run-0001"),
        )

        state = launcher._state_payload()

        self.assertTrue(state["session_mode"])
        self.assertEqual(state["runs"][0]["run_id"], run_id)
        self.assertEqual(state["runs"][0]["status"], "running")
        self.assertEqual(state["runs"][0]["route_name"], "democritus")
        self.assertEqual(state["runs"][0]["outdir"], "/tmp/ff2-run-0001")

    def test_request_session_run_deepen_queues_deep_followup(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 5 studies of GLP-1",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("Find me 5 studies of GLP-1", execution_mode="quick")
        launcher.wait_for_next_submission(timeout=0.01)
        launcher.update_session_run(
            run_id,
            status="complete",
            route_name="democritus",
            note="Finished.",
            artifact_path=Path("/tmp/result.html"),
            outdir=Path("/tmp/ff2-run-0001"),
        )

        deep_run_id = launcher.request_session_run_deepen(run_id)
        followup = launcher.wait_for_next_submission(timeout=0.01)

        self.assertIsNotNone(deep_run_id)
        self.assertEqual(followup, (deep_run_id, "Find me 5 studies of GLP-1", "deep"))
        state = launcher._state_payload()
        self.assertEqual(state["runs"][0]["execution_mode"], "deep")
        self.assertEqual(state["runs"][0]["parent_run_id"], run_id)
        self.assertEqual(
            launcher.submission_overrides_for_run(str(deep_run_id)),
            {"route": "democritus"},
        )

    def test_request_session_run_deepen_accepts_completed_interactive_runs(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of minimum wage",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("Find me 10 studies of minimum wage", execution_mode="interactive")
        launcher.wait_for_next_submission(timeout=0.01)
        launcher.update_session_run(
            run_id,
            status="complete",
            route_name="democritus",
            note="Interactive checkpoint ready.",
            artifact_path=Path("/tmp/checkpoint.html"),
            outdir=Path("/tmp/ff2-run-0002"),
        )

        deep_run_id = launcher.request_session_run_deepen(run_id)
        followup = launcher.wait_for_next_submission(timeout=0.01)

        self.assertIsNotNone(deep_run_id)
        self.assertEqual(followup, (deep_run_id, "Find me 10 studies of minimum wage", "deep"))
        self.assertEqual(
            launcher.submission_overrides_for_run(str(deep_run_id)),
            {"route": "democritus"},
        )

    def test_request_session_run_deepen_stores_curated_manifest_overrides(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of minimum wage",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("Find me 10 studies of minimum wage", execution_mode="interactive")
        launcher.wait_for_next_submission(timeout=0.01)
        launcher.update_session_run(
            run_id,
            status="complete",
            route_name="democritus",
            note="Interactive checkpoint ready.",
            artifact_path=Path("/tmp/checkpoint.html"),
            outdir=Path("/tmp/ff2-run-0002"),
        )

        deep_run_id = launcher.request_session_run_deepen(
            run_id,
            democritus_manifest_path=Path("/tmp/curated_manifest.json"),
            democritus_target_docs=2,
        )
        followup = launcher.wait_for_next_submission(timeout=0.01)

        self.assertEqual(followup, (deep_run_id, "Find me 10 studies of minimum wage", "deep"))
        self.assertEqual(
            launcher.submission_overrides_for_run(str(deep_run_id)),
            {
                "route": "democritus",
                "democritus_manifest": str(Path("/tmp/curated_manifest.json").resolve()),
                "democritus_target_docs": 2,
            },
        )

    def test_request_session_run_deepen_stores_company_similarity_year_overrides(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How similar is Adobe to Nike?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("How similar is Adobe to Nike?", execution_mode="interactive")
        launcher.wait_for_next_submission(timeout=0.01)
        launcher.update_session_run(
            run_id,
            status="complete",
            route_name="company_similarity",
            note="Interactive checkpoint ready.",
            artifact_path=Path("/tmp/company_similarity_checkpoint.html"),
            outdir=Path("/tmp/ff2-run-0003"),
        )

        deep_run_id = launcher.request_session_run_deepen(
            run_id,
            company_similarity_year_start=2011,
            company_similarity_year_end=2018,
        )
        followup = launcher.wait_for_next_submission(timeout=0.01)

        self.assertEqual(followup, (deep_run_id, "How similar is Adobe to Nike?", "deep"))
        self.assertEqual(
            launcher.submission_overrides_for_run(str(deep_run_id)),
            {
                "route": "company_similarity",
                "company_similarity_year_start": 2011,
                "company_similarity_year_end": 2018,
            },
        )

    def test_state_payload_discovers_archived_runs_from_worker_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = Path(tmpdir) / "archive"
            archived_run = archive_root / "cliff_session1-run-0007-20260411-090000-mediterranean_diet"
            artifact_path = archived_run / "democritus" / "democritus_runs" / "corpus_synthesis" / "democritus_corpus_synthesis.html"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("<html><body>Mediterranean archive</body></html>", encoding="utf-8")
            (archived_run / "cliff_worker_result.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "system_name": "CLIFF",
                        "query": "Analyze 10 recent studies of the Mediterranean diet and synthesize their joint support",
                        "route_decision": {"route_name": "democritus"},
                        "route_outdir": str((archived_run / "democritus").resolve()),
                        "artifact_path": str(artifact_path.resolve()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            launcher = DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle="Test session",
                    query_label="CLIFF query",
                    query_placeholder="Ask a question",
                    submit_label="Ask CLIFF",
                    waiting_message="Runs stay in the background.",
                    session_mode=True,
                    enable_execution_mode=True,
                    archive_roots=(archive_root,),
                )
            )
            self.addCleanup(launcher.close)

            payload = launcher._state_payload()
            rendered = launcher._render_run_artifact_page(archived_run.name)

        self.assertEqual(len(payload["archived_runs"]), 1)
        self.assertEqual(payload["archived_runs"][0]["run_id"], archived_run.name)
        self.assertEqual(payload["archived_runs"][0]["route_name"], "democritus")
        self.assertTrue(payload["archived_runs"][0]["archived"])
        self.assertIn("Mediterranean archive", rendered)

    def test_request_archived_run_rerun_queues_followup_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = Path(tmpdir) / "archive"
            archived_run = archive_root / "cliff_session1-run-0008-20260411-093000-kan_extension"
            archived_run.mkdir(parents=True, exist_ok=True)
            (archived_run / "cliff_worker_result.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "system_name": "CLIFF",
                        "query": "Explain the Kan Extension Transformer",
                        "route_decision": {"route_name": "course_demo"},
                        "route_outdir": str((archived_run / "course_demo").resolve()),
                        "artifact_path": str((archived_run / "course_demo" / "course_demo_dashboard.html").resolve()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            launcher = DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle="Test session",
                    query_label="CLIFF query",
                    query_placeholder="Ask a question",
                    submit_label="Ask CLIFF",
                    waiting_message="Runs stay in the background.",
                    session_mode=True,
                    enable_execution_mode=True,
                    archive_roots=(archive_root,),
                )
            )
            self.addCleanup(launcher.close)

            launcher._state_payload()
            rerun_id = launcher.request_archived_run_rerun(archived_run.name)
            queued = launcher.wait_for_next_submission(timeout=0.01)

        self.assertIsNotNone(rerun_id)
        self.assertEqual(queued, (rerun_id, "Explain the Kan Extension Transformer", "quick"))
        self.assertEqual(
            launcher.submission_overrides_for_run(str(rerun_id)),
            {"route": "course_demo"},
        )

    def test_archived_failed_worker_result_recovers_local_artifact_from_copied_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_root = Path(tmpdir) / "archive"
            archived_run = archive_root / "cliff_session1-run-0002-20260409-111432-global_warming"
            democritus_gui = archived_run / "democritus" / "democritus_runs" / "democritus_gui.html"
            democritus_gui.parent.mkdir(parents=True, exist_ok=True)
            democritus_gui.write_text("<html><body>Recovered archive GUI</body></html>", encoding="utf-8")
            (archived_run / "router_error.html").write_text("<html><body>Old error</body></html>", encoding="utf-8")
            (archived_run / "cliff_worker_result.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "system_name": "CLIFF",
                        "query": "Analyze 20 recent studies of global warming and what they jointly support",
                        "route_decision": {"route_name": "democritus"},
                        "error_artifact_path": (
                            "/private/tmp/cliff_session1-run-0002-20260409-111432-global_warming/router_error.html"
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            launcher = DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle="Test session",
                    query_label="CLIFF query",
                    query_placeholder="Ask a question",
                    submit_label="Ask CLIFF",
                    waiting_message="Runs stay in the background.",
                    session_mode=True,
                    enable_execution_mode=True,
                    archive_roots=(archive_root,),
                )
            )
            self.addCleanup(launcher.close)

            payload = launcher._state_payload()
            rendered = launcher._render_run_artifact_page(archived_run.name)

        self.assertEqual(len(payload["archived_runs"]), 1)
        archived = payload["archived_runs"][0]
        self.assertEqual(archived["status"], "complete")
        self.assertEqual(Path(archived["artifact_path"]).resolve(), democritus_gui.resolve())
        self.assertIn("Recovered archive GUI", rendered)

    def test_cached_archive_index_reloads_archived_runs_without_rescanning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_root = root / "archive"
            cache_dir = root / "cache"
            archived_run = archive_root / "cliff_session1-run-0009-20260411-101500-red_wine"
            archived_run.mkdir(parents=True, exist_ok=True)
            worker_result = archived_run / "cliff_worker_result.json"
            worker_result.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "system_name": "CLIFF",
                        "query": "Analyze 10 recent studies on red wine and synthesize what they jointly support",
                        "route_decision": {"route_name": "democritus"},
                        "route_outdir": str((archived_run / "democritus").resolve()),
                        "artifact_path": str((archived_run / "democritus" / "result.html").resolve()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            launcher = DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle="Test session",
                    query_label="CLIFF query",
                    query_placeholder="Ask a question",
                    submit_label="Ask CLIFF",
                    waiting_message="Runs stay in the background.",
                    session_mode=True,
                    enable_execution_mode=True,
                    archive_roots=(archive_root,),
                    archive_cache_dir=cache_dir,
                )
            )
            self.addCleanup(launcher.close)
            payload = launcher._state_payload()

            worker_result.unlink()

            launcher_cached = DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle="Test session",
                    query_label="CLIFF query",
                    query_placeholder="Ask a question",
                    submit_label="Ask CLIFF",
                    waiting_message="Runs stay in the background.",
                    session_mode=True,
                    enable_execution_mode=True,
                    archive_roots=(archive_root,),
                    archive_cache_dir=cache_dir,
                )
            )
            self.addCleanup(launcher_cached.close)
            cached_payload = launcher_cached._state_payload()

        self.assertEqual(len(payload["archived_runs"]), 1)
        self.assertEqual(len(cached_payload["archived_runs"]), 1)
        self.assertEqual(
            cached_payload["archived_runs"][0]["query"],
            "Analyze 10 recent studies on red wine and synthesize what they jointly support",
        )

    def test_render_run_artifact_page_for_democritus_checkpoint_includes_curation_controls(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            alpha_pdf = root / "alpha.pdf"
            beta_pdf = root / "beta.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\nalpha\n")
            beta_pdf.write_bytes(b"%PDF-1.4\nbeta\n")
            artifact_path = checkpoint_dir / "democritus_topic_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "democritus_topic_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "Find me 10 studies of minimum wage",
                        "stage_label": "Root Topic Checkpoint",
                        "summary_text": "Checkpoint ready.",
                        "query_focus_terms": ["minimum", "wage"],
                        "suspicious_topics": [{"topic": "household income effects"}],
                        "drift_metrics": {
                            "total_topic_count": 3,
                            "aligned_topic_count": 2,
                            "suspicious_topic_count": 1,
                            "aligned_topic_ratio": 0.667,
                            "mean_alignment_score": 0.55,
                            "synthesis_readiness_proxy": 0.55,
                        },
                        "documents": [
                            {
                                "run_name": "run_1",
                                "title": "Alpha study of minimum wage policy",
                                "pdf_path": str(alpha_pdf.resolve()),
                                "topics": ["minimum wage increases", "employment floor effects"],
                                "guide_summary": "Alpha summary",
                                "causal_gestalt": "Alpha gestalt",
                            },
                            {
                                "run_name": "run_2",
                                "title": "Beta study of household income effects",
                                "pdf_path": str(beta_pdf.resolve()),
                                "topics": ["minimum wage increases", "household income effects"],
                                "guide_summary": "Beta summary",
                                "causal_gestalt": "Beta gestalt",
                            },
                        ],
                        "top_topics": [{"topic": "minimum wage increases", "document_count": 2}],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Find me 10 studies of minimum wage", execution_mode="interactive")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="democritus",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

        self.assertIn('/checkpoint-action', rendered)
        self.assertIn('Include in deeper analysis', rendered)
        self.assertIn('Retrieve more documents', rendered)
        self.assertIn('Retrieve again from topic choices', rendered)
        self.assertIn('data-topic-state="neutral"', rendered)
        self.assertIn("selected_topic", rendered)
        self.assertIn("Atlas Drift Signal", rendered)
        self.assertIn("household income effects", rendered)
        self.assertIn('Inspect PDF', rendered)
        self.assertIn('/run-file?run_id=', rendered)

    def test_handle_checkpoint_action_deepen_writes_curated_manifest_and_queues_followup(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            alpha_pdf = root / "alpha.pdf"
            beta_pdf = root / "beta.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\nalpha\n")
            beta_pdf.write_bytes(b"%PDF-1.4\nbeta\n")
            artifact_path = checkpoint_dir / "democritus_topic_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "democritus_topic_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "Find me 10 studies of minimum wage",
                        "stage_label": "Root Topic Checkpoint",
                        "summary_text": "Checkpoint ready.",
                        "drift_metrics": {
                            "total_topic_count": 2,
                            "aligned_topic_count": 1,
                            "suspicious_topic_count": 1,
                            "aligned_topic_ratio": 0.5,
                            "mean_alignment_score": 0.5,
                            "synthesis_readiness_proxy": 0.5,
                        },
                        "documents": [
                            {
                                "run_name": "run_1",
                                "title": "Alpha study",
                                "pdf_path": str(alpha_pdf.resolve()),
                                "topics": ["minimum wage increases"],
                                "guide_summary": "Alpha summary",
                                "causal_gestalt": "Alpha gestalt",
                            },
                            {
                                "run_name": "run_2",
                                "title": "Beta study",
                                "pdf_path": str(beta_pdf.resolve()),
                                "topics": ["household income effects"],
                                "guide_summary": "Beta summary",
                                "causal_gestalt": "Beta gestalt",
                            },
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Find me 10 studies of minimum wage", execution_mode="interactive")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="democritus",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            body, status = launcher._handle_checkpoint_action(
                run_id=run_id,
                action_kind="deepen",
                selected_pdf_paths=(str(alpha_pdf.resolve()),),
                selected_topics=(),
                rejected_topics=(),
                additional_documents=3,
                retrieval_refinement="",
            )
            followup = launcher.wait_for_next_submission(timeout=0.01)

            self.assertEqual(status, module.HTTPStatus.OK)
            self.assertIsNotNone(followup)
            deep_run_id, followup_query, followup_mode = followup
            self.assertEqual(followup_query, "Find me 10 studies of minimum wage")
            self.assertEqual(followup_mode, "deep")
            self.assertIn("Deep Democritus Run Queued", body)
            overrides = launcher.submission_overrides_for_run(deep_run_id)
            curated_manifest_path = Path(str(overrides["democritus_manifest"]))
            payload = json.loads(curated_manifest_path.read_text(encoding="utf-8"))
            telemetry_log_path = checkpoint_dir / module._DEMOCRITUS_CURATION_TELEMETRY
            telemetry_summary_path = checkpoint_dir / module._DEMOCRITUS_CURATION_SUMMARY
            telemetry_events = [
                json.loads(line)
                for line in telemetry_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            telemetry_summary = json.loads(telemetry_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(overrides["democritus_target_docs"], 1)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["pdf_path"], str(alpha_pdf.resolve()))
        self.assertEqual(len(telemetry_events), 1)
        self.assertEqual(telemetry_events[0]["action_kind"], "deepen")
        self.assertEqual(telemetry_events[0]["action_status"], "queued")
        self.assertEqual(telemetry_events[0]["selected_count"], 1)
        self.assertEqual(telemetry_events[0]["rejected_count"], 1)
        self.assertEqual(telemetry_events[0]["topic_preference_signal"]["minimum wage increases"], 1)
        self.assertEqual(telemetry_events[0]["topic_preference_signal"]["household income effects"], -1)
        self.assertEqual(telemetry_events[0]["atlas_drift_metrics"]["suspicious_topic_count"], 1)
        self.assertEqual(telemetry_summary["event_count"], 1)
        self.assertEqual(telemetry_summary["action_counts"]["deepen"], 1)
        self.assertEqual(telemetry_summary["cumulative_topic_preference_signal"]["minimum wage increases"], 1)
        self.assertEqual(telemetry_summary["cumulative_topic_preference_signal"]["household income effects"], -1)

    def test_handle_checkpoint_action_topic_guided_retrieval_persists_topic_preferences_and_queues_followup(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            alpha_pdf = root / "alpha.pdf"
            beta_pdf = root / "beta.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\nalpha\n")
            beta_pdf.write_bytes(b"%PDF-1.4\nbeta\n")
            artifact_path = checkpoint_dir / "democritus_topic_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "democritus_topic_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "Find me 5 studies of unemployment in the tech sector",
                        "stage_label": "Root Topic Checkpoint",
                        "summary_text": "Checkpoint ready.",
                        "documents": [
                            {
                                "run_name": "run_1",
                                "title": "Alpha study",
                                "pdf_path": str(alpha_pdf.resolve()),
                                "topics": ["tech sector layoffs", "software hiring slowdown"],
                                "guide_summary": "Alpha summary",
                                "causal_gestalt": "Alpha gestalt",
                            },
                            {
                                "run_name": "run_2",
                                "title": "Beta study",
                                "pdf_path": str(beta_pdf.resolve()),
                                "topics": ["youth entrepreneurship", "software hiring slowdown"],
                                "guide_summary": "Beta summary",
                                "causal_gestalt": "Beta gestalt",
                            },
                        ],
                        "top_topics": [
                            {"topic": "tech sector layoffs", "document_count": 1},
                            {"topic": "software hiring slowdown", "document_count": 2},
                            {"topic": "youth entrepreneurship", "document_count": 1},
                        ],
                        "drift_metrics": {
                            "total_topic_count": 3,
                            "aligned_topic_count": 2,
                            "suspicious_topic_count": 1,
                            "aligned_topic_ratio": 0.667,
                            "mean_alignment_score": 0.611,
                            "synthesis_readiness_proxy": 0.611,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query(
                "Find me 5 studies of unemployment in the tech sector",
                execution_mode="interactive",
            )
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="democritus",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            body, status = launcher._handle_checkpoint_action(
                run_id=run_id,
                action_kind="topic_guided_retrieval",
                selected_pdf_paths=(str(alpha_pdf.resolve()), str(beta_pdf.resolve())),
                selected_topics=("tech sector layoffs", "software hiring slowdown"),
                rejected_topics=("youth entrepreneurship",),
                additional_documents=4,
                retrieval_refinement="peer reviewed longitudinal evidence",
            )
            followup = launcher.wait_for_next_submission(timeout=0.01)
            curation_payload = json.loads(
                (checkpoint_dir / module._DEMOCRITUS_CURATION_STATE).read_text(encoding="utf-8")
            )
            telemetry_log_path = checkpoint_dir / module._DEMOCRITUS_CURATION_TELEMETRY
            telemetry_events = [
                json.loads(line)
                for line in telemetry_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(status, module.HTTPStatus.OK)
        self.assertIsNotNone(followup)
        followup_run_id, followup_query, followup_mode = followup
        self.assertEqual(followup_mode, "interactive")
        self.assertIn("focus on topics: tech sector layoffs; software hiring slowdown", followup_query)
        self.assertIn("avoid topics: youth entrepreneurship", followup_query)
        self.assertIn("peer reviewed longitudinal evidence", followup_query)
        self.assertIn("Topic-Guided Retrieval Queued", body)
        overrides = launcher.submission_overrides_for_run(followup_run_id)
        self.assertEqual(overrides["route"], "democritus")
        self.assertEqual(overrides["democritus_target_docs"], 6)
        self.assertEqual(overrides["democritus_atlas_baseline"]["suspicious_topic_count"], 1)
        self.assertEqual(overrides["democritus_base_query"], "Find me 5 studies of unemployment in the tech sector")
        self.assertEqual(
            overrides["democritus_selected_topics"],
            ["tech sector layoffs", "software hiring slowdown"],
        )
        self.assertEqual(overrides["democritus_rejected_topics"], ["youth entrepreneurship"])
        self.assertEqual(overrides["democritus_retrieval_refinement"], "peer reviewed longitudinal evidence")
        self.assertEqual(
            curation_payload["selected_topics"],
            ["tech sector layoffs", "software hiring slowdown"],
        )
        self.assertEqual(curation_payload["rejected_topics"], ["youth entrepreneurship"])
        self.assertEqual(telemetry_events[0]["action_kind"], "topic_guided_retrieval")
        self.assertEqual(
            telemetry_events[0]["explicit_selected_topics"],
            ["tech sector layoffs", "software hiring slowdown"],
        )
        self.assertEqual(telemetry_events[0]["explicit_rejected_topics"], ["youth entrepreneurship"])
        self.assertEqual(telemetry_events[0]["topic_preference_signal"]["tech sector layoffs"], 2)

    def test_render_run_artifact_page_reports_drift_reduction_against_prior_atlas_baseline(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = checkpoint_dir / "democritus_topic_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "democritus_topic_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "Analyze five studies of climate change",
                        "stage_label": "Atlas Drift Checkpoint",
                        "summary_text": "Checkpoint ready.",
                        "query_focus_terms": ["climate", "change"],
                        "suspicious_topics": [],
                        "drift_metrics": {
                            "total_topic_count": 3,
                            "aligned_topic_count": 3,
                            "suspicious_topic_count": 0,
                            "aligned_topic_ratio": 1.0,
                            "mean_alignment_score": 0.8,
                            "synthesis_readiness_proxy": 0.8,
                        },
                        "documents": [],
                        "top_topics": [{"topic": "climate adaptation", "document_count": 2}],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query(
                "Analyze five studies of climate change",
                execution_mode="interactive",
                submission_overrides={
                    "democritus_atlas_baseline": {
                        "total_topic_count": 4,
                        "aligned_topic_count": 2,
                        "suspicious_topic_count": 2,
                        "aligned_topic_ratio": 0.5,
                        "mean_alignment_score": 0.45,
                        "synthesis_readiness_proxy": 0.45,
                    }
                },
            )
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="democritus",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

        self.assertIn("Drift tightened from 2 suspicious topics to 0.", rendered)
        self.assertIn("alignment 0.5", rendered)
        self.assertIn("readiness 0.45", rendered)

    def test_second_pass_checkpoint_telemetry_logs_drift_comparison(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            alpha_pdf = root / "alpha.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\nalpha\n")
            artifact_path = checkpoint_dir / "democritus_topic_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "democritus_topic_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "Analyze five studies of climate change",
                        "stage_label": "Atlas Drift Checkpoint",
                        "summary_text": "Checkpoint ready.",
                        "query_focus_terms": ["climate", "change"],
                        "suspicious_topics": [],
                        "drift_metrics": {
                            "total_topic_count": 2,
                            "aligned_topic_count": 2,
                            "suspicious_topic_count": 0,
                            "aligned_topic_ratio": 1.0,
                            "mean_alignment_score": 0.8,
                            "synthesis_readiness_proxy": 0.8,
                        },
                        "documents": [
                            {
                                "run_name": "run_1",
                                "title": "Alpha climate study",
                                "pdf_path": str(alpha_pdf.resolve()),
                                "topics": ["climate adaptation"],
                                "guide_summary": "Alpha summary",
                                "causal_gestalt": "Alpha gestalt",
                            }
                        ],
                        "top_topics": [{"topic": "climate adaptation", "document_count": 1}],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query(
                "Analyze five studies of climate change",
                execution_mode="interactive",
                submission_overrides={
                    "democritus_atlas_baseline": {
                        "total_topic_count": 3,
                        "aligned_topic_count": 1,
                        "suspicious_topic_count": 2,
                        "aligned_topic_ratio": 0.333,
                        "mean_alignment_score": 0.35,
                        "synthesis_readiness_proxy": 0.35,
                    }
                },
            )
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="democritus",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            launcher._handle_checkpoint_action(
                run_id=run_id,
                action_kind="save",
                selected_pdf_paths=(str(alpha_pdf.resolve()),),
                selected_topics=(),
                rejected_topics=(),
                additional_documents=2,
                retrieval_refinement="",
            )
            telemetry_log_path = checkpoint_dir / module._DEMOCRITUS_CURATION_TELEMETRY
            telemetry_events = [
                json.loads(line)
                for line in telemetry_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(telemetry_events[0]["atlas_drift_comparison"]["reduced_drift"])
        self.assertEqual(telemetry_events[0]["atlas_drift_comparison"]["suspicious_topic_delta"], 2)

    def test_render_run_artifact_page_for_company_similarity_checkpoint_includes_year_controls(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = checkpoint_dir / "company_similarity_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "company_similarity_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "How similar is Adobe to Nike?",
                        "company_a": "Adobe",
                        "company_b": "Nike",
                        "year_window": {"start": 2023, "end": 2025},
                        "suggested_year_window": {"start": 2015, "end": 2020},
                        "available_overlap_years": [2015, 2016, 2017],
                        "summary_text": "Initial similarity read.",
                        "partial_preview": {"status": "ready", "shared_edge_basis_size": 7},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("How similar is Adobe to Nike?", execution_mode="interactive")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="company_similarity",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

        self.assertIn('name="company_similarity_year_start"', rendered)
        self.assertIn('name="company_similarity_year_end"', rendered)
        self.assertIn("Go deeper on this window", rendered)
        self.assertIn("Available Overlap", rendered)

    def test_handle_company_similarity_checkpoint_action_deepen_queues_followup_with_year_window(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "interactive_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = checkpoint_dir / "company_similarity_checkpoint.html"
            artifact_path.write_text("<html>placeholder</html>", encoding="utf-8")
            (checkpoint_dir / "company_similarity_checkpoint.json").write_text(
                json.dumps(
                    {
                        "query": "How similar is Adobe to Nike?",
                        "company_a": "Adobe",
                        "company_b": "Nike",
                        "year_window": {"start": 2023, "end": 2025},
                        "suggested_year_window": {"start": 2015, "end": 2020},
                        "available_overlap_years": [2015, 2016, 2017],
                        "summary_text": "Initial similarity read.",
                        "partial_preview": {"status": "ready", "shared_edge_basis_size": 7},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("How similar is Adobe to Nike?", execution_mode="interactive")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="complete",
                route_name="company_similarity",
                note="Interactive checkpoint ready.",
                artifact_path=artifact_path,
                outdir=root,
            )

            body, status = launcher._handle_checkpoint_action(
                run_id=run_id,
                action_kind="deepen",
                selected_pdf_paths=(),
                selected_topics=(),
                rejected_topics=(),
                additional_documents=3,
                retrieval_refinement="",
                company_similarity_year_start=2014,
                company_similarity_year_end=2018,
            )
            followup = launcher.wait_for_next_submission(timeout=0.01)

        self.assertEqual(status, module.HTTPStatus.OK)
        self.assertIsNotNone(followup)
        deep_run_id, followup_query, followup_mode = followup
        self.assertEqual(followup_query, "How similar is Adobe to Nike?")
        self.assertEqual(followup_mode, "deep")
        self.assertIn("Deep Company-Similarity Run Queued", body)
        self.assertEqual(
            launcher.submission_overrides_for_run(deep_run_id),
            {
                "route": "company_similarity",
                "company_similarity_year_start": 2014,
                "company_similarity_year_end": 2018,
            },
        )

    def test_render_launcher_page_includes_demo_tour_queries(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                demo_queries=(
                    "Explain the Geometric Transformer on the Sudoku problem",
                    "Explain how the Kan Extension Transformer works",
                ),
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        markup = launcher._render_launcher_page()

        self.assertIn("Take the 2-minute tour", markup)
        self.assertIn("Explain the Geometric Transformer on the Sudoku problem", markup)
        self.assertIn("demo-tour-button", markup)

    def test_render_launcher_page_includes_latency_guide_for_execution_modes(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                enable_execution_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        markup = launcher._render_launcher_page()

        self.assertIn("Latency guide:", markup)
        self.assertIn("may still take several minutes even in quick mode", markup)

    def test_render_launcher_page_expands_cliff_in_hero_banner(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Ask a question",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        markup = launcher._render_launcher_page()

        self.assertIn("Conscious Layer Interface to Functor Flow", markup)
        self.assertIn("the conscious interface sits on top of the Functor Flow causal engine", markup)

    def test_render_session_runs_markup_highlights_completed_results(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How comfortable is it to drive the Mazda Miata 3?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        markup = launcher._render_session_runs_markup(
            [
                {
                    "run_id": "run-0001",
                    "query": "How comfortable is it to drive the Mazda Miata 3?",
                    "status": "complete",
                    "mind_layer": "conscious",
                    "route_name": "product_feedback",
                    "note": "Finished.",
                    "artifact_path": "/tmp/result.html",
                    "outdir": "/tmp/cliff-run-0001",
                }
            ]
        )

        self.assertIn("run-card-complete", markup)
        self.assertIn("status-complete", markup)
        self.assertIn("Open result", markup)
        self.assertIn("/run-artifact?run_id=run-0001", markup)
        self.assertIn("Longer analysis", markup)
        self.assertIn("Structured evidence and synthesis can take a few minutes.", markup)

    def test_render_session_runs_markup_uses_inspect_label_for_running_artifact(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of red wine",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        markup = launcher._render_session_runs_markup(
            [
                {
                    "run_id": "run-0002",
                    "query": "Find me 10 studies of red wine",
                    "status": "running",
                    "mind_layer": "unconscious",
                    "route_name": "democritus",
                    "note": "Running.",
                    "artifact_path": "/tmp/democritus_runs/democritus_gui.html",
                    "outdir": "/tmp/cliff-run-0002",
                }
            ]
        )

        self.assertIn("Inspect run", markup)
        self.assertNotIn("Open result", markup)
        self.assertIn("Deep research", markup)
        self.assertIn("Even quick mode may take several minutes while CLIFF builds causal state.", markup)

    def test_state_payload_promotes_route_eta_back_to_session_runs(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of red wine",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            democritus_telemetry = root / "democritus" / "democritus_runs" / "telemetry.json"
            democritus_telemetry.parent.mkdir(parents=True, exist_ok=True)
            democritus_telemetry.write_text(
                json.dumps(
                    {
                        "timing": {
                            "eta_ready": True,
                            "eta_human": "12m 00s",
                            "effective_parallelism": 4.7,
                            "peak_parallelism": 6.0,
                            "current_stage": "Triple Extraction Agent",
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Find me 10 studies of red wine")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="running",
                route_name="democritus",
                note="Running.",
                outdir=root,
                artifact_path=root / "democritus" / "democritus_runs" / "democritus_gui.html",
            )

            state = launcher._state_payload()
            markup = launcher._render_session_runs_markup(state["runs"])

        self.assertEqual(state["runs"][0]["eta_label"], "about 12m 00s remaining")
        self.assertEqual(state["runs"][0]["parallelism"], 4.7)
        self.assertEqual(state["runs"][0]["peak_parallelism"], 6.0)
        self.assertEqual(state["runs"][0]["current_stage"], "Triple Extraction Agent")
        self.assertIn("about 12m 00s remaining", markup)
        self.assertIn("parallelism 4.7 (peak 6.0)", markup)
        self.assertIn("stage Triple Extraction Agent", markup)

    def test_state_payload_promotes_democritus_llm_usage_summary(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of red wine",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            democritus_telemetry = root / "democritus" / "democritus_runs" / "telemetry.json"
            democritus_telemetry.parent.mkdir(parents=True, exist_ok=True)
            democritus_telemetry.write_text(
                json.dumps(
                    {
                        "llm_usage": {
                            "request_count": 18,
                            "requests_with_usage": 18,
                            "total_tokens": 12345,
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Find me 10 studies of red wine")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="running",
                route_name="democritus",
                note="Running.",
                outdir=root,
                artifact_path=root / "democritus" / "democritus_runs" / "democritus_gui.html",
            )

            state = launcher._state_payload()
            markup = launcher._render_session_runs_markup(state["runs"])

        self.assertEqual(state["runs"][0]["llm_usage_label"], "12,345 tokens across 18 LLM requests")
        self.assertIn("LLM usage:", markup)
        self.assertIn("12,345 tokens across 18 LLM requests", markup)

    def test_state_payload_promotes_company_similarity_peak_parallelism_and_stage(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Compare Adobe and Nike",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            telemetry_path = root / "company_similarity" / "company_similarity_telemetry.json"
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_path.write_text(
                json.dumps(
                    {
                        "timing": {
                            "observed_parallelism": 1.0,
                            "peak_parallelism": 2.0,
                            "current_stage": "Cross-company functor comparison",
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Compare Adobe and Nike")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="running",
                route_name="company_similarity",
                note="Running.",
                outdir=root,
                artifact_path=root / "company_similarity" / "company_similarity.html",
            )

            state = launcher._state_payload()
            markup = launcher._render_session_runs_markup(state["runs"])

        self.assertEqual(state["runs"][0]["parallelism"], 1.0)
        self.assertEqual(state["runs"][0]["peak_parallelism"], 2.0)
        self.assertEqual(state["runs"][0]["current_stage"], "Cross-company functor comparison")
        self.assertIn("parallelism 1.0 (peak 2.0)", markup)
        self.assertIn("stage Cross-company functor comparison", markup)

    def test_state_payload_promotes_company_similarity_inner_democritus_stage_from_logs(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Compare Adobe and Nike",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cliff_worker_first_pass_stdout.log").write_text(
                "\n".join(
                    [
                        "[company_similarity] resolved query to Adobe vs Nike",
                        "[company_similarity] ensuring company analysis for Adobe",
                        "[company_similarity][Adobe] [Module 4] Recovered 480 triples from 900 statements so far…",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            telemetry_path = root / "company_similarity" / "company_similarity_telemetry.json"
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_path.write_text(
                json.dumps(
                    {
                        "timing": {
                            "observed_parallelism": 1.0,
                            "peak_parallelism": 2.0,
                            "current_stage": "Adobe build",
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_id = launcher.submit_query("Compare Adobe and Nike")
            launcher.wait_for_next_submission(timeout=0.01)
            launcher.update_session_run(
                run_id,
                status="running",
                route_name="company_similarity",
                note="Running.",
                outdir=root,
                artifact_path=root / "company_similarity" / "company_similarity.html",
            )

            state = launcher._state_payload()
            markup = launcher._render_session_runs_markup(state["runs"])

        self.assertEqual(state["runs"][0]["current_stage"], "Adobe: recovering triples (480)")
        self.assertIn("stage Adobe: recovering triples (480)", markup)

    def test_render_run_artifact_page_returns_specific_completed_artifact(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How comfortable is it to drive the Mazda Miata 3?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        artifact_path = Path("/tmp/cliff-run-artifact-test.html")
        artifact_path.write_text("<html><body>Miata result</body></html>", encoding="utf-8")
        self.addCleanup(lambda: artifact_path.unlink(missing_ok=True))

        run_id = launcher.submit_query("How comfortable is it to drive the Mazda Miata 3?")
        launcher.update_session_run(
            run_id,
            status="complete",
            mind_layer="conscious",
            route_name="product_feedback",
            note="Finished.",
            artifact_path=artifact_path,
            outdir=Path("/tmp/cliff-run-0001"),
        )

        html = launcher._render_run_artifact_page(run_id)

        self.assertIn("Miata result", html)

    def test_render_run_artifact_page_wraps_running_artifact_with_refresh_shell(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 studies of red wine",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "democritus_runs" / "democritus_gui.html"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("<html><body>Partial Democritus GUI</body></html>", encoding="utf-8")

            run_id = launcher.submit_query("Find me 10 studies of red wine")
            launcher.update_session_run(
                run_id,
                status="running",
                mind_layer="unconscious",
                route_name="democritus",
                note="Running.",
                artifact_path=artifact_path,
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

            self.assertIn("Inspecting Run", rendered)
            self.assertIn("/run-file?run_id=", rendered)
            self.assertIn("setInterval", rendered)
            self.assertIn("/state?ts=", rendered)
            self.assertIn("clearInterval(refreshTimer)", rendered)
            self.assertIn("terminalStatuses", rendered)

    def test_render_run_artifact_page_shows_company_similarity_progress_dashboard_while_running(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How similar is Apple to Tesla?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cliff_worker_first_pass_stdout.log").write_text(
                "\n".join(
                    [
                        "[company_similarity] resolved query to Apple vs Tesla",
                        "[company_similarity] ensuring company analysis for Apple",
                        "[company_similarity] ensuring company analysis for Tesla",
                        "[company_similarity][Apple] [run_brand_financial_filings] year=2002 staged_pdfs=1 rows=1",
                        "[company_similarity][Tesla] [run_brand_financial_filings] year=2002 staged_pdfs=1 rows=1",
                        "[company_similarity][Apple] [run_brand_financial_filings] year=2002 launching atlas build outdir=/tmp/apple/atlas_apple_2002",
                        "[company_similarity][Apple] [run_brand_financial_filings] year=2002 atlas build completed outdir=/tmp/apple/atlas_apple_2002",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            route_root = root / "company_similarity" / "apple_vs_tesla_functors"
            route_root.mkdir(parents=True, exist_ok=True)
            (route_root / "cross_company_functors_summary.md").write_text("# Partial summary\n", encoding="utf-8")
            (route_root / "partial").mkdir(parents=True, exist_ok=True)
            (route_root / "partial" / "cross_company_functors_summary.md").write_text(
                "# Initial similarity read\n\n- Mean yearly cosine similarity: 0.62\n",
                encoding="utf-8",
            )
            (route_root / "partial" / "cross_company_functors_manifest.json").write_text(
                '{"overlap_years":[2002],"shared_edge_basis_size":18}',
                encoding="utf-8",
            )
            (root / "company_similarity" / "company_similarity_telemetry.json").write_text(
                json.dumps(
                    {
                        "partial_preview": {
                            "status": "ready",
                            "note": "Initial similarity read is ready from 1 overlapping year and 18 shared basis edges.",
                            "summary_path": str((route_root / "partial" / "cross_company_functors_summary.md").resolve()),
                            "manifest_path": str((route_root / "partial" / "cross_company_functors_manifest.json").resolve()),
                            "overlap_years": [2002],
                            "shared_edge_basis_size": 18,
                        },
                        "slowest_stages": [{"label": "Apple build", "duration_human": "3m 0s", "duration_seconds": 180.0}],
                        "timing": {
                            "elapsed_human": "4m 0s",
                            "eta_human": "6m 0s",
                            "completed_work_human": "<1s",
                            "observed_work_human": "4m 0s",
                            "observed_parallelism": 1.75,
                        },
                    }
                ),
                encoding="utf-8",
            )

            run_id = launcher.submit_query("How similar is Apple to Tesla?")
            launcher.update_session_run(
                run_id,
                status="running",
                mind_layer="unconscious",
                route_name="company_similarity",
                note="Running.",
                artifact_path=root / "company_similarity" / "company_similarity_dashboard.html",
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

            self.assertIn("Progress", rendered)
            self.assertIn("Performance", rendered)
            self.assertIn("Recent Activity", rendered)
            self.assertIn("Current phase", rendered)
            self.assertIn("Democritus stage", rendered)
            self.assertIn("Rough ETA", rendered)
            self.assertIn("about ", rendered)
            self.assertIn("Apple vs Tesla", rendered)
            self.assertIn("Apple build", rendered)
            self.assertIn("Tesla build", rendered)
            self.assertIn("Building Apple and Tesla: yearly atlas", rendered)
            self.assertIn("Apple and Tesla: yearly atlas", rendered)
            self.assertIn("Initial Similarity Read", rendered)
            self.assertIn("Initial similarity read is ready from 1 overlapping year and 18 shared basis edges.", rendered)
            self.assertIn("Mean yearly cosine similarity: 0.62", rendered)
            self.assertIn("partial summary markdown", rendered)
            self.assertIn("Active builds", rendered)
            self.assertIn("Atlas years ready", rendered)
            self.assertIn(">1</strong>", rendered)
            self.assertIn("Observed parallelism", rendered)
            self.assertIn("1.75", rendered)
            self.assertIn("Observed work", rendered)
            self.assertIn("4m 0s", rendered)
            self.assertIn("Apple build", rendered)
            self.assertIn("3m 0s", rendered)
            self.assertIn("Live Files", rendered)
            self.assertIn("cliff_worker_first_pass_stdout.log", rendered)
            self.assertIn("cross_company_functors_summary.md", rendered)
            self.assertIn("company_similarity_telemetry.json", rendered)
            self.assertIn("/run-file?run_id=", rendered)

    def test_render_run_artifact_page_clarifies_waiting_for_completed_yearly_atlas(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How similar is Adobe to Walmart?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cliff_worker_first_pass_stdout.log").write_text(
                "\n".join(
                    [
                        "[company_similarity] resolved query to Adobe vs Walmart",
                        "[company_similarity] ensuring company analysis for Adobe",
                        "[company_similarity] ensuring company analysis for Walmart",
                        "[company_similarity][Adobe] [run_brand_financial_filings] year=2025 staged_pdfs=1 rows=1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "company_similarity").mkdir(parents=True, exist_ok=True)
            (root / "company_similarity" / "company_similarity_telemetry.json").write_text(
                json.dumps(
                    {
                        "partial_preview": {
                            "status": "warming_up",
                            "note": "Waiting for the first usable yearly atlas slice from Adobe.",
                        },
                        "timing": {
                            "elapsed_human": "3m 29s",
                            "eta_human": "9m 40s",
                            "observed_work_human": "6m 57s",
                            "observed_parallelism": 2.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            run_id = launcher.submit_query("How similar is Adobe to Walmart?")
            launcher.update_session_run(
                run_id,
                status="running",
                mind_layer="unconscious",
                route_name="company_similarity",
                note="Running.",
                artifact_path=root / "company_similarity" / "company_similarity_dashboard.html",
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

            self.assertIn("Waiting for the first completed yearly atlas from Adobe.", rendered)
            self.assertIn("Filings have been staged through 2025, but the first yearly atlas is not complete yet.", rendered)
            self.assertIn("Atlas years ready", rendered)
            self.assertIn(">0</strong>", rendered)

    def test_render_run_artifact_page_surfaces_inner_democritus_substage_for_company_similarity(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How similar is Apple to Adobe?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cliff_worker_first_pass_stdout.log").write_text(
                "\n".join(
                    [
                        "[company_similarity] resolved query to Apple vs Adobe",
                        "[company_similarity] ensuring company analysis for Apple",
                        "[company_similarity][Apple] [run_brand_financial_filings] year=2025 staged_pdfs=1 rows=1",
                        "[company_similarity][Apple] [Module 4] Recovered 480 triples from 900 statements so far…",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "company_similarity").mkdir(parents=True, exist_ok=True)
            (root / "company_similarity" / "company_similarity_telemetry.json").write_text(
                json.dumps(
                    {
                        "partial_preview": {
                            "status": "warming_up",
                            "note": "Waiting for the first usable yearly atlas slice from Apple.",
                        },
                        "timing": {
                            "elapsed_human": "4m 00s",
                            "eta_human": "8m 00s",
                            "observed_work_human": "4m 00s",
                            "observed_parallelism": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            run_id = launcher.submit_query("How similar is Apple to Adobe?")
            launcher.update_session_run(
                run_id,
                status="running",
                mind_layer="unconscious",
                route_name="company_similarity",
                note="Running.",
                artifact_path=root / "company_similarity" / "company_similarity_dashboard.html",
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

            self.assertIn("Building Apple: recovering triples (480)", rendered)
            self.assertIn("Democritus stage", rendered)
            self.assertIn("Apple: recovering triples (480)", rendered)
            self.assertIn("Apple is still recovering triples (480).", rendered)

    def test_render_run_artifact_page_rewrites_file_links_to_launcher_endpoint(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 Adobe 10-K filings",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visualizations = root / "visualizations"
            visualizations.mkdir(parents=True, exist_ok=True)
            artifact_path = root / "basket_rocket_gui.html"
            linked_path = visualizations / "index.html"
            linked_path.write_text("<html><body>Visualization suite</body></html>", encoding="utf-8")
            artifact_path.write_text(
                (
                    '<html><body>'
                    f'<a href="file://{linked_path.resolve()}">Open suite</a>'
                    '</body></html>'
                ),
                encoding="utf-8",
            )

            run_id = launcher.submit_query("Find me 10 Adobe 10-K filings")
            launcher.update_session_run(
                run_id,
                status="complete",
                mind_layer="conscious",
                route_name="basket_rocket_sec",
                note="Finished.",
                artifact_path=artifact_path,
                outdir=root,
            )

            rendered = launcher._render_run_artifact_page(run_id)

            self.assertIn("/run-file?run_id=", rendered)
            self.assertIn("Visualization suite", launcher._render_run_file_response(run_id, str(linked_path))[0].decode("utf-8"))

    def test_render_run_file_response_wraps_markdown_as_html(self) -> None:
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="Find me 10 Adobe 10-K filings",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
            )
        )
        self.addCleanup(launcher.close)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "basket_rocket_gui.html"
            report_path = root / "workflow_report.md"
            artifact_path.write_text("<html><body>Root</body></html>", encoding="utf-8")
            report_path.write_text("# Workflow report\n\nAdobe filing summary", encoding="utf-8")

            run_id = launcher.submit_query("Find me 10 Adobe 10-K filings")
            launcher.update_session_run(
                run_id,
                status="complete",
                mind_layer="conscious",
                route_name="basket_rocket_sec",
                note="Finished.",
                artifact_path=artifact_path,
                outdir=root,
            )

            body, content_type, status = launcher._render_run_file_response(run_id, str(report_path))

            self.assertEqual(status, module.HTTPStatus.OK)
            self.assertEqual(content_type, "text/html; charset=utf-8")
            self.assertIn("Workflow report", body.decode("utf-8"))

    def test_request_session_run_stop_updates_state_and_invokes_handler(self) -> None:
        calls: list[tuple[str, str]] = []
        launcher = DashboardQueryLauncher(
            DashboardQueryLauncherConfig(
                title="CLIFF",
                subtitle="Test session",
                query_label="CLIFF query",
                query_placeholder="How comfortable is it to drive the Mazda Miata 3?",
                submit_label="Ask CLIFF",
                waiting_message="Runs stay in the background.",
                session_mode=True,
                run_control_handler=lambda action, run_id: calls.append((action, run_id)),
            )
        )
        self.addCleanup(launcher.close)

        run_id = launcher.submit_query("How easy is it to drive the Mazda Miata 3?")

        stopped = launcher.request_session_run_stop(run_id)
        state = launcher._state_payload()

        self.assertTrue(stopped)
        self.assertEqual(calls, [("stop", run_id)])
        self.assertEqual(state["runs"][0]["status"], "stopping")
        self.assertEqual(state["runs"][0]["mind_layer"], "conscious")

    def test_main_opens_launcher_and_prints_submission(self) -> None:
        events: list[str] = []

        class FakeLauncher:
            url = "http://127.0.0.1:9999/"

            def __init__(self, config) -> None:
                self.config = config

            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

            def wait_for_submission(self) -> str:
                events.append("wait")
                return "Find me 10 studies on resveratrol"

        stdout = io.StringIO()
        with patch.object(module, "_parse_args", return_value=SimpleNamespace(
            title="FunctorFlow Query Launcher",
            subtitle="Test subtitle",
            query_label="Query",
            query_placeholder="placeholder",
            submit_label="Submit",
            waiting_message="waiting",
            eyebrow="FunctorFlow Launcher",
        )):
            with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
                with patch("sys.stdout", stdout):
                    module.main()

        payload = stdout.getvalue()
        self.assertIn('"url": "http://127.0.0.1:9999/"', payload)
        self.assertIn('"query": "Find me 10 studies on resveratrol"', payload)
        self.assertEqual(events, ["enter", "wait", "exit"])


if __name__ == "__main__":
    unittest.main()
