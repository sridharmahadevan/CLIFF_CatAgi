"""Tests for the dashboard query launcher."""

from __future__ import annotations

import io
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

        self.assertEqual(submission, (run_id, "Find me 5 studies of GLP-1"))

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
