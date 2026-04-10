"""Tests for company similarity portability helpers."""

from __future__ import annotations

import csv
import io
import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    from functorflow_v3 import company_similarity_agentic as module
except ModuleNotFoundError:
    from ..functorflow_v3 import company_similarity_agentic as module


class CompanySimilarityAgenticTests(unittest.TestCase):
    def test_extract_companies_from_query_accepts_short_numeric_brand_alias(self) -> None:
        adobe = module._CompanyRecord(
            brand="Adobe",
            slug="adobe",
            aliases=("adobe", "adbe"),
            ticker="adbe",
            outdir=Path("/tmp/adobe"),
        )
        three_m = module._CompanyRecord(
            brand="3M",
            slug="3m",
            aliases=("3m", "mmm"),
            ticker="mmm",
            outdir=Path("/tmp/3m"),
        )

        company_a, company_b = module._extract_companies_from_query(
            "How similar is Adobe to 3M?",
            {"adobe": adobe, "3m": three_m},
        )

        self.assertEqual(company_a.brand, "Adobe")
        self.assertEqual(company_b.brand, "3M")

    def test_quick_mode_profile_uses_recent_year_window(self) -> None:
        profile = module._company_similarity_mode_profile("quick")

        self.assertEqual(profile["execution_mode"], "quick")
        self.assertGreaterEqual(int(profile["year_end"]) - int(profile["year_start"]), 0)
        self.assertLessEqual(int(profile["year_end"]) - int(profile["year_start"]), 3)
        self.assertEqual(profile["jobs"], 3)
        self.assertEqual(profile["llm_jobs"], 3)
        self.assertEqual(profile["epochs"], 1)
        self.assertEqual(profile["batch_size"], 6)
        self.assertEqual(profile["skip_visualization"], 1)
        self.assertEqual(profile["skip_branch_visuals"], 1)

    def test_interactive_mode_profile_preserves_explicit_year_window(self) -> None:
        profile = module._company_similarity_mode_profile("interactive", year_start=2011, year_end=2014)

        self.assertEqual(profile["execution_mode"], "interactive")
        self.assertEqual(profile["year_start"], 2011)
        self.assertEqual(profile["year_end"], 2014)
        self.assertEqual(profile["jobs"], 3)

    def test_format_duration_preserves_subsecond_work(self) -> None:
        self.assertEqual(module._format_duration(0.25), "<1s")
        self.assertEqual(module._format_duration(1.0), "1s")

    def test_build_telemetry_payload_reports_observed_parallelism(self) -> None:
        runner = module.CompanySimilarityAgenticRunner(
            "Compare Adobe and Nike",
            Path("/tmp/company_similarity"),
        )
        plan = module.CompanySimilarityQueryPlan(
            query="Compare Adobe and Nike",
            company_a="Adobe",
            company_b="Nike",
            company_a_slug="adobe",
            company_b_slug="nike",
        )
        stage_state = {
            "query": {
                "label": "Resolve query",
                "status": "complete",
                "started_at_epoch": 100.0,
                "ended_at_epoch": 100.0,
            },
            "company_a_analysis": {
                "label": "Adobe build",
                "status": "complete",
                "started_at_epoch": 100.0,
                "ended_at_epoch": 160.0,
            },
            "company_b_analysis": {
                "label": "Nike build",
                "status": "complete",
                "started_at_epoch": 100.0,
                "ended_at_epoch": 160.0,
            },
            "functor_analysis": {
                "label": "Cross-company functor comparison",
                "status": "pending",
                "started_at_epoch": 0.0,
                "ended_at_epoch": 0.0,
            },
            "visualization": {
                "label": "Visualization",
                "status": "pending",
                "started_at_epoch": 0.0,
                "ended_at_epoch": 0.0,
            },
        }

        with mock.patch.object(module.time, "time", return_value=160.0):
            payload = runner._build_telemetry_payload(
                plan=plan,
                started_at=100.0,
                stage_state=stage_state,
                status="running",
                note="Testing parallelism.",
            )

        timing = payload["timing"]
        self.assertEqual(timing["completed_work_seconds"], 120.0)
        self.assertEqual(timing["observed_work_seconds"], 120.0)
        self.assertEqual(timing["observed_parallelism"], 2.0)
        self.assertEqual(timing["peak_parallelism"], 2.0)
        self.assertEqual(timing["current_stage"], "Cross-company functor comparison")

    def test_render_company_similarity_performance_html_prefers_observed_work(self) -> None:
        html = module._render_company_similarity_performance_html(
            {
                "status": "running",
                "note": "Testing observed work.",
                "timing": {
                    "elapsed_human": "1m 36s",
                    "current_stage": "Walmart build",
                    "observed_work_human": "1m 36s",
                    "completed_work_human": "<1s",
                    "observed_parallelism": 1.0,
                    "peak_parallelism": 2.0,
                    "eta_human": "6m 28s",
                },
                "stages": [],
                "slowest_stages": [],
            }
        )

        self.assertIn("Observed Work", html)
        self.assertIn("1m 36s", html)
        self.assertIn("Completed Stages", html)
        self.assertIn("&lt;1s", html)

    def test_refresh_partial_similarity_preview_builds_provisional_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = module.CompanySimilarityAgenticRunner(
                "Compare Adobe and Nike",
                root / "company_similarity",
            )
            record_a = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe",),
                outdir=root / "outputs" / "adobe",
            )
            record_b = module._CompanyRecord(
                brand="Nike",
                slug="nike",
                aliases=("nike",),
                outdir=root / "outputs" / "nike",
            )
            for record, years in ((record_a, (2023,)), (record_b, (2023, 2024))):
                filings_outdir = record.outdir / f"runs_{record.slug}_financial_filings"
                filings_outdir.mkdir(parents=True, exist_ok=True)
                for year in years:
                    atlas_dir = filings_outdir / f"atlas_{record.slug}_{year}"
                    atlas_dir.mkdir(parents=True, exist_ok=True)
                    (atlas_dir / "atlas_edges.parquet").write_text("", encoding="utf-8")

            commands: list[list[str]] = []

            def fake_run_command(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
                del cwd, stream_label
                commands.append(command)
                outdir = Path(command[command.index("--outdir") + 1])
                outdir.mkdir(parents=True, exist_ok=True)
                if "combine_atlas_parquets" in " ".join(command):
                    (outdir / "atlas_edges.parquet").write_text("", encoding="utf-8")
                    return
                if "cross_company_functors" in " ".join(command):
                    (outdir / "cross_company_functors_summary.md").write_text("# Partial similarity\n", encoding="utf-8")
                    (outdir / "cross_company_functors_manifest.json").write_text(
                        '{"overlap_years":[2023],"shared_edge_basis_size":12}',
                        encoding="utf-8",
                    )

            partial_preview_state: dict[str, object] = {}
            with mock.patch.object(module, "_run_command", side_effect=fake_run_command):
                runner._refresh_partial_similarity_preview(
                    record_a=record_a,
                    record_b=record_b,
                    workspace_root=root,
                    python_executable="python3",
                    mode_profile={"year_start": 2023, "year_end": 2025},
                    analysis_dir=runner.outdir / "adobe_vs_nike_functors",
                    stage_state={"functor_analysis": {"status": "pending"}},
                    partial_preview_state=partial_preview_state,
                )

            self.assertEqual(partial_preview_state["status"], "ready")
            self.assertIn("Initial similarity read is ready", str(partial_preview_state["note"]))
            self.assertEqual(partial_preview_state["overlap_years"], [2023])
            self.assertEqual(partial_preview_state["shared_edge_basis_size"], 12)
            self.assertTrue(str(partial_preview_state["summary_path"]).endswith("partial/cross_company_functors_summary.md"))
            self.assertEqual(len(commands), 3)

    def test_load_company_registry_rehomes_absolute_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = root / "configs"
            configs.mkdir(parents=True, exist_ok=True)
            registry_path = configs / "company_batch_registry.csv"
            with registry_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "brand",
                        "company_aliases",
                        "edgar_ticker",
                        "index_url",
                        "existing_combined_dir",
                        "outdir",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "brand": "Adobe",
                        "company_aliases": "adobe,adbe",
                        "edgar_ticker": "",
                        "index_url": "https://example.com/adobe",
                        "existing_combined_dir": "/old/machine/brand_democritus_block_denoise/outputs/adobe/runs_adobe_financial_filings/atlas_adobe_financial_combined",
                        "outdir": "/old/machine/brand_democritus_block_denoise/outputs/adobe",
                    }
                )

            records = module._load_company_registry(root)
            adobe = records["adobe"]

        self.assertEqual(adobe.outdir, (root / "outputs" / "adobe").resolve())
        self.assertEqual(
            adobe.existing_combined_dir,
            (root / "outputs" / "adobe" / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined").resolve(),
        )
        self.assertEqual(adobe.ticker, "adbe")

    def test_ensure_company_analysis_prefers_edgar_ticker_over_index_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "outputs" / "adobe"
            record = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe", "adbe"),
                ticker="adbe",
                index_url="https://stocklight.example/adobe",
                outdir=outdir,
                existing_combined_dir=None,
            )

            def fake_run(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
                manifest_path = outdir / "add_company_analysis_manifest.json"
                combined_dir = outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined"
                combined_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text('{"combined_dir": "%s"}' % str(combined_dir), encoding="utf-8")
                self.captured = (command, cwd, stream_label)

            self.captured = None
            with mock.patch.object(module, "_run_command", side_effect=fake_run):
                combined_dir, manifest_path = module._ensure_company_analysis(
                    record=record,
                    sec_user_agent="Test Agent",
                    workspace_root=root,
                    python_executable="python3",
                    execution_mode="quick",
                )

            self.assertIsNotNone(self.captured)
            command, _cwd, stream_label = self.captured
            self.assertIn("--edgar-ticker", command)
            self.assertIn("adbe", command)
            self.assertNotIn("--index-url", command)
            self.assertIn("--filings-only", command)
            self.assertIn("--skip-visuals", command)
            self.assertEqual(stream_label, "Adobe")
            self.assertEqual(combined_dir, (outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined").resolve())
            self.assertEqual(manifest_path.resolve(), (outdir / "add_company_analysis_manifest.json").resolve())

    def test_ensure_company_analysis_ignores_stale_local_manifest_when_year_window_expands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "outputs" / "adobe"
            filings_outdir = outdir / "runs_adobe_financial_filings"
            filings_outdir.mkdir(parents=True, exist_ok=True)
            stale_manifest = filings_outdir / "adobe_sec_edgar_manifest.csv"
            stale_manifest.write_text(
                "doc_id,brand,fiscal_year,filing_type,file_path\n"
                "adobe_10k_2024,Adobe,2024,10k,/tmp/2024.pdf\n"
                "adobe_10k_2025,Adobe,2025,10k,/tmp/2025.pdf\n",
                encoding="utf-8",
            )
            record = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe", "adbe"),
                ticker="adbe",
                index_url="https://stocklight.example/adobe",
                outdir=outdir,
                existing_combined_dir=None,
            )

            def fake_run(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
                manifest_path = outdir / "add_company_analysis_manifest.json"
                combined_dir = outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined"
                combined_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text('{"combined_dir": "%s"}' % str(combined_dir), encoding="utf-8")
                self.captured = (command, cwd, stream_label)

            self.captured = None
            with mock.patch.object(module, "_run_command", side_effect=fake_run):
                module._ensure_company_analysis(
                    record=record,
                    sec_user_agent="Test Agent",
                    workspace_root=root,
                    python_executable="python3",
                    execution_mode="deep",
                    year_start=2019,
                    year_end=2025,
                )

            command, _cwd, _stream_label = self.captured
            self.assertIn("--edgar-ticker", command)
            self.assertIn("--filings-only", command)
            self.assertNotIn("--manifest", command)

    def test_preflight_company_similarity_backend_requires_key_when_no_cached_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe", "adbe"),
                ticker="adbe",
                index_url="https://stocklight.example/adobe",
                outdir=root / "outputs" / "adobe",
                existing_combined_dir=None,
            )

            with mock.patch.dict(module.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    module._preflight_company_similarity_backend((record,))

    def test_preflight_company_similarity_backend_allows_cached_outputs_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            combined_dir = root / "outputs" / "adobe" / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            record = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe", "adbe"),
                ticker="adbe",
                index_url="https://stocklight.example/adobe",
                outdir=root / "outputs" / "adobe",
                existing_combined_dir=None,
            )

            with mock.patch.dict(module.os.environ, {}, clear=True):
                module._preflight_company_similarity_backend((record,))

    def test_preflight_company_similarity_backend_rejects_insufficient_cached_year_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "outputs" / "adobe"
            combined_dir = outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            temporal_blocks_dir = outdir / "temporal_blocks"
            temporal_blocks_dir.mkdir(parents=True, exist_ok=True)
            (temporal_blocks_dir / "temporal_block_summary.json").write_text(
                '{"year_min": 2023, "year_max": 2024}',
                encoding="utf-8",
            )
            (outdir / "add_company_analysis_manifest.json").write_text(
                json.dumps({"temporal_summary_path": str(temporal_blocks_dir / "temporal_block_summary.json")}),
                encoding="utf-8",
            )
            record = module._CompanyRecord(
                brand="Adobe",
                slug="adobe",
                aliases=("adobe", "adbe"),
                ticker="adbe",
                index_url="https://stocklight.example/adobe",
                outdir=outdir,
                existing_combined_dir=None,
            )

            with mock.patch.dict(module.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "requested years"):
                    module._preflight_company_similarity_backend((record,), year_start=2020, year_end=2025)

    def test_run_command_streams_stdout_and_raises_with_tail(self) -> None:
        class FakePopen:
            def __init__(self, *args, **kwargs) -> None:
                self.stdout = io.StringIO("line one\nline two\n")

            def wait(self) -> int:
                return 1

        with mock.patch.object(module.subprocess, "Popen", return_value=FakePopen()):
            with mock.patch("builtins.print") as print_mock:
                with self.assertRaisesRegex(RuntimeError, "line one\nline two"):
                    module._run_command(["python3", "-m", "demo"], cwd=Path("/tmp"))

        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("[company_similarity] command: python3 -m demo", printed)
        self.assertIn("line one", printed)
        self.assertIn("line two", printed)

    def test_ensure_company_analyses_runs_company_builds_in_parallel(self) -> None:
        record_a = module._CompanyRecord(
            brand="Adobe",
            slug="adobe",
            aliases=("adobe", "adbe"),
            outdir=Path("/tmp/adobe"),
        )
        record_b = module._CompanyRecord(
            brand="Nike",
            slug="nike",
            aliases=("nike", "nke"),
            outdir=Path("/tmp/nike"),
        )
        gate = threading.Event()
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_ensure(**kwargs):
            nonlocal active, max_active
            record = kwargs["record"]
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                if max_active >= 2:
                    gate.set()
            self.assertTrue(gate.wait(timeout=1.0))
            with state_lock:
                active -= 1
            return Path(f"/tmp/{record.slug}_combined"), None

        messages: list[str] = []
        with mock.patch.object(module, "_ensure_company_analysis", side_effect=fake_ensure):
            results = module._ensure_company_analyses(
                records=(record_a, record_b),
                sec_user_agent="Test Agent",
                workspace_root=Path("/tmp"),
                python_executable="python3",
                execution_mode="quick",
                log=messages.append,
            )

        self.assertEqual(max_active, 2)
        self.assertEqual(results["adobe"][0], Path("/tmp/adobe_combined"))
        self.assertEqual(results["nike"][0], Path("/tmp/nike_combined"))
        joined = "\n".join(messages)
        self.assertIn("ensuring company analysis for Adobe", joined)
        self.assertIn("ensuring company analysis for Nike", joined)
        self.assertIn("company analysis ready for Adobe", joined)
        self.assertIn("company analysis ready for Nike", joined)

    def test_quick_run_skips_visualization_subprocess(self) -> None:
        runner = module.CompanySimilarityAgenticRunner(
            "Compare Adobe and Nike",
            Path("/tmp/company_similarity_quick"),
            execution_mode="quick",
        )
        plan = module.CompanySimilarityQueryPlan(
            query="Compare Adobe and Nike",
            company_a="Adobe",
            company_b="Nike",
            company_a_slug="adobe",
            company_b_slug="nike",
        )
        analysis_dir = runner.outdir / "adobe_vs_nike_functors"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "cross_company_functors_summary.md").write_text("summary", encoding="utf-8")
        (analysis_dir / "cross_company_functors_manifest.json").write_text("{}", encoding="utf-8")
        (analysis_dir / "cross_company_year_metrics.csv").write_text("cosine_similarity,js_divergence\n0.5,0.1\n", encoding="utf-8")
        (analysis_dir / "naturality_defects.csv").write_text("naturality_defect_rel\n0.2\n", encoding="utf-8")

        commands: list[list[str]] = []

        def fake_run_command(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
            del cwd, stream_label
            commands.append(command)

        with mock.patch.object(module, "interpret_company_similarity_query", return_value=plan):
            with mock.patch.object(module, "_resolve_brand_workspace_root", return_value=Path("/tmp/brand_root")):
                with mock.patch.object(module, "repo_root", return_value=Path("/tmp/repo_root/subdir")):
                    with mock.patch.object(module, "_select_python_for_brand_pipeline", return_value="python3"):
                        with mock.patch.object(module, "_find_company_record") as find_mock:
                            with mock.patch.object(module, "_preflight_company_similarity_backend"):
                                with mock.patch.object(module, "_ensure_company_analyses", return_value={
                                    "adobe": (Path("/tmp/adobe_combined"), None),
                                    "nike": (Path("/tmp/nike_combined"), None),
                                }):
                                    with mock.patch.object(module, "_run_command", side_effect=fake_run_command):
                                        with mock.patch.object(module, "_write_html_report", return_value=runner.outdir / "company_similarity.html"):
                                            find_mock.side_effect = [
                                                module._CompanyRecord("Adobe", "adobe", ("adobe",), outdir=Path("/tmp/adobe")),
                                                module._CompanyRecord("Nike", "nike", ("nike",), outdir=Path("/tmp/nike")),
                                            ]
                                            result = runner.run()

        self.assertIsNotNone(result.artifact_path)
        self.assertEqual(len(commands), 1)
        self.assertIn("brand_democritus_block_denoise.cross_company_functors", " ".join(commands[0]))

    def test_run_heartbeat_refreshes_active_stage_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "company_similarity"
            runner = module.CompanySimilarityAgenticRunner(
                "How similar is Nike to Walmart?",
                outdir,
                execution_mode="quick",
            )
            runner._telemetry_heartbeat_interval_seconds = 0.01
            plan = module.CompanySimilarityQueryPlan(
                query="How similar is Nike to Walmart?",
                company_a="Nike",
                company_b="Walmart",
                company_a_slug="nike",
                company_b_slug="walmart",
            )
            nike = module._CompanyRecord("Nike", "nike", ("nike",), outdir=Path(tmpdir) / "nike")
            walmart = module._CompanyRecord("Walmart", "walmart", ("walmart",), outdir=Path(tmpdir) / "walmart")
            write_payloads: list[dict[str, object]] = []
            original_write = runner._write_telemetry

            def wrapped_write(**kwargs):
                payload = original_write(**kwargs)
                write_payloads.append(payload)
                return payload

            def fake_ensure_company_analyses(**kwargs):
                on_record_complete = kwargs["on_record_complete"]
                time.sleep(0.04)
                on_record_complete(nike, Path(tmpdir) / "nike_combined")
                time.sleep(0.05)
                on_record_complete(walmart, Path(tmpdir) / "walmart_combined")
                return {
                    "nike": (Path(tmpdir) / "nike_combined", None),
                    "walmart": (Path(tmpdir) / "walmart_combined", None),
                }

            def fake_run_command(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
                del cwd, stream_label
                if "cross_company_functors" in " ".join(command):
                    analysis_dir = outdir / "nike_vs_walmart_functors"
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    (analysis_dir / "cross_company_functors_summary.md").write_text("summary", encoding="utf-8")
                    (analysis_dir / "cross_company_functors_manifest.json").write_text("{}", encoding="utf-8")
                    (analysis_dir / "cross_company_year_metrics.csv").write_text(
                        "cosine_similarity,js_divergence\n0.5,0.1\n",
                        encoding="utf-8",
                    )
                    (analysis_dir / "naturality_defects.csv").write_text(
                        "naturality_defect_rel\n0.2\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(module, "interpret_company_similarity_query", return_value=plan):
                with mock.patch.object(module, "_resolve_brand_workspace_root", return_value=Path(tmpdir) / "brand_root"):
                    with mock.patch.object(module, "repo_root", return_value=Path(tmpdir) / "repo_root" / "subdir"):
                        with mock.patch.object(module, "_select_python_for_brand_pipeline", return_value="python3"):
                            with mock.patch.object(module, "_find_company_record", side_effect=[nike, walmart]):
                                with mock.patch.object(module, "_preflight_company_similarity_backend"):
                                    with mock.patch.object(module, "_ensure_company_analyses", side_effect=fake_ensure_company_analyses):
                                        with mock.patch.object(module, "_run_command", side_effect=fake_run_command):
                                            with mock.patch.object(runner, "_write_telemetry", side_effect=wrapped_write):
                                                with mock.patch.object(
                                                    module,
                                                    "_write_html_report",
                                                    return_value=outdir / "company_similarity_dashboard.html",
                                                ):
                                                    result = runner.run()

            self.assertIsNotNone(result.artifact_path)
            running_payloads = [payload for payload in write_payloads if payload.get("status") == "running"]
            self.assertGreaterEqual(len(running_payloads), 3)
            self.assertTrue(
                any(float(payload.get("elapsed_seconds") or 0.0) > 0.0 for payload in running_payloads),
                "expected a heartbeat telemetry refresh with nonzero elapsed time",
            )
            self.assertTrue(
                any(
                    any(
                        stage.get("status") == "active" and float(stage.get("duration_seconds") or 0.0) > 0.0
                        for stage in payload.get("stages", [])
                        if isinstance(stage, dict)
                    )
                    for payload in running_payloads
                ),
                "expected a heartbeat telemetry refresh with a ticking active stage duration",
            )

    def test_interactive_run_returns_checkpoint_before_final_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "company_similarity"
            runner = module.CompanySimilarityAgenticRunner(
                "Compare Adobe and Nike",
                outdir,
                execution_mode="interactive",
                year_start=2012,
                year_end=2015,
            )
            plan = module.CompanySimilarityQueryPlan(
                query="Compare Adobe and Nike",
                company_a="Adobe",
                company_b="Nike",
                company_a_slug="adobe",
                company_b_slug="nike",
            )
            adobe = module._CompanyRecord("Adobe", "adobe", ("adobe",), outdir=Path(tmpdir) / "adobe")
            nike = module._CompanyRecord("Nike", "nike", ("nike",), outdir=Path(tmpdir) / "nike")

            with mock.patch.object(module, "interpret_company_similarity_query", return_value=plan):
                with mock.patch.object(module, "_resolve_brand_workspace_root", return_value=Path(tmpdir) / "brand_root"):
                    with mock.patch.object(module, "repo_root", return_value=Path(tmpdir) / "repo_root" / "subdir"):
                        with mock.patch.object(module, "_select_python_for_brand_pipeline", return_value="python3"):
                            with mock.patch.object(module, "_find_company_record", side_effect=[adobe, nike]):
                                with mock.patch.object(module, "_preflight_company_similarity_backend"):
                                    with mock.patch.object(
                                        module,
                                        "_ensure_company_analyses",
                                        return_value={
                                            "adobe": (Path(tmpdir) / "adobe_combined", None),
                                            "nike": (Path(tmpdir) / "nike_combined", None),
                                        },
                                    ):
                                        with mock.patch.object(
                                            runner,
                                            "_refresh_partial_similarity_preview",
                                            side_effect=lambda **kwargs: kwargs["partial_preview_state"].update(
                                                {
                                                    "status": "ready",
                                                    "note": "Initial similarity read is ready.",
                                                    "summary_path": str(outdir / "partial" / "cross_company_functors_summary.md"),
                                                    "manifest_path": str(outdir / "partial" / "cross_company_functors_manifest.json"),
                                                    "overlap_years": [2013, 2014],
                                                    "shared_edge_basis_size": 9,
                                                }
                                            ),
                                        ):
                                            with mock.patch.object(
                                                module,
                                                "_build_company_similarity_checkpoint",
                                                return_value=(
                                                    outdir / "interactive_checkpoint" / "company_similarity_checkpoint.json",
                                                    outdir / "interactive_checkpoint" / "company_similarity_checkpoint.html",
                                                ),
                                            ):
                                                result = runner.run()

        self.assertEqual(result.artifact_path, outdir / "interactive_checkpoint" / "company_similarity_checkpoint.html")
        self.assertEqual(result.checkpoint_manifest_path, outdir / "interactive_checkpoint" / "company_similarity_checkpoint.json")
        self.assertEqual(result.checkpoint_dashboard_path, outdir / "interactive_checkpoint" / "company_similarity_checkpoint.html")


if __name__ == "__main__":
    unittest.main()
