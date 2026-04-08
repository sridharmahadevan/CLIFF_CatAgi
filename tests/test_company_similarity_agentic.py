"""Tests for company similarity portability helpers."""

from __future__ import annotations

import csv
import io
import threading
import tempfile
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
        self.assertLessEqual(int(profile["year_end"]) - int(profile["year_start"]), 5)
        self.assertEqual(profile["jobs"], 2)

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
            self.assertEqual(stream_label, "Adobe")
            self.assertEqual(combined_dir, (outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined").resolve())
            self.assertEqual(manifest_path.resolve(), (outdir / "add_company_analysis_manifest.json").resolve())

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


if __name__ == "__main__":
    unittest.main()
