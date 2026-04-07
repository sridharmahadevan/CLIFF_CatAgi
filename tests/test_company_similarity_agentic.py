"""Tests for company similarity portability helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from functorflow_v3 import company_similarity_agentic as module
except ModuleNotFoundError:
    from ..functorflow_v3 import company_similarity_agentic as module


class CompanySimilarityAgenticTests(unittest.TestCase):
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

            def fake_run(command: list[str], *, cwd: Path) -> None:
                manifest_path = outdir / "add_company_analysis_manifest.json"
                combined_dir = outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined"
                combined_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text('{"combined_dir": "%s"}' % str(combined_dir), encoding="utf-8")
                self.captured = (command, cwd)

            self.captured = None
            with mock.patch.object(module, "_run_command", side_effect=fake_run):
                combined_dir, manifest_path = module._ensure_company_analysis(
                    record=record,
                    sec_user_agent="Test Agent",
                    workspace_root=root,
                    python_executable="python3",
                )

            self.assertIsNotNone(self.captured)
            command, _cwd = self.captured
            self.assertIn("--edgar-ticker", command)
            self.assertIn("adbe", command)
            self.assertNotIn("--index-url", command)
            self.assertEqual(combined_dir, (outdir / "runs_adobe_financial_filings" / "atlas_adobe_financial_combined").resolve())
            self.assertEqual(manifest_path.resolve(), (outdir / "add_company_analysis_manifest.json").resolve())


if __name__ == "__main__":
    unittest.main()
