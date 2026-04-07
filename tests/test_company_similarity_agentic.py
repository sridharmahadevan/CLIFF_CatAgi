"""Tests for company similarity portability helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
