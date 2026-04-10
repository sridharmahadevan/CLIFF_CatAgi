"""Tests for plain-language cross-company similarity reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class CrossCompanyFunctorsSummaryTests(unittest.TestCase):
    def test_write_summary_uses_plain_language_sections(self) -> None:
        try:
            import pandas as pd
            from brand_democritus_block_denoise import cross_company_functors as module
        except ModuleNotFoundError as exc:
            self.skipTest(f"cross-company summary test requires analytics dependencies: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            basis_df = pd.DataFrame({"edge_key": ["a\tREL\tb", "c\tREL\td"]})
            overlap_df = pd.DataFrame(
                [
                    {
                        "fiscal_year": 2024,
                        "cosine_similarity": 0.0003,
                        "js_divergence": 0.9990,
                        "shared_mass_l1": 0.0002,
                    }
                ]
            )
            naturality_df = pd.DataFrame(
                [
                    {
                        "fiscal_year": 2024,
                        "year_to": 2025,
                        "naturality_defect_rel": 0.0898,
                        "naturality_defect_abs": 0.0090,
                    }
                ]
            )

            module.write_summary(
                outdir,
                company_a_name="adobe",
                company_b_name="nike",
                years_a=[2024, 2025],
                years_b=[2024, 2025],
                basis_df=basis_df,
                overlap_df=overlap_df,
                trans_a=(None, None, 0.0),
                trans_b=(None, None, 0.0),
                cross_ab=(None, None, 0.000021),
                naturality_df=naturality_df,
                latent_rank=4,
            )

            rendered = (outdir / "cross_company_functors_summary.md").read_text(encoding="utf-8")

        self.assertIn("# Company Similarity Summary", rendered)
        self.assertIn("## Headline Read", rendered)
        self.assertIn("Average alignment score: 0.0003", rendered)
        self.assertIn("Average distribution gap: 0.9990", rendered)
        self.assertIn("Average consistency gap: 0.0898", rendered)
        self.assertIn("relative_gap", rendered)
        self.assertIn("structurally quite different", rendered)
        self.assertNotIn("Cross-Company Functor Prototype", rendered)
        self.assertNotIn("Approximate Naturality", rendered)


if __name__ == "__main__":
    unittest.main()
