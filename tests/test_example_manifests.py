"""Validation tests for checked-in example manifests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class ExampleManifestTests(unittest.TestCase):
    def test_culinary_stop_manifest_has_required_fields(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "culinary_tour"
            / "culinary_stop_manifest.jsonl"
        )

        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 8)

        records = [json.loads(line) for line in lines]
        for record in records:
            self.assertIn("name", record)
            self.assertIn("destination", record)
            self.assertIn("district", record)
            self.assertIn("specialty", record)
            self.assertIn("estimated_cost", record)
            self.assertIn("tags", record)
            self.assertIn("url", record)
            self.assertTrue(str(record["name"]).strip())
            self.assertTrue(str(record["destination"]).strip())
            self.assertTrue(str(record["district"]).strip())
            self.assertTrue(str(record["url"]).startswith("https://"))
            self.assertIsInstance(record["tags"], list)

    def test_lovesac_review_manifest_has_required_fields(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "product_feedback"
            / "lovesac_review_manifest.jsonl"
        )

        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)

        records = [json.loads(line) for line in lines]
        for record in records:
            self.assertIn("title", record)
            self.assertIn("url", record)
            self.assertIn("product", record)
            self.assertIn("brand", record)
            self.assertIn("summary", record)
            self.assertTrue(str(record["title"]).strip())
            self.assertTrue(str(record["url"]).startswith("https://"))
            self.assertEqual(record["brand"], "Lovesac")
            self.assertEqual(record["product"], "Lovesac Sactional")

    def test_nike_pegasus_review_manifest_has_required_fields(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "product_feedback"
            / "nike_pegasus_41_review_manifest.jsonl"
        )

        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)

        records = [json.loads(line) for line in lines]
        for record in records:
            self.assertIn("title", record)
            self.assertIn("url", record)
            self.assertIn("product", record)
            self.assertIn("brand", record)
            self.assertIn("summary", record)
            self.assertTrue(str(record["title"]).strip())
            self.assertTrue(str(record["url"]).startswith("https://"))
            self.assertEqual(record["brand"], "Nike")
            self.assertEqual(record["product"], "Nike Pegasus 41")


if __name__ == "__main__":
    unittest.main()
