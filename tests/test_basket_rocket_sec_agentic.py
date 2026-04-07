"""Tests for SEC-backed BASKET/ROCKET ingress scaffolding."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import basket_rocket_sec_agentic as module
    from functorflow_v3 import BasketRocketSECAgenticConfig, BasketRocketSECAgenticRunner
    from functorflow_v3.basket_rocket_sec_agentic import BasketRocketBatchAgenticRunner, MaterializedSECFiling
    from functorflow_v3.democritus_query_agentic import DemocritusQueryRunResult, DiscoveredDocument, QueryPlan
except ModuleNotFoundError:
    from ..functorflow_v3 import basket_rocket_sec_agentic as module
    from ..functorflow_v3 import BasketRocketSECAgenticConfig, BasketRocketSECAgenticRunner
    from ..functorflow_v3.basket_rocket_sec_agentic import BasketRocketBatchAgenticRunner, MaterializedSECFiling
    from ..functorflow_v3.democritus_query_agentic import DemocritusQueryRunResult, DiscoveredDocument, QueryPlan


class BasketRocketSECAgenticTests(unittest.TestCase):
    def test_resolve_query_for_main_uses_dashboard_launcher_when_query_missing(self) -> None:
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
                return "Find me 10 recent 10-K filings for Adobe"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
                query = module._resolve_query_for_main(SimpleNamespace(query="", outdir=tmpdir))

        self.assertEqual(query, "Find me 10 recent 10-K filings for Adobe")
        self.assertIsNotNone(captured_config)
        self.assertEqual(
            captured_config.artifact_path,
            Path(tmpdir).resolve() / "workflow_batches" / "corpus_synthesis" / "basket_rocket_corpus_synthesis.html",
        )

    def test_batch_runner_uses_legacy_financial_panel_when_available(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        required_paths = (
            repo_root / "BASKET" / "src" / "rocket.py",
            repo_root / "BASKET" / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "workflow_extractions.jsonl",
            repo_root / "BASKET" / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "macro_skills.csv",
            repo_root
            / "brand_democritus_block_denoise"
            / "outputs"
            / "company_panel_26"
            / "rocket_company_outcomes.csv",
        )
        if not all(path.exists() for path in required_paths):
            self.skipTest("legacy BASKET financial panel artifacts are not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            filing_path = root / "ibm_10k.html"
            text_path = root / "ibm_10k.txt"
            filing_text = "\n".join(
                (
                    "ITEM 1. BUSINESS",
                    "Digital platform strategy and software growth.",
                    "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
                    "Revenue, margin, operations, and cloud demand improved.",
                )
            )
            filing_path.write_text(filing_text, encoding="utf-8")
            text_path.write_text(filing_text, encoding="utf-8")
            filing = MaterializedSECFiling(
                title="IBM 10-K 2024-02-26",
                filing_path=str(filing_path),
                text_path=str(text_path),
                source_url="https://example.com/ibm10k",
                retrieval_backend="sec",
                company="International Business Machines Corp",
                ticker="IBM",
                cik="0000051143",
                accession_number="0000051143-24-000001",
                form_type="10-K",
                filing_date="2024-02-26",
                filing_year="2024",
                anchor_year="2024",
                semantic_role="annual_anchor",
                workset_name="ibm_2024_10k",
            )
            manifest = root / "materialized_filing_manifest.json"
            company_context = root / "company_context.json"
            manifest.write_text("[]", encoding="utf-8")
            company_context.write_text("[]", encoding="utf-8")

            BasketRocketBatchAgenticRunner(
                filings=(filing,),
                outdir=root / "workflow_batches",
                filing_manifest_path=manifest,
                company_context_path=company_context,
                rocket_reward_source="legacy",
            ).run()

            rankings_payload = json.loads(
                (root / "workflow_batches" / "ibm_2024_10k" / "rocket_rankings.json").read_text(encoding="utf-8")
            )
            top = rankings_payload["rankings"][0]
            self.assertEqual(rankings_payload["reward_backend"], "legacy_financial_real")
            self.assertEqual(top["reward_backend"], "legacy_financial_real")
            self.assertIn("legacy_company_id", top)
            self.assertGreaterEqual(float(top["selected_score"]), float(top["base_score"]))
            self.assertEqual(
                rankings_payload["financial_targets"],
                [
                    "revenue_yoy",
                    "operating_margin",
                    "free_cash_flow_margin",
                    "return_on_assets",
                    "debt_to_assets",
                ],
            )

    def test_batch_runner_falls_back_to_heuristic_when_legacy_company_lookup_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            filing_path = root / "adobe_10k.html"
            text_path = root / "adobe_10k.txt"
            filing_text = "\n".join(
                (
                    "ITEM 1. BUSINESS",
                    "Creative software, subscriptions, cloud platform, and customer demand.",
                    "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
                    "Revenue, margin, and operations improved.",
                )
            )
            filing_path.write_text(filing_text, encoding="utf-8")
            text_path.write_text(filing_text, encoding="utf-8")
            filing = MaterializedSECFiling(
                title="Adobe 10-K 2024-01-19",
                filing_path=str(filing_path),
                text_path=str(text_path),
                source_url="https://example.com/adobe10k",
                retrieval_backend="sec",
                company="Adobe Inc.",
                ticker="ADBE",
                cik="0000796343",
                accession_number="0000796343-24-000001",
                form_type="10-K",
                filing_date="2024-01-19",
                filing_year="2024",
                anchor_year="2024",
                semantic_role="annual_anchor",
                workset_name="adobe_2024_10k",
            )
            manifest = root / "materialized_filing_manifest.json"
            company_context = root / "company_context.json"
            manifest.write_text("[]", encoding="utf-8")
            company_context.write_text("[]", encoding="utf-8")

            runner = BasketRocketBatchAgenticRunner(
                filings=(filing,),
                outdir=root / "workflow_batches",
                filing_manifest_path=manifest,
                company_context_path=company_context,
                rocket_reward_source="legacy",
            )
            runner._legacy_company_lookup = {}

            with patch.object(runner, "_get_legacy_rocket_module", return_value=SimpleNamespace()):
                with patch.object(
                    runner,
                    "_get_legacy_reward_context",
                    return_value=SimpleNamespace(financial_targets=("revenue_yoy",)),
                ):
                    workset = runner.worksets[0]
                    candidate = {
                        "candidate_id": "adobe_candidate_01",
                        "filing_title": filing.title,
                        "workflow_stages": ["digitize", "optimize", "market", "realize_revenue"],
                        "event_item_codes": [],
                        "evidence_sections": ["Cloud platform", "Revenue and operations"],
                        "score_basis": {"section_count": 2},
                    }
                    reranked = runner._rerank_candidate_workflow(candidate, workset)

            self.assertEqual(reranked["reward_backend"], "heuristic")
            self.assertNotIn("legacy_company_id", reranked)

    def test_sec_runner_groups_discovered_filings_into_company_year_form_worksets(self) -> None:
        class FakeRunner(BasketRocketSECAgenticRunner):
            def _run_sec_discovery_agent(self) -> DemocritusQueryRunResult:
                plan = QueryPlan(
                    query="find me IBM and Coca-Cola 10-K filings",
                    normalized_query="find me ibm and coca-cola 10-k filings",
                    keyword_tokens=("ibm", "coca", "cola", "10", "k"),
                    target_documents=3,
                    requested_forms=("10-K",),
                )
                selected = (
                    DiscoveredDocument(
                        title="IBM 10-K 2024-02-20",
                        score=9.0,
                        retrieval_backend="sec",
                        download_url="https://www.sec.gov/ibm10k2024.htm",
                        url="https://www.sec.gov/ibm10k2024.htm",
                        document_format="html",
                        identifier="0000051143-24-000001",
                        metadata={
                            "company": "International Business Machines Corp",
                            "ticker": "IBM",
                            "cik": "0000051143",
                            "form": "10-K",
                            "filing_date": "2024-02-20",
                        },
                    ),
                    DiscoveredDocument(
                        title="IBM 8-K 2024-05-01",
                        score=8.0,
                        retrieval_backend="sec",
                        download_url="https://www.sec.gov/ibm8k2024.htm",
                        url="https://www.sec.gov/ibm8k2024.htm",
                        document_format="html",
                        identifier="0000051143-24-000010",
                        metadata={
                            "company": "International Business Machines Corp",
                            "ticker": "IBM",
                            "cik": "0000051143",
                            "form": "8-K",
                            "filing_date": "2024-05-01",
                        },
                    ),
                    DiscoveredDocument(
                        title="Coca-Cola 10-K 2023-02-10",
                        score=7.0,
                        retrieval_backend="sec",
                        download_url="https://www.sec.gov/ko10k2023.htm",
                        url="https://www.sec.gov/ko10k2023.htm",
                        document_format="html",
                        identifier="0000021344-23-000001",
                        metadata={
                            "company": "Coca Cola Co",
                            "ticker": "KO",
                            "cik": "0000021344",
                            "form": "10-K",
                            "filing_date": "2023-02-10",
                        },
                    ),
                )
                summary_path = self.discovery_outdir / "query_run_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps({"retrieval_backend": "sec"}, indent=2), encoding="utf-8")
                return DemocritusQueryRunResult(
                    query_plan=plan,
                    selected_documents=selected,
                    acquired_documents=(),
                    batch_records=(),
                    pdf_dir=self.discovery_outdir / "unused_pdfs",
                    batch_outdir=self.discovery_outdir / "unused_batch",
                    summary_path=summary_path,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "basket_run"
            runner = FakeRunner(
                BasketRocketSECAgenticConfig(
                    query="find me IBM and Coca-Cola 10-K filings",
                    outdir=outdir,
                    target_filings=3,
                    rocket_reward_source="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_filings), 3)
            self.assertEqual(len(result.materialized_filings), 3)
            self.assertEqual(len(result.batch_records), 21)
            self.assertTrue((outdir / "materialized_filing_manifest.json").exists())
            self.assertTrue((outdir / "company_context" / "company_year_form_index.json").exists())
            self.assertTrue((outdir / "workflow_batches" / "workset_index.json").exists())
            self.assertTrue((outdir / "workflow_batches" / "basket_rocket_gui.html").exists())
            self.assertIn(
                "BASKET/ROCKET 10-K GUI",
                (outdir / "workflow_batches" / "basket_rocket_gui.html").read_text(encoding="utf-8"),
            )

            worksets = json.loads((outdir / "workflow_batches" / "workset_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(worksets), 3)
            self.assertEqual({item["form_type"] for item in worksets}, {"10-K", "8-K"})
            roles = {(item["form_type"], item["semantic_role"]) for item in worksets}
            self.assertIn(("10-K", "annual_anchor"), roles)
            self.assertIn(("8-K", "event_patch"), roles)

            agent_names = {record.agent_name for record in result.batch_records}
            self.assertIn("filing_collection_agent", agent_names)
            self.assertIn("workflow_reporting_agent", agent_names)

    def test_sec_runner_materializes_html_filings_and_writes_scaffold_outputs(self) -> None:
        class FakeRunner(BasketRocketSECAgenticRunner):
            def _run_sec_discovery_agent(self) -> DemocritusQueryRunResult:
                plan = QueryPlan(
                    query="find me IBM 10-K filings",
                    normalized_query="find me ibm 10-k filings",
                    keyword_tokens=("ibm", "10", "k"),
                    target_documents=2,
                    requested_forms=("10-K",),
                )
                selected = (
                    DiscoveredDocument(
                        title="IBM 10-K 2024-02-20",
                        score=9.0,
                        retrieval_backend="sec",
                        download_url="https://www.sec.gov/ibm10k2024.htm",
                        url="https://www.sec.gov/ibm10k2024.htm",
                        document_format="html",
                        identifier="0000051143-24-000001",
                        metadata={
                            "company": "International Business Machines Corp",
                            "ticker": "IBM",
                            "cik": "0000051143",
                            "form": "10-K",
                            "filing_date": "2024-02-20",
                        },
                    ),
                    DiscoveredDocument(
                        title="IBM 8-K 2024-03-01",
                        score=8.5,
                        retrieval_backend="sec",
                        download_url="https://www.sec.gov/ibm8k2024a.htm",
                        url="https://www.sec.gov/ibm8k2024a.htm",
                        document_format="html",
                        identifier="0000051143-24-000002",
                        metadata={
                            "company": "International Business Machines Corp",
                            "ticker": "IBM",
                            "cik": "0000051143",
                            "form": "8-K",
                            "filing_date": "2024-03-01",
                        },
                    ),
                )
                summary_path = self.discovery_outdir / "query_run_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps({"retrieval_backend": "sec"}, indent=2), encoding="utf-8")
                return DemocritusQueryRunResult(
                    query_plan=plan,
                    selected_documents=selected,
                    acquired_documents=(),
                    batch_records=(),
                    pdf_dir=self.discovery_outdir / "unused_pdfs",
                    batch_outdir=self.discovery_outdir / "unused_batch",
                    summary_path=summary_path,
                )

            def _download_filing(self, url: str, *, referer: str | None = None) -> bytes:
                del url, referer
                return (
                    b"<html><body><h1>ITEM 1.01 ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT</h1><p>Cloud and consulting.</p>"
                    b"<h2>ITEM 2.02 RESULTS OF OPERATIONS AND FINANCIAL CONDITION</h2><p>Execution risk and AI transition.</p>"
                    b"<h2>ITEM 7. MANAGEMENT DISCUSSION</h2><p>Year-over-year analysis.</p>"
                    b"</body></html>"
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "basket_run"
            runner = FakeRunner(
                BasketRocketSECAgenticConfig(
                    query="find me IBM 10-K filings",
                    outdir=outdir,
                    target_filings=2,
                    rocket_reward_source="heuristic",
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_filings), 2)
            self.assertEqual(len(result.materialized_filings), 2)
            self.assertGreater(len(result.batch_records), 0)

            first_filing = Path(result.materialized_filings[0].filing_path)
            first_text = Path(result.materialized_filings[0].text_path or "")
            self.assertTrue(first_filing.exists())
            self.assertTrue(first_text.exists())
            self.assertIn("ITEM 1.01", first_text.read_text(encoding="utf-8"))

            workset_index = json.loads((outdir / "workflow_batches" / "workset_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(workset_index), 2)
            annual_rows = [row for row in workset_index if row["semantic_role"] == "annual_anchor"]
            event_rows = [row for row in workset_index if row["semantic_role"] == "event_patch"]
            self.assertEqual(len(annual_rows), 1)
            self.assertEqual(len(event_rows), 1)
            self.assertEqual(event_rows[0]["filings"][0]["event_item_codes"], ["1.01", "2.02"])

            annual_candidates = json.loads(
                (
                    outdir
                    / "workflow_batches"
                    / annual_rows[0]["workset_name"]
                    / "candidate_workflows.json"
                ).read_text(encoding="utf-8")
            )["candidates"]
            event_candidates = json.loads(
                (
                    outdir
                    / "workflow_batches"
                    / event_rows[0]["workset_name"]
                    / "candidate_workflows.json"
                ).read_text(encoding="utf-8")
            )["candidates"]
            annual_stages = annual_candidates[0]["workflow_stages"]
            event_stages = event_candidates[0]["workflow_stages"]
            self.assertGreaterEqual(len(annual_candidates), 2)
            self.assertGreaterEqual(len(event_candidates), 1)
            self.assertIn(
                "grounded_evidence_order",
                {candidate["candidate_source"] for candidate in annual_candidates},
            )
            self.assertIn("optimize", annual_stages)
            self.assertIn("realize_revenue", annual_stages)
            self.assertNotIn("annual_baseline_read", annual_stages)
            self.assertIn("invest", event_stages)
            self.assertIn("price", event_stages)
            self.assertIn("realize_revenue", event_stages)
            self.assertNotIn("event_triage", event_stages)
            self.assertTrue(annual_candidates[0]["evidence_spans"])
            self.assertEqual(annual_candidates[0]["candidate_source"], "grounded_filing_backbone")

            annual_rankings = json.loads(
                (
                    outdir
                    / "workflow_batches"
                    / annual_rows[0]["workset_name"]
                    / "rocket_rankings.json"
                ).read_text(encoding="utf-8")
            )["rankings"]
            self.assertGreater(len(annual_rankings), 0)
            self.assertIn("base_workflow_stages", annual_rankings[0])
            self.assertIn("selected_score", annual_rankings[0])
            self.assertIn("score_gain", annual_rankings[0])
            self.assertIn("candidate_summaries", annual_rankings[0])
            self.assertIn("financial_targets", annual_rankings[0])
            self.assertIn("evidence_spans", annual_rankings[0])
            self.assertIn("candidate_source", annual_rankings[0])
            self.assertGreaterEqual(float(annual_rankings[0]["selected_score"]), float(annual_rankings[0]["base_score"]))

            visualization_index_path = outdir / "workflow_batches" / "visualizations" / "index.html"
            visualization_summary_path = outdir / "workflow_batches" / "visualizations" / "index_summary.json"
            company_summary_path = outdir / "workflow_batches" / "visualizations" / "company_index_summary.json"
            live_gui_path = outdir / "workflow_batches" / "basket_rocket_gui.html"
            visualization_rows = json.loads(visualization_summary_path.read_text(encoding="utf-8"))
            company_rows = json.loads(company_summary_path.read_text(encoding="utf-8"))
            workset_page_path = outdir / "workflow_batches" / "visualizations" / visualization_rows[0]["html"]
            aggregate_page_path = outdir / "workflow_batches" / "visualizations" / visualization_rows[0]["aggregate_html"]
            year_page_path = outdir / "workflow_batches" / "visualizations" / visualization_rows[0]["year_html"]
            company_page_path = outdir / "workflow_batches" / "visualizations" / company_rows[0]["html"]
            company_aggregate_path = outdir / "workflow_batches" / "visualizations" / company_rows[0]["aggregate_html"]
            report_path = outdir / "workflow_batches" / annual_rows[0]["workset_name"] / "workflow_report.md"
            rankings_path = outdir / "workflow_batches" / event_rows[0]["workset_name"] / "rocket_rankings.json"
            self.assertTrue(report_path.exists())
            self.assertTrue(rankings_path.exists())
            self.assertTrue(live_gui_path.exists())
            self.assertTrue(visualization_index_path.exists())
            self.assertTrue(visualization_summary_path.exists())
            self.assertTrue(company_summary_path.exists())
            self.assertTrue(workset_page_path.exists())
            self.assertTrue(aggregate_page_path.exists())
            self.assertTrue(year_page_path.exists())
            self.assertTrue(company_page_path.exists())
            self.assertTrue(company_aggregate_path.exists())
            self.assertIn("BASKET/ROCKET Scaffold Report", report_path.read_text(encoding="utf-8"))
            self.assertIn("BAFFLE BASKET/ROCKET Visualizers", visualization_index_path.read_text(encoding="utf-8"))
            self.assertIn("Recovered Workflow Backbone", live_gui_path.read_text(encoding="utf-8"))
            self.assertIn("Open final BASKET/ROCKET visualization suite", live_gui_path.read_text(encoding="utf-8"))
            self.assertIn("Mean Gain", workset_page_path.read_text(encoding="utf-8"))
            self.assertIn("Grounded evidence spans", workset_page_path.read_text(encoding="utf-8"))
            self.assertIn("Reward Backend", workset_page_path.read_text(encoding="utf-8"))
            self.assertIn("ROCKET Reranking Visualizer", company_page_path.read_text(encoding="utf-8"))
            self.assertIn("ROCKET Aggregate Plans", company_aggregate_path.read_text(encoding="utf-8"))
            self.assertIn("BAFFLE Year Backbone", year_page_path.read_text(encoding="utf-8"))
            self.assertEqual(result.batch_live_gui_path.resolve(), live_gui_path.resolve())
            self.assertEqual(result.batch_visualization_index_path.resolve(), visualization_index_path.resolve())
            self.assertEqual(result.batch_visualization_summary_path.resolve(), visualization_summary_path.resolve())


if __name__ == "__main__":
    unittest.main()
