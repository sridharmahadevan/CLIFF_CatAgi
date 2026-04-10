"""Tests for batch-oriented Democritus agentic execution."""

from __future__ import annotations

import json
import time
import tempfile
import unittest
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

try:
    from functorflow_v3 import (
        DemocritusBatchAgenticRunner,
        DemocritusBatchConfig,
        DemocritusBatchRecord,
        DemocritusAgentRecord,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import (
        DemocritusBatchAgenticRunner,
        DemocritusBatchConfig,
        DemocritusBatchRecord,
        DemocritusAgentRecord,
    )

try:
    from functorflow_v3 import democritus_batch_agentic as module
except ModuleNotFoundError:
    from ..functorflow_v3 import democritus_batch_agentic as module

try:
    from functorflow_v3.democritus_batch_agentic import DemocritusBatchDocument
except ModuleNotFoundError:
    from ..functorflow_v3.democritus_batch_agentic import DemocritusBatchDocument
from types import SimpleNamespace


class DemocritusBatchAgenticTests(unittest.TestCase):
    def test_batch_config_defaults_to_eight_workers(self) -> None:
        config = DemocritusBatchConfig(pdf_dir=Path("/tmp/pdfs"), outdir=Path("/tmp/out"))

        self.assertEqual(config.max_workers, 8)

    def test_batch_runner_discovers_pdfs_and_creates_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\n")
            (pdf_dir / "beta.pdf").write_bytes(b"%PDF-1.4\n")

            runner = DemocritusBatchAgenticRunner(
                DemocritusBatchConfig(
                    pdf_dir=pdf_dir,
                    outdir=outdir,
                    max_workers=2,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )

            self.assertEqual(len(runner.documents), 2)
            self.assertTrue(all(document.run_name.startswith(("0001_", "0002_")) for document in runner.documents))

    def test_batch_runner_passes_topic_budget_into_document_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\n")

            runner = DemocritusBatchAgenticRunner(
                DemocritusBatchConfig(
                    pdf_dir=pdf_dir,
                    outdir=outdir,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    depth_limit=2,
                    max_total_topics=40,
                    dry_run=True,
                )
            )

            document_runner = runner.documents[0].runner
            self.assertEqual(document_runner.config.depth_limit, 2)
            self.assertEqual(document_runner.config.max_total_topics, 40)
            self.assertEqual(document_runner.config.statements_per_question, 2)
            self.assertEqual(document_runner.config.statement_batch_size, 16)
            self.assertEqual(document_runner.config.statement_max_tokens, 192)

    def test_batch_dry_run_produces_records_for_all_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\n")
            (pdf_dir / "beta.pdf").write_bytes(b"%PDF-1.4\n")

            runner = DemocritusBatchAgenticRunner(
                DemocritusBatchConfig(
                    pdf_dir=pdf_dir,
                    outdir=outdir,
                    max_workers=2,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            records = runner.run()

            self.assertTrue(records)
            run_names = {record.run_name for record in records}
            self.assertEqual(len(run_names), 2)
            self.assertTrue(all(record.agent_record.status == "planned" for record in records))
            self.assertTrue((outdir / "telemetry.json").exists())
            self.assertTrue((outdir / "dashboard.html").exists())
            self.assertTrue((outdir / "democritus_gui.html").exists())

    def test_per_agent_concurrency_limits_are_enforced(self) -> None:
        active_counts = {"root_topic_discovery_agent": 0}
        max_seen = {"root_topic_discovery_agent": 0}

        class FakeRunner:
            def __init__(self, name: str) -> None:
                self.name = name

            def _execute_agent(self, agent_name: str, frontier_index: int):
                active_counts[agent_name] = active_counts.get(agent_name, 0) + 1
                max_seen[agent_name] = max(max_seen.get(agent_name, 0), active_counts[agent_name])
                time.sleep(0.02)
                active_counts[agent_name] -= 1
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=0.0,
                    ended_at=0.0,
                    outputs=(),
                    log_path=None,
                    notes="",
                )

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                documents = []
                for index in range(3):
                    run_name = f"run_{index}"
                    documents.append(
                        DemocritusBatchDocument(
                            index=index,
                            pdf_path=Path(f"/tmp/{run_name}.pdf"),
                            run_name=run_name,
                            outdir=Path(f"/tmp/{run_name}"),
                            runner=FakeRunner(run_name),
                            plan=((SimpleNamespace(name="root_topic_discovery_agent"),),),
                        )
                    )
                return tuple(documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=3,
                    agent_concurrency_limits=(("root_topic_discovery_agent", 1),),
                    dry_run=False,
                )
            )
            records = runner.run()

            self.assertEqual(len(records), 3)
            self.assertEqual(max_seen["root_topic_discovery_agent"], 1)

    def test_telemetry_snapshot_includes_timing_forecast_and_slowest_stages(self) -> None:
        class FakeRunner:
            def _execute_agent(self, agent_name: str, frontier_index: int):
                raise NotImplementedError

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                return (
                    DemocritusBatchDocument(
                        index=1,
                        pdf_path=Path("/tmp/alpha.pdf"),
                        run_name="run_alpha",
                        outdir=Path("/tmp/run_alpha"),
                        runner=FakeRunner(),
                        plan=(
                            (SimpleNamespace(name="root_topic_discovery_agent"),),
                            (SimpleNamespace(name="causal_question_agent"),),
                            (SimpleNamespace(name="causal_statement_agent"),),
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=3,
                    dry_run=False,
                )
            )
            completed_records = [
                DemocritusBatchRecord(
                    run_name="run_alpha",
                    pdf_path="/tmp/alpha.pdf",
                    agent_record=DemocritusAgentRecord(
                        agent_name="root_topic_discovery_agent",
                        frontier_index=0,
                        status="ok",
                        started_at=100.0,
                        ended_at=125.0,
                    ),
                ),
                DemocritusBatchRecord(
                    run_name="run_alpha",
                    pdf_path="/tmp/alpha.pdf",
                    agent_record=DemocritusAgentRecord(
                        agent_name="causal_question_agent",
                        frontier_index=1,
                        status="ok",
                        started_at=130.0,
                        ended_at=190.0,
                    ),
                ),
            ]
            snapshot = runner._build_telemetry_snapshot(
                batch_started_at=100.0,
                pending_frontiers={"run_alpha": 2},
                active_agent_counts={},
                active_futures={},
                ready_queue=deque(),
                completed_records=completed_records,
                status="running",
            )

            self.assertIn("elapsed_human", snapshot)
            self.assertIn("started_at_local", snapshot)
            self.assertIn("timing", snapshot)
            self.assertIn("slowest_stages", snapshot)
            self.assertTrue(snapshot["timing"]["eta_ready"])
            self.assertGreater(snapshot["timing"]["remaining_work_seconds_estimate"], 0.0)
            self.assertGreater(snapshot["timing"]["eta_seconds"], 0.0)
            self.assertEqual(snapshot["slowest_stages"][0]["agent_name"], "causal_question_agent")

    def test_batch_runner_builds_csql_bundle_from_triple_outputs(self) -> None:
        class FakeRunner:
            def __init__(self, outdir: Path, name: str) -> None:
                self.outdir = outdir
                self.name = name

            def _execute_agent(self, agent_name: str, frontier_index: int):
                triples_path = self.outdir / "relational_triples.jsonl"
                triples_path.parent.mkdir(parents=True, exist_ok=True)
                triples_path.write_text(
                    "\n".join(
                        [
                            '{"topic":"climate","path":["climate"],"question":"q","statement":"s","subj":"carbon","rel":"increases","obj":"warming","domain":"climate"}',
                            '{"topic":"climate","path":["climate"],"question":"q2","statement":"s2","subj":"warming","rel":"drives","obj":"migration","domain":"climate"}',
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_path),),
                    log_path=None,
                    notes="",
                )

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                documents = []
                for index in range(2):
                    run_name = f"run_{index}"
                    outdir = Path(self.config.outdir) / run_name
                    documents.append(
                        DemocritusBatchDocument(
                            index=index,
                            pdf_path=Path(f"/tmp/{run_name}.pdf"),
                            run_name=run_name,
                            outdir=outdir,
                            runner=FakeRunner(outdir, run_name),
                            plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                        )
                    )
                return tuple(documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=2,
                    dry_run=False,
                )
            )
            result = runner.run_with_artifacts()

            self.assertEqual(len(result.records), 2)
            self.assertIsNotNone(result.csql_bundle)
            self.assertIsNotNone(result.corpus_synthesis)
            sqlite_path = result.csql_bundle.sqlite_path
            summary_path = result.csql_bundle.summary_path
            self.assertTrue(sqlite_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(result.corpus_synthesis.summary_path.exists())
            self.assertTrue(result.corpus_synthesis.dashboard_path.exists())
            connection = sqlite3.connect(str(sqlite_path))
            try:
                claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
                edge_support = connection.execute(
                    "SELECT document_support FROM aggregated_edges WHERE subj = 'carbon' AND rel = 'increases' AND obj = 'warming'"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(claim_count, 4)
            self.assertEqual(edge_support, 2)
            synthesis_payload = json.loads(result.corpus_synthesis.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(synthesis_payload["diagnostic_supported"], [])
            synthesis_html = result.corpus_synthesis.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Democritus Corpus Synthesis", synthesis_html)
            self.assertIn("carbon", synthesis_html)

    def test_corpus_synthesis_surfaces_normalized_diagnostic_claims(self) -> None:
        class FakeRunner:
            def __init__(self, outdir: Path, name: str) -> None:
                self.outdir = outdir
                self.name = name

            def _execute_agent(self, agent_name: str, frontier_index: int):
                triples_path = self.outdir / "relational_triples.jsonl"
                triples_path.parent.mkdir(parents=True, exist_ok=True)
                payload_by_name = {
                    "run_0": (
                        '{"topic":"weight loss","path":["weight loss"],"question":"q","statement":"The use of GLP-1 receptor agonists increases weight loss.","subj":"the use of glp-1 receptor agonists","rel":"increases","obj":"weight loss","domain":"GLP-1 receptor agonists effects"}'
                    ),
                    "run_1": (
                        '{"topic":"weight loss","path":["weight loss"],"question":"q","statement":"Treatment with glucagon-like peptide-1 receptor agonists increases weight loss.","subj":"treatment with glucagon-like peptide-1 receptor agonists","rel":"increases","obj":"weight loss","domain":"Glucagon-Like Peptide-1 receptor agonists"}'
                    ),
                }
                triples_path.write_text(payload_by_name[self.name] + "\n", encoding="utf-8")
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_path),),
                    log_path=None,
                    notes="",
                )

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                documents = []
                for index in range(2):
                    run_name = f"run_{index}"
                    outdir = Path(self.config.outdir) / run_name
                    documents.append(
                        DemocritusBatchDocument(
                            index=index,
                            pdf_path=Path(f"/tmp/{run_name}.pdf"),
                            run_name=run_name,
                            outdir=outdir,
                            runner=FakeRunner(outdir, run_name),
                            plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                        )
                    )
                return tuple(documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=2,
                    dry_run=False,
                )
            )
            result = runner.run_with_artifacts()

            self.assertIsNotNone(result.corpus_synthesis)
            synthesis_payload = json.loads(result.corpus_synthesis.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(synthesis_payload["strongly_supported"], [])
            self.assertEqual(len(synthesis_payload["diagnostic_supported"]), 1)
            diagnostic_claim = synthesis_payload["diagnostic_supported"][0]
            self.assertEqual(diagnostic_claim["document_support"], 2)
            self.assertEqual(diagnostic_claim["exact_document_support_max"], 1)
            self.assertEqual(diagnostic_claim["canonical_subj"], "glp1 receptor agonist")
            synthesis_html = result.corpus_synthesis.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Normalized Diagnostic Support", synthesis_html)
            self.assertIn("glp1 receptor agonist", synthesis_html)

    def test_single_document_corpus_synthesis_coalesces_near_duplicate_claim_cards(self) -> None:
        class FakeRunner:
            def __init__(self, outdir: Path) -> None:
                self.outdir = outdir

            def _execute_agent(self, agent_name: str, frontier_index: int):
                triples_path = self.outdir / "relational_triples.jsonl"
                triples_path.parent.mkdir(parents=True, exist_ok=True)
                triples_path.write_text(
                    "\n".join(
                        [
                            '{"topic":"antarctic warming","path":["antarctic warming"],"question":"q","statement":"Rising ocean temperatures lead to krill moving to deeper waters, which reduces the food availability for Antarctic fur seals.","subj":"rising ocean temperatures","rel":"leads_to","obj":"krill moving to deeper waters, which reduces the food availability for antarctic fur seals","domain":"No relevant content on sea ice loss"}',
                            '{"topic":"antarctic warming","path":["antarctic warming"],"question":"q","statement":"Rising ocean temperatures cause krill to move to deeper waters, which reduces food availability for Antarctic fur seals.","subj":"rising ocean temperatures","rel":"causes","obj":"krill to move to deeper waters, which reduces food availability for antarctic fur seals","domain":"Climate change effects on species"}',
                            '{"topic":"antarctic warming","path":["antarctic warming"],"question":"q","statement":"Rising ocean temperatures cause krill to move to deeper waters, which reduces the availability of food for Antarctic fur seals.","subj":"rising ocean temperatures","rel":"causes","obj":"krill to move to deeper waters, which reduces the availability of food for antarctic fur seals","domain":"No relevant content on Antarctic species"}',
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_path),),
                    log_path=None,
                    notes="",
                )

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                run_name = "run_0"
                outdir = Path(self.config.outdir) / run_name
                return (
                    DemocritusBatchDocument(
                        index=0,
                        pdf_path=Path("/tmp/run_0.pdf"),
                        run_name=run_name,
                        outdir=outdir,
                        runner=FakeRunner(outdir),
                        plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=1,
                    dry_run=False,
                )
            )

            result = runner.run_with_artifacts()

            self.assertIsNotNone(result.corpus_synthesis)
            synthesis_payload = json.loads(result.corpus_synthesis.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(synthesis_payload["weakly_supported"]), 1)
            weak_claim = synthesis_payload["weakly_supported"][0]
            self.assertEqual(weak_claim["document_support"], 1)
            self.assertEqual(weak_claim["claim_count"], 3)
            self.assertEqual(weak_claim["domain"], "Climate change effects on species")
            self.assertIn("Rising ocean temperatures", weak_claim["statement"])
            synthesis_html = result.corpus_synthesis.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Climate change effects on species", synthesis_html)

    def test_corpus_synthesis_surfaces_causal_equivalence_classes_separately_from_disagreements(self) -> None:
        class FakeRunner:
            def __init__(self, outdir: Path, name: str) -> None:
                self.outdir = outdir
                self.name = name

            def _execute_agent(self, agent_name: str, frontier_index: int):
                triples_path = self.outdir / "relational_triples.jsonl"
                triples_path.parent.mkdir(parents=True, exist_ok=True)
                payload_by_name = {
                    "run_0": (
                        '{"topic":"red wine polyphenols","path":["red wine polyphenols"],"question":"q","statement":"Moderate red wine consumption increases the intake of polyphenolic compounds such as resveratrol.","subj":"moderate red wine consumption","rel":"increases","obj":"the intake of polyphenolic compounds such as resveratrol","domain":"Moderate red wine consumption"}'
                    ),
                    "run_1": (
                        '{"topic":"red wine polyphenols","path":["red wine polyphenols"],"question":"q","statement":"Moderate red wine consumption leads to the intake of polyphenolic compounds such as resveratrol.","subj":"moderate red wine consumption","rel":"leads_to","obj":"the intake of polyphenolic compounds such as resveratrol","domain":"Moderate red wine consumption"}'
                    ),
                }
                triples_path.write_text(payload_by_name[self.name] + "\n", encoding="utf-8")
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_path),),
                    log_path=None,
                    notes="",
                )

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                documents = []
                for index in range(2):
                    run_name = f"run_{index}"
                    outdir = Path(self.config.outdir) / run_name
                    documents.append(
                        DemocritusBatchDocument(
                            index=index,
                            pdf_path=Path(f"/tmp/{run_name}.pdf"),
                            run_name=run_name,
                            outdir=outdir,
                            runner=FakeRunner(outdir, run_name),
                            plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                        )
                    )
                return tuple(documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=Path(tmpdir) / "runs",
                    max_workers=2,
                    dry_run=False,
                )
            )
            result = runner.run_with_artifacts()

            self.assertIsNotNone(result.corpus_synthesis)
            synthesis_payload = json.loads(result.corpus_synthesis.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(synthesis_payload["weakly_supported"]), 1)
            self.assertEqual(len(synthesis_payload["equivalence_classes"]), 1)
            self.assertEqual(synthesis_payload["disagreements"], [])
            equivalence_class = synthesis_payload["equivalence_classes"][0]
            self.assertEqual(equivalence_class["subj"], "moderate red wine consumption")
            self.assertEqual(len(equivalence_class["variants"]), 2)
            synthesis_html = result.corpus_synthesis.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Causal Equivalence Classes", synthesis_html)
            self.assertIn("Backbone claim:", synthesis_html)
            self.assertIn("same-direction variant", synthesis_html)
            self.assertIn("Disagreements", synthesis_html)

    def test_incremental_corpus_synthesis_refreshes_only_when_new_triple_documents_arrive(self) -> None:
        class FakeRunner:
            def _execute_agent(self, agent_name: str, frontier_index: int):
                raise NotImplementedError

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                return ()

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "runs"
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=outdir,
                    max_workers=1,
                    dry_run=False,
                )
            )
            run_alpha = outdir / "run_alpha"
            run_beta = outdir / "run_beta"
            run_alpha.mkdir(parents=True, exist_ok=True)
            run_beta.mkdir(parents=True, exist_ok=True)
            triples_alpha = run_alpha / "relational_triples.jsonl"
            triples_beta = run_beta / "relational_triples.jsonl"
            triples_alpha.write_text('{"subj":"a","rel":"supports","obj":"b","domain":"x"}\n', encoding="utf-8")
            triples_beta.write_text('{"subj":"c","rel":"supports","obj":"d","domain":"y"}\n', encoding="utf-8")
            record_alpha = DemocritusBatchRecord(
                run_name="run_alpha",
                pdf_path="/tmp/run_alpha.pdf",
                agent_record=DemocritusAgentRecord(
                    agent_name="triple_extraction_agent",
                    frontier_index=0,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_alpha),),
                    log_path=None,
                    notes="",
                ),
            )
            record_beta = DemocritusBatchRecord(
                run_name="run_beta",
                pdf_path="/tmp/run_beta.pdf",
                agent_record=DemocritusAgentRecord(
                    agent_name="triple_extraction_agent",
                    frontier_index=0,
                    status="ok",
                    started_at=0.0,
                    ended_at=1.0,
                    outputs=(str(triples_beta),),
                    log_path=None,
                    notes="",
                ),
            )
            fake_bundle = module.BatchCSQLBundleResult(
                sqlite_path=outdir / "csql" / "democritus_csql.sqlite",
                summary_path=outdir / "csql" / "democritus_csql_summary.json",
            )
            fake_synthesis = module.DemocritusCorpusSynthesisResult(
                summary_path=outdir / "corpus_synthesis" / "democritus_corpus_synthesis.json",
                dashboard_path=outdir / "corpus_synthesis" / "democritus_corpus_synthesis.html",
            )
            fake_synthesis.summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_synthesis.summary_path.write_text('{"n_documents": 1}', encoding="utf-8")
            fake_synthesis.dashboard_path.write_text("<html></html>", encoding="utf-8")

            with mock.patch.object(runner, "_build_csql_bundle", return_value=fake_bundle) as bundle_mock:
                with mock.patch.object(runner, "_build_corpus_synthesis", return_value=fake_synthesis) as synthesis_mock:
                    runner._maybe_refresh_incremental_corpus_synthesis([record_alpha])
                    runner._maybe_refresh_incremental_corpus_synthesis([record_alpha])
                    runner._maybe_refresh_incremental_corpus_synthesis([record_alpha, record_beta])

            self.assertEqual(bundle_mock.call_count, 2)
            self.assertEqual(synthesis_mock.call_count, 2)

    def test_incremental_corpus_synthesis_can_refresh_from_partial_triples_before_completion(self) -> None:
        class FakeRunner:
            def _execute_agent(self, agent_name: str, frontier_index: int):
                raise NotImplementedError

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                run_outdir = Path(self.config.outdir) / "run_alpha"
                return (
                    DemocritusBatchDocument(
                        index=1,
                        pdf_path=Path("/tmp/run_alpha.pdf"),
                        run_name="run_alpha",
                        outdir=run_outdir,
                        runner=FakeRunner(),
                        plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "runs"
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=outdir,
                    max_workers=1,
                    dry_run=False,
                )
            )
            run_alpha = runner.documents[0].outdir
            run_alpha.mkdir(parents=True, exist_ok=True)
            triples_alpha = run_alpha / "relational_triples.jsonl"
            triples_alpha.write_text(
                "\n".join(
                    [
                        '{"subj":"a","rel":"supports","obj":"b","domain":"x"}',
                        '{"subj":"b","rel":"supports","obj":"c","domain":"x"}',
                        '{"subj":"c","rel":"supports","obj":"d","domain":"x"}',
                        '{"subj":"d","rel":"supports","obj":"e","domain":"x"}',
                        '{"subj":"e","rel":"supports","obj":"f","domain":"x"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_bundle = module.BatchCSQLBundleResult(
                sqlite_path=outdir / "csql" / "democritus_csql.sqlite",
                summary_path=outdir / "csql" / "democritus_csql_summary.json",
            )
            fake_synthesis = module.DemocritusCorpusSynthesisResult(
                summary_path=outdir / "corpus_synthesis" / "democritus_corpus_synthesis.json",
                dashboard_path=outdir / "corpus_synthesis" / "democritus_corpus_synthesis.html",
            )
            fake_synthesis.summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_synthesis.summary_path.write_text('{"n_documents": 1}', encoding="utf-8")
            fake_synthesis.dashboard_path.write_text("<html></html>", encoding="utf-8")

            with mock.patch.object(runner, "_build_csql_bundle", return_value=fake_bundle) as bundle_mock:
                with mock.patch.object(runner, "_build_corpus_synthesis", return_value=fake_synthesis) as synthesis_mock:
                    result = runner._maybe_refresh_incremental_corpus_synthesis([])

            self.assertIs(result, fake_synthesis)
            self.assertEqual(bundle_mock.call_count, 1)
            self.assertEqual(synthesis_mock.call_count, 1)

    def test_incremental_document_admission_starts_work_before_stream_closes(self) -> None:
        event_log: list[tuple[str, str, float]] = []

        class FakeRunner:
            def __init__(self, pdf_path: Path) -> None:
                self.pdf_path = pdf_path
                self.plan = ((SimpleNamespace(name="root_topic_discovery_agent"),),)

            def _execute_agent(self, agent_name: str, frontier_index: int):
                started_at = time.time()
                event_log.append(("start", self.pdf_path.name, started_at))
                time.sleep(0.05)
                ended_at = time.time()
                return DemocritusAgentRecord(
                    agent_name=agent_name,
                    frontier_index=frontier_index,
                    status="ok",
                    started_at=started_at,
                    ended_at=ended_at,
                    outputs=(),
                    log_path=None,
                    notes="",
                )

        class StreamingBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                return ()

            def _build_document_runner(self, *, pdf_path: Path, run_name: str, run_outdir: Path):
                del run_name, run_outdir
                return FakeRunner(pdf_path)

            def _plan_for_runner(self, runner):
                return runner.plan

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir()
            alpha_pdf = pdf_dir / "alpha.pdf"
            beta_pdf = pdf_dir / "beta.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\n")
            beta_pdf.write_bytes(b"%PDF-1.4\n")

            runner = StreamingBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=pdf_dir,
                    outdir=outdir,
                    max_workers=2,
                    discover_existing_documents=False,
                    allow_incremental_admission=True,
                    include_phase2=False,
                    dry_run=False,
                )
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(runner.run_with_artifacts)
                runner.register_document(alpha_pdf)
                time.sleep(0.02)
                second_registered_at = time.time()
                runner.register_document(beta_pdf)
                runner.close_document_stream()
                result = future.result()

            self.assertEqual(len(result.records), 2)
            first_start = min(timestamp for event, _, timestamp in event_log if event == "start")
            self.assertLess(first_start, second_registered_at)
            telemetry = (outdir / "telemetry.json").read_text(encoding="utf-8")
            self.assertIn('"allow_incremental_admission": true', telemetry)

    def test_batch_runner_bootstraps_dashboard_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir()

            runner = DemocritusBatchAgenticRunner(
                DemocritusBatchConfig(
                    pdf_dir=pdf_dir,
                    outdir=outdir,
                    discover_existing_documents=False,
                    allow_incremental_admission=True,
                    dry_run=False,
                )
            )

            self.assertTrue(runner.telemetry_path.exists())
            self.assertTrue(runner.dashboard_path.exists())
            telemetry = runner.telemetry_path.read_text(encoding="utf-8")
            self.assertIn('"status": "waiting_for_documents"', telemetry)
            self.assertTrue(runner.gui_path.exists())

    def test_democritus_gui_surfaces_triples_and_lcm_focus(self) -> None:
        pdf_fixture = Path("/tmp/glp1_study.pdf")

        class FakeRunner:
            def _execute_agent(self, agent_name: str, frontier_index: int):
                raise NotImplementedError

        class FakeBatchRunner(DemocritusBatchAgenticRunner):
            def _discover_documents(self):
                run_outdir = Path(self.config.outdir) / "run_alpha"
                return (
                    DemocritusBatchDocument(
                        index=1,
                        pdf_path=pdf_fixture,
                        run_name="run_alpha",
                        outdir=run_outdir,
                        runner=FakeRunner(),
                        plan=((SimpleNamespace(name="triple_extraction_agent"),),),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "runs"
            pdf_fixture.write_bytes(b"%PDF-1.4\n")
            runner = FakeBatchRunner(
                DemocritusBatchConfig(
                    pdf_dir=Path(tmpdir),
                    outdir=outdir,
                    request_query="Give me 5 studies of the health benefits of resveratrol in red wine.",
                    max_workers=1,
                    dry_run=False,
                )
            )
            run_dir = runner.documents[0].outdir
            (run_dir / "input.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
            (run_dir / "configs").mkdir(parents=True, exist_ok=True)
            (run_dir / "configs" / "root_topics.txt").write_text("GLP-1 agonists\nweight loss\n", encoding="utf-8")
            (run_dir / "causal_questions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"topic": "GLP-1 agonists", "path": ["GLP-1 agonists"], "questions": ["How do GLP-1 agonists influence appetite?"]}),
                        json.dumps({"topic": "weight loss", "path": ["weight loss"], "questions": ["What causes improved satiety during GLP-1 therapy?"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "causal_statements.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "topic": "GLP-1 agonists",
                                "path": ["GLP-1 agonists"],
                                "question": "How do GLP-1 agonists influence appetite?",
                                "statements": ["GLP-1 agonists increase satiety and reduce caloric intake."],
                            }
                        ),
                        json.dumps(
                            {
                                "topic": "weight loss",
                                "path": ["weight loss"],
                                "question": "What causes improved satiety during GLP-1 therapy?",
                                "statements": ["Improved satiety influences weight loss adherence."],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "relational_triples.jsonl").write_text(
                '{"subj":"GLP-1 agonists","rel":"increase","obj":"satiety","statement":"GLP-1 agonists increase satiety and reduce caloric intake.","domain":"weight loss"}\n',
                encoding="utf-8",
            )
            (run_dir / "sweep").mkdir(parents=True, exist_ok=True)
            (run_dir / "sweep" / "scores.csv").write_text(
                "file,focus,radius,n_nodes,n_edges,score_raw,coupling,score,lcm_json\n"
                "glp1.json,appetite suppression pathway,2,8,5,0.1,0.25,0.42,glp1.json\n",
                encoding="utf-8",
            )
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)
            (run_dir / "reports" / "run_alpha_executive_summary.md").write_text(
                "\n".join(
                    [
                        "# Executive Credibility Summary — run_alpha",
                        "",
                        "This one-page summary ranks causal claims by credibility.",
                        "",
                        "### Models used (top-K)",
                        "",
                        "- appetite suppression pathway",
                        "",
                        "## Tier 1 Claims",
                        "",
                        "**1. (1.00) GLP-1 agonists —increase→ satiety**",
                        "",
                        "> GLP-1 agonists increase satiety and reduce caloric intake.",
                        "",
                        "## Tier 2 Claims",
                        "",
                        "_No items in this tier._",
                        "",
                        "## Notes and caveats",
                        "",
                        "- Placeholder note.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "reports" / "run_alpha_credibility_report.md").write_text(
                "# Credibility Report\n\n- Appetite suppression is supported by multiple statements.\n",
                encoding="utf-8",
            )
            (outdir / "corpus_synthesis").mkdir(parents=True, exist_ok=True)
            (outdir / "corpus_synthesis" / "democritus_corpus_synthesis.json").write_text(
                json.dumps(
                    {
                        "query": "Give me 5 studies of the health benefits of resveratrol in red wine.",
                        "n_documents": 1,
                        "strongly_supported": [
                            {
                                "subj": "GLP-1 agonists",
                                "rel": "increase",
                                "obj": "satiety",
                                "domain": "weight loss",
                                "statement": "GLP-1 agonists increase satiety and reduce caloric intake.",
                                "document_support": 1,
                                "supporting_runs": ["run_alpha"],
                                "truth_value": "provisional_support",
                            }
                        ],
                        "weakly_supported": [],
                        "disagreements": [],
                    }
                ),
                encoding="utf-8",
            )
            (outdir / "corpus_synthesis" / "democritus_corpus_synthesis.html").write_text(
                "<html><body>Rolling synthesis</body></html>",
                encoding="utf-8",
            )
            (run_dir / "viz").mkdir(parents=True, exist_ok=True)
            (run_dir / "viz" / "relational_manifold_2d.png").write_bytes(b"fakepng")

            runner._write_telemetry(
                batch_started_at=100.0,
                pending_frontiers={"run_alpha": 1},
                active_agent_counts={},
                active_futures={},
                ready_queue=deque(),
                completed_records=[],
                status="running",
            )

            gui_html = runner.gui_path.read_text(encoding="utf-8")
            self.assertIn("BAFFLE Democritus GUI", gui_html)
            self.assertIn("Give me 5 studies of the health benefits of resveratrol in red wine.", gui_html)
            self.assertNotIn("Recovered Causal Structure Across the Current Batch", gui_html)
            self.assertIn("Batch performance dashboard", gui_html)
            self.assertIn("Telemetry JSON", gui_html)
            self.assertIn("Performance Snapshot", gui_html)
            self.assertIn("Slowest Stages", gui_html)
            self.assertIn("Initial Corpus Answer", gui_html)
            self.assertIn("Current Cross-Study Claims", gui_html)
            self.assertIn("Open rolling synthesis", gui_html)
            self.assertIn("near-final", gui_html)
            self.assertIn("2 questions", gui_html)
            self.assertIn("2 statements", gui_html)
            self.assertIn("1 triples", gui_html)
            self.assertIn("appetite suppression pathway", gui_html)
            self.assertIn("GLP-1 agonists", gui_html)
            self.assertIn("This one-page summary ranks causal claims by credibility.", gui_html)
            self.assertIn("Open PDF", gui_html)
            self.assertNotIn("file://", gui_html)
            self.assertIn(f'{run_dir.name}/input.pdf', gui_html)
            self.assertIn("run_alpha_executive_summary.html", gui_html)
            self.assertIn("relational_manifold_viewer.html", gui_html)
            summary_viewer = run_dir / "reports" / "run_alpha_executive_summary.html"
            credibility_viewer = run_dir / "reports" / "run_alpha_credibility_report.html"
            manifold_viewer = run_dir / "viz" / "relational_manifold_viewer.html"
            self.assertTrue(summary_viewer.exists())
            self.assertTrue(credibility_viewer.exists())
            self.assertTrue(manifold_viewer.exists())
            summary_html = summary_viewer.read_text(encoding="utf-8")
            self.assertIn("BAFFLE Democritus Reader", summary_html)
            self.assertIn("font-size: 22px", summary_html)
            self.assertIn("Tier 1 Claims", summary_html)
            self.assertIn("GLP-1 agonists", summary_html)
            self.assertNotIn("Models used (top-K)", summary_html)
            self.assertNotIn("Notes and caveats", summary_html)
            self.assertNotIn("Executive Credibility Summary — run_alpha", summary_html)
            self.assertNotIn("glp1 study executive summary", summary_html.lower())
            credibility_html = credibility_viewer.read_text(encoding="utf-8")
            self.assertIn("Credibility Report", credibility_html)
            self.assertIn("Appetite suppression is supported by multiple statements.", credibility_html)
            self.assertNotIn('<article class="article">\n      <h1>', credibility_html)
            self.assertIn("Topic Labels", manifold_viewer.read_text(encoding="utf-8"))
            self.assertIn("GLP-1 agonists", manifold_viewer.read_text(encoding="utf-8"))
            pdf_fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
