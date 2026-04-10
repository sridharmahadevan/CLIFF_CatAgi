"""Tests for query-driven Democritus corpus acquisition."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from functorflow_v3 import democritus_query_agentic as democritus_query_agentic_module
    from functorflow_v3 import DemocritusQueryAgenticConfig, DemocritusQueryAgenticRunner
    from functorflow_v3.democritus_agentic import DemocritusAgentRecord
    from functorflow_v3.democritus_batch_agentic import DemocritusBatchRecord, DemocritusBatchRunResult
    from functorflow_v3.democritus_query_agentic import (
        CrossrefRetrievalBackend,
        AcquiredCorpusDocument,
        DemocritusEvidenceSnapshot,
        DirectFileRetrievalBackend,
        DirectURLRetrievalBackend,
        DiscoveredDocument,
        DirectDirectoryRetrievalBackend,
        EuropePMCOARetrievalBackend,
        QueryPlan,
        SECFilingRetrievalBackend,
        ScholarlyRetrievalBackend,
        _match_score,
        _derive_retrieval_query,
        infer_requested_result_count,
        _resolve_sec_user_agent,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import democritus_query_agentic as democritus_query_agentic_module
    from ..functorflow_v3 import DemocritusQueryAgenticConfig, DemocritusQueryAgenticRunner
    from ..functorflow_v3.democritus_agentic import DemocritusAgentRecord
    from ..functorflow_v3.democritus_batch_agentic import DemocritusBatchRecord, DemocritusBatchRunResult
    from ..functorflow_v3.democritus_query_agentic import (
        CrossrefRetrievalBackend,
        AcquiredCorpusDocument,
        DemocritusEvidenceSnapshot,
        DirectFileRetrievalBackend,
        DirectURLRetrievalBackend,
        DiscoveredDocument,
        DirectDirectoryRetrievalBackend,
        EuropePMCOARetrievalBackend,
        QueryPlan,
        SECFilingRetrievalBackend,
        ScholarlyRetrievalBackend,
        _direct_document_path_candidates,
        _match_score,
        _derive_retrieval_query,
        infer_requested_result_count,
        _resolve_sec_user_agent,
    )


class DemocritusQueryAgenticTests(unittest.TestCase):
    def test_topic_checkpoint_uses_wider_document_cards(self) -> None:
        html = democritus_query_agentic_module._render_democritus_topic_checkpoint_html(
            {
                "query": "What are the health effects of red wine?",
                "stage_label": "Topic checkpoint",
                "summary_text": "Preview the topic atlas before going deeper.",
                "n_documents": 2,
                "top_topics": [{"topic": "polyphenols", "document_count": 2}],
                "documents": [
                    {
                        "run_name": "run_0",
                        "title": "Study of moderate red wine consumption",
                        "guide_summary": "A longer abstract-style snapshot that should have room to breathe.",
                        "topics": ["polyphenols", "cardiovascular outcomes"],
                    }
                ],
            }
        )

        self.assertIn("minmax(min(100%, 360px), 1fr)", html)
        self.assertIn("max-width: 68ch", html)

    def test_query_config_defaults_to_eight_workers(self) -> None:
        config = DemocritusQueryAgenticConfig(
            query="find me 10 recent studies on glp-1",
            outdir=Path("/tmp/democritus-query"),
        )

        self.assertEqual(config.max_workers, 8)

    def test_query_config_quick_mode_caps_document_count_and_disables_phase2(self) -> None:
        config = DemocritusQueryAgenticConfig(
            query="find me 10 recent studies on climate change",
            outdir=Path("/tmp/democritus-query"),
            execution_mode="quick",
            target_documents=10,
            max_docs=12,
        ).resolved()

        self.assertEqual(config.execution_mode, "quick")
        self.assertEqual(config.target_documents, 3)
        self.assertEqual(config.max_docs, 5)
        self.assertFalse(config.include_phase2)
        self.assertEqual(config.root_topic_strategy, "heuristic")
        self.assertEqual(config.depth_limit, 2)
        self.assertEqual(config.max_total_topics, 40)
        self.assertEqual(config.statements_per_question, 1)
        self.assertEqual(config.statement_batch_size, 32)
        self.assertEqual(config.statement_max_tokens, 72)
        self.assertEqual(config.intra_document_shards, 2)

    def test_query_config_quick_mode_preserves_full_document_pipeline_for_direct_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "uploaded_paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nuploaded\n")

            config = DemocritusQueryAgenticConfig(
                query="Analyze this uploaded PDF",
                outdir=Path(tmpdir) / "query_run",
                execution_mode="quick",
                input_pdf_path=pdf_path,
            ).resolved()

        self.assertEqual(config.execution_mode, "quick")
        self.assertEqual(config.root_topic_strategy, "summary_guided")
        self.assertFalse(config.include_phase2)
        self.assertEqual(config.depth_limit, 3)
        self.assertEqual(config.max_total_topics, 100)
        self.assertEqual(config.statements_per_question, 2)
        self.assertEqual(config.statement_batch_size, 16)
        self.assertEqual(config.statement_max_tokens, 192)
        self.assertEqual(config.intra_document_shards, 1)

    def test_query_config_quick_mode_preserves_full_document_pipeline_for_direct_pdf_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "papers"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\nalpha\n")

            config = DemocritusQueryAgenticConfig(
                query="Analyze these uploaded PDFs",
                outdir=Path(tmpdir) / "query_run",
                execution_mode="quick",
                input_pdf_dir=pdf_dir,
            ).resolved()

        self.assertEqual(config.execution_mode, "quick")
        self.assertEqual(config.depth_limit, 3)
        self.assertEqual(config.max_total_topics, 100)
        self.assertEqual(config.statements_per_question, 2)
        self.assertEqual(config.statement_batch_size, 16)
        self.assertEqual(config.statement_max_tokens, 192)
        self.assertEqual(config.intra_document_shards, 1)

    def test_query_config_interactive_mode_keeps_requested_corpus_size_and_pauses_before_deep_stages(self) -> None:
        config = DemocritusQueryAgenticConfig(
            query="find me 10 recent studies on minimum wage",
            outdir=Path("/tmp/democritus-query"),
            execution_mode="interactive",
            target_documents=10,
            max_docs=15,
        ).resolved()

        self.assertEqual(config.execution_mode, "interactive")
        self.assertEqual(config.target_documents, 10)
        self.assertEqual(config.max_docs, 12)
        self.assertFalse(config.include_phase2)
        self.assertEqual(config.root_topic_strategy, "summary_guided")

    def test_resolve_query_for_main_uses_cli_query_when_present(self) -> None:
        try:
            from functorflow_v3 import democritus_query_agentic as module
        except ImportError:
            from ..functorflow_v3 import democritus_query_agentic as module

        query = module._resolve_query_for_main(
            SimpleNamespace(query="  red wine benefits  ", outdir="/tmp/ignored")
        )

        self.assertEqual(query, "red wine benefits")

    def test_resolve_query_for_main_falls_back_to_dashboard_launcher(self) -> None:
        try:
            from functorflow_v3 import democritus_query_agentic as module
        except ImportError:
            from ..functorflow_v3 import democritus_query_agentic as module

        class FakeLauncher:
            instances: list["FakeLauncher"] = []

            def __init__(self, config):
                self.config = config
                type(self).instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def wait_for_submission(self) -> str:
                return "Find me 10 studies on the benefits of drinking red wine"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(module, "DashboardQueryLauncher", FakeLauncher):
                query = module._resolve_query_for_main(
                    SimpleNamespace(query="", outdir=tmpdir)
                )

        self.assertEqual(query, "Find me 10 studies on the benefits of drinking red wine")
        self.assertEqual(len(FakeLauncher.instances), 1)
        self.assertEqual(
            FakeLauncher.instances[0].config.artifact_path,
            Path(tmpdir).resolve() / "democritus_runs" / "corpus_synthesis" / "democritus_corpus_synthesis.html",
        )

    def test_query_interpretation_rewrites_open_ended_question_for_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="I'd like to know the health impacts of drinking red wine",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="manifest",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.query, "I'd like to know the health impacts of drinking red wine")
            self.assertEqual(plan.retrieval_query, "health impacts red wine")
            self.assertEqual(plan.keyword_tokens, ("health", "impacts", "red", "wine"))

    def test_query_interpretation_extracts_strict_sec_company_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the recent 10-K filings from Adobe and extract workflows",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="sec",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.requested_forms, ("10-K",))
            self.assertEqual(plan.sec_company_targets, (("adobe",),))

    def test_query_interpretation_expands_djia_company_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the recent 10-K filings from the DJIA companies",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="sec",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.requested_forms, ("10-K",))
            self.assertGreaterEqual(len(plan.sec_company_targets), 30)
            self.assertEqual(plan.sec_cohort_mode, "latest_per_company")
            self.assertEqual(plan.target_documents, len(plan.sec_company_targets))
            self.assertIn(("apple", "aapl"), plan.sec_company_targets)
            self.assertIn(("ibm", "international business machines"), plan.sec_company_targets)

    def test_infer_requested_result_count_reads_numeric_request(self) -> None:
        count = infer_requested_result_count(
            "Give me 5 studies of the health benefits of resveratrol in red wine",
            nouns=("study", "studies", "paper", "papers"),
        )

        self.assertEqual(count, 5)

    def test_infer_requested_result_count_reads_number_words(self) -> None:
        count = infer_requested_result_count(
            "Find me five recent studies on red wine polyphenols",
            nouns=("study", "studies", "paper", "papers"),
        )

        self.assertEqual(count, 5)

    def test_derive_retrieval_query_preserves_glp1_and_drops_synthesis_boilerplate(self) -> None:
        query = "Analyze 5 recent studies of the weight loss drug GLP-1 and synthesize what they jointly support"

        retrieval_query = _derive_retrieval_query(query)

        self.assertEqual(retrieval_query, "weight loss drug glp-1")

    def test_derive_retrieval_query_drops_direct_document_url_only_boilerplate(self) -> None:
        query = "Analyze the document at https://example.org/news/story-about-water"

        retrieval_query = _derive_retrieval_query(query)

        self.assertEqual(retrieval_query, "")

    def test_match_score_prefers_title_anchor_phrase_over_generic_context_overlap(self) -> None:
        plan = QueryPlan(
            query="health impacts of drinking red wine",
            normalized_query="health impacts red wine",
            keyword_tokens=("health", "impacts", "red", "wine"),
            target_documents=5,
            retrieval_query="health impacts red wine",
        )

        specific_score, specific_evidence = _match_score(
            plan,
            "Red Wine Polyphenols and Cardiovascular Outcomes",
            "Study of red wine health impacts and mechanisms.",
        )
        generic_score, generic_evidence = _match_score(
            plan,
            "Mediterranean Diet and Cardiovascular Health",
            "Background discussion mentions red wine consumption among several diet components.",
        )

        self.assertGreater(specific_score, generic_score)
        self.assertTrue(any(item.startswith("title_phrase:red wine") for item in specific_evidence))
        self.assertFalse(any(item.startswith("title:") for item in generic_evidence))

    def test_default_download_headers_use_browser_user_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine benefits",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="scholarly",
                    dry_run=True,
                )
            )

            headers = runner._download_request_headers(referer="https://example.org/paper")

            self.assertIn("Mozilla/5.0", headers["User-Agent"])
            self.assertEqual(headers["Referer"], "https://example.org/paper")

    def test_query_interpretation_extracts_direct_document_url_and_forces_single_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://example.org/news/story-about-water",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_urls, ("https://example.org/news/story-about-water",))
            self.assertEqual(plan.target_documents, 1)
            self.assertEqual(runner._backend_name(), "direct_url")

    def test_query_interpretation_extracts_local_pdf_path_and_forces_single_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "uploaded_paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nuploaded\n")
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query=f"Analyze the PDF at `{pdf_path}`",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_paths, (str(pdf_path.resolve()),))
            self.assertEqual(plan.target_documents, 1)
            self.assertEqual(runner._backend_name(), "direct_file")

    def test_query_interpretation_uses_explicit_input_pdf_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "uploaded_paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nuploaded\n")
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze this uploaded PDF",
                    outdir=Path(tmpdir) / "query_run",
                    input_pdf_path=pdf_path,
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_paths, (str(pdf_path.resolve()),))
            self.assertEqual(plan.target_documents, 1)
            self.assertEqual(runner._backend_name(), "direct_file")

    def test_direct_document_path_candidates_accept_macos_path_without_leading_slash(self) -> None:
        candidates = _direct_document_path_candidates(
            "Analyze the PDF document at Users/sridharmahadevan/Desktop/WaPo_EmperorPenguin.pdf"
        )

        self.assertEqual(candidates, ("Users/sridharmahadevan/Desktop/WaPo_EmperorPenguin.pdf",))

    def test_query_interpretation_recovers_macos_path_without_leading_slash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            real_pdf = Path(tmpdir) / "WaPo_EmperorPenguin.pdf"
            real_pdf.write_bytes(b"%PDF-1.4\nuploaded\n")
            query_path = str(real_pdf.resolve()).lstrip("/")
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query=f"Analyze the PDF document at {query_path}",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_paths, (str(real_pdf.resolve()),))
            self.assertEqual(plan.target_documents, 1)
            self.assertEqual(runner._backend_name(), "direct_file")

    def test_query_interpretation_fails_fast_for_unresolved_local_pdf_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the PDF document at Users/sridharmahadevan/Desktop/DoesNotExist.pdf",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            with self.assertRaisesRegex(ValueError, "Could not resolve the requested local PDF path"):
                runner._run_query_interpretation_agent()

    def test_query_interpretation_extracts_local_pdf_directory_and_counts_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "papers"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\nalpha\n")
            (pdf_dir / "beta.pdf").write_bytes(b"%PDF-1.4\nbeta\n")
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query=f"Analyze the PDFs in `{pdf_dir}`",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_directories, (str(pdf_dir.resolve()),))
            self.assertEqual(plan.target_documents, 2)
            self.assertEqual(runner._backend_name(), "direct_directory")

    def test_query_interpretation_uses_explicit_input_pdf_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "papers"
            pdf_dir.mkdir()
            (pdf_dir / "alpha.pdf").write_bytes(b"%PDF-1.4\nalpha\n")
            (pdf_dir / "beta.pdf").write_bytes(b"%PDF-1.4\nbeta\n")
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze this uploaded PDF folder",
                    outdir=Path(tmpdir) / "query_run",
                    input_pdf_dir=pdf_dir,
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            plan = runner._run_query_interpretation_agent()

            self.assertEqual(plan.direct_document_directories, (str(pdf_dir.resolve()),))
            self.assertEqual(plan.target_documents, 2)
            self.assertEqual(runner._backend_name(), "direct_directory")

    def test_direct_url_backend_emits_direct_document_candidates(self) -> None:
        backend = DirectURLRetrievalBackend()

        results = backend.search(
            QueryPlan(
                query="Analyze the document at https://example.org/news/story-about-water",
                normalized_query="",
                keyword_tokens=(),
                target_documents=1,
                direct_document_urls=("https://example.org/news/story-about-water",),
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "direct_url")
        self.assertEqual(results[0].url, "https://example.org/news/story-about-water")
        self.assertEqual(results[0].document_format, "unknown")

    def test_direct_file_backend_emits_local_pdf_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "uploaded_paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nuploaded\n")
            backend = DirectFileRetrievalBackend()

            results = backend.search(
                QueryPlan(
                    query=f"Analyze the PDF at {pdf_path}",
                    normalized_query="",
                    keyword_tokens=(),
                    target_documents=1,
                    direct_document_paths=(str(pdf_path.resolve()),),
                ),
                limit=1,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retrieval_backend, "direct_file")
            self.assertEqual(results[0].source_path, str(pdf_path.resolve()))
            self.assertEqual(results[0].document_format, "pdf")

    def test_direct_directory_backend_emits_local_pdf_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "papers"
            pdf_dir.mkdir()
            alpha = pdf_dir / "alpha.pdf"
            beta = pdf_dir / "nested" / "beta.pdf"
            beta.parent.mkdir()
            alpha.write_bytes(b"%PDF-1.4\nalpha\n")
            beta.write_bytes(b"%PDF-1.4\nbeta\n")
            backend = DirectDirectoryRetrievalBackend()

            results = backend.search(
                QueryPlan(
                    query=f"Analyze the PDFs in {pdf_dir}",
                    normalized_query="",
                    keyword_tokens=(),
                    target_documents=2,
                    direct_document_directories=(str(pdf_dir.resolve()),),
                ),
                limit=2,
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(all(item.retrieval_backend == "direct_directory" for item in results))
            self.assertEqual({item.source_path for item in results}, {str(alpha.resolve()), str(beta.resolve())})

    def test_runner_materializes_local_pdf_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_pdf = Path(tmpdir) / "uploaded_paper.pdf"
            source_pdf.write_bytes(b"%PDF-1.4\nuploaded\n")
            outdir = Path(tmpdir) / "query_run"
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query=f"Analyze the PDF at `{source_pdf}`",
                    outdir=outdir,
                    retrieval_backend="auto",
                    dry_run=True,
                    auto_topics_from_pdf=False,
                )
            )

            plan = runner._run_query_interpretation_agent()
            selected, acquired, *_ = runner._run_corpus_materialization_agent(plan)

            self.assertEqual(len(selected), 1)
            self.assertEqual(len(acquired), 1)
            acquired_pdf = Path(acquired[0].acquired_pdf_path)
            self.assertTrue(acquired_pdf.exists())
            self.assertEqual(acquired_pdf.read_bytes(), source_pdf.read_bytes())

    def test_runner_materializes_html_article_url_into_extractable_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://example.org/news/story-about-water",
                    outdir=outdir,
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )
            plan = runner._run_query_interpretation_agent()
            article_html = """
                <html>
                  <head><title>Winter Storms Ease Drought in California</title></head>
                  <body>
                    <article>
                      <p>Winter storms improved reservoir levels across California.</p>
                      <p>The article links heavy rainfall to lower drought pressure and reduced wildfire risk.</p>
                    </article>
                  </body>
                </html>
            """
            runner._fetch_url_payload = lambda url, referer=None: (  # type: ignore[assignment]
                article_html.encode("utf-8"),
                "text/html; charset=utf-8",
                url,
            )

            selected, acquired, *_ = runner._run_corpus_materialization_agent(plan)

            self.assertEqual(len(selected), 1)
            self.assertEqual(len(acquired), 1)
            acquired_pdf = Path(acquired[0].acquired_pdf_path)
            self.assertTrue(acquired_pdf.exists())
            self.assertTrue(acquired_pdf.read_bytes().startswith(b"%PDF-"))
            extracted_text = runner._extract_pdf_text_for_validation(acquired_pdf)
            if extracted_text is not None:
                self.assertIn("Winter storms improved reservoir levels across California.", extracted_text)
                self.assertIn("reduced wildfire risk", extracted_text)

    def test_direct_url_fetch_retries_with_longer_timeouts_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://example.org/story",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    retrieval_timeout_seconds=7.0,
                    dry_run=True,
                )
            )
            seen_timeouts: list[float] = []

            class FakeHeaders:
                def get(self, key: str, default=None):
                    if key.lower() == "content-type":
                        return "text/html; charset=utf-8"
                    return default

            class FakeResponse:
                headers = FakeHeaders()

                def geturl(self) -> str:
                    return "https://example.org/story"

                def read(self, size: int = -1) -> bytes:
                    del size
                    return b"<html><body><article>hello</article></body></html>"

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def fake_urlopen(request, timeout):
                del request
                seen_timeouts.append(timeout)
                if len(seen_timeouts) == 1:
                    raise TimeoutError("The read operation timed out")
                return FakeResponse()

            with patch.object(democritus_query_agentic_module, "urlopen", side_effect=fake_urlopen):
                payload, content_type, source_reference = runner._fetch_url_payload("https://example.org/story")

            self.assertEqual(payload, b"<html><body><article>hello</article></body></html>")
            self.assertEqual(content_type, "text/html; charset=utf-8")
            self.assertEqual(source_reference, "https://example.org/story")
            self.assertEqual(seen_timeouts, [7.0, 45.0])

    def test_non_pdf_remote_payload_reads_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://example.org/story",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            class FakeResponse:
                def __init__(self) -> None:
                    self.sizes: list[int] = []

                def read(self, size: int = -1) -> bytes:
                    self.sizes.append(size)
                    return b"<html><body><article><p>ok</p></article></body></html>" if len(self.sizes) == 1 else b""

            response = FakeResponse()

            payload = runner._read_remote_payload(
                response,
                content_type="text/html; charset=utf-8",
                source_reference="https://example.org/story",
            )

            self.assertIn(b"<article>", payload)
            self.assertEqual(response.sizes, [democritus_query_agentic_module._NON_PDF_STREAM_CHUNK_BYTES])

    def test_washington_post_uses_smaller_non_pdf_byte_budget(self) -> None:
        limit = democritus_query_agentic_module.DemocritusQueryAgenticRunner._non_pdf_remote_byte_limit(
            "https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits/"
        )

        self.assertEqual(limit, 768 * 1024)

    def test_washington_post_direct_url_request_specs_include_range_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits/",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            specs = runner._direct_document_request_specs(
                "https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits/"
            )

            self.assertEqual(len(specs), 3)
            self.assertTrue(any("Range" in headers for headers in specs))
            self.assertTrue(any(headers.get("Connection") == "close" for headers in specs))

    def test_washington_post_direct_url_uses_host_specific_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits/",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    retrieval_timeout_seconds=7.0,
                    dry_run=True,
                )
            )

            timeouts = runner._request_timeouts_for_url(
                "https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits/",
                kind="direct_url",
            )

            self.assertEqual(timeouts, (8.0, 20.0, 40.0))

    def test_non_pdf_chunked_reader_stops_once_article_markup_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="Analyze the document at https://example.org/story",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="auto",
                    dry_run=True,
                )
            )

            class FakeResponse:
                def __init__(self) -> None:
                    self.calls = 0

                def read(self, size: int = -1) -> bytes:
                    self.calls += 1
                    if self.calls == 1:
                        return (
                            b"<html><body><article><p>One</p><p>Two</p><p>Three</p>"
                            b"<p>Four</p><p>Five</p><p>Six</p></article>"
                        )
                    return b""

            response = FakeResponse()

            payload = runner._read_non_pdf_payload_chunked(
                response,
                source_reference="https://example.org/story",
            )

            self.assertIn(b"</article>", payload)
            self.assertEqual(response.calls, 1)

    def test_query_runner_acquires_manifest_documents_and_hands_off_to_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source_pdfs"
            outdir = root / "query_run"
            source_dir.mkdir()

            alpha_pdf = source_dir / "red_wine_benefits_study.pdf"
            beta_pdf = source_dir / "red_wine_cardiovascular_review.pdf"
            gamma_pdf = source_dir / "coffee_habits_report.pdf"
            for path in (alpha_pdf, beta_pdf, gamma_pdf):
                path.write_bytes(b"%PDF-1.4\n")

            manifest_path = root / "studies.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "title": "Benefits of Red Wine on Cardiovascular Health",
                                "abstract": "A study of moderate red wine consumption and cardiovascular outcomes.",
                                "pdf_path": str(alpha_pdf),
                                "year": 2022,
                            }
                        ),
                        json.dumps(
                            {
                                "title": "Red Wine Polyphenols Review",
                                "abstract": "A review of mechanisms and benefits linked to red wine polyphenols.",
                                "pdf_path": str(beta_pdf),
                                "year": 2021,
                            }
                        ),
                        json.dumps(
                            {
                                "title": "Coffee Consumption Report",
                                "abstract": "Not about red wine.",
                                "pdf_path": str(gamma_pdf),
                                "year": 2020,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="find me 2 studies of the benefits of drinking red wine",
                    outdir=outdir,
                    target_documents=2,
                    manifest_path=manifest_path,
                    max_workers=2,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertGreaterEqual(len(result.selected_documents), 2)
            self.assertGreaterEqual(len(result.acquired_documents), 2)
            selected_titles = {item.title for item in result.selected_documents}
            self.assertIn("Benefits of Red Wine on Cardiovascular Health", selected_titles)
            self.assertIn("Red Wine Polyphenols Review", selected_titles)
            self.assertTrue((outdir / "query_plan.json").exists())
            self.assertTrue((outdir / "acquired_corpus_manifest.json").exists())
            self.assertTrue((outdir / "democritus_runs" / "telemetry.json").exists())
            self.assertTrue(any("red" in token for token in result.query_plan.keyword_tokens))
            self.assertTrue(all(Path(item.acquired_pdf_path).exists() for item in result.acquired_documents))

    def test_query_runner_can_search_pdf_filenames_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source_pdfs"
            outdir = root / "query_run"
            source_dir.mkdir()
            (source_dir / "red_wine_benefits_01.pdf").write_bytes(b"%PDF-1.4\n")
            (source_dir / "red_wine_benefits_02.pdf").write_bytes(b"%PDF-1.4\n")
            (source_dir / "industrial_supply_chain.pdf").write_bytes(b"%PDF-1.4\n")

            runner = DemocritusQueryAgenticRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine benefits",
                    outdir=outdir,
                    target_documents=2,
                    source_pdf_root=source_dir,
                    max_workers=2,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_documents), 2)
            titles = {item.title for item in result.selected_documents}
            self.assertIn("red_wine_benefits_01", titles)
            self.assertIn("red_wine_benefits_02", titles)

    def test_query_runner_skips_unextractable_pdfs_and_continues_materialization(self) -> None:
        class FakeValidationRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "manifest"

                    def __init__(self, docs):
                        self._docs = docs

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan
                        return tuple(self._docs[:limit])

                return FakeProvider(self._fake_docs)

            def _validate_materialized_pdf(self, path: Path) -> None:
                self._validate_pdf_file(path)
                if "bad_study" in path.name:
                    raise RuntimeError("PDF 'bad_study.pdf' does not contain extractable text for Democritus topic discovery")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source_pdfs"
            outdir = root / "query_run"
            source_dir.mkdir()

            bad_pdf = source_dir / "bad_study.pdf"
            good_pdf_1 = source_dir / "good_study_1.pdf"
            good_pdf_2 = source_dir / "good_study_2.pdf"
            for path in (bad_pdf, good_pdf_1, good_pdf_2):
                path.write_bytes(b"%PDF-1.4\n")

            runner = FakeValidationRunner(
                DemocritusQueryAgenticConfig(
                    query="find me 2 studies of the benefits of drinking red wine",
                    outdir=outdir,
                    target_documents=2,
                    retrieval_backend="manifest",
                    include_phase2=False,
                    dry_run=True,
                )
            )
            runner._fake_docs = (
                DiscoveredDocument(
                    title="Bad Study",
                    score=9.0,
                    retrieval_backend="manifest",
                    source_path=str(bad_pdf),
                    document_format="pdf",
                ),
                DiscoveredDocument(
                    title="Good Study 1",
                    score=8.0,
                    retrieval_backend="manifest",
                    source_path=str(good_pdf_1),
                    document_format="pdf",
                ),
                DiscoveredDocument(
                    title="Good Study 2",
                    score=7.0,
                    retrieval_backend="manifest",
                    source_path=str(good_pdf_2),
                    document_format="pdf",
                ),
            )

            result = runner.run()

            self.assertEqual(len(result.acquired_documents), 2)
            acquired_titles = {item.title for item in result.selected_documents}
            self.assertNotIn("Bad Study", acquired_titles)
            self.assertIn("Good Study 1", acquired_titles)
            self.assertIn("Good Study 2", acquired_titles)
            materialization_log = (outdir / "query_agent_logs" / "corpus_materialization_agent.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("[SKIP]", materialization_log)
            self.assertIn("does not contain extractable text", materialization_log)

    def test_crossref_backend_extracts_pdf_links_from_remote_metadata(self) -> None:
        class FakeCrossrefBackend(CrossrefRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del url, params
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/redwine",
                                "title": ["Red Wine and Cardiovascular Outcomes"],
                                "abstract": "Study of red wine benefits.",
                                "URL": "https://doi.org/10.1000/redwine",
                                "link": [
                                    {
                                        "URL": "https://example.org/red_wine.pdf",
                                        "content-type": "application/pdf",
                                    }
                                ],
                                "is-referenced-by-count": 120,
                                "type": "journal-article",
                                "published-online": {"date-parts": [[2024, 5, 1]]},
                            }
                        ]
                    }
                }

        backend = FakeCrossrefBackend(user_agent="test-agent", timeout_seconds=5.0)
        results = backend.search(
            QueryPlan(
                query="red wine benefits",
                normalized_query="red wine benefits",
                keyword_tokens=("red", "wine", "benefits"),
                target_documents=1,
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "crossref")
        self.assertEqual(results[0].download_url, "https://example.org/red_wine.pdf")
        self.assertEqual(results[0].document_format, "pdf")

    def test_crossref_backend_downranks_generic_highly_cited_matches_for_red_wine_queries(self) -> None:
        class FakeCrossrefBackend(CrossrefRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del url, params
                return {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/specific-red-wine",
                                "title": ["Red Wine Polyphenols and Cardiovascular Outcomes"],
                                "abstract": "Study of red wine health impacts and polyphenol mechanisms.",
                                "URL": "https://doi.org/10.1000/specific-red-wine",
                                "link": [{"URL": "https://example.org/red_wine.pdf", "content-type": "application/pdf"}],
                                "is-referenced-by-count": 12,
                                "type": "journal-article",
                                "published-online": {"date-parts": [[2024, 5, 1]]},
                            },
                            {
                                "DOI": "10.1000/generic-diet",
                                "title": ["Mediterranean Diet and Cardiovascular Health"],
                                "abstract": "A broad diet review that briefly mentions red wine consumption.",
                                "URL": "https://doi.org/10.1000/generic-diet",
                                "link": [{"URL": "https://example.org/generic.pdf", "content-type": "application/pdf"}],
                                "is-referenced-by-count": 5000,
                                "type": "journal-article",
                                "published-online": {"date-parts": [[2024, 4, 1]]},
                            },
                        ]
                    }
                }

        backend = FakeCrossrefBackend(user_agent="test-agent", timeout_seconds=5.0)
        results = backend.search(
            QueryPlan(
                query="health impacts of drinking red wine",
                normalized_query="health impacts red wine",
                keyword_tokens=("health", "impacts", "red", "wine"),
                target_documents=2,
                retrieval_query="health impacts red wine",
            ),
            limit=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].identifier, "10.1000/specific-red-wine")
        self.assertGreater(results[0].score, results[1].score)

    def test_europe_pmc_backend_prefers_pmc_oa_pdf_links(self) -> None:
        class FakeEuropePMCBackend(EuropePMCOARetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del url, params
                return {
                    "resultList": {
                        "result": [
                            {
                                "id": "123456",
                                "source": "MED",
                                "pmcid": "PMC9999999",
                                "doi": "10.1000/redwine",
                                "title": "Red Wine and Cardiovascular Outcomes",
                                "abstractText": "Study of red wine health impacts.",
                                "journalTitle": "Open Cardiology",
                                "pubYear": "2024",
                                "isOpenAccess": "Y",
                                "citedByCount": 50,
                            }
                        ]
                    }
                }

            def _fetch_text(self, url: str, params=None, *, accept="text/plain"):
                del url, params, accept
                return """<OA><records><record id='PMC9999999'><link format='pdf' href='ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/test/redwine.pdf'/></record></records></OA>"""

        backend = FakeEuropePMCBackend(user_agent="test-agent", timeout_seconds=5.0)
        results = backend.search(
            QueryPlan(
                query="health impacts of drinking red wine",
                normalized_query="health impacts red wine",
                keyword_tokens=("health", "impacts", "red", "wine"),
                target_documents=1,
                retrieval_query="health impacts red wine",
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "europe_pmc")
        self.assertEqual(results[0].download_url, "https://ftp.ncbi.nlm.nih.gov/pub/pmc/test/redwine.pdf")
        self.assertEqual(results[0].document_format, "pdf")

    def test_sec_backend_parses_recent_filings(self) -> None:
        class FakeSECRetrievalBackend(SECFilingRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del params
                if url.endswith("company_tickers.json"):
                    return {
                        "0": {"ticker": "IBM", "title": "International Business Machines Corp", "cik_str": 51143},
                        "1": {"ticker": "KO", "title": "Coca Cola Co", "cik_str": 21344},
                    }
                if "CIK0000051143.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000051143-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-02-20"],
                                "primaryDocument": ["ibm10k2024.htm"],
                            }
                        }
                    }
                return {"filings": {"recent": {}}}

        backend = FakeSECRetrievalBackend(
            user_agent="test-agent",
            timeout_seconds=5.0,
            form_types=("10-K",),
            company_limit=2,
        )
        results = backend.search(
            QueryPlan(
                query="find me IBM 10-K filings",
                normalized_query="find me ibm 10-k filings",
                keyword_tokens=("ibm", "10", "k"),
                target_documents=2,
                requested_forms=("10-K",),
            ),
            limit=2,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "sec")
        self.assertEqual(results[0].document_format, "html")
        self.assertEqual(results[0].metadata["ticker"], "IBM")
        self.assertEqual(results[0].metadata["form"], "10-K")

    def test_sec_backend_does_not_match_numeric_form_token_to_wrong_company(self) -> None:
        class FakeSECRetrievalBackend(SECFilingRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del params
                if url.endswith("company_tickers.json"):
                    return {
                        "0": {"ticker": "ADBE", "title": "Adobe Inc.", "cik_str": 796343},
                        "1": {"ticker": "TXG", "title": "10x Genomics, Inc.", "cik_str": 1770787},
                    }
                if "CIK0000796343.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000796343-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-01-19"],
                                "primaryDocument": ["adbe10k.htm"],
                            }
                        }
                    }
                if "CIK0001770787.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0001770787-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-02-28"],
                                "primaryDocument": ["txg10k.htm"],
                            }
                        }
                    }
                return {"filings": {"recent": {}}}

        backend = FakeSECRetrievalBackend(
            user_agent="test-agent",
            timeout_seconds=5.0,
            form_types=("10-K",),
            company_limit=1,
        )
        results = backend.search(
            QueryPlan(
                query="find me 10 recent 10-K filings for Adobe",
                normalized_query="find me 10 recent 10-k filings for adobe",
                keyword_tokens=("10", "recent", "adobe"),
                target_documents=1,
                requested_forms=("10-K",),
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["ticker"], "ADBE")
        self.assertIn("adobe", results[0].evidence)

    def test_sec_backend_uses_strict_company_targets_when_query_mentions_workflows(self) -> None:
        class FakeSECRetrievalBackend(SECFilingRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del params
                if url.endswith("company_tickers.json"):
                    return {
                        "0": {"ticker": "ADBE", "title": "Adobe Inc.", "cik_str": 796343},
                        "1": {"ticker": "BSTX", "title": "Bespoke Extracts, Inc.", "cik_str": 999999},
                    }
                if "CIK0000796343.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000796343-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-01-19"],
                                "primaryDocument": ["adbe10k.htm"],
                            }
                        }
                    }
                if "CIK0000999999.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000999999-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2009-05-15"],
                                "primaryDocument": ["bstx10k.htm"],
                            }
                        }
                    }
                return {"filings": {"recent": {}}}

        backend = FakeSECRetrievalBackend(
            user_agent="test-agent",
            timeout_seconds=5.0,
            form_types=("10-K",),
            company_limit=3,
        )
        results = backend.search(
            QueryPlan(
                query="Analyze the recent 10-K filings from Adobe and extract workflows",
                normalized_query="analyze recent 10-k filings from adobe and extract workflows",
                retrieval_query="analyze recent 10 k filings from adobe and extract workflows",
                keyword_tokens=("analyze", "recent", "10", "filings", "adobe", "extract", "workflows"),
                target_documents=2,
                requested_forms=("10-K",),
                sec_company_targets=(("adobe",),),
            ),
            limit=2,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["ticker"], "ADBE")

    def test_sec_backend_expands_djia_targets_beyond_default_company_limit(self) -> None:
        class FakeSECRetrievalBackend(SECFilingRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del params
                if url.endswith("company_tickers.json"):
                    return {
                        "0": {"ticker": "AAPL", "title": "Apple Inc.", "cik_str": 320193},
                        "1": {"ticker": "IBM", "title": "International Business Machines Corp", "cik_str": 51143},
                        "2": {"ticker": "ADBE", "title": "Adobe Inc.", "cik_str": 796343},
                    }
                if "CIK0000320193.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000320193-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-11-01"],
                                "primaryDocument": ["aapl10k.htm"],
                            }
                        }
                    }
                if "CIK0000051143.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000051143-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-02-20"],
                                "primaryDocument": ["ibm10k.htm"],
                            }
                        }
                    }
                if "CIK0000796343.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000796343-24-000001"],
                                "form": ["10-K"],
                                "filingDate": ["2024-01-19"],
                                "primaryDocument": ["adbe10k.htm"],
                            }
                        }
                    }
                return {"filings": {"recent": {}}}

        backend = FakeSECRetrievalBackend(
            user_agent="test-agent",
            timeout_seconds=5.0,
            form_types=("10-K",),
            company_limit=1,
        )
        results = backend.search(
            QueryPlan(
                query="Analyze the recent 10-K filings from the DJIA companies",
                normalized_query="analyze the recent 10-k filings from the djia companies",
                retrieval_query="analyze recent 10 k filings djia companies",
                keyword_tokens=("analyze", "recent", "10", "filings", "djia", "companies"),
                target_documents=5,
                requested_forms=("10-K",),
                sec_company_targets=(("apple", "aapl"), ("ibm", "international business machines")),
                sec_cohort_mode="latest_per_company",
            ),
            limit=5,
        )

        self.assertEqual({item.metadata["ticker"] for item in results}, {"AAPL", "IBM"})

    def test_sec_backend_returns_latest_matching_filing_per_company_in_cohort_mode(self) -> None:
        class FakeSECRetrievalBackend(SECFilingRetrievalBackend):
            def _fetch_json(self, url: str, params=None):
                del params
                if url.endswith("company_tickers.json"):
                    return {
                        "0": {"ticker": "IBM", "title": "International Business Machines Corp", "cik_str": 51143},
                        "1": {"ticker": "AXP", "title": "American Express Co", "cik_str": 4962},
                    }
                if "CIK0000051143.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000051143-24-000002", "0000051143-24-000001"],
                                "form": ["10-K", "10-K"],
                                "filingDate": ["2024-02-21", "2023-02-22"],
                                "primaryDocument": ["ibm10k2024.htm", "ibm10k2023.htm"],
                            }
                        }
                    }
                if "CIK0000004962.json" in url:
                    return {
                        "filings": {
                            "recent": {
                                "accessionNumber": ["0000004962-24-000001", "0000004962-23-000001"],
                                "form": ["10-K", "10-K"],
                                "filingDate": ["2024-02-09", "2023-02-10"],
                                "primaryDocument": ["axp10k2024.htm", "axp10k2023.htm"],
                            }
                        }
                    }
                return {"filings": {"recent": {}}}

        backend = FakeSECRetrievalBackend(
            user_agent="test-agent",
            timeout_seconds=5.0,
            form_types=("10-K",),
            company_limit=1,
        )
        results = backend.search(
            QueryPlan(
                query="Analyze the recent 10-K filings from the DJIA companies",
                normalized_query="analyze the recent 10-k filings from the djia companies",
                retrieval_query="analyze recent 10 k filings djia companies",
                keyword_tokens=("analyze", "recent", "10", "filings", "djia", "companies"),
                target_documents=2,
                requested_forms=("10-K",),
                sec_company_targets=(("ibm", "international business machines"), ("american express", "axp")),
                sec_cohort_mode="latest_per_company",
            ),
            limit=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual({item.metadata["ticker"] for item in results}, {"IBM", "AXP"})
        titles = {item.title for item in results}
        self.assertIn("International Business Machines Corp 10-K 2024-02-21", titles)
        self.assertIn("American Express Co 10-K 2024-02-09", titles)

    def test_resolve_sec_user_agent_uses_env_identity_when_placeholder_is_passed(self) -> None:
        with patch.dict(os.environ, {"FF2_SEC_USER_AGENT": "Research User researcher@example.com"}, clear=False):
            resolved = _resolve_sec_user_agent("FunctorFlow_v2/0.1 (agentic SEC retrieval; local use)")

        self.assertEqual(resolved, "Research User researcher@example.com")

    def test_resolve_sec_user_agent_requires_contactable_identity(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FF2_SEC_USER_AGENT": "",
                "SEC_USER_AGENT": "",
                "SEC_IDENTITY": "",
                "SEC_CONTACT_NAME": "",
                "SEC_CONTACT_EMAIL": "",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError) as ctx:
                _resolve_sec_user_agent("FunctorFlow_v2/0.1 (agentic SEC retrieval; local use)")

        self.assertIn("SEC retrieval requires an identifying User-Agent", str(ctx.exception))

    def test_discovery_only_runner_supports_non_pdf_results_for_future_systems(self) -> None:
        class FakeSECRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "sec"

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredDocument(
                                title="IBM 10-K 2024-02-20",
                                score=8.0,
                                retrieval_backend="sec",
                                download_url="https://www.sec.gov/Archives/edgar/data/51143/000005114324000001/ibm10k2024.htm",
                                url="https://www.sec.gov/Archives/edgar/data/51143/000005114324000001/ibm10k2024.htm",
                                document_format="html",
                                metadata={"ticker": "IBM", "form": "10-K"},
                            ),
                        )

                return FakeProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = FakeSECRunner(
                DemocritusQueryAgenticConfig(
                    query="find me IBM 10-K filings",
                    outdir=outdir,
                    target_documents=1,
                    retrieval_backend="sec",
                    discovery_only=True,
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_documents), 1)
            self.assertEqual(len(result.acquired_documents), 0)
            self.assertEqual(len(result.batch_records), 0)
            summary = json.loads((outdir / "query_run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["discovery_only"])
            self.assertEqual(summary["retrieval_backend"], "sec")

    def test_scholarly_backend_prefers_pdf_results_across_sources(self) -> None:
        class StaticBackend:
            def __init__(self, backend_name: str, results: tuple[DiscoveredDocument, ...]) -> None:
                self.backend_name = backend_name
                self._results = results

            def search(self, plan: QueryPlan, *, limit: int):
                del plan
                return self._results[:limit]

        backend = ScholarlyRetrievalBackend(
            (
                StaticBackend(
                    "semantic_scholar",
                    (
                        DiscoveredDocument(
                            title="Red Wine Outcomes",
                            score=7.0,
                            retrieval_backend="semantic_scholar",
                            url="https://example.org/html",
                            document_format="html",
                            identifier="same-paper",
                        ),
                    ),
                ),
                StaticBackend(
                    "crossref",
                    (
                        DiscoveredDocument(
                            title="Red Wine Outcomes",
                            score=6.0,
                            retrieval_backend="crossref",
                            download_url="https://example.org/red_wine.pdf",
                            document_format="pdf",
                            identifier="same-paper",
                        ),
                    ),
                ),
            )
        )

        results = backend.search(
            QueryPlan(
                query="red wine benefits",
                normalized_query="red wine benefits",
                keyword_tokens=("red", "wine", "benefits"),
                target_documents=1,
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "crossref")
        self.assertEqual(results[0].document_format, "pdf")

    def test_scholarly_backend_prefers_europe_pmc_pdf_over_other_pdf_sources(self) -> None:
        class StaticBackend:
            def __init__(self, name: str, results: tuple[DiscoveredDocument, ...]) -> None:
                self.backend_name = name
                self._results = results

            def search(self, plan: QueryPlan, *, limit: int):
                del plan, limit
                return self._results

        backend = ScholarlyRetrievalBackend(
            (
                StaticBackend(
                    "semantic_scholar",
                    (
                        DiscoveredDocument(
                            title="Red Wine Outcomes",
                            score=9.0,
                            retrieval_backend="semantic_scholar",
                            download_url="https://example.org/semantic.pdf",
                            document_format="pdf",
                            identifier="same-paper",
                        ),
                    ),
                ),
                StaticBackend(
                    "europe_pmc",
                    (
                        DiscoveredDocument(
                            title="Red Wine Outcomes",
                            score=8.0,
                            retrieval_backend="europe_pmc",
                            download_url="https://ftp.ncbi.nlm.nih.gov/pub/pmc/test/redwine.pdf",
                            document_format="pdf",
                            identifier="same-paper",
                        ),
                    ),
                ),
            )
        )

        results = backend.search(
            QueryPlan(
                query="red wine benefits",
                normalized_query="red wine benefits",
                keyword_tokens=("red", "wine", "benefits"),
                target_documents=1,
            ),
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_backend, "europe_pmc")

    def test_scholarly_backend_caps_backend_limit_for_large_targets(self) -> None:
        class RecordingBackend:
            def __init__(self, name: str) -> None:
                self.backend_name = name
                self.limits: list[int] = []

            def search(self, plan: QueryPlan, *, limit: int):
                del plan
                self.limits.append(limit)
                return ()

        semantic = RecordingBackend("semantic_scholar")
        crossref = RecordingBackend("crossref")
        backend = ScholarlyRetrievalBackend((semantic, crossref))

        results = backend.search(
            QueryPlan(
                query="health impacts of drinking red wine",
                normalized_query="health impacts red wine",
                keyword_tokens=("health", "impacts", "red", "wine"),
                target_documents=50,
                retrieval_query="health impacts red wine",
            ),
            limit=600,
        )

        self.assertEqual(results, ())
        self.assertEqual(semantic.limits, [100])
        self.assertEqual(crossref.limits, [100])

    def test_materialization_skips_blocked_downloads_and_uses_next_candidate(self) -> None:
        class FallbackRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "scholarly"

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredDocument(
                                title="Blocked PDF",
                                score=9.0,
                                retrieval_backend="crossref",
                                download_url="https://example.org/blocked.pdf",
                                url="https://doi.org/blocked",
                                document_format="pdf",
                            ),
                            DiscoveredDocument(
                                title="Working PDF",
                                score=8.0,
                                retrieval_backend="semantic_scholar",
                                download_url="https://example.org/working.pdf",
                                url="https://example.org/working",
                                document_format="pdf",
                            ),
                        )

                return FakeProvider()

            def _download_file(self, url: str, target_path: Path, *, referer: str | None = None) -> None:
                del referer
                if "blocked" in url:
                    raise RuntimeError("HTTP Error 403: Forbidden")
                target_path.write_bytes(b"%PDF-1.4\n")

            def _validate_materialized_pdf(self, path: Path) -> None:
                self._validate_pdf_file(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = FallbackRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine benefits",
                    outdir=outdir,
                    target_documents=1,
                    retrieval_backend="scholarly",
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_documents), 1)
            self.assertEqual(result.selected_documents[0].title, "Working PDF")
            self.assertEqual(len(result.acquired_documents), 1)
            self.assertTrue(Path(result.acquired_documents[0].acquired_pdf_path).exists())

    def test_materialization_falls_back_to_landing_page_when_pdf_link_is_broken(self) -> None:
        class FallbackRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "scholarly"

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredDocument(
                                title="Fallback Article",
                                score=9.0,
                                retrieval_backend="crossref",
                                download_url="https://example.org/broken.pdf",
                                url="https://example.org/article",
                                document_format="pdf",
                            ),
                        )

                return FakeProvider()

            def _download_file(self, url: str, target_path: Path, *, referer: str | None = None) -> None:
                del url, target_path, referer
                raise RuntimeError("HTTP Error 404: Not Found")

            def _validate_materialized_pdf(self, path: Path) -> None:
                self._validate_pdf_file(path)

            def _fetch_url_payload(
                self,
                url: str,
                *,
                referer: str | None = None,
            ) -> tuple[bytes, str, str]:
                del referer
                if url != "https://example.org/article":
                    raise AssertionError(f"unexpected fallback URL: {url}")
                return (
                    b"<html><head><title>Fallback Article</title></head>"
                    b"<body><article><p>GLP-1 treatment reduces body weight.</p></article></body></html>",
                    "text/html; charset=utf-8",
                    url,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = FallbackRunner(
                DemocritusQueryAgenticConfig(
                    query="glp-1 weight loss",
                    outdir=outdir,
                    target_documents=1,
                    retrieval_backend="scholarly",
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_documents), 1)
            self.assertEqual(result.selected_documents[0].title, "Fallback Article")
            self.assertEqual(len(result.acquired_documents), 1)
            self.assertTrue(Path(result.acquired_documents[0].acquired_pdf_path).exists())

    def test_scholarly_discovery_limit_overfetches_for_materialization(self) -> None:
        runner = DemocritusQueryAgenticRunner(
            DemocritusQueryAgenticConfig(
                query="red wine benefits",
                outdir=Path("/tmp/unused-query-run"),
                target_documents=5,
                retrieval_backend="scholarly",
                dry_run=True,
            )
        )
        plan = QueryPlan(
            query="red wine benefits",
            normalized_query="red wine benefits",
            keyword_tokens=("red", "wine", "benefits"),
            target_documents=5,
        )
        self.assertEqual(runner._discovery_limit(plan), 60)

    def test_discovery_error_mentions_backend_and_retrieval_query(self) -> None:
        class FailingRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "scholarly"

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan, limit
                        raise RuntimeError("simulated backend failure")

                return FakeProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = FailingRunner(
                DemocritusQueryAgenticConfig(
                    query="health impacts of drinking red wine",
                    outdir=Path(tmpdir) / "query_run",
                    retrieval_backend="scholarly",
                    dry_run=True,
                )
            )
            plan = runner._run_query_interpretation_agent()

            with self.assertRaises(RuntimeError) as ctx:
                runner._run_document_discovery_agent(plan)

            message = str(ctx.exception)
            self.assertIn("backend 'scholarly'", message)
            self.assertIn("health impacts red wine", message)

    def test_materialization_rejects_empty_or_non_pdf_payloads(self) -> None:
        class ValidationRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "scholarly"

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan, limit
                        return (
                            DiscoveredDocument(
                                title="Empty Payload",
                                score=9.0,
                                retrieval_backend="crossref",
                                download_url="https://example.org/empty.pdf",
                                document_format="pdf",
                            ),
                            DiscoveredDocument(
                                title="HTML Payload",
                                score=8.0,
                                retrieval_backend="crossref",
                                download_url="https://example.org/html.pdf",
                                document_format="pdf",
                            ),
                            DiscoveredDocument(
                                title="Valid Payload",
                                score=7.0,
                                retrieval_backend="semantic_scholar",
                                download_url="https://example.org/valid.pdf",
                                document_format="pdf",
                            ),
                        )

                return FakeProvider()

            def _download_file(self, url: str, target_path: Path, *, referer: str | None = None) -> None:
                del referer
                if "empty" in url:
                    target_path.write_bytes(b"")
                    self._validate_pdf_file(target_path)
                    return
                if "html" in url:
                    target_path.write_text("<html>blocked</html>", encoding="utf-8")
                    self._validate_pdf_file(target_path)
                    return
                target_path.write_bytes(b"%PDF-1.4\n")
                self._validate_pdf_file(target_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = ValidationRunner(
                DemocritusQueryAgenticConfig(
                    query="climate change",
                    outdir=outdir,
                    target_documents=1,
                    retrieval_backend="scholarly",
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            result = runner.run()

            self.assertEqual(len(result.selected_documents), 1)
            self.assertEqual(result.selected_documents[0].title, "Valid Payload")
            self.assertEqual(len(result.acquired_documents), 1)

    def test_query_runner_streams_documents_into_live_batch_runner(self) -> None:
        class FakeStreamingBatchRunner:
            def __init__(self) -> None:
                self.run_started_at: float | None = None
                self.close_called_at: float | None = None
                self.registered_paths: list[tuple[str, float]] = []
                self._closed = threading.Event()

            def register_document(self, pdf_path: Path):
                self.registered_paths.append((pdf_path.name, time.time()))
                return None

            def close_document_stream(self) -> None:
                self.close_called_at = time.time()
                self._closed.set()

            def run_with_artifacts(self) -> DemocritusBatchRunResult:
                self.run_started_at = time.time()
                self._closed.wait(timeout=1.0)
                records = tuple(
                    DemocritusBatchRecord(
                        run_name=f"run_{index}",
                        pdf_path=name,
                        agent_record=DemocritusAgentRecord(
                            agent_name="document_intake_agent",
                            frontier_index=0,
                            status="ok",
                            started_at=self.run_started_at,
                            ended_at=self.close_called_at or self.run_started_at,
                        ),
                    )
                    for index, (name, _) in enumerate(self.registered_paths, start=1)
                )
                return DemocritusBatchRunResult(records=records, csql_bundle=None)

        class StreamingQueryRunner(DemocritusQueryAgenticRunner):
            def __init__(self, config: DemocritusQueryAgenticConfig) -> None:
                super().__init__(config)
                self.fake_batch_runner = FakeStreamingBatchRunner()
                self.streaming_flag: bool | None = None
                self.discovery_started_at: float | None = None

            def _provider(self):
                class FakeProvider:
                    backend_name = "manifest"

                    def __init__(
                        self,
                        docs: tuple[DiscoveredDocument, ...],
                        runner: "StreamingQueryRunner",
                    ) -> None:
                        self._docs = docs
                        self._runner = runner

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan
                        self._runner.discovery_started_at = time.time()
                        return self._docs[:limit]

                return FakeProvider(self._fake_documents, self)

            def _build_batch_runner(self, *, streaming: bool):
                self.streaming_flag = streaming
                return self.fake_batch_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "query_run"
            source_dir = root / "source"
            source_dir.mkdir()
            alpha_pdf = source_dir / "alpha.pdf"
            beta_pdf = source_dir / "beta.pdf"
            alpha_pdf.write_bytes(b"%PDF-1.4\n")
            beta_pdf.write_bytes(b"%PDF-1.4\n")

            runner = StreamingQueryRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine benefits",
                    outdir=outdir,
                    target_documents=2,
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=False,
                )
            )
            runner._fake_documents = (
                DiscoveredDocument(
                    title="Alpha Study",
                    score=10.0,
                    retrieval_backend="manifest",
                    source_path=str(alpha_pdf),
                    document_format="pdf",
                ),
                DiscoveredDocument(
                    title="Beta Study",
                    score=9.0,
                    retrieval_backend="manifest",
                    source_path=str(beta_pdf),
                    document_format="pdf",
                ),
            )

            result = runner.run()

            self.assertEqual(len(result.acquired_documents), 2)
            self.assertEqual(len(result.batch_records), 2)
            self.assertTrue(runner.streaming_flag)
            self.assertIsNotNone(runner.fake_batch_runner.run_started_at)
            self.assertIsNotNone(runner.fake_batch_runner.close_called_at)
            self.assertEqual(len(runner.fake_batch_runner.registered_paths), 2)
            first_registered_at = runner.fake_batch_runner.registered_paths[0][1]
            second_registered_at = runner.fake_batch_runner.registered_paths[1][1]
            self.assertIsNotNone(runner.discovery_started_at)
            self.assertLessEqual(runner.fake_batch_runner.run_started_at, runner.discovery_started_at)
            self.assertLessEqual(runner.fake_batch_runner.run_started_at, first_registered_at)
            self.assertLessEqual(first_registered_at, runner.fake_batch_runner.close_called_at)
            self.assertLessEqual(second_registered_at, runner.fake_batch_runner.close_called_at)

    def test_query_runner_interactive_mode_writes_topic_checkpoint_artifact(self) -> None:
        class FakeInteractiveBatchRunner:
            def __init__(self, runner: "InteractiveQueryRunner") -> None:
                self.runner = runner
                self.documents: list[SimpleNamespace] = []
                self.closed = threading.Event()
                self.stop_after_frontier_index: int | None = None
                self.enable_corpus_synthesis: bool | None = None

            def register_document(self, pdf_path: Path):
                run_name = f"run_{len(self.documents) + 1}"
                outdir = self.runner.batch_outdir / run_name
                configs_dir = outdir / "configs"
                configs_dir.mkdir(parents=True, exist_ok=True)
                if "alpha" in pdf_path.name:
                    topics = "minimum wage increases\nemployment floor effects\n"
                    guide = {
                        "summary": "The article focuses on minimum wage policy and labor-market outcomes.",
                        "causal_gestalt": "Minimum wage changes alter wage floors and can reshape employer responses.",
                    }
                else:
                    topics = "minimum wage increases\nhousehold income effects\n"
                    guide = {
                        "summary": "This paper tracks income and labor consequences of wage-floor changes.",
                        "causal_gestalt": "Raising the wage floor can increase household earnings while shifting employment margins.",
                    }
                (configs_dir / "root_topics.txt").write_text(topics, encoding="utf-8")
                (configs_dir / "document_topic_guide.json").write_text(json.dumps(guide), encoding="utf-8")
                self.documents.append(
                    SimpleNamespace(
                        run_name=run_name,
                        outdir=outdir,
                        pdf_path=pdf_path,
                    )
                )
                return None

            def close_document_stream(self) -> None:
                self.closed.set()

            def _documents_snapshot(self):
                return tuple(self.documents)

            def run_with_artifacts(self) -> DemocritusBatchRunResult:
                self.closed.wait(timeout=1.0)
                return DemocritusBatchRunResult(records=(), csql_bundle=None, corpus_synthesis=None)

        class InteractiveQueryRunner(DemocritusQueryAgenticRunner):
            def __init__(self, config: DemocritusQueryAgenticConfig) -> None:
                super().__init__(config)
                self.fake_batch_runner = FakeInteractiveBatchRunner(self)

            def _run_query_interpretation_agent(self) -> QueryPlan:
                return QueryPlan(
                    query=self.config.query,
                    normalized_query=self.config.query.lower(),
                    keyword_tokens=("minimum", "wage"),
                    target_documents=2,
                )

            def _run_corpus_materialization_agent(self, plan: QueryPlan, on_document_acquired=None):
                del plan
                alpha_pdf = self.pdf_dir / "alpha_minimum_wage.pdf"
                beta_pdf = self.pdf_dir / "beta_minimum_wage.pdf"
                alpha_pdf.parent.mkdir(parents=True, exist_ok=True)
                alpha_pdf.write_bytes(b"%PDF-1.4\n")
                beta_pdf.write_bytes(b"%PDF-1.4\n")
                acquired = (
                    AcquiredCorpusDocument(
                        title="Alpha Minimum Wage Study",
                        acquired_pdf_path=str(alpha_pdf),
                        source_path=str(alpha_pdf),
                        score=9.0,
                        run_name_hint="alpha",
                        retrieval_backend="manifest",
                    ),
                    AcquiredCorpusDocument(
                        title="Beta Minimum Wage Study",
                        acquired_pdf_path=str(beta_pdf),
                        source_path=str(beta_pdf),
                        score=8.5,
                        run_name_hint="beta",
                        retrieval_backend="manifest",
                    ),
                )
                if on_document_acquired is not None:
                    for item in acquired:
                        on_document_acquired(item)
                selected = tuple(
                    DiscoveredDocument(
                        title=item.title,
                        score=item.score,
                        retrieval_backend=item.retrieval_backend,
                        source_path=item.source_path,
                        document_format="pdf",
                    )
                    for item in acquired
                )
                return selected, acquired, 1, False, None

            def _build_batch_runner(self, *, streaming: bool):
                del streaming
                return self.fake_batch_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "query_run"
            runner = InteractiveQueryRunner(
                DemocritusQueryAgenticConfig(
                    query="Find 10 studies of minimum wage and synthesize what they jointly support",
                    outdir=outdir,
                    execution_mode="interactive",
                    target_documents=2,
                )
            )

            result = runner.run()

            self.assertIsNotNone(result.checkpoint_manifest_path)
            self.assertIsNotNone(result.checkpoint_dashboard_path)
            self.assertTrue(result.checkpoint_manifest_path.exists())
            self.assertTrue(result.checkpoint_dashboard_path.exists())
            self.assertIsNone(result.corpus_synthesis_dashboard_path)
            payload = json.loads(result.checkpoint_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage_id"], "root_topics")
            self.assertEqual(payload["n_documents"], 2)
            self.assertTrue(any(item["topic"] == "minimum wage increases" for item in payload["top_topics"]))
            checkpoint_html = result.checkpoint_dashboard_path.read_text(encoding="utf-8")
            alpha_pdf_uri = (outdir / "acquired_pdfs" / "alpha_minimum_wage.pdf").resolve().as_uri()
            self.assertIn('class="doc-title"', checkpoint_html)
            self.assertIn("Inspect PDF", checkpoint_html)
            self.assertIn(alpha_pdf_uri, checkpoint_html)
            summary = json.loads((outdir / "query_run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["execution_mode"], "interactive")
            self.assertEqual(summary["checkpoint_manifest_path"], str(result.checkpoint_manifest_path))

    def test_query_runner_stops_after_democlritus_retrieval_state_stabilizes(self) -> None:
        class FakeBatchRunner:
            def run_with_artifacts(self) -> DemocritusBatchRunResult:
                return DemocritusBatchRunResult(records=(), csql_bundle=None)

        class ConvergingRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "manifest"

                    def __init__(self, docs: tuple[DiscoveredDocument, ...]) -> None:
                        self._docs = docs

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan
                        return self._docs[:limit]

                return FakeProvider(self._fake_documents)

            def _build_batch_runner(self, *, streaming: bool):
                del streaming
                return FakeBatchRunner()

            def _democritus_convergence_snapshot(self, plan: QueryPlan, selected_documents):
                del plan
                count = len(selected_documents)
                if count <= 2:
                    return DemocritusEvidenceSnapshot(
                        keyword_support=("red", "wine"),
                        top_context_terms=("polyphenols", "cardio"),
                        average_score=8.000,
                        evidence_count=count,
                    )
                return DemocritusEvidenceSnapshot(
                    keyword_support=("red", "wine"),
                    top_context_terms=("polyphenols", "cardio"),
                    average_score=8.020,
                    evidence_count=count,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "query_run"
            source_dir = root / "source_pdfs"
            source_dir.mkdir()
            fake_docs = []
            for index in range(4):
                pdf_path = source_dir / f"redwine_{index}.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n")
                fake_docs.append(
                    DiscoveredDocument(
                        title=f"Red wine study {index}",
                        score=10.0 - index,
                        retrieval_backend="manifest",
                        source_path=str(pdf_path),
                        document_format="pdf",
                    )
                )

            runner = ConvergingRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine health benefits",
                    outdir=outdir,
                    target_documents=2,
                    max_docs=4,
                    retrieval_backend="manifest",
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            runner._fake_documents = tuple(fake_docs)

            result = runner.run()

            self.assertEqual(len(result.selected_documents), 3)
            self.assertEqual(len(result.acquired_documents), 3)
            self.assertEqual(result.analysis_iterations, 2)
            self.assertTrue(result.consensus_reached)
            self.assertEqual(result.convergence_assessment["stop_trigger"], "stability")

    def test_query_runner_stops_at_budget_without_claiming_stability(self) -> None:
        class FakeBatchRunner:
            def run_with_artifacts(self) -> DemocritusBatchRunResult:
                return DemocritusBatchRunResult(records=(), csql_bundle=None)

        class BudgetStopRunner(DemocritusQueryAgenticRunner):
            def _provider(self):
                class FakeProvider:
                    backend_name = "manifest"

                    def __init__(self, docs: tuple[DiscoveredDocument, ...]) -> None:
                        self._docs = docs

                    def search(self, plan: QueryPlan, *, limit: int):
                        del plan
                        return self._docs[:limit]

                return FakeProvider(self._fake_documents)

            def _build_batch_runner(self, *, streaming: bool):
                del streaming
                return FakeBatchRunner()

            def _democritus_convergence_snapshot(self, plan: QueryPlan, selected_documents):
                del plan
                count = len(selected_documents)
                if count <= 2:
                    return DemocritusEvidenceSnapshot(
                        keyword_support=("red", "wine"),
                        top_context_terms=("polyphenols", "cardio"),
                        average_score=8.000,
                        evidence_count=count,
                    )
                return DemocritusEvidenceSnapshot(
                    keyword_support=("health",),
                    top_context_terms=("metabolism",),
                    average_score=6.500,
                    evidence_count=count,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "query_run"
            source_dir = root / "source_pdfs"
            source_dir.mkdir()
            fake_docs = []
            for index in range(3):
                pdf_path = source_dir / f"redwine_{index}.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n")
                fake_docs.append(
                    DiscoveredDocument(
                        title=f"Red wine study {index}",
                        score=10.0 - index,
                        retrieval_backend="manifest",
                        source_path=str(pdf_path),
                        document_format="pdf",
                    )
                )

            runner = BudgetStopRunner(
                DemocritusQueryAgenticConfig(
                    query="red wine health benefits",
                    outdir=outdir,
                    target_documents=2,
                    max_docs=3,
                    retrieval_backend="manifest",
                    include_phase2=False,
                    root_topic_strategy="heuristic",
                    dry_run=True,
                )
            )
            runner._fake_documents = tuple(fake_docs)

            result = runner.run()

            self.assertEqual(len(result.selected_documents), 3)
            self.assertEqual(len(result.acquired_documents), 3)
            self.assertEqual(result.analysis_iterations, 2)
            self.assertFalse(result.consensus_reached)
            self.assertEqual(result.convergence_assessment["stop_trigger"], "max_evidence")


if __name__ == "__main__":
    unittest.main()
