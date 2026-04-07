"""Batch-oriented agentic runner for Democritus."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from .csql_bundle import BatchCSQLBundleResult, build_batch_csql_bundle
from .democritus_agentic import DemocritusAgentRecord, DemocritusAgenticConfig, DemocritusAgenticRunner
from .democritus_corpus_synthesis import (
    DemocritusCorpusSynthesisResult,
    build_democritus_corpus_synthesis,
)
from .repo_layout import resolve_democritus_seed_pdf_root


def _default_pdf_dir() -> Path:
    return resolve_democritus_seed_pdf_root()


def _slugify(name: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "document"


def _sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _markdown_excerpt(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    paragraph_parts: list[str] = []
    for line in lines:
        if not line:
            if paragraph_parts:
                break
            continue
        if line.startswith("#") or line.startswith("|") or line.startswith("- ") or line.startswith(">"):
            continue
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        if cleaned:
            paragraph_parts.append(cleaned)
    return " ".join(paragraph_parts)[:420]


def _inline_markdown_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',
        escaped,
    )
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _pretty_document_title(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", name).strip())


def _summary_tiers_only(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    start = None
    end = len(lines)
    for index, raw_line in enumerate(lines):
        if re.match(r"^#{2,6}\s+Tier\s+1\b", raw_line.strip(), re.IGNORECASE):
            start = index
            break
    if start is None:
        return markdown_text
    for index in range(start + 1, len(lines)):
        if re.match(r"^#{2,6}\s+Notes and caveats\b", lines[index].strip(), re.IGNORECASE):
            end = index
            break
    trimmed = "\n".join(lines[start:end]).strip()
    return trimmed or markdown_text


def _strip_leading_markdown_title(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and re.match(r"^#\s+", lines[start].strip()):
        start += 1
        while start < len(lines) and not lines[start].strip():
            start += 1
    trimmed = "\n".join(lines[start:]).strip()
    return trimmed or markdown_text


def _markdown_text_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{' '.join(_inline_markdown_html(line) for line in paragraph)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            close_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{_inline_markdown_html(heading_match.group(2))}</h{level}>")
            continue
        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{_inline_markdown_html(stripped[2:].strip())}</li>")
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            close_list()
            parts.append(f"<blockquote>{_inline_markdown_html(stripped[1:].strip())}</blockquote>")
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            close_list()
            parts.append(f"<pre>{html.escape(stripped)}</pre>")
            continue
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    if in_code:
        flush_code()
    return "\n".join(parts) if parts else "<p>Artifact not available yet.</p>"


def _markdown_document_html(
    path: Path,
    *,
    tiers_only: bool = False,
    strip_leading_title: bool = False,
) -> str:
    if not path.exists():
        return "<p>Artifact not available yet.</p>"
    markdown_text = path.read_text(encoding="utf-8")
    if tiers_only:
        markdown_text = _summary_tiers_only(markdown_text)
    if strip_leading_title:
        markdown_text = _strip_leading_markdown_title(markdown_text)
    return _markdown_text_html(markdown_text)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DemocritusBatchConfig:
    """Configuration for running the agentic Democritus pipeline over a PDF directory."""

    pdf_dir: Path
    outdir: Path
    request_query: str = ""
    max_docs: int = 0
    max_workers: int = 8
    agent_concurrency_limits: tuple[tuple[str, int], ...] = ()
    include_phase2: bool = True
    auto_topics_from_pdf: bool = True
    root_topic_strategy: str = "v0_openai"
    discover_existing_documents: bool = True
    allow_incremental_admission: bool = False
    idle_poll_seconds: float = 0.5
    dry_run: bool = False
    intra_document_shards: int = 1
    enable_corpus_synthesis: bool = True

    def resolved(self) -> "DemocritusBatchConfig":
        return DemocritusBatchConfig(
            pdf_dir=self.pdf_dir.resolve(),
            outdir=self.outdir.resolve(),
            request_query=" ".join(self.request_query.split()),
            max_docs=self.max_docs,
            max_workers=self.max_workers,
            agent_concurrency_limits=tuple(
                (agent_name, limit) for agent_name, limit in self.agent_concurrency_limits
            ),
            include_phase2=self.include_phase2,
            auto_topics_from_pdf=self.auto_topics_from_pdf,
            root_topic_strategy=self.root_topic_strategy,
            discover_existing_documents=self.discover_existing_documents,
            allow_incremental_admission=self.allow_incremental_admission,
            idle_poll_seconds=self.idle_poll_seconds,
            dry_run=self.dry_run,
            intra_document_shards=max(1, int(self.intra_document_shards)),
            enable_corpus_synthesis=self.enable_corpus_synthesis,
        )


@dataclass(frozen=True)
class DemocritusBatchDocument:
    """One document in a batch run."""

    index: int
    pdf_path: Path
    run_name: str
    outdir: Path
    runner: DemocritusAgenticRunner
    plan: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DemocritusBatchRecord:
    """Execution record for one agent on one document."""

    run_name: str
    pdf_path: str
    agent_record: DemocritusAgentRecord


@dataclass(frozen=True)
class DemocritusBatchRunResult:
    """Result of a batch run, including optional integrated CSQL outputs."""

    records: tuple[DemocritusBatchRecord, ...]
    csql_bundle: BatchCSQLBundleResult | None = None
    corpus_synthesis: DemocritusCorpusSynthesisResult | None = None


class DemocritusBatchAgenticRunner:
    """Pipeline-parallel batch runner over a directory of Democritus PDFs."""

    def __init__(self, config: DemocritusBatchConfig) -> None:
        self.config = config.resolved()
        self._agent_limits = self._normalize_agent_limits(self.config.agent_concurrency_limits)
        self._state_lock = threading.Lock()
        self._state_cv = threading.Condition(self._state_lock)
        self.documents: list[DemocritusBatchDocument] = []
        self._documents_by_run_name: dict[str, DemocritusBatchDocument] = {}
        self._documents_by_pdf_path: dict[str, DemocritusBatchDocument] = {}
        self._pending_admissions: deque[DemocritusBatchDocument] = deque()
        self._next_document_index = 1
        self._admission_closed = not self.config.allow_incremental_admission
        if self.config.discover_existing_documents:
            for document in self._discover_documents():
                self._admit_document(document, enqueue=True)
        if self.config.discover_existing_documents and not self.config.allow_incremental_admission:
            self._admission_closed = True
        self.summary_path = self.config.outdir / "batch_agent_run_summary.json"
        self.telemetry_path = self.config.outdir / "telemetry.json"
        self.dashboard_path = self.config.outdir / "dashboard.html"
        self.gui_path = self.config.outdir / "democritus_gui.html"
        self.csql_summary_path = self.config.outdir / "csql" / "democritus_csql_summary.json"
        self.csql_sqlite_path = self.config.outdir / "csql" / "democritus_csql.sqlite"
        self._bootstrap_dashboard()

    def _document_by_run_name(self, run_name: str) -> DemocritusBatchDocument:
        with self._state_lock:
            document = self._documents_by_run_name.get(run_name)
        if document is None:
            raise KeyError(run_name)
        return document

    @staticmethod
    def _normalize_agent_limits(
        limits: tuple[tuple[str, int], ...],
    ) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for agent_name, limit in limits:
            if limit < 1:
                raise ValueError(f"Agent concurrency limit for {agent_name!r} must be >= 1.")
            normalized[agent_name] = limit
        return normalized

    def _agent_limit(self, agent_name: str) -> int:
        return self._agent_limits.get(agent_name, self.config.max_workers)

    def _discover_documents(self) -> tuple[DemocritusBatchDocument, ...]:
        if not self.config.pdf_dir.exists():
            raise FileNotFoundError(self.config.pdf_dir)
        if not self.config.pdf_dir.is_dir():
            raise NotADirectoryError(self.config.pdf_dir)

        pdf_paths = sorted(path for path in self.config.pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        if self.config.max_docs > 0:
            pdf_paths = pdf_paths[: self.config.max_docs]
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in directory: {self.config.pdf_dir}")

        documents: list[DemocritusBatchDocument] = []
        for index, pdf_path in enumerate(pdf_paths, start=1):
            documents.append(self._make_document(pdf_path, index=index))
        return tuple(documents)

    def _build_document_runner(
        self,
        *,
        pdf_path: Path,
        run_name: str,
        run_outdir: Path,
    ) -> DemocritusAgenticRunner:
        return DemocritusAgenticRunner(
            DemocritusAgenticConfig(
                outdir=run_outdir,
                domain_name=run_name,
                input_pdf=pdf_path,
                auto_topics_from_pdf=self.config.auto_topics_from_pdf,
                root_topic_strategy=self.config.root_topic_strategy,
                include_phase2=self.config.include_phase2,
                intra_document_shards=self.config.intra_document_shards,
            )
        )

    @staticmethod
    def _plan_for_runner(runner: DemocritusAgenticRunner) -> tuple[tuple[object, ...], ...]:
        return runner.workflow.parallel_frontiers()

    def _make_document(self, pdf_path: Path, *, index: int) -> DemocritusBatchDocument:
        resolved_pdf_path = pdf_path.resolve()
        run_name = f"{index:04d}_{_slugify(resolved_pdf_path.stem)}_{_sha256_prefix(resolved_pdf_path)}"
        run_outdir = self.config.outdir / run_name
        runner = self._build_document_runner(
            pdf_path=resolved_pdf_path,
            run_name=run_name,
            run_outdir=run_outdir,
        )
        return DemocritusBatchDocument(
            index=index,
            pdf_path=resolved_pdf_path,
            run_name=run_name,
            outdir=run_outdir,
            runner=runner,
            plan=self._plan_for_runner(runner),
        )

    def _admit_document(self, document: DemocritusBatchDocument, *, enqueue: bool) -> DemocritusBatchDocument:
        with self._state_cv:
            existing = self._documents_by_pdf_path.get(str(document.pdf_path))
            if existing is not None:
                return existing
            self.documents.append(document)
            self._documents_by_run_name[document.run_name] = document
            self._documents_by_pdf_path[str(document.pdf_path)] = document
            self._next_document_index = max(self._next_document_index, document.index + 1)
            if enqueue:
                self._pending_admissions.append(document)
                self._state_cv.notify_all()
            return document

    def register_document(self, pdf_path: Path) -> DemocritusBatchDocument:
        resolved_pdf_path = pdf_path.resolve()
        if not resolved_pdf_path.exists():
            raise FileNotFoundError(resolved_pdf_path)
        if not resolved_pdf_path.is_file():
            raise IsADirectoryError(resolved_pdf_path)
        if resolved_pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF path, got: {resolved_pdf_path}")
        with self._state_lock:
            if self._admission_closed:
                raise RuntimeError("Document admission is already closed for this batch runner.")
            existing = self._documents_by_pdf_path.get(str(resolved_pdf_path))
            if existing is not None:
                return existing
            index = self._next_document_index
        document = self._make_document(resolved_pdf_path, index=index)
        return self._admit_document(document, enqueue=True)

    def close_document_stream(self) -> None:
        with self._state_cv:
            self._admission_closed = True
            self._state_cv.notify_all()

    def _admission_is_closed(self) -> bool:
        with self._state_lock:
            return self._admission_closed

    def _documents_snapshot(self) -> tuple[DemocritusBatchDocument, ...]:
        with self._state_lock:
            return tuple(self.documents)

    def _drain_pending_admissions(self) -> list[DemocritusBatchDocument]:
        with self._state_lock:
            pending = list(self._pending_admissions)
            self._pending_admissions.clear()
        return pending

    def _wait_for_more_work(self) -> None:
        timeout = max(0.05, float(self.config.idle_poll_seconds))
        with self._state_cv:
            if self._pending_admissions or self._admission_closed:
                return
            self._state_cv.wait(timeout=timeout)

    def run(self) -> tuple[DemocritusBatchRecord, ...]:
        return self.run_with_artifacts().records

    def _bootstrap_dashboard(self) -> None:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self._write_telemetry(
            batch_started_at=time.time(),
            pending_frontiers={document.run_name: 0 for document in self._documents_snapshot()},
            active_agent_counts={},
            active_futures={},
            ready_queue=deque(),
            completed_records=[],
            status="starting" if self._admission_is_closed() else "waiting_for_documents",
        )

    def run_with_artifacts(self) -> DemocritusBatchRunResult:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        batch_started_at = time.time()
        if self.config.dry_run:
            records = []
            now = time.time()
            for document in self._documents_snapshot():
                for frontier_index, frontier in enumerate(document.plan):
                    for agent in frontier:
                        records.append(
                            DemocritusBatchRecord(
                                run_name=document.run_name,
                                pdf_path=str(document.pdf_path),
                                agent_record=DemocritusAgentRecord(
                                    agent_name=agent.name,
                                    frontier_index=frontier_index,
                                    status="planned",
                                    started_at=now,
                                    ended_at=now,
                                ),
                            )
                        )
            self._write_summary(tuple(records))
            self._write_telemetry(
                batch_started_at=batch_started_at,
                pending_frontiers={document.run_name: 0 for document in self.documents},
                active_agent_counts={},
                active_futures={},
                ready_queue=deque(),
                completed_records=records,
                status="dry_run_complete",
            )
            return DemocritusBatchRunResult(records=tuple(records), csql_bundle=None, corpus_synthesis=None)

        pending_frontiers = {document.run_name: 0 for document in self.documents}
        frontier_completion_counts: dict[tuple[str, int], int] = {}
        active_futures = {}
        active_agent_counts: dict[str, int] = {}
        completed_records: list[DemocritusBatchRecord] = []
        ready_queue: deque[tuple[DemocritusBatchDocument, int, str]] = deque()

        def enqueue_ready_frontier(document: DemocritusBatchDocument) -> None:
            frontier_index = pending_frontiers[document.run_name]
            if frontier_index >= len(document.plan):
                return
            frontier = document.plan[frontier_index]
            for agent in frontier:
                ready_queue.append((document, frontier_index, agent.name))

        def submit_ready_agents(executor: ThreadPoolExecutor) -> None:
            while ready_queue and len(active_futures) < max(1, self.config.max_workers):
                selected_index: int | None = None
                for index, (_, _, agent_name) in enumerate(ready_queue):
                    if active_agent_counts.get(agent_name, 0) < self._agent_limit(agent_name):
                        selected_index = index
                        break
                if selected_index is None:
                    break
                document, frontier_index, agent_name = ready_queue[selected_index]
                del ready_queue[selected_index]
                future = executor.submit(document.runner._execute_agent, agent_name, frontier_index)
                active_futures[future] = (document, frontier_index, agent_name)
                active_agent_counts[agent_name] = active_agent_counts.get(agent_name, 0) + 1

        def admit_new_documents() -> bool:
            admitted = False
            for document in self._drain_pending_admissions():
                admitted = True
                pending_frontiers.setdefault(document.run_name, 0)
                enqueue_ready_frontier(document)
            return admitted

        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as executor:
            admit_new_documents()
            self._write_telemetry(
                batch_started_at=batch_started_at,
                pending_frontiers=pending_frontiers,
                active_agent_counts=active_agent_counts,
                active_futures=active_futures,
                ready_queue=ready_queue,
                completed_records=completed_records,
                status="running",
            )
            submit_ready_agents(executor)
            self._write_telemetry(
                batch_started_at=batch_started_at,
                pending_frontiers=pending_frontiers,
                active_agent_counts=active_agent_counts,
                active_futures=active_futures,
                ready_queue=ready_queue,
                completed_records=completed_records,
                status="running",
            )

            while True:
                if not active_futures and ready_queue:
                    raise RuntimeError(
                        "Ready agents remain queued, but none can be scheduled. "
                        "Check `agent_concurrency_limits` for impossible settings."
                    )
                if not active_futures and not ready_queue:
                    admitted = admit_new_documents()
                    if admitted:
                        submit_ready_agents(executor)
                        self._write_telemetry(
                            batch_started_at=batch_started_at,
                            pending_frontiers=pending_frontiers,
                            active_agent_counts=active_agent_counts,
                            active_futures=active_futures,
                            ready_queue=ready_queue,
                            completed_records=completed_records,
                            status="running",
                        )
                        continue
                    if self._admission_is_closed():
                        break
                    self._write_telemetry(
                        batch_started_at=batch_started_at,
                        pending_frontiers=pending_frontiers,
                        active_agent_counts=active_agent_counts,
                        active_futures=active_futures,
                        ready_queue=ready_queue,
                        completed_records=completed_records,
                        status="waiting_for_documents",
                    )
                    self._wait_for_more_work()
                    continue
                done, _ = wait(
                    tuple(active_futures.keys()),
                    timeout=max(0.05, float(self.config.idle_poll_seconds)),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if admit_new_documents():
                        submit_ready_agents(executor)
                    self._write_telemetry(
                        batch_started_at=batch_started_at,
                        pending_frontiers=pending_frontiers,
                        active_agent_counts=active_agent_counts,
                        active_futures=active_futures,
                        ready_queue=ready_queue,
                        completed_records=completed_records,
                        status="running",
                    )
                    continue
                for future in done:
                    document, frontier_index, agent_name = active_futures.pop(future)
                    active_agent_counts[agent_name] -= 1
                    if active_agent_counts[agent_name] == 0:
                        del active_agent_counts[agent_name]
                    record = future.result()
                    completed_records.append(
                        DemocritusBatchRecord(
                            run_name=document.run_name,
                            pdf_path=str(document.pdf_path),
                            agent_record=record,
                        )
                    )
                    frontier_key = (document.run_name, frontier_index)
                    frontier_completion_counts[frontier_key] = frontier_completion_counts.get(frontier_key, 0) + 1
                    run_name = document.run_name
                    document = self._document_by_run_name(run_name)
                    frontier_size = len(document.plan[frontier_index])
                    if frontier_completion_counts[frontier_key] == frontier_size:
                        pending_frontiers[run_name] = frontier_index + 1
                        enqueue_ready_frontier(document)
                admit_new_documents()
                submit_ready_agents(executor)
                self._write_telemetry(
                    batch_started_at=batch_started_at,
                    pending_frontiers=pending_frontiers,
                    active_agent_counts=active_agent_counts,
                    active_futures=active_futures,
                    ready_queue=ready_queue,
                    completed_records=completed_records,
                    status="running",
                )

        ordered = tuple(
            sorted(
                completed_records,
                key=lambda record: (
                    record.run_name,
                    record.agent_record.frontier_index,
                    record.agent_record.agent_name,
                ),
            )
        )
        self._write_summary(ordered)
        csql_bundle = self._build_csql_bundle(ordered)
        corpus_synthesis = self._build_corpus_synthesis(csql_bundle) if self.config.enable_corpus_synthesis else None
        self._write_telemetry(
            batch_started_at=batch_started_at,
            pending_frontiers=pending_frontiers,
            active_agent_counts={},
            active_futures={},
            ready_queue=deque(),
            completed_records=list(ordered),
            status="complete",
        )
        return DemocritusBatchRunResult(
            records=ordered,
            csql_bundle=csql_bundle,
            corpus_synthesis=corpus_synthesis,
        )

    def _write_summary(self, records: tuple[DemocritusBatchRecord, ...]) -> None:
        self.summary_path.write_text(
            json.dumps(
                [
                    {
                        "run_name": record.run_name,
                        "pdf_path": record.pdf_path,
                        "agent_record": asdict(record.agent_record),
                    }
                    for record in records
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _build_csql_bundle(
        self,
        records: tuple[DemocritusBatchRecord, ...],
    ) -> BatchCSQLBundleResult | None:
        triples_by_run: dict[str, Path] = {}
        pdf_by_run: dict[str, str] = {}
        for record in records:
            pdf_by_run.setdefault(record.run_name, record.pdf_path)
            if record.agent_record.agent_name != "triple_extraction_agent":
                continue
            if not record.agent_record.outputs:
                continue
            triples_path = Path(record.agent_record.outputs[0])
            if triples_path.exists():
                triples_by_run[record.run_name] = triples_path
        if not triples_by_run:
            return None
        bundle_records = [
            {
                "run_name": run_name,
                "pdf_path": pdf_by_run.get(run_name, ""),
                "triples_path": str(triples_path),
            }
            for run_name, triples_path in sorted(triples_by_run.items())
        ]
        return build_batch_csql_bundle(
            batch_outdir=self.config.outdir,
            records=bundle_records,
            pdf_dir=self.config.pdf_dir,
        )

    def _build_corpus_synthesis(
        self,
        csql_bundle: BatchCSQLBundleResult | None,
    ) -> DemocritusCorpusSynthesisResult | None:
        if csql_bundle is None:
            return None
        return build_democritus_corpus_synthesis(
            query=self.config.request_query or "Democritus corpus synthesis",
            batch_outdir=self.config.outdir,
            csql_sqlite_path=csql_bundle.sqlite_path,
        )

    def _build_telemetry_snapshot(
        self,
        *,
        batch_started_at: float,
        pending_frontiers: dict[str, int],
        active_agent_counts: dict[str, int],
        active_futures: dict,
        ready_queue: deque[tuple[DemocritusBatchDocument, int, str]],
        completed_records: list[DemocritusBatchRecord],
        status: str,
    ) -> dict[str, object]:
        now = time.time()
        started_at_local = dt.datetime.fromtimestamp(batch_started_at).astimezone()
        documents = self._documents_snapshot()
        completed_by_run: dict[str, list[DemocritusBatchRecord]] = {}
        for record in completed_records:
            completed_by_run.setdefault(record.run_name, []).append(record)

        active_by_run: dict[str, list[dict[str, object]]] = {}
        for document, frontier_index, agent_name in active_futures.values():
            active_by_run.setdefault(document.run_name, []).append(
                {"agent_name": agent_name, "frontier_index": frontier_index}
            )

        queued_by_run: dict[str, list[dict[str, object]]] = {}
        for document, frontier_index, agent_name in ready_queue:
            queued_by_run.setdefault(document.run_name, []).append(
                {"agent_name": agent_name, "frontier_index": frontier_index}
            )

        documents_payload: list[dict[str, object]] = []
        for document in documents:
            completed = completed_by_run.get(document.run_name, [])
            active = active_by_run.get(document.run_name, [])
            queued = queued_by_run.get(document.run_name, [])
            next_frontier_index = pending_frontiers.get(document.run_name, 0)
            total_agents = sum(len(frontier) for frontier in document.plan)
            status_label = "pending"
            if len(completed) == total_agents:
                status_label = "complete"
            elif active:
                status_label = "running"
            elif queued:
                status_label = "queued"
            elif completed:
                status_label = "waiting"

            last_completed = None
            if completed:
                latest = max(completed, key=lambda item: item.agent_record.ended_at)
                last_completed = {
                    "agent_name": latest.agent_record.agent_name,
                    "frontier_index": latest.agent_record.frontier_index,
                    "duration_seconds": round(
                        latest.agent_record.ended_at - latest.agent_record.started_at,
                        3,
                    ),
                }

            documents_payload.append(
                {
                    "run_name": document.run_name,
                    "pdf_path": str(document.pdf_path),
                    "outdir": str(document.outdir),
                    "status": status_label,
                    "next_frontier_index": next_frontier_index,
                    "completed_agents": len(completed),
                    "total_agents": total_agents,
                    "active_agents": active,
                    "queued_agents": queued,
                    "last_completed": last_completed,
                }
            )

        agent_metrics: dict[str, dict[str, object]] = {}
        total_completed_work_seconds = 0.0
        for record in completed_records:
            metrics = agent_metrics.setdefault(
                record.agent_record.agent_name,
                {
                    "completed": 0,
                    "total_duration_seconds": 0.0,
                    "max_duration_seconds": 0.0,
                },
            )
            duration = record.agent_record.ended_at - record.agent_record.started_at
            total_completed_work_seconds += duration
            metrics["completed"] += 1
            metrics["total_duration_seconds"] += duration
            metrics["max_duration_seconds"] = max(metrics["max_duration_seconds"], duration)

        for agent_name, metrics in agent_metrics.items():
            completed = int(metrics["completed"])
            total = float(metrics["total_duration_seconds"])
            metrics["avg_duration_seconds"] = round(total / completed, 3) if completed else 0.0
            metrics["total_duration_seconds"] = round(total, 3)
            metrics["max_duration_seconds"] = round(float(metrics["max_duration_seconds"]), 3)
            metrics["active"] = active_agent_counts.get(agent_name, 0)
            metrics["limit"] = self._agent_limit(agent_name)

        for agent_name, active_count in active_agent_counts.items():
            agent_metrics.setdefault(
                agent_name,
                {
                    "completed": 0,
                    "total_duration_seconds": 0.0,
                    "max_duration_seconds": 0.0,
                    "avg_duration_seconds": 0.0,
                    "active": active_count,
                    "limit": self._agent_limit(agent_name),
                },
            )

        queue_counts: dict[str, int] = {}
        for _, _, agent_name in ready_queue:
            queue_counts[agent_name] = queue_counts.get(agent_name, 0) + 1

        all_agent_names = {
            agent.name
            for document in documents
            for frontier in document.plan
            for agent in frontier
        }
        global_avg_duration = (
            total_completed_work_seconds / len(completed_records) if completed_records else 0.0
        )
        remaining_work_seconds = 0.0
        remaining_agents = 0
        for document in documents:
            completed_names = {record.agent_record.agent_name for record in completed_by_run.get(document.run_name, [])}
            active_names = {item["agent_name"] for item in active_by_run.get(document.run_name, [])}
            queued_names = {item["agent_name"] for item in queued_by_run.get(document.run_name, [])}
            seen_names = completed_names | active_names | queued_names
            for frontier in document.plan:
                for agent in frontier:
                    if agent.name in seen_names:
                        continue
                    metrics = agent_metrics.get(agent.name)
                    estimated_duration = global_avg_duration
                    if metrics and metrics.get("completed", 0):
                        estimated_duration = float(metrics["avg_duration_seconds"])
                    remaining_work_seconds += estimated_duration
                    remaining_agents += 1
            for item in active_by_run.get(document.run_name, []):
                agent_name = item["agent_name"]
                metrics = agent_metrics.get(agent_name)
                estimated_duration = global_avg_duration
                if metrics and metrics.get("completed", 0):
                    estimated_duration = float(metrics["avg_duration_seconds"])
                remaining_work_seconds += max(estimated_duration * 0.5, 0.0)

        effective_parallelism = 1.0
        elapsed_seconds = max(now - batch_started_at, 0.001)
        if total_completed_work_seconds > 0:
            effective_parallelism = max(
                1.0,
                min(float(self.config.max_workers), total_completed_work_seconds / elapsed_seconds),
            )
        eta_seconds = remaining_work_seconds / effective_parallelism if remaining_work_seconds > 0 else 0.0
        slowest_stages = [
            {
                "agent_name": agent_name,
                "avg_duration_seconds": metrics["avg_duration_seconds"],
                "max_duration_seconds": metrics["max_duration_seconds"],
                "total_duration_seconds": metrics["total_duration_seconds"],
                "completed": metrics["completed"],
            }
            for agent_name, metrics in agent_metrics.items()
            if int(metrics.get("completed", 0)) > 0
        ]
        slowest_stages.sort(
            key=lambda item: (
                float(item["avg_duration_seconds"]),
                float(item["max_duration_seconds"]),
                float(item["total_duration_seconds"]),
            ),
            reverse=True,
        )
        slowest_stages = slowest_stages[:5]

        return {
            "status": status,
            "updated_at_epoch": now,
            "started_at_epoch": round(batch_started_at, 6),
            "started_at_local": started_at_local.isoformat(timespec="seconds"),
            "elapsed_seconds": round(now - batch_started_at, 3),
            "elapsed_human": _format_duration(now - batch_started_at),
            "config": {
                "pdf_dir": str(self.config.pdf_dir),
                "outdir": str(self.config.outdir),
                "max_docs": self.config.max_docs,
                "max_workers": self.config.max_workers,
                "include_phase2": self.config.include_phase2,
                "root_topic_strategy": self.config.root_topic_strategy,
                "discover_existing_documents": self.config.discover_existing_documents,
                "allow_incremental_admission": self.config.allow_incremental_admission,
                "admission_closed": self._admission_is_closed(),
                "idle_poll_seconds": self.config.idle_poll_seconds,
                "agent_concurrency_limits": {
                    agent_name: limit for agent_name, limit in self.config.agent_concurrency_limits
                },
            },
            "summary": {
                "n_documents": len(documents),
                "n_completed_records": len(completed_records),
                "n_active_agents": len(active_futures),
                "n_queued_agents": len(ready_queue),
            },
            "timing": {
                "effective_parallelism": round(effective_parallelism, 3),
                "remaining_agents_estimate": remaining_agents,
                "remaining_work_seconds_estimate": round(remaining_work_seconds, 3),
                "remaining_work_human": _format_duration(remaining_work_seconds),
                "eta_seconds": round(eta_seconds, 3),
                "eta_human": _format_duration(eta_seconds),
                "eta_ready": bool(completed_records),
                "total_completed_work_seconds": round(total_completed_work_seconds, 3),
                "total_completed_work_human": _format_duration(total_completed_work_seconds),
            },
            "active_agent_counts": dict(sorted(active_agent_counts.items())),
            "queued_agent_counts": dict(sorted(queue_counts.items())),
            "agent_metrics": dict(sorted(agent_metrics.items())),
            "slowest_stages": slowest_stages,
            "documents": documents_payload,
        }

    def _render_dashboard_html(self, snapshot: dict[str, object]) -> str:
        summary = snapshot["summary"]
        config = snapshot["config"]
        documents = snapshot["documents"]
        agent_metrics = snapshot["agent_metrics"]
        active_counts = snapshot["active_agent_counts"]
        queued_counts = snapshot["queued_agent_counts"]
        timing = snapshot["timing"]
        slowest_stages = snapshot["slowest_stages"]

        def esc(value: object) -> str:
            return html.escape(str(value))

        def render_kv_table(rows: list[tuple[str, object]]) -> str:
            body = "".join(
                f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in rows
            )
            return f"<table>{body}</table>"

        document_rows = []
        for document in documents:
            active_agents = ", ".join(
                item["agent_name"] for item in document["active_agents"]
            ) or "-"
            queued_agents = ", ".join(
                item["agent_name"] for item in document["queued_agents"]
            ) or "-"
            progress_text = f"{document['completed_agents']}/{document['total_agents']}"
            last_completed = document["last_completed"]
            last_completed_text = "-"
            if last_completed:
                last_completed_text = (
                    f"{last_completed['agent_name']} "
                    f"({last_completed['duration_seconds']}s)"
                )
            document_rows.append(
                "<tr>"
                f"<td>{esc(document['run_name'])}</td>"
                f"<td>{esc(document['status'])}</td>"
                f"<td>{esc(progress_text)}</td>"
                f"<td>{esc(document['next_frontier_index'])}</td>"
                f"<td>{esc(active_agents)}</td>"
                f"<td>{esc(queued_agents)}</td>"
                f"<td>{esc(last_completed_text)}</td>"
                "</tr>"
            )
        documents_table = (
            "<table><thead><tr>"
            "<th>Run</th><th>Status</th><th>Progress</th><th>Next Frontier</th>"
            "<th>Active Agents</th><th>Queued Agents</th><th>Last Completed</th>"
            "</tr></thead><tbody>"
            + "".join(document_rows)
            + "</tbody></table>"
        )

        metric_rows = []
        for agent_name, metrics in agent_metrics.items():
            metric_rows.append(
                "<tr>"
                f"<td>{esc(agent_name)}</td>"
                f"<td>{esc(metrics['active'])}</td>"
                f"<td>{esc(metrics['limit'])}</td>"
                f"<td>{esc(queued_counts.get(agent_name, 0))}</td>"
                f"<td>{esc(metrics['completed'])}</td>"
                f"<td>{esc(metrics['avg_duration_seconds'])}</td>"
                f"<td>{esc(metrics['max_duration_seconds'])}</td>"
                f"<td>{esc(metrics['total_duration_seconds'])}</td>"
                "</tr>"
            )
        metrics_table = (
            "<table><thead><tr>"
            "<th>Agent</th><th>Active</th><th>Limit</th><th>Queued</th>"
            "<th>Completed</th><th>Avg Sec</th><th>Max Sec</th><th>Total Sec</th>"
            "</tr></thead><tbody>"
            + "".join(metric_rows)
            + "</tbody></table>"
        )
        slowest_stage_rows = []
        for stage in slowest_stages:
            slowest_stage_rows.append(
                "<tr>"
                f"<td>{esc(stage['agent_name'])}</td>"
                f"<td>{esc(stage['completed'])}</td>"
                f"<td>{esc(stage['avg_duration_seconds'])}</td>"
                f"<td>{esc(stage['max_duration_seconds'])}</td>"
                f"<td>{esc(stage['total_duration_seconds'])}</td>"
                "</tr>"
            )
        slowest_stages_table = (
            "<table><thead><tr>"
            "<th>Agent</th><th>Completed</th><th>Avg Sec</th><th>Max Sec</th><th>Total Sec</th>"
            "</tr></thead><tbody>"
            + ("".join(slowest_stage_rows) if slowest_stage_rows else "<tr><td colspan='5'>No completed stages yet.</td></tr>")
            + "</tbody></table>"
        )

        config_table = render_kv_table(
            [
                ("Status", snapshot["status"]),
                ("Started At", snapshot["started_at_local"]),
                ("Elapsed Wall Clock", snapshot["elapsed_human"]),
                ("Elapsed Seconds", snapshot["elapsed_seconds"]),
                ("Updated At Epoch", snapshot["updated_at_epoch"]),
                ("PDF Dir", config["pdf_dir"]),
                ("Outdir", config["outdir"]),
                ("Max Docs", config["max_docs"]),
                ("Max Workers", config["max_workers"]),
                ("Phase 2 Enabled", config["include_phase2"]),
                ("Root Topic Strategy", config["root_topic_strategy"]),
                ("Incremental Admission", config["allow_incremental_admission"]),
                ("Admission Closed", config["admission_closed"]),
                ("Idle Poll Seconds", config["idle_poll_seconds"]),
            ]
        )
        summary_table = render_kv_table(
            [
                ("Documents", summary["n_documents"]),
                ("Completed Agent Records", summary["n_completed_records"]),
                ("Active Agents", summary["n_active_agents"]),
                ("Queued Agents", summary["n_queued_agents"]),
                ("Active Counts", json.dumps(active_counts)),
                ("Queued Counts", json.dumps(queued_counts)),
            ]
        )
        timing_table = render_kv_table(
            [
                ("ETA", timing["eta_human"] if timing["eta_ready"] else "Learning from first completions..."),
                ("Remaining Agents", timing["remaining_agents_estimate"]),
                ("Remaining Work", timing["remaining_work_human"]),
                ("Observed Parallelism", timing["effective_parallelism"]),
                ("Completed Work", timing["total_completed_work_human"]),
            ]
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <title>Democritus Batch Dashboard</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #12212f;
      background: #f6f8fb;
    }}
    h1, h2 {{
      margin-bottom: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #eef2f7;
    }}
    .note {{
      color: #52606d;
      font-size: 13px;
      margin-bottom: 16px;
    }}
    code {{
      background: #eef2f7;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>Democritus Batch Dashboard</h1>
  <p class="note">Auto-refreshes every 5 seconds. JSON source: <code>{esc(self.telemetry_path)}</code></p>
  <div class="grid">
    <div class="card">
      <h2>Run Summary</h2>
      {summary_table}
    </div>
    <div class="card">
      <h2>Configuration</h2>
      {config_table}
    </div>
    <div class="card">
      <h2>Timing Forecast</h2>
      {timing_table}
    </div>
  </div>
  <div class="card">
    <h2>Agent Metrics</h2>
    {metrics_table}
  </div>
  <div class="card" style="margin-top: 24px;">
    <h2>Slowest Stages</h2>
    {slowest_stages_table}
  </div>
  <div class="card" style="margin-top: 24px;">
    <h2>Documents</h2>
    {documents_table}
  </div>
</body>
</html>
"""

    def _document_gui_payload(
        self,
        document: DemocritusBatchDocument,
        *,
        document_snapshot: dict[str, object] | None,
    ) -> dict[str, object]:
        run_dir = document.outdir
        root_topics = _read_lines(run_dir / "configs" / "root_topics.txt")[:8]
        triples = _read_jsonl_rows(run_dir / "relational_triples.jsonl")
        top_triples = [
            {
                "subj": str(row.get("subj") or ""),
                "rel": str(row.get("rel") or ""),
                "obj": str(row.get("obj") or ""),
                "statement": str(row.get("statement") or ""),
                "domain": str(row.get("domain") or row.get("topic") or ""),
            }
            for row in triples[:4]
        ]
        score_rows = _read_csv_rows(run_dir / "sweep" / "scores.csv")
        score_rows.sort(
            key=lambda row: (
                _safe_float(row.get("score")),
                _safe_float(row.get("coupling")),
                _safe_int(row.get("n_edges")),
            ),
            reverse=True,
        )
        top_lcms = [
            {
                "focus": str(row.get("focus") or ""),
                "score": round(_safe_float(row.get("score")), 3),
                "coupling": round(_safe_float(row.get("coupling")), 3),
                "n_nodes": _safe_int(row.get("n_nodes")),
                "n_edges": _safe_int(row.get("n_edges")),
            }
            for row in score_rows[:3]
        ]
        summary_path = run_dir / "reports" / f"{document.run_name}_executive_summary.md"
        credibility_path = run_dir / "reports" / f"{document.run_name}_credibility_report.md"
        manifold_path = run_dir / "viz" / "relational_manifold_2d.png"
        summary_viewer_path, credibility_viewer_path, manifold_viewer_path = self._write_artifact_viewers(
            document,
            summary_path=summary_path,
            credibility_path=credibility_path,
            manifold_path=manifold_path,
            root_topics=root_topics,
            top_triples=top_triples,
            top_lcms=top_lcms,
        )
        bundled_pdf_path = run_dir / "input.pdf"
        pdf_target_path = bundled_pdf_path if bundled_pdf_path.exists() else document.pdf_path
        return {
            "run_name": document.run_name,
            "title": document.pdf_path.stem.replace("_", " "),
            "pdf_name": document.pdf_path.name,
            "pdf_href": self._bundle_relative_href(pdf_target_path),
            "status": str((document_snapshot or {}).get("status") or "pending"),
            "progress_text": (
                f"{(document_snapshot or {}).get('completed_agents', 0)}/"
                f"{(document_snapshot or {}).get('total_agents', 0)} agents"
            ),
            "summary_excerpt": _markdown_excerpt(summary_path) or _markdown_excerpt(credibility_path),
            "root_topics": root_topics,
            "top_triples": top_triples,
            "top_lcms": top_lcms,
            "triple_count": len(triples),
            "lcm_count": len(score_rows),
            "summary_href": self._bundle_relative_href(summary_viewer_path),
            "credibility_href": self._bundle_relative_href(credibility_viewer_path),
            "manifold_href": self._bundle_relative_href(manifold_viewer_path),
        }

    def _bundle_relative_href(self, target_path: Path) -> str:
        if not target_path.exists():
            return ""
        return os.path.relpath(target_path.resolve(), start=self.gui_path.parent.resolve())

    def _write_artifact_viewers(
        self,
        document: DemocritusBatchDocument,
        *,
        summary_path: Path,
        credibility_path: Path,
        manifold_path: Path,
        root_topics: list[str],
        top_triples: list[dict[str, object]],
        top_lcms: list[dict[str, object]],
    ) -> tuple[Path, Path, Path]:
        summary_viewer_path = summary_path.with_suffix(".html")
        credibility_viewer_path = credibility_path.with_suffix(".html")
        manifold_viewer_path = manifold_path.with_name("relational_manifold_viewer.html")
        summary_viewer_path.parent.mkdir(parents=True, exist_ok=True)
        manifold_viewer_path.parent.mkdir(parents=True, exist_ok=True)
        summary_viewer_path.write_text(
            self._render_report_viewer_html(
                title=_pretty_document_title(document.pdf_path.stem),
                artifact_label="Executive Summary",
                markdown_path=summary_path,
                tiers_only=True,
                strip_leading_title=True,
            ),
            encoding="utf-8",
        )
        credibility_viewer_path.write_text(
            self._render_report_viewer_html(
                title=_pretty_document_title(document.pdf_path.stem),
                artifact_label="Credibility Report",
                markdown_path=credibility_path,
                strip_leading_title=True,
            ),
            encoding="utf-8",
        )
        manifold_viewer_path.write_text(
            self._render_manifold_viewer_html(
                title=document.pdf_path.stem.replace("_", " "),
                manifold_path=manifold_path,
                root_topics=root_topics,
                top_triples=top_triples,
                top_lcms=top_lcms,
            ),
            encoding="utf-8",
        )
        return summary_viewer_path, credibility_viewer_path, manifold_viewer_path

    def _render_report_viewer_html(
        self,
        *,
        title: str,
        artifact_label: str,
        markdown_path: Path,
        tiers_only: bool = False,
        strip_leading_title: bool = False,
    ) -> str:
        def esc(value: object) -> str:
            return html.escape(str(value))

        body_html = _markdown_document_html(
            markdown_path,
            tiers_only=tiers_only,
            strip_leading_title=strip_leading_title,
        )
        hero_subtitle = f"<p>{esc(title)}</p>" if title.strip() else ""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(artifact_label)} · {esc(title)}</title>
  <style>
    :root {{
      --ink: #162433;
      --muted: #5d7083;
      --paper: #f7f0e6;
      --card: rgba(255, 252, 246, 0.97);
      --line: #d8ccbc;
      --accent: #0f766e;
      --shadow: 0 18px 44px rgba(22, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 26%),
        linear-gradient(180deg, #fbf6ee 0%, #efe5d7 100%);
    }}
    .shell {{ max-width: 980px; margin: 0 auto; padding: 28px 18px 44px; }}
    .hero, .article {{
      background: var(--card);
      border: 1px solid rgba(216, 204, 188, 0.96);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 26px; margin-bottom: 20px; }}
    .eyebrow {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(32px, 4vw, 52px); line-height: 0.98; }}
    .hero p {{ margin: 0; font-size: 18px; line-height: 1.7; color: var(--muted); }}
    .article {{ padding: 30px; }}
    .article h1, .article h2, .article h3 {{
      line-height: 1.12;
      margin-top: 28px;
      margin-bottom: 12px;
    }}
    .article h1:first-child, .article h2:first-child, .article h3:first-child {{ margin-top: 0; }}
    .article p, .article li, .article blockquote {{
      font-size: 22px;
      line-height: 1.75;
    }}
    .article ul {{ padding-left: 24px; }}
    .article blockquote {{
      margin: 18px 0;
      padding: 16px 18px;
      border-left: 4px solid var(--accent);
      background: rgba(15,118,110,0.06);
    }}
    .article pre {{
      overflow: auto;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
      font-size: 16px;
      line-height: 1.6;
      white-space: pre-wrap;
    }}
    .article code {{
      background: rgba(0,0,0,0.05);
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 0.92em;
    }}
    .article a {{ color: var(--accent); text-decoration: none; }}
    .article a:hover {{ text-decoration: underline; }}
    @media (max-width: 720px) {{
      .article p, .article li, .article blockquote {{ font-size: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Democritus Reader</div>
      <h1>{esc(artifact_label)}</h1>
      {hero_subtitle}
    </section>
    <article class="article">
      {body_html}
    </article>
  </div>
</body>
</html>"""

    def _render_manifold_viewer_html(
        self,
        *,
        title: str,
        manifold_path: Path,
        root_topics: list[str],
        top_triples: list[dict[str, object]],
        top_lcms: list[dict[str, object]],
    ) -> str:
        def esc(value: object) -> str:
            return html.escape(str(value))

        topic_markup = "".join(f'<span class="chip">{esc(topic)}</span>' for topic in root_topics[:10])
        triple_markup = "".join(
            '<article class="mini-card">'
            f'<div class="mini-title">{esc(item.get("subj") or "")} <span class="arrow">→</span> {esc(item.get("obj") or "")}</div>'
            f'<p>{esc(item.get("statement") or item.get("rel") or "")}</p>'
            "</article>"
            for item in top_triples[:4]
        )
        lcm_markup = "".join(
            '<article class="lcm-row">'
            f'<div><strong>{esc(item.get("focus") or "local causal model")}</strong></div>'
            f'<div class="lcm-meta">score {esc(item.get("score") or 0)} · coupling {esc(item.get("coupling") or 0)} · {esc(item.get("n_nodes") or 0)} nodes · {esc(item.get("n_edges") or 0)} edges</div>'
            "</article>"
            for item in top_lcms[:4]
        )
        image_markup = (
            f'<img src="{esc(manifold_path.name)}" alt="{esc(title)} relational manifold" loading="eager">'
            if manifold_path.exists()
            else '<div class="empty">The manifold image is not available yet.</div>'
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manifold View · {esc(title)}</title>
  <style>
    :root {{
      --ink: #142433;
      --muted: #5a7184;
      --paper: #f5efe5;
      --card: rgba(255, 252, 246, 0.97);
      --line: #d8ccbc;
      --accent: #0f766e;
      --shadow: 0 18px 44px rgba(20, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 28%),
        linear-gradient(180deg, #fbf6ee 0%, #eee4d6 100%);
    }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 44px; }}
    .hero, .viewer, .panel {{
      background: var(--card);
      border: 1px solid rgba(216, 204, 188, 0.96);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 26px; margin-bottom: 20px; }}
    .eyebrow {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.16em; color: var(--muted); }}
    h1 {{ margin: 10px 0 8px; font-size: clamp(32px, 4vw, 54px); line-height: 0.98; }}
    .hero p {{ margin: 0; color: var(--muted); font-size: 17px; line-height: 1.65; max-width: 920px; }}
    .layout {{ display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 18px; }}
    .viewer {{ padding: 18px; }}
    .viewer img {{ width: 100%; border-radius: 22px; display: block; background: white; }}
    .panel {{ padding: 22px; }}
    .section-label {{
      margin: 0 0 10px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      background: #edf2f7;
      color: var(--ink);
    }}
    .mini-card, .lcm-row {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      margin-bottom: 12px;
    }}
    .mini-title {{ font-size: 18px; line-height: 1.35; }}
    .mini-card p, .lcm-meta {{
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Democritus Viewer</div>
      <h1>Relational Manifold</h1>
      <p>{esc(title)}. This viewer keeps the manifold large enough for presentation use and adds readable label context from the recovered root topics, triples, and top local causal models.</p>
    </section>
    <div class="layout">
      <section class="viewer">
        {image_markup}
      </section>
      <aside class="panel">
        <div class="section-label">Topic Labels</div>
        <div class="chip-row">{topic_markup or '<span class="chip">Root topics pending</span>'}</div>
        <div class="section-label">Key Evidence Labels</div>
        {triple_markup or '<div class="empty">Relational triples will appear here after extraction.</div>'}
        <div class="section-label">Top Local Causal Models</div>
        {lcm_markup or '<div class="empty">LCM scores will appear here after sweep scoring.</div>'}
      </aside>
    </div>
  </div>
</body>
</html>"""

    def _render_gui_html(self, snapshot: dict[str, object]) -> str:
        def esc(value: object) -> str:
            return html.escape(str(value))

        documents_by_run = {
            str(item.get("run_name")): item
            for item in list(snapshot.get("documents") or [])
        }
        cards = [
            self._document_gui_payload(
                document,
                document_snapshot=documents_by_run.get(document.run_name),
            )
            for document in self._documents_snapshot()
        ]
        completed_docs = sum(1 for card in cards if card["status"] == "complete")
        total_triples = sum(int(card["triple_count"]) for card in cards)
        total_lcms = sum(int(card["lcm_count"]) for card in cards)
        query_banner = self.config.request_query
        hero_title = query_banner or "Recovered Causal Structure Across the Current Batch"
        hero_note = (
            "Results aligned to the original Democritus request so you can judge how closely "
            "the recovered evidence, summaries, and local causal models match the query."
            if query_banner
            else "Recovered evidence, executive summaries, and local causal model candidates "
            "from the current Democritus document batch."
        )

        def tone_for_status(status: str) -> str:
            return {
                "complete": "success",
                "running": "mixed",
                "queued": "neutral",
                "waiting": "neutral",
                "pending": "neutral",
            }.get(status, "neutral")

        def chip(label: str, tone: str) -> str:
            return f'<span class="chip {tone}">{esc(label)}</span>'

        def metric(label: str, value: object) -> str:
            return (
                '<div class="metric">'
                f'<div class="metric-label">{esc(label)}</div>'
                f'<div class="metric-value">{esc(value)}</div>'
                "</div>"
            )

        document_cards: list[str] = []
        for card in cards:
            root_topic_markup = "".join(chip(topic, "neutral") for topic in card["root_topics"]) or chip("root topics pending", "neutral")
            triple_markup = "".join(
                '<article class="mini-card">'
                f'<div class="mini-kicker">{esc(item["domain"] or "causal claim")}</div>'
                f'<div class="mini-title">{esc(item["subj"])} <span class="arrow">→</span> {esc(item["obj"])}</div>'
                f'<p>{esc(item["statement"] or item["rel"])}</p>'
                "</article>"
                for item in card["top_triples"]
            ) or '<div class="empty">Relational triples will appear here after extraction.</div>'
            lcm_markup = "".join(
                '<article class="lcm-row">'
                f'<div><strong>{esc(item["focus"] or "local causal model")}</strong></div>'
                f'<div class="lcm-meta">score {esc(item["score"])} · coupling {esc(item["coupling"])} · {esc(item["n_nodes"])} nodes · {esc(item["n_edges"])} edges</div>'
                "</article>"
                for item in card["top_lcms"]
            ) or '<div class="empty">LCM scores will appear here after sweep scoring.</div>'
            links = []
            if card["pdf_href"]:
                links.append(f'<a href="{esc(card["pdf_href"])}" target="_blank" rel="noreferrer">Open PDF</a>')
            if card["summary_href"]:
                links.append(f'<a href="{esc(card["summary_href"])}" target="_blank" rel="noreferrer">Executive summary</a>')
            if card["credibility_href"]:
                links.append(f'<a href="{esc(card["credibility_href"])}" target="_blank" rel="noreferrer">Credibility report</a>')
            if card["manifold_href"]:
                links.append(f'<a href="{esc(card["manifold_href"])}" target="_blank" rel="noreferrer">Manifold view</a>')
            link_markup = " · ".join(links) if links else "Artifacts will appear as this run completes."
            status_chip = chip(str(card["status"]).replace("_", " "), tone_for_status(str(card["status"])))
            progress_chip = chip(str(card["progress_text"]), "neutral")
            triple_chip = chip(f'{card["triple_count"]} triples', "neutral")
            lcm_chip = chip(f'{card["lcm_count"]} LCMs', "neutral")
            document_cards.append(
                '<section class="doc-card">'
                '<div class="doc-header">'
                '<div>'
                f'<div class="doc-eyebrow">{esc(card["pdf_name"])}</div>'
                f'<h2>{esc(card["title"])}</h2>'
                "</div>"
                '<div class="doc-status">'
                f"{status_chip}"
                f"{progress_chip}"
                f"{triple_chip}"
                f"{lcm_chip}"
                "</div>"
                "</div>"
                f'<p class="summary">{esc(card["summary_excerpt"] or "Document summary is still being prepared. Once the report stage finishes, this card will surface the recovered argument structure.")}</p>'
                '<div class="section-label">Root Topics</div>'
                f'<div class="chip-row">{root_topic_markup}</div>'
                '<div class="two-up">'
                '<div>'
                '<div class="section-label">Evidence Preview</div>'
                f'<div class="mini-grid">{triple_markup}</div>'
                "</div>"
                '<div>'
                '<div class="section-label">Recovered Local Causal Models</div>'
                f'<div class="lcm-list">{lcm_markup}</div>'
                "</div>"
                "</div>"
                f'<div class="links">{link_markup}</div>'
                "</section>"
            )

        refresh_meta = '<meta http-equiv="refresh" content="5">' if snapshot.get("status") != "complete" else ""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {refresh_meta}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE Democritus GUI</title>
  <style>
    :root {{
      --ink: #142433;
      --muted: #5a7184;
      --paper: #f4efe7;
      --card: rgba(255, 252, 247, 0.96);
      --line: #d7cec2;
      --accent: #0f766e;
      --accent-soft: #d7f0eb;
      --mixed: #a16207;
      --mixed-soft: #fef3c7;
      --shadow: 0 18px 44px rgba(20, 36, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 28%),
        linear-gradient(180deg, #f8f3ea 0%, #efe6d7 100%);
    }}
    .shell {{ max-width: 1240px; margin: 0 auto; padding: 28px 18px 48px; }}
    .hero, .doc-card {{
      background: var(--card);
      border: 1px solid rgba(215, 206, 194, 0.95);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 26px; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 4vw, 52px); line-height: 0.98; }}
    .hero p {{ margin: 0; color: var(--muted); font-size: 17px; line-height: 1.6; max-width: 860px; }}
    .chip-row, .hero-meta, .doc-status {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .hero-meta {{ margin-top: 18px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      background: #edf2f7;
      border: 1px solid transparent;
    }}
    .chip.success {{ background: var(--accent-soft); color: var(--accent); }}
    .chip.mixed {{ background: var(--mixed-soft); color: var(--mixed); }}
    .chip.neutral {{ background: #edf2f7; color: var(--ink); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .metric {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .metric-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ margin-top: 8px; font-size: 30px; }}
    .docs {{ display: grid; gap: 18px; margin-top: 24px; }}
    .doc-card {{ padding: 22px; }}
    .doc-header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
    }}
    .doc-eyebrow {{ color: var(--muted); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }}
    h2 {{ margin: 6px 0 0; font-size: 28px; }}
    .summary {{ margin: 16px 0 18px; color: var(--muted); line-height: 1.6; font-size: 16px; }}
    .section-label {{
      margin: 18px 0 10px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-top: 8px;
    }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .mini-card, .lcm-row {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .mini-kicker {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .mini-title {{ margin-top: 8px; font-size: 16px; line-height: 1.35; }}
    .mini-card p, .lcm-meta {{ margin: 10px 0 0; color: var(--muted); line-height: 1.5; font-size: 14px; }}
    .lcm-list {{ display: grid; gap: 10px; }}
    .links {{ margin-top: 16px; color: var(--muted); font-size: 14px; }}
    .links a {{ color: var(--accent); text-decoration: none; }}
    .links a:hover {{ text-decoration: underline; }}
    .empty {{
      background: white;
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .arrow {{ color: var(--accent); }}
    @media (max-width: 980px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-up, .mini-grid {{ grid-template-columns: 1fr; }}
      .doc-header {{ flex-direction: column; }}
    }}
    @media (max-width: 680px) {{
      .shell {{ padding: 16px 12px 36px; }}
      .hero, .doc-card {{ border-radius: 18px; padding: 18px; }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Democritus GUI</div>
      <h1>{esc(hero_title)}</h1>
      <p>{esc(hero_note)}</p>
      <div class="hero-meta">
        {chip(str(snapshot.get("status", "unknown")).replace("_", " "), tone_for_status(str(snapshot.get("status", "unknown"))))}
        {chip(f"{completed_docs}/{len(cards)} complete", "neutral")}
        {chip(f"{total_triples} triples recovered", "neutral")}
        {chip(f"{total_lcms} scored LCMs", "neutral")}
      </div>
      <div class="metrics">
        {metric("Documents", len(cards))}
        {metric("Completed Agent Records", snapshot.get("summary", {}).get("n_completed_records", 0))}
        {metric("ETA", snapshot.get("timing", {}).get("eta_human", "n/a") if snapshot.get("timing", {}).get("eta_ready") else "warming up")}
        {metric("Observed Parallelism", snapshot.get("timing", {}).get("effective_parallelism", 1.0))}
      </div>
    </section>
    <section class="docs">
      {"".join(document_cards) if document_cards else '<div class="empty">No documents have been admitted to this batch yet.</div>'}
    </section>
  </div>
</body>
</html>"""

    def _write_telemetry(
        self,
        *,
        batch_started_at: float,
        pending_frontiers: dict[str, int],
        active_agent_counts: dict[str, int],
        active_futures: dict,
        ready_queue: deque[tuple[DemocritusBatchDocument, int, str]],
        completed_records: list[DemocritusBatchRecord],
        status: str,
    ) -> None:
        snapshot = self._build_telemetry_snapshot(
            batch_started_at=batch_started_at,
            pending_frontiers=pending_frontiers,
            active_agent_counts=active_agent_counts,
            active_futures=active_futures,
            ready_queue=ready_queue,
            completed_records=completed_records,
            status=status,
        )
        self.telemetry_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        self.dashboard_path.write_text(self._render_dashboard_html(snapshot), encoding="utf-8")
        self.gui_path.write_text(self._render_gui_html(snapshot), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic Democritus pipeline over a PDF directory.")
    parser.add_argument("--pdf-dir", default=str(_default_pdf_dir()))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--agent-limit",
        action="append",
        default=[],
        help="Per-agent concurrency limit in the form agent_name=limit. May be repeated.",
    )
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--root-topic-strategy", default="v0_openai", choices=["v0_openai", "heuristic"])
    parser.add_argument("--no-auto-topics", action="store_true")
    parser.add_argument("--intra-document-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_agent_limit_args(values: list[str]) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --agent-limit value {value!r}; expected agent_name=limit.")
        agent_name, raw_limit = value.split("=", 1)
        agent_name = agent_name.strip()
        if not agent_name:
            raise ValueError(f"Invalid --agent-limit value {value!r}; missing agent name.")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"Invalid --agent-limit value {value!r}; limit must be an integer.") from exc
        parsed.append((agent_name, limit))
    return tuple(parsed)


def main() -> None:
    args = _parse_args()
    runner = DemocritusBatchAgenticRunner(
        DemocritusBatchConfig(
            pdf_dir=Path(args.pdf_dir),
            outdir=Path(args.outdir),
            max_docs=args.max_docs,
            max_workers=args.max_workers,
            agent_concurrency_limits=_parse_agent_limit_args(args.agent_limit),
            include_phase2=not args.skip_phase2,
            auto_topics_from_pdf=not args.no_auto_topics,
            root_topic_strategy=args.root_topic_strategy,
            intra_document_shards=args.intra_document_shards,
            dry_run=args.dry_run,
        )
    )
    records = runner.run()
    print(
        json.dumps(
            [
                {
                    "run_name": record.run_name,
                    "pdf_path": record.pdf_path,
                    "agent_name": record.agent_record.agent_name,
                    "frontier_index": record.agent_record.frontier_index,
                    "status": record.agent_record.status,
                }
                for record in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
