"""Local dashboard launcher for query-first BAFFLE workflows."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import re
import threading
import time
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

_ARTIFACT_REFRESH_SECONDS = 15
_ARTIFACT_REFRESH_MS = _ARTIFACT_REFRESH_SECONDS * 1000
_SESSION_REFRESH_SECONDS = 5
_SESSION_REFRESH_MS = _SESSION_REFRESH_SECONDS * 1000
_ARCHIVE_REFRESH_SECONDS = 20
_ARCHIVE_INDEX_FRESH_SECONDS = 12 * 60 * 60
_ARCHIVE_RECORD_FILENAME = "cliff_run_record.json"
_WORKER_RESULT_FILENAME = "cliff_worker_result.json"
_DEMOCRITUS_CHECKPOINT_HTML = "democritus_topic_checkpoint.html"
_DEMOCRITUS_CHECKPOINT_MANIFEST = "democritus_topic_checkpoint.json"
_DEMOCRITUS_CURATION_STATE = "user_curation.json"
_DEMOCRITUS_CURATED_MANIFEST = "selected_documents_manifest.json"
_DEMOCRITUS_CURATION_TELEMETRY = "user_curation_telemetry.jsonl"
_DEMOCRITUS_CURATION_SUMMARY = "user_curation_summary.json"
_COMPANY_SIMILARITY_CHECKPOINT_HTML = "company_similarity_checkpoint.html"
_COMPANY_SIMILARITY_CHECKPOINT_MANIFEST = "company_similarity_checkpoint.json"
_COMPANY_SIMILARITY_CURATION_STATE = "company_similarity_year_window.json"
_ROUTER_FEEDBACK_LOG = "router_feedback.jsonl"
_ROUTER_FEEDBACK_SUMMARY = "router_feedback_summary.json"
_SUPPORTED_ROUTE_NAMES = (
    "democritus",
    "basket_rocket_sec",
    "culinary_tour",
    "product_feedback",
    "company_similarity",
    "course_demo",
)


@dataclass(frozen=True)
class DashboardQueryLauncherConfig:
    """Configuration for a local dashboard query prompt."""

    title: str
    subtitle: str
    query_label: str
    query_placeholder: str
    submit_label: str
    waiting_message: str
    demo_queries: tuple[str, ...] = ()
    eyebrow: str = "FunctorFlow Dashboard Launch"
    artifact_path: Path | None = None
    session_mode: bool = False
    run_control_handler: Callable[[str, str], None] | None = None
    enable_execution_mode: bool = False
    default_execution_mode: str = "quick"
    archive_roots: tuple[Path, ...] = ()
    archive_max_runs: int = 120
    archive_cache_dir: Path | None = None


class DashboardQueryLauncher:
    """Serve a lightweight local dashboard form for one-shot or session-based querying."""

    def __init__(
        self,
        config: DashboardQueryLauncherConfig,
        *,
        browser_opener: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        self.browser_opener = browser_opener or (lambda url: webbrowser.open(url, new=1, autoraise=True))
        self._submitted_event = threading.Event()
        self._lock = threading.Lock()
        self._submitted_query = ""
        self._submitted_execution_mode = self._normalize_execution_mode(config.default_execution_mode)
        self._query_received = False
        self._artifact_path = config.artifact_path
        self._submission_counter = 0
        self._submission_queue: deque[tuple[str, str, str]] = deque()
        self._session_runs: list[dict[str, object]] = []
        self._session_runs_by_id: dict[str, dict[str, object]] = {}
        self._archived_runs: list[dict[str, object]] = []
        self._archived_runs_by_id: dict[str, dict[str, object]] = {}
        self._archive_last_scan_at = 0.0
        self._archive_cache_refresh_pending = False
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> "DashboardQueryLauncher":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._server_thread is not None:
            return
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._server.daemon_threads = True
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/"
        self._server_thread = threading.Thread(target=self._server.serve_forever, name="ff2-dashboard-query", daemon=True)
        self._server_thread.start()
        try:
            self.browser_opener(self.url)
        except Exception:
            pass

    @staticmethod
    def _normalize_execution_mode(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "deep":
            return "deep"
        if normalized == "interactive":
            return "interactive"
        return "quick"

    @staticmethod
    def _normalize_llm_token_budget(value: object) -> int | None:
        try:
            budget_tokens = int(str(value or "").strip())
        except (TypeError, ValueError):
            return None
        return budget_tokens if budget_tokens > 0 else None

    def _run_llm_token_budget(self, run_state: dict[str, object]) -> int | None:
        overrides = dict(run_state.get("submission_overrides") or {})
        return self._normalize_llm_token_budget(overrides.get("llm_token_budget"))

    @staticmethod
    def _route_research_profile(route_name: object) -> dict[str, str]:
        normalized = str(route_name or "").strip().lower()
        if normalized in {"basket_rocket_sec", "course_demo", "culinary_tour"}:
            return {
                "class_name": "quick-answer",
                "label": "Quick answer",
                "note": "Usually one of the faster CLIFF routes.",
            }
        if normalized == "product_feedback":
            return {
                "class_name": "longer-analysis",
                "label": "Longer analysis",
                "note": "Structured evidence and synthesis can take a few minutes.",
            }
        if normalized in {"democritus", "company_similarity"}:
            return {
                "class_name": "deep-research",
                "label": "Deep research",
                "note": "Even quick mode may take several minutes while CLIFF builds causal state.",
            }
        return {
            "class_name": "route-warming-up",
            "label": "Route warming up",
            "note": "CLIFF is still determining how much work this question requires.",
        }

    def wait_for_submission(self) -> str:
        self._submitted_event.wait()
        with self._lock:
            return self._submitted_query

    def wait_for_submission_payload(self) -> tuple[str, str]:
        self._submitted_event.wait()
        with self._lock:
            return self._submitted_query, self._submitted_execution_mode

    def wait_for_next_submission(self, timeout: float | None = None) -> tuple[str, str, str] | None:
        if not self._submitted_event.wait(timeout):
            return None
        with self._lock:
            if not self._submission_queue:
                self._submitted_event.clear()
                return None
            submission = self._submission_queue.popleft()
            if not self._submission_queue:
                self._submitted_event.clear()
            return submission

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def set_artifact_path(self, artifact_path: Path | None) -> None:
        with self._lock:
            self._artifact_path = artifact_path

    def submit_query(
        self,
        query: str,
        *,
        execution_mode: str | None = None,
        parent_run_id: str | None = None,
        submission_overrides: dict[str, object] | None = None,
        queued_note: str | None = None,
    ) -> str:
        normalized_query = " ".join(str(query).split()).strip()
        if not normalized_query:
            raise ValueError("Query must be non-empty.")
        normalized_mode = self._normalize_execution_mode(
            execution_mode if self.config.enable_execution_mode else self.config.default_execution_mode
        )
        with self._lock:
            if not self.config.session_mode:
                self._submitted_query = normalized_query
                self._submitted_execution_mode = normalized_mode
                self._query_received = True
                self._submitted_event.set()
                return "run-0001"
            self._submission_counter += 1
            run_id = f"run-{self._submission_counter:04d}"
            run_state = {
                "run_id": run_id,
                "query": normalized_query,
                "status": "queued",
                "mind_layer": "conscious",
                "route_name": "",
                "note": "Waiting for the router to start.",
                "artifact_path": None,
                "outdir": None,
                "execution_mode": normalized_mode,
                "parent_run_id": parent_run_id,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            if parent_run_id:
                run_state["note"] = queued_note or f"Queued as a follow-up to {parent_run_id}."
            elif queued_note:
                run_state["note"] = queued_note
            if submission_overrides:
                run_state["submission_overrides"] = dict(submission_overrides)
            self._submission_queue.append((run_id, normalized_query, normalized_mode))
            self._session_runs.insert(0, run_state)
            self._session_runs_by_id[run_id] = run_state
            self._query_received = True
            self._submitted_event.set()
            return run_id

    def submission_overrides_for_run(self, run_id: str) -> dict[str, object]:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None:
                return {}
            overrides = run_state.get("submission_overrides")
            return dict(overrides) if isinstance(overrides, dict) else {}

    def update_session_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        mind_layer: str | None = None,
        route_name: str | None = None,
        note: str | None = None,
        artifact_path: Path | None = None,
        outdir: Path | None = None,
    ) -> None:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None:
                return
            if status is not None:
                run_state["status"] = status
            if mind_layer is not None:
                run_state["mind_layer"] = mind_layer
            if route_name is not None:
                run_state["route_name"] = route_name
            if note is not None:
                run_state["note"] = note
            if artifact_path is not None:
                run_state["artifact_path"] = str(artifact_path)
            if outdir is not None:
                run_state["outdir"] = str(outdir)
            run_state["updated_at"] = time.time()
            self._persist_run_record(run_state)

    def _persist_run_record(self, run_state: dict[str, object]) -> None:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return
        try:
            outdir = Path(outdir_value).resolve()
        except Exception:
            return
        payload = {
            "run_id": str(run_state.get("run_id") or outdir.name),
            "query": str(run_state.get("query") or "").strip(),
            "status": str(run_state.get("status") or "").strip(),
            "route_name": str(run_state.get("route_name") or "").strip(),
            "artifact_path": str(run_state.get("artifact_path") or "").strip() or None,
            "outdir": str(outdir),
            "execution_mode": self._normalize_execution_mode(run_state.get("execution_mode")),
            "parent_run_id": str(run_state.get("parent_run_id") or "").strip() or None,
            "note": str(run_state.get("note") or "").strip(),
            "created_at": run_state.get("created_at"),
            "updated_at": run_state.get("updated_at"),
            "submission_overrides": dict(run_state.get("submission_overrides") or {}),
            "source": "dashboard_session",
        }
        self._write_json_file(outdir / _ARCHIVE_RECORD_FILENAME, payload)

    def _archive_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        seen: set[str] = set()
        for root in self.config.archive_roots:
            try:
                resolved = root.expanduser().resolve()
            except Exception:
                continue
            key = str(resolved)
            if key in seen or not resolved.exists():
                continue
            seen.add(key)
            roots.append(resolved)
        return tuple(roots)

    def _archive_cache_file(self, root: Path) -> Path | None:
        cache_dir = self.config.archive_cache_dir
        if cache_dir is None:
            return None
        try:
            resolved_cache_dir = cache_dir.expanduser().resolve()
        except Exception:
            return None
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
        return resolved_cache_dir / f"{digest}.json"

    def _load_cached_archive_entries(self, roots: tuple[Path, ...]) -> tuple[list[dict[str, object]], bool]:
        discovered: dict[str, dict[str, object]] = {}
        all_fresh = True
        limit = max(1, int(self.config.archive_max_runs))
        for root in roots:
            cache_file = self._archive_cache_file(root)
            if cache_file is None or not cache_file.exists():
                all_fresh = False
                continue
            payload = self._read_json_dict(cache_file)
            if not payload:
                all_fresh = False
                continue
            generated_at = float(payload.get("generated_at") or 0.0)
            if (time.time() - generated_at) > _ARCHIVE_INDEX_FRESH_SECONDS:
                all_fresh = False
            for item in list(payload.get("runs") or []):
                entry = dict(item) if isinstance(item, dict) else {}
                if not entry:
                    continue
                entry = self._normalize_archived_entry(entry)
                outdir_key = str(entry.get("outdir") or "")
                if not outdir_key or outdir_key in discovered:
                    continue
                if str(entry.get("run_id") or "") in self._session_runs_by_id:
                    continue
                discovered[outdir_key] = entry
                if len(discovered) >= limit:
                    break
            if len(discovered) >= limit:
                break
        cached = sorted(
            discovered.values(),
            key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
            reverse=True,
        )
        return cached, all_fresh and bool(cached)

    def _write_cached_archive_entries(self, root: Path, runs: list[dict[str, object]]) -> None:
        cache_file = self._archive_cache_file(root)
        if cache_file is None:
            return
        payload = {
            "root": str(root),
            "generated_at": time.time(),
            "runs": runs,
        }
        self._write_json_file(cache_file, payload)

    def _set_archived_runs(self, archived: list[dict[str, object]]) -> None:
        self._archived_runs = archived
        self._archived_runs_by_id = {str(item.get("run_id") or ""): item for item in archived if item.get("run_id")}

    def _rebase_archived_path(self, run_dir: Path, stored_path: object) -> Path | None:
        text = " ".join(str(stored_path or "").split()).strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return resolved
        parts = list(candidate.parts)
        for index, part in enumerate(parts):
            if part != run_dir.name:
                continue
            mapped = run_dir.parent.joinpath(*parts[index:])
            if mapped.exists():
                return mapped.resolve()
        direct = run_dir / candidate.name
        if candidate.name and direct.exists():
            return direct.resolve()
        return None

    def _preferred_archived_artifact_path(self, run_dir: Path, route_name: str) -> Path | None:
        route = " ".join(str(route_name or "").split()).strip().lower()
        candidates: list[Path] = []
        if route == "democritus":
            democritus_root = run_dir / "democritus" / "democritus_runs"
            candidates.extend(
                [
                    democritus_root / "corpus_synthesis" / "democritus_corpus_synthesis.html",
                    democritus_root / "democritus_gui.html",
                    democritus_root / "dashboard.html",
                    democritus_root / "democritus_query_clarification.html",
                    democritus_root / _DEMOCRITUS_CHECKPOINT_HTML,
                ]
            )
        elif route == "company_similarity":
            company_root = run_dir / "company_similarity"
            candidates.extend(
                [
                    company_root / "company_similarity_dashboard.html",
                    company_root / _COMPANY_SIMILARITY_CHECKPOINT_HTML,
                ]
            )
        else:
            candidates.append(run_dir / route / f"{route}_dashboard.html")
        candidates.extend(
            [
                run_dir / "selected_route_artifact.html",
                run_dir / "router_error.html",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _normalize_archived_entry(self, entry: dict[str, object]) -> dict[str, object]:
        normalized = dict(entry)
        if not normalized.get("archived"):
            return normalized
        source_path_value = str(normalized.get("archive_source_path") or "").strip()
        if not source_path_value:
            return normalized
        try:
            run_dir = Path(source_path_value).expanduser().resolve().parent
        except Exception:
            return normalized
        if not run_dir.exists():
            return normalized
        route_name = " ".join(str(normalized.get("route_name") or "").split()).strip()
        current_artifact = self._rebase_archived_path(run_dir, normalized.get("artifact_path"))
        preferred_artifact = self._preferred_archived_artifact_path(run_dir, route_name)
        recovered_artifact = False
        if preferred_artifact is not None and (
            current_artifact is None
            or current_artifact.name == "router_error.html"
            or not current_artifact.exists()
        ):
            current_artifact = preferred_artifact
            recovered_artifact = current_artifact.name != "router_error.html"
        if current_artifact is not None:
            normalized["artifact_path"] = str(current_artifact)
        rebased_outdir = self._rebase_archived_path(run_dir, normalized.get("outdir"))
        normalized["outdir"] = str((rebased_outdir or run_dir).resolve())
        status = " ".join(str(normalized.get("status") or "complete").split()).strip() or "complete"
        if status == "failed" and recovered_artifact:
            normalized["status"] = "complete"
            normalized["note"] = "Archived run recovered from CLIFF worker output and copied artifacts."
        return normalized

    def _load_archive_record(self, path: Path) -> dict[str, object]:
        payload = self._read_json_dict(path)
        if not payload:
            return {}
        run_id = " ".join(str(payload.get("run_id") or path.parent.name).split()).strip() or path.parent.name
        outdir_value = " ".join(str(payload.get("outdir") or path.parent).split()).strip()
        artifact_value = " ".join(str(payload.get("artifact_path") or "").split()).strip()
        route_name = " ".join(str(payload.get("route_name") or "").split()).strip()
        execution_mode = self._normalize_execution_mode(payload.get("execution_mode"))
        status = " ".join(str(payload.get("status") or "complete").split()).strip() or "complete"
        query = " ".join(str(payload.get("query") or "").split()).strip()
        if not query:
            return {}
        return self._normalize_archived_entry(
            {
            "run_id": run_id,
            "query": query,
            "status": status,
            "mind_layer": "conscious",
            "route_name": route_name,
            "note": "Archived run discovered from saved CLIFF metadata.",
            "artifact_path": artifact_value or None,
            "outdir": outdir_value or str(path.parent),
            "execution_mode": execution_mode,
            "parent_run_id": str(payload.get("parent_run_id") or "").strip(),
            "created_at": payload.get("created_at") or path.stat().st_mtime,
            "updated_at": payload.get("updated_at") or path.stat().st_mtime,
            "submission_overrides": dict(payload.get("submission_overrides") or {}),
            "archived": True,
            "archive_source_path": str(path.resolve()),
            }
        )

    def _load_worker_result_archive_record(self, path: Path) -> dict[str, object]:
        payload = self._read_json_dict(path)
        if not payload:
            return {}
        query = " ".join(str(payload.get("query") or "").split()).strip()
        if not query:
            return {}
        route_name = " ".join(str(dict(payload.get("route_decision") or {}).get("route_name") or "").split()).strip()
        artifact_value = " ".join(
            str(payload.get("artifact_path") or payload.get("error_artifact_path") or "").split()
        ).strip()
        status = " ".join(str(payload.get("status") or "complete").split()).strip() or "complete"
        outdir_value = " ".join(str(payload.get("route_outdir") or path.parent).split()).strip()
        note = (
            "Archived run discovered from CLIFF worker output."
            if status == "complete"
            else "Archived run discovered from CLIFF worker output (failed run)."
        )
        return self._normalize_archived_entry(
            {
            "run_id": path.parent.name,
            "query": query,
            "status": status,
            "mind_layer": "conscious",
            "route_name": route_name,
            "note": note,
            "artifact_path": artifact_value or None,
            "outdir": outdir_value or str(path.parent),
            "execution_mode": "quick",
            "parent_run_id": "",
            "created_at": path.stat().st_mtime,
            "updated_at": path.stat().st_mtime,
            "submission_overrides": {
                key: value
                for key, value in (
                    ("route", route_name or None),
                    ("llm_token_budget", self._normalize_llm_token_budget(payload.get("llm_token_budget"))),
                )
                if value is not None
            },
            "archived": True,
            "archive_source_path": str(path.resolve()),
            }
        )

    def _refresh_archived_runs(self, *, force: bool = False) -> None:
        if not self.config.session_mode:
            return
        now = time.time()
        if (
            not force
            and (now - self._archive_last_scan_at) < _ARCHIVE_REFRESH_SECONDS
            and not self._archive_cache_refresh_pending
        ):
            return
        roots = self._archive_roots()
        if not force and not self._archived_runs:
            cached, all_fresh = self._load_cached_archive_entries(roots)
            if cached:
                self._set_archived_runs(cached)
                self._archive_last_scan_at = now
                self._archive_cache_refresh_pending = not all_fresh
                if all_fresh:
                    return
                return
        discovered: dict[str, dict[str, object]] = {}
        limit = max(1, int(self.config.archive_max_runs))
        for root in roots:
            root_entries: list[dict[str, object]] = []
            for pattern, loader in (
                (_ARCHIVE_RECORD_FILENAME, self._load_archive_record),
                (_WORKER_RESULT_FILENAME, self._load_worker_result_archive_record),
            ):
                try:
                    matches = sorted(root.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
                except Exception:
                    continue
                for candidate in matches:
                    entry = loader(candidate)
                    if not entry:
                        continue
                    outdir_key = str(entry.get("outdir") or candidate.parent)
                    if outdir_key in discovered:
                        continue
                    if str(entry.get("run_id") or "") in self._session_runs_by_id:
                        continue
                    discovered[outdir_key] = entry
                    root_entries.append(entry)
                    if len(discovered) >= limit:
                        break
                if len(discovered) >= limit:
                    break
            self._write_cached_archive_entries(
                root,
                sorted(
                    root_entries,
                    key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
                    reverse=True,
                ),
            )
            if len(discovered) >= limit:
                break
        archived = sorted(
            discovered.values(),
            key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
            reverse=True,
        )
        self._set_archived_runs(archived)
        self._archive_last_scan_at = now
        self._archive_cache_refresh_pending = False

    def _lookup_run_state(self, run_id: str) -> dict[str, object]:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is not None:
                return dict(run_state)
            self._refresh_archived_runs()
            archived = self._archived_runs_by_id.get(run_id)
            return dict(archived or {})

    def request_archived_run_rerun(self, run_id: str) -> str | None:
        with self._lock:
            self._refresh_archived_runs()
            run_state = dict(self._archived_runs_by_id.get(run_id) or {})
        query = " ".join(str(run_state.get("query") or "").split()).strip()
        if not query:
            return None
        submission_overrides = dict(run_state.get("submission_overrides") or {})
        route_name = " ".join(str(run_state.get("route_name") or "").split()).strip()
        if route_name and "route" not in submission_overrides:
            submission_overrides["route"] = route_name
        llm_token_budget = self._run_llm_token_budget(run_state)
        if llm_token_budget is not None:
            submission_overrides.setdefault("llm_token_budget", llm_token_budget)
        return self.submit_query(
            query,
            execution_mode=self._normalize_execution_mode(run_state.get("execution_mode")),
            parent_run_id=run_id,
            submission_overrides=submission_overrides or None,
            queued_note=f"Queued by re-running archived run {run_id}.",
        )

    def _router_feedback_root_for_run(self, run_state: dict[str, object]) -> Path | None:
        archive_source = str(run_state.get("archive_source_path") or "").strip()
        if archive_source:
            try:
                return Path(archive_source).expanduser().resolve().parent
            except Exception:
                return None
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return None
        try:
            return Path(outdir_value).expanduser().resolve()
        except Exception:
            return None

    @staticmethod
    def _normalized_excluded_routes(values: object) -> tuple[str, ...]:
        routes: list[str] = []
        for value in list(values or []):
            normalized = " ".join(str(value or "").split()).strip().lower()
            if normalized in _SUPPORTED_ROUTE_NAMES:
                routes.append(normalized)
        return tuple(dict.fromkeys(routes))

    def _record_router_feedback(
        self,
        run_state: dict[str, object],
        *,
        feedback_kind: str,
        feedback_status: str,
        rejected_route: str,
        excluded_routes: tuple[str, ...],
        queued_followup_run_id: str | None = None,
    ) -> None:
        feedback_root = self._router_feedback_root_for_run(run_state)
        if feedback_root is None:
            return
        log_path = feedback_root / _ROUTER_FEEDBACK_LOG
        summary_path = feedback_root / _ROUTER_FEEDBACK_SUMMARY
        event = {
            "timestamp": time.time(),
            "run_id": str(run_state.get("run_id") or ""),
            "parent_run_id": str(run_state.get("parent_run_id") or ""),
            "query": str(run_state.get("query") or ""),
            "route_name": str(run_state.get("route_name") or ""),
            "execution_mode": str(run_state.get("execution_mode") or ""),
            "feedback_kind": str(feedback_kind),
            "feedback_status": str(feedback_status),
            "rejected_route": str(rejected_route),
            "excluded_routes": list(excluded_routes),
            "queued_followup_run_id": str(queued_followup_run_id or ""),
            "artifact_path": str(run_state.get("artifact_path") or ""),
            "outdir": str(run_state.get("outdir") or ""),
            "archived": bool(run_state.get("archived")),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

        existing_summary = self._read_json_dict(summary_path) if summary_path.exists() else {}
        feedback_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("feedback_counts") or {}).items()
                if str(key).strip()
            }
        )
        status_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("feedback_status_counts") or {}).items()
                if str(key).strip()
            }
        )
        rejected_route_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("rejected_route_counts") or {}).items()
                if str(key).strip()
            }
        )
        feedback_counts[str(feedback_kind)] += 1
        status_counts[str(feedback_status)] += 1
        if rejected_route:
            rejected_route_counts[str(rejected_route)] += 1
        summary = {
            "event_count": int(existing_summary.get("event_count") or 0) + 1,
            "feedback_counts": dict(feedback_counts),
            "feedback_status_counts": dict(status_counts),
            "rejected_route_counts": dict(rejected_route_counts),
            "latest_event": event,
            "latest_excluded_routes": list(excluded_routes),
        }
        self._write_json_file(summary_path, summary)

    def request_run_wrong_route(self, run_id: str) -> str | None:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None:
                self._refresh_archived_runs()
                run_state = self._archived_runs_by_id.get(run_id)
            run_state = dict(run_state or {})
        query = " ".join(str(run_state.get("query") or "").split()).strip()
        route_name = " ".join(str(run_state.get("route_name") or "").split()).strip().lower()
        if (
            not query
            or not route_name
            or route_name not in _SUPPORTED_ROUTE_NAMES
            or str(run_state.get("status") or "").strip().lower() != "complete"
        ):
            self._record_router_feedback(
                run_state,
                feedback_kind="wrong_route",
                feedback_status="ignored",
                rejected_route=route_name,
                excluded_routes=(),
            )
            return None
        overrides = dict(run_state.get("submission_overrides") or {})
        existing_excluded = self._normalized_excluded_routes(overrides.get("router_excluded_routes"))
        excluded_routes = tuple(dict.fromkeys(existing_excluded + (route_name,)))
        execution_mode = self._normalize_execution_mode(run_state.get("execution_mode"))
        llm_token_budget = self._run_llm_token_budget(run_state)
        submission_overrides = {"route": "auto", "router_excluded_routes": list(excluded_routes)}
        if llm_token_budget is not None:
            submission_overrides["llm_token_budget"] = llm_token_budget
        new_run_id = self.submit_query(
            query,
            execution_mode=execution_mode,
            parent_run_id=run_id,
            submission_overrides=submission_overrides,
            queued_note=f"Queued after marking the {route_name} route as wrong for {run_id}.",
        )
        self._record_router_feedback(
            run_state,
            feedback_kind="wrong_route",
            feedback_status="queued" if new_run_id else "ignored",
            rejected_route=route_name,
            excluded_routes=excluded_routes,
            queued_followup_run_id=new_run_id,
        )
        return new_run_id

    def request_session_run_deepen(
        self,
        run_id: str,
        *,
        query_override: str | None = None,
        democritus_manifest_path: Path | None = None,
        democritus_target_docs: int | None = None,
        democritus_base_query: str | None = None,
        democritus_selected_topics: tuple[str, ...] = (),
        democritus_rejected_topics: tuple[str, ...] = (),
        democritus_retrieval_refinement: str = "",
        company_similarity_year_start: int | None = None,
        company_similarity_year_end: int | None = None,
    ) -> str | None:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None or not self.config.session_mode or not self.config.enable_execution_mode:
                return None
            if str(run_state.get("status") or "") != "complete":
                return None
            if self._normalize_execution_mode(run_state.get("execution_mode")) not in {"quick", "interactive"}:
                return None
            route_name = str(run_state.get("route_name") or "")
            if route_name not in {"democritus", "company_similarity"}:
                return None
            query = " ".join(str(query_override or run_state.get("query") or "").split()).strip()
            llm_token_budget = self._run_llm_token_budget(dict(run_state))
        if not query:
            return None
        submission_overrides: dict[str, object] = {"route": route_name}
        if llm_token_budget is not None:
            submission_overrides["llm_token_budget"] = llm_token_budget
        if democritus_manifest_path is not None:
            submission_overrides["democritus_manifest"] = str(democritus_manifest_path.resolve())
        if democritus_target_docs is not None:
            submission_overrides["democritus_target_docs"] = max(1, int(democritus_target_docs))
        if democritus_base_query:
            submission_overrides["democritus_base_query"] = " ".join(str(democritus_base_query).split()).strip()
        if democritus_selected_topics:
            submission_overrides["democritus_selected_topics"] = list(democritus_selected_topics)
        if democritus_rejected_topics:
            submission_overrides["democritus_rejected_topics"] = list(democritus_rejected_topics)
        if democritus_retrieval_refinement:
            submission_overrides["democritus_retrieval_refinement"] = " ".join(
                str(democritus_retrieval_refinement).split()
            ).strip()
        if company_similarity_year_start is not None:
            submission_overrides["company_similarity_year_start"] = int(company_similarity_year_start)
        if company_similarity_year_end is not None:
            submission_overrides["company_similarity_year_end"] = int(company_similarity_year_end)
        queued_note = (
            f"Queued as a deep follow-up to {run_id}."
            if not submission_overrides
            else f"Queued as a deep follow-up to {run_id} using the curated checkpoint selection."
        )
        return self.submit_query(
            query,
            execution_mode="deep",
            parent_run_id=run_id,
            submission_overrides=submission_overrides or None,
            queued_note=queued_note,
        )

    def request_session_run_retrieve_more(
        self,
        run_id: str,
        *,
        query_override: str | None = None,
        democritus_target_docs: int,
        democritus_atlas_baseline: dict[str, object] | None = None,
        democritus_base_query: str | None = None,
        democritus_selected_topics: tuple[str, ...] = (),
        democritus_rejected_topics: tuple[str, ...] = (),
        democritus_retrieval_refinement: str = "",
    ) -> str | None:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None or not self.config.session_mode or not self.config.enable_execution_mode:
                return None
            route_name = str(run_state.get("route_name") or "")
            if route_name != "democritus":
                return None
            query = " ".join(str(query_override or run_state.get("query") or "").split()).strip()
            llm_token_budget = self._run_llm_token_budget(dict(run_state))
        if not query:
            return None
        submission_overrides: dict[str, object] = {
            "route": "democritus",
            "democritus_target_docs": max(1, int(democritus_target_docs)),
        }
        if llm_token_budget is not None:
            submission_overrides["llm_token_budget"] = llm_token_budget
        if democritus_atlas_baseline:
            submission_overrides["democritus_atlas_baseline"] = dict(democritus_atlas_baseline)
        if democritus_base_query:
            submission_overrides["democritus_base_query"] = " ".join(str(democritus_base_query).split()).strip()
        if democritus_selected_topics:
            submission_overrides["democritus_selected_topics"] = list(democritus_selected_topics)
        if democritus_rejected_topics:
            submission_overrides["democritus_rejected_topics"] = list(democritus_rejected_topics)
        if democritus_retrieval_refinement:
            submission_overrides["democritus_retrieval_refinement"] = " ".join(
                str(democritus_retrieval_refinement).split()
            ).strip()
        return self.submit_query(
            query,
            execution_mode="interactive",
            parent_run_id=run_id,
            submission_overrides=submission_overrides,
            queued_note=f"Queued to retrieve more Democritus evidence after {run_id}.",
        )

    @staticmethod
    def _checkpoint_manifest_for_html(artifact_path: Path) -> Path | None:
        if artifact_path.name != _DEMOCRITUS_CHECKPOINT_HTML:
            return None
        manifest_path = artifact_path.with_name(_DEMOCRITUS_CHECKPOINT_MANIFEST)
        return manifest_path if manifest_path.exists() else None

    @staticmethod
    def _company_similarity_checkpoint_manifest_for_html(artifact_path: Path) -> Path | None:
        if artifact_path.name != _COMPANY_SIMILARITY_CHECKPOINT_HTML:
            return None
        manifest_path = artifact_path.with_name(_COMPANY_SIMILARITY_CHECKPOINT_MANIFEST)
        return manifest_path if manifest_path.exists() else None

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json_file(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_democritus_checkpoint_payload(self, artifact_path: Path) -> dict[str, object]:
        manifest_path = self._checkpoint_manifest_for_html(artifact_path)
        if manifest_path is None:
            return {}
        payload = self._read_json_dict(manifest_path)
        payload["_manifest_path"] = str(manifest_path)
        return payload

    def _load_company_similarity_checkpoint_payload(self, artifact_path: Path) -> dict[str, object]:
        manifest_path = self._company_similarity_checkpoint_manifest_for_html(artifact_path)
        if manifest_path is None:
            return {}
        payload = self._read_json_dict(manifest_path)
        payload["_manifest_path"] = str(manifest_path)
        return payload

    def _default_selected_checkpoint_paths(self, payload: dict[str, object]) -> tuple[str, ...]:
        selected: list[str] = []
        for item in list(payload.get("documents") or []):
            pdf_path = " ".join(str(dict(item).get("pdf_path") or "").split()).strip()
            if pdf_path:
                selected.append(pdf_path)
        return tuple(dict.fromkeys(selected))

    def _checkpoint_curation_path(self, payload: dict[str, object]) -> Path | None:
        manifest_path_raw = str(payload.get("_manifest_path") or "").strip()
        if not manifest_path_raw:
            return None
        return Path(manifest_path_raw).resolve().with_name(_DEMOCRITUS_CURATION_STATE)

    def _load_democritus_checkpoint_curation(self, payload: dict[str, object]) -> dict[str, object]:
        curation_path = self._checkpoint_curation_path(payload)
        selected_defaults = self._default_selected_checkpoint_paths(payload)
        if curation_path is None or not curation_path.exists():
            return {
                "selected_pdf_paths": list(selected_defaults),
                "selected_topics": [],
                "rejected_topics": [],
                "retrieval_refinement": "",
            }
        curation = self._read_json_dict(curation_path)
        selected_paths = [
            " ".join(str(value).split()).strip()
            for value in list(curation.get("selected_pdf_paths") or [])
            if " ".join(str(value).split()).strip()
        ]
        if "selected_pdf_paths" not in curation:
            selected_paths = list(selected_defaults)
        selected_topics = self._unique_checkpoint_topics(list(curation.get("selected_topics") or []))
        rejected_topics = tuple(
            topic for topic in self._unique_checkpoint_topics(list(curation.get("rejected_topics") or []))
            if topic not in set(selected_topics)
        )
        return {
            "selected_pdf_paths": list(dict.fromkeys(selected_paths)),
            "selected_topics": list(selected_topics),
            "rejected_topics": list(rejected_topics),
            "retrieval_refinement": " ".join(str(curation.get("retrieval_refinement") or "").split()).strip(),
        }

    def _save_democritus_checkpoint_curation(
        self,
        payload: dict[str, object],
        *,
        selected_pdf_paths: tuple[str, ...],
        selected_topics: tuple[str, ...] = (),
        rejected_topics: tuple[str, ...] = (),
        retrieval_refinement: str = "",
    ) -> Path | None:
        curation_path = self._checkpoint_curation_path(payload)
        if curation_path is None:
            return None
        self._write_json_file(
            curation_path,
            {
                "selected_pdf_paths": list(selected_pdf_paths),
                "selected_topics": list(selected_topics),
                "rejected_topics": list(rejected_topics),
                "retrieval_refinement": retrieval_refinement,
                "updated_at": time.time(),
            },
        )
        return curation_path

    def _company_similarity_checkpoint_curation_path(self, payload: dict[str, object]) -> Path | None:
        manifest_path_raw = str(payload.get("_manifest_path") or "").strip()
        if not manifest_path_raw:
            return None
        return Path(manifest_path_raw).resolve().with_name(_COMPANY_SIMILARITY_CURATION_STATE)

    def _load_company_similarity_checkpoint_curation(self, payload: dict[str, object]) -> dict[str, int]:
        suggested = dict(payload.get("suggested_year_window") or {})
        default_start = int(suggested.get("start") or dict(payload.get("year_window") or {}).get("start") or 2002)
        default_end = int(suggested.get("end") or dict(payload.get("year_window") or {}).get("end") or default_start)
        curation_path = self._company_similarity_checkpoint_curation_path(payload)
        if curation_path is None or not curation_path.exists():
            return {"year_start": default_start, "year_end": default_end}
        curation = self._read_json_dict(curation_path)
        try:
            return {
                "year_start": int(curation.get("year_start") or default_start),
                "year_end": int(curation.get("year_end") or default_end),
            }
        except (TypeError, ValueError):
            return {"year_start": default_start, "year_end": default_end}

    def _save_company_similarity_checkpoint_curation(
        self,
        payload: dict[str, object],
        *,
        year_start: int,
        year_end: int,
    ) -> Path | None:
        curation_path = self._company_similarity_checkpoint_curation_path(payload)
        if curation_path is None:
            return None
        self._write_json_file(
            curation_path,
            {
                "year_start": int(year_start),
                "year_end": int(year_end),
                "updated_at": time.time(),
            },
        )
        return curation_path

    def _write_democritus_curated_manifest(
        self,
        payload: dict[str, object],
        *,
        selected_pdf_paths: tuple[str, ...],
    ) -> Path | None:
        manifest_path_raw = str(payload.get("_manifest_path") or "").strip()
        if not manifest_path_raw:
            return None
        selected_lookup = {item for item in selected_pdf_paths if item}
        rows: list[dict[str, object]] = []
        for item in list(payload.get("documents") or []):
            record = dict(item)
            pdf_path = " ".join(str(record.get("pdf_path") or "").split()).strip()
            if not pdf_path or pdf_path not in selected_lookup:
                continue
            summary = " ".join(
                str(value).strip()
                for value in (
                    record.get("guide_summary"),
                    record.get("causal_gestalt"),
                )
                if str(value or "").strip()
            ).strip()
            rows.append(
                {
                    "id": str(record.get("run_name") or Path(pdf_path).stem),
                    "title": str(record.get("title") or Path(pdf_path).stem),
                    "summary": summary,
                    "abstract": summary,
                    "keywords": ", ".join(str(topic) for topic in list(record.get("topics") or [])),
                    "pdf_path": pdf_path,
                    "source_path": pdf_path,
                    "document_format": "pdf",
                }
            )
        target_path = Path(manifest_path_raw).resolve().with_name(_DEMOCRITUS_CURATED_MANIFEST)
        self._write_json_file(target_path, rows)
        return target_path

    def _checkpoint_telemetry_log_path(self, payload: dict[str, object]) -> Path | None:
        manifest_path_raw = str(payload.get("_manifest_path") or "").strip()
        if not manifest_path_raw:
            return None
        return Path(manifest_path_raw).resolve().with_name(_DEMOCRITUS_CURATION_TELEMETRY)

    def _checkpoint_telemetry_summary_path(self, payload: dict[str, object]) -> Path | None:
        manifest_path_raw = str(payload.get("_manifest_path") or "").strip()
        if not manifest_path_raw:
            return None
        return Path(manifest_path_raw).resolve().with_name(_DEMOCRITUS_CURATION_SUMMARY)

    @staticmethod
    def _topic_counter_payload(counter: Counter[str], *, limit: int = 24) -> dict[str, int]:
        return {topic: int(count) for topic, count in counter.most_common(limit)}

    @staticmethod
    def _atlas_drift_metrics(payload: dict[str, object]) -> dict[str, object]:
        metrics = dict(payload.get("drift_metrics") or {})
        return {
            "total_topic_count": int(metrics.get("total_topic_count") or 0),
            "aligned_topic_count": int(metrics.get("aligned_topic_count") or 0),
            "suspicious_topic_count": int(metrics.get("suspicious_topic_count") or 0),
            "aligned_topic_ratio": float(metrics.get("aligned_topic_ratio") or 0.0),
            "mean_alignment_score": float(metrics.get("mean_alignment_score") or 0.0),
            "synthesis_readiness_proxy": float(metrics.get("synthesis_readiness_proxy") or 0.0),
        }

    def _atlas_drift_comparison(
        self,
        *,
        payload: dict[str, object],
        run_state: dict[str, object],
    ) -> dict[str, object]:
        overrides = dict(run_state.get("submission_overrides") or {})
        baseline_raw = dict(overrides.get("democritus_atlas_baseline") or {})
        if not baseline_raw:
            return {}
        current = self._atlas_drift_metrics(payload)
        previous = {
            "total_topic_count": int(baseline_raw.get("total_topic_count") or 0),
            "aligned_topic_count": int(baseline_raw.get("aligned_topic_count") or 0),
            "suspicious_topic_count": int(baseline_raw.get("suspicious_topic_count") or 0),
            "aligned_topic_ratio": float(baseline_raw.get("aligned_topic_ratio") or 0.0),
            "mean_alignment_score": float(baseline_raw.get("mean_alignment_score") or 0.0),
            "synthesis_readiness_proxy": float(baseline_raw.get("synthesis_readiness_proxy") or 0.0),
        }
        suspicious_delta = previous["suspicious_topic_count"] - current["suspicious_topic_count"]
        alignment_delta = round(current["aligned_topic_ratio"] - previous["aligned_topic_ratio"], 3)
        readiness_delta = round(
            current["synthesis_readiness_proxy"] - previous["synthesis_readiness_proxy"],
            3,
        )
        reduced_drift = suspicious_delta > 0 or alignment_delta > 0.0 or readiness_delta > 0.0
        return {
            "previous": previous,
            "current": current,
            "suspicious_topic_delta": suspicious_delta,
            "alignment_ratio_delta": alignment_delta,
            "synthesis_readiness_delta": readiness_delta,
            "reduced_drift": reduced_drift,
        }

    @staticmethod
    def _normalize_checkpoint_topic(value: object) -> str:
        return " ".join(str(value or "").split()).strip()

    def _unique_checkpoint_topics(self, values: list[object] | tuple[object, ...]) -> tuple[str, ...]:
        topics: list[str] = []
        for value in values:
            normalized = self._normalize_checkpoint_topic(value)
            if normalized:
                topics.append(normalized)
        return tuple(dict.fromkeys(topics))

    def _available_checkpoint_topics(self, payload: dict[str, object]) -> tuple[str, ...]:
        topics: list[str] = []
        for item in list(payload.get("top_topics") or []):
            topic = self._normalize_checkpoint_topic(dict(item).get("topic"))
            if topic:
                topics.append(topic)
        for item in list(payload.get("documents") or []):
            for topic in list(dict(item).get("topics") or []):
                normalized = self._normalize_checkpoint_topic(topic)
                if normalized:
                    topics.append(normalized)
        return tuple(dict.fromkeys(topics))

    def _filter_checkpoint_topics(
        self,
        payload: dict[str, object],
        *,
        selected_topics: tuple[str, ...],
        rejected_topics: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        valid_topics = set(self._available_checkpoint_topics(payload))
        filtered_selected = tuple(topic for topic in selected_topics if topic in valid_topics)
        filtered_selected_lookup = set(filtered_selected)
        filtered_rejected = tuple(
            topic for topic in rejected_topics if topic in valid_topics and topic not in filtered_selected_lookup
        )
        return filtered_selected, filtered_rejected

    def _topic_guidance_refinement(
        self,
        *,
        selected_topics: tuple[str, ...],
        rejected_topics: tuple[str, ...],
        retrieval_refinement: str,
    ) -> str:
        parts: list[str] = []
        if selected_topics:
            parts.append("focus on topics: " + "; ".join(selected_topics))
        if rejected_topics:
            parts.append("avoid topics: " + "; ".join(rejected_topics))
        normalized_refinement = " ".join(str(retrieval_refinement or "").split()).strip()
        if normalized_refinement:
            parts.append(normalized_refinement)
        return ". ".join(part for part in parts if part)

    def _democritus_checkpoint_selection_snapshot(
        self,
        payload: dict[str, object],
        *,
        selected_pdf_paths: tuple[str, ...],
        selected_topics: tuple[str, ...] = (),
        rejected_topics: tuple[str, ...] = (),
    ) -> dict[str, object]:
        selected_lookup = {path for path in selected_pdf_paths if path}
        selected_docs: list[dict[str, object]] = []
        rejected_docs: list[dict[str, object]] = []
        selected_topic_counts: Counter[str] = Counter()
        rejected_topic_counts: Counter[str] = Counter()
        for item in list(payload.get("documents") or []):
            record = dict(item)
            pdf_path = " ".join(str(record.get("pdf_path") or "").split()).strip()
            normalized_topics = [
                " ".join(str(topic).split()).strip()
                for topic in list(record.get("topics") or [])
                if " ".join(str(topic).split()).strip()
            ]
            normalized_record = {
                "run_name": str(record.get("run_name") or ""),
                "title": str(record.get("title") or ""),
                "pdf_path": pdf_path,
                "topics": normalized_topics,
            }
            if pdf_path in selected_lookup:
                selected_docs.append(normalized_record)
                for topic in normalized_topics:
                    selected_topic_counts[topic] += 1
            else:
                rejected_docs.append(normalized_record)
                for topic in normalized_topics:
                    rejected_topic_counts[topic] += 1
        explicit_selected_topics = self._unique_checkpoint_topics(selected_topics)
        explicit_rejected_topics = tuple(
            topic for topic in self._unique_checkpoint_topics(rejected_topics)
            if topic not in set(explicit_selected_topics)
        )
        for topic in explicit_selected_topics:
            selected_topic_counts[topic] += 1
        for topic in explicit_rejected_topics:
            rejected_topic_counts[topic] += 1
        topic_preference_signal = Counter(selected_topic_counts)
        topic_preference_signal.subtract(rejected_topic_counts)
        return {
            "selected_documents": selected_docs,
            "rejected_documents": rejected_docs,
            "explicit_selected_topics": list(explicit_selected_topics),
            "explicit_rejected_topics": list(explicit_rejected_topics),
            "selected_topic_counts": self._topic_counter_payload(selected_topic_counts),
            "rejected_topic_counts": self._topic_counter_payload(rejected_topic_counts),
            "topic_preference_signal": {
                topic: int(value)
                for topic, value in topic_preference_signal.items()
                if int(value) != 0
            },
        }

    def _record_democritus_checkpoint_telemetry(
        self,
        payload: dict[str, object],
        *,
        run_state: dict[str, object],
        action_kind: str,
        action_status: str,
        selected_pdf_paths: tuple[str, ...],
        selected_topics: tuple[str, ...],
        rejected_topics: tuple[str, ...],
        retrieval_refinement: str,
        additional_documents: int,
        queued_followup_run_id: str | None = None,
        curated_manifest_path: Path | None = None,
    ) -> None:
        log_path = self._checkpoint_telemetry_log_path(payload)
        summary_path = self._checkpoint_telemetry_summary_path(payload)
        if log_path is None or summary_path is None:
            return
        snapshot = self._democritus_checkpoint_selection_snapshot(
            payload,
            selected_pdf_paths=selected_pdf_paths,
            selected_topics=selected_topics,
            rejected_topics=rejected_topics,
        )
        selected_documents = list(snapshot.get("selected_documents") or [])
        rejected_documents = list(snapshot.get("rejected_documents") or [])
        atlas_drift_metrics = self._atlas_drift_metrics(payload)
        atlas_drift_comparison = self._atlas_drift_comparison(payload=payload, run_state=run_state)
        event = {
            "timestamp": time.time(),
            "run_id": str(run_state.get("run_id") or ""),
            "parent_run_id": str(run_state.get("parent_run_id") or ""),
            "route_name": str(run_state.get("route_name") or ""),
            "execution_mode": str(run_state.get("execution_mode") or ""),
            "query": str(payload.get("query") or run_state.get("query") or ""),
            "stage_label": str(payload.get("stage_label") or ""),
            "action_kind": str(action_kind),
            "action_status": str(action_status),
            "selected_count": len(selected_documents),
            "rejected_count": len(rejected_documents),
            "selected_documents": selected_documents,
            "rejected_documents": rejected_documents,
            "explicit_selected_topics": list(snapshot.get("explicit_selected_topics") or []),
            "explicit_rejected_topics": list(snapshot.get("explicit_rejected_topics") or []),
            "selected_topic_counts": dict(snapshot.get("selected_topic_counts") or {}),
            "rejected_topic_counts": dict(snapshot.get("rejected_topic_counts") or {}),
            "topic_preference_signal": dict(snapshot.get("topic_preference_signal") or {}),
            "atlas_drift_metrics": atlas_drift_metrics,
            "atlas_drift_comparison": atlas_drift_comparison,
            "retrieval_refinement": retrieval_refinement,
            "additional_documents_requested": int(max(1, additional_documents)),
            "queued_followup_run_id": str(queued_followup_run_id or ""),
            "curated_manifest_path": str(curated_manifest_path.resolve()) if curated_manifest_path else "",
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

        existing_summary = self._read_json_dict(summary_path) if summary_path.exists() else {}
        action_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("action_counts") or {}).items()
                if str(key).strip()
            }
        )
        status_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("action_status_counts") or {}).items()
                if str(key).strip()
            }
        )
        topic_preferences = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("cumulative_topic_preference_signal") or {}).items()
                if str(key).strip()
            }
        )
        document_selection_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("document_selection_counts") or {}).items()
                if str(key).strip()
            }
        )
        rejected_document_counts = Counter(
            {
                str(key): int(value)
                for key, value in dict(existing_summary.get("document_rejection_counts") or {}).items()
                if str(key).strip()
            }
        )
        action_counts[str(action_kind)] += 1
        status_counts[str(action_status)] += 1
        for topic, delta in dict(event.get("topic_preference_signal") or {}).items():
            topic_preferences[str(topic)] += int(delta)
        for item in selected_documents:
            title = " ".join(str(dict(item).get("title") or "").split()).strip()
            if title:
                document_selection_counts[title] += 1
        for item in rejected_documents:
            title = " ".join(str(dict(item).get("title") or "").split()).strip()
            if title:
                rejected_document_counts[title] += 1
        summary = {
            "event_count": int(existing_summary.get("event_count") or 0) + 1,
            "action_counts": dict(action_counts),
            "action_status_counts": dict(status_counts),
            "latest_event": event,
            "latest_selected_count": len(selected_documents),
            "latest_rejected_count": len(rejected_documents),
            "latest_retrieval_refinement": retrieval_refinement,
            "latest_atlas_drift_metrics": atlas_drift_metrics,
            "latest_atlas_drift_comparison": atlas_drift_comparison,
            "cumulative_topic_preference_signal": dict(topic_preferences),
            "document_selection_counts": dict(document_selection_counts),
            "document_rejection_counts": dict(rejected_document_counts),
        }
        self._write_json_file(summary_path, summary)

    @staticmethod
    def _combine_query_with_refinement(query: str, refinement: str) -> str:
        base = " ".join(str(query or "").split()).strip()
        suffix = " ".join(str(refinement or "").split()).strip()
        if not suffix:
            return base
        if not base:
            return suffix
        return f"{base} {suffix}"

    @staticmethod
    def _canonical_guided_query(
        *,
        base_query: str,
        selected_topics: tuple[str, ...],
        rejected_topics: tuple[str, ...],
        retrieval_refinement: str,
    ) -> str:
        parts = [" ".join(str(base_query or "").split()).strip()]
        if selected_topics:
            parts.append("focus on topics: " + "; ".join(selected_topics))
        if rejected_topics:
            parts.append("avoid topics: " + "; ".join(rejected_topics))
        refinement = " ".join(str(retrieval_refinement or "").split()).strip()
        if refinement:
            parts.append(refinement)
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _democritus_base_query(payload: dict[str, object], run_state: dict[str, object]) -> str:
        overrides = dict(run_state.get("submission_overrides") or {})
        return " ".join(
            str(
                overrides.get("democritus_base_query")
                or payload.get("base_query")
                or run_state.get("query")
                or payload.get("query")
                or ""
            ).split()
        ).strip()

    def _render_democritus_checkpoint_page(
        self,
        run_id: str,
        *,
        artifact_path: Path,
        banner_message: str = "",
        banner_tone: str = "info",
    ) -> str:
        payload = self._load_democritus_checkpoint_payload(artifact_path)
        if not payload:
            return artifact_path.read_text(encoding="utf-8", errors="replace")
        with self._lock:
            run_state = dict(self._session_runs_by_id.get(run_id) or {})
        curation = self._load_democritus_checkpoint_curation(payload)
        selected_pdf_paths = set(str(item) for item in list(curation.get("selected_pdf_paths") or []))
        selected_topics = set(
            self._unique_checkpoint_topics(list(curation.get("selected_topics") or []))
        )
        rejected_topics = set(
            self._unique_checkpoint_topics(list(curation.get("rejected_topics") or []))
        )
        documents = list(payload.get("documents") or [])
        n_documents = len(documents)
        selected_count = sum(
            1
            for item in documents
            if " ".join(str(dict(item).get("pdf_path") or "").split()).strip() in selected_pdf_paths
        )
        query = html.escape(str(payload.get("query") or "Democritus interactive checkpoint"))
        stage_label = html.escape(str(payload.get("stage_label") or "Topic checkpoint"))
        summary_text = html.escape(str(payload.get("summary_text") or ""))
        query_focus_terms = list(payload.get("query_focus_terms") or [])
        suspicious_topics = list(payload.get("suspicious_topics") or [])
        drift_metrics = self._atlas_drift_metrics(payload)
        drift_comparison = self._atlas_drift_comparison(payload=payload, run_state=run_state)
        topic_counts: Counter[str] = Counter()
        for item in documents:
            for topic in list(dict(item).get("topics") or []):
                normalized = self._normalize_checkpoint_topic(topic)
                if normalized:
                    topic_counts[normalized] += 1
        atlas_topics: list[tuple[str, int, tuple[str, ...], int]] = []
        seen_topics: set[str] = set()
        for item in list(payload.get("top_topics") or []):
            item_dict = dict(item)
            topic = self._normalize_checkpoint_topic(item_dict.get("topic"))
            if not topic or topic in seen_topics:
                continue
            count = int(item_dict.get("document_count") or topic_counts.get(topic) or 0)
            aliases = tuple(
                self._normalize_checkpoint_topic(alias)
                for alias in list(item_dict.get("aliases") or [])
                if self._normalize_checkpoint_topic(alias)
            )
            equivalence_class_size = int(item_dict.get("equivalence_class_size") or len(aliases) or 1)
            atlas_topics.append((topic, count, aliases, equivalence_class_size))
            seen_topics.add(topic)
        for topic, count in topic_counts.most_common(16):
            if topic in seen_topics:
                continue
            atlas_topics.append((topic, int(count), (topic,), 1))
            seen_topics.add(topic)
        for topic in list(selected_topics) + list(rejected_topics):
            if topic in seen_topics:
                continue
            atlas_topics.append((topic, int(topic_counts.get(topic) or 0), (topic,), 1))
            seen_topics.add(topic)
        topic_hidden_inputs = "".join(
            f'<input type="hidden" name="selected_topic" value="{html.escape(topic)}" data-topic-hidden="selected" />'
            for topic in selected_topics
        ) + "".join(
            f'<input type="hidden" name="rejected_topic" value="{html.escape(topic)}" data-topic-hidden="rejected" />'
            for topic in rejected_topics
        )
        suspicious_lookup = {
            self._normalize_checkpoint_topic(dict(item).get("topic")) for item in suspicious_topics
        }
        focus_term_markup = "".join(
            f'<span class="checkpoint-chip checkpoint-chip-focus">{html.escape(str(term))}</span>'
            for term in query_focus_terms[:8]
        ) or '<span class="checkpoint-chip">No strong query anchors extracted</span>'
        suspicious_topic_markup = "".join(
            f'<span class="checkpoint-chip checkpoint-chip-warn">{html.escape(str(dict(item).get("topic") or ""))}</span>'
            for item in suspicious_topics[:8]
            if str(dict(item).get("topic") or "").strip()
        ) or '<span class="checkpoint-chip">No obvious off-scope topics flagged</span>'
        drift_comparison_markup = ""
        if drift_comparison:
            previous = dict(drift_comparison.get("previous") or {})
            current = dict(drift_comparison.get("current") or {})
            reduced_drift = bool(drift_comparison.get("reduced_drift"))
            drift_tone_class = "checkpoint-chip-focus" if reduced_drift else "checkpoint-chip-warn"
            drift_message = (
                f"Drift tightened from {int(previous.get('suspicious_topic_count') or 0)} suspicious topics to "
                f"{int(current.get('suspicious_topic_count') or 0)}."
                if reduced_drift
                else f"Drift did not tighten yet: {int(previous.get('suspicious_topic_count') or 0)} suspicious topics before, "
                f"{int(current.get('suspicious_topic_count') or 0)} now."
            )
            drift_comparison_markup = (
                '<div class="checkpoint-chips" style="margin-top:14px;">'
                f'<span class="checkpoint-chip {drift_tone_class}">{html.escape(drift_message)}</span>'
                f'<span class="checkpoint-chip">alignment {html.escape(str(previous.get("aligned_topic_ratio") or 0.0))} → {html.escape(str(current.get("aligned_topic_ratio") or 0.0))}</span>'
                f'<span class="checkpoint-chip">readiness {html.escape(str(previous.get("synthesis_readiness_proxy") or 0.0))} → {html.escape(str(current.get("synthesis_readiness_proxy") or 0.0))}</span>'
                '</div>'
            )
        topic_chips = "".join(
            '<button type="button" class="topic-chip-button'
            + (
                ' is-selected'
                if topic in selected_topics
                else ' is-rejected'
                if topic in rejected_topics
                else ' is-suspicious'
                if topic in suspicious_lookup
                else ''
            )
            + f'" data-topic="{html.escape(topic)}"'
            + f' title="{html.escape("Aliases: " + " | ".join(alias for alias in aliases[:4]))}"'
            + f' data-topic-state="{html.escape("selected" if topic in selected_topics else "rejected" if topic in rejected_topics else "neutral")}">'
            + f'<span class="topic-chip-label">{html.escape(topic)}</span>'
            + f'<span class="topic-chip-count">{html.escape(str(count))} doc{"s" if int(count) != 1 else ""}'
            + (
                f' · {html.escape(str(equivalence_class_size))} variants'
                if int(equivalence_class_size) > 1
                else ""
            )
            + '</span>'
            + "</button>"
            for topic, count, aliases, equivalence_class_size in atlas_topics[:24]
        ) or '<span class="chip">No recurring topics detected yet</span>'
        document_cards = []
        for item in documents:
            record = dict(item)
            pdf_path_raw = " ".join(str(record.get("pdf_path") or "").split()).strip()
            checked_attr = " checked" if pdf_path_raw in selected_pdf_paths else ""
            pdf_href = (
                html.escape(self._launcher_href_for_run_file(run_id, Path(pdf_path_raw).resolve()))
                if pdf_path_raw
                else ""
            )
            topics_markup = "".join(
                f'<span class="topic-pill">{html.escape(str(topic))}</span>'
                for topic in list(record.get("topics") or [])[:12]
            )
            document_cards.append(
                '<article class="doc-card">'
                f'<label class="doc-select"><input type="checkbox" name="selected_pdf_path" value="{html.escape(pdf_path_raw)}"{checked_attr} /> Include in deeper analysis</label>'
                f'<div class="doc-meta">{html.escape(str(record.get("run_name") or ""))}</div>'
                f'<h3 class="doc-title" title="{html.escape(str(record.get("title") or ""))}">{html.escape(str(record.get("title") or ""))}</h3>'
                + (
                    f'<div class="doc-actions"><a href="{pdf_href}" target="_blank" rel="noopener noreferrer">Inspect PDF</a></div>'
                    if pdf_href
                    else ""
                )
                + (
                    f'<p class="guide">{html.escape(str(record.get("guide_summary") or ""))}</p>'
                    if str(record.get("guide_summary") or "").strip()
                    else ""
                )
                + (
                    f'<p class="guide"><strong>Causal gestalt:</strong> {html.escape(str(record.get("causal_gestalt") or ""))}</p>'
                    if str(record.get("causal_gestalt") or "").strip()
                    else ""
                )
                + f'<div class="topic-list">{topics_markup}</div>'
                + "</article>"
            )
        banner_markup = (
            f'<section class="banner banner-{html.escape(banner_tone)}">{html.escape(banner_message)}</section>'
            if banner_message
            else ""
        )
        refinement_value = html.escape(str(curation.get("retrieval_refinement") or ""))
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Democritus Interactive Checkpoint</title>
    <style>
      :root {{
        --ink: #18222d;
        --muted: #5b6874;
        --paper: #f6f1e8;
        --card: rgba(255,255,255,0.9);
        --line: #d7ccb8;
        --accent: #93451e;
        --green: #1f6a56;
        --warn: #8b4a1f;
        --soft: #f8ede0;
        --ok: #e8f4ee;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(147,69,30,0.12), transparent 24%),
          linear-gradient(180deg, #fbf7f0 0%, var(--paper) 100%);
      }}
      main {{ width: min(1240px, calc(100vw - 32px)); margin: 32px auto 48px; display: grid; gap: 18px; }}
      .panel, .banner {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; box-shadow: 0 24px 60px rgba(30,25,18,0.08); }}
      .banner-info {{ background: #fdf8ef; }}
      .banner-success {{ background: var(--ok); }}
      .banner-warn {{ background: var(--soft); }}
      .eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      .hero-grid {{ display: grid; gap: 18px; grid-template-columns: 1.35fr 1fr; }}
      h1, h2, h3, p {{ margin: 0; }}
      .trace {{ color: var(--muted); line-height: 1.6; }}
      .chip-row, .topic-list, .checkpoint-chips, .form-actions {{ display: flex; flex-wrap: wrap; gap: 10px; min-width: 0; }}
      .chip, .topic-pill, .checkpoint-chip {{ border-radius: 999px; padding: 8px 12px; background: #efe7d9; font-size: 0.92rem; color: #64492b; }}
      .topic-pill {{ background: #f5efe4; max-width: 100%; overflow-wrap: anywhere; }}
      .checkpoint-chip {{ background: #f6f0e5; }}
      .checkpoint-chip-focus {{ background: #e8f4ee; color: #204d41; }}
      .checkpoint-chip-warn {{ background: #f8ede0; color: #8b4a1f; }}
      .topic-chip-button {{
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 10px 14px;
        background: #fff8ef;
        color: #5c4128;
        font: inherit;
        display: inline-flex;
        gap: 10px;
        align-items: center;
        cursor: pointer;
      }}
      .topic-chip-button.is-selected {{ background: #e8f4ee; border-color: #8eb7aa; color: #204d41; }}
      .topic-chip-button.is-rejected {{ background: #f8ede0; border-color: #d1aa8b; color: #8b4a1f; }}
      .topic-chip-button.is-suspicious {{ border-style: dashed; border-color: #d1aa8b; }}
      .topic-chip-label {{ font-weight: 700; }}
      .topic-chip-count {{ font-size: 0.84rem; opacity: 0.85; }}
      .legend {{ color: var(--muted); font-size: 0.92rem; line-height: 1.55; }}
      .controls-grid {{ display: grid; gap: 16px; grid-template-columns: 1.2fr 1fr; }}
      .control-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 18px; background: #fffdf9; display: grid; gap: 12px; }}
      .control-label {{ font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
      .control-copy {{ color: var(--muted); line-height: 1.55; }}
      .refinement-input, .additional-input {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        background: #fffdfa;
        color: var(--ink);
      }}
      .doc-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); }}
      .doc-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 16px; background: #fffdf9; display: grid; gap: 10px; min-width: 0; align-content: start; overflow: hidden; }}
      .doc-meta {{ color: var(--muted); font-size: 0.9rem; }}
      .doc-title {{
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.25;
        min-width: 0;
      }}
      .doc-select {{ display: flex; gap: 10px; align-items: center; font-size: 0.95rem; color: var(--ink); }}
      .doc-actions {{ min-width: 0; }}
      .guide {{ color: var(--ink); line-height: 1.5; min-width: 0; overflow-wrap: anywhere; }}
      .callout {{ background: #f8ede0; }}
      .empty {{ color: var(--muted); line-height: 1.6; }}
      .primary-button, .secondary-button {{
        border-radius: 999px;
        border: 1px solid var(--line);
        padding: 10px 16px;
        font: inherit;
        cursor: pointer;
      }}
      .primary-button {{ background: #204d41; color: #fff; border-color: #204d41; }}
      .secondary-button {{ background: #fff8ef; color: var(--ink); }}
      a {{ color: var(--green); text-decoration: none; font-weight: 700; }}
      a:hover {{ text-decoration: underline; }}
      @media (max-width: 980px) {{
        .hero-grid, .controls-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      {banner_markup}
      <form method="post" action="/checkpoint-action">
        <input type="hidden" name="run_id" value="{html.escape(run_id)}" />
        <section class="panel hero-grid">
          <div>
            <p class="eyebrow">Democritus Interactive Mode</p>
            <h1>{query}</h1>
            <p class="trace">Democritus paused at the <strong>{stage_label}</strong>. Curate the corpus here before launching the deeper causal extraction and synthesis stages.</p>
            <div class="checkpoint-chips" style="margin-top:14px;">
              <span class="checkpoint-chip">{selected_count} selected</span>
              <span class="checkpoint-chip">{n_documents} retrieved</span>
              <span class="checkpoint-chip">{len(selected_topics)} topics included</span>
              <span class="checkpoint-chip">{len(rejected_topics)} topics excluded</span>
            </div>
          </div>
          <div class="panel callout">
            <p class="eyebrow">Next Step</p>
            <p class="trace">{summary_text}</p>
            <p class="trace" style="margin-top:12px;">Use <strong>Go deeper on selected</strong> to run the longer Democritus pass only on the documents you keep.</p>
            {drift_comparison_markup}
          </div>
        </section>
        <section class="panel">
          <p class="eyebrow">Atlas Drift Signal</p>
          <p class="trace">This atlas pass is now the explicit anti-drift stage in Democritus. Suspicious topics are weakly aligned to the query anchors below and are good candidates to exclude before the deep run.</p>
          <div class="checkpoint-chips" style="margin-top:14px;">
            <span class="checkpoint-chip">{int(drift_metrics.get("suspicious_topic_count") or 0)} suspicious topic{'s' if int(drift_metrics.get("suspicious_topic_count") or 0) != 1 else ''}</span>
            <span class="checkpoint-chip">{int(drift_metrics.get("aligned_topic_count") or 0)} aligned topic{'s' if int(drift_metrics.get("aligned_topic_count") or 0) != 1 else ''}</span>
            <span class="checkpoint-chip">readiness {html.escape(str(drift_metrics.get("synthesis_readiness_proxy") or 0.0))}</span>
          </div>
          <div class="checkpoint-chips" style="margin-top:12px;">{focus_term_markup}</div>
          <div class="checkpoint-chips" style="margin-top:12px;">{suspicious_topic_markup}</div>
        </section>
        <section class="panel">
          <p class="eyebrow">Corpus Topic Atlas</p>
          <p class="trace">{n_documents} documents reached the root-topic frontier. These recurring themes are the first shared causal surface Democritus recovered.</p>
          <p class="legend" style="margin-top:12px;">Click a topic once to include it in the next retrieval, twice to exclude it, and a third time to clear it.</p>
          <div id="topic-guidance-inputs">{topic_hidden_inputs}</div>
          <div class="chip-row" style="margin-top:14px;">{topic_chips}</div>
        </section>
        <section class="panel">
          <p class="eyebrow">Interactive Controls</p>
          <div class="controls-grid" style="margin-top:12px;">
            <div class="control-card">
              <span class="control-label">Selection</span>
              <p class="control-copy">Keep the documents that look useful for the final deep run. The saved selection persists for this checkpoint.</p>
              <div class="form-actions">
                <button type="submit" class="secondary-button" name="action_kind" value="save">Save selection</button>
                <button type="submit" class="primary-button" name="action_kind" value="deepen">Go deeper on selected</button>
              </div>
            </div>
            <div class="control-card">
              <span class="control-label">Guided Retrieval</span>
              <p class="control-copy">Queue another interactive pass with more documents. Topic choices become soft guidance for the next retrieval, and the optional refinement text still appends to the original query.</p>
              <input class="additional-input" type="number" name="additional_documents" min="1" max="25" step="1" value="3" />
              <input class="refinement-input" type="text" name="retrieval_refinement" value="{refinement_value}" placeholder="Optional retrieval refinement, e.g. peer reviewed natural experiments" />
              <div class="form-actions">
                <button type="submit" class="primary-button" name="action_kind" value="topic_guided_retrieval">Retrieve again from topic choices</button>
                <button type="submit" class="secondary-button" name="action_kind" value="retrieve_more">Retrieve more documents</button>
              </div>
            </div>
          </div>
        </section>
        <section class="panel">
          <p class="eyebrow">Per-Document Topics</p>
          <div class="doc-grid" style="margin-top:12px;">{''.join(document_cards) or '<div class="empty">Root topics have not been materialized yet.</div>'}</div>
        </section>
      </form>
      <script>
        (() => {{
          const hiddenRoot = document.getElementById('topic-guidance-inputs');
          const buttons = Array.from(document.querySelectorAll('.topic-chip-button[data-topic]'));
          if (!hiddenRoot || !buttons.length) {{
            return;
          }}
          const stateByTopic = new Map();
          const cycleState = (current) => {{
            if (current === 'selected') {{
              return 'rejected';
            }}
            if (current === 'rejected') {{
              return 'neutral';
            }}
            return 'selected';
          }};
          hiddenRoot.querySelectorAll('input[data-topic-hidden]').forEach((node) => {{
            const topic = (node.getAttribute('value') || '').trim();
            const state = (node.getAttribute('data-topic-hidden') || '').trim();
            if (topic && state) {{
              stateByTopic.set(topic, state);
            }}
          }});
          const render = () => {{
            hiddenRoot.innerHTML = '';
            buttons.forEach((button) => {{
              const topic = (button.getAttribute('data-topic') || '').trim();
              const state = stateByTopic.get(topic) || 'neutral';
              button.setAttribute('data-topic-state', state);
              button.classList.toggle('is-selected', state === 'selected');
              button.classList.toggle('is-rejected', state === 'rejected');
              if (state === 'neutral') {{
                return;
              }}
              const input = document.createElement('input');
              input.type = 'hidden';
              input.name = state === 'selected' ? 'selected_topic' : 'rejected_topic';
              input.value = topic;
              input.setAttribute('data-topic-hidden', state);
              hiddenRoot.appendChild(input);
            }});
          }};
          buttons.forEach((button) => {{
            button.addEventListener('click', () => {{
              const topic = (button.getAttribute('data-topic') || '').trim();
              if (!topic) {{
                return;
              }}
              stateByTopic.set(topic, cycleState(stateByTopic.get(topic) || button.getAttribute('data-topic-state') || 'neutral'));
              render();
            }});
          }});
          render();
        }})();
      </script>
    </main>
  </body>
</html>"""

    def _render_company_similarity_checkpoint_page(
        self,
        run_id: str,
        *,
        artifact_path: Path,
        banner_message: str = "",
        banner_tone: str = "info",
    ) -> str:
        payload = self._load_company_similarity_checkpoint_payload(artifact_path)
        if not payload:
            return artifact_path.read_text(encoding="utf-8", errors="replace")
        curation = self._load_company_similarity_checkpoint_curation(payload)
        year_start = int(curation.get("year_start") or 2002)
        year_end = int(curation.get("year_end") or year_start)
        suggested = dict(payload.get("suggested_year_window") or {})
        default_window = dict(payload.get("year_window") or {})
        overlap_years = [
            int(year)
            for year in list(payload.get("available_overlap_years") or [])
            if str(year).strip()
        ]
        summary_text = str(payload.get("summary_text") or "").strip()
        partial_preview = dict(payload.get("partial_preview") or {})
        partial_status = html.escape(str(partial_preview.get("status") or "ready").replace("_", " "))
        basis_size = html.escape(str(partial_preview.get("shared_edge_basis_size") or 0))
        summary_path_raw = str(partial_preview.get("summary_path") or "").strip()
        manifest_path_raw = str(partial_preview.get("manifest_path") or "").strip()
        links_markup_parts = []
        if summary_path_raw:
            links_markup_parts.append(
                '<a href="'
                + html.escape(self._launcher_href_for_run_file(run_id, Path(summary_path_raw).resolve()))
                + '" target="_blank" rel="noopener noreferrer">partial summary markdown</a>'
            )
        if manifest_path_raw:
            links_markup_parts.append(
                '<a href="'
                + html.escape(self._launcher_href_for_run_file(run_id, Path(manifest_path_raw).resolve()))
                + '" target="_blank" rel="noopener noreferrer">partial manifest</a>'
            )
        links_markup = (
            '<p class="muted">' + ", ".join(links_markup_parts) + ".</p>"
            if links_markup_parts
            else ""
        )
        overlap_markup = (
            "".join(f'<span class="chip">{html.escape(str(year))}</span>' for year in overlap_years[:20])
            if overlap_years
            else '<span class="chip">No overlap years detected yet</span>'
        )
        banner_markup = (
            f'<section class="banner banner-{html.escape(banner_tone)}">{html.escape(banner_message)}</section>'
            if banner_message
            else ""
        )
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Company Similarity Interactive Checkpoint</title>
    <style>
      :root {{
        --ink: #18222d;
        --muted: #5b6874;
        --paper: #f6f1e8;
        --card: rgba(255,255,255,0.9);
        --line: #d7ccb8;
        --accent: #93451e;
        --green: #1f6a56;
        --soft: #f8ede0;
        --ok: #e8f4ee;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(147,69,30,0.12), transparent 24%),
          linear-gradient(180deg, #fbf7f0 0%, var(--paper) 100%);
      }}
      main {{ width: min(1180px, calc(100vw - 32px)); margin: 32px auto 48px; display: grid; gap: 18px; }}
      .panel, .banner {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 24px; box-shadow: 0 24px 60px rgba(30,25,18,0.08); }}
      .banner-info {{ background: #fdf8ef; }}
      .banner-success {{ background: var(--ok); }}
      .banner-warn {{ background: var(--soft); }}
      .eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      .hero-grid, .control-grid {{ display: grid; gap: 18px; grid-template-columns: 1.2fr 1fr; }}
      .trace {{ color: var(--muted); line-height: 1.6; }}
      h1, h2, h3, p {{ margin: 0; }}
      .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; min-width: 0; }}
      .chip {{ border-radius: 999px; padding: 8px 12px; background: #efe7d9; font-size: 0.92rem; color: #64492b; }}
      .control-card {{ border: 1px solid var(--line); border-radius: 20px; padding: 18px; background: #fffdf9; display: grid; gap: 12px; }}
      .field-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      label {{ display: grid; gap: 8px; color: var(--muted); font-size: 0.92rem; }}
      input {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        background: #fffdfa;
        color: var(--ink);
      }}
      .form-actions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
      .primary-button, .secondary-button {{
        border-radius: 999px;
        border: 1px solid var(--line);
        padding: 10px 16px;
        font: inherit;
        cursor: pointer;
      }}
      .primary-button {{ background: #204d41; color: #fff; border-color: #204d41; }}
      .secondary-button {{ background: #fff8ef; color: var(--ink); }}
      pre {{
        white-space: pre-wrap;
        background: #fbf7ef;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
        color: #425048;
        line-height: 1.55;
      }}
      .muted {{ color: var(--muted); }}
      a {{ color: var(--green); text-decoration: none; font-weight: 700; }}
      @media (max-width: 900px) {{
        .hero-grid, .control-grid, .field-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      {banner_markup}
      <form method="post" action="/checkpoint-action">
        <input type="hidden" name="run_id" value="{html.escape(run_id)}" />
        <section class="panel">
          <p class="eyebrow">Company Similarity Interactive Mode</p>
          <div class="hero-grid">
            <div>
              <h1>{html.escape(str(payload.get("company_a") or "Company A"))} vs {html.escape(str(payload.get("company_b") or "Company B"))}</h1>
              <p class="trace" style="margin-top:12px;">CLIFF paused after the initial similarity read. Adjust the fiscal-year window here before launching the deeper cross-company comparison.</p>
              <div class="chip-row" style="margin-top:14px;">
                <span class="chip">{partial_status}</span>
                <span class="chip">{len(overlap_years)} overlap year{'s' if len(overlap_years) != 1 else ''}</span>
                <span class="chip">{basis_size} shared basis edge{'s' if str(partial_preview.get("shared_edge_basis_size") or 0) != "1" else ''}</span>
              </div>
            </div>
            <div class="control-card">
              <p class="eyebrow">Suggested Window</p>
              <p class="trace">Current checkpoint: <strong>{html.escape(str(default_window.get("start") or ""))}</strong> to <strong>{html.escape(str(default_window.get("end") or ""))}</strong>.</p>
              <p class="trace">Suggested overlap: <strong>{html.escape(str(suggested.get("start") or year_start))}</strong> to <strong>{html.escape(str(suggested.get("end") or year_end))}</strong>.</p>
            </div>
          </div>
        </section>
        <section class="panel">
          <p class="eyebrow">Choose Years</p>
          <div class="control-grid">
            <div class="control-card">
              <div class="field-grid">
                <label>Start year
                  <input type="number" name="company_similarity_year_start" min="2002" max="2100" value="{year_start}" />
                </label>
                <label>End year
                  <input type="number" name="company_similarity_year_end" min="2002" max="2100" value="{year_end}" />
                </label>
              </div>
              <p class="trace">If the deeper run needs more years than the current cached branches cover, CLIFF will rebuild the missing range before the final comparison.</p>
              <div class="form-actions">
                <button type="submit" class="secondary-button" name="action_kind" value="save">Save year window</button>
                <button type="submit" class="primary-button" name="action_kind" value="deepen">Go deeper on this window</button>
              </div>
            </div>
            <div class="control-card">
              <p class="eyebrow">Available Overlap</p>
              <div class="chip-row">{overlap_markup}</div>
            </div>
          </div>
        </section>
      </form>
      <section class="panel">
        <p class="eyebrow">Initial Similarity Read</p>
        <pre>{html.escape(summary_text or "No provisional summary text is available yet.")}</pre>
        {links_markup}
      </section>
    </main>
  </body>
</html>"""

    def _handle_company_similarity_checkpoint_action(
        self,
        *,
        run_id: str,
        artifact_path: Path,
        action_kind: str,
        year_start: int,
        year_end: int,
    ) -> tuple[str, HTTPStatus]:
        payload = self._load_company_similarity_checkpoint_payload(artifact_path)
        if not payload:
            return self._render_text_file_as_html(
                self.config.title,
                "This company-similarity checkpoint is no longer available from the current session.",
            ), HTTPStatus.NOT_FOUND
        year_start = max(2002, int(year_start))
        year_end = max(2002, int(year_end))
        if year_start > year_end:
            year_start, year_end = year_end, year_start
        self._save_company_similarity_checkpoint_curation(
            payload,
            year_start=year_start,
            year_end=year_end,
        )
        if action_kind == "save":
            return self._render_company_similarity_checkpoint_page(
                run_id,
                artifact_path=artifact_path,
                banner_message=f"Saved the year window {year_start} to {year_end} for this checkpoint.",
                banner_tone="success",
            ), HTTPStatus.OK
        if action_kind == "deepen":
            new_run_id = self.request_session_run_deepen(
                run_id,
                company_similarity_year_start=year_start,
                company_similarity_year_end=year_end,
            )
            return self._render_checkpoint_followup_queued_page(
                heading="Deep Company-Similarity Run Queued",
                message=f"Queued a deep company comparison for fiscal years {year_start} through {year_end}.",
                run_id=run_id,
                new_run_id=new_run_id,
            ), HTTPStatus.OK
        return self._render_company_similarity_checkpoint_page(
            run_id,
            artifact_path=artifact_path,
            banner_message="No checkpoint action was applied.",
            banner_tone="warn",
        ), HTTPStatus.BAD_REQUEST

    def _render_checkpoint_followup_queued_page(
        self,
        *,
        heading: str,
        message: str,
        run_id: str,
        new_run_id: str | None,
    ) -> str:
        new_run_markup = (
            f'<p><strong>Queued run:</strong> {html.escape(new_run_id)}</p>'
            if new_run_id
            else ""
        )
        artifact_link = (
            f'<p><a href="/run-artifact?run_id={html.escape(new_run_id)}" target="_blank" rel="noopener noreferrer">Open the new run</a></p>'
            if new_run_id
            else ""
        )
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="1; url=/" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(self.config.title)}</title>
    <style>
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; background: #f7f1e5; color: #1a1f1a; }}
      main {{ max-width: 860px; margin: 48px auto; padding: 0 18px; }}
      .card {{ background: #fffdf6; border: 1px solid #d5c8a4; border-radius: 24px; padding: 24px; }}
      h1 {{ margin: 0 0 12px; }}
      p {{ margin: 12px 0 0; color: #5e675d; line-height: 1.6; }}
      a {{ color: #1f6a56; font-weight: 700; }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>{html.escape(heading)}</h1>
        <p>{html.escape(message)}</p>
        {new_run_markup}
        {artifact_link}
        <p><a href="/">Return to the CLIFF session</a></p>
        <p><a href="/run-artifact?run_id={html.escape(run_id)}">Back to the checkpoint</a></p>
      </section>
    </main>
  </body>
</html>"""

    def _handle_checkpoint_action(
        self,
        *,
        run_id: str,
        action_kind: str,
        selected_pdf_paths: tuple[str, ...],
        selected_topics: tuple[str, ...],
        rejected_topics: tuple[str, ...],
        additional_documents: int,
        retrieval_refinement: str,
        company_similarity_year_start: int | None = None,
        company_similarity_year_end: int | None = None,
    ) -> tuple[str, HTTPStatus]:
        with self._lock:
            run_state = dict(self._session_runs_by_id.get(run_id) or {})
        artifact_path_value = str(run_state.get("artifact_path") or "").strip()
        artifact_path = Path(artifact_path_value).resolve() if artifact_path_value else None
        if artifact_path is None or not artifact_path.exists():
            body = self._render_text_file_as_html(
                self.config.title,
                "This interactive checkpoint is no longer available from the current session.",
            )
            return body, HTTPStatus.NOT_FOUND
        if self._company_similarity_checkpoint_manifest_for_html(artifact_path) is not None:
            return self._handle_company_similarity_checkpoint_action(
                run_id=run_id,
                artifact_path=artifact_path,
                action_kind=action_kind,
                year_start=int(company_similarity_year_start or 2002),
                year_end=int(company_similarity_year_end or company_similarity_year_start or 2002),
            )
        payload = self._load_democritus_checkpoint_payload(artifact_path)
        documents = list(payload.get("documents") or [])
        valid_paths = {
            " ".join(str(dict(item).get("pdf_path") or "").split()).strip()
            for item in documents
            if " ".join(str(dict(item).get("pdf_path") or "").split()).strip()
        }
        filtered_selected = tuple(
            path for path in dict.fromkeys(selected_pdf_paths) if path in valid_paths
        )
        refinement = " ".join(str(retrieval_refinement or "").split()).strip()
        filtered_selected_topics, filtered_rejected_topics = self._filter_checkpoint_topics(
            payload,
            selected_topics=self._unique_checkpoint_topics(selected_topics),
            rejected_topics=self._unique_checkpoint_topics(rejected_topics),
        )
        self._save_democritus_checkpoint_curation(
            payload,
            selected_pdf_paths=filtered_selected,
            selected_topics=filtered_selected_topics,
            rejected_topics=filtered_rejected_topics,
            retrieval_refinement=refinement,
        )
        if action_kind == "save":
            count = len(filtered_selected)
            tone = "success" if count else "warn"
            message = (
                f"Saved the checkpoint selection for {count} document{'s' if count != 1 else ''}."
                if count
                else "No documents are currently selected. Choose at least one document before going deeper."
            )
            self._record_democritus_checkpoint_telemetry(
                payload,
                run_state=run_state,
                action_kind="save",
                action_status=("saved" if count else "empty_selection"),
                selected_pdf_paths=filtered_selected,
                selected_topics=filtered_selected_topics,
                rejected_topics=filtered_rejected_topics,
                retrieval_refinement=refinement,
                additional_documents=additional_documents,
            )
            return self._render_democritus_checkpoint_page(
                run_id,
                artifact_path=artifact_path,
                banner_message=message,
                banner_tone=tone,
            ), HTTPStatus.OK
        if action_kind == "deepen":
            if not filtered_selected:
                self._record_democritus_checkpoint_telemetry(
                    payload,
                    run_state=run_state,
                    action_kind="deepen",
                    action_status="blocked_empty_selection",
                    selected_pdf_paths=filtered_selected,
                    selected_topics=filtered_selected_topics,
                    rejected_topics=filtered_rejected_topics,
                    retrieval_refinement=refinement,
                    additional_documents=additional_documents,
                )
                return self._render_democritus_checkpoint_page(
                    run_id,
                    artifact_path=artifact_path,
                    banner_message="Select at least one document before launching the deeper Democritus pass.",
                    banner_tone="warn",
                ), HTTPStatus.BAD_REQUEST
            curated_manifest_path = self._write_democritus_curated_manifest(
                payload,
                selected_pdf_paths=filtered_selected,
            )
            base_query = self._democritus_base_query(payload, run_state)
            new_run_id = self.request_session_run_deepen(
                run_id,
                democritus_manifest_path=curated_manifest_path,
                democritus_target_docs=len(filtered_selected),
                democritus_base_query=base_query,
                democritus_selected_topics=filtered_selected_topics,
                democritus_rejected_topics=filtered_rejected_topics,
                democritus_retrieval_refinement=refinement,
            )
            self._record_democritus_checkpoint_telemetry(
                payload,
                run_state=run_state,
                action_kind="deepen",
                action_status=("queued" if new_run_id else "not_queued"),
                selected_pdf_paths=filtered_selected,
                selected_topics=filtered_selected_topics,
                rejected_topics=filtered_rejected_topics,
                retrieval_refinement=refinement,
                additional_documents=additional_documents,
                queued_followup_run_id=new_run_id,
                curated_manifest_path=curated_manifest_path,
            )
            return self._render_checkpoint_followup_queued_page(
                heading="Deep Democritus Run Queued",
                message=f"Queued a deep run on {len(filtered_selected)} selected document{'s' if len(filtered_selected) != 1 else ''}.",
                run_id=run_id,
                new_run_id=new_run_id,
            ), HTTPStatus.OK
        if action_kind == "topic_guided_retrieval":
            base_query = self._democritus_base_query(payload, run_state)
            if not (filtered_selected_topics or filtered_rejected_topics or refinement):
                self._record_democritus_checkpoint_telemetry(
                    payload,
                    run_state=run_state,
                    action_kind="topic_guided_retrieval",
                    action_status="blocked_missing_guidance",
                    selected_pdf_paths=filtered_selected,
                    selected_topics=filtered_selected_topics,
                    rejected_topics=filtered_rejected_topics,
                    retrieval_refinement=refinement,
                    additional_documents=additional_documents,
                )
                return self._render_democritus_checkpoint_page(
                    run_id,
                    artifact_path=artifact_path,
                    banner_message="Choose at least one topic or add refinement text before launching a topic-guided retrieval.",
                    banner_tone="warn",
                ), HTTPStatus.BAD_REQUEST
            base_count = max(1, len(documents))
            target_documents = base_count + max(1, int(additional_documents or 1))
            query = self._canonical_guided_query(
                base_query=base_query,
                selected_topics=filtered_selected_topics,
                rejected_topics=filtered_rejected_topics,
                retrieval_refinement=refinement,
            )
            new_run_id = self.request_session_run_retrieve_more(
                run_id,
                query_override=query,
                democritus_target_docs=target_documents,
                democritus_atlas_baseline=self._atlas_drift_metrics(payload),
                democritus_base_query=base_query,
                democritus_selected_topics=filtered_selected_topics,
                democritus_rejected_topics=filtered_rejected_topics,
                democritus_retrieval_refinement=refinement,
            )
            self._record_democritus_checkpoint_telemetry(
                payload,
                run_state=run_state,
                action_kind="topic_guided_retrieval",
                action_status=("queued" if new_run_id else "not_queued"),
                selected_pdf_paths=filtered_selected,
                selected_topics=filtered_selected_topics,
                rejected_topics=filtered_rejected_topics,
                retrieval_refinement=refinement,
                additional_documents=additional_documents,
                queued_followup_run_id=new_run_id,
            )
            return self._render_checkpoint_followup_queued_page(
                heading="Topic-Guided Retrieval Queued",
                message=(
                    f"Queued another interactive run targeting {target_documents} documents"
                    + (" using your topic guidance." if filtered_selected_topics or filtered_rejected_topics else ".")
                ),
                run_id=run_id,
                new_run_id=new_run_id,
            ), HTTPStatus.OK
        if action_kind == "retrieve_more":
            base_count = max(1, len(documents))
            target_documents = base_count + max(1, int(additional_documents or 1))
            base_query = self._democritus_base_query(payload, run_state)
            query = self._canonical_guided_query(
                base_query=base_query,
                selected_topics=(),
                rejected_topics=(),
                retrieval_refinement=refinement,
            )
            new_run_id = self.request_session_run_retrieve_more(
                run_id,
                query_override=query,
                democritus_target_docs=target_documents,
                democritus_atlas_baseline=self._atlas_drift_metrics(payload),
                democritus_base_query=base_query,
                democritus_retrieval_refinement=refinement,
            )
            self._record_democritus_checkpoint_telemetry(
                payload,
                run_state=run_state,
                action_kind="retrieve_more",
                action_status=("queued" if new_run_id else "not_queued"),
                selected_pdf_paths=filtered_selected,
                selected_topics=filtered_selected_topics,
                rejected_topics=filtered_rejected_topics,
                retrieval_refinement=refinement,
                additional_documents=additional_documents,
                queued_followup_run_id=new_run_id,
            )
            return self._render_checkpoint_followup_queued_page(
                heading="Interactive Retrieval Expansion Queued",
                message=(
                    f"Queued another interactive run targeting {target_documents} documents"
                    + (f" with the refinement '{refinement}'." if refinement else ".")
                ),
                run_id=run_id,
                new_run_id=new_run_id,
            ), HTTPStatus.OK
        return self._render_democritus_checkpoint_page(
            run_id,
            artifact_path=artifact_path,
            banner_message="No checkpoint action was applied.",
            banner_tone="warn",
        ), HTTPStatus.BAD_REQUEST

    def request_session_run_stop(self, run_id: str) -> bool:
        handler = None
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None:
                return False
            if str(run_state.get("status") or "") in {"complete", "failed", "stopped"}:
                return False
            run_state["status"] = "stopping"
            run_state["mind_layer"] = "conscious"
            run_state["note"] = "Stop requested from CLIFF's conscious layer."
            handler = self.config.run_control_handler
        if handler is not None:
            try:
                handler("stop", run_id)
            except Exception:
                return False
        return True

    def _make_handler(self):
        launcher = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(launcher._render_launcher_page())
                    return
                if parsed.path == "/artifact":
                    self._send_html(launcher._render_artifact_page())
                    return
                if parsed.path == "/run-artifact":
                    run_id = " ".join(parse_qs(parsed.query).get("run_id", [""])[0].split()).strip()
                    self._send_html(launcher._render_run_artifact_page(run_id), status=HTTPStatus.OK)
                    return
                if parsed.path == "/run-file":
                    query = parse_qs(parsed.query)
                    run_id = " ".join(query.get("run_id", [""])[0].split()).strip()
                    requested_path = " ".join(query.get("path", [""])[0].split()).strip()
                    body, content_type, status = launcher._render_run_file_response(run_id, requested_path)
                    self._send_bytes(body, content_type=content_type, status=status)
                    return
                if parsed.path == "/state":
                    self._send_json(launcher._state_payload())
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/checkpoint-action":
                    content_length = int(self.headers.get("Content-Length") or "0")
                    payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                    parsed_payload = parse_qs(payload)
                    run_id = " ".join(parsed_payload.get("run_id", [""])[0].split()).strip()
                    action_kind = " ".join(parsed_payload.get("action_kind", [""])[0].split()).strip().lower()
                    selected_pdf_paths = tuple(
                        " ".join(item.split()).strip()
                        for item in parsed_payload.get("selected_pdf_path", [])
                        if " ".join(item.split()).strip()
                    )
                    selected_topics = tuple(
                        " ".join(item.split()).strip()
                        for item in parsed_payload.get("selected_topic", [])
                        if " ".join(item.split()).strip()
                    )
                    rejected_topics = tuple(
                        " ".join(item.split()).strip()
                        for item in parsed_payload.get("rejected_topic", [])
                        if " ".join(item.split()).strip()
                    )
                    additional_raw = " ".join(parsed_payload.get("additional_documents", ["3"])[0].split()).strip()
                    try:
                        additional_documents = max(1, int(additional_raw or "3"))
                    except ValueError:
                        additional_documents = 3
                    retrieval_refinement = parsed_payload.get("retrieval_refinement", [""])[0]
                    year_start_raw = " ".join(parsed_payload.get("company_similarity_year_start", [""])[0].split()).strip()
                    year_end_raw = " ".join(parsed_payload.get("company_similarity_year_end", [""])[0].split()).strip()
                    try:
                        company_similarity_year_start = int(year_start_raw) if year_start_raw else None
                    except ValueError:
                        company_similarity_year_start = None
                    try:
                        company_similarity_year_end = int(year_end_raw) if year_end_raw else None
                    except ValueError:
                        company_similarity_year_end = None
                    body, status = launcher._handle_checkpoint_action(
                        run_id=run_id,
                        action_kind=action_kind,
                        selected_pdf_paths=selected_pdf_paths,
                        selected_topics=selected_topics,
                        rejected_topics=rejected_topics,
                        additional_documents=additional_documents,
                        retrieval_refinement=retrieval_refinement,
                        company_similarity_year_start=company_similarity_year_start,
                        company_similarity_year_end=company_similarity_year_end,
                    )
                    self._send_html(body, status=status)
                    return
                if parsed.path == "/deepen-run":
                    content_length = int(self.headers.get("Content-Length") or "0")
                    payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                    run_id = " ".join(parse_qs(payload).get("run_id", [""])[0].split()).strip()
                    new_run_id = launcher.request_session_run_deepen(run_id)
                    self._send_json({"ok": bool(new_run_id), "run_id": run_id, "new_run_id": new_run_id})
                    return
                if parsed.path == "/stop-run":
                    content_length = int(self.headers.get("Content-Length") or "0")
                    payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                    run_id = " ".join(parse_qs(payload).get("run_id", [""])[0].split()).strip()
                    ok = launcher.request_session_run_stop(run_id)
                    self._send_json({"ok": ok, "run_id": run_id})
                    return
                if parsed.path == "/rerun-archived":
                    content_length = int(self.headers.get("Content-Length") or "0")
                    payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                    run_id = " ".join(parse_qs(payload).get("run_id", [""])[0].split()).strip()
                    new_run_id = launcher.request_archived_run_rerun(run_id)
                    self._send_json({"ok": bool(new_run_id), "run_id": run_id, "new_run_id": new_run_id})
                    return
                if parsed.path == "/wrong-route":
                    content_length = int(self.headers.get("Content-Length") or "0")
                    payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                    run_id = " ".join(parse_qs(payload).get("run_id", [""])[0].split()).strip()
                    new_run_id = launcher.request_run_wrong_route(run_id)
                    self._send_json({"ok": bool(new_run_id), "run_id": run_id, "new_run_id": new_run_id})
                    return
                if parsed.path != "/submit":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_length = int(self.headers.get("Content-Length") or "0")
                payload = self.rfile.read(content_length).decode("utf-8", errors="replace")
                parsed_payload = parse_qs(payload)
                query = " ".join(parsed_payload.get("query", [""])[0].split()).strip()
                execution_mode = launcher._normalize_execution_mode(
                    parsed_payload.get("execution_mode", [launcher.config.default_execution_mode])[0]
                )
                if not query:
                    self._send_html(
                        launcher._render_launcher_page(error_message="Enter a query before starting the run."),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                llm_token_budget = launcher._normalize_llm_token_budget(
                    parsed_payload.get("llm_token_budget", [""])[0]
                )
                submission_overrides = (
                    {"llm_token_budget": llm_token_budget}
                    if llm_token_budget is not None
                    else None
                )
                launcher.submit_query(
                    query,
                    execution_mode=execution_mode,
                    submission_overrides=submission_overrides,
                )
                self._send_html(launcher._render_launcher_page())

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                del format, args

            def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
                encoded = body.encode("utf-8")
                self._send_bytes(encoded, content_type="text/html; charset=utf-8", status=status)

            def _send_bytes(
                self,
                body: bytes,
                *,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, object]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler

    def _state_payload(self) -> dict[str, object]:
        with self._lock:
            self._refresh_archived_runs()
            payload = {
                "session_mode": self.config.session_mode,
                "execution_mode_enabled": self.config.enable_execution_mode,
                "query_received": self._query_received,
                "artifact_ready": bool(self._artifact_path and self._artifact_path.exists()),
                "artifact_path": str(self._artifact_path) if self._artifact_path else None,
            }
            if self.config.session_mode:
                payload["runs"] = [self._enriched_run_state(dict(item)) for item in self._session_runs]
                payload["archived_runs"] = [self._enriched_run_state(dict(item)) for item in self._archived_runs]
            return payload

    def _democritus_telemetry(self, run_state: dict[str, object]) -> dict[str, object]:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return {}
        telemetry_path = (Path(outdir_value).resolve() / "democritus" / "democritus_runs" / "telemetry.json").resolve()
        if not telemetry_path.exists():
            return {}
        try:
            return dict(json.loads(telemetry_path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    def _route_llm_usage(self, run_state: dict[str, object]) -> dict[str, object]:
        route_name = str(run_state.get("route_name") or "").strip()
        telemetry: dict[str, object] = {}
        if route_name == "democritus":
            telemetry = self._democritus_telemetry(run_state)
        elif route_name == "company_similarity":
            telemetry = self._company_similarity_telemetry(run_state)
        llm_usage = dict(telemetry.get("llm_usage") or {})
        return llm_usage

    @staticmethod
    def _llm_usage_summary(llm_usage: dict[str, object]) -> dict[str, object]:
        request_count = int(llm_usage.get("request_count") or 0)
        total_tokens = int(llm_usage.get("total_tokens") or 0)
        requests_with_usage = int(llm_usage.get("requests_with_usage") or 0)
        if request_count <= 0 and total_tokens <= 0:
            return {}
        tracked_requests = requests_with_usage or request_count
        label = f"{total_tokens:,} tokens across {tracked_requests:,} LLM request"
        if tracked_requests != 1:
            label += "s"
        if request_count > tracked_requests:
            label += f" ({request_count - tracked_requests} pending usage rows)"
        return {
            "request_count": request_count,
            "requests_with_usage": requests_with_usage,
            "total_tokens": total_tokens,
            "label": label,
        }

    def _llm_budget_summary(self, run_state: dict[str, object], llm_usage: dict[str, object]) -> dict[str, object]:
        budget_tokens = self._run_llm_token_budget(run_state)
        if budget_tokens is None:
            return {}
        spent_tokens = int(llm_usage.get("total_tokens") or 0)
        remaining_tokens = max(0, budget_tokens - spent_tokens)
        label = f"{spent_tokens:,} of {budget_tokens:,} tokens used · {remaining_tokens:,} remaining"
        if spent_tokens > budget_tokens:
            label += f" ({spent_tokens - budget_tokens:,} over budget)"
        return {
            "budget_tokens": budget_tokens,
            "spent_tokens": spent_tokens,
            "remaining_tokens": remaining_tokens,
            "exhausted": spent_tokens >= budget_tokens,
            "label": label,
        }

    def _run_eta_summary(self, run_state: dict[str, object]) -> dict[str, object]:
        route_name = str(run_state.get("route_name") or "").strip()
        status = str(run_state.get("status") or "").strip().lower()
        if status not in {"queued", "routing", "running", "stopping"}:
            return {}
        if route_name == "democritus":
            telemetry = self._democritus_telemetry(run_state)
            timing = dict(telemetry.get("timing") or {})
            eta_ready = bool(timing.get("eta_ready"))
            current_stage = str(timing.get("current_stage") or "").strip()
            current_parallelism = timing.get("effective_parallelism")
            peak_parallelism = timing.get("peak_parallelism")
            return {
                "eta_human": str(timing.get("eta_human") or ("warming up" if not eta_ready else "n/a")),
                "eta_label": (
                    f"about {timing.get('eta_human')} remaining"
                    if eta_ready and timing.get("eta_human")
                    else "ETA warming up"
                ),
                "parallelism": current_parallelism,
                "parallelism_label": (
                    f"parallelism {current_parallelism}"
                    + (f" (peak {peak_parallelism})" if peak_parallelism is not None else "")
                    if current_parallelism is not None
                    else ""
                ),
                "current_stage": current_stage,
                "current_stage_label": (f"stage {current_stage}" if current_stage else ""),
                "peak_parallelism": peak_parallelism,
            }
        if route_name == "company_similarity":
            progress = self._company_similarity_progress(run_state)
            eta = self._company_similarity_eta(run_state, progress)
            telemetry = dict(progress.get("telemetry") or {})
            timing = dict(telemetry.get("timing") or {})
            progress_stage = str(progress.get("current_phase") or "").strip()
            if progress_stage in {"Preparing company inputs", "Preparing First company and Second company"}:
                progress_stage = ""
            current_stage = str(
                self._company_similarity_democritus_stage_summary(progress)
                or progress_stage
                or timing.get("current_stage")
                or ""
            ).strip()
            current_parallelism = timing.get("observed_parallelism")
            peak_parallelism = timing.get("peak_parallelism")
            return {
                "eta_human": str(eta.get("eta_human") or "warming up"),
                "eta_label": str(eta.get("eta_label") or "ETA warming up"),
                "parallelism": current_parallelism,
                "parallelism_label": (
                    f"parallelism {current_parallelism}"
                    + (f" (peak {peak_parallelism})" if peak_parallelism is not None else "")
                    if current_parallelism is not None
                    else ""
                ),
                "current_stage": current_stage,
                "current_stage_label": (f"stage {current_stage}" if current_stage else ""),
                "peak_parallelism": peak_parallelism,
            }
        return {}

    def _enriched_run_state(self, run_state: dict[str, object]) -> dict[str, object]:
        eta_summary = self._run_eta_summary(run_state)
        if eta_summary:
            run_state["eta_human"] = eta_summary.get("eta_human")
            run_state["eta_label"] = eta_summary.get("eta_label")
            run_state["parallelism"] = eta_summary.get("parallelism")
            run_state["parallelism_label"] = eta_summary.get("parallelism_label")
            run_state["current_stage"] = eta_summary.get("current_stage")
            run_state["current_stage_label"] = eta_summary.get("current_stage_label")
            run_state["peak_parallelism"] = eta_summary.get("peak_parallelism")
        llm_usage = self._route_llm_usage(run_state)
        llm_budget_summary = self._llm_budget_summary(run_state, llm_usage)
        if llm_budget_summary:
            run_state["llm_budget_label"] = llm_budget_summary.get("label")
            run_state["llm_budget_tokens"] = llm_budget_summary.get("budget_tokens")
            run_state["llm_spent_tokens"] = llm_budget_summary.get("spent_tokens")
            run_state["llm_remaining_tokens"] = llm_budget_summary.get("remaining_tokens")
        llm_usage_summary = self._llm_usage_summary(llm_usage)
        if llm_usage_summary:
            run_state["llm_usage_label"] = llm_usage_summary.get("label")
            run_state["llm_total_tokens"] = llm_usage_summary.get("total_tokens")
            run_state["llm_request_count"] = llm_usage_summary.get("request_count")
        research_profile = self._route_research_profile(run_state.get("route_name"))
        run_state["research_profile_class"] = research_profile.get("class_name")
        run_state["research_profile_label"] = research_profile.get("label")
        run_state["research_profile_note"] = research_profile.get("note")
        return run_state

    def _render_artifact_page(self) -> str:
        with self._lock:
            artifact_path = self._artifact_path
        if artifact_path and artifact_path.exists():
            return artifact_path.read_text(encoding="utf-8")
        title = html.escape(self.config.title)
        message = html.escape(self.config.waiting_message)
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="{_ARTIFACT_REFRESH_SECONDS}" />
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #1a1f1a;
        --muted: #5e675d;
        --paper: #f7f1e5;
        --card: #fffdf6;
        --line: #d5c8a4;
        --accent: #8a3b12;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: radial-gradient(circle at top, #fff7dd 0, var(--paper) 48%, #efe4cf 100%);
        color: var(--ink);
      }}
      main {{
        max-width: 880px;
        margin: 64px auto;
        padding: 32px;
      }}
      .card {{
        background: rgba(255, 253, 246, 0.94);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 18px 50px rgba(57, 40, 16, 0.12);
      }}
      h1 {{
        margin: 0 0 12px 0;
        font-size: 2rem;
      }}
      p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>{title}</h1>
        <p>{message}</p>
      </section>
    </main>
  </body>
</html>
"""

    def _render_run_artifact_page(self, run_id: str) -> str:
        run_state = self._enriched_run_state(self._lookup_run_state(run_id))
        run_status = str(run_state.get("status") or "").strip()
        route_name = str(run_state.get("route_name") or "").strip()
        artifact_path_value = str(run_state.get("artifact_path") or "").strip()
        artifact_path = Path(artifact_path_value) if artifact_path_value else None
        if route_name == "company_similarity" and run_status in {"queued", "routing", "running", "stopping"}:
            return self._render_company_similarity_live_page(run_id, run_state=run_state)
        if artifact_path and run_status in {"queued", "routing", "running", "stopping"}:
            return self._render_live_run_artifact_shell(
                run_id,
                artifact_path=artifact_path,
                run_state=run_state,
            )
        if artifact_path and artifact_path.exists():
            return self._render_html_file_for_run(run_id, artifact_path)
        title = html.escape(self.config.title)
        message = html.escape(
            "This run does not have an accessible artifact yet. Return to the CLIFF session and wait for the run to complete."
        )
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: #f7f1e5;
        color: #1a1f1a;
      }}
      main {{
        max-width: 880px;
        margin: 64px auto;
        padding: 32px;
      }}
      .card {{
        background: #fffdf6;
        border: 1px solid #d5c8a4;
        border-radius: 24px;
        padding: 28px;
      }}
      p {{
        margin: 0;
        color: #5e675d;
        line-height: 1.6;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p>{message}</p>
      </section>
    </main>
  </body>
</html>
"""

    def _company_similarity_live_files(self, run_state: dict[str, object]) -> tuple[Path, ...]:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return ()
        run_root = Path(outdir_value).resolve()
        route_root = run_root / "company_similarity"
        candidates = [
            run_root / "cliff_worker_first_pass_stdout.log",
            run_root / "cliff_worker_first_pass_stderr.log",
            run_root / "cliff_worker_synthesis_pass_stdout.log",
            run_root / "cliff_worker_synthesis_pass_stderr.log",
            route_root / "company_similarity_summary.json",
            route_root / "company_similarity_telemetry.json",
            route_root / "company_similarity_performance.html",
            route_root / "company_similarity_dashboard.html",
        ]
        if route_root.exists():
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/*.md")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/*.json")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/*.csv")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/*.png")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/partial/*.md")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/partial/*.json")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/partial/*.csv")))
            candidates.extend(sorted(route_root.glob("*_vs_*_functors/partial/*.png")))
        seen: set[Path] = set()
        ordered: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.exists() or resolved in seen:
                continue
            seen.add(resolved)
            ordered.append(resolved)
        return tuple(ordered)

    def _company_similarity_stdout_log(self, run_state: dict[str, object]) -> Path | None:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return None
        candidate = (Path(outdir_value).resolve() / "cliff_worker_first_pass_stdout.log").resolve()
        return candidate if candidate.exists() else None

    def _company_similarity_log_lines(self, run_state: dict[str, object]) -> tuple[str, ...]:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return ()
        run_root = Path(outdir_value).resolve()
        candidates = (
            run_root / "cliff_worker_first_pass_stdout.log",
            run_root / "cliff_worker_first_pass_stderr.log",
            run_root / "cliff_worker_synthesis_pass_stdout.log",
            run_root / "cliff_worker_synthesis_pass_stderr.log",
        )
        lines: list[str] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                lines.extend(candidate.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
        return tuple(lines)

    def _company_similarity_telemetry(self, run_state: dict[str, object]) -> dict[str, object]:
        outdir_value = str(run_state.get("outdir") or "").strip()
        if not outdir_value:
            return {}
        telemetry_path = (Path(outdir_value).resolve() / "company_similarity" / "company_similarity_telemetry.json").resolve()
        if not telemetry_path.exists():
            return {}
        try:
            return dict(json.loads(telemetry_path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    def _company_similarity_partial_preview(self, run_state: dict[str, object]) -> dict[str, object]:
        telemetry = self._company_similarity_telemetry(run_state)
        preview = dict(telemetry.get("partial_preview") or {})
        summary_path_raw = str(preview.get("summary_path") or "").strip()
        manifest_path_raw = str(preview.get("manifest_path") or "").strip()
        summary_text = ""
        if summary_path_raw:
            summary_path = Path(summary_path_raw).resolve()
            if summary_path.exists():
                try:
                    summary_text = summary_path.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    summary_text = ""
        preview["summary_text"] = summary_text
        preview["summary_path"] = summary_path_raw
        preview["manifest_path"] = manifest_path_raw
        return preview

    @staticmethod
    def _company_similarity_stream_parts(line: str) -> tuple[str, str]:
        stripped = line.strip()
        match = re.match(
            r"^\[company_similarity\](?:\[(?P<label>[^\]]+)\])?\s*(?P<body>.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if not match:
            return "", stripped
        return str(match.group("label") or "").strip(), str(match.group("body") or "").strip()

    @staticmethod
    def _company_similarity_activity_label(line: str) -> str:
        stream_label, stripped = DashboardQueryLauncher._company_similarity_stream_parts(line)
        prefix = f"{stream_label}: " if stream_label else ""
        if stripped.startswith("[run_brand_financial_filings] "):
            return prefix + "filings pipeline: " + stripped[len("[run_brand_financial_filings] ") :]
        return prefix + stripped

    @staticmethod
    def _format_eta_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 60:
            return "under 1 min"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes} min" if seconds < 30 else f"{minutes + 1} min"
        hours, minutes = divmod(minutes, 60)
        if minutes == 0:
            return f"{hours} hr"
        return f"{hours} hr {minutes} min"

    def _company_similarity_eta(self, run_state: dict[str, object], progress: dict[str, object]) -> dict[str, object]:
        phase_seconds = {
            "query": 20.0,
            "company_a": 8.0 * 60.0,
            "company_b": 8.0 * 60.0,
            "functors": 4.0 * 60.0,
            "viz": 90.0,
            "done": 15.0,
        }
        phase_weights = {
            "query": 0.05,
            "company_a": 0.30,
            "company_b": 0.30,
            "functors": 0.22,
            "viz": 0.10,
            "done": 0.03,
        }
        phases = tuple(progress.get("phases") or ())
        progress_fraction = 0.0
        fallback_remaining = 0.0
        company_build_remaining: list[float] = []
        active_phase_key = ""

        for key, _label, state in phases:
            progress_fraction += phase_weights.get(str(key), 0.0) * (
                1.0 if state == "complete" else 0.35 if state == "active" else 0.0
            )
            if state == "active" and not active_phase_key:
                active_phase_key = str(key)
            if state == "complete":
                continue
            phase_remaining = phase_seconds.get(str(key), 0.0) * (0.55 if state == "active" else 1.0)
            if str(key) in {"company_a", "company_b"}:
                company_build_remaining.append(phase_remaining)
            else:
                fallback_remaining += phase_remaining
        if company_build_remaining:
            fallback_remaining += max(company_build_remaining)

        created_at = float(run_state.get("created_at") or time.time())
        elapsed_seconds = max(0.0, time.time() - created_at)
        eta_ready = progress_fraction >= 0.15 or bool(progress.get("latest_batch")) or bool(progress.get("staged_years"))
        if progress_fraction >= 0.08 and elapsed_seconds >= 20.0:
            inferred_remaining = elapsed_seconds * max(0.0, (1.0 - progress_fraction) / max(progress_fraction, 0.05))
            remaining_seconds = min(
                max(fallback_remaining * 0.4, inferred_remaining),
                max(fallback_remaining * 1.6, inferred_remaining),
            )
            eta_ready = True
        else:
            remaining_seconds = fallback_remaining

        if any(state != "complete" for _key, _label, state in phases):
            remaining_seconds = max(45.0 if eta_ready else 90.0, remaining_seconds)
        else:
            remaining_seconds = 0.0

        if progress.get("staged_years") and not progress.get("latest_batch") and active_phase_key in {"company_a", "company_b"}:
            remaining_seconds = max(60.0, remaining_seconds * 0.85)

        return {
            "eta_ready": eta_ready,
            "remaining_seconds": round(remaining_seconds, 1),
            "eta_human": self._format_eta_duration(remaining_seconds) if eta_ready else "warming up",
            "eta_label": (
                f"about {self._format_eta_duration(remaining_seconds)} remaining"
                if eta_ready
                else "warming up from first progress signals"
            ),
            "elapsed_human": self._format_eta_duration(elapsed_seconds),
        }

    def _company_similarity_progress(self, run_state: dict[str, object]) -> dict[str, object]:
        log_path = self._company_similarity_stdout_log(run_state)
        lines = list(self._company_similarity_log_lines(run_state))
        telemetry = self._company_similarity_telemetry(run_state)
        company_a = ""
        company_b = ""
        query_resolved = False
        first_company_started = False
        first_company_ready = False
        second_company_started = False
        second_company_ready = False
        comparison_started = False
        comparison_completed = False
        visualization_started = False
        visualization_completed = False
        dashboard_ready = False
        staged_year_values: set[tuple[str, str]] = set()
        atlas_year_values: set[tuple[str, str]] = set()
        latest_year = ""
        latest_atlas_year = ""
        active_companies: list[str] = []
        active_atlas_companies: list[str] = []
        company_substage: dict[str, str] = {}
        company_triples: dict[str, int] = {}
        latest_batch = ""
        recent_lines = [
            self._company_similarity_activity_label(line)
            for line in lines
            if line.strip()
        ][-8:]
        if not recent_lines:
            for stage in list(telemetry.get("stages") or []):
                if not isinstance(stage, dict):
                    continue
                if str(stage.get("status") or "") == "complete":
                    recent_lines.append(f"{stage.get('label')}: {stage.get('duration_human')}")
            recent_lines = recent_lines[-8:]

        for line in lines:
            stream_label, parsed_line = self._company_similarity_stream_parts(line)
            lowered = parsed_line.lower()
            resolved_match = re.search(r"resolved query to\s+(.+?)\s+vs\s+(.+)$", parsed_line, flags=re.IGNORECASE)
            if resolved_match:
                company_a = resolved_match.group(1).strip()
                company_b = resolved_match.group(2).strip()
                query_resolved = True
            start_match = re.search(r"ensuring company analysis for\s+(.+)$", parsed_line, flags=re.IGNORECASE)
            if start_match:
                company = start_match.group(1).strip()
                if company_a and company.lower() == company_a.lower():
                    first_company_started = True
                elif company_b and company.lower() == company_b.lower():
                    second_company_started = True
                elif not company_a:
                    company_a = company
                    first_company_started = True
                elif not company_b:
                    company_b = company
                    second_company_started = True
                if company not in active_companies:
                    active_companies.append(company)
            ready_match = re.search(r"company analysis ready for\s+(.+?):", parsed_line, flags=re.IGNORECASE)
            if ready_match:
                company = ready_match.group(1).strip()
                if company_a and company.lower() == company_a.lower():
                    first_company_ready = True
                elif company_b and company.lower() == company_b.lower():
                    second_company_ready = True
                elif not company_a:
                    company_a = company
                    first_company_ready = True
                elif not company_b:
                    company_b = company
                    second_company_ready = True
                active_companies = [item for item in active_companies if item.lower() != company.lower()]
            if "running cross-company functor analysis" in lowered:
                comparison_started = True
                active_companies = []
            if "cross-company functor analysis completed" in lowered:
                comparison_started = True
                comparison_completed = True
            if "visualize_cross_company_functors" in lowered:
                visualization_started = True
            if "cross-company visualization completed" in lowered:
                visualization_started = True
                visualization_completed = True
            if "company similarity dashboard ready:" in lowered:
                dashboard_ready = True
            if "[run_brand_financial_filings] year=" in parsed_line and "staged_pdfs=" in parsed_line:
                year_match = re.search(r"year=(\d{4})", parsed_line)
                if year_match:
                    latest_year = year_match.group(1)
                    staged_year_values.add((stream_label or "unknown", latest_year))
            if "pipelines.batch_pipeline" in parsed_line:
                batch_match = re.search(r"year_(\d{4})", parsed_line)
                if batch_match:
                    latest_batch = batch_match.group(1)
            if "launching atlas build outdir=" in lowered:
                year_match = re.search(r"year=(\d{4})", parsed_line)
                if year_match:
                    latest_atlas_year = year_match.group(1)
                if stream_label and stream_label not in active_atlas_companies:
                    active_atlas_companies.append(stream_label)
            if "atlas build completed" in lowered:
                year_match = re.search(r"year=(\d{4})", parsed_line)
                if year_match:
                    latest_atlas_year = year_match.group(1)
                    atlas_year_values.add((stream_label or "unknown", latest_atlas_year))
                if stream_label:
                    active_atlas_companies = [item for item in active_atlas_companies if item.lower() != stream_label.lower()]
            if stream_label:
                substage = ""
                if "root_topic_discovery_agent" in parsed_line or "[module 1]" in lowered:
                    substage = "root topics"
                elif "topic_graph_agent" in parsed_line:
                    substage = "topic graph"
                elif "causal_question_agent" in parsed_line or "[module 2]" in lowered:
                    substage = "causal questions"
                elif "causal_statement_agent" in parsed_line or "[module 3]" in lowered:
                    substage = "causal statements"
                elif "triple_extraction_agent" in parsed_line or "[module 4]" in lowered:
                    substage = "relational triples"
                elif "manifold_builder_agent" in parsed_line:
                    substage = "manifold"
                elif "lcm_sweep_agent" in parsed_line or "lcm_scoring_agent" in parsed_line:
                    substage = "lcm scoring"
                elif "launching atlas build outdir=" in lowered or "atlas build completed" in lowered:
                    substage = "yearly atlas"
                elif "brand_democritus_block_denoise.temporal_train" in parsed_line:
                    substage = "temporal denoiser"
                elif "brand_democritus_block_denoise.temporal_infer" in parsed_line:
                    substage = "temporal inference"
                if substage:
                    company_substage[stream_label] = substage
                triple_match = re.search(r"Recovered\s+(\d+)\s+triples", parsed_line, flags=re.IGNORECASE)
                if triple_match:
                    company_triples[stream_label] = int(triple_match.group(1))

        company_a = company_a or "First company"
        company_b = company_b or "Second company"

        def phase_state(*, complete: bool, active: bool) -> str:
            if complete:
                return "complete"
            if active:
                return "active"
            return "pending"

        phases = (
            ("query", "Query resolved", phase_state(complete=query_resolved, active=bool(lines) and not query_resolved)),
            (
                "company_a",
                f"{company_a} build",
                phase_state(complete=first_company_ready, active=first_company_started and not first_company_ready),
            ),
            (
                "company_b",
                f"{company_b} build",
                phase_state(complete=second_company_ready, active=second_company_started and not second_company_ready),
            ),
            (
                "functors",
                "Functor comparison",
                phase_state(complete=comparison_completed, active=comparison_started and not comparison_completed),
            ),
            (
                "viz",
                "Visualization",
                phase_state(complete=visualization_completed, active=visualization_started and not visualization_completed),
            ),
            (
                "done",
                "Dashboard ready",
                phase_state(complete=dashboard_ready, active=visualization_completed and not dashboard_ready),
            ),
        )

        current_phase = "Preparing company inputs"
        if dashboard_ready:
            current_phase = "Dashboard ready"
        elif visualization_started:
            current_phase = "Rendering visualization"
        elif comparison_started and latest_batch:
            current_phase = f"Comparing fiscal year {latest_batch}"
        elif comparison_started:
            current_phase = "Comparing yearly functors"
        elif active_atlas_companies:
            current_phase = f"Building yearly atlas for {active_atlas_companies[0]}"
        elif len(active_companies) >= 2:
            active_substages = [company_substage.get(company, "") for company in active_companies]
            active_substages = [stage for stage in active_substages if stage]
            if active_substages and len(set(active_substages)) == 1:
                current_phase = f"Building {company_a} and {company_b}: {active_substages[0]}"
            else:
                current_phase = f"Building {company_a} and {company_b} in parallel"
        elif active_companies:
            active_company = active_companies[0]
            substage = company_substage.get(active_company, "")
            triples = company_triples.get(active_company)
            if substage == "relational triples" and triples:
                current_phase = f"Building {active_company}: recovering triples ({triples})"
            elif substage:
                current_phase = f"Building {active_company}: {substage}"
            else:
                current_phase = f"Building {active_company}"
        elif query_resolved:
            current_phase = f"Preparing {company_a} and {company_b}"

        return {
            "phases": tuple(phases),
            "current_phase": current_phase,
            "company_a": company_a,
            "company_b": company_b,
            "active_companies": tuple(active_companies),
            "active_company": " and ".join(active_companies),
            "staged_years": len(staged_year_values),
            "atlas_years_ready": len(atlas_year_values),
            "latest_year": latest_year,
            "latest_atlas_year": latest_atlas_year,
            "latest_batch": latest_batch,
            "active_atlas_companies": tuple(active_atlas_companies),
            "company_substage": dict(company_substage),
            "company_triples": dict(company_triples),
            "recent_lines": tuple(recent_lines),
            "log_path": log_path,
            "telemetry": telemetry,
        }

    @staticmethod
    def _company_similarity_democritus_stage_summary(progress: dict[str, object]) -> str:
        active_companies = list(progress.get("active_companies") or ())
        company_substage = dict(progress.get("company_substage") or {})
        company_triples = dict(progress.get("company_triples") or {})
        active_atlas_companies = list(progress.get("active_atlas_companies") or ())

        def render_company_stage(company: str) -> str:
            substage = str(company_substage.get(company) or "").strip()
            triples = company_triples.get(company)
            if substage == "relational triples" and triples:
                return f"{company}: recovering triples ({triples})"
            if substage:
                return f"{company}: {substage}"
            return company

        if active_atlas_companies:
            return " · ".join(f"{company}: yearly atlas" for company in active_atlas_companies[:2])
        if len(active_companies) >= 2:
            substages = [str(company_substage.get(company) or "").strip() for company in active_companies[:2]]
            substages = [substage for substage in substages if substage]
            if len(substages) == 2 and len(set(substages)) == 1:
                return f'{" and ".join(active_companies[:2])}: {substages[0]}'
            return " · ".join(render_company_stage(company) for company in active_companies[:2])
        if active_companies:
            return render_company_stage(active_companies[0])
        return ""

    def _render_company_similarity_live_page(self, run_id: str, *, run_state: dict[str, object]) -> str:
        query = html.escape(str(run_state.get("query") or ""))
        note = html.escape(
            str(run_state.get("note") or "CLIFF is building the cross-company comparison and will surface partial artifacts here as they appear.")
        )
        status = html.escape(str(run_state.get("status") or "running"))
        route_name = html.escape(str(run_state.get("route_name") or "company_similarity"))
        execution_mode = html.escape(str(run_state.get("execution_mode") or self.config.default_execution_mode))
        files = self._company_similarity_live_files(run_state)
        progress = self._company_similarity_progress(run_state)
        eta = self._company_similarity_eta(run_state, progress)
        partial_preview = self._company_similarity_partial_preview(run_state)
        phases = progress["phases"]
        phase_markup = "".join(
            (
                '<div class="phase-card'
                + f' phase-{state}'
                + '"><span class="phase-label">'
                + html.escape(label)
                + "</span><span class=\"phase-state\">"
                + html.escape(state.title())
                + "</span></div>"
            )
            for _key, label, state in phases
        )
        metric_bits = [
            ("Current phase", str(progress["current_phase"])),
            ("Democritus stage", self._company_similarity_democritus_stage_summary(progress) or "Waiting for inner pipeline output"),
            ("Rough ETA", str(eta["eta_human"])),
            ("Companies", f'{progress["company_a"]} vs {progress["company_b"]}'),
            ("Active builds", str(progress["active_company"] or "Waiting")),
            ("Observed parallelism", str(dict(progress.get("telemetry") or {}).get("timing", {}).get("observed_parallelism", 1.0))),
            ("Years staged", str(progress["staged_years"])),
            ("Atlas years ready", str(progress["atlas_years_ready"])),
            ("Latest year", str(progress["latest_atlas_year"] or progress["latest_batch"] or progress["latest_year"] or "n/a")),
        ]
        metric_markup = "".join(
            '<div class="metric"><span class="metric-label">'
            + html.escape(label)
            + "</span><strong>"
            + html.escape(value)
            + "</strong></div>"
            for label, value in metric_bits
        )
        telemetry = dict(progress.get("telemetry") or {})
        slowest_stage_markup = ""
        slowest_stages = list(telemetry.get("slowest_stages") or [])
        timing = dict(telemetry.get("timing") or {})
        performance_summary_markup = (
            "<div class=\"metrics\">"
            + "".join(
                '<div class="metric"><span class="metric-label">'
                + html.escape(label)
                + "</span><strong>"
                + html.escape(value)
                + "</strong></div>"
                for label, value in (
                    ("Elapsed", str(timing.get("elapsed_human") or "n/a")),
                    ("Observed work", str(timing.get("observed_work_human") or "n/a")),
                    ("Observed parallelism", str(timing.get("observed_parallelism") or 1.0)),
                    ("ETA", str(timing.get("eta_human") or "n/a")),
                )
            )
            + "</div>"
        )
        if slowest_stages:
            slowest_stage_markup = (
                "<ul>"
                + "".join(
                    "<li><strong>"
                    + html.escape(str(stage.get("label") or stage.get("stage_key") or "stage"))
                    + "</strong><span class=\"meta\">"
                    + html.escape(str(stage.get("duration_human") or stage.get("duration_seconds") or ""))
                    + "</span></li>"
                    for stage in slowest_stages[:5]
                    if isinstance(stage, dict)
                )
                + "</ul>"
            )
        else:
            slowest_stage_markup = '<p class="muted">No completed timing stages yet.</p>'
        recent_log_markup = (
            "<ul class=\"activity-feed\">"
            + "".join(f"<li>{html.escape(line)}</li>" for line in progress["recent_lines"])
            + "</ul>"
            if progress["recent_lines"]
            else '<p class="muted">No activity lines yet. The worker may still be starting.</p>'
        )
        file_markup = "".join(
            (
                '<li><a href="'
                + html.escape(self._launcher_href_for_run_file(run_id, path))
                + '" target="_blank" rel="noopener noreferrer">'
                + html.escape(path.name)
                + "</a><span class=\"meta\">"
                + html.escape(str(path))
                + "</span></li>"
            )
            for path in files
        )
        empty_markup = (
            "<p class=\"muted\">No partial files are available yet. The worker logs usually appear first while CLIFF is preparing company data.</p>"
            if not files
            else ""
        )
        partial_preview_note_text = str(
            partial_preview.get("note")
            or "CLIFF is waiting for enough overlap to assemble an initial cross-company read."
        )
        if str(partial_preview.get("status") or "") == "warming_up":
            atlas_years_ready = int(progress.get("atlas_years_ready") or 0)
            latest_year_hint = str(progress.get("latest_atlas_year") or progress.get("latest_year") or "").strip()
            active_company_names = list(progress.get("active_companies") or ())
            active_company_name = active_company_names[0] if active_company_names else ""
            active_company_substage = str(dict(progress.get("company_substage") or {}).get(active_company_name, "") or "")
            active_company_triples = dict(progress.get("company_triples") or {}).get(active_company_name)
            if atlas_years_ready <= 0:
                company_hint = str(partial_preview_note_text).replace("Waiting for the first usable yearly atlas slice from ", "").rstrip(".")
                substage_suffix = ""
                if active_company_name and company_hint.lower() == active_company_name.lower() and active_company_substage:
                    if active_company_substage == "relational triples" and active_company_triples:
                        substage_suffix = f" {active_company_name} is still recovering triples ({active_company_triples})."
                    else:
                        substage_suffix = f" {active_company_name} is still in {active_company_substage}."
                suffix = (
                    f" Filings have been staged through {latest_year_hint}, but the first yearly atlas is not complete yet."
                    if latest_year_hint
                    else ""
                )
                partial_preview_note_text = f"Waiting for the first completed yearly atlas from {company_hint}.{suffix}{substage_suffix}".strip()
            else:
                partial_preview_note_text = (
                    f"{atlas_years_ready} yearly atlas slice"
                    f"{'s are' if atlas_years_ready != 1 else ' is'} ready, but CLIFF still needs overlap from both companies "
                    "before it can assemble the initial similarity read."
                )
        partial_preview_note = html.escape(partial_preview_note_text)
        partial_preview_status = html.escape(str(partial_preview.get("status") or "warming_up").replace("_", " "))
        partial_preview_overlap = list(partial_preview.get("overlap_years") or [])
        partial_preview_basis = html.escape(str(partial_preview.get("shared_edge_basis_size") or 0))
        partial_preview_summary = str(partial_preview.get("summary_text") or "").strip()
        partial_preview_links = []
        summary_path_raw = str(partial_preview.get("summary_path") or "").strip()
        if summary_path_raw:
            partial_preview_links.append(
                '<a href="'
                + html.escape(self._launcher_href_for_run_file(run_id, Path(summary_path_raw).resolve()))
                + '" target="_blank" rel="noopener noreferrer">partial summary markdown</a>'
            )
        manifest_path_raw = str(partial_preview.get("manifest_path") or "").strip()
        if manifest_path_raw:
            partial_preview_links.append(
                '<a href="'
                + html.escape(self._launcher_href_for_run_file(run_id, Path(manifest_path_raw).resolve()))
                + '" target="_blank" rel="noopener noreferrer">partial manifest</a>'
            )
        partial_preview_links_markup = (
            "<p class=\"muted\">"
            + ", ".join(partial_preview_links)
            + ".</p>"
            if partial_preview_links
            else ""
        )
        partial_preview_summary_markup = (
            "<pre>"
            + html.escape(partial_preview_summary)
            + "</pre>"
            if partial_preview_summary
            else '<p class="muted">No provisional summary text yet. CLIFF will surface it here as soon as the first overlap is usable.</p>'
        )
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(self.config.title)}</title>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: #f7f1e5;
        color: #1a1f1a;
      }}
      main {{
        max-width: 1120px;
        margin: 28px auto;
        padding: 0 18px 24px;
        display: grid;
        gap: 16px;
      }}
      .panel {{
        background: #fffdf6;
        border: 1px solid #d5c8a4;
        border-radius: 24px;
        padding: 22px;
      }}
      .eyebrow {{
        margin: 0 0 10px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: #0f6d63;
      }}
      h1 {{
        margin: 0;
        font-size: 1.8rem;
      }}
      p {{
        margin: 12px 0 0;
        color: #5e675d;
        line-height: 1.6;
      }}
      .chips {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
      }}
      .chip {{
        border: 1px solid #c8b88e;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6f5b2a;
        background: #faf4e3;
      }}
      ul {{
        margin: 0;
        padding-left: 20px;
        display: grid;
        gap: 10px;
      }}
      li {{
        line-height: 1.5;
      }}
      a {{
        color: #7d4306;
      }}
      .meta {{
        display: block;
        color: #6e756c;
        font-size: 12px;
        word-break: break-word;
      }}
      .muted {{
        color: #6e756c;
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 16px;
      }}
      .metric {{
        border: 1px solid #d5c8a4;
        border-radius: 18px;
        padding: 14px;
        background: #fffaf0;
      }}
      .metric-label {{
        display: block;
        color: #6e756c;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .metric strong {{
        display: block;
        margin-top: 6px;
        font-size: 20px;
      }}
      .phases {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
      }}
      .phase-card {{
        border: 1px solid #d5c8a4;
        border-radius: 18px;
        padding: 14px;
        background: #fffaf2;
      }}
      .phase-complete {{
        border-color: #8fb7a9;
        background: #eef8f3;
      }}
      .phase-active {{
        border-color: #d1b86e;
        background: #fff7dc;
      }}
      .phase-label {{
        display: block;
        font-size: 14px;
      }}
      .phase-state {{
        display: block;
        margin-top: 6px;
        color: #6e756c;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .activity-feed {{
        margin: 0;
        padding-left: 20px;
        display: grid;
        gap: 10px;
      }}
      .activity-feed li {{
        color: #344039;
        line-height: 1.6;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">Inspecting Run</p>
        <h1>{query}</h1>
        <p>{note}</p>
        <div class="chips">
          <span class="chip">{route_name}</span>
          <span class="chip">{execution_mode}</span>
          <span class="chip">{status}</span>
          <span class="chip">{html.escape(str(eta["eta_label"]))}</span>
          <span class="chip">{len(files)} partial file{'s' if len(files) != 1 else ''}</span>
        </div>
        <div class="metrics">{metric_markup}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Progress</p>
        <div class="phases">{phase_markup}</div>
      </section>
      <section class="panel">
        <p class="eyebrow">Initial Similarity Read</p>
        <div class="chips">
          <span class="chip">{partial_preview_status}</span>
          <span class="chip">{len(partial_preview_overlap)} overlap year{'s' if len(partial_preview_overlap) != 1 else ''}</span>
          <span class="chip">{partial_preview_basis} shared basis edge{'s' if str(partial_preview.get("shared_edge_basis_size") or 0) != "1" else ''}</span>
        </div>
        <p>{partial_preview_note}</p>
        {partial_preview_summary_markup}
        {partial_preview_links_markup}
      </section>
      <section class="panel">
        <p class="eyebrow">Performance</p>
        {performance_summary_markup}
        {slowest_stage_markup}
      </section>
      <section class="panel">
        <p class="eyebrow">Recent Activity</p>
        {recent_log_markup}
      </section>
      <section class="panel">
        <p class="eyebrow">Live Files</p>
        {empty_markup}
        {"<ul>" + file_markup + "</ul>" if file_markup else ""}
      </section>
    </main>
    <script>
      var terminalStatuses = new Set(["complete", "failed", "stopped"]);
      var refreshTimer = window.setInterval(function () {{
        fetch("/state?ts=" + Date.now(), {{ cache: "no-store" }})
          .then(function (response) {{ return response.json(); }})
          .then(function (payload) {{
            var runs = Array.isArray(payload.runs) ? payload.runs : [];
            var runState = runs.find(function (item) {{ return item.run_id === {json.dumps(run_id)}; }});
            if (runState && terminalStatuses.has(String(runState.status || "").toLowerCase())) {{
              window.clearInterval(refreshTimer);
              return;
            }}
            window.location.replace("/run-artifact?run_id=" + encodeURIComponent({json.dumps(run_id)}) + "&ts=" + Date.now());
          }})
          .catch(function () {{}});
      }}, {_ARTIFACT_REFRESH_MS});
    </script>
  </body>
</html>
"""

    def _render_live_run_artifact_shell(
        self,
        run_id: str,
        *,
        artifact_path: Path,
        run_state: dict[str, object],
    ) -> str:
        title = html.escape(self.config.title)
        query = html.escape(str(run_state.get("query") or ""))
        note = html.escape(str(run_state.get("note") or "CLIFF is gathering live partial outputs from this run."))
        llm_budget_label = html.escape(str(run_state.get("llm_budget_label") or ""))
        llm_usage_label = html.escape(str(run_state.get("llm_usage_label") or ""))
        iframe_src = self._launcher_href_for_run_file(run_id, artifact_path)
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: #f7f1e5;
        color: #1a1f1a;
      }}
      main {{
        max-width: 1180px;
        margin: 28px auto;
        padding: 0 18px 24px;
        display: grid;
        gap: 16px;
      }}
      .panel {{
        background: #fffdf6;
        border: 1px solid #d5c8a4;
        border-radius: 24px;
        padding: 22px;
      }}
      .eyebrow {{
        margin: 0 0 10px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: #0f6d63;
      }}
      h1 {{
        margin: 0;
        font-size: 1.8rem;
      }}
      p {{
        margin: 12px 0 0;
        color: #5e675d;
        line-height: 1.6;
      }}
      .artifact-shell {{
        border: 1px solid #d5c8a4;
        border-radius: 22px;
        overflow: hidden;
        min-height: 720px;
        background: #fffdf8;
      }}
      .artifact-shell iframe {{
        width: 100%;
        min-height: 720px;
        border: 0;
        display: block;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <p class="eyebrow">Inspecting Run</p>
        <h1>{query}</h1>
        <p>{note}</p>
        {'<p><strong>LLM budget:</strong> ' + llm_budget_label + '</p>' if llm_budget_label else ''}
        {'<p><strong>LLM usage:</strong> ' + llm_usage_label + '</p>' if llm_usage_label else ''}
      </section>
      <section class="artifact-shell">
        <iframe id="artifact-frame" src="{html.escape(iframe_src)}" title="Live CLIFF artifact"></iframe>
      </section>
    </main>
    <script>
      var terminalStatuses = new Set(["complete", "failed", "stopped"]);
      var refreshTimer = window.setInterval(function () {{
        fetch("/state?ts=" + Date.now(), {{ cache: "no-store" }})
          .then(function (response) {{ return response.json(); }})
          .then(function (payload) {{
            var runs = Array.isArray(payload.runs) ? payload.runs : [];
            var runState = runs.find(function (item) {{ return item.run_id === {json.dumps(run_id)}; }});
            if (runState && terminalStatuses.has(String(runState.status || "").toLowerCase())) {{
              window.clearInterval(refreshTimer);
              return;
            }}
            var frame = document.getElementById("artifact-frame");
            if (!frame) {{
              return;
            }}
            frame.src = "{html.escape(iframe_src)}" + "&ts=" + Date.now();
          }})
          .catch(function () {{}});
      }}, {_ARTIFACT_REFRESH_MS});
    </script>
  </body>
</html>
"""

    def _resolve_run_file_path(self, run_id: str, requested_path: str) -> Path | None:
        run_state = self._lookup_run_state(run_id)
        if not run_state:
            return None
        normalized_request = str(requested_path).strip()
        if not normalized_request:
            artifact_path_value = str(run_state.get("artifact_path") or "").strip()
            return Path(artifact_path_value).resolve() if artifact_path_value else None
        candidate = Path(unquote(normalized_request)).expanduser()
        if not candidate.is_absolute():
            artifact_path_value = str(run_state.get("artifact_path") or "").strip()
            artifact_path = Path(artifact_path_value).resolve() if artifact_path_value else None
            if artifact_path is None:
                return None
            candidate = (artifact_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots: list[Path] = []
        outdir_value = str(run_state.get("outdir") or "").strip()
        if outdir_value:
            allowed_roots.append(Path(outdir_value).resolve())
        artifact_path_value = str(run_state.get("artifact_path") or "").strip()
        if artifact_path_value:
            artifact_path = Path(artifact_path_value).resolve()
            allowed_roots.append(artifact_path.parent)
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return candidate
            except ValueError:
                continue
        return None

    def _launcher_href_for_run_file(self, run_id: str, file_path: Path) -> str:
        return "/run-file?run_id=" + quote(run_id, safe="") + "&path=" + quote(str(file_path.resolve()), safe="")

    def _rewrite_artifact_links(self, html_body: str, *, run_id: str, source_path: Path) -> str:
        pattern = re.compile(r'(?P<attr>href|src)=(?P<quote>["\'])(?P<target>.+?)(?P=quote)', flags=re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            attr = match.group("attr")
            quote_char = match.group("quote")
            target = match.group("target").strip()
            lowered = target.lower()
            if (
                not target
                or target.startswith("#")
                or lowered.startswith("http://")
                or lowered.startswith("https://")
                or lowered.startswith("mailto:")
                or lowered.startswith("javascript:")
                or lowered.startswith("data:")
                or target.startswith("/run-file?")
            ):
                return match.group(0)
            resolved: Path | None = None
            if lowered.startswith("file://"):
                parsed = urlparse(target)
                local_path = unquote(parsed.path or "")
                if local_path:
                    resolved = Path(local_path).resolve()
            elif not target.startswith("/"):
                resolved = (source_path.parent / target).resolve()
            if resolved is None:
                return match.group(0)
            safe_target = self._resolve_run_file_path(run_id, str(resolved))
            if safe_target is None or not safe_target.exists():
                return match.group(0)
            proxied = self._launcher_href_for_run_file(run_id, safe_target)
            return f"{attr}={quote_char}{proxied}{quote_char}"

        return pattern.sub(replace, html_body)

    def _render_text_file_as_html(self, title: str, body: str) -> str:
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: #f7f1e5;
        color: #1a1f1a;
      }}
      main {{
        max-width: 980px;
        margin: 32px auto;
        padding: 0 18px;
      }}
      .card {{
        background: #fffdf6;
        border: 1px solid #d5c8a4;
        border-radius: 24px;
        padding: 24px;
      }}
      h1 {{
        margin: 0 0 14px;
        font-size: 1.7rem;
      }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.6;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>{html.escape(title)}</h1>
        <pre>{html.escape(body)}</pre>
      </section>
    </main>
  </body>
</html>
"""

    def _render_html_file_for_run(self, run_id: str, file_path: Path) -> str:
        checkpoint_manifest = self._checkpoint_manifest_for_html(file_path)
        if checkpoint_manifest is not None:
            return self._render_democritus_checkpoint_page(run_id, artifact_path=file_path)
        company_checkpoint_manifest = self._company_similarity_checkpoint_manifest_for_html(file_path)
        if company_checkpoint_manifest is not None:
            return self._render_company_similarity_checkpoint_page(run_id, artifact_path=file_path)
        return self._rewrite_artifact_links(
            file_path.read_text(encoding="utf-8", errors="replace"),
            run_id=run_id,
            source_path=file_path,
        )

    def _render_run_file_response(self, run_id: str, requested_path: str) -> tuple[bytes, str, HTTPStatus]:
        file_path = self._resolve_run_file_path(run_id, requested_path)
        if file_path is None or not file_path.exists():
            body = self._render_text_file_as_html(
                self.config.title,
                "This linked artifact is not available from the current CLIFF session.",
            )
            return body.encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.NOT_FOUND
        suffix = file_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            body = self._render_html_file_for_run(run_id, file_path)
            return body.encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.OK
        if suffix in {".md", ".txt", ".json", ".jsonl", ".csv", ".log"}:
            body = self._render_text_file_as_html(file_path.name, file_path.read_text(encoding="utf-8", errors="replace"))
            return body.encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.OK
        content_type = mimetypes.guess_type(str(file_path.name))[0] or "application/octet-stream"
        return file_path.read_bytes(), content_type, HTTPStatus.OK

    def _render_launcher_page(self, *, error_message: str = "") -> str:
        with self._lock:
            query_received = self._query_received
            self._refresh_archived_runs()
            session_runs = [self._enriched_run_state(dict(item)) for item in self._session_runs]
            archived_runs = [self._enriched_run_state(dict(item)) for item in self._archived_runs]
        title = html.escape(self.config.title)
        subtitle = html.escape(self.config.subtitle)
        eyebrow = html.escape(self.config.eyebrow)
        title_raw = str(self.config.title or "").strip()
        cliff_expansion_markup = ""
        if title_raw.upper() == "CLIFF":
            cliff_expansion_markup = (
                '<p class="hero-expansion">Conscious Layer Interface to Functor Flow</p>'
                '<p class="hero-architecture">Architecture: the conscious interface sits on top of the Functor Flow causal engine.</p>'
            )
        query_label = html.escape(self.config.query_label)
        placeholder = html.escape(self.config.query_placeholder)
        submit_label = html.escape(self.config.submit_label)
        waiting_message = html.escape(self.config.waiting_message)
        demo_queries = tuple(query for query in self.config.demo_queries if str(query).strip())
        session_mode = self.config.session_mode
        execution_mode_enabled = self.config.enable_execution_mode
        error_html = (
            f'<p class="error" role="alert">{html.escape(error_message)}</p>'
            if error_message
            else ""
        )
        demo_tour_markup = self._render_demo_tour_markup(demo_queries)
        execution_mode_markup = ""
        if execution_mode_enabled:
            execution_mode_markup = """
            <fieldset class="execution-mode-fieldset">
              <legend>Execution depth</legend>
              <label class="execution-mode-option">
                <input type="radio" name="execution_mode" value="quick" checked />
                <span><strong>Quick</strong> Start with a smaller sample and lighter pass so the first answer arrives sooner.</span>
              </label>
              <label class="execution-mode-option">
                <input type="radio" name="execution_mode" value="interactive" />
                <span><strong>Interactive</strong> Stop at the first meaningful checkpoint so you can inspect topics and decide whether to continue deeper.</span>
              </label>
              <label class="execution-mode-option">
                <input type="radio" name="execution_mode" value="deep" />
                <span><strong>Deep</strong> Run the fuller causal build from the start for maximum coverage.</span>
              </label>
              <p class="execution-mode-hint">
                Latency guide: textbook and filing lookups are usually quickest, product evaluation can take longer,
                interactive mode pauses earlier for inspection, and deep research routes like Democritus or company similarity may still take several minutes even in quick mode.
              </p>
            </fieldset>
            """
        llm_budget_markup = """
            <fieldset class="execution-mode-fieldset">
              <legend>LLM token budget</legend>
              <label class="query-form-inline-label" for="llm_token_budget">Maximum total tokens for this run</label>
              <input
                id="llm_token_budget"
                name="llm_token_budget"
                type="number"
                min="1"
                step="1"
                placeholder="Optional, e.g. 20000"
              />
              <p class="execution-mode-hint">If set, CLIFF will stop once this run exhausts the shared OpenAI token budget.</p>
            </fieldset>
        """
        form_or_status = f"""
          <form method="post" action="/submit" class="query-form">
            <label for="query">{query_label}</label>
            <textarea
              id="query"
              name="query"
              rows="5"
              placeholder="{placeholder}"
              autofocus
              required
            ></textarea>
            {execution_mode_markup}
            {llm_budget_markup}
            {error_html}
            <button type="submit">{submit_label}</button>
          </form>
          {demo_tour_markup}
        """
        if session_mode:
            form_or_status = f"""
          <section class="session-shell">
            <div class="session-intro">
              <p class="status-kicker">Persistent Session</p>
              <h2>The prompt window stays open for your next question.</h2>
              <p>{waiting_message} Longer runs can keep working in the background, and you can open completed results from the session run list whenever they are ready.</p>
            </div>
            <form method="post" action="/submit" class="query-form">
              <label for="query">{query_label}</label>
              <textarea
                id="query"
                name="query"
                rows="5"
                placeholder="{placeholder}"
                autofocus
                required
              ></textarea>
              {execution_mode_markup}
              {llm_budget_markup}
              {error_html}
              <button type="submit">{submit_label}</button>
            </form>
            {demo_tour_markup}
            <section class="session-runs">
              <div class="session-header">
                <h2>Session Runs</h2>
                <p>Queued, running, completed, and failed requests appear here.</p>
              </div>
              <div id="session-runs">{self._render_session_runs_markup(session_runs)}</div>
            </section>
            <section class="session-runs">
              <div class="session-header">
                <h2>Archived Runs</h2>
                <p>Saved CLIFF runs discovered from archive roots can be reopened or rerun here.</p>
              </div>
              <div id="archived-runs">{self._render_session_runs_markup(archived_runs, empty_message="No archived CLIFF runs were discovered under the configured archive roots yet.")}</div>
            </section>
          </section>
          <script>
            function escapeHtml(value) {{
              return String(value || "")
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
            }}

            function renderRuns(runs, emptyMessage) {{
              if (!Array.isArray(runs) || runs.length === 0) {{
                return '<div class="empty-state">' + escapeHtml(emptyMessage || 'No queries have been submitted in this session yet.') + '</div>';
              }}
              return runs.map(function (run) {{
                var archived = !!run.archived;
                var cardClass = 'run-card';
                if ((run.status || '') === 'complete') {{
                  cardClass += ' run-card-complete';
                }} else if ((run.status || '') === 'failed') {{
                  cardClass += ' run-card-failed';
                }}
                var executionMode = String(run.execution_mode || 'quick');
                var route = run.route_name ? '<span class="run-chip">' + escapeHtml(run.route_name) + '</span>' : '';
                var archiveChip = archived ? '<span class="run-chip archived-chip">archived</span>' : '';
                var researchProfile = run.research_profile_label
                  ? '<span class="run-chip route-profile-' + escapeHtml(run.research_profile_class || 'route-warming-up') + '">' + escapeHtml(run.research_profile_label) + '</span>'
                  : '';
                var mode = executionMode ? '<span class="run-chip mode-' + escapeHtml(executionMode) + '">' + escapeHtml(executionMode) + '</span>' : '';
                var eta = run.eta_label ? '<span class="run-chip eta-chip">' + escapeHtml(run.eta_label) + '</span>' : '';
                var stopAction = (!archived && ((run.status || '') === 'queued' || (run.status || '') === 'routing' || (run.status || '') === 'running'))
                  ? '<div class="run-actions"><button type="button" class="run-stop-button" onclick="requestStopRun(\\'' + escapeHtml(run.run_id || '') + '\\')">Stop query</button></div>'
                  : '';
                var deepenAction = (!archived && (run.status || '') === 'complete' && (executionMode === 'quick' || executionMode === 'interactive') && ((run.route_name || '') === 'democritus' || (run.route_name || '') === 'company_similarity'))
                  ? '<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestDeepRun(\\'' + escapeHtml(run.run_id || '') + '\\')">Go deeper</button></div>'
                  : '';
                var rerunAction = archived
                  ? '<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestArchivedRerun(\\'' + escapeHtml(run.run_id || '') + '\\')">Re-run query</button></div>'
                  : '';
                var wrongRouteAction = ((run.status || '') === 'complete' && (run.route_name || ''))
                  ? '<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestWrongRoute(\\'' + escapeHtml(run.run_id || '') + '\\')">Wrong route</button></div>'
                  : '';
                var inspectLabel = ((run.status || '') === 'queued' || (run.status || '') === 'routing' || (run.status || '') === 'running' || (run.status || '') === 'stopping')
                  ? 'Inspect run'
                  : 'Open result';
                var openAction = run.artifact_path
                  ? '<div class="run-actions"><a class="run-link" href="/run-artifact?run_id=' + encodeURIComponent(run.run_id || '') + '" target="_blank" rel="noopener noreferrer">' + inspectLabel + '</a></div>'
                  : '';
                var outdir = run.outdir ? '<div class="run-meta"><strong>Output:</strong> <code>' + escapeHtml(run.outdir) + '</code></div>' : '';
                var artifact = run.artifact_path ? '<div class="run-meta"><strong>Artifact:</strong> <code>' + escapeHtml(run.artifact_path) + '</code></div>' : '';
                var researchNote = run.research_profile_note
                  ? '<div class="run-meta"><strong>Latency class:</strong> ' + escapeHtml(run.research_profile_note) + '</div>'
                  : '';
                var llmBudget = run.llm_budget_label
                  ? '<div class="run-meta"><strong>LLM budget:</strong> ' + escapeHtml(run.llm_budget_label) + '</div>'
                  : '';
                var llmUsage = run.llm_usage_label
                  ? '<div class="run-meta"><strong>LLM usage:</strong> ' + escapeHtml(run.llm_usage_label) + '</div>'
                  : '';
                var unconscious = (run.eta_label || run.parallelism_label || run.current_stage_label)
                  ? '<div class="run-meta"><strong>Unconscious report:</strong> '
                    + (run.eta_label ? escapeHtml(run.eta_label) : 'ETA warming up')
                    + (run.parallelism_label ? ' · ' + escapeHtml(run.parallelism_label) : '')
                    + (run.current_stage_label ? ' · ' + escapeHtml(run.current_stage_label) : '')
                    + '</div>'
                  : '';
                return ''
                  + '<article class="' + cardClass + '">'
                  + '<div class="run-topline">'
                  + '<div class="run-id">' + escapeHtml(run.run_id) + '</div>'
                  + '<div class="run-badges"><span class="run-chip status-' + escapeHtml(run.status || 'queued') + '">' + escapeHtml(run.status || 'queued') + '</span>' + archiveChip + route + researchProfile + mode + eta + '</div>'
                  + '</div>'
                  + '<div class="run-query">' + escapeHtml(run.query || '') + '</div>'
                  + '<div class="run-note">' + escapeHtml(run.note || '') + '</div>'
                  + stopAction
                  + deepenAction
                  + rerunAction
                  + wrongRouteAction
                  + openAction
                  + researchNote
                  + llmBudget
                  + llmUsage
                  + unconscious
                  + outdir
                  + artifact
                  + '</article>';
              }}).join('');
            }}

            function requestStopRun(runId) {{
              fetch('/stop-run', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }},
                body: 'run_id=' + encodeURIComponent(runId || '')
              }})
                .then(function () {{
                  return fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }});
                }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (!container) {{
                    return;
                  }}
                  container.innerHTML = renderRuns(payload.runs || [], 'No queries have been submitted in this session yet.');
                }})
                .catch(function () {{}});
            }}

            function requestDeepRun(runId) {{
              fetch('/deepen-run', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }},
                body: 'run_id=' + encodeURIComponent(runId || '')
              }})
                .then(function () {{
                  return fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }});
                }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (!container) {{
                    return;
                  }}
                  container.innerHTML = renderRuns(payload.runs || [], 'No queries have been submitted in this session yet.');
                }})
                .catch(function () {{}});
            }}

            function requestArchivedRerun(runId) {{
              fetch('/rerun-archived', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }},
                body: 'run_id=' + encodeURIComponent(runId || '')
              }})
                .then(function () {{
                  return fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }});
                }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (container) {{
                    container.innerHTML = renderRuns(payload.runs || [], 'No queries have been submitted in this session yet.');
                  }}
                  var archived = document.getElementById('archived-runs');
                  if (archived) {{
                    archived.innerHTML = renderRuns(payload.archived_runs || [], 'No archived CLIFF runs were discovered under the configured archive roots yet.');
                  }}
                }})
                .catch(function () {{}});
            }}

            function requestWrongRoute(runId) {{
              fetch('/wrong-route', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8' }},
                body: 'run_id=' + encodeURIComponent(runId || '')
              }})
                .then(function () {{
                  return fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }});
                }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (container) {{
                    container.innerHTML = renderRuns(payload.runs || [], 'No queries have been submitted in this session yet.');
                  }}
                  var archived = document.getElementById('archived-runs');
                  if (archived) {{
                    archived.innerHTML = renderRuns(payload.archived_runs || [], 'No archived CLIFF runs were discovered under the configured archive roots yet.');
                  }}
                }})
                .catch(function () {{}});
            }}

            window.setInterval(function () {{
              fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (container) {{
                    container.innerHTML = renderRuns(payload.runs || [], 'No queries have been submitted in this session yet.');
                  }}
                  var archived = document.getElementById('archived-runs');
                  if (archived) {{
                    archived.innerHTML = renderRuns(payload.archived_runs || [], 'No archived CLIFF runs were discovered under the configured archive roots yet.');
                  }}
                }})
                .catch(function () {{}});
            }}, {_SESSION_REFRESH_MS});
          </script>
        """
        elif query_received:
            form_or_status = f"""
          <section class="status-card">
            <p class="status-kicker">Run accepted</p>
            <h2>The analysis is starting.</h2>
            <p>{waiting_message}</p>
          </section>
          <section class="artifact-shell">
            <iframe
              id="artifact-frame"
              src="/artifact"
              title="BAFFLE dashboard"
              loading="eager"
            ></iframe>
          </section>
          <script>
            var refreshTimer = window.setInterval(function () {{
              fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  if (payload.artifact_ready) {{
                    window.clearInterval(refreshTimer);
                  }}
                  var frame = document.getElementById("artifact-frame");
                  if (!frame) {{
                    return;
                  }}
                  frame.src = "/artifact?ts=" + Date.now();
                }})
                .catch(function () {{}});
            }}, {_ARTIFACT_REFRESH_MS});
          </script>
        """
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #16211f;
        --muted: #5a6661;
        --paper: #f3ead7;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d0c0a0;
        --accent: #0f6d63;
        --accent-strong: #0a4f48;
        --error: #a22222;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at 15% 20%, rgba(15, 109, 99, 0.12), transparent 28%),
          radial-gradient(circle at 85% 12%, rgba(153, 96, 38, 0.14), transparent 24%),
          linear-gradient(180deg, #f9f3e6 0%, var(--paper) 62%, #eadcc0 100%);
      }}
      main {{
        width: min(1100px, calc(100vw - 32px));
        margin: 40px auto;
      }}
      .hero {{
        display: grid;
        gap: 22px;
      }}
      .panel {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 24px 60px rgba(33, 26, 18, 0.12);
      }}
      .eyebrow {{
        margin: 0 0 10px 0;
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
      }}
      h1 {{
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.5rem);
        line-height: 1.02;
      }}
      .subtitle {{
        margin: 16px 0 0 0;
        max-width: 60rem;
        font-size: 1.08rem;
        line-height: 1.7;
        color: var(--muted);
      }}
      .hero-expansion {{
        margin: 14px 0 0 0;
        font-size: 0.95rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-strong);
      }}
      .hero-architecture {{
        margin: 10px 0 0 0;
        max-width: 56rem;
        line-height: 1.6;
        color: var(--muted);
      }}
      .query-form {{
        display: grid;
        gap: 14px;
      }}
      .query-form label {{
        font-size: 0.95rem;
        font-weight: 600;
      }}
      .query-form textarea {{
        width: 100%;
        padding: 16px 18px;
        border-radius: 18px;
        border: 1px solid #baa77e;
        background: #fffdf9;
        color: var(--ink);
        font: inherit;
        resize: vertical;
        min-height: 150px;
        box-shadow: inset 0 1px 2px rgba(40, 28, 11, 0.06);
      }}
      .query-form input[type="number"] {{
        width: min(280px, 100%);
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid #baa77e;
        background: #fffdf9;
        color: var(--ink);
        font: inherit;
        box-shadow: inset 0 1px 2px rgba(40, 28, 11, 0.06);
      }}
      .query-form button {{
        width: fit-content;
        padding: 12px 20px;
        border: 0;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        color: white;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
      }}
      .execution-mode-fieldset {{
        margin: 0;
        padding: 14px 16px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255, 253, 248, 0.82);
      }}
      .execution-mode-fieldset legend {{
        padding: 0 8px;
        font-size: 0.88rem;
        color: var(--muted);
      }}
      .execution-mode-option {{
        display: flex;
        gap: 10px;
        align-items: flex-start;
        line-height: 1.6;
        color: var(--muted);
      }}
      .execution-mode-option + .execution-mode-option {{
        margin-top: 10px;
      }}
      .execution-mode-option input {{
        margin-top: 0.28rem;
      }}
      .execution-mode-hint {{
        margin: 12px 0 0 0;
        line-height: 1.6;
        color: var(--muted);
        font-size: 0.95rem;
      }}
      .query-form-inline-label {{
        display: block;
        margin-bottom: 8px;
      }}
      .demo-tour {{
        margin-top: 18px;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 20px;
        background: rgba(255, 250, 240, 0.88);
      }}
      .demo-tour h3 {{
        margin: 0 0 8px 0;
        font-size: 1.12rem;
      }}
      .demo-tour p {{
        margin: 0 0 14px 0;
        line-height: 1.6;
        color: var(--muted);
      }}
      .demo-tour-list {{
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 10px;
      }}
      .demo-tour-step {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 12px;
        align-items: start;
      }}
      .demo-tour-index {{
        width: 28px;
        height: 28px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #e2efe9;
        color: var(--accent-strong);
        font-size: 0.88rem;
        font-weight: 700;
      }}
      .demo-tour-button {{
        width: 100%;
        text-align: left;
        border: 1px solid #ccb892;
        border-radius: 16px;
        background: #fffdf8;
        padding: 12px 14px;
        color: var(--ink);
        font: inherit;
        line-height: 1.5;
        cursor: pointer;
      }}
      .demo-tour-button:hover {{
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(15, 109, 99, 0.08);
      }}
      .error {{
        margin: 0;
        color: var(--error);
      }}
      .status-card {{
        margin-bottom: 18px;
      }}
      .status-kicker {{
        margin: 0 0 10px 0;
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
      }}
      .status-card h2 {{
        margin: 0 0 8px 0;
        font-size: 1.5rem;
      }}
      .status-card p {{
        margin: 0;
        line-height: 1.6;
        color: var(--muted);
      }}
      .session-shell {{
        display: grid;
        gap: 22px;
      }}
      .session-intro h2,
      .session-header h2 {{
        margin: 0 0 8px 0;
        font-size: 1.5rem;
      }}
      .session-intro p,
      .session-header p {{
        margin: 0;
        line-height: 1.6;
        color: var(--muted);
      }}
      .session-runs {{
        display: grid;
        gap: 14px;
      }}
      #session-runs {{
        display: grid;
        gap: 12px;
      }}
      .run-card {{
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 16px 18px;
        background: #fffdf8;
      }}
      .run-card-complete {{
        border-color: #62a578;
        box-shadow: 0 0 0 3px rgba(98, 165, 120, 0.10);
        background: linear-gradient(180deg, #fcfffc 0%, #f4fbf5 100%);
      }}
      .run-card-failed {{
        border-color: #d38c8c;
        box-shadow: 0 0 0 3px rgba(162, 34, 34, 0.08);
      }}
      .run-topline {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
      }}
      .run-id {{
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
      }}
      .run-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .run-chip {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1.1;
        background: #edf3f1;
        color: var(--accent-strong);
      }}
      .archived-chip {{
        background: #eee5f5;
        color: #62427c;
      }}
      .status-queued,
      .status-routing {{
        background: #f5ecd6;
        color: #8a5c0a;
      }}
      .status-running {{
        background: #dcefe9;
        color: var(--accent-strong);
      }}
      .status-stopping {{
        background: #f5e7c9;
        color: #9a6700;
      }}
      .status-complete {{
        background: #dff3e5;
        color: #166534;
      }}
      .status-stopped {{
        background: #e7edf0;
        color: #44525c;
      }}
      .mode-quick {{
        background: #e6f1ee;
        color: var(--accent-strong);
      }}
      .mode-deep {{
        background: #efe3d4;
        color: #8a4b10;
      }}
      .eta-chip {{
        background: #eef4db;
        color: #556b17;
      }}
      .route-profile-quick-answer {{
        background: #e7f3eb;
        color: #2e6a47;
      }}
      .route-profile-longer-analysis {{
        background: #f4ead8;
        color: #8a5a18;
      }}
      .route-profile-deep-research {{
        background: #f6e0dc;
        color: #8a3325;
      }}
      .route-profile-route-warming-up {{
        background: #ece8de;
        color: #6b6251;
      }}
      .status-failed {{
        background: #f8dddd;
        color: var(--error);
      }}
      .run-query {{
        margin-top: 12px;
        font-size: 1.05rem;
        line-height: 1.5;
      }}
      .run-actions {{
        margin-top: 10px;
      }}
      .run-link {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 9px 14px;
        background: #0f6d63;
        color: #fff;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 700;
      }}
      .run-link:hover {{
        background: #0a4f48;
      }}
      .run-stop-button {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 9px 14px;
        background: #fff4d6;
        color: #8a5c0a;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid #e4c98a;
        cursor: pointer;
      }}
      .run-stop-button:hover {{
        background: #f7e7b8;
      }}
      .run-deepen-button {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 9px 14px;
        background: #e7f1ef;
        color: var(--accent-strong);
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid #b8d3cb;
        cursor: pointer;
      }}
      .run-deepen-button:hover {{
        background: #d8ebe7;
      }}
      .run-note,
      .run-meta {{
        margin-top: 8px;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.95rem;
      }}
      .run-meta code {{
        background: rgba(22, 33, 31, 0.06);
        padding: 2px 6px;
        border-radius: 6px;
      }}
      .empty-state {{
        border: 1px dashed var(--line);
        border-radius: 18px;
        padding: 16px;
        color: var(--muted);
        background: rgba(255, 253, 248, 0.72);
      }}
      .artifact-shell {{
        border: 1px solid var(--line);
        border-radius: 22px;
        overflow: hidden;
        min-height: 680px;
        background: #fffdf8;
      }}
      .artifact-shell iframe {{
        width: 100%;
        min-height: 680px;
        border: 0;
        display: block;
      }}
      @media (max-width: 720px) {{
        main {{
          width: min(100vw - 20px, 1100px);
          margin: 20px auto;
        }}
        .panel {{
          padding: 20px;
          border-radius: 22px;
        }}
        .artifact-shell,
        .artifact-shell iframe {{
          min-height: 560px;
        }}
        .demo-tour-step {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <article class="panel">
          <p class="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          {cliff_expansion_markup}
          <p class="subtitle">{subtitle}</p>
        </article>
        <article class="panel">
          {form_or_status}
        </article>
      </section>
    </main>
    <script>
      document.querySelectorAll('.demo-tour-button').forEach(function (button) {{
        button.addEventListener('click', function () {{
          var query = button.getAttribute('data-query') || '';
          var textarea = document.getElementById('query');
          if (!textarea) {{
            return;
          }}
          textarea.value = query;
          textarea.focus();
          if (typeof textarea.setSelectionRange === 'function') {{
            textarea.setSelectionRange(textarea.value.length, textarea.value.length);
          }}
        }});
      }});
    </script>
  </body>
</html>
"""

    def _render_demo_tour_markup(self, demo_queries: tuple[str, ...]) -> str:
        if not demo_queries:
            return ""
        steps: list[str] = []
        for index, query in enumerate(demo_queries, start=1):
            escaped_query = html.escape(query)
            steps.append(
                "<li class=\"demo-tour-step\">"
                f"<span class=\"demo-tour-index\">{index}</span>"
                "<div>"
                f"<button type=\"button\" class=\"demo-tour-button\" data-query=\"{escaped_query}\">{escaped_query}</button>"
                "</div>"
                "</li>"
            )
        return (
            "<section class=\"demo-tour\">"
            "<h3>Take the 2-minute tour</h3>"
            "<p>Try these in order to see how CLIFF moves from book guidance to demos, code, and applied examples.</p>"
            f"<ol class=\"demo-tour-list\">{''.join(steps)}</ol>"
            "</section>"
        )

    def _render_session_runs_markup(self, runs: list[dict[str, object]], *, empty_message: str = "No queries have been submitted in this session yet.") -> str:
        if not runs:
            return f'<div class="empty-state">{html.escape(empty_message)}</div>'

        def esc(value: object) -> str:
            return html.escape(str(value))

        cards: list[str] = []
        for run in (self._enriched_run_state(dict(item)) for item in runs):
            archived = bool(run.get("archived"))
            card_class = "run-card"
            if run.get("status") == "complete":
                card_class += " run-card-complete"
            elif run.get("status") == "failed":
                card_class += " run-card-failed"
            route_markup = (
                f'<span class="run-chip">{esc(run.get("route_name"))}</span>'
                if run.get("route_name")
                else ""
            )
            archived_markup = (
                '<span class="run-chip archived-chip">archived</span>'
                if archived
                else ""
            )
            research_profile_markup = (
                f'<span class="run-chip route-profile-{esc(run.get("research_profile_class") or "route-warming-up")}">{esc(run.get("research_profile_label"))}</span>'
                if run.get("research_profile_label")
                else ""
            )
            mode_markup = f'<span class="run-chip mode-{esc(run.get("execution_mode") or "quick")}">{esc(run.get("execution_mode") or "quick")}</span>'
            eta_markup = (
                f'<span class="run-chip eta-chip">{esc(run.get("eta_label") or "")}</span>'
                if run.get("eta_label")
                else ""
            )
            stop_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-stop-button" onclick="requestStopRun(\'{esc(run.get("run_id") or "")}\')">Stop query</button></div>'
                if (not archived and run.get("status") in {"queued", "routing", "running"})
                else ""
            )
            deepen_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestDeepRun(\'{esc(run.get("run_id") or "")}\')">Go deeper</button></div>'
                if (not archived and run.get("status") == "complete")
                and run.get("execution_mode") in {"quick", "interactive"}
                and run.get("route_name") in {"democritus", "company_similarity"}
                else ""
            )
            rerun_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestArchivedRerun(\'{esc(run.get("run_id") or "")}\')">Re-run query</button></div>'
                if archived
                else ""
            )
            wrong_route_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestWrongRoute(\'{esc(run.get("run_id") or "")}\')">Wrong route</button></div>'
                if run.get("status") == "complete" and run.get("route_name")
                else ""
            )
            open_action_markup = (
                f'<div class="run-actions"><a class="run-link" href="/run-artifact?run_id={esc(run.get("run_id") or "")}" target="_blank" rel="noopener noreferrer">{"Inspect run" if run.get("status") in {"queued", "routing", "running", "stopping"} else "Open result"}</a></div>'
                if run.get("artifact_path")
                else ""
            )
            outdir_markup = (
                f'<div class="run-meta"><strong>Output:</strong> <code>{esc(run.get("outdir"))}</code></div>'
                if run.get("outdir")
                else ""
            )
            artifact_markup = (
                f'<div class="run-meta"><strong>Artifact:</strong> <code>{esc(run.get("artifact_path"))}</code></div>'
                if run.get("artifact_path")
                else ""
            )
            unconscious_markup = (
                f'<div class="run-meta"><strong>Unconscious report:</strong> {esc(run.get("eta_label") or "ETA warming up")}'
                + (f' · {esc(run.get("parallelism_label"))}' if run.get("parallelism_label") else "")
                + (f' · {esc(run.get("current_stage_label"))}' if run.get("current_stage_label") else "")
                + "</div>"
                if run.get("eta_label") or run.get("parallelism_label") or run.get("current_stage_label")
                else ""
            )
            research_note_markup = (
                f'<div class="run-meta"><strong>Latency class:</strong> {esc(run.get("research_profile_note"))}</div>'
                if run.get("research_profile_note")
                else ""
            )
            llm_budget_markup = (
                f'<div class="run-meta"><strong>LLM budget:</strong> {esc(run.get("llm_budget_label"))}</div>'
                if run.get("llm_budget_label")
                else ""
            )
            llm_usage_markup = (
                f'<div class="run-meta"><strong>LLM usage:</strong> {esc(run.get("llm_usage_label"))}</div>'
                if run.get("llm_usage_label")
                else ""
            )
            cards.append(
                f'<article class="{card_class}">'
                '<div class="run-topline">'
                f'<div class="run-id">{esc(run.get("run_id") or "")}</div>'
                '<div class="run-badges">'
                f'<span class="run-chip status-{esc(run.get("status") or "queued")}">{esc(run.get("status") or "queued")}</span>'
                f"{archived_markup}"
                f"{route_markup}"
                f"{research_profile_markup}"
                f"{mode_markup}"
                f"{eta_markup}"
                "</div>"
                "</div>"
                f'<div class="run-query">{esc(run.get("query") or "")}</div>'
                f'<div class="run-note">{esc(run.get("note") or "")}</div>'
                f"{stop_action_markup}"
                f"{deepen_action_markup}"
                f"{rerun_action_markup}"
                f"{wrong_route_action_markup}"
                f"{open_action_markup}"
                f"{research_note_markup}"
                f"{llm_budget_markup}"
                f"{llm_usage_markup}"
                f"{unconscious_markup}"
                f"{outdir_markup}"
                f"{artifact_markup}"
                "</article>"
            )
        return "".join(cards)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a standalone FunctorFlow query launcher GUI and wait for a natural-language submission."
    )
    parser.add_argument("--title", default="FunctorFlow Query Launcher")
    parser.add_argument(
        "--subtitle",
        default=(
            "Enter a natural-language query to confirm the launcher GUI is working. "
            "For full CLIFF orchestration, use `python -m functorflow_v3.cliff --outdir ...`."
        ),
    )
    parser.add_argument("--query-label", default="Query")
    parser.add_argument(
        "--query-placeholder",
        default=(
            "Find me 10 recent AMD 10-K filings\n"
            "or\n"
            "Find me 10 recent studies of the benefits of red wine"
        ),
    )
    parser.add_argument("--submit-label", default="Submit Query")
    parser.add_argument(
        "--waiting-message",
        default="The launcher received the query and is keeping this window open.",
    )
    parser.add_argument("--eyebrow", default="FunctorFlow Launcher")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with DashboardQueryLauncher(
        DashboardQueryLauncherConfig(
            title=args.title,
            subtitle=args.subtitle,
            query_label=args.query_label,
            query_placeholder=args.query_placeholder,
            submit_label=args.submit_label,
            waiting_message=args.waiting_message,
            eyebrow=args.eyebrow,
        )
    ) as launcher:
        print(json.dumps({"url": launcher.url, "mode": "one_shot_launcher"}, indent=2), flush=True)
        query = launcher.wait_for_submission()
        print(json.dumps({"query": query}, indent=2), flush=True)


if __name__ == "__main__":
    main()
