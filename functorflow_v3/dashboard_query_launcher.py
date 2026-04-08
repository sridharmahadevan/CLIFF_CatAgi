"""Local dashboard launcher for query-first BAFFLE workflows."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlparse


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
        self._submission_queue: deque[tuple[str, str]] = deque()
        self._session_runs: list[dict[str, object]] = []
        self._session_runs_by_id: dict[str, dict[str, object]] = {}
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
        return "deep" if str(value or "").strip().lower() == "deep" else "quick"

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

    def submit_query(self, query: str, *, execution_mode: str | None = None, parent_run_id: str | None = None) -> str:
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
                run_state["note"] = f"Queued as a deep follow-up to {parent_run_id}."
            self._submission_queue.append((run_id, normalized_query, normalized_mode))
            self._session_runs.insert(0, run_state)
            self._session_runs_by_id[run_id] = run_state
            self._query_received = True
            self._submitted_event.set()
            return run_id

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

    def request_session_run_deepen(self, run_id: str) -> str | None:
        with self._lock:
            run_state = self._session_runs_by_id.get(run_id)
            if run_state is None or not self.config.session_mode or not self.config.enable_execution_mode:
                return None
            if str(run_state.get("status") or "") != "complete":
                return None
            if self._normalize_execution_mode(run_state.get("execution_mode")) != "quick":
                return None
            if str(run_state.get("route_name") or "") not in {"democritus", "company_similarity"}:
                return None
            query = str(run_state.get("query") or "").strip()
        if not query:
            return None
        return self.submit_query(query, execution_mode="deep", parent_run_id=run_id)

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
                launcher.submit_query(query, execution_mode=execution_mode)
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
            payload = {
                "session_mode": self.config.session_mode,
                "execution_mode_enabled": self.config.enable_execution_mode,
                "query_received": self._query_received,
                "artifact_ready": bool(self._artifact_path and self._artifact_path.exists()),
                "artifact_path": str(self._artifact_path) if self._artifact_path else None,
            }
            if self.config.session_mode:
                payload["runs"] = [self._enriched_run_state(dict(item)) for item in self._session_runs]
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
    <meta http-equiv="refresh" content="3" />
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
        with self._lock:
            run_state = dict(self._session_runs_by_id.get(run_id) or {})
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
      }}, 3000);
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
      }}, 3000);
    </script>
  </body>
</html>
"""

    def _resolve_run_file_path(self, run_id: str, requested_path: str) -> Path | None:
        with self._lock:
            run_state = dict(self._session_runs_by_id.get(run_id) or {})
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
            session_runs = [self._enriched_run_state(dict(item)) for item in self._session_runs]
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
                <input type="radio" name="execution_mode" value="deep" />
                <span><strong>Deep</strong> Run the fuller causal build from the start for maximum coverage.</span>
              </label>
              <p class="execution-mode-hint">
                Latency guide: textbook and filing lookups are usually quickest, product evaluation can take longer,
                and deep research routes like Democritus or company similarity may still take several minutes even in quick mode.
              </p>
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

            function renderRuns(runs) {{
              if (!Array.isArray(runs) || runs.length === 0) {{
                return '<div class="empty-state">No queries have been submitted in this session yet.</div>';
              }}
              return runs.map(function (run) {{
                var cardClass = 'run-card';
                if ((run.status || '') === 'complete') {{
                  cardClass += ' run-card-complete';
                }} else if ((run.status || '') === 'failed') {{
                  cardClass += ' run-card-failed';
                }}
                var executionMode = String(run.execution_mode || 'quick');
                var route = run.route_name ? '<span class="run-chip">' + escapeHtml(run.route_name) + '</span>' : '';
                var researchProfile = run.research_profile_label
                  ? '<span class="run-chip route-profile-' + escapeHtml(run.research_profile_class || 'route-warming-up') + '">' + escapeHtml(run.research_profile_label) + '</span>'
                  : '';
                var mode = executionMode ? '<span class="run-chip mode-' + escapeHtml(executionMode) + '">' + escapeHtml(executionMode) + '</span>' : '';
                var eta = run.eta_label ? '<span class="run-chip eta-chip">' + escapeHtml(run.eta_label) + '</span>' : '';
                var stopAction = ((run.status || '') === 'queued' || (run.status || '') === 'routing' || (run.status || '') === 'running')
                  ? '<div class="run-actions"><button type="button" class="run-stop-button" onclick="requestStopRun(\\'' + escapeHtml(run.run_id || '') + '\\')">Stop query</button></div>'
                  : '';
                var deepenAction = ((run.status || '') === 'complete' && executionMode === 'quick' && ((run.route_name || '') === 'democritus' || (run.route_name || '') === 'company_similarity'))
                  ? '<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestDeepRun(\\'' + escapeHtml(run.run_id || '') + '\\')">Go deeper</button></div>'
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
                  + '<div class="run-badges"><span class="run-chip status-' + escapeHtml(run.status || 'queued') + '">' + escapeHtml(run.status || 'queued') + '</span>' + route + researchProfile + mode + eta + '</div>'
                  + '</div>'
                  + '<div class="run-query">' + escapeHtml(run.query || '') + '</div>'
                  + '<div class="run-note">' + escapeHtml(run.note || '') + '</div>'
                  + stopAction
                  + deepenAction
                  + openAction
                  + researchNote
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
                  container.innerHTML = renderRuns(payload.runs || []);
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
                  container.innerHTML = renderRuns(payload.runs || []);
                }})
                .catch(function () {{}});
            }}

            window.setInterval(function () {{
              fetch('/state?ts=' + Date.now(), {{ cache: 'no-store' }})
                .then(function (response) {{ return response.json(); }})
                .then(function (payload) {{
                  var container = document.getElementById('session-runs');
                  if (!container) {{
                    return;
                  }}
                  container.innerHTML = renderRuns(payload.runs || []);
                }})
                .catch(function () {{}});
            }}, 2000);
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
            }}, 3000);
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

    def _render_session_runs_markup(self, runs: list[dict[str, object]]) -> str:
        if not runs:
            return '<div class="empty-state">No queries have been submitted in this session yet.</div>'

        def esc(value: object) -> str:
            return html.escape(str(value))

        cards: list[str] = []
        for run in (self._enriched_run_state(dict(item)) for item in runs):
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
                if run.get("status") in {"queued", "routing", "running"}
                else ""
            )
            deepen_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-deepen-button" onclick="requestDeepRun(\'{esc(run.get("run_id") or "")}\')">Go deeper</button></div>'
                if run.get("status") == "complete"
                and run.get("execution_mode") == "quick"
                and run.get("route_name") in {"democritus", "company_similarity"}
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
            cards.append(
                f'<article class="{card_class}">'
                '<div class="run-topline">'
                f'<div class="run-id">{esc(run.get("run_id") or "")}</div>'
                '<div class="run-badges">'
                f'<span class="run-chip status-{esc(run.get("status") or "queued")}">{esc(run.get("status") or "queued")}</span>'
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
                f"{open_action_markup}"
                f"{research_note_markup}"
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
