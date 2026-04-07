"""Local dashboard launcher for query-first BAFFLE workflows."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import threading
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
    eyebrow: str = "FunctorFlow Dashboard Launch"
    artifact_path: Path | None = None
    session_mode: bool = False
    run_control_handler: Callable[[str, str], None] | None = None


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

    def wait_for_submission(self) -> str:
        self._submitted_event.wait()
        with self._lock:
            return self._submitted_query

    def wait_for_next_submission(self, timeout: float | None = None) -> tuple[str, str] | None:
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

    def submit_query(self, query: str) -> str:
        normalized_query = " ".join(str(query).split()).strip()
        if not normalized_query:
            raise ValueError("Query must be non-empty.")
        with self._lock:
            if not self.config.session_mode:
                self._submitted_query = normalized_query
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
            }
            self._submission_queue.append((run_id, normalized_query))
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
                query = " ".join(parse_qs(payload).get("query", [""])[0].split()).strip()
                if not query:
                    self._send_html(
                        launcher._render_launcher_page(error_message="Enter a query before starting the run."),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                launcher.submit_query(query)
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
                "query_received": self._query_received,
                "artifact_ready": bool(self._artifact_path and self._artifact_path.exists()),
                "artifact_path": str(self._artifact_path) if self._artifact_path else None,
            }
            if self.config.session_mode:
                payload["runs"] = [dict(item) for item in self._session_runs]
            return payload

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
        artifact_path_value = str(run_state.get("artifact_path") or "").strip()
        artifact_path = Path(artifact_path_value) if artifact_path_value else None
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

    def _render_live_run_artifact_shell(
        self,
        run_id: str,
        *,
        artifact_path: Path,
        run_state: dict[str, object],
    ) -> str:
        title = html.escape(self.config.title)
        query = html.escape(str(run_state.get("query") or ""))
        note = html.escape(str(run_state.get("note") or "CLIFF is gathering live partial outputs from the unconscious run."))
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
        <p class="eyebrow">Inspecting Unconscious Run</p>
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
            session_runs = [dict(item) for item in self._session_runs]
        title = html.escape(self.config.title)
        subtitle = html.escape(self.config.subtitle)
        eyebrow = html.escape(self.config.eyebrow)
        query_label = html.escape(self.config.query_label)
        placeholder = html.escape(self.config.query_placeholder)
        submit_label = html.escape(self.config.submit_label)
        waiting_message = html.escape(self.config.waiting_message)
        session_mode = self.config.session_mode
        error_html = (
            f'<p class="error" role="alert">{html.escape(error_message)}</p>'
            if error_message
            else ""
        )
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
            {error_html}
            <button type="submit">{submit_label}</button>
          </form>
        """
        if session_mode:
            form_or_status = f"""
          <section class="session-shell">
            <div class="session-intro">
              <p class="status-kicker">Persistent Session</p>
              <h2>The prompt window stays open for the next query.</h2>
              <p>{waiting_message} Long-running jobs keep working in the background and completed results can be opened from the conscious-layer run list.</p>
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
              {error_html}
              <button type="submit">{submit_label}</button>
            </form>
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
                var route = run.route_name ? '<span class="run-chip">' + escapeHtml(run.route_name) + '</span>' : '';
                var mind = run.mind_layer ? '<span class="run-chip mind-' + escapeHtml(run.mind_layer) + '">' + escapeHtml(run.mind_layer) + '</span>' : '';
                var attention = (run.status || '') === 'complete'
                  ? '<span class="run-chip attention-ready">ready in conscious layer</span>'
                  : '';
                var stopAction = ((run.status || '') === 'queued' || (run.status || '') === 'routing' || (run.status || '') === 'running')
                  ? '<div class="run-actions"><button type="button" class="run-stop-button" onclick="requestStopRun(\\'' + escapeHtml(run.run_id || '') + '\\')">Stop query</button></div>'
                  : '';
                var inspectLabel = ((run.status || '') === 'queued' || (run.status || '') === 'routing' || (run.status || '') === 'running' || (run.status || '') === 'stopping')
                  ? 'Inspect run'
                  : 'Open result';
                var openAction = run.artifact_path
                  ? '<div class="run-actions"><a class="run-link" href="/run-artifact?run_id=' + encodeURIComponent(run.run_id || '') + '" target="_blank" rel="noopener noreferrer">' + inspectLabel + '</a></div>'
                  : '';
                var outdir = run.outdir ? '<div class="run-meta"><strong>Output:</strong> <code>' + escapeHtml(run.outdir) + '</code></div>' : '';
                var artifact = run.artifact_path ? '<div class="run-meta"><strong>Artifact:</strong> <code>' + escapeHtml(run.artifact_path) + '</code></div>' : '';
                return ''
                  + '<article class="' + cardClass + '">'
                  + '<div class="run-topline">'
                  + '<div class="run-id">' + escapeHtml(run.run_id) + '</div>'
                  + '<div class="run-badges"><span class="run-chip status-' + escapeHtml(run.status || 'queued') + '">' + escapeHtml(run.status || 'queued') + '</span>' + attention + mind + route + '</div>'
                  + '</div>'
                  + '<div class="run-query">' + escapeHtml(run.query || '') + '</div>'
                  + '<div class="run-note">' + escapeHtml(run.note || '') + '</div>'
                  + stopAction
                  + openAction
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
      .status-failed {{
        background: #f8dddd;
        color: var(--error);
      }}
      .attention-ready {{
        background: #e4f6e7;
        color: #166534;
        font-weight: 700;
      }}
      .mind-conscious {{
        background: #e5eefc;
        color: #1d4f91;
      }}
      .mind-unconscious {{
        background: #eee3fb;
        color: #6b21a8;
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
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <article class="panel">
          <p class="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p class="subtitle">{subtitle}</p>
        </article>
        <article class="panel">
          {form_or_status}
        </article>
      </section>
    </main>
  </body>
</html>
"""

    def _render_session_runs_markup(self, runs: list[dict[str, object]]) -> str:
        if not runs:
            return '<div class="empty-state">No queries have been submitted in this session yet.</div>'

        def esc(value: object) -> str:
            return html.escape(str(value))

        cards: list[str] = []
        for run in runs:
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
            mind_markup = (
                f'<span class="run-chip mind-{esc(run.get("mind_layer"))}">{esc(run.get("mind_layer"))}</span>'
                if run.get("mind_layer")
                else ""
            )
            attention_markup = (
                '<span class="run-chip attention-ready">ready in conscious layer</span>'
                if run.get("status") == "complete"
                else ""
            )
            stop_action_markup = (
                f'<div class="run-actions"><button type="button" class="run-stop-button" onclick="requestStopRun(\'{esc(run.get("run_id") or "")}\')">Stop query</button></div>'
                if run.get("status") in {"queued", "routing", "running"}
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
            cards.append(
                f'<article class="{card_class}">'
                '<div class="run-topline">'
                f'<div class="run-id">{esc(run.get("run_id") or "")}</div>'
                '<div class="run-badges">'
                f'<span class="run-chip status-{esc(run.get("status") or "queued")}">{esc(run.get("status") or "queued")}</span>'
                f"{attention_markup}"
                f"{mind_markup}"
                f"{route_markup}"
                "</div>"
                "</div>"
                f'<div class="run-query">{esc(run.get("query") or "")}</div>'
                f'<div class="run-note">{esc(run.get("note") or "")}</div>'
                f"{stop_action_markup}"
                f"{open_action_markup}"
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
