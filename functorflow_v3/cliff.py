"""CLIFF entrypoint: Conscious Learning in Functor Flow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .basket_rocket_corpus_synthesis import build_basket_rocket_corpus_synthesis
from .consciousness import ConsciousFieldOfView, ConsciousWorkspaceState, ConsciousnessFunctor, UnconsciousProcess
from .dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
from .democritus_corpus_synthesis import build_democritus_corpus_synthesis
from .product_feedback_corpus_synthesis import build_product_feedback_corpus_synthesis
from .query_router_agentic import (
    FF3QueryRouter,
    FF3QueryRouterConfig,
    FF3QueryRouterRunResult,
    FF3RouteDecision,
    _artifact_path_for_result,
    _build_router_from_args,
    _build_router_from_args_with_outdir,
    _open_artifact,
    _predicted_artifact_path,
    _router_launcher_artifact_path,
    _session_query_outdir,
    _write_router_error_artifact,
    route_ff3_query,
)


CLIFFRouteDecision = FF3RouteDecision
CLIFFRouterConfig = FF3QueryRouterConfig
CLIFFRouterRunResult = FF3QueryRouterRunResult
CLIFFRouter = FF3QueryRouter
_WORKER_RESULT_FILENAME = "cliff_worker_result.json"
_REDISPATCHABLE_ROUTES = {"democritus", "basket_rocket_sec", "product_feedback"}


@dataclass(frozen=True)
class CLIFFConsciousReturn:
    """A report accepted back into CLIFF's conscious layer."""

    query: str
    route_decision: CLIFFRouteDecision
    workspace_state: ConsciousWorkspaceState
    artifact_path: Path | None = None


@dataclass
class _ActiveCLIFFWorker:
    process: subprocess.Popen[str]
    stdout_handle: object
    stderr_handle: object
    run_outdir: Path
    stage: str = "first_pass"
    stop_requested: bool = False


def route_cliff_query(query: str, *, route_override: str = "auto") -> CLIFFRouteDecision:
    """Route an NLP query through CLIFF's unconscious orchestrator."""

    return route_ff3_query(query, route_override=route_override)


def report_to_cliff_consciousness(
    query: str,
    route_decision: CLIFFRouteDecision,
    *,
    artifact_path: Path | None = None,
) -> CLIFFConsciousReturn:
    """Model a completed unconscious run returning to CLIFF's conscious layer."""

    process = UnconsciousProcess(
        name=f"{route_decision.route_name}_report",
        source_agent=route_decision.route_name,
        summary=query,
        artifact_refs=(str(artifact_path),) if artifact_path else (),
        salience=0.95,
        relevance=1.0,
        novelty=0.60,
        urgency=0.80,
    )
    workspace_state = ConsciousnessFunctor(ConsciousFieldOfView(capacity=3)).competition_for_access([process])
    return CLIFFConsciousReturn(
        query=query,
        route_decision=route_decision,
        workspace_state=workspace_state,
        artifact_path=artifact_path,
    )


def _cliff_complete_note(report: CLIFFConsciousReturn) -> str:
    if report.workspace_state.selected:
        return "CLIFF finished the query and brought the result back into your session."
    return "CLIFF finished the background work, but the result is still waiting for your attention."


def _worker_result_path(run_outdir: Path) -> Path:
    return run_outdir / _WORKER_RESULT_FILENAME


def _cliff_cycle_complete_note(report: CLIFFConsciousReturn) -> str:
    base = _cliff_complete_note(report)
    return (
        "CLIFF completed a first pass, then ran a second synthesis pass to refine the result. "
        + base
    )


def _decision_supports_conscious_redispatch(decision: CLIFFRouteDecision) -> bool:
    return decision.route_name in _REDISPATCHABLE_ROUTES


def _synthesis_artifact_path_for_decision(run_outdir: Path, decision: CLIFFRouteDecision) -> Path | None:
    route_outdir = run_outdir / decision.route_name
    if decision.route_name == "democritus":
        return route_outdir / "democritus_runs" / "corpus_synthesis" / "democritus_corpus_synthesis.html"
    if decision.route_name == "basket_rocket_sec":
        return route_outdir / "workflow_batches" / "corpus_synthesis" / "basket_rocket_corpus_synthesis.html"
    if decision.route_name == "product_feedback":
        return route_outdir / "product_feedback_run" / "corpus_synthesis" / "product_feedback_corpus_synthesis.html"
    return _predicted_artifact_path(run_outdir, decision)


def _build_cliff_synthesis_from_first_pass(
    *,
    query: str,
    decision: CLIFFRouteDecision,
    run_outdir: Path,
) -> Path | None:
    route_outdir = run_outdir / decision.route_name
    try:
        if decision.route_name == "democritus":
            summary_path = route_outdir / "query_run_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            csql_sqlite_raw = str(payload.get("csql_sqlite_path") or "").strip()
            if not csql_sqlite_raw:
                return None
            csql_sqlite_path = Path(csql_sqlite_raw).resolve()
            batch_outdir = Path(str(payload.get("batch_outdir") or route_outdir / "democritus_runs")).resolve()
            if not csql_sqlite_path.exists():
                return None
            return build_democritus_corpus_synthesis(
                query=query,
                batch_outdir=batch_outdir,
                csql_sqlite_path=csql_sqlite_path,
            ).dashboard_path
        if decision.route_name == "basket_rocket_sec":
            summary_path = route_outdir / "basket_rocket_run_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            batch_outdir = route_outdir / "workflow_batches"
            workset_index_raw = str(payload.get("batch_workset_index_path") or "").strip()
            if not workset_index_raw:
                return None
            workset_index_path = Path(workset_index_raw).resolve()
            visualization_summary_raw = str(payload.get("batch_visualization_summary_path") or "").strip()
            visualization_summary_path = Path(visualization_summary_raw).resolve() if visualization_summary_raw else None
            company_summary_path = batch_outdir / "visualizations" / "company_index_summary.json"
            if not workset_index_path.exists():
                return None
            return build_basket_rocket_corpus_synthesis(
                query=query,
                batch_outdir=batch_outdir,
                workset_index_path=workset_index_path,
                visualization_summary_path=(
                    visualization_summary_path if visualization_summary_path and visualization_summary_path.exists() else None
                ),
                company_summary_path=company_summary_path if company_summary_path.exists() else None,
            ).dashboard_path
        if decision.route_name == "product_feedback":
            analysis_outdir = route_outdir / "product_feedback_run"
            return build_product_feedback_corpus_synthesis(
                query=query,
                outdir=analysis_outdir,
                analysis_outdir=analysis_outdir,
            ).dashboard_path
    except FileNotFoundError:
        return None
    return None


def _build_worker_command(
    args: argparse.Namespace,
    *,
    run_outdir: Path,
    query: str,
    cycle_stage: str = "first_pass",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "functorflow_v3.cliff_worker",
        "--query",
        query,
        "--outdir",
        str(run_outdir),
        "--cycle-stage",
        cycle_stage,
        "--execution-mode",
        str(getattr(args, "execution_mode", "quick")),
        "--route",
        str(args.route),
        "--democritus-retrieval-backend",
        str(args.democritus_retrieval_backend),
        "--democritus-intra-document-shards",
        str(args.democritus_intra_document_shards),
        "--democritus-manifold-mode",
        str(getattr(args, "democritus_manifold_mode", "full")),
        "--democritus-topk",
        str(getattr(args, "democritus_topk", 200)),
        "--democritus-radii",
        str(getattr(args, "democritus_radii", "1,2,3")),
        "--democritus-maxnodes",
        str(getattr(args, "democritus_maxnodes", "10,20,30,40,60")),
        "--democritus-lambda-edge",
        str(getattr(args, "democritus_lambda_edge", 0.25)),
        "--democritus-topk-models",
        str(getattr(args, "democritus_topk_models", 5)),
        "--democritus-topk-claims",
        str(getattr(args, "democritus_topk_claims", 30)),
        "--democritus-alpha",
        str(getattr(args, "democritus_alpha", 1.0)),
        "--democritus-tier1",
        str(getattr(args, "democritus_tier1", 0.60)),
        "--democritus-tier2",
        str(getattr(args, "democritus_tier2", 0.30)),
        "--democritus-assets-dir",
        str(getattr(args, "democritus_assets_dir", "assets")),
        "--democritus-png-dpi",
        str(getattr(args, "democritus_png_dpi", 200)),
        "--sec-company-limit",
        str(args.sec_company_limit),
    ]
    if cycle_stage == "first_pass":
        command.append("--cliff-defer-final-synthesis")
    if getattr(args, "democritus_input_pdf", ""):
        command.extend(["--democritus-input-pdf", str(args.democritus_input_pdf)])
    if getattr(args, "democritus_input_pdf_dir", ""):
        command.extend(["--democritus-input-pdf-dir", str(args.democritus_input_pdf_dir)])
    if args.democritus_manifest:
        command.extend(["--democritus-manifest", str(args.democritus_manifest)])
    if args.democritus_source_pdf_root:
        command.extend(["--democritus-source-pdf-root", str(args.democritus_source_pdf_root)])
    if args.democritus_target_docs is not None:
        command.extend(["--democritus-target-docs", str(args.democritus_target_docs)])
    if args.democritus_max_docs is not None:
        command.extend(["--democritus-max-docs", str(args.democritus_max_docs)])
    if getattr(args, "democritus_anchors", ""):
        command.extend(["--democritus-anchors", str(args.democritus_anchors)])
    if getattr(args, "democritus_title", ""):
        command.extend(["--democritus-title", str(args.democritus_title)])
    if getattr(args, "democritus_dedupe_focus", False):
        command.append("--democritus-dedupe-focus")
    if getattr(args, "democritus_require_anchor_in_focus", False):
        command.append("--democritus-require-anchor-in-focus")
    if getattr(args, "democritus_focus_blacklist_regex", ""):
        command.extend(["--democritus-focus-blacklist-regex", str(args.democritus_focus_blacklist_regex)])
    if getattr(args, "democritus_render_topk_pngs", False):
        command.append("--democritus-render-topk-pngs")
    if getattr(args, "democritus_write_deep_dive", False):
        command.append("--democritus-write-deep-dive")
    if getattr(args, "democritus_deep_dive_max_bullets", None) is not None:
        command.extend(
            [
                "--democritus-deep-dive-max-bullets",
                str(getattr(args, "democritus_deep_dive_max_bullets")),
            ]
        )
    if args.democritus_discovery_only:
        command.append("--democritus-discovery-only")
    if args.democritus_dry_run:
        command.append("--democritus-dry-run")
    if args.product_manifest:
        command.extend(["--product-manifest", str(args.product_manifest)])
    if getattr(args, "culinary_manifest", ""):
        command.extend(["--culinary-manifest", str(args.culinary_manifest)])
    if args.product_target_docs is not None:
        command.extend(["--product-target-docs", str(args.product_target_docs)])
    if args.product_max_docs is not None:
        command.extend(["--product-max-docs", str(args.product_max_docs)])
    if args.product_name:
        command.extend(["--product-name", str(args.product_name)])
    if args.brand_name:
        command.extend(["--brand-name", str(args.brand_name)])
    if args.analysis_question:
        command.extend(["--analysis-question", str(args.analysis_question)])
    if args.product_discovery_only:
        command.append("--product-discovery-only")
    if args.sec_target_filings is not None:
        command.extend(["--sec-target-filings", str(args.sec_target_filings)])
    if args.sec_retrieval_user_agent:
        command.extend(["--sec-retrieval-user-agent", str(args.sec_retrieval_user_agent)])
    for sec_form in list(args.sec_form or []):
        command.extend(["--sec-form", str(sec_form)])
    if args.sec_discovery_only:
        command.append("--sec-discovery-only")
    if args.sec_dry_run:
        command.append("--sec-dry-run")
    if getattr(args, "course_repo_root", ""):
        command.extend(["--course-repo-root", str(args.course_repo_root)])
    if getattr(args, "course_book_pdf_path", ""):
        command.extend(["--course-book-pdf-path", str(args.course_book_pdf_path)])
    if getattr(args, "course_no_execute", False):
        command.append("--course-no-execute")
    if getattr(args, "course_timeout_sec", None) is not None:
        command.extend(["--course-timeout-sec", str(args.course_timeout_sec)])
    return command


def _launch_cliff_worker(
    args: argparse.Namespace,
    *,
    run_outdir: Path,
    query: str,
    cycle_stage: str,
) -> _ActiveCLIFFWorker:
    stdout_path = run_outdir / f"cliff_worker_{cycle_stage}_stdout.log"
    stderr_path = run_outdir / f"cliff_worker_{cycle_stage}_stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        _build_worker_command(args, run_outdir=run_outdir, query=query, cycle_stage=cycle_stage),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env=env,
    )
    return _ActiveCLIFFWorker(
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        run_outdir=run_outdir,
        stage=cycle_stage,
    )


def _monitor_cliff_session_worker(
    launcher: DashboardQueryLauncher,
    *,
    args: argparse.Namespace,
    run_id: str,
    query: str,
    decision: CLIFFRouteDecision,
    active_runs: dict[str, _ActiveCLIFFWorker],
    active_runs_lock: threading.Lock,
) -> None:
    with active_runs_lock:
        worker = active_runs.get(run_id)
    if worker is None:
        return
    return_code = worker.process.wait()
    try:
        if worker.stdout_handle:
            worker.stdout_handle.close()
        if worker.stderr_handle:
            worker.stderr_handle.close()
    finally:
        pass

    with active_runs_lock:
        worker = active_runs.pop(run_id, worker)

    if worker.stop_requested:
        launcher.update_session_run(
            run_id,
            status="stopped",
            mind_layer="conscious",
            route_name=decision.route_name,
            note="The unconscious run was stopped before it could report a final result back to CLIFF.",
            outdir=worker.run_outdir,
        )
        return

    result_path = _worker_result_path(worker.run_outdir)
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "phase1_complete":
            artifact_value = str(payload.get("artifact_path") or "").strip()
            artifact_path = Path(artifact_value) if artifact_value else None
            conscious_report = report_to_cliff_consciousness(query, decision, artifact_path=artifact_path)
            launcher.update_session_run(
                run_id,
                status="routing",
                mind_layer="conscious",
                route_name=decision.route_name,
                note=(
                    "The unconscious orchestrator completed a first pass and reported its partial result into "
                    "CLIFF's conscious layer. Consciousness is now initiating a second unconscious synthesis pass."
                ),
                artifact_path=artifact_path,
                outdir=worker.run_outdir,
            )
            resumed_worker = _launch_cliff_worker(
                args,
                run_outdir=worker.run_outdir,
                query=query,
                cycle_stage="synthesis_pass",
            )
            with active_runs_lock:
                active_runs[run_id] = resumed_worker
            launcher.update_session_run(
                run_id,
                status="running",
                mind_layer="unconscious",
                route_name=decision.route_name,
                note="CLIFF's conscious layer has re-dispatched the integrated corpus/workset synthesis into the unconscious layer.",
                artifact_path=_synthesis_artifact_path_for_decision(worker.run_outdir, decision),
                outdir=worker.run_outdir,
            )
            threading.Thread(
                target=_monitor_cliff_session_worker,
                kwargs={
                    "launcher": launcher,
                    "args": args,
                    "run_id": run_id,
                    "query": query,
                    "decision": decision,
                    "active_runs": active_runs,
                    "active_runs_lock": active_runs_lock,
                },
                name=f"cliff-monitor-{run_id}-phase2",
                daemon=True,
            ).start()
            return
        if payload.get("status") == "complete":
            artifact_value = str(payload.get("artifact_path") or "").strip()
            artifact_path = Path(artifact_value) if artifact_value else None
            conscious_report = report_to_cliff_consciousness(query, decision, artifact_path=artifact_path)
            launcher.update_session_run(
                run_id,
                status="complete",
                mind_layer="conscious",
                route_name=decision.route_name,
                note=(
                    (
                        _cliff_cycle_complete_note(conscious_report)
                        if worker.stage == "synthesis_pass"
                        else _cliff_complete_note(conscious_report)
                    )
                    + " The result is ready in the session list and won't open automatically."
                ),
                artifact_path=artifact_path,
                outdir=worker.run_outdir,
            )
            return
        error_artifact_value = str(payload.get("error_artifact_path") or "").strip()
        error_artifact_path = Path(error_artifact_value) if error_artifact_value else None
        launcher.update_session_run(
            run_id,
            status="failed",
            mind_layer="conscious",
            route_name=decision.route_name,
            note=(
                "A background run failed: "
                + str(payload.get("error") or f"worker exited with code {return_code}")
            ),
            artifact_path=error_artifact_path,
            outdir=worker.run_outdir,
        )
        return

    launcher.update_session_run(
        run_id,
        status="failed",
        mind_layer="conscious",
        route_name=decision.route_name,
        note=f"A background worker exited unexpectedly with code {return_code}.",
        outdir=worker.run_outdir,
    )


def _start_cliff_session_query(
    launcher: DashboardQueryLauncher,
    args: argparse.Namespace,
    *,
    run_id: str,
    query: str,
    active_runs: dict[str, _ActiveCLIFFWorker],
    active_runs_lock: threading.Lock,
) -> None:
    run_outdir = _session_query_outdir(Path(args.outdir), run_id=run_id, query=query)
    decision = route_cliff_query(query, route_override=args.route)
    predicted_artifact_path = _predicted_artifact_path(run_outdir, decision)
    run_outdir.mkdir(parents=True, exist_ok=True)
    launcher.update_session_run(
        run_id,
        status="routing",
        mind_layer="conscious",
        route_name=decision.route_name,
        note="CLIFF accepted the query and is sending it to the background runner.",
        artifact_path=predicted_artifact_path,
        outdir=run_outdir,
    )
    worker = _launch_cliff_worker(
        args,
        run_outdir=run_outdir,
        query=query,
        cycle_stage="first_pass",
    )
    with active_runs_lock:
        active_runs[run_id] = worker
    launcher.update_session_run(
        run_id,
        status="running",
        mind_layer="unconscious",
        route_name=decision.route_name,
        note=f"CLIFF is running the {decision.route_name} workflow in the background.",
        artifact_path=predicted_artifact_path,
        outdir=run_outdir,
    )
    threading.Thread(
        target=_monitor_cliff_session_worker,
        kwargs={
            "launcher": launcher,
            "args": args,
            "run_id": run_id,
            "query": query,
            "decision": decision,
            "active_runs": active_runs,
            "active_runs_lock": active_runs_lock,
        },
        name=f"cliff-monitor-{run_id}",
        daemon=True,
    ).start()


def _handle_cliff_run_control(
    action: str,
    run_id: str,
    *,
    active_runs: dict[str, _ActiveCLIFFWorker],
    active_runs_lock: threading.Lock,
) -> None:
    if action != "stop":
        return
    with active_runs_lock:
        worker = active_runs.get(run_id)
        if worker is None:
            return
        worker.stop_requested = True
        process = worker.process
    if process.poll() is None:
        process.terminate()

        def _escalate_stop() -> None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()

        threading.Thread(
            target=_escalate_stop,
            name=f"cliff-stop-{run_id}",
            daemon=True,
        ).start()


def _run_cliff_session_query(
    launcher: DashboardQueryLauncher,
    args: argparse.Namespace,
    *,
    run_id: str,
    query: str,
) -> None:
    run_outdir = _session_query_outdir(Path(args.outdir), run_id=run_id, query=query)
    decision = route_cliff_query(query, route_override=args.route)
    predicted_artifact_path = _predicted_artifact_path(run_outdir, decision)
    launcher.update_session_run(
        run_id,
        status="routing",
        mind_layer="conscious",
        route_name=decision.route_name,
        note="CLIFF accepted the query and is sending it to the background runner.",
        artifact_path=predicted_artifact_path,
        outdir=run_outdir,
    )
    try:
        launcher.update_session_run(
            run_id,
            status="running",
            mind_layer="unconscious",
            route_name=decision.route_name,
            note=f"CLIFF is running the {decision.route_name} workflow in the background.",
            artifact_path=predicted_artifact_path,
            outdir=run_outdir,
        )
        if _decision_supports_conscious_redispatch(decision):
            first_pass_args = argparse.Namespace(**vars(args), cliff_defer_final_synthesis=True)
            result = _build_router_from_args_with_outdir(first_pass_args, query=query, outdir=run_outdir).run()
            first_pass_artifact_path = _artifact_path_for_result(result)
            launcher.update_session_run(
                run_id,
                status="routing",
                mind_layer="conscious",
                route_name=decision.route_name,
                note=(
                    "The unconscious orchestrator completed a first pass and reported its partial result into "
                    "CLIFF's conscious layer. Consciousness is now initiating a second unconscious synthesis pass."
                ),
                artifact_path=first_pass_artifact_path,
                outdir=run_outdir,
            )
            synthesis_artifact_path = _build_cliff_synthesis_from_first_pass(
                query=query,
                decision=decision,
                run_outdir=run_outdir,
            )
            artifact_path = synthesis_artifact_path or first_pass_artifact_path
        else:
            result = _build_router_from_args_with_outdir(args, query=query, outdir=run_outdir).run()
            artifact_path = _artifact_path_for_result(result)
        conscious_report = report_to_cliff_consciousness(query, decision, artifact_path=artifact_path)
        launcher.update_session_run(
            run_id,
            status="complete",
            mind_layer="conscious",
            route_name=decision.route_name,
            note=(
                (
                    _cliff_cycle_complete_note(conscious_report)
                    if _decision_supports_conscious_redispatch(decision)
                    else _cliff_complete_note(conscious_report)
                )
                + " The result is ready in the session list and won't open automatically."
            ),
            artifact_path=artifact_path,
            outdir=run_outdir,
        )
        print(
            json.dumps(
                {
                    "system_name": "CLIFF",
                    "run_id": run_id,
                    "query": query,
                    "route_decision": asdict(result.route_decision),
                    "route_outdir": str(result.route_outdir),
                    "summary_path": str(result.summary_path),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        hints: tuple[str, ...] = ()
        if "SEC retrieval requires an identifying User-Agent" in message:
            hints = (
                "Retry with `--sec-retrieval-user-agent 'Your Name your_email@example.com'`.",
                "Or export `FF3_SEC_USER_AGENT='Your Name your_email@example.com'` before running CLIFF.",
            )
        error_path = _write_router_error_artifact(
            run_outdir,
            title="CLIFF could not launch the selected workflow",
            message=message,
            detail=repr(exc),
            hints=hints,
        )
        launcher.update_session_run(
            run_id,
            status="failed",
            mind_layer="conscious",
            route_name=decision.route_name,
            note=f"A background run failed: {message}",
            artifact_path=error_path,
            outdir=run_outdir,
        )
        print(
            json.dumps(
                {
                    "system_name": "CLIFF",
                    "run_id": run_id,
                    "query": query,
                    "error": message,
                    "error_artifact_path": str(error_path),
                },
                indent=2,
            ),
            file=sys.stderr,
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch CLIFF (Conscious Learning in Functor Flow) as the single NLP entrypoint for FunctorFlow v3."
    )
    parser.add_argument(
        "--query",
        default="",
        help="Natural-language request for CLIFF. If omitted, CLIFF opens its GUI conscious interface.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--execution-mode", choices=("quick", "deep"), default="quick")
    parser.add_argument("--route", choices=("auto", "democritus", "basket_rocket_sec", "culinary_tour", "product_feedback", "company_similarity", "course_demo"), default="auto")
    parser.add_argument("--democritus-input-pdf", default="")
    parser.add_argument("--democritus-input-pdf-dir", default="")
    parser.add_argument("--democritus-manifest", default="")
    parser.add_argument("--democritus-source-pdf-root", default="")
    parser.add_argument("--democritus-target-docs", type=int, default=None)
    parser.add_argument("--democritus-retrieval-backend", default="auto")
    parser.add_argument("--democritus-max-docs", type=int, default=None)
    parser.add_argument("--democritus-intra-document-shards", type=int, default=1)
    parser.add_argument("--democritus-manifold-mode", default="full", choices=("full", "lite", "moe"))
    parser.add_argument("--democritus-topk", type=int, default=200)
    parser.add_argument("--democritus-radii", default="1,2,3")
    parser.add_argument("--democritus-maxnodes", default="10,20,30,40,60")
    parser.add_argument("--democritus-lambda-edge", type=float, default=0.25)
    parser.add_argument("--democritus-topk-models", type=int, default=5)
    parser.add_argument("--democritus-topk-claims", type=int, default=30)
    parser.add_argument("--democritus-alpha", type=float, default=1.0)
    parser.add_argument("--democritus-tier1", type=float, default=0.60)
    parser.add_argument("--democritus-tier2", type=float, default=0.30)
    parser.add_argument("--democritus-anchors", default="")
    parser.add_argument("--democritus-title", default="")
    parser.add_argument("--democritus-dedupe-focus", action="store_true")
    parser.add_argument("--democritus-require-anchor-in-focus", action="store_true")
    parser.add_argument("--democritus-focus-blacklist-regex", default="")
    parser.add_argument("--democritus-render-topk-pngs", action="store_true")
    parser.add_argument("--democritus-assets-dir", default="assets")
    parser.add_argument("--democritus-png-dpi", type=int, default=200)
    parser.add_argument("--democritus-write-deep-dive", action="store_true")
    parser.add_argument("--democritus-deep-dive-max-bullets", type=int, default=8)
    parser.add_argument("--democritus-discovery-only", action="store_true")
    parser.add_argument("--democritus-dry-run", action="store_true")
    parser.add_argument("--product-manifest", default="")
    parser.add_argument("--culinary-manifest", default="")
    parser.add_argument("--product-target-docs", type=int, default=None)
    parser.add_argument("--product-max-docs", type=int, default=None)
    parser.add_argument("--product-name", default="")
    parser.add_argument("--brand-name", default="")
    parser.add_argument("--analysis-question", default="")
    parser.add_argument("--product-discovery-only", action="store_true")
    parser.add_argument("--sec-target-filings", type=int, default=None)
    parser.add_argument(
        "--sec-retrieval-user-agent",
        "--retrieval-user-agent",
        dest="sec_retrieval_user_agent",
        default="",
        help=(
            "SEC identity string with contact info, for example "
            "'Your Name your_email@example.com'. This is required for SEC-backed routes "
            "unless FF3_SEC_USER_AGENT, FF2_SEC_USER_AGENT, SEC_USER_AGENT, SEC_IDENTITY, or SEC_CONTACT_NAME plus "
            "SEC_CONTACT_EMAIL is already set."
        ),
    )
    parser.add_argument("--sec-form", action="append", default=[])
    parser.add_argument("--sec-company-limit", type=int, default=3)
    parser.add_argument("--sec-discovery-only", action="store_true")
    parser.add_argument("--sec-dry-run", action="store_true")
    parser.add_argument("--course-repo-root", default="")
    parser.add_argument("--course-book-pdf-path", default="")
    parser.add_argument("--course-no-execute", action="store_true")
    parser.add_argument("--course-timeout-sec", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    launcher_artifact_path: Path | None = None
    active_runs: dict[str, _ActiveCLIFFWorker] = {}
    active_runs_lock = threading.Lock()
    try:
        inline_query = " ".join(str(args.query or "").split()).strip()
        if inline_query:
            query = inline_query
            result = _build_router_from_args(args, query=query).run()
        else:
            launcher_artifact_path = _router_launcher_artifact_path(Path(args.outdir))
            with DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="CLIFF",
                    subtitle=(
                        "CLIFF for Categories for AGI asks about ideas from the textbook, course demos, project directions, "
                        "and supporting applications. It helps connect the book to runnable examples, code fragments, "
                        "comparisons, and guided explorations. CLIFF is a research prototype, and its analyses should not "
                        "be treated as product endorsements, product criticism, or professional advice. AI-based analyses "
                        "can make mistakes."
                    ),
                    eyebrow="CLIFF",
                    query_label="CLIFF query",
                    query_placeholder=(
                        "Analyze 10 recent Adobe 10-K filings and extract their workflows\n"
                        "or\n"
                        "Analyze the PDF at /absolute/path/to/document.pdf\n"
                        "or\n"
                        "Analyze the PDFs in /absolute/path/to/folder\n"
                        "or\n"
                        "Plan a seafood tour for Boston from July 6th-10th\n"
                        "or\n"
                        "Analyze 10 recent studies on red wine and synthesize what they jointly support\n"
                        "or\n"
                        "How comfortable is the Lovesac sectional sofa?"
                    ),
                    submit_label="Ask CLIFF",
                    waiting_message="CLIFF keeps this workspace open while the background analysis runs, gathers what it needs, and prepares the result for you.",
                    demo_queries=(
                        "Explain the Geometric Transformer on the Sudoku problem",
                        "Explain how the Kan Extension Transformer works",
                        "What demo should I use for causality?",
                        "How comfortable is the Lovesac sectional sofa?",
                    ),
                    artifact_path=launcher_artifact_path,
                    session_mode=True,
                    enable_execution_mode=True,
                    run_control_handler=lambda action, run_id: _handle_cliff_run_control(
                        action,
                        run_id,
                        active_runs=active_runs,
                        active_runs_lock=active_runs_lock,
                    ),
                )
            ) as launcher:
                print(
                    json.dumps(
                        {
                            "system_name": "CLIFF",
                            "session_url": launcher.url,
                            "session_outdir_root": str(Path(args.outdir).resolve()),
                            "mode": "interactive_session",
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                while True:
                    submission = launcher.wait_for_next_submission(timeout=0.5)
                    if submission is None:
                        continue
                    run_id, query, execution_mode = submission
                    _start_cliff_session_query(
                        launcher,
                        argparse.Namespace(**dict(vars(args), execution_mode=execution_mode)),
                        run_id=run_id,
                        query=query,
                        active_runs=active_runs,
                        active_runs_lock=active_runs_lock,
                    )
                return
        _open_artifact(_artifact_path_for_result(result))
        print(
            json.dumps(
                {
                    "system_name": "CLIFF",
                    "query": query,
                    "route_decision": asdict(result.route_decision),
                    "route_outdir": str(result.route_outdir),
                    "summary_path": str(result.summary_path),
                },
                indent=2,
            )
        )
    except KeyboardInterrupt:
        print(json.dumps({"system_name": "CLIFF", "status": "session_stopped"}, indent=2), flush=True)
        raise SystemExit(0) from None
    except Exception as exc:
        outdir = Path(args.outdir).resolve()
        message = str(exc).strip() or exc.__class__.__name__
        hints: tuple[str, ...] = ()
        if "SEC retrieval requires an identifying User-Agent" in message:
            hints = (
                "Retry CLIFF with `--sec-retrieval-user-agent 'Your Name your_email@example.com'`.",
                "Or export `FF3_SEC_USER_AGENT='Your Name your_email@example.com'` before running CLIFF.",
                "You can also set `SEC_CONTACT_NAME` and `SEC_CONTACT_EMAIL` instead of a combined user-agent string.",
            )
        error_path = _write_router_error_artifact(
            outdir,
            title="CLIFF could not launch the selected workflow",
            message=message,
            detail=repr(exc),
            hints=hints,
        )
        _open_artifact(error_path)
        print(
            json.dumps(
                {
                    "system_name": "CLIFF",
                    "error": message,
                    "error_artifact_path": str(error_path),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
