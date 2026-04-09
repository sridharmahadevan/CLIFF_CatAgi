"""CLIFF worker process for one unconscious query run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .cliff import (
    _WORKER_RESULT_FILENAME,
    _build_cliff_synthesis_from_first_pass,
    _decision_supports_conscious_redispatch,
    route_cliff_query,
)
from .query_router_agentic import (
    _artifact_path_for_result,
    _build_router_from_args_with_outdir,
    _write_router_error_artifact,
)


def _worker_result_path(run_outdir: Path) -> Path:
    return run_outdir / _WORKER_RESULT_FILENAME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one unconscious CLIFF query worker.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cycle-stage", choices=("first_pass", "synthesis_pass"), default="first_pass")
    parser.add_argument("--execution-mode", choices=("quick", "deep"), default="quick")
    parser.add_argument("--cliff-defer-final-synthesis", action="store_true")
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
    parser.add_argument("--sec-retrieval-user-agent", default="")
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
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    decision = route_cliff_query(args.query, route_override=args.route)
    result_path = _worker_result_path(outdir)
    try:
        if args.cycle_stage == "synthesis_pass":
            artifact_path = _build_cliff_synthesis_from_first_pass(
                query=args.query,
                decision=decision,
                run_outdir=outdir,
            )
            payload = {
                "status": "complete",
                "system_name": "CLIFF",
                "query": args.query,
                "route_decision": asdict(decision),
                "route_outdir": str((outdir / decision.route_name).resolve()),
                "summary_path": str(outdir / "ff2_query_router_summary.json"),
                "artifact_path": str(artifact_path) if artifact_path else None,
            }
        else:
            result = _build_router_from_args_with_outdir(args, query=args.query, outdir=outdir).run()
            artifact_path = _artifact_path_for_result(result)
            payload = {
                "status": (
                    "phase1_complete"
                    if args.cliff_defer_final_synthesis and _decision_supports_conscious_redispatch(result.route_decision)
                    else "complete"
                ),
                "system_name": "CLIFF",
                "query": args.query,
                "route_decision": asdict(result.route_decision),
                "route_outdir": str(result.route_outdir),
                "summary_path": str(result.summary_path),
                "artifact_path": str(artifact_path) if artifact_path else None,
            }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        hints: tuple[str, ...] = ()
        if "SEC retrieval requires an identifying User-Agent" in message:
            hints = (
                "Retry CLIFF with `--sec-retrieval-user-agent 'Your Name your_email@example.com'`.",
                "Or export `FF3_SEC_USER_AGENT='Your Name your_email@example.com'` before running CLIFF.",
            )
        error_path = _write_router_error_artifact(
            outdir,
            title="CLIFF unconscious worker could not complete the selected workflow",
            message=message,
            detail=repr(exc),
            hints=hints,
        )
        payload = {
            "status": "failed",
            "system_name": "CLIFF",
            "query": args.query,
            "route_decision": asdict(decision),
            "error": message,
            "error_artifact_path": str(error_path),
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
