"""Top-level NLP router for the inherited FF2 baseline and FF3 transition."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import threading
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

from .basket_rocket_sec_agentic import (
    BasketRocketSECAgenticConfig,
    BasketRocketSECAgenticRunner,
    BasketRocketSECRunResult,
)
from .company_similarity_agentic import (
    CompanySimilarityAgenticRunner,
    CompanySimilarityRunResult,
    looks_like_company_similarity_query,
)
from .course_demo_agentic import (
    CourseDemoAgenticConfig,
    CourseDemoAgenticRunner,
    CourseDemoRunResult,
    looks_like_course_demo_query,
)
from .dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
from .democritus_query_agentic import (
    DemocritusQueryAgenticConfig,
    DemocritusQueryAgenticRunner,
    DemocritusQueryRunResult,
    infer_requested_result_count,
)
from .culinary_tour_agentic import (
    CulinaryTourAgenticRunner,
    CulinaryTourRunResult,
    interpret_culinary_query,
)
from .product_feedback_query_agentic import (
    ProductFeedbackQueryAgenticConfig,
    ProductFeedbackQueryAgenticRunner,
    ProductFeedbackQueryRunResult,
)

_SEC_PATTERNS = (
    r"\b10-k\b",
    r"\b10-q\b",
    r"\b8-k\b",
    r"\bedgar\b",
    r"\bsec\b",
    r"\bfiling\b",
    r"\bfilings\b",
    r"\bannual report\b",
    r"\bquarterly report\b",
)

_PRODUCT_PATTERNS = (
    r"\breview\b",
    r"\breviews\b",
    r"\beasy to drive\b",
    r"\beasy to run\b",
    r"\brun with\b",
    r"\brunning shoe\b",
    r"\brunning shoes\b",
    r"\bshoe\b",
    r"\bshoes\b",
    r"\bdrive\b",
    r"\bdriving\b",
    r"\bcar\b",
    r"\bvehicle\b",
    r"\bsteering\b",
    r"\bhandling\b",
    r"\bseat comfort\b",
    r"\bcomfortable\b",
    r"\bcomfort\b",
    r"\bsofa\b",
    r"\bcouch\b",
    r"\bsectional\b",
    r"\bmattress\b",
    r"\bchair\b",
    r"\breturn risk\b",
    r"\breturns\b",
    r"\bdurability\b",
    r"\bowners say\b",
    r"\bfeedback\b",
)

_CULINARY_PATTERNS = (
    r"\bculinary\b",
    r"\bfood tour\b",
    r"\brestaurant\b",
    r"\brestaurants\b",
    r"\bmeal\b",
    r"\bmeals\b",
    r"\bdining\b",
    r"\bdine\b",
    r"\bitinerary\b",
    r"\btravel\b",
    r"\bfood\b",
)


def _looks_like_culinary_tour_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    if any(re.search(pattern, normalized) for pattern in _CULINARY_PATTERNS):
        return True
    if "tour" not in normalized:
        return False
    plan = interpret_culinary_query(query)
    has_destination = plan.destination != "Destination TBD"
    has_scheduling_constraint = bool(plan.time_window) or plan.budget_per_meal is not None
    return has_destination and has_scheduling_constraint

@dataclass(frozen=True)
class FF2RouteDecision:
    """Router decision for a single natural-language FF2 query."""

    route_name: str
    module_name: str
    rationale: str


@dataclass(frozen=True)
class FF2QueryRouterConfig:
    """Configuration for the top-level FF2 query router."""

    query: str
    outdir: Path
    execution_mode: str = "quick"
    route_override: str = "auto"
    democritus_input_pdf_path: Path | None = None
    democritus_input_pdf_dir: Path | None = None
    democritus_manifest_path: Path | None = None
    democritus_source_pdf_root: Path | None = None
    democritus_target_documents: int = 10
    democritus_base_query: str = ""
    democritus_selected_topics: tuple[str, ...] = ()
    democritus_rejected_topics: tuple[str, ...] = ()
    democritus_retrieval_refinement: str = ""
    democritus_retrieval_backend: str = "auto"
    democritus_max_docs: int = 0
    democritus_intra_document_shards: int = 1
    democritus_manifold_mode: str = "full"
    democritus_topk: int = 200
    democritus_radii: str = "1,2,3"
    democritus_maxnodes: str = "10,20,30,40,60"
    democritus_lambda_edge: float = 0.25
    democritus_topk_models: int = 5
    democritus_topk_claims: int = 30
    democritus_alpha: float = 1.0
    democritus_tier1: float = 0.60
    democritus_tier2: float = 0.30
    democritus_anchors: str = ""
    democritus_title: str = ""
    democritus_dedupe_focus: bool = False
    democritus_require_anchor_in_focus: bool = False
    democritus_focus_blacklist_regex: str = ""
    democritus_render_topk_pngs: bool = False
    democritus_assets_dir: str = "assets"
    democritus_png_dpi: int = 200
    democritus_write_deep_dive: bool = False
    democritus_deep_dive_max_bullets: int = 8
    defer_final_synthesis_to_cliff: bool = False
    democritus_discovery_only: bool = False
    democritus_dry_run: bool = False
    product_manifest_path: Path | None = None
    culinary_manifest_path: Path | None = None
    product_target_documents: int = 5
    product_max_documents: int = 0
    product_name: str = ""
    brand_name: str = ""
    analysis_question: str = ""
    product_discovery_only: bool = False
    sec_target_filings: int = 10
    sec_retrieval_user_agent: str = ""
    company_similarity_year_start: int | None = None
    company_similarity_year_end: int | None = None
    sec_form_types: tuple[str, ...] = ("10-K", "10-Q")
    sec_company_limit: int = 3
    sec_discovery_only: bool = False
    sec_dry_run: bool = False
    course_repo_root: Path | None = None
    course_book_pdf_path: Path | None = None
    course_execute_demo: bool = True
    course_execution_timeout_sec: int = 900

    def resolved(self) -> "FF2QueryRouterConfig":
        normalized_mode = str(self.execution_mode).strip().lower()
        return FF2QueryRouterConfig(
            query=" ".join(self.query.split()),
            outdir=self.outdir.resolve(),
            execution_mode=(
                "deep"
                if normalized_mode == "deep"
                else ("interactive" if normalized_mode == "interactive" else "quick")
            ),
            route_override=self.route_override,
            democritus_input_pdf_path=self.democritus_input_pdf_path.resolve() if self.democritus_input_pdf_path else None,
            democritus_input_pdf_dir=self.democritus_input_pdf_dir.resolve() if self.democritus_input_pdf_dir else None,
            democritus_manifest_path=self.democritus_manifest_path.resolve() if self.democritus_manifest_path else None,
            democritus_source_pdf_root=self.democritus_source_pdf_root.resolve() if self.democritus_source_pdf_root else None,
            democritus_target_documents=self.democritus_target_documents,
            democritus_base_query=" ".join(str(self.democritus_base_query).split()).strip(),
            democritus_selected_topics=tuple(
                " ".join(str(topic).split()).strip()
                for topic in tuple(self.democritus_selected_topics)
                if " ".join(str(topic).split()).strip()
            ),
            democritus_rejected_topics=tuple(
                " ".join(str(topic).split()).strip()
                for topic in tuple(self.democritus_rejected_topics)
                if " ".join(str(topic).split()).strip()
            ),
            democritus_retrieval_refinement=" ".join(str(self.democritus_retrieval_refinement).split()).strip(),
            democritus_retrieval_backend=self.democritus_retrieval_backend,
            democritus_max_docs=self.democritus_max_docs,
            democritus_intra_document_shards=max(1, int(self.democritus_intra_document_shards)),
            democritus_manifold_mode=self.democritus_manifold_mode,
            democritus_topk=max(1, int(self.democritus_topk)),
            democritus_radii=self.democritus_radii,
            democritus_maxnodes=self.democritus_maxnodes,
            democritus_lambda_edge=float(self.democritus_lambda_edge),
            democritus_topk_models=max(1, int(self.democritus_topk_models)),
            democritus_topk_claims=max(1, int(self.democritus_topk_claims)),
            democritus_alpha=float(self.democritus_alpha),
            democritus_tier1=float(self.democritus_tier1),
            democritus_tier2=float(self.democritus_tier2),
            democritus_anchors=self.democritus_anchors,
            democritus_title=self.democritus_title,
            democritus_dedupe_focus=bool(self.democritus_dedupe_focus),
            democritus_require_anchor_in_focus=bool(self.democritus_require_anchor_in_focus),
            democritus_focus_blacklist_regex=self.democritus_focus_blacklist_regex,
            democritus_render_topk_pngs=bool(self.democritus_render_topk_pngs),
            democritus_assets_dir=self.democritus_assets_dir,
            democritus_png_dpi=max(72, int(self.democritus_png_dpi)),
            democritus_write_deep_dive=bool(self.democritus_write_deep_dive),
            democritus_deep_dive_max_bullets=max(1, int(self.democritus_deep_dive_max_bullets)),
            defer_final_synthesis_to_cliff=self.defer_final_synthesis_to_cliff,
            democritus_discovery_only=self.democritus_discovery_only,
            democritus_dry_run=self.democritus_dry_run,
            product_manifest_path=self.product_manifest_path.resolve() if self.product_manifest_path else None,
            culinary_manifest_path=self.culinary_manifest_path.resolve() if self.culinary_manifest_path else None,
            product_target_documents=self.product_target_documents,
            product_max_documents=self.product_max_documents,
            product_name=self.product_name.strip(),
            brand_name=self.brand_name.strip(),
            analysis_question=self.analysis_question.strip(),
            product_discovery_only=self.product_discovery_only,
            sec_target_filings=self.sec_target_filings,
            sec_retrieval_user_agent=" ".join(str(self.sec_retrieval_user_agent).split()).strip(),
            company_similarity_year_start=(
                int(self.company_similarity_year_start)
                if self.company_similarity_year_start is not None
                else None
            ),
            company_similarity_year_end=(
                int(self.company_similarity_year_end)
                if self.company_similarity_year_end is not None
                else None
            ),
            sec_form_types=tuple(self.sec_form_types),
            sec_company_limit=self.sec_company_limit,
            sec_discovery_only=self.sec_discovery_only,
            sec_dry_run=self.sec_dry_run,
            course_repo_root=self.course_repo_root.resolve() if self.course_repo_root else None,
            course_book_pdf_path=self.course_book_pdf_path.resolve() if self.course_book_pdf_path else None,
            course_execute_demo=bool(self.course_execute_demo),
            course_execution_timeout_sec=max(30, int(self.course_execution_timeout_sec)),
        )


@dataclass(frozen=True)
class FF2QueryRouterRunResult:
    """Execution result for the routed FF2 query."""

    route_decision: FF2RouteDecision
    route_outdir: Path
    summary_path: Path
    democritus_result: DemocritusQueryRunResult | None = None
    basket_rocket_sec_result: BasketRocketSECRunResult | None = None
    product_feedback_result: ProductFeedbackQueryRunResult | None = None
    culinary_tour_result: CulinaryTourRunResult | None = None
    company_similarity_result: CompanySimilarityRunResult | None = None
    course_demo_result: CourseDemoRunResult | None = None


def route_ff2_query(query: str, *, route_override: str = "auto") -> FF2RouteDecision:
    """Classify an FF2 query into one of the supported routed entry points."""

    normalized = " ".join(query.lower().split())
    if route_override != "auto":
        return _decision_for_override(route_override)
    if looks_like_company_similarity_query(query):
        return FF2RouteDecision(
            route_name="company_similarity",
            module_name="functorflow_v3.company_similarity_agentic",
            rationale="Query asks for cross-company similarity, so route to temporal diffusion construction and cross-company functor comparison.",
        )
    if any(re.search(pattern, normalized) for pattern in _SEC_PATTERNS):
        return FF2RouteDecision(
            route_name="basket_rocket_sec",
            module_name="functorflow_v3.basket_rocket_sec_agentic",
            rationale="Query mentions SEC or filing-specific language, so route to the SEC-backed BASKET/ROCKET ingress.",
        )
    if _looks_like_culinary_tour_query(query):
        return FF2RouteDecision(
            route_name="culinary_tour",
            module_name="functorflow_v3.culinary_tour_agentic",
            rationale="Query looks like a food, travel, or itinerary planning request, so route to the CLIFF culinary tour orchestrator.",
        )
    if looks_like_course_demo_query(query):
        return FF2RouteDecision(
            route_name="course_demo",
            module_name="functorflow_v3.course_demo_agentic",
            rationale="Query matches a registered Category Theory for AGI course demo, so route to the course notebook launcher.",
        )
    if any(re.search(pattern, normalized) for pattern in _PRODUCT_PATTERNS):
        return FF2RouteDecision(
            route_name="product_feedback",
            module_name="functorflow_v3.product_feedback_query_agentic",
            rationale="Query looks like a consumer product/review question, so route to product-feedback retrieval and analysis.",
        )
    return FF2RouteDecision(
        route_name="democritus",
        module_name="functorflow_v3.democritus_query_agentic",
        rationale="Default route to Democritus for study, paper, corpus, and open-ended evidence acquisition queries.",
    )


def _decision_for_override(route_override: str) -> FF2RouteDecision:
    decisions = {
        "democritus": FF2RouteDecision(
            route_name="democritus",
            module_name="functorflow_v3.democritus_query_agentic",
            rationale="Route override selected the Democritus query runner.",
        ),
        "basket_rocket_sec": FF2RouteDecision(
            route_name="basket_rocket_sec",
            module_name="functorflow_v3.basket_rocket_sec_agentic",
            rationale="Route override selected the SEC-backed BASKET/ROCKET runner.",
        ),
        "culinary_tour": FF2RouteDecision(
            route_name="culinary_tour",
            module_name="functorflow_v3.culinary_tour_agentic",
            rationale="Route override selected the CLIFF culinary tour orchestrator.",
        ),
        "product_feedback": FF2RouteDecision(
            route_name="product_feedback",
            module_name="functorflow_v3.product_feedback_query_agentic",
            rationale="Route override selected the product-feedback query runner.",
        ),
        "course_demo": FF2RouteDecision(
            route_name="course_demo",
            module_name="functorflow_v3.course_demo_agentic",
            rationale="Route override selected the Category Theory for AGI course demo runner.",
        ),
        "company_similarity": FF2RouteDecision(
            route_name="company_similarity",
            module_name="functorflow_v3.company_similarity_agentic",
            rationale="Route override selected the cross-company temporal diffusion similarity runner.",
        ),
    }
    if route_override not in decisions:
        raise ValueError(f"Unsupported route override: {route_override}")
    return decisions[route_override]


class FF2QueryRouter:
    """Run a natural-language query through the correct FF2 entry point."""

    def __init__(self, config: FF2QueryRouterConfig) -> None:
        self.config = config.resolved()
        if not self.config.query:
            raise ValueError("A non-empty FF2 query is required.")
        self.summary_path = self.config.outdir / "ff2_query_router_summary.json"

    def run(self) -> FF2QueryRouterRunResult:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        decision = route_ff2_query(self.config.query, route_override=self.config.route_override)
        if decision.route_name == "basket_rocket_sec":
            result = self._run_basket_rocket_sec(decision)
        elif decision.route_name == "company_similarity":
            result = self._run_company_similarity(decision)
        elif decision.route_name == "culinary_tour":
            result = self._run_culinary_tour(decision)
        elif decision.route_name == "course_demo":
            result = self._run_course_demo(decision)
        elif decision.route_name == "product_feedback":
            result = self._run_product_feedback(decision)
        else:
            result = self._run_democritus(decision)
        self.summary_path.write_text(
            json.dumps(
                {
                    "query": self.config.query,
                    "execution_mode": self.config.execution_mode,
                    "route_decision": asdict(result.route_decision),
                    "route_outdir": str(result.route_outdir),
                    "democritus_summary_path": (
                        str(result.democritus_result.summary_path) if result.democritus_result else None
                    ),
                    "basket_rocket_sec_summary_path": (
                        str(result.basket_rocket_sec_result.summary_path) if result.basket_rocket_sec_result else None
                    ),
                    "product_feedback_summary_path": (
                        str(result.product_feedback_result.summary_path) if result.product_feedback_result else None
                    ),
                    "culinary_tour_summary_path": (
                        str(result.culinary_tour_result.summary_path) if result.culinary_tour_result else None
                    ),
                    "company_similarity_summary_path": (
                        str(result.company_similarity_result.summary_path) if result.company_similarity_result else None
                    ),
                    "course_demo_summary_path": (
                        str(result.course_demo_result.summary_path) if result.course_demo_result else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _run_democritus(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "democritus"
        result = DemocritusQueryAgenticRunner(
            DemocritusQueryAgenticConfig(
                query=self.config.query,
                outdir=route_outdir,
                target_documents=self.config.democritus_target_documents,
                base_query=self.config.democritus_base_query,
                selected_topics=self.config.democritus_selected_topics,
                rejected_topics=self.config.democritus_rejected_topics,
                retrieval_refinement=self.config.democritus_retrieval_refinement,
                input_pdf_path=self.config.democritus_input_pdf_path,
                input_pdf_dir=self.config.democritus_input_pdf_dir,
                manifest_path=self.config.democritus_manifest_path,
                source_pdf_root=self.config.democritus_source_pdf_root,
                retrieval_backend=self.config.democritus_retrieval_backend,
                max_docs=self.config.democritus_max_docs,
                intra_document_shards=self.config.democritus_intra_document_shards,
                manifold_mode=self.config.democritus_manifold_mode,
                topk=self.config.democritus_topk,
                radii=self.config.democritus_radii,
                maxnodes=self.config.democritus_maxnodes,
                lambda_edge=self.config.democritus_lambda_edge,
                topk_models=self.config.democritus_topk_models,
                topk_claims=self.config.democritus_topk_claims,
                alpha=self.config.democritus_alpha,
                tier1=self.config.democritus_tier1,
                tier2=self.config.democritus_tier2,
                anchors=self.config.democritus_anchors,
                title=self.config.democritus_title,
                dedupe_focus=self.config.democritus_dedupe_focus,
                require_anchor_in_focus=self.config.democritus_require_anchor_in_focus,
                focus_blacklist_regex=self.config.democritus_focus_blacklist_regex,
                render_topk_pngs=self.config.democritus_render_topk_pngs,
                assets_dir=self.config.democritus_assets_dir,
                png_dpi=self.config.democritus_png_dpi,
                write_deep_dive=self.config.democritus_write_deep_dive,
                deep_dive_max_bullets=self.config.democritus_deep_dive_max_bullets,
                enable_corpus_synthesis=not self.config.defer_final_synthesis_to_cliff,
                discovery_only=self.config.democritus_discovery_only,
                dry_run=self.config.democritus_dry_run,
                execution_mode=self.config.execution_mode,
            )
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            democritus_result=result,
        )

    def _run_basket_rocket_sec(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "basket_rocket_sec"
        result = BasketRocketSECAgenticRunner(
            BasketRocketSECAgenticConfig(
                query=self.config.query,
                outdir=route_outdir,
                target_filings=self.config.sec_target_filings,
                retrieval_user_agent=self.config.sec_retrieval_user_agent,
                sec_form_types=self.config.sec_form_types,
                sec_company_limit=self.config.sec_company_limit,
                discovery_only=self.config.sec_discovery_only,
                dry_run=self.config.sec_dry_run,
                enable_corpus_synthesis=not self.config.defer_final_synthesis_to_cliff,
            )
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            basket_rocket_sec_result=result,
        )

    def _run_product_feedback(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "product_feedback"
        result = ProductFeedbackQueryAgenticRunner(
            ProductFeedbackQueryAgenticConfig(
                query=self.config.query,
                outdir=route_outdir,
                manifest_path=self.config.product_manifest_path,
                target_documents=self.config.product_target_documents,
                max_documents=self.config.product_max_documents,
                product_name=self.config.product_name,
                brand_name=self.config.brand_name,
                analysis_question=self.config.analysis_question,
                discovery_only=self.config.product_discovery_only,
                enable_corpus_synthesis=not self.config.defer_final_synthesis_to_cliff,
            )
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            product_feedback_result=result,
        )

    def _run_company_similarity(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "company_similarity"
        result = CompanySimilarityAgenticRunner(
            self.config.query,
            route_outdir,
            sec_user_agent=self.config.sec_retrieval_user_agent,
            execution_mode=self.config.execution_mode,
            year_start=self.config.company_similarity_year_start,
            year_end=self.config.company_similarity_year_end,
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            company_similarity_result=result,
        )

    def _run_culinary_tour(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "culinary_tour"
        result = CulinaryTourAgenticRunner(
            self.config.query,
            route_outdir,
            manifest_path=self.config.culinary_manifest_path,
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            culinary_tour_result=result,
        )

    def _run_course_demo(self, decision: FF2RouteDecision) -> FF2QueryRouterRunResult:
        route_outdir = self.config.outdir / "course_demo"
        result = CourseDemoAgenticRunner(
            CourseDemoAgenticConfig(
                query=self.config.query,
                outdir=route_outdir,
                course_repo_root=self.config.course_repo_root,
                book_pdf_path=self.config.course_book_pdf_path,
                execute_demo=self.config.course_execute_demo,
                execution_timeout_sec=self.config.course_execution_timeout_sec,
            )
        ).run()
        return FF2QueryRouterRunResult(
            route_decision=decision,
            route_outdir=route_outdir,
            summary_path=self.summary_path,
            course_demo_result=result,
        )


def _resolve_query_for_main(args: argparse.Namespace, *, artifact_path: Path | None = None) -> str:
    query = " ".join(str(args.query or "").split()).strip()
    if query:
        return query
    with DashboardQueryLauncher(
        DashboardQueryLauncherConfig(
            title="BAFFLE Query Router",
            subtitle=(
                "Enter one natural-language request and BAFFLE will choose the right runner. "
                "Document requests trigger retrieval plus analysis: studies and papers go to Democritus for "
                "per-document analysis and corpus synthesis, SEC filing requests go to BASKET/ROCKET SEC for "
                "workflow extraction, company-to-company similarity requests go to the temporal diffusion "
                "comparison runner, and product/review questions go to product feedback."
            ),
            query_label="BAFFLE query",
            query_placeholder=(
                "Analyze 10 recent Adobe 10-K filings and extract their workflows\n"
                "or\n"
                "How similar is Adobe to Nike?\n"
                "or\n"
                "Analyze 10 recent studies on red wine and synthesize what they jointly support\n"
                "or\n"
                "How comfortable is the Lovesac sectional sofa?"
            ),
            submit_label="Route Query",
            waiting_message="BAFFLE is routing the query, retrieving the needed documents, and launching the selected analysis workflow.",
            artifact_path=artifact_path,
        )
    ) as launcher:
        return launcher.wait_for_submission()


def _artifact_path_for_result(result: FF2QueryRouterRunResult) -> Path | None:
    if result.product_feedback_result:
        if result.product_feedback_result.corpus_synthesis_result:
            return result.product_feedback_result.corpus_synthesis_result.dashboard_path
        if result.product_feedback_result.product_feedback_result:
            return result.product_feedback_result.product_feedback_result.dashboard_path
    if result.company_similarity_result:
        return result.company_similarity_result.artifact_path
    if result.culinary_tour_result:
        return result.culinary_tour_result.dashboard_path
    if result.course_demo_result:
        return result.course_demo_result.dashboard_path
    if result.democritus_result:
        clarification_candidate = getattr(result.democritus_result, "clarification_dashboard_path", None)
        if clarification_candidate and clarification_candidate.exists():
            return clarification_candidate
        checkpoint_candidate = getattr(result.democritus_result, "checkpoint_dashboard_path", None)
        if checkpoint_candidate and checkpoint_candidate.exists():
            return checkpoint_candidate
        corpus_candidate = getattr(result.democritus_result, "corpus_synthesis_dashboard_path", None)
        if corpus_candidate and corpus_candidate.exists():
            return corpus_candidate
        gui_candidate = getattr(result.democritus_result, "batch_outdir", None)
        gui_candidate = gui_candidate / "democritus_gui.html" if gui_candidate else None
        if gui_candidate and gui_candidate.exists():
            return gui_candidate
        candidate_root = getattr(result.democritus_result, "batch_outdir", None)
        candidate = candidate_root / "dashboard.html" if candidate_root else None
        return candidate if candidate and candidate.exists() else None
    if result.basket_rocket_sec_result:
        if (
            result.basket_rocket_sec_result.corpus_synthesis_dashboard_path
            and result.basket_rocket_sec_result.corpus_synthesis_dashboard_path.exists()
        ):
            return result.basket_rocket_sec_result.corpus_synthesis_dashboard_path
        if result.basket_rocket_sec_result.batch_live_gui_path and result.basket_rocket_sec_result.batch_live_gui_path.exists():
            return result.basket_rocket_sec_result.batch_live_gui_path
        if result.basket_rocket_sec_result.batch_visualization_index_path:
            return result.basket_rocket_sec_result.batch_visualization_index_path
    return None


def _open_artifact(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=1, autoraise=True)
    except Exception:
        pass


def _router_launcher_artifact_path(outdir: Path) -> Path:
    return outdir.resolve() / "selected_route_artifact.html"


def _write_router_error_artifact(
    outdir: Path,
    *,
    title: str,
    message: str,
    detail: str = "",
    hints: tuple[str, ...] = (),
) -> Path:
    path = outdir.resolve() / "router_error.html"
    path.parent.mkdir(parents=True, exist_ok=True)

    def esc(value: object) -> str:
        return html.escape(str(value))

    hint_markup = "".join(f"<li>{esc(item)}</li>" for item in hints)
    detail_markup = f"<pre>{esc(detail)}</pre>" if detail else ""
    path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)}</title>
    <style>
      :root {{
        --ink: #1f2320;
        --muted: #5a6661;
        --paper: #f7efe1;
        --card: rgba(255, 252, 246, 0.97);
        --line: #d6c4a7;
        --warn: #9a3f12;
        --warn-soft: #f7e3d5;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top, rgba(154,63,18,0.10), transparent 24%),
          linear-gradient(180deg, #fbf4e8 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 980px;
        margin: 44px auto;
        padding: 0 18px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 20px 54px rgba(44, 30, 11, 0.10);
      }}
      .eyebrow {{
        margin: 0 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--warn);
      }}
      h1 {{
        margin: 0 0 12px 0;
        font-size: clamp(30px, 4vw, 50px);
        line-height: 1.0;
      }}
      p, li {{
        color: var(--muted);
        line-height: 1.6;
        font-size: 16px;
      }}
      ul {{
        margin: 16px 0 0 18px;
        padding: 0;
      }}
      pre {{
        margin: 20px 0 0;
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: var(--warn-soft);
        white-space: pre-wrap;
        word-break: break-word;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
      code {{
        background: rgba(0,0,0,0.04);
        padding: 2px 5px;
        border-radius: 6px;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">BAFFLE Launch Error</p>
        <h1>{esc(title)}</h1>
        <p>{esc(message)}</p>
        {"<ul>" + hint_markup + "</ul>" if hint_markup else ""}
        {detail_markup}
      </section>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return path


def _predicted_artifact_path(outdir: Path, decision: FF2RouteDecision) -> Path | None:
    resolved_outdir = outdir.resolve()
    if decision.route_name == "democritus":
        # The live session preview should point at the batch GUI because it exists
        # throughout the run; the corpus synthesis artifact is selected after completion.
        return resolved_outdir / "democritus" / "democritus_runs" / "democritus_gui.html"
    if decision.route_name == "product_feedback":
        return resolved_outdir / "product_feedback" / "product_feedback_run" / "product_feedback_dashboard.html"
    if decision.route_name == "culinary_tour":
        return resolved_outdir / "culinary_tour" / "culinary_tour_dashboard.html"
    if decision.route_name == "basket_rocket_sec":
        return resolved_outdir / "basket_rocket_sec" / "workflow_batches" / "basket_rocket_gui.html"
    if decision.route_name == "company_similarity":
        return resolved_outdir / "company_similarity" / "company_similarity_dashboard.html"
    if decision.route_name == "course_demo":
        return resolved_outdir / "course_demo" / "course_demo_dashboard.html"
    return None


def _materialize_router_artifact(source_path: Path | None, target_path: Path | None) -> None:
    if source_path is None or target_path is None or not source_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _build_router_from_args(args: argparse.Namespace, *, query: str) -> FF2QueryRouter:
    return _build_router_from_args_with_outdir(args, query=query, outdir=Path(args.outdir))


def _build_router_from_args_with_outdir(
    args: argparse.Namespace,
    *,
    query: str,
    outdir: Path,
) -> FF2QueryRouter:
    democritus_inferred_target = infer_requested_result_count(
        query,
        nouns=("study", "studies", "paper", "papers", "article", "articles", "document", "documents"),
    )
    product_inferred_target = infer_requested_result_count(
        query,
        nouns=("review", "reviews", "document", "documents"),
    )
    sec_inferred_target = infer_requested_result_count(
        query,
        nouns=("filing", "filings", "report", "reports", "document", "documents"),
    )
    democritus_target_documents = (
        int(args.democritus_target_docs)
        if args.democritus_target_docs is not None
        else (democritus_inferred_target or 10)
    )
    if args.democritus_max_docs is not None:
        democritus_max_docs = int(args.democritus_max_docs)
    else:
        democritus_default_target = democritus_inferred_target or 10
        has_fixed_democritus_corpus = bool(
            getattr(args, "democritus_input_pdf", "")
            or getattr(args, "democritus_input_pdf_dir", "")
            or getattr(args, "democritus_manifest", "")
            or getattr(args, "democritus_source_pdf_root", "")
        )
        backend_name = str(getattr(args, "democritus_retrieval_backend", "auto") or "auto")
        if has_fixed_democritus_corpus:
            democritus_max_docs = democritus_inferred_target or 0
        elif backend_name in {"auto", "scholarly", "crossref", "semantic_scholar", "europe_pmc"}:
            democritus_max_docs = min(max(democritus_default_target * 3, democritus_default_target + 10), 100)
        else:
            democritus_max_docs = democritus_inferred_target or 0
    product_target_documents = (
        int(args.product_target_docs)
        if args.product_target_docs is not None
        else (product_inferred_target or 5)
    )
    product_max_documents = (
        int(args.product_max_docs)
        if args.product_max_docs is not None
        else (product_inferred_target or 0)
    )
    sec_target_filings = (
        int(args.sec_target_filings)
        if args.sec_target_filings is not None
        else (sec_inferred_target or 10)
    )
    return FF2QueryRouter(
        FF2QueryRouterConfig(
            query=query,
            outdir=Path(outdir),
            execution_mode=getattr(args, "execution_mode", "quick"),
            route_override=args.route,
            democritus_input_pdf_path=(
                Path(getattr(args, "democritus_input_pdf", ""))
                if getattr(args, "democritus_input_pdf", "")
                else None
            ),
            democritus_input_pdf_dir=(
                Path(getattr(args, "democritus_input_pdf_dir", ""))
                if getattr(args, "democritus_input_pdf_dir", "")
                else None
            ),
            democritus_manifest_path=Path(args.democritus_manifest) if args.democritus_manifest else None,
            democritus_source_pdf_root=Path(args.democritus_source_pdf_root) if args.democritus_source_pdf_root else None,
            democritus_target_documents=democritus_target_documents,
            democritus_base_query=getattr(args, "democritus_base_query", ""),
            democritus_selected_topics=tuple(getattr(args, "democritus_selected_topics", ()) or ()),
            democritus_rejected_topics=tuple(getattr(args, "democritus_rejected_topics", ()) or ()),
            democritus_retrieval_refinement=getattr(args, "democritus_retrieval_refinement", ""),
            democritus_retrieval_backend=args.democritus_retrieval_backend,
            democritus_max_docs=democritus_max_docs,
            democritus_intra_document_shards=args.democritus_intra_document_shards,
            democritus_manifold_mode=getattr(args, "democritus_manifold_mode", "full"),
            democritus_topk=getattr(args, "democritus_topk", 200),
            democritus_radii=getattr(args, "democritus_radii", "1,2,3"),
            democritus_maxnodes=getattr(args, "democritus_maxnodes", "10,20,30,40,60"),
            democritus_lambda_edge=getattr(args, "democritus_lambda_edge", 0.25),
            democritus_topk_models=getattr(args, "democritus_topk_models", 5),
            democritus_topk_claims=getattr(args, "democritus_topk_claims", 30),
            democritus_alpha=getattr(args, "democritus_alpha", 1.0),
            democritus_tier1=getattr(args, "democritus_tier1", 0.60),
            democritus_tier2=getattr(args, "democritus_tier2", 0.30),
            democritus_anchors=getattr(args, "democritus_anchors", ""),
            democritus_title=getattr(args, "democritus_title", ""),
            democritus_dedupe_focus=bool(getattr(args, "democritus_dedupe_focus", False)),
            democritus_require_anchor_in_focus=bool(getattr(args, "democritus_require_anchor_in_focus", False)),
            democritus_focus_blacklist_regex=getattr(args, "democritus_focus_blacklist_regex", ""),
            democritus_render_topk_pngs=bool(getattr(args, "democritus_render_topk_pngs", False)),
            democritus_assets_dir=getattr(args, "democritus_assets_dir", "assets"),
            democritus_png_dpi=getattr(args, "democritus_png_dpi", 200),
            democritus_write_deep_dive=bool(getattr(args, "democritus_write_deep_dive", False)),
            democritus_deep_dive_max_bullets=getattr(args, "democritus_deep_dive_max_bullets", 8),
            defer_final_synthesis_to_cliff=bool(getattr(args, "cliff_defer_final_synthesis", False)),
            democritus_discovery_only=args.democritus_discovery_only,
            democritus_dry_run=args.democritus_dry_run,
            product_manifest_path=Path(args.product_manifest) if args.product_manifest else None,
            culinary_manifest_path=Path(getattr(args, "culinary_manifest", "")) if getattr(args, "culinary_manifest", "") else None,
            product_target_documents=product_target_documents,
            product_max_documents=product_max_documents,
            product_name=args.product_name,
            brand_name=args.brand_name,
            analysis_question=args.analysis_question,
            product_discovery_only=args.product_discovery_only,
            sec_target_filings=sec_target_filings,
            sec_retrieval_user_agent=args.sec_retrieval_user_agent,
            company_similarity_year_start=getattr(args, "company_similarity_year_start", None),
            company_similarity_year_end=getattr(args, "company_similarity_year_end", None),
            sec_form_types=tuple(args.sec_form) if args.sec_form else ("10-K", "10-Q"),
            sec_company_limit=args.sec_company_limit,
            sec_discovery_only=args.sec_discovery_only,
            sec_dry_run=args.sec_dry_run,
            course_repo_root=Path(getattr(args, "course_repo_root", "")) if getattr(args, "course_repo_root", "") else None,
            course_book_pdf_path=Path(getattr(args, "course_book_pdf_path", "")) if getattr(args, "course_book_pdf_path", "") else None,
            course_execute_demo=not bool(getattr(args, "course_no_execute", False)),
            course_execution_timeout_sec=int(getattr(args, "course_timeout_sec", 900)),
        )
    )


def _slugify_query(query: str, *, maxlen: int = 48) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return cleaned[:maxlen] or "query"


def _session_query_outdir(session_outdir: Path, *, run_id: str, query: str) -> Path:
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    root = session_outdir.expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root.parent / f"{root.name}-{run_id}-{timestamp}-{_slugify_query(query)}"


def _run_session_query(
    launcher: DashboardQueryLauncher,
    args: argparse.Namespace,
    *,
    run_id: str,
    query: str,
) -> None:
    run_outdir = _session_query_outdir(Path(args.outdir), run_id=run_id, query=query)
    decision = route_ff2_query(query, route_override=args.route)
    launcher.update_session_run(
        run_id,
        status="routing",
        route_name=decision.route_name,
        note=f"Routing this request to {decision.route_name}.",
        outdir=run_outdir,
    )
    try:
        launcher.update_session_run(
            run_id,
            status="running",
            route_name=decision.route_name,
            note=f"Running {decision.route_name} in the background.",
            outdir=run_outdir,
        )
        result = _build_router_from_args_with_outdir(args, query=query, outdir=run_outdir).run()
        artifact_path = _artifact_path_for_result(result)
        launcher.update_session_run(
            run_id,
            status="complete",
            route_name=decision.route_name,
            note="Run complete. BAFFLE opened the result window.",
            artifact_path=artifact_path,
            outdir=run_outdir,
        )
        _open_artifact(artifact_path)
        print(
            json.dumps(
                {
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
                "Or export `FF3_SEC_USER_AGENT='Your Name your_email@example.com'` before running BAFFLE.",
            )
        error_path = _write_router_error_artifact(
            run_outdir,
            title="BAFFLE could not launch the selected workflow",
            message=message,
            detail=repr(exc),
            hints=hints,
        )
        launcher.update_session_run(
            run_id,
            status="failed",
            route_name=decision.route_name,
            note=message,
            artifact_path=error_path,
            outdir=run_outdir,
        )
        _open_artifact(error_path)
        print(
            json.dumps(
                {
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
    parser = argparse.ArgumentParser(description="Route one natural-language query to the appropriate FF2 entry point.")
    parser.add_argument(
        "--query",
        default="",
        help="Natural-language FF2 request. If omitted, a local dashboard will open to collect it.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--execution-mode", choices=("quick", "interactive", "deep"), default="quick")
    parser.add_argument("--route", choices=("auto", "democritus", "basket_rocket_sec", "culinary_tour", "product_feedback", "company_similarity", "course_demo"), default="auto")
    parser.add_argument("--democritus-manifest", default="")
    parser.add_argument("--democritus-source-pdf-root", default="")
    parser.add_argument("--democritus-target-docs", type=int, default=None)
    parser.add_argument("--democritus-retrieval-backend", default="auto")
    parser.add_argument("--democritus-max-docs", type=int, default=None)
    parser.add_argument("--democritus-intra-document-shards", type=int, default=1)
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
    parser.add_argument("--company-similarity-year-start", type=int, default=None)
    parser.add_argument("--company-similarity-year-end", type=int, default=None)
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
    try:
        inline_query = " ".join(str(args.query or "").split()).strip()
        if inline_query:
            query = inline_query
            result = _build_router_from_args(args, query=query).run()
        else:
            launcher_artifact_path = _router_launcher_artifact_path(Path(args.outdir))
            with DashboardQueryLauncher(
                DashboardQueryLauncherConfig(
                    title="BAFFLE Query Router",
                    subtitle=(
                        "Enter repeated natural-language requests and BAFFLE will route each one into its own "
                        "background run. Document requests trigger retrieval plus analysis: studies and papers "
                        "go to Democritus, SEC filing requests go to BASKET/ROCKET SEC for workflow extraction, "
                        "and product/review questions go to product feedback."
                    ),
                    query_label="BAFFLE query",
                    query_placeholder=(
                        "Analyze 10 recent Adobe 10-K filings and extract their workflows\n"
                        "or\n"
                        "Analyze 10 recent studies on red wine and synthesize what they jointly support\n"
                        "or\n"
                        "How comfortable is the Lovesac sectional sofa?"
                    ),
                    submit_label="Route Query",
                    waiting_message="BAFFLE keeps this session window open while each query retrieves its documents and runs its analysis in a separate output directory.",
                    artifact_path=launcher_artifact_path,
                    session_mode=True,
                    enable_execution_mode=True,
                )
            ) as launcher:
                print(
                    json.dumps(
                        {
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
                    threading.Thread(
                        target=_run_session_query,
                        args=(launcher, argparse.Namespace(**dict(vars(args), execution_mode=execution_mode))),
                        kwargs={"run_id": run_id, "query": query},
                        name=f"ff2-session-{run_id}",
                        daemon=True,
                    ).start()
                return
        _open_artifact(_artifact_path_for_result(result))
        print(
            json.dumps(
                {
                    "query": query,
                    "route_decision": asdict(result.route_decision),
                    "route_outdir": str(result.route_outdir),
                    "summary_path": str(result.summary_path),
                },
                indent=2,
            )
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "session_stopped"}, indent=2), flush=True)
        raise SystemExit(0) from None
    except Exception as exc:
        outdir = Path(args.outdir).resolve()
        message = str(exc).strip() or exc.__class__.__name__
        hints: tuple[str, ...] = ()
        if "SEC retrieval requires an identifying User-Agent" in message:
            hints = (
                "Retry the router with `--sec-retrieval-user-agent 'Your Name your_email@example.com'`.",
                "Or export `FF3_SEC_USER_AGENT='Your Name your_email@example.com'` before running BAFFLE.",
                "You can also set `SEC_CONTACT_NAME` and `SEC_CONTACT_EMAIL` instead of a combined user-agent string.",
            )
        error_path = _write_router_error_artifact(
            outdir,
            title="BAFFLE could not launch the selected workflow",
            message=message,
            detail=repr(exc),
            hints=hints,
        )
        _open_artifact(error_path)
        print(json.dumps({"error": message, "error_artifact_path": str(error_path)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()


FF3RouteDecision = FF2RouteDecision
FF3QueryRouterConfig = FF2QueryRouterConfig
FF3QueryRouterRunResult = FF2QueryRouterRunResult
FF3QueryRouter = FF2QueryRouter


def route_ff3_query(query: str, *, route_override: str = "auto") -> FF3RouteDecision:
    """Forward-looking FF3 alias for the inherited query router."""

    return route_ff2_query(query, route_override=route_override)
