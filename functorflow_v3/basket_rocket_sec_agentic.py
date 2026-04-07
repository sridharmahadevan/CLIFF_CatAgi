"""SEC-backed ingress and batch scaffolding for agentic BASKET/ROCKET."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .basket_rocket_corpus_synthesis import (
    BasketRocketCorpusSynthesisResult,
    build_basket_rocket_corpus_synthesis,
)
from .basket_rocket_visualizations import BasketRocketVisualizationResult, generate_basket_rocket_visualizations
from .blueprints import build_basket_rocket_workflow
from .dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
from .democritus_query_agentic import (
    DemocritusQueryAgenticConfig,
    DemocritusQueryAgenticRunner,
    DemocritusQueryRunResult,
    DiscoveredDocument,
    QueryPlan,
    _resolve_sec_user_agent,
)
from .repo_layout import resolve_basket_root, resolve_brand_panel_root


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASKET_ROOT = resolve_basket_root()
_BRAND_PANEL_ROOT = resolve_brand_panel_root()
_DEFAULT_LEGACY_EXTRACTIONS_PATH = (
    _BASKET_ROOT / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "workflow_extractions.jsonl.gz"
    if (_BASKET_ROOT / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "workflow_extractions.jsonl.gz").exists()
    else _BASKET_ROOT / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "workflow_extractions.jsonl"
)
_DEFAULT_LEGACY_MACRO_SKILLS_PATH = (
    _BASKET_ROOT / "outputs" / "tenk_rawpdf_fullpanel_monitored" / "macro_skills.csv"
)
_DEFAULT_LEGACY_PANEL_PATH = (
    _BRAND_PANEL_ROOT / "outputs" / "company_panel_26" / "rocket_company_outcomes.csv"
)
_DEFAULT_LEGACY_KET_MODEL_PATH = (
    _BASKET_ROOT / "outputs" / "rocket_repair_policy_torch_closed_loop_v1_small" / "summary.json"
)


def _slugify(name: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "filing"


def _form_slug(value: str) -> str:
    return _slugify(value.replace("/", "-"))


def _safe_year(value: str) -> str:
    year = str(value).split("-", 1)[0].strip()
    return year or "unknown_year"


def _form_semantic_role(form_type: str) -> str:
    normalized = str(form_type).upper()
    if normalized == "10-K":
        return "annual_anchor"
    if normalized == "8-K":
        return "event_patch"
    if normalized == "10-Q":
        return "quarterly_update"
    return "filing_update"


def _extract_sec_item_codes(text: str) -> tuple[str, ...]:
    matches = re.findall(r"\bitem\s+(\d+\.\d+)\b", text, flags=re.IGNORECASE)
    unique = []
    seen = set()
    for match in matches:
        code = match.strip()
        if code and code not in seen:
            seen.add(code)
            unique.append(code)
    return tuple(unique)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_json(path: Path) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    rows: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


_LEGACY_ACTION_ORDER = (
    "invest",
    "innovate",
    "digitize",
    "optimize",
    "expand",
    "distribute",
    "market",
    "price",
    "realize_revenue",
)

_ACTION_KEYWORDS = {
    "invest": ("investment", "capital", "acquisition", "research", "development", "agreement"),
    "innovate": ("innovation", "innov", "product", "research", "development", "technology"),
    "digitize": ("digital", "cloud", "software", "platform", "data", "ai", "automation"),
    "optimize": ("risk", "management", "discussion", "operations", "efficiency", "controls", "restructuring"),
    "expand": ("growth", "strategy", "international", "segment", "business", "geographic", "acquisition"),
    "distribute": ("distribution", "channel", "logistics", "supply", "fulfillment", "partner"),
    "market": ("marketing", "sales", "customer", "brand", "demand"),
    "price": ("pricing", "price", "margin", "financial", "contract", "subscription", "fee"),
    "realize_revenue": ("revenue", "results", "sales", "financial", "earnings"),
}

_ACTION_ITEM_PREFIXES = {
    "invest": ("1",),
    "expand": ("1",),
    "price": ("2",),
    "realize_revenue": ("2",),
    "optimize": ("5",),
    "market": ("8",),
}

_ACTION_PRIORITY = {action: index for index, action in enumerate(_LEGACY_ACTION_ORDER)}

_MACRO_TEMPLATES = {
    "annual_anchor": (
        ("digitize", "optimize", "market", "realize_revenue"),
        ("expand", "market", "price", "realize_revenue"),
        ("innovate", "optimize", "market", "realize_revenue"),
    ),
    "event_patch": (
        ("invest", "expand", "price", "realize_revenue"),
        ("optimize", "market", "price", "realize_revenue"),
        ("invest", "optimize", "realize_revenue"),
    ),
    "quarterly_update": (
        ("digitize", "optimize", "price", "realize_revenue"),
        ("expand", "market", "realize_revenue"),
    ),
    "filing_update": (
        ("optimize", "market", "realize_revenue"),
    ),
}


def _append_unique(actions: list[str], *new_actions: str) -> None:
    for action in new_actions:
        value = str(action).strip()
        if value and value not in actions:
            actions.append(value)


def _ordered_actions(actions: list[str]) -> list[str]:
    seen = set(actions)
    ordered = [action for action in _LEGACY_ACTION_ORDER if action in seen]
    return ordered if ordered else ["optimize", "realize_revenue"]


def _linear_edges(actions: list[str]) -> list[tuple[str, str]]:
    return [(actions[index], actions[index + 1]) for index in range(len(actions) - 1)]


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ordered_unique(actions: list[str]) -> list[str]:
    return _ordered_actions(list(dict.fromkeys(str(action).strip() for action in actions if str(action).strip())))


def _insert_action(actions: list[str], action: str, *, before: tuple[str, ...] = ("realize_revenue",)) -> list[str]:
    if action in actions:
        return list(actions)
    updated = list(actions)
    for index, existing in enumerate(updated):
        if existing in before:
            updated.insert(index, action)
            return _ordered_unique(updated)
    updated.append(action)
    return _ordered_unique(updated)


def _normalized_name_token(value: str) -> str:
    token = _slugify(value).replace("_", "")
    for suffix in ("inc", "corp", "corporation", "company", "co", "ltd", "plc", "holdings", "group"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
    return token


@dataclass(frozen=True)
class MaterializedSECFiling:
    """A filing staged locally for downstream BASKET/ROCKET workflows."""

    title: str
    filing_path: str
    text_path: str | None
    source_url: str
    retrieval_backend: str
    company: str
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    filing_year: str
    anchor_year: str
    semantic_role: str
    workset_name: str
    event_item_codes: tuple[str, ...] = ()
    status: str = "materialized"


@dataclass(frozen=True)
class BasketRocketBatchRecord:
    """Execution record for one batch-stage agent on one workset."""

    workset_name: str
    company: str
    filing_year: str
    form_type: str
    agent_name: str
    frontier_index: int
    status: str
    started_at: float
    ended_at: float
    outputs: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class BasketRocketBatchRunResult:
    """Output bundle for the grouped BASKET/ROCKET batch scaffold."""

    records: tuple[BasketRocketBatchRecord, ...]
    workset_index_path: Path
    summary_path: Path
    live_gui_path: Path
    visualizations: BasketRocketVisualizationResult | None = None
    corpus_synthesis: BasketRocketCorpusSynthesisResult | None = None


@dataclass(frozen=True)
class BasketRocketSECRunResult:
    """Result bundle for SEC-backed BASKET/ROCKET ingress."""

    query_plan: QueryPlan
    selected_filings: tuple[DiscoveredDocument, ...]
    materialized_filings: tuple[MaterializedSECFiling, ...]
    batch_records: tuple[BasketRocketBatchRecord, ...]
    discovery_summary_path: Path
    filing_manifest_path: Path | None
    company_context_path: Path | None
    batch_workset_index_path: Path | None
    batch_summary_path: Path | None
    batch_live_gui_path: Path | None
    batch_visualization_index_path: Path | None
    batch_visualization_summary_path: Path | None
    summary_path: Path
    corpus_synthesis_summary_path: Path | None = None
    corpus_synthesis_dashboard_path: Path | None = None


@dataclass(frozen=True)
class BasketRocketSECAgenticConfig:
    """Configuration for the SEC-backed BASKET/ROCKET ingress scaffold."""

    query: str
    outdir: Path
    target_filings: int = 10
    retrieval_user_agent: str = "FunctorFlow_v2/0.1 (agentic SEC retrieval; local use)"
    retrieval_timeout_seconds: float = 20.0
    sec_form_types: tuple[str, ...] = ("10-K", "10-Q")
    sec_company_limit: int = 3
    rocket_reward_source: str = "legacy"
    rocket_legacy_extractions_path: Path = _DEFAULT_LEGACY_EXTRACTIONS_PATH
    rocket_legacy_macro_skills_path: Path = _DEFAULT_LEGACY_MACRO_SKILLS_PATH
    rocket_panel_path: Path = _DEFAULT_LEGACY_PANEL_PATH
    rocket_financial_targets: tuple[str, ...] = (
        "+revenue_yoy",
        "+operating_margin",
        "+free_cash_flow_margin",
        "+return_on_assets",
        "-debt_to_assets",
    )
    rocket_financial_horizon: str = "next_year"
    rocket_ket_model_path: Path | None = None
    rocket_ket_skill_basis_path: Path | None = None
    rocket_ket_action_vocab_path: Path | None = None
    rocket_ket_device: str = "cpu"
    rocket_ket_weight: float = 0.20
    discovery_only: bool = False
    dry_run: bool = False
    enable_corpus_synthesis: bool = True

    def resolved(self) -> "BasketRocketSECAgenticConfig":
        return BasketRocketSECAgenticConfig(
            query=" ".join(self.query.split()),
            outdir=self.outdir.resolve(),
            target_filings=max(1, self.target_filings),
            retrieval_user_agent=" ".join(str(self.retrieval_user_agent).split()).strip()
            or "FunctorFlow_v2/0.1 (agentic SEC retrieval; local use)",
            retrieval_timeout_seconds=self.retrieval_timeout_seconds,
            sec_form_types=tuple(self.sec_form_types),
            sec_company_limit=max(1, self.sec_company_limit),
            rocket_reward_source=str(self.rocket_reward_source or "legacy").strip().lower(),
            rocket_legacy_extractions_path=Path(self.rocket_legacy_extractions_path).expanduser().resolve(),
            rocket_legacy_macro_skills_path=Path(self.rocket_legacy_macro_skills_path).expanduser().resolve(),
            rocket_panel_path=Path(self.rocket_panel_path).expanduser().resolve(),
            rocket_financial_targets=tuple(str(item).strip() for item in self.rocket_financial_targets if str(item).strip()),
            rocket_financial_horizon=str(self.rocket_financial_horizon or "next_year").strip(),
            rocket_ket_model_path=(Path(self.rocket_ket_model_path).expanduser().resolve() if self.rocket_ket_model_path else None),
            rocket_ket_skill_basis_path=(
                Path(self.rocket_ket_skill_basis_path).expanduser().resolve() if self.rocket_ket_skill_basis_path else None
            ),
            rocket_ket_action_vocab_path=(
                Path(self.rocket_ket_action_vocab_path).expanduser().resolve() if self.rocket_ket_action_vocab_path else None
            ),
            rocket_ket_device=str(self.rocket_ket_device or "cpu").strip(),
            rocket_ket_weight=float(max(0.0, self.rocket_ket_weight)),
            discovery_only=self.discovery_only,
            dry_run=self.dry_run,
            enable_corpus_synthesis=self.enable_corpus_synthesis,
        )


@dataclass(frozen=True)
class _BasketRocketWorkset:
    """Grouped company/year/form work unit."""

    workset_name: str
    company: str
    filing_year: str
    form_type: str
    semantic_role: str
    filings: tuple[MaterializedSECFiling, ...]
    outdir: Path


class BasketRocketBatchAgenticRunner:
    """Batch scaffold over materialized SEC filings grouped by company/year/form."""

    def __init__(
        self,
        *,
        filings: tuple[MaterializedSECFiling, ...],
        outdir: Path,
        filing_manifest_path: Path,
        company_context_path: Path,
        rocket_reward_source: str = "legacy",
        rocket_legacy_extractions_path: Path = _DEFAULT_LEGACY_EXTRACTIONS_PATH,
        rocket_legacy_macro_skills_path: Path = _DEFAULT_LEGACY_MACRO_SKILLS_PATH,
        rocket_panel_path: Path = _DEFAULT_LEGACY_PANEL_PATH,
        rocket_financial_targets: tuple[str, ...] = (
            "+revenue_yoy",
            "+operating_margin",
            "+free_cash_flow_margin",
            "+return_on_assets",
            "-debt_to_assets",
        ),
        rocket_financial_horizon: str = "next_year",
        rocket_ket_model_path: Path | None = None,
        rocket_ket_skill_basis_path: Path | None = None,
        rocket_ket_action_vocab_path: Path | None = None,
        rocket_ket_device: str = "cpu",
        rocket_ket_weight: float = 0.20,
        dry_run: bool = False,
        enable_corpus_synthesis: bool = True,
    ) -> None:
        self.filings = tuple(filings)
        self.outdir = outdir.resolve()
        self.filing_manifest_path = filing_manifest_path.resolve()
        self.company_context_path = company_context_path.resolve()
        self.rocket_reward_source = str(rocket_reward_source or "legacy").strip().lower()
        self.rocket_legacy_extractions_path = Path(rocket_legacy_extractions_path).expanduser().resolve()
        self.rocket_legacy_macro_skills_path = Path(rocket_legacy_macro_skills_path).expanduser().resolve()
        self.rocket_panel_path = Path(rocket_panel_path).expanduser().resolve()
        self.rocket_financial_targets = tuple(str(item).strip() for item in rocket_financial_targets if str(item).strip())
        self.rocket_financial_horizon = str(rocket_financial_horizon or "next_year").strip()
        self.rocket_ket_model_path = Path(rocket_ket_model_path).expanduser().resolve() if rocket_ket_model_path else None
        self.rocket_ket_skill_basis_path = (
            Path(rocket_ket_skill_basis_path).expanduser().resolve() if rocket_ket_skill_basis_path else None
        )
        self.rocket_ket_action_vocab_path = (
            Path(rocket_ket_action_vocab_path).expanduser().resolve() if rocket_ket_action_vocab_path else None
        )
        self.rocket_ket_device = str(rocket_ket_device or "cpu").strip()
        self.rocket_ket_weight = float(max(0.0, rocket_ket_weight))
        self.dry_run = dry_run
        self.enable_corpus_synthesis = enable_corpus_synthesis
        self.workflow = build_basket_rocket_workflow()
        self.plan = tuple(tuple(agent.name for agent in frontier) for frontier in self.workflow.parallel_frontiers())
        self.summary_path = self.outdir / "basket_rocket_batch_summary.json"
        self.workset_index_path = self.outdir / "workset_index.json"
        self.live_gui_path = self.outdir / "basket_rocket_gui.html"
        self.worksets = self._build_worksets()
        self._legacy_rocket_module = None
        self._basket_tenk_module = None
        self._legacy_reward_context = None
        self._legacy_company_lookup = None

    def _build_worksets(self) -> tuple[_BasketRocketWorkset, ...]:
        grouped: dict[tuple[str, str, str], list[MaterializedSECFiling]] = {}
        for filing in self.filings:
            key = (filing.company, filing.anchor_year, filing.form_type)
            grouped.setdefault(key, []).append(filing)
        worksets = []
        for company, filing_year, form_type in sorted(grouped):
            filings = tuple(sorted(grouped[(company, filing_year, form_type)], key=lambda item: item.filing_date))
            workset_name = filings[0].workset_name
            worksets.append(
                _BasketRocketWorkset(
                    workset_name=workset_name,
                    company=company,
                    filing_year=filing_year,
                    form_type=form_type,
                    semantic_role=filings[0].semantic_role,
                    filings=filings,
                    outdir=self.outdir / workset_name,
                )
            )
        return tuple(worksets)

    def run(self) -> BasketRocketBatchRunResult:
        self.outdir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.workset_index_path,
            [
                {
                    "workset_name": workset.workset_name,
                    "company": workset.company,
                    "filing_year": workset.filing_year,
                    "form_type": workset.form_type,
                    "semantic_role": workset.semantic_role,
                    "filing_count": len(workset.filings),
                    "source_filing_years": sorted({filing.filing_year for filing in workset.filings}),
                    "filings": [asdict(filing) for filing in workset.filings],
                }
                for workset in self.worksets
            ],
        )

        records: list[BasketRocketBatchRecord] = []
        self._write_progress_snapshot(records, status="starting")
        for workset in self.worksets:
            workset.outdir.mkdir(parents=True, exist_ok=True)
            self._write_workset_inputs(workset)
            self._write_progress_snapshot(records, status="running")
            for frontier_index, frontier in enumerate(self.plan):
                for agent_name in frontier:
                    started_at = time.time()
                    if self.dry_run and agent_name != "filing_collection_agent":
                        records.append(
                            BasketRocketBatchRecord(
                                workset_name=workset.workset_name,
                                company=workset.company,
                                filing_year=workset.filing_year,
                                form_type=workset.form_type,
                                agent_name=agent_name,
                                frontier_index=frontier_index,
                                status="planned",
                                started_at=started_at,
                                ended_at=started_at,
                            )
                        )
                        self._write_progress_snapshot(records, status="running")
                        continue
                    status, outputs, notes = self._run_agent(workset, agent_name)
                    ended_at = time.time()
                    records.append(
                        BasketRocketBatchRecord(
                            workset_name=workset.workset_name,
                            company=workset.company,
                            filing_year=workset.filing_year,
                            form_type=workset.form_type,
                            agent_name=agent_name,
                            frontier_index=frontier_index,
                            status=status,
                            started_at=started_at,
                            ended_at=ended_at,
                            outputs=outputs,
                            notes=notes,
                        )
                    )
                    self._write_progress_snapshot(records, status="running")

        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.workset_name,
                    item.frontier_index,
                    item.agent_name,
                ),
            )
        )
        _write_json(self.summary_path, [asdict(record) for record in ordered])
        visualizations = None
        if not self.dry_run and self.worksets:
            visualizations = generate_basket_rocket_visualizations(self.outdir)
        self._write_progress_snapshot(
            ordered,
            status="complete",
            visualization_index_path=visualizations.index_path if visualizations else None,
        )
        corpus_synthesis = None
        if self.enable_corpus_synthesis and not self.dry_run and self.worksets:
            corpus_synthesis = build_basket_rocket_corpus_synthesis(
                query="BASKET/ROCKET corpus synthesis",
                batch_outdir=self.outdir,
                workset_index_path=self.workset_index_path,
                visualization_summary_path=visualizations.summary_path if visualizations else None,
                company_summary_path=visualizations.company_summary_path if visualizations else None,
            )
        return BasketRocketBatchRunResult(
            records=ordered,
            workset_index_path=self.workset_index_path,
            summary_path=self.summary_path,
            live_gui_path=self.live_gui_path,
            visualizations=visualizations,
            corpus_synthesis=corpus_synthesis,
        )

    def _write_progress_snapshot(
        self,
        records: tuple[BasketRocketBatchRecord, ...] | list[BasketRocketBatchRecord],
        *,
        status: str,
        visualization_index_path: Path | None = None,
    ) -> None:
        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.workset_name,
                    item.frontier_index,
                    item.agent_name,
                    item.started_at,
                ),
            )
        )
        _write_json(self.summary_path, [asdict(record) for record in ordered])
        self.live_gui_path.write_text(
            self._render_live_gui_html(
                ordered,
                status=status,
                visualization_index_path=visualization_index_path,
            ),
            encoding="utf-8",
        )

    def _render_live_gui_html(
        self,
        records: tuple[BasketRocketBatchRecord, ...],
        *,
        status: str,
        visualization_index_path: Path | None = None,
    ) -> str:
        def esc(value: object) -> str:
            return html.escape(str(value))

        refresh_meta = '<meta http-equiv="refresh" content="5">' if status != "complete" else ""
        records_by_workset: dict[str, list[BasketRocketBatchRecord]] = {}
        for record in records:
            records_by_workset.setdefault(record.workset_name, []).append(record)

        total_worksets = len(self.worksets)
        total_agents = total_worksets * sum(len(frontier) for frontier in self.plan)
        completed_agents = sum(1 for record in records if record.status in {"completed", "planned"})
        completed_worksets = sum(
            1
            for workset in self.worksets
            if len(records_by_workset.get(workset.workset_name, ())) >= sum(len(frontier) for frontier in self.plan)
        )
        reranked_worksets = sum(
            1
            for workset in self.worksets
            if (workset.outdir / "rocket_rankings.json").exists()
        )
        report_ready_worksets = sum(
            1
            for workset in self.worksets
            if (workset.outdir / "workflow_report.md").exists()
        )

        hero_link = ""
        if visualization_index_path and visualization_index_path.exists():
            hero_link = (
                f'<a class="hero-link" href="{esc(visualization_index_path.resolve().as_uri())}" '
                'target="_blank" rel="noreferrer">Open final BASKET/ROCKET visualization suite</a>'
            )

        plan_labels = " -> ".join(" + ".join(frontier) for frontier in self.plan)

        workset_cards: list[str] = []
        for workset in self.worksets:
            workset_records = records_by_workset.get(workset.workset_name, [])
            workset_status = "queued"
            if len(workset_records) >= sum(len(frontier) for frontier in self.plan):
                workset_status = "complete"
            elif workset_records:
                workset_status = "running"

            latest_record = max(workset_records, key=lambda item: item.ended_at or item.started_at, default=None)
            input_payload = _read_json(workset.outdir / "input_filings.json")
            candidate_payload = _read_json(workset.outdir / "candidate_workflows.json")
            rankings_payload = _read_json(workset.outdir / "rocket_rankings.json")
            psr_payload = _read_json(workset.outdir / "psr_models.json")
            top_candidate = next(iter((candidate_payload.get("candidates") or []) if isinstance(candidate_payload, dict) else []), {})
            top_ranking = next(iter((rankings_payload.get("rankings") or []) if isinstance(rankings_payload, dict) else []), {})
            filing_rows = list((input_payload.get("filings") or []) if isinstance(input_payload, dict) else [])
            filing_chips = "".join(
                f'<span class="chip neutral">{esc(row.get("form_type", ""))} · {esc(row.get("filing_date", ""))}</span>'
                for row in filing_rows[:4]
            ) or '<span class="chip neutral">filings pending</span>'
            evidence_chips = "".join(
                f'<span class="chip neutral">{esc(section)}</span>'
                for section in list(top_candidate.get("evidence_sections") or [])[:5]
            ) or '<span class="chip neutral">Evidence preview will appear after extraction.</span>'
            workflow_stage_markup = "".join(
                f'<span class="workflow-stage">{esc(stage)}</span>'
                for stage in list(top_ranking.get("workflow_stages") or psr_payload.get("latent_states") or [])[:8]
            ) or '<div class="empty">Recovered workflow backbone will appear after ROCKET reranking.</div>'
            financial_target_markup = "".join(
                f'<span class="chip mixed">{esc(target)}</span>'
                for target in list(top_ranking.get("financial_targets") or [])[:5]
            ) or '<span class="chip mixed">financial targets pending</span>'
            progress_label = (
                f"{len(workset_records)}/{sum(len(frontier) for frontier in self.plan)} agents"
                if workset_records
                else "waiting for first agent"
            )
            top_line = "Workflow report will appear after the reporting stage finishes."
            if top_ranking:
                top_line = (
                    f"Selected source {top_ranking.get('selected_source', 'unknown')} "
                    f"with label {top_ranking.get('selected_label', 'n/a')} and "
                    f"score gain {float(top_ranking.get('score_gain') or 0.0):.3f}."
                )
            note = latest_record.notes if latest_record and latest_record.notes else "SEC workset staged for BASKET/ROCKET analysis."
            links = []
            report_path = workset.outdir / "workflow_report.md"
            if report_path.exists():
                links.append(
                    f'<a href="{esc(report_path.resolve().as_uri())}" target="_blank" rel="noreferrer">Workflow report</a>'
                )
            if visualization_index_path and visualization_index_path.exists():
                links.append(
                    f'<a href="{esc(visualization_index_path.resolve().as_uri())}" target="_blank" rel="noreferrer">Visualization suite</a>'
                )
            link_markup = " · ".join(links) if links else "More artifacts will appear as this workset progresses."
            status_tone = "success" if workset_status == "complete" else ("mixed" if workset_status == "running" else "neutral")
            workset_cards.append(
                '<section class="workset-card">'
                '<div class="workset-header">'
                '<div>'
                f'<div class="eyebrow">{esc(workset.form_type)} · {esc(workset.semantic_role.replace("_", " "))}</div>'
                f'<h2>{esc(workset.company)} · {esc(workset.filing_year)}</h2>'
                f'<p class="summary">{esc(note)}</p>'
                "</div>"
                '<div class="status-strip">'
                f'<span class="chip {status_tone}">{esc(workset_status)}</span>'
                f'<span class="chip neutral">{esc(progress_label)}</span>'
                f'<span class="chip neutral">{esc(workset.workset_name)}</span>'
                "</div>"
                "</div>"
                '<div class="section-label">Source Filings</div>'
                f'<div class="chip-row">{filing_chips}</div>'
                '<div class="grid two-up">'
                '<div class="panel">'
                '<div class="section-label">Evidence Preview</div>'
                f'<div class="chip-row">{evidence_chips}</div>'
                f'<p class="detail">{esc(top_line)}</p>'
                "</div>"
                '<div class="panel">'
                '<div class="section-label">Recovered Workflow Backbone</div>'
                f'<div class="workflow-row">{workflow_stage_markup}</div>'
                '<div class="section-label">Financial Targets</div>'
                f'<div class="chip-row">{financial_target_markup}</div>'
                "</div>"
                "</div>"
                f'<div class="links">{link_markup}</div>'
                "</section>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {refresh_meta}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE BASKET/ROCKET GUI</title>
  <style>
    :root {{
      --ink: #172432;
      --muted: #5c6d7d;
      --paper: #f7f1e7;
      --card: rgba(255, 252, 247, 0.97);
      --line: #d9cdbf;
      --accent: #0b6e4f;
      --accent-soft: #dff4ea;
      --mixed: #9a5b16;
      --mixed-soft: #f9ead0;
      --shadow: 0 18px 44px rgba(23, 36, 50, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
      background:
        radial-gradient(circle at top right, rgba(11,110,79,0.10), transparent 24%),
        radial-gradient(circle at left center, rgba(154,91,22,0.08), transparent 22%),
        linear-gradient(180deg, #fbf6ee 0%, #efe4d3 100%);
    }}
    .shell {{ max-width: 1240px; margin: 0 auto; padding: 28px 18px 48px; }}
    .hero, .workset-card {{
      background: var(--card);
      border: 1px solid rgba(217, 205, 191, 0.96);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    h1 {{ margin: 10px 0 12px; font-size: clamp(32px, 4vw, 54px); line-height: 0.98; }}
    h2 {{ margin: 6px 0 0; font-size: 29px; }}
    .hero p, .summary, .detail {{
      color: var(--muted);
      line-height: 1.6;
      font-size: 16px;
    }}
    .hero-meta, .chip-row, .status-strip, .workflow-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .hero-meta {{ margin-top: 18px; }}
    .hero-link {{
      display: inline-block;
      margin-top: 18px;
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .hero-link:hover {{ text-decoration: underline; }}
    .metrics {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .metric {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .metric-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{ margin-top: 8px; font-size: 30px; }}
    .chip, .workflow-stage {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      border: 1px solid transparent;
      background: #edf2f7;
    }}
    .chip.success {{ background: var(--accent-soft); color: var(--accent); }}
    .chip.mixed {{ background: var(--mixed-soft); color: var(--mixed); }}
    .chip.neutral {{ background: #edf2f7; color: var(--ink); }}
    .workflow-stage {{
      background: white;
      border-color: var(--line);
      color: var(--ink);
      font-weight: 600;
    }}
    .worksets {{ display: grid; gap: 18px; margin-top: 24px; }}
    .workset-card {{ padding: 22px; }}
    .workset-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
    }}
    .section-label {{
      margin: 18px 0 10px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .grid.two-up {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
      margin-top: 8px;
    }}
    .panel {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .links {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .links a:hover {{ text-decoration: underline; }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 14px;
      color: var(--muted);
      background: rgba(255,255,255,0.72);
    }}
    @media (max-width: 960px) {{
      .metrics, .grid.two-up {{ grid-template-columns: 1fr 1fr; }}
      .workset-header {{ flex-direction: column; }}
    }}
    @media (max-width: 680px) {{
      .metrics, .grid.two-up {{ grid-template-columns: 1fr; }}
      h2 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Financial Workflow Reader</div>
      <h1>BASKET/ROCKET 10-K GUI</h1>
      <p>Recovered SEC filing evidence, reranked workflow backbones, and predictive-state summaries for the current financial batch. This page refreshes while the run is active so the router can surface meaningful progress before the final visualization suite is finished.</p>
      <div class="hero-meta">
        <span class="chip {'success' if status == 'complete' else 'mixed'}">{esc(status.replace('_', ' '))}</span>
        <span class="chip neutral">{esc(completed_agents)}/{esc(total_agents)} agent steps materialized</span>
        <span class="chip neutral">{esc(plan_labels)}</span>
      </div>
      {hero_link}
      <div class="metrics">
        <div class="metric"><div class="metric-label">Worksets</div><div class="metric-value">{esc(total_worksets)}</div></div>
        <div class="metric"><div class="metric-label">Completed Worksets</div><div class="metric-value">{esc(completed_worksets)}</div></div>
        <div class="metric"><div class="metric-label">Reranked</div><div class="metric-value">{esc(reranked_worksets)}</div></div>
        <div class="metric"><div class="metric-label">Reports Ready</div><div class="metric-value">{esc(report_ready_worksets)}</div></div>
      </div>
    </section>
    <div class="worksets">
      {"".join(workset_cards) if workset_cards else '<section class="workset-card"><div class="empty">Worksets will appear here after SEC filings are grouped into company-year-form batches.</div></section>'}
    </div>
  </div>
</body>
</html>
"""

    def _write_workset_inputs(self, workset: _BasketRocketWorkset) -> None:
        _write_json(
            workset.outdir / "input_filings.json",
            {
                "workset_name": workset.workset_name,
                "company": workset.company,
                "filing_year": workset.filing_year,
                "form_type": workset.form_type,
                "semantic_role": workset.semantic_role,
                "filing_count": len(workset.filings),
                "filings": [asdict(filing) for filing in workset.filings],
            },
        )

    def _run_agent(self, workset: _BasketRocketWorkset, agent_name: str) -> tuple[str, tuple[str, ...], str]:
        handlers = {
            "filing_collection_agent": self._run_filing_collection_agent,
            "filing_chunking_agent": self._run_filing_chunking_agent,
            "basket_artifact_builder_agent": self._run_basket_artifact_builder_agent,
            "workflow_extraction_agent": self._run_workflow_extraction_agent,
            "rocket_reranking_agent": self._run_rocket_reranking_agent,
            "psr_modeling_agent": self._run_psr_modeling_agent,
            "workflow_reporting_agent": self._run_workflow_reporting_agent,
        }
        handler = handlers[agent_name]
        return handler(workset)

    def _run_filing_collection_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        path = workset.outdir / "collection_bundle.json"
        _write_json(
            path,
            {
                "workset_name": workset.workset_name,
                "company": workset.company,
                "filing_year": workset.filing_year,
                "form_type": workset.form_type,
                "semantic_role": workset.semantic_role,
                "filing_manifest_path": str(self.filing_manifest_path),
                "company_context_path": str(self.company_context_path),
                "filings": [asdict(filing) for filing in workset.filings],
            },
        )
        return "completed", (str(path),), "reused SEC ingress artifacts"

    def _run_filing_chunking_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        path = workset.outdir / "filing_chunks.json"
        chunks = []
        for filing in workset.filings:
            text = self._filing_text(filing)
            for index, chunk in enumerate(self._split_into_chunks(text), start=1):
                chunks.append(
                    {
                        "chunk_id": f"{_slugify(Path(filing.filing_path).stem)}_{index:03d}",
                        "chunk_index": index,
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "char_count": len(chunk["text"]),
                        "filing_title": filing.title,
                        "filing_path": filing.filing_path,
                        "form_type": filing.form_type,
                        "filing_date": filing.filing_date,
                        "semantic_role": filing.semantic_role,
                        "event_item_codes": list(filing.event_item_codes),
                    }
                )
        _write_json(path, {"workset_name": workset.workset_name, "chunks": chunks})
        return "completed", (str(path),), f"chunked {len(chunks)} filing sections"

    def _run_basket_artifact_builder_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        chunks_path = workset.outdir / "filing_chunks.json"
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = list(payload.get("chunks") or [])
        by_filing: dict[str, list[dict[str, object]]] = {}
        for chunk in chunks:
            by_filing.setdefault(str(chunk["filing_title"]), []).append(chunk)
        path = workset.outdir / "basket_artifacts.json"
        artifacts = []
        for filing in workset.filings:
            filing_chunks = by_filing.get(filing.title, [])
            section_titles = [str(chunk["title"]) for chunk in filing_chunks]
            grounded_action_evidence = self._grounded_action_evidence_for_chunks(
                filing_chunks,
                semantic_role=filing.semantic_role,
                event_item_codes=tuple(filing.event_item_codes),
            )
            dominant_chunk_candidates = self._dominant_chunk_candidates(
                filing_chunks,
                semantic_role=filing.semantic_role,
                event_item_codes=tuple(filing.event_item_codes),
            )
            artifacts.append(
                {
                    "filing_title": filing.title,
                    "company": filing.company,
                    "ticker": filing.ticker,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date,
                    "semantic_role": filing.semantic_role,
                    "anchor_year": filing.anchor_year,
                    "event_item_codes": list(filing.event_item_codes),
                    "section_titles": section_titles,
                    "chunk_count": len(filing_chunks),
                    "text_path": filing.text_path,
                    "grounded_action_evidence": grounded_action_evidence,
                    "dominant_chunk_candidates": dominant_chunk_candidates,
                    "artifact_status": "grounded_scaffold",
                }
            )
        _write_json(
            path,
            {
                "workset_name": workset.workset_name,
                "artifact_family": "BASKET",
                "artifacts": artifacts,
            },
        )
        return "completed", (str(path),), f"built {len(artifacts)} scaffolded filing artifacts"

    def _run_workflow_extraction_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        scaffold_candidates = self._build_scaffold_candidates(workset)
        basket_rows = self._build_basket_workflow_extractions(workset)
        if basket_rows:
            workflow_extractions_path = workset.outdir / "workflow_extractions.jsonl"
            candidate_path = workset.outdir / "candidate_workflows.json"
            summary_path = workset.outdir / "workflow_extractions_summary.json"
            _write_jsonl(workflow_extractions_path, basket_rows)
            candidates = list(scaffold_candidates)
            candidates.extend(self._candidate_from_basket_extraction(row) for row in basket_rows)
            _write_json(
                candidate_path,
                {
                    "workset_name": workset.workset_name,
                    "candidate_family": "BASKET",
                    "candidates": candidates,
                },
            )
            _write_json(
                summary_path,
                {
                    "workset_name": workset.workset_name,
                    "extraction_backend": "basket_tenk_real",
                    "extraction_count": len(basket_rows),
                    "scaffold_candidate_count": len(scaffold_candidates),
                    "candidate_count": len(candidates),
                },
            )
            return (
                "completed",
                (str(workflow_extractions_path), str(candidate_path), str(summary_path)),
                f"extracted {len(basket_rows)} BASKET workflow rows",
            )

        candidates = self._build_scaffold_candidates(workset)
        path = workset.outdir / "candidate_workflows.json"
        _write_json(path, {"workset_name": workset.workset_name, "candidates": candidates})
        return "completed", (str(path),), f"extracted {len(candidates)} workflow candidates"

    def _build_scaffold_candidates(self, workset: _BasketRocketWorkset) -> list[dict[str, object]]:
        artifacts_payload = json.loads((workset.outdir / "basket_artifacts.json").read_text(encoding="utf-8"))
        artifacts = list(artifacts_payload.get("artifacts") or [])
        candidates = []
        for index, artifact in enumerate(artifacts, start=1):
            section_titles = [str(title).lower() for title in artifact.get("section_titles") or []]
            artifact_candidates = self._workflow_candidates_for_artifact(
                artifact,
                section_titles=section_titles,
            )
            for candidate_offset, candidate in enumerate(artifact_candidates, start=1):
                candidate["candidate_id"] = f"{workset.workset_name}_candidate_{index:02d}_{candidate_offset:02d}"
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _candidate_stages_for_artifact(
        artifact: dict[str, object],
        section_titles: list[str],
    ) -> list[str]:
        semantic_role = str(artifact.get("semantic_role") or "")
        event_item_codes = [str(item) for item in artifact.get("event_item_codes") or [] if str(item).strip()]
        section_text = " || ".join(section_titles)
        stages: list[str] = []

        def has_any(*keywords: str) -> bool:
            return any(keyword in section_text for keyword in keywords)

        if semantic_role == "annual_anchor":
            if has_any("investment", "capital", "acquisition", "research", "development"):
                _append_unique(stages, "invest")
            if has_any("innovation", "innov", "product", "research", "development", "technology"):
                _append_unique(stages, "innovate")
            if has_any("digital", "cloud", "software", "platform", "data", "ai", "automation"):
                _append_unique(stages, "digitize")
            if has_any("risk", "management", "discussion", "operations", "efficiency", "controls", "restructuring"):
                _append_unique(stages, "optimize")
            if has_any("growth", "strategy", "international", "segment", "business", "geographic", "acquisition"):
                _append_unique(stages, "expand")
            if has_any("distribution", "channel", "logistics", "supply", "fulfillment", "partner"):
                _append_unique(stages, "distribute")
            if has_any("marketing", "sales", "customer", "brand", "demand", "segment"):
                _append_unique(stages, "market")
            if has_any("pricing", "price", "margin", "financial", "revenue", "contract", "subscription", "fee"):
                _append_unique(stages, "price")
            if "optimize" not in stages:
                _append_unique(stages, "optimize")
            _append_unique(stages, "realize_revenue")
            return _ordered_actions(stages)

        if semantic_role == "event_patch":
            item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
            if "1" in item_prefixes:
                _append_unique(stages, "invest", "expand")
            if "2" in item_prefixes:
                _append_unique(stages, "price", "realize_revenue")
            if "5" in item_prefixes:
                _append_unique(stages, "optimize")
            if "8" in item_prefixes:
                _append_unique(stages, "market")
            if has_any("innovation", "innov", "product", "research", "development", "technology"):
                _append_unique(stages, "innovate")
            if has_any("digital", "cloud", "software", "platform", "data", "ai", "automation"):
                _append_unique(stages, "digitize")
            if has_any("risk", "management", "discussion", "operations", "controls", "integration", "restructuring"):
                _append_unique(stages, "optimize")
            if has_any("distribution", "channel", "logistics", "supply", "partner"):
                _append_unique(stages, "distribute")
            if has_any("marketing", "sales", "customer", "brand", "demand"):
                _append_unique(stages, "market")
            if has_any("pricing", "price", "margin", "financial", "revenue", "contract", "fee"):
                _append_unique(stages, "price")
            if not stages:
                _append_unique(stages, "optimize")
            if any(action in stages for action in ("invest", "innovate", "digitize", "expand", "distribute", "market", "price")):
                _append_unique(stages, "realize_revenue")
            return _ordered_actions(stages)

        if semantic_role == "quarterly_update":
            if has_any("digital", "cloud", "software", "platform", "data", "ai"):
                _append_unique(stages, "digitize")
            if has_any("risk", "management", "discussion", "operations", "controls"):
                _append_unique(stages, "optimize")
            if has_any("growth", "strategy", "segment", "international"):
                _append_unique(stages, "expand")
            if has_any("marketing", "sales", "customer", "brand", "demand"):
                _append_unique(stages, "market")
            if has_any("pricing", "price", "margin", "financial", "revenue", "fee"):
                _append_unique(stages, "price")
            if "optimize" not in stages:
                _append_unique(stages, "optimize")
            _append_unique(stages, "realize_revenue")
            return _ordered_actions(stages)

        if has_any("innovation", "innov", "product", "research", "development", "technology"):
            _append_unique(stages, "innovate")
        if has_any("digital", "cloud", "software", "platform", "data", "ai"):
            _append_unique(stages, "digitize")
        if has_any("operations", "management", "risk", "controls"):
            _append_unique(stages, "optimize")
        if has_any("growth", "strategy", "market", "sales", "customer", "brand"):
            _append_unique(stages, "expand", "market")
        if has_any("pricing", "price", "margin", "financial", "revenue"):
            _append_unique(stages, "price")
        _append_unique(stages, "realize_revenue")
        return _ordered_actions(stages)

    @classmethod
    def _workflow_candidates_for_artifact(
        cls,
        artifact: dict[str, object],
        *,
        section_titles: list[str],
    ) -> list[dict[str, object]]:
        evidence_map = dict(artifact.get("grounded_action_evidence") or {})
        base_stages = cls._candidate_stages_for_artifact(artifact, section_titles)
        base_evidence = cls._flatten_grounded_evidence(evidence_map, actions=base_stages)
        section_count = len(section_titles)
        filing_title = str(artifact.get("filing_title") or "")
        semantic_role = str(artifact.get("semantic_role") or "")
        event_item_codes = list(artifact.get("event_item_codes") or [])
        evidence_sections = list(dict.fromkeys(str(item.get("section_title") or "").strip() for item in base_evidence if str(item.get("section_title") or "").strip()))
        candidates = [
            {
                "filing_title": filing_title,
                "workflow_stages": base_stages,
                "semantic_role": semantic_role,
                "event_item_codes": event_item_codes,
                "evidence_sections": evidence_sections or list(artifact.get("section_titles") or []),
                "evidence_spans": base_evidence,
                "score_basis": {"stage_count": len(base_stages), "section_count": section_count, "evidence_span_count": len(base_evidence)},
                "candidate_source": "grounded_filing_backbone",
            }
        ]

        evidence_order_actions = cls._workflow_actions_from_evidence_order(
            evidence_map,
            semantic_role=semantic_role,
            event_item_codes=tuple(str(item) for item in event_item_codes if str(item).strip()),
        )
        if evidence_order_actions and tuple(evidence_order_actions) != tuple(base_stages):
            evidence_order_spans = cls._flatten_grounded_evidence(evidence_map, actions=evidence_order_actions)
            evidence_order_sections = list(
                dict.fromkeys(
                    str(item.get("section_title") or "").strip()
                    for item in evidence_order_spans
                    if str(item.get("section_title") or "").strip()
                )
            )
            candidates.append(
                {
                    "filing_title": filing_title,
                    "workflow_stages": evidence_order_actions,
                    "semantic_role": semantic_role,
                    "event_item_codes": event_item_codes,
                    "evidence_sections": evidence_order_sections or list(artifact.get("section_titles") or []),
                    "evidence_spans": evidence_order_spans,
                    "score_basis": {
                        "stage_count": len(evidence_order_actions),
                        "section_count": section_count,
                        "evidence_span_count": len(evidence_order_spans),
                    },
                    "candidate_source": "grounded_evidence_order",
                }
            )

        seen = {tuple(base_stages)}
        dominant_chunk_candidates = list(artifact.get("dominant_chunk_candidates") or [])
        for chunk_candidate in dominant_chunk_candidates[:2]:
            chunk_stages = [str(item) for item in chunk_candidate.get("workflow_stages") or [] if str(item).strip()]
            if not chunk_stages:
                continue
            ordered_chunk_stages = _ordered_actions(chunk_stages)
            key = tuple(ordered_chunk_stages)
            if key in seen:
                continue
            seen.add(key)
            evidence_spans = [
                {
                    "action": str(item.get("action") or ""),
                    "section_title": str(chunk_candidate.get("title") or ""),
                    "snippet": str(item.get("snippet") or ""),
                    "score": float(item.get("score") or 0.0),
                    "match_basis": str(item.get("match_basis") or ""),
                }
                for item in chunk_candidate.get("action_evidence") or []
                if str(item.get("action") or "").strip()
            ]
            candidates.append(
                {
                    "filing_title": filing_title,
                    "workflow_stages": ordered_chunk_stages,
                    "semantic_role": semantic_role,
                    "event_item_codes": event_item_codes,
                    "evidence_sections": [str(chunk_candidate.get("title") or "")] if str(chunk_candidate.get("title") or "").strip() else [],
                    "evidence_spans": evidence_spans,
                    "score_basis": {"stage_count": len(ordered_chunk_stages), "section_count": section_count, "evidence_span_count": len(evidence_spans)},
                    "candidate_source": "grounded_chunk_focus",
                }
            )
        return candidates

    @classmethod
    def _flatten_grounded_evidence(
        cls,
        evidence_map: dict[str, object],
        *,
        actions: list[str],
        limit_per_action: int = 2,
        limit_total: int = 8,
    ) -> list[dict[str, object]]:
        spans: list[dict[str, object]] = []
        for action in actions:
            rows = list(evidence_map.get(action) or [])
            for row in rows[:limit_per_action]:
                spans.append(
                    {
                        "action": action,
                        "section_title": str(row.get("section_title") or ""),
                        "snippet": str(row.get("snippet") or ""),
                        "score": float(row.get("score") or 0.0),
                        "match_basis": str(row.get("match_basis") or ""),
                        "chunk_index": int(row.get("chunk_index") or 0),
                        "sentence_index": int(row.get("sentence_index") or 0),
                    }
                )
        spans.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                _ACTION_PRIORITY.get(str(item.get("action") or ""), len(_ACTION_PRIORITY)),
                str(item.get("section_title") or ""),
            )
        )
        return spans[:limit_total]

    @classmethod
    def _workflow_actions_from_evidence_order(
        cls,
        evidence_map: dict[str, object],
        *,
        semantic_role: str,
        event_item_codes: tuple[str, ...],
    ) -> list[str]:
        ordered_rows: list[dict[str, object]] = []
        for action, rows in evidence_map.items():
            for row in rows:
                ordered_rows.append(
                    {
                        "action": str(action),
                        "chunk_index": int(row.get("chunk_index") or 0),
                        "sentence_index": int(row.get("sentence_index") or 0),
                        "score": float(row.get("score") or 0.0),
                    }
                )
        ordered_rows.sort(
            key=lambda item: (
                int(item.get("chunk_index") or 0),
                int(item.get("sentence_index") or 0),
                -float(item.get("score") or 0.0),
                _ACTION_PRIORITY.get(str(item.get("action") or ""), len(_ACTION_PRIORITY)),
            )
        )
        actions: list[str] = []
        for row in ordered_rows:
            action = str(row.get("action") or "").strip()
            if action and action not in actions:
                actions.append(action)
        item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
        if semantic_role == "event_patch" and "1" in item_prefixes:
            actions = _insert_action(actions, "invest", before=("expand", "market", "price", "realize_revenue"))
        if semantic_role == "event_patch" and "2" in item_prefixes:
            actions = _insert_action(actions, "price", before=("realize_revenue",))
        if semantic_role in {"annual_anchor", "quarterly_update"} and "realize_revenue" not in actions and any(
            action in actions for action in ("market", "price", "expand", "optimize", "digitize")
        ):
            actions = _insert_action(actions, "realize_revenue")
        return actions

    @classmethod
    def _grounded_action_evidence_for_chunks(
        cls,
        chunks: list[dict[str, object]],
        *,
        semantic_role: str,
        event_item_codes: tuple[str, ...],
    ) -> dict[str, list[dict[str, object]]]:
        evidence_by_action: dict[str, list[dict[str, object]]] = {}
        for chunk in chunks:
            title = str(chunk.get("title") or "").strip()
            text = str(chunk.get("text") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            sentences = cls._sentences_from_text(text)
            if not sentences and text.strip():
                sentences = [text.strip()]
            for sentence_index, sentence in enumerate(sentences[:24], start=1):
                lowered = sentence.lower()
                for action, keywords in _ACTION_KEYWORDS.items():
                    score = 0.0
                    basis: list[str] = []
                    title_lower = title.lower()
                    if any(keyword in title_lower for keyword in keywords):
                        score += 1.35
                        basis.append("section_title")
                    keyword_hits = [keyword for keyword in keywords if keyword in lowered]
                    if keyword_hits:
                        score += 1.0 + (0.2 * min(len(keyword_hits), 3))
                        basis.append("sentence")
                    if any(prefix in {code.split(".", 1)[0] for code in event_item_codes} for prefix in _ACTION_ITEM_PREFIXES.get(action, ())):
                        score += 0.55
                        basis.append("item_code")
                    if semantic_role == "annual_anchor" and action in {"optimize", "market", "realize_revenue"}:
                        score += 0.1
                    if score <= 0.0:
                        continue
                    evidence_by_action.setdefault(action, []).append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": int(chunk.get("chunk_index") or 0),
                            "sentence_index": sentence_index,
                            "section_title": title,
                            "snippet": cls._compact_snippet(sentence),
                            "score": round(score, 4),
                            "match_basis": "+".join(basis),
                        }
                    )
        for action, rows in evidence_by_action.items():
            deduped: list[dict[str, object]] = []
            seen = set()
            for row in sorted(rows, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("snippet") or ""))):
                key = (str(row.get("section_title") or ""), str(row.get("snippet") or ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            evidence_by_action[action] = deduped[:6]
        return evidence_by_action

    @classmethod
    def _dominant_chunk_candidates(
        cls,
        chunks: list[dict[str, object]],
        *,
        semantic_role: str,
        event_item_codes: tuple[str, ...],
    ) -> list[dict[str, object]]:
        chunk_candidates: list[dict[str, object]] = []
        item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
        for chunk in chunks:
            title = str(chunk.get("title") or "")
            text = str(chunk.get("text") or "")
            local_evidence = cls._grounded_action_evidence_for_chunks(
                [chunk],
                semantic_role=semantic_role,
                event_item_codes=event_item_codes,
            )
            actions = [action for action in _LEGACY_ACTION_ORDER if local_evidence.get(action)]
            if semantic_role == "event_patch" and "1" in item_prefixes:
                actions = _insert_action(actions, "invest", before=("expand", "market", "price", "realize_revenue"))
                actions = _insert_action(actions, "expand", before=("market", "price", "realize_revenue"))
            if semantic_role == "event_patch" and "2" in item_prefixes:
                actions = _insert_action(actions, "price", before=("realize_revenue",))
            if actions and actions[-1] != "realize_revenue" and ("revenue" in text.lower() or "financial" in text.lower()):
                actions = _insert_action(actions, "realize_revenue")
            if len(actions) < 2:
                continue
            top_score = sum(float(rows[0].get("score") or 0.0) for rows in local_evidence.values() if rows)
            action_evidence = cls._flatten_grounded_evidence(local_evidence, actions=actions, limit_per_action=1, limit_total=5)
            chunk_candidates.append(
                {
                    "title": title,
                    "workflow_stages": actions,
                    "action_evidence": action_evidence,
                    "chunk_score": round(top_score, 4),
                }
            )
        chunk_candidates.sort(key=lambda item: (-float(item.get("chunk_score") or 0.0), str(item.get("title") or "")))
        return chunk_candidates[:3]

    @staticmethod
    def _sentences_from_text(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[\.\?!;])\s+", normalized)
        return [part.strip() for part in parts if len(part.strip()) >= 30]

    @staticmethod
    def _compact_snippet(text: str, max_chars: int = 220) -> str:
        compact = " ".join(str(text or "").split()).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _run_rocket_reranking_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        real_rerank = self._run_real_rocket_reranking_agent(workset)
        if real_rerank is not None:
            return real_rerank

        candidates_payload = json.loads((workset.outdir / "candidate_workflows.json").read_text(encoding="utf-8"))
        candidates = list(candidates_payload.get("candidates") or [])
        rankings = []
        for candidate in candidates:
            reranked = self._rerank_candidate_workflow(candidate, workset)
            rankings.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "filing_title": candidate["filing_title"],
                    "candidate_source": str(candidate.get("candidate_source") or ""),
                    "evidence_spans": list(candidate.get("evidence_spans") or []),
                    **reranked,
                }
            )
        ranked = sorted(
            rankings,
            key=lambda row: (
                float(row.get("selected_score") or row.get("score") or 0.0),
                float(row.get("score_gain") or 0.0),
                int(len(row.get("workflow_stages") or [])),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        path = workset.outdir / "rocket_rankings.json"
        _write_json(
            path,
            {
                "workset_name": workset.workset_name,
                "ranking_family": "ROCKET",
                "reward_mode": "financial",
                "reward_backend": str(ranked[0].get("reward_backend") or "heuristic") if ranked else "heuristic",
                "financial_targets": (
                    ranked[0].get("financial_targets")
                    if ranked and ranked[0].get("financial_targets")
                    else list(self.rocket_financial_targets or self._financial_targets_for_role(workset.semantic_role, ()))
                ),
                "rankings": ranked,
            },
        )
        return "completed", (str(path),), f"ranked {len(ranked)} workflow candidates"

    def _run_real_rocket_reranking_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str] | None:
        workflow_extractions_path = workset.outdir / "workflow_extractions.jsonl"
        if not workflow_extractions_path.exists():
            return None
        rocket = self._get_legacy_rocket_module()
        if rocket is None:
            return None
        reward_context = self._get_legacy_reward_context(rocket)
        if reward_context is None:
            return None

        try:
            extraction_rows = rocket.load_jsonl(workflow_extractions_path)
        except Exception:
            return None
        if not extraction_rows:
            return None

        candidate_payload = _read_json(workset.outdir / "candidate_workflows.json")
        candidate_rows = (
            list(candidate_payload.get("candidates") or [])
            if isinstance(candidate_payload, dict)
            else []
        )
        candidate_lookup = {
            str(row.get("candidate_id") or ""): row
            for row in candidate_rows
            if str(row.get("candidate_id") or "").strip()
        }

        reranked_rows: list[dict[str, object]] = []
        ranking_rows: list[dict[str, object]] = []
        for extraction_row in extraction_rows:
            try:
                reranked = rocket.rerank_extraction(
                    extraction_row,
                    reward_context,
                    reward_mode="financial",
                    max_candidates=int(getattr(rocket, "DEFAULT_MAX_CANDIDATES", 12)),
                )
            except Exception:
                return None
            reranked_rows.append(reranked)
            ranking_rows.append(
                self._ranking_from_real_rerank(
                    reranked,
                    extraction_row=extraction_row,
                    candidate_row=candidate_lookup.get(str(extraction_row.get("statement_id") or "")),
                    financial_targets=list(getattr(reward_context, "financial_targets", ()) or ()),
                )
            )

        ranking_rows.sort(
            key=lambda row: (
                float(row.get("selected_score") or row.get("score") or 0.0),
                float(row.get("score_gain") or 0.0),
                int(len(row.get("workflow_stages") or [])),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranking_rows, start=1):
            row["rank"] = rank

        reranked_path = workset.outdir / "reranked_workflows.jsonl"
        reranked_summary_path = workset.outdir / "reranked_summary.json"
        rankings_path = workset.outdir / "rocket_rankings.json"
        _write_jsonl(reranked_path, reranked_rows)
        _write_json(reranked_summary_path, rocket.summarize_reranking(reranked_rows))
        _write_json(
            rankings_path,
            {
                "workset_name": workset.workset_name,
                "ranking_family": "ROCKET",
                "reward_mode": "financial",
                "reward_backend": "legacy_financial_real",
                "extraction_backend": "basket_tenk_real",
                "financial_targets": list(getattr(reward_context, "financial_targets", ()) or ()),
                "rankings": ranking_rows,
            },
        )
        return (
            "completed",
            (str(rankings_path), str(reranked_path), str(reranked_summary_path)),
            f"ranked {len(ranking_rows)} BASKET extraction rows with real ROCKET",
        )

    def _rerank_candidate_workflow(
        self,
        candidate: dict[str, object],
        workset: _BasketRocketWorkset,
    ) -> dict[str, object]:
        if self.rocket_reward_source == "legacy":
            legacy = self._legacy_rerank_candidate_workflow(candidate, workset)
            if legacy is not None:
                return legacy
        reranked = self._heuristic_rerank_candidate_workflow(candidate, workset)
        reranked["reward_backend"] = "heuristic"
        return reranked

    def _legacy_rerank_candidate_workflow(
        self,
        candidate: dict[str, object],
        workset: _BasketRocketWorkset,
    ) -> dict[str, object] | None:
        rocket = self._get_legacy_rocket_module()
        if rocket is None:
            return None
        reward_context = self._get_legacy_reward_context(rocket)
        if reward_context is None:
            return None

        company_id = self._resolve_legacy_company_id(
            company=workset.company,
            ticker=next((filing.ticker for filing in workset.filings if filing.ticker), ""),
        )
        if not company_id:
            return None
        if not str(workset.filing_year).isdigit():
            return None

        base_actions = _ordered_unique([str(stage) for stage in candidate.get("workflow_stages") or [] if str(stage).strip()])
        extraction_row = {
            "statement_id": str(candidate.get("candidate_id") or ""),
            "company": company_id,
            "year": int(workset.filing_year),
            "topic": str(candidate.get("filing_title") or ""),
            "statement": " ".join(str(item) for item in candidate.get("evidence_sections") or [] if str(item).strip()),
            "workflow": {
                "actions": list(base_actions),
                "sequence": list(base_actions),
                "edges": [list(edge) for edge in _linear_edges(base_actions)],
            },
            "match_rows": self._build_legacy_match_rows(
                rocket,
                evidence_sections=[str(item) for item in candidate.get("evidence_sections") or [] if str(item).strip()],
                event_item_codes=tuple(str(item) for item in candidate.get("event_item_codes") or [] if str(item).strip()),
            ),
        }
        try:
            reranked = rocket.rerank_extraction(
                extraction_row,
                reward_context,
                reward_mode="financial",
                max_candidates=int(getattr(rocket, "DEFAULT_MAX_CANDIDATES", 12)),
            )
        except Exception:
            return None

        selected = reranked.get("selected_plan") or {}
        base = reranked.get("base_plan") or {}
        selected_scores = dict((name, float(value)) for name, value in (selected.get("scores") or {}).items())
        candidate_summaries = [
            [
                str(plan.get("label") or ""),
                {
                    "source": str(plan.get("source") or ""),
                    "rank_seed": int(plan.get("rank_seed") or 0),
                    "actions": [str(item) for item in plan.get("actions") or [] if str(item).strip()],
                    "edges": [list(edge) for edge in plan.get("edges") or []],
                    "scores": [[name, float(value)] for name, value in (plan.get("scores") or {}).items()],
                },
            ]
            for plan in reranked.get("candidates") or []
        ]
        return {
            "score": float(reranked.get("selected_score") or 0.0),
            "workflow_stages": [str(item) for item in selected.get("actions") or [] if str(item).strip()],
            "base_workflow_stages": [str(item) for item in base.get("actions") or [] if str(item).strip()],
            "base_score": float(reranked.get("base_score") or 0.0),
            "selected_score": float(reranked.get("selected_score") or 0.0),
            "score_gain": float(reranked.get("score_gain") or 0.0),
            "selected_label": str(selected.get("label") or ""),
            "selected_source": str(selected.get("source") or ""),
            "changed": bool(reranked.get("changed", False)),
            "candidate_summaries": candidate_summaries,
            "selected_score_components": [[name, float(value)] for name, value in selected_scores.items()],
            "reward_mode": "financial",
            "financial_targets": list(getattr(reward_context, "financial_targets", ()) or ()),
            "reward_backend": "legacy_financial_real",
            "legacy_company_id": company_id,
        }

    @classmethod
    def _heuristic_rerank_candidate_workflow(
        cls,
        candidate: dict[str, object],
        workset: _BasketRocketWorkset,
    ) -> dict[str, object]:
        base_actions = _ordered_unique([str(stage) for stage in candidate.get("workflow_stages") or [] if str(stage).strip()])
        section_titles = [str(title).lower() for title in candidate.get("evidence_sections") or [] if str(title).strip()]
        section_text = " || ".join(section_titles)
        event_item_codes = tuple(str(item) for item in candidate.get("event_item_codes") or [] if str(item).strip())
        evidence_spans = list(candidate.get("evidence_spans") or [])
        candidate_source = str(candidate.get("candidate_source") or "basket")
        variants = cls._workflow_variants(
            base_actions,
            semantic_role=workset.semantic_role,
            section_text=section_text,
            event_item_codes=event_item_codes,
        )
        scored_variants = []
        for rank_seed, variant in enumerate(variants, start=1):
            scores = cls._score_workflow_variant(
                variant["workflow_stages"],
                semantic_role=workset.semantic_role,
                section_text=section_text,
                event_item_codes=event_item_codes,
                section_count=int(((candidate.get("score_basis") or {}).get("section_count") or 0)),
                evidence_spans=evidence_spans,
                variant_source=str(variant.get("source") or ""),
                candidate_source=candidate_source,
            )
            scored_variants.append(
                {
                    **variant,
                    "rank_seed": rank_seed,
                    "score_components": scores,
                    "selected_score": float(scores["total"]),
                    "edges": _linear_edges(variant["workflow_stages"]),
                }
            )
        base_variant = scored_variants[0]
        selected_variant = max(
            scored_variants,
            key=lambda row: (float(row["selected_score"]), -int(row["rank_seed"])),
        )
        candidate_summaries = [
            [
                str(variant["label"]),
                {
                    "source": str(variant["source"]),
                    "rank_seed": int(variant["rank_seed"]),
                    "actions": list(variant["workflow_stages"]),
                    "edges": list(variant["edges"]),
                    "scores": [[name, float(value)] for name, value in variant["score_components"].items()],
                },
            ]
            for variant in sorted(scored_variants, key=lambda row: (-float(row["selected_score"]), int(row["rank_seed"])))
        ]
        return {
            "score": float(selected_variant["selected_score"]),
            "workflow_stages": list(selected_variant["workflow_stages"]),
            "base_workflow_stages": list(base_actions),
            "base_score": float(base_variant["selected_score"]),
            "selected_score": float(selected_variant["selected_score"]),
            "score_gain": float(selected_variant["selected_score"] - base_variant["selected_score"]),
            "selected_label": str(selected_variant["label"]),
            "selected_source": str(selected_variant["source"]),
            "changed": list(selected_variant["workflow_stages"]) != list(base_actions),
            "candidate_summaries": candidate_summaries,
            "selected_score_components": [[name, float(value)] for name, value in selected_variant["score_components"].items()],
            "reward_mode": "financial",
            "financial_targets": list(cls._financial_targets_for_role(workset.semantic_role, event_item_codes)),
        }

    @staticmethod
    def _financial_targets_for_role(semantic_role: str, event_item_codes: tuple[str, ...]) -> tuple[str, ...]:
        targets = ["financial_alignment", "workflow_coherence", "local_evidence_support"]
        item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
        if semantic_role == "annual_anchor":
            targets.insert(0, "next_year_financial_alignment")
        if semantic_role == "event_patch" and "2" in item_prefixes:
            targets.insert(0, "event_financial_alignment")
        return tuple(dict.fromkeys(targets))

    def _get_legacy_rocket_module(self):
        if self._legacy_rocket_module is not None:
            return self._legacy_rocket_module
        rocket_src = _BASKET_ROOT / "src"
        if not rocket_src.exists():
            return None
        if str(rocket_src) not in sys.path:
            sys.path.insert(0, str(rocket_src))
        try:
            import rocket  # type: ignore
        except Exception:
            return None
        self._legacy_rocket_module = rocket
        return rocket

    def _get_basket_tenk_module(self):
        if self._basket_tenk_module is not None:
            return self._basket_tenk_module
        rocket_src = _BASKET_ROOT / "src"
        if not rocket_src.exists():
            return None
        if str(rocket_src) not in sys.path:
            sys.path.insert(0, str(rocket_src))
        try:
            import tenk  # type: ignore
        except Exception:
            return None
        self._basket_tenk_module = tenk
        return tenk

    def _get_legacy_reward_context(self, rocket):
        if self._legacy_reward_context is not None:
            return self._legacy_reward_context
        if not self.rocket_legacy_extractions_path.exists() or not self.rocket_legacy_macro_skills_path.exists():
            return None
        try:
            if self.rocket_legacy_extractions_path.suffix == ".gz":
                extraction_rows = _load_jsonl_rows(self.rocket_legacy_extractions_path)
            else:
                extraction_rows = rocket.load_jsonl(self.rocket_legacy_extractions_path)
            macro_support, max_macro_support = rocket.load_macro_support(self.rocket_legacy_macro_skills_path)
            panel_rows = rocket.load_csv_rows(self.rocket_panel_path) if self.rocket_panel_path.exists() else []
            financial_target_specs = (
                rocket.parse_financial_target_specs(",".join(self.rocket_financial_targets))
                if panel_rows and self.rocket_financial_targets
                else ()
            )
            ket_scorer = None
            if self.rocket_ket_model_path and self.rocket_ket_model_path.exists():
                try:
                    import plan_ket_scoring  # type: ignore
                except Exception:
                    plan_ket_scoring = None
                if plan_ket_scoring is not None:
                    ket_scorer = plan_ket_scoring.PlanKETScorer.load(
                        model_path=str(self.rocket_ket_model_path),
                        macro_skills_path=str(self.rocket_legacy_macro_skills_path),
                        skill_basis_path=str(self.rocket_ket_skill_basis_path or ""),
                        action_vocab_path=str(self.rocket_ket_action_vocab_path or ""),
                        device=self.rocket_ket_device,
                    )
            self._legacy_reward_context = rocket.build_reward_context(
                extraction_rows,
                macro_support=macro_support,
                max_macro_support=max_macro_support,
                panel_rows=panel_rows,
                financial_target_specs=financial_target_specs,
                financial_horizon=self.rocket_financial_horizon,
                ket_scorer=ket_scorer,
                ket_weight=self.rocket_ket_weight,
            )
            self._legacy_company_lookup = self._build_legacy_company_lookup(panel_rows)
        except Exception:
            self._legacy_reward_context = None
        return self._legacy_reward_context

    @staticmethod
    def _build_legacy_company_lookup(panel_rows: list[dict[str, str]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for row in panel_rows:
            company_id = str(row.get("company_id", "")).strip().lower()
            ticker = str(row.get("ticker", "")).strip().lower()
            if company_id:
                lookup[company_id] = company_id
                lookup[_normalized_name_token(company_id)] = company_id
            if ticker and company_id:
                lookup[ticker] = company_id
        return lookup

    def _resolve_legacy_company_id(self, *, company: str, ticker: str) -> str:
        lookup = self._legacy_company_lookup or {}
        ticker_key = str(ticker).strip().lower()
        company_key = str(company).strip().lower()
        normalized_company_key = _normalized_name_token(company_key)
        for key in (ticker_key, company_key, normalized_company_key):
            if key and key in lookup:
                return lookup[key]
        if ticker_key and ticker_key in {"mmm", "ko", "pg"}:
            return {"mmm": "3m", "ko": "coca_cola", "pg": "procter_gamble"}[ticker_key]
        return ""

    def _resolve_basket_company_id(self, *, company: str, ticker: str) -> str:
        resolved = self._resolve_legacy_company_id(company=company, ticker=ticker)
        if resolved:
            return resolved
        ticker_key = str(ticker).strip().lower()
        if ticker_key:
            return ticker_key
        return _slugify(company)

    def _build_basket_workflow_extractions(self, workset: _BasketRocketWorkset) -> list[dict[str, object]]:
        tenk = self._get_basket_tenk_module()
        if tenk is None:
            return []

        chunks_payload = _read_json(workset.outdir / "filing_chunks.json")
        chunks = list(chunks_payload.get("chunks") or []) if isinstance(chunks_payload, dict) else []
        if not chunks:
            return []

        extraction_rows: list[dict[str, object]] = []
        statement_index = 0
        for chunk in chunks:
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            sequence_hint = str(chunk.get("title") or "filing_chunk").strip() or "filing_chunk"
            filing_title = str(chunk.get("filing_title") or workset.workset_name)
            filing_path = str(chunk.get("filing_path") or "")
            form_type = str(chunk.get("form_type") or workset.form_type)
            semantic_role = str(chunk.get("semantic_role") or workset.semantic_role)
            event_item_codes = [str(item) for item in chunk.get("event_item_codes") or [] if str(item).strip()]
            company_name = next((filing.company for filing in workset.filings if filing.title == filing_title), workset.company)
            ticker = next((filing.ticker for filing in workset.filings if filing.title == filing_title), "")
            company_id = self._resolve_basket_company_id(company=company_name, ticker=ticker)
            topic = sequence_hint
            section_meta = tenk.infer_sec_section_metadata(topic, text, path=[topic, form_type, semantic_role])
            prompt = tenk.build_workflow_prompt(
                topic=str(section_meta.get("section_display") or topic),
                domain="sec_filing_chunk",
                question=f"What operational workflow is described in this {form_type} chunk?",
                statement=text,
            )
            extraction = tenk.heuristic_extract_plan(
                topic=str(section_meta.get("section_display") or topic),
                domain="sec_filing_chunk",
                question="Extract the workflow from the filing chunk.",
                statement=text,
                max_actions=8,
            )
            sequence = [str(item) for item in extraction.get("sequence", []) if str(item).strip()]
            if not sequence:
                continue
            statement_index += 1
            extraction_rows.append(
                {
                    "statement_id": f"{workset.workset_name}:chunk{statement_index:04d}",
                    "company": company_id,
                    "year": int(workset.filing_year) if str(workset.filing_year).isdigit() else 0,
                    "topic": topic,
                    "section_id": section_meta.get("section_id"),
                    "section_label": section_meta.get("section_label"),
                    "section_display": section_meta.get("section_display"),
                    "section_family": section_meta.get("section_family"),
                    "section_order": int(section_meta.get("section_order") or 0),
                    "section_confidence": float(section_meta.get("section_confidence") or 0.0),
                    "section_is_standard": bool(section_meta.get("section_is_standard")),
                    "section_subheading": section_meta.get("section_subheading"),
                    "section_path": list(section_meta.get("section_path") or []),
                    "raw_section": section_meta.get("raw_section"),
                    "domain": "sec_filing_chunk",
                    "path": [topic, form_type, semantic_role],
                    "question": "Extract the workflow from the filing chunk.",
                    "statement": text,
                    "source_file": filing_path,
                    "prompt": prompt,
                    "workflow": {
                        "actions": extraction.get("actions", []),
                        "edges": extraction.get("edges", []),
                        "sequence": extraction.get("sequence", []),
                    },
                    "rich_workflow": extraction.get("rich_workflow", {}),
                    "action_types": extraction.get("action_types", {}),
                    "rich_action_types": extraction.get("rich_action_types", {}),
                    "confidence": float(extraction.get("confidence") or 0.0),
                    "match_rows": extraction.get("match_rows", []),
                    "all_match_rows": extraction.get("all_match_rows", []),
                    "extractor": extraction.get("extractor", "heuristic_prompt_aligned"),
                    "source_mode": "sec_materialized_chunk",
                    "filing_title": filing_title,
                    "form_type": form_type,
                    "semantic_role": semantic_role,
                    "event_item_codes": event_item_codes,
                    "chunk_index": int(chunk.get("chunk_index") or statement_index),
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                }
            )
        return extraction_rows

    @staticmethod
    def _match_rows_to_evidence_spans(
        extraction_row: dict[str, object],
        *,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        section_title = str(
            extraction_row.get("section_display")
            or extraction_row.get("section_label")
            or extraction_row.get("topic")
            or ""
        )
        statement = " ".join(str(extraction_row.get("statement") or "").split())
        snippet = statement[:220] + ("..." if len(statement) > 220 else "")
        spans: list[dict[str, object]] = []
        for row in list(extraction_row.get("match_rows") or [])[:limit]:
            spans.append(
                {
                    "action": str(row.get("action") or ""),
                    "section_title": section_title,
                    "snippet": snippet,
                    "score": 1.0,
                    "match_basis": str(row.get("match_type") or ""),
                    "chunk_index": int(extraction_row.get("chunk_index") or 0),
                    "sentence_index": 0,
                }
            )
        return spans

    def _candidate_from_basket_extraction(self, extraction_row: dict[str, object]) -> dict[str, object]:
        sequence = [
            str(item)
            for item in ((extraction_row.get("workflow") or {}).get("sequence") or [])
            if str(item).strip()
        ]
        return {
            "candidate_id": str(extraction_row.get("statement_id") or ""),
            "statement_id": str(extraction_row.get("statement_id") or ""),
            "filing_title": str(extraction_row.get("filing_title") or extraction_row.get("topic") or ""),
            "workflow_stages": sequence,
            "semantic_role": str(extraction_row.get("semantic_role") or ""),
            "event_item_codes": list(extraction_row.get("event_item_codes") or []),
            "evidence_sections": [
                str(
                    extraction_row.get("section_display")
                    or extraction_row.get("section_label")
                    or extraction_row.get("topic")
                    or ""
                )
            ],
            "evidence_spans": self._match_rows_to_evidence_spans(extraction_row),
            "score_basis": {
                "stage_count": len(sequence),
                "section_count": 1,
                "evidence_span_count": len(list(extraction_row.get("match_rows") or [])),
            },
            "candidate_source": "basket_workflow_extraction",
            "topic": str(extraction_row.get("topic") or ""),
            "company": str(extraction_row.get("company") or ""),
            "year": int(extraction_row.get("year") or 0),
        }

    @staticmethod
    def _ranking_from_real_rerank(
        reranked: dict[str, object],
        *,
        extraction_row: dict[str, object],
        candidate_row: dict[str, object] | None,
        financial_targets: list[str],
    ) -> dict[str, object]:
        selected = dict(reranked.get("selected_plan") or {})
        base = dict(reranked.get("base_plan") or {})
        candidate_summaries = [
            [
                str(plan.get("label") or ""),
                {
                    "source": str(plan.get("source") or ""),
                    "rank_seed": int(plan.get("rank_seed") or 0),
                    "actions": [str(item) for item in plan.get("actions") or [] if str(item).strip()],
                    "edges": [list(edge) for edge in plan.get("edges") or []],
                    "scores": [
                        [name, float(value)]
                        for name, value in (plan.get("scores") or {}).items()
                    ],
                },
            ]
            for plan in reranked.get("candidates") or []
        ]
        evidence_spans = (
            list(candidate_row.get("evidence_spans") or [])
            if isinstance(candidate_row, dict)
            else []
        )
        if not evidence_spans:
            evidence_spans = BasketRocketBatchAgenticRunner._match_rows_to_evidence_spans(extraction_row)
        return {
            "candidate_id": str(extraction_row.get("statement_id") or ""),
            "statement_id": str(extraction_row.get("statement_id") or ""),
            "filing_title": str(extraction_row.get("filing_title") or extraction_row.get("topic") or ""),
            "topic": str(extraction_row.get("topic") or ""),
            "candidate_source": (
                str(candidate_row.get("candidate_source") or "")
                if isinstance(candidate_row, dict)
                else "basket_workflow_extraction"
            ),
            "score": float(reranked.get("selected_score") or 0.0),
            "workflow_stages": [str(item) for item in selected.get("actions") or [] if str(item).strip()],
            "selected_edges": [list(edge) for edge in selected.get("edges") or []],
            "base_workflow_stages": [str(item) for item in base.get("actions") or [] if str(item).strip()],
            "base_edges": [list(edge) for edge in base.get("edges") or []],
            "base_score": float(reranked.get("base_score") or 0.0),
            "selected_score": float(reranked.get("selected_score") or 0.0),
            "score_gain": float(reranked.get("score_gain") or 0.0),
            "selected_label": str(selected.get("label") or ""),
            "selected_source": str(selected.get("source") or ""),
            "changed": bool(reranked.get("changed", False)),
            "candidate_summaries": candidate_summaries,
            "selected_score_components": [
                [name, float(value)]
                for name, value in (selected.get("scores") or {}).items()
            ],
            "reward_mode": "financial",
            "financial_targets": financial_targets,
            "reward_backend": "legacy_financial_real",
            "extraction_backend": "basket_tenk_real",
            "evidence_spans": evidence_spans,
            "evidence_sections": (
                list(candidate_row.get("evidence_sections") or [])
                if isinstance(candidate_row, dict)
                else [str(extraction_row.get("section_display") or extraction_row.get("topic") or "")]
            ),
            "company": str(extraction_row.get("company") or ""),
            "year": int(extraction_row.get("year") or 0),
            "legacy_company_id": str(extraction_row.get("company") or ""),
        }

    @staticmethod
    def _build_legacy_match_rows(rocket, *, evidence_sections: list[str], event_item_codes: tuple[str, ...]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        position = 0
        section_texts = [str(section).strip().lower() for section in evidence_sections if str(section).strip()]
        for section_text in section_texts:
            for rule in getattr(rocket, "ACTION_RULES", ()):
                patterns = tuple(str(pattern).lower() for pattern in getattr(rule, "patterns", ()) if str(pattern).strip())
                if any(pattern in section_text for pattern in patterns):
                    rows.append(
                        {
                            "action": str(getattr(rule, "action", "")).strip(),
                            "stage_rank": int(getattr(rule, "stage_rank", 10_000)),
                            "position": position,
                            "match_type": "text_match",
                            "evidence": section_text,
                        }
                    )
                    position += 1
        prefix_action_map = {
            "1": ("invest", "acquire", "expand"),
            "2": ("price", "realize_revenue"),
            "5": ("optimize",),
            "8": ("market",),
        }
        for code in event_item_codes:
            for action in prefix_action_map.get(str(code).split(".", 1)[0], ()):
                rows.append(
                    {
                        "action": action,
                        "stage_rank": int(getattr(rocket, "ACTION_RANKS", {}).get(action, 10_000)),
                        "position": position,
                        "match_type": "topic_hint",
                        "evidence": str(code),
                    }
                )
                position += 1
        return rows

    @classmethod
    def _workflow_variants(
        cls,
        base_actions: list[str],
        *,
        semantic_role: str,
        section_text: str,
        event_item_codes: tuple[str, ...],
    ) -> list[dict[str, object]]:
        variants = [
            {"label": "base", "source": "basket", "workflow_stages": list(base_actions)},
        ]
        item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
        current = list(base_actions)

        if semantic_role == "annual_anchor":
            if "expand" not in current and any(word in section_text for word in ("growth", "strategy", "business", "segment")):
                variants.append(
                    {
                        "label": "add_expand",
                        "source": "local_insert",
                        "workflow_stages": _insert_action(current, "expand", before=("market", "price", "realize_revenue")),
                    }
                )
            macro = list(current)
            if "digitize" not in macro and any(word in section_text for word in ("digital", "cloud", "software", "ai", "data")):
                macro = _insert_action(macro, "digitize", before=("optimize", "expand", "market", "price", "realize_revenue"))
            if "market" not in macro:
                macro = _insert_action(macro, "market", before=("price", "realize_revenue"))
            if "price" not in macro:
                macro = _insert_action(macro, "price", before=("realize_revenue",))
            variants.append({"label": "panel_macro", "source": "macro_completion", "workflow_stages": macro})

        if semantic_role == "event_patch":
            local = list(current)
            if "1" in item_prefixes:
                local = _insert_action(local, "invest", before=("innovate", "digitize", "optimize", "expand", "market", "price", "realize_revenue"))
                local = _insert_action(local, "expand", before=("market", "price", "realize_revenue"))
            if "2" in item_prefixes:
                local = _insert_action(local, "price", before=("realize_revenue",))
            if "5" in item_prefixes:
                local = _insert_action(local, "optimize", before=("expand", "market", "price", "realize_revenue"))
            variants.append({"label": "panel_partial_macro_merge", "source": "macro_merge", "workflow_stages": local})

        if "optimize" not in current:
            variants.append(
                {
                    "label": "add_optimize",
                    "source": "local_insert",
                    "workflow_stages": _insert_action(current, "optimize", before=("expand", "market", "price", "realize_revenue")),
                }
            )
        if "realize_revenue" not in current:
            variants.append(
                {
                    "label": "add_realize_revenue",
                    "source": "local_insert",
                    "workflow_stages": _insert_action(current, "realize_revenue"),
                }
            )
        if "market" not in current and any(word in section_text for word in ("sales", "customer", "brand", "demand")):
            variants.append(
                {
                    "label": "add_market",
                    "source": "local_insert",
                    "workflow_stages": _insert_action(current, "market", before=("price", "realize_revenue")),
                }
            )
        if "price" not in current and (("2" in item_prefixes) or any(word in section_text for word in ("price", "pricing", "margin", "financial", "revenue"))):
            variants.append(
                {
                    "label": "add_price",
                    "source": "local_insert",
                    "workflow_stages": _insert_action(current, "price", before=("realize_revenue",)),
                }
            )

        deduped = []
        seen = set()
        for variant in variants:
            actions = tuple(_ordered_unique([str(action) for action in variant["workflow_stages"]]))
            key = (str(variant["label"]), actions)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "label": str(variant["label"]),
                    "source": str(variant["source"]),
                    "workflow_stages": list(actions),
                }
            )
        return deduped

    @staticmethod
    def _score_workflow_variant(
        actions: list[str],
        *,
        semantic_role: str,
        section_text: str,
        event_item_codes: tuple[str, ...],
        section_count: int,
        evidence_spans: list[dict[str, object]],
        variant_source: str,
        candidate_source: str,
    ) -> dict[str, float]:
        item_prefixes = {code.split(".", 1)[0] for code in event_item_codes}
        evidence_hits = 0
        for action in actions:
            if any(keyword in section_text for keyword in _ACTION_KEYWORDS.get(action, ())):
                evidence_hits += 1
                continue
            if any(prefix in item_prefixes for prefix in _ACTION_ITEM_PREFIXES.get(action, ())):
                evidence_hits += 1
        local = evidence_hits / max(len(actions), 1)

        target_actions = {
            "annual_anchor": ("optimize", "market", "realize_revenue"),
            "event_patch": ("invest", "price", "realize_revenue"),
            "quarterly_update": ("optimize", "price", "realize_revenue"),
        }.get(semantic_role, ("optimize", "realize_revenue"))
        target_hits = sum(1 for action in target_actions if action in actions)
        terminal_bonus = 1.0 if actions and actions[-1] == "realize_revenue" else 0.0
        financial = (target_hits + terminal_bonus) / (len(target_actions) + 1.0)

        canonical_pairs = sum(
            1
            for left, right in zip(actions, actions[1:])
            if _LEGACY_ACTION_ORDER.index(left) < _LEGACY_ACTION_ORDER.index(right)
        )
        struct = canonical_pairs / max(len(actions) - 1, 1)

        templates = _MACRO_TEMPLATES.get(semantic_role, _MACRO_TEMPLATES["filing_update"])
        macro = max(
            (
                sum(1 for action in template if action in actions) / max(len(template), 1)
                for template in templates
            ),
            default=0.0,
        )
        panel = sum(1 for action in actions if action in {"expand", "market", "price", "realize_revenue"}) / max(len(actions), 1)
        simp = _clamp_score(1.0 - (abs(len(actions) - 4) / 4.0))
        text = _clamp_score((section_count / 6.0) + 0.35 * local)
        supported_actions = {str(span.get("action") or "") for span in evidence_spans if str(span.get("action") or "").strip()}
        evidence_density = sum(1 for action in actions if action in supported_actions) / max(len(actions), 1)
        span_bonus = _clamp_score(len(evidence_spans) / 6.0)
        source_bias = 0.0
        if candidate_source == "grounded_evidence_order":
            source_bias += 0.06
        elif candidate_source == "grounded_chunk_focus":
            source_bias += 0.04
        elif candidate_source == "grounded_filing_backbone":
            source_bias += 0.02
        if variant_source == "macro_completion":
            source_bias -= 0.08 if evidence_density < 0.85 else 0.01
        elif variant_source == "macro_merge":
            source_bias -= 0.04 if evidence_density < 0.8 else 0.0
        elif variant_source == "local_insert":
            source_bias += 0.02 if evidence_density >= 0.75 else -0.02
        total = (
            0.22 * financial
            + 0.18 * local
            + 0.12 * macro
            + 0.12 * panel
            + 0.12 * struct
            + 0.08 * simp
            + 0.08 * text
            + 0.06 * evidence_density
            + 0.02 * span_bonus
            + source_bias
        )
        return {
            "financial": round(financial, 6),
            "local": round(local, 6),
            "macro": round(macro, 6),
            "panel": round(panel, 6),
            "struct": round(struct, 6),
            "simp": round(simp, 6),
            "text": round(text, 6),
            "evidence_density": round(evidence_density, 6),
            "span_bonus": round(span_bonus, 6),
            "source_bias": round(source_bias, 6),
            "total": round(total, 6),
        }

    def _run_psr_modeling_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        rankings_payload = json.loads((workset.outdir / "rocket_rankings.json").read_text(encoding="utf-8"))
        top_ranking = next(iter(rankings_payload.get("rankings") or []), None)
        reward_backend = str(
            rankings_payload.get("reward_backend")
            or (top_ranking.get("reward_backend") if top_ranking else "heuristic")
        )
        model = {
            "workset_name": workset.workset_name,
            "model_family": "PSR",
            "status": "legacy_financial_alignment" if reward_backend == "legacy_financial_real" else "heuristic_financial_alignment",
            "observed_inputs": [filing.title for filing in workset.filings],
            "latent_states": list(top_ranking.get("workflow_stages") or []) if top_ranking else ["collect", "review", "compare", "escalate"],
            "top_ranked_candidate": top_ranking,
            "semantic_role": workset.semantic_role,
            "transition_note": (
                "Predictive-state summary derived from the original panel-driven ROCKET candidate."
                if reward_backend == "legacy_financial_real"
                else "Heuristic predictive-state summary derived from the financially reranked ROCKET candidate."
            ),
        }
        path = workset.outdir / "psr_models.json"
        _write_json(path, model)
        return "completed", (str(path),), "wrote scaffolded PSR state summary"

    def _run_workflow_reporting_agent(self, workset: _BasketRocketWorkset) -> tuple[str, tuple[str, ...], str]:
        rankings_payload = json.loads((workset.outdir / "rocket_rankings.json").read_text(encoding="utf-8"))
        psr_payload = json.loads((workset.outdir / "psr_models.json").read_text(encoding="utf-8"))
        top = next(iter(rankings_payload.get("rankings") or []), None)
        evidence_lines = []
        for span in list(top.get("evidence_spans") or [])[:5] if top else []:
            evidence_lines.append(
                f"  - {span.get('action', 'evidence')}: {span.get('section_title', '')} :: {span.get('snippet', '')}"
            )
        lines = [
            f"# BASKET/ROCKET Scaffold Report: {workset.company} {workset.form_type} {workset.filing_year}",
            "",
            f"- Workset: `{workset.workset_name}`",
            f"- Filings staged: {len(workset.filings)}",
            f"- Form type: `{workset.form_type}`",
            f"- Semantic role: `{workset.semantic_role}`",
            f"- Top candidate: `{top['candidate_id']}`" if top else "- Top candidate: none",
            f"- Workflow stages: {', '.join(top['workflow_stages'])}" if top else "- Workflow stages: none",
            f"- Base stages: {', '.join(top['base_workflow_stages'])}" if top and top.get("base_workflow_stages") else "- Base stages: none",
            f"- Selected source: `{top['selected_source']}` · label `{top['selected_label']}`" if top else "- Selected source: none",
            f"- Score gain: `{top['score_gain']:.3f}`" if top else "- Score gain: none",
            f"- Reward backend: `{top['reward_backend']}`" if top and top.get("reward_backend") else "- Reward backend: heuristic",
            f"- PSR status: `{psr_payload.get('status', 'unknown')}`",
            f"- Candidate source: `{top.get('candidate_source', 'unknown')}`" if top else "- Candidate source: none",
            "",
            "This report is a grounded FF2 ingress output. It groups SEC filings into",
            "company/year/form worksets, extracts workflow candidates from chunked filing text and evidence spans, and then applies",
            "ROCKET reranking against the original panel-driven financial outcome data when available.",
        ]
        if evidence_lines:
            lines.extend(["", "Top grounded evidence spans:"])
            lines.extend(evidence_lines)
        path = workset.outdir / "workflow_report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "completed", (str(path),), "wrote scaffolded workflow report"

    def _filing_text(self, filing: MaterializedSECFiling) -> str:
        if filing.text_path and Path(filing.text_path).exists():
            return Path(filing.text_path).read_text(encoding="utf-8")
        path = Path(filing.filing_path)
        if path.exists():
            payload = path.read_bytes()
            return BasketRocketSECAgenticRunner._normalize_filing_text(payload, "html")
        return ""

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int = 3000) -> list[dict[str, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return [{"title": "full_filing", "text": ""}]

        sections: list[dict[str, str]] = []
        current_title = "full_filing"
        current_lines: list[str] = []
        for line in lines:
            if BasketRocketBatchAgenticRunner._looks_like_heading(line):
                if current_lines:
                    sections.append({"title": current_title, "text": "\n".join(current_lines)})
                    current_lines = []
                current_title = line[:160]
                continue
            current_lines.append(line)
        if current_lines:
            sections.append({"title": current_title, "text": "\n".join(current_lines)})
        if not sections:
            sections = [{"title": "full_filing", "text": "\n".join(lines)}]

        bounded: list[dict[str, str]] = []
        for section in sections:
            text_payload = section["text"]
            if len(text_payload) <= max_chars:
                bounded.append(section)
                continue
            for start in range(0, len(text_payload), max_chars):
                bounded.append(
                    {
                        "title": section["title"],
                        "text": text_payload[start : start + max_chars],
                    }
                )
        return bounded

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        if len(line) < 4 or len(line) > 120:
            return False
        lowered = line.lower()
        if lowered.startswith("item ") or lowered.startswith("part "):
            return True
        letters = [char for char in line if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        return uppercase_ratio >= 0.8 and len(letters) >= 4


class BasketRocketSECAgenticRunner:
    """SEC-specific ingress runner for the first BASKET/ROCKET FF2 workflow."""

    def __init__(self, config: BasketRocketSECAgenticConfig) -> None:
        self.config = config.resolved()
        if not self.config.query:
            raise ValueError("A non-empty SEC acquisition query is required.")
        self.discovery_outdir = self.config.outdir / "sec_discovery"
        self.logs_dir = self.config.outdir / "agent_logs"
        self.materialized_filings_dir = self.config.outdir / "materialized_filings"
        self.filing_manifest_path = self.config.outdir / "materialized_filing_manifest.json"
        self.company_context_path = self.config.outdir / "company_context" / "company_year_form_index.json"
        self.batch_outdir = self.config.outdir / "workflow_batches"
        self.batch_live_gui_path = self.batch_outdir / "basket_rocket_gui.html"
        self.summary_path = self.config.outdir / "basket_rocket_run_summary.json"

    def run(self) -> BasketRocketSECRunResult:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self._write_launch_gui(status="starting", stage_label="Discovering SEC filings", note=(
            "BAFFLE is retrieving candidate SEC filings and preparing the BASKET/ROCKET workset."
        ))
        discovery_result = self._run_sec_discovery_agent()
        selected_filings = discovery_result.selected_documents
        self._write_launch_gui(
            status="running",
            stage_label="Materializing filings",
            note="BAFFLE has selected SEC filings and is now materializing the filing bundle.",
            selected_filings=selected_filings,
        )
        materialized_filings: tuple[MaterializedSECFiling, ...] = ()
        batch_result: BasketRocketBatchRunResult | None = None
        if not self.config.discovery_only:
            materialized_filings = self._run_filing_materialization_agent(selected_filings)
            self._write_company_context(materialized_filings)
            self._write_launch_gui(
                status="running",
                stage_label="Launching BASKET/ROCKET analysis",
                note="Materialized filings are ready. BAFFLE is grouping worksets and starting workflow extraction and reranking.",
                selected_filings=selected_filings,
                materialized_filings=materialized_filings,
            )
            batch_result = BasketRocketBatchAgenticRunner(
                filings=materialized_filings,
                outdir=self.batch_outdir,
                filing_manifest_path=self.filing_manifest_path,
                company_context_path=self.company_context_path,
                rocket_reward_source=self.config.rocket_reward_source,
                rocket_legacy_extractions_path=self.config.rocket_legacy_extractions_path,
                rocket_legacy_macro_skills_path=self.config.rocket_legacy_macro_skills_path,
                rocket_panel_path=self.config.rocket_panel_path,
                rocket_financial_targets=self.config.rocket_financial_targets,
                rocket_financial_horizon=self.config.rocket_financial_horizon,
                rocket_ket_model_path=self.config.rocket_ket_model_path,
                rocket_ket_skill_basis_path=self.config.rocket_ket_skill_basis_path,
                rocket_ket_action_vocab_path=self.config.rocket_ket_action_vocab_path,
                rocket_ket_device=self.config.rocket_ket_device,
                rocket_ket_weight=self.config.rocket_ket_weight,
                dry_run=self.config.dry_run,
                enable_corpus_synthesis=self.config.enable_corpus_synthesis,
            ).run()

        result = BasketRocketSECRunResult(
            query_plan=discovery_result.query_plan,
            selected_filings=selected_filings,
            materialized_filings=materialized_filings,
            batch_records=batch_result.records if batch_result else (),
            discovery_summary_path=discovery_result.summary_path,
            filing_manifest_path=self.filing_manifest_path if self.filing_manifest_path.exists() else None,
            company_context_path=self.company_context_path if self.company_context_path.exists() else None,
            batch_workset_index_path=batch_result.workset_index_path if batch_result else None,
            batch_summary_path=batch_result.summary_path if batch_result else None,
            batch_live_gui_path=(
                batch_result.live_gui_path if batch_result else (self.batch_live_gui_path if self.batch_live_gui_path.exists() else None)
            ),
            batch_visualization_index_path=batch_result.visualizations.index_path if batch_result and batch_result.visualizations else None,
            batch_visualization_summary_path=(
                batch_result.visualizations.summary_path if batch_result and batch_result.visualizations else None
            ),
            corpus_synthesis_summary_path=(
                batch_result.corpus_synthesis.summary_path if batch_result and batch_result.corpus_synthesis else None
            ),
            corpus_synthesis_dashboard_path=(
                batch_result.corpus_synthesis.dashboard_path if batch_result and batch_result.corpus_synthesis else None
            ),
            summary_path=self.summary_path,
        )
        _write_json(
            self.summary_path,
            {
                "query_plan": asdict(result.query_plan),
                "discovery_summary_path": str(result.discovery_summary_path),
                "selected_filings": [asdict(item) for item in result.selected_filings],
                "materialized_filings": [asdict(item) for item in result.materialized_filings],
                "discovery_only": self.config.discovery_only,
                "dry_run": self.config.dry_run,
                "retrieval_backend": "sec",
                "filing_manifest_path": str(result.filing_manifest_path) if result.filing_manifest_path else None,
                "company_context_path": str(result.company_context_path) if result.company_context_path else None,
                "batch_workset_index_path": (
                    str(result.batch_workset_index_path) if result.batch_workset_index_path else None
                ),
                "batch_summary_path": str(result.batch_summary_path) if result.batch_summary_path else None,
                "batch_live_gui_path": (
                    str(result.batch_live_gui_path) if result.batch_live_gui_path else None
                ),
                "batch_visualization_index_path": (
                    str(result.batch_visualization_index_path) if result.batch_visualization_index_path else None
                ),
                "batch_visualization_summary_path": (
                    str(result.batch_visualization_summary_path) if result.batch_visualization_summary_path else None
                ),
                "corpus_synthesis_summary_path": (
                    str(result.corpus_synthesis_summary_path) if result.corpus_synthesis_summary_path else None
                ),
                "corpus_synthesis_dashboard_path": (
                    str(result.corpus_synthesis_dashboard_path) if result.corpus_synthesis_dashboard_path else None
                ),
                "batch_record_count": len(result.batch_records),
            },
        )
        return result

    def _write_launch_gui(
        self,
        *,
        status: str,
        stage_label: str,
        note: str,
        selected_filings: tuple[DiscoveredDocument, ...] = (),
        materialized_filings: tuple[MaterializedSECFiling, ...] = (),
    ) -> None:
        self.batch_outdir.mkdir(parents=True, exist_ok=True)

        def esc(value: object) -> str:
            return html.escape(str(value))

        refresh_meta = '<meta http-equiv="refresh" content="5">' if status != "complete" else ""
        selected_cards = "".join(
            '<article class="filing-card">'
            f'<div class="eyebrow">{esc(item.metadata.get("form", "SEC filing"))} · {esc(item.metadata.get("filing_date", item.year))}</div>'
            f'<h2>{esc(item.metadata.get("company", item.title))}</h2>'
            f'<p>{esc(item.title)}</p>'
            f'<div class="chip-row"><span class="chip neutral">{esc(item.retrieval_backend)}</span><span class="chip neutral">{esc(item.identifier or "identifier pending")}</span></div>'
            "</article>"
            for item in selected_filings[:6]
        )
        materialized_cards = "".join(
            '<article class="filing-card">'
            f'<div class="eyebrow">{esc(item.form_type)} · {esc(item.filing_date)}</div>'
            f'<h2>{esc(item.company)}</h2>'
            f'<p>{esc(item.title)}</p>'
            f'<div class="chip-row"><span class="chip success">{esc(item.semantic_role.replace("_", " "))}</span><span class="chip neutral">{esc(item.workset_name)}</span></div>'
            "</article>"
            for item in materialized_filings[:6]
        )

        self.batch_live_gui_path.write_text(
            f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {refresh_meta}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BAFFLE BASKET/ROCKET GUI</title>
  <style>
    :root {{
      --ink: #172432;
      --muted: #5c6d7d;
      --paper: #f7f1e7;
      --card: rgba(255, 252, 247, 0.97);
      --line: #d9cdbf;
      --accent: #0b6e4f;
      --accent-soft: #dff4ea;
      --mixed: #9a5b16;
      --mixed-soft: #f9ead0;
      --shadow: 0 18px 44px rgba(23, 36, 50, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", serif;
      background:
        radial-gradient(circle at top right, rgba(11,110,79,0.10), transparent 24%),
        radial-gradient(circle at left center, rgba(154,91,22,0.08), transparent 22%),
        linear-gradient(180deg, #fbf6ee 0%, #efe4d3 100%);
    }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 48px; }}
    .hero, .panel {{
      background: var(--card);
      border: 1px solid rgba(217, 205, 191, 0.96);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    h1 {{ margin: 10px 0 12px; font-size: clamp(32px, 4vw, 54px); line-height: 0.98; }}
    h2 {{ margin: 6px 0 8px; font-size: 24px; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .metrics, .two-up, .filing-grid {{
      display: grid;
      gap: 16px;
    }}
    .metrics {{
      margin-top: 22px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .metric, .filing-card {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
    }}
    .metric-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{ margin-top: 8px; font-size: 30px; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      border: 1px solid transparent;
      background: #edf2f7;
    }}
    .chip.success {{ background: var(--accent-soft); color: var(--accent); }}
    .chip.mixed {{ background: var(--mixed-soft); color: var(--mixed); }}
    .chip.neutral {{ background: #edf2f7; color: var(--ink); }}
    .two-up {{
      margin-top: 24px;
      grid-template-columns: 1fr 1fr;
    }}
    .panel {{ padding: 22px; }}
    .filing-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 14px;
    }}
    .empty {{
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .metrics, .two-up, .filing-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">BAFFLE Financial Workflow Reader</div>
      <h1>BASKET/ROCKET 10-K GUI</h1>
      <p>{esc(note)}</p>
      <div class="chip-row">
        <span class="chip {'success' if status == 'complete' else 'mixed'}">{esc(status.replace("_", " "))}</span>
        <span class="chip mixed">{esc(stage_label)}</span>
        <span class="chip neutral">{esc(self.config.query)}</span>
      </div>
      <div class="metrics">
        <div class="metric"><div class="metric-label">Selected Filings</div><div class="metric-value">{esc(len(selected_filings))}</div></div>
        <div class="metric"><div class="metric-label">Materialized</div><div class="metric-value">{esc(len(materialized_filings))}</div></div>
        <div class="metric"><div class="metric-label">Target</div><div class="metric-value">{esc(self.config.target_filings)}</div></div>
      </div>
    </section>
    <div class="two-up">
      <section class="panel">
        <div class="eyebrow">Discovery Preview</div>
        <h2>Selected SEC Filings</h2>
        <div class="filing-grid">
          {selected_cards or '<div class="empty">Candidate filings will appear here as discovery completes.</div>'}
        </div>
      </section>
      <section class="panel">
        <div class="eyebrow">Ingress Preview</div>
        <h2>Materialized Workset Inputs</h2>
        <div class="filing-grid">
          {materialized_cards or '<div class="empty">Materialized filings and workset assignments will appear here before BASKET/ROCKET analysis starts.</div>'}
        </div>
      </section>
    </div>
  </div>
</body>
</html>
""",
            encoding="utf-8",
        )

    def _log(self, agent_name: str, lines: list[str]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"{agent_name}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_sec_discovery_agent(self) -> DemocritusQueryRunResult:
        runner = DemocritusQueryAgenticRunner(
            DemocritusQueryAgenticConfig(
                query=self.config.query,
                outdir=self.discovery_outdir,
                target_documents=self.config.target_filings,
                retrieval_backend="sec",
                retrieval_user_agent=_resolve_sec_user_agent(self.config.retrieval_user_agent),
                retrieval_timeout_seconds=self.config.retrieval_timeout_seconds,
                discovery_only=True,
                dry_run=self.config.dry_run,
                sec_form_types=self.config.sec_form_types,
                sec_company_limit=self.config.sec_company_limit,
            )
        )
        return runner.run()

    def _run_filing_materialization_agent(
        self,
        filings: tuple[DiscoveredDocument, ...],
    ) -> tuple[MaterializedSECFiling, ...]:
        self.materialized_filings_dir.mkdir(parents=True, exist_ok=True)
        anchors_by_company: dict[str, list[str]] = {}
        for filing in filings:
            company = filing.metadata.get("company", "unknown_company")
            form_type = filing.metadata.get("form", "UNKNOWN")
            filing_year = _safe_year(filing.metadata.get("filing_date", filing.year) or filing.year)
            if str(form_type).upper() == "10-K":
                anchors_by_company.setdefault(company, []).append(filing_year)
        normalized_anchors = {
            company: sorted({year for year in years if year.isdigit()}, key=int)
            for company, years in anchors_by_company.items()
        }
        materialized: list[MaterializedSECFiling] = []
        failures: list[str] = []
        log_lines = [
            f"[QUERY] {self.config.query}",
            f"[TARGET_DIR] {self.materialized_filings_dir}",
            f"[TARGET_FILINGS] {self.config.target_filings}",
            f"[DRY_RUN] {self.config.dry_run}",
        ]
        for index, filing in enumerate(filings, start=1):
            company = filing.metadata.get("company", "unknown_company")
            ticker = filing.metadata.get("ticker", "unknown")
            cik = filing.metadata.get("cik", "")
            accession = filing.identifier or filing.metadata.get("accession_number", "")
            form_type = filing.metadata.get("form", "UNKNOWN")
            filing_date = filing.metadata.get("filing_date", filing.year)
            filing_year = _safe_year(filing_date or filing.year)
            semantic_role = _form_semantic_role(form_type)
            anchor_year = self._anchor_year_for_company(company, filing_year, normalized_anchors)
            workset_name = _slugify(f"{company}_{anchor_year}_{form_type}")
            base_name = f"{index:04d}_{_slugify(ticker)}_{_form_slug(form_type)}_{_slugify(filing_date or filing_year)}"
            suffix = {
                "html": ".html",
                "txt": ".txt",
                "pdf": ".pdf",
            }.get(filing.document_format, ".dat")
            target_dir = self.materialized_filings_dir / workset_name
            filing_path = target_dir / f"{base_name}{suffix}"
            text_path = target_dir / f"{base_name}.txt"
            target_dir.mkdir(parents=True, exist_ok=True)

            if self.config.dry_run:
                materialized.append(
                    MaterializedSECFiling(
                        title=filing.title,
                        filing_path=str(filing_path),
                        text_path=str(text_path),
                        source_url=filing.download_url or filing.url or "",
                        retrieval_backend=filing.retrieval_backend,
                        company=company,
                        ticker=ticker,
                        cik=cik,
                        accession_number=accession,
                        form_type=form_type,
                        filing_date=filing_date,
                        filing_year=filing_year,
                        anchor_year=anchor_year,
                        semantic_role=semantic_role,
                        event_item_codes=(),
                        workset_name=workset_name,
                        status="planned",
                    )
                )
                log_lines.append(f"[PLAN] {filing.title} -> {filing_path}")
                continue

            if not filing.download_url:
                failures.append(f"{filing.title}: missing download_url")
                log_lines.append(f"[SKIP] missing download_url for {filing.title}")
                continue
            try:
                payload = self._download_filing(filing.download_url, referer=filing.url)
            except Exception as exc:
                failures.append(f"{filing.title}: {exc}")
                log_lines.append(f"[SKIP] download failed for {filing.title}: {exc}")
                continue
            filing_path.write_bytes(payload)
            normalized_text = self._normalize_filing_text(payload, filing.document_format)
            event_item_codes = _extract_sec_item_codes(normalized_text)
            if normalized_text:
                text_path.write_text(normalized_text, encoding="utf-8")
                stored_text_path = str(text_path)
            else:
                stored_text_path = None
            materialized.append(
                MaterializedSECFiling(
                    title=filing.title,
                    filing_path=str(filing_path),
                    text_path=stored_text_path,
                    source_url=filing.download_url,
                    retrieval_backend=filing.retrieval_backend,
                    company=company,
                    ticker=ticker,
                    cik=cik,
                    accession_number=accession,
                    form_type=form_type,
                    filing_date=filing_date,
                    filing_year=filing_year,
                    anchor_year=anchor_year,
                    semantic_role=semantic_role,
                    event_item_codes=event_item_codes,
                    workset_name=workset_name,
                )
            )
            log_lines.append(f"[DOWNLOAD] {filing.download_url} -> {filing_path}")

        if not materialized:
            self._log("filing_materialization_agent", log_lines + ["[ERROR] no filings materialized"])
            preview = "; ".join(failures[:3]) if failures else "no SEC filings were materialized"
            raise RuntimeError(f"Failed to materialize any SEC filings. First issues: {preview}")

        _write_json(self.filing_manifest_path, [asdict(item) for item in materialized])
        self._log("filing_materialization_agent", log_lines)
        return tuple(materialized)

    @staticmethod
    def _anchor_year_for_company(
        company: str,
        filing_year: str,
        anchors_by_company: dict[str, list[str]],
    ) -> str:
        anchor_years = anchors_by_company.get(company, [])
        if filing_year in anchor_years:
            return filing_year
        if filing_year.isdigit() and anchor_years:
            filing_year_int = int(filing_year)
            return min(anchor_years, key=lambda value: abs(int(value) - filing_year_int))
        return filing_year

    def _write_company_context(self, materialized_filings: tuple[MaterializedSECFiling, ...]) -> None:
        grouped: dict[tuple[str, str, str], list[MaterializedSECFiling]] = {}
        for filing in materialized_filings:
            grouped.setdefault((filing.company, filing.filing_year, filing.form_type), []).append(filing)
        payload = []
        for company, filing_year, form_type in sorted(grouped):
            filings = tuple(sorted(grouped[(company, filing_year, form_type)], key=lambda item: item.filing_date))
            payload.append(
                {
                    "workset_name": filings[0].workset_name,
                    "company": company,
                    "ticker": filings[0].ticker,
                    "filing_year": filing_year,
                    "form_type": form_type,
                    "semantic_role": filings[0].semantic_role,
                    "anchor_year": filings[0].anchor_year,
                    "filing_count": len(filings),
                    "event_item_codes": sorted({code for filing in filings for code in filing.event_item_codes}),
                    "filings": [asdict(filing) for filing in filings],
                }
            )
        _write_json(self.company_context_path, payload)
        self._log(
            "company_context_agent",
            [
                f"[WORKSETS] {len(payload)}",
                *[
                    f"[WORKSET] {item['workset_name']} company={item['company']} year={item['filing_year']} "
                    f"form={item['form_type']} filings={item['filing_count']}"
                    for item in payload
                ],
            ],
        )

    def _download_filing(self, url: str, *, referer: str | None = None) -> bytes:
        candidate_urls = [url]
        if url.startswith("http://"):
            candidate_urls.insert(0, "https://" + url[len("http://") :])
        last_error: Exception | None = None
        for candidate_url in dict.fromkeys(candidate_urls):
            headers = {
                "User-Agent": _resolve_sec_user_agent(self.config.retrieval_user_agent),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            }
            if referer:
                headers["Referer"] = referer
            request = Request(candidate_url, headers=headers)
            try:
                with urlopen(request, timeout=self.config.retrieval_timeout_seconds) as response:
                    payload = response.read()
            except (HTTPError, URLError) as exc:
                last_error = exc
                continue
            if not payload or not payload.strip():
                last_error = RuntimeError(f"downloaded filing from {candidate_url!r} is empty")
                continue
            return payload
        if last_error is None:
            raise RuntimeError(f"Could not download {url}")
        raise last_error

    @staticmethod
    def _normalize_filing_text(payload: bytes, document_format: str) -> str:
        text = payload.decode("utf-8", errors="ignore")
        if document_format == "html":
            text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
            text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
            text = re.sub(r"(?s)<!--.*?-->", " ", text)
            text = re.sub(r"(?i)<br\s*/?>", "\n", text)
            text = re.sub(r"(?i)</p\s*>", "\n", text)
            text = re.sub(r"(?i)</div\s*>", "\n", text)
            text = re.sub(r"(?i)</tr\s*>", "\n", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = html.unescape(text)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        filtered = [line for line in lines if line]
        return "\n".join(filtered)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SEC-backed BASKET/ROCKET ingress scaffold.")
    parser.add_argument("--query", default="", help="Natural-language SEC filing request.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--target-filings", type=int, default=10)
    parser.add_argument(
        "--retrieval-user-agent",
        default="FunctorFlow_v2/0.1 (agentic SEC retrieval; local use)",
        help=(
            "SEC identity string. This should include contact info, "
            "for example 'Your Name your_email@example.com'."
        ),
    )
    parser.add_argument("--retrieval-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sec-form", action="append", default=[], help="SEC form to request, e.g. 10-K. May be repeated.")
    parser.add_argument("--sec-company-limit", type=int, default=3)
    parser.add_argument("--rocket-reward-source", choices=("legacy", "heuristic"), default="legacy")
    parser.add_argument("--rocket-legacy-extractions", default=str(_DEFAULT_LEGACY_EXTRACTIONS_PATH))
    parser.add_argument("--rocket-legacy-macro-skills", default=str(_DEFAULT_LEGACY_MACRO_SKILLS_PATH))
    parser.add_argument("--rocket-panel", default=str(_DEFAULT_LEGACY_PANEL_PATH))
    parser.add_argument(
        "--rocket-financial-target",
        action="append",
        default=[],
        help="Financial target spec passed into the original ROCKET reranker, e.g. +revenue_yoy or -debt_to_assets.",
    )
    parser.add_argument("--rocket-financial-horizon", choices=("same_year", "next_year", "next_year_delta"), default="next_year")
    parser.add_argument("--rocket-ket-model", default="")
    parser.add_argument("--rocket-ket-skill-basis", default="")
    parser.add_argument("--rocket-ket-action-vocab", default="")
    parser.add_argument("--rocket-ket-device", default="cpu")
    parser.add_argument("--rocket-ket-weight", type=float, default=0.20)
    return parser.parse_args()


def _resolve_query_for_main(args: argparse.Namespace) -> str:
    query = " ".join(str(args.query or "").split()).strip()
    if query:
        return query
    artifact_path = Path(args.outdir).resolve() / "workflow_batches" / "corpus_synthesis" / "basket_rocket_corpus_synthesis.html"
    with DashboardQueryLauncher(
        DashboardQueryLauncherConfig(
            title="BASKET/ROCKET SEC Dashboard",
            subtitle=(
                "Describe the SEC filing batch you want BAFFLE to assemble and analyze. "
                "Natural-language 10-K and 10-Q requests work well here."
            ),
            query_label="SEC filing query",
            query_placeholder=(
                "Find me 10 recent 10-K filings for Adobe\n"
                "or\n"
                "Find me 8 recent 10-Q filings for Nvidia and AMD"
            ),
            submit_label="Launch BASKET/ROCKET Run",
            waiting_message=(
                "The query has been captured. BAFFLE will retrieve the filings, extract individual workflows, and open the synthesized BASKET/ROCKET result when the run finishes."
            ),
            artifact_path=artifact_path,
        )
    ) as launcher:
        return launcher.wait_for_submission()


def _open_dashboard_artifact(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=1, autoraise=True)
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    query = _resolve_query_for_main(args)
    runner = BasketRocketSECAgenticRunner(
        BasketRocketSECAgenticConfig(
            query=query,
            outdir=Path(args.outdir),
            target_filings=args.target_filings,
            retrieval_user_agent=args.retrieval_user_agent,
            retrieval_timeout_seconds=args.retrieval_timeout_seconds,
            sec_form_types=tuple(args.sec_form) if args.sec_form else ("10-K", "10-Q"),
            sec_company_limit=args.sec_company_limit,
            rocket_reward_source=args.rocket_reward_source,
            rocket_legacy_extractions_path=Path(args.rocket_legacy_extractions),
            rocket_legacy_macro_skills_path=Path(args.rocket_legacy_macro_skills),
            rocket_panel_path=Path(args.rocket_panel),
            rocket_financial_targets=tuple(args.rocket_financial_target)
            if args.rocket_financial_target
            else (
                "+revenue_yoy",
                "+operating_margin",
                "+free_cash_flow_margin",
                "+return_on_assets",
                "-debt_to_assets",
            ),
            rocket_financial_horizon=args.rocket_financial_horizon,
            rocket_ket_model_path=Path(args.rocket_ket_model) if args.rocket_ket_model else None,
            rocket_ket_skill_basis_path=Path(args.rocket_ket_skill_basis) if args.rocket_ket_skill_basis else None,
            rocket_ket_action_vocab_path=Path(args.rocket_ket_action_vocab) if args.rocket_ket_action_vocab else None,
            rocket_ket_device=args.rocket_ket_device,
            rocket_ket_weight=args.rocket_ket_weight,
            discovery_only=args.discovery_only,
            dry_run=args.dry_run,
        )
    )
    result = runner.run()
    _open_dashboard_artifact(
        result.corpus_synthesis_dashboard_path
        or result.batch_visualization_index_path
        or result.batch_live_gui_path
    )
    print(
        json.dumps(
            {
                "query": result.query_plan.query,
                "selected_filings": len(result.selected_filings),
                "materialized_filings": len(result.materialized_filings),
                "batch_records": len(result.batch_records),
                "corpus_synthesis_dashboard_path": (
                    str(result.corpus_synthesis_dashboard_path) if result.corpus_synthesis_dashboard_path else None
                ),
                "summary_path": str(result.summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
