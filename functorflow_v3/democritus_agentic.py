"""Agentic Democritus runner built on top of the current public pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from .agentic_workflows import AgenticWorkflow, AgentSpec, ArtifactSpec, build_agentic_workflow
from .repo_layout import resolve_democritus_root, workspace_root


_MATPLOTLIB_RENDER_LOCK = threading.Lock()


def _workspace_root() -> Path:
    return workspace_root()


def _democritus_repo_root() -> Path:
    return resolve_democritus_root()


def _default_topics_path() -> Path:
    return _workspace_root() / "democritus_atlas" / "causal_claims" / "root_topics.txt"


@dataclass(frozen=True)
class DemocritusAgenticConfig:
    """Runtime configuration for the agentic Democritus runner."""

    outdir: Path
    domain_name: str = "topics"
    topics_file: Path | None = None
    root_topics: tuple[str, ...] = ()
    input_pdf: Path | None = None
    auto_topics_from_pdf: bool = False
    root_topic_strategy: str = "summary_guided"
    include_phase2: bool = True
    depth_limit: int = 3
    max_total_topics: int = 100
    statements_per_question: int = 2
    statement_batch_size: int = 16
    statement_max_tokens: int = 192
    manifold_mode: str = "full"
    topk: int = 200
    radii: str = "1,2,3"
    maxnodes: str = "10,20,30,40,60"
    lambda_edge: float = 0.25
    topk_models: int = 5
    topk_claims: int = 30
    alpha: float = 1.0
    tier1: float = 0.60
    tier2: float = 0.30
    anchors: str = ""
    title: str = ""
    dedupe_focus: bool = False
    require_anchor_in_focus: bool = False
    focus_blacklist_regex: str = ""
    render_topk_pngs: bool = True
    assets_dir: str = "assets"
    png_dpi: int = 200
    write_deep_dive: bool = False
    deep_dive_max_bullets: int = 8
    intra_document_shards: int = 1
    llm_usage_log_path: Path | None = None

    def resolved(self) -> "DemocritusAgenticConfig":
        return DemocritusAgenticConfig(
            outdir=self.outdir.resolve(),
            domain_name=self.domain_name,
            topics_file=self.topics_file.resolve() if self.topics_file else None,
            root_topics=self.root_topics,
            input_pdf=self.input_pdf.resolve() if self.input_pdf else None,
            auto_topics_from_pdf=self.auto_topics_from_pdf,
            root_topic_strategy=self.root_topic_strategy,
            include_phase2=self.include_phase2,
            depth_limit=self.depth_limit,
            max_total_topics=self.max_total_topics,
            statements_per_question=max(1, int(self.statements_per_question)),
            statement_batch_size=max(1, int(self.statement_batch_size)),
            statement_max_tokens=max(48, int(self.statement_max_tokens)),
            manifold_mode=self.manifold_mode,
            topk=self.topk,
            radii=self.radii,
            maxnodes=self.maxnodes,
            lambda_edge=self.lambda_edge,
            topk_models=self.topk_models,
            topk_claims=self.topk_claims,
            alpha=self.alpha,
            tier1=self.tier1,
            tier2=self.tier2,
            anchors=self.anchors,
            title=self.title,
            dedupe_focus=self.dedupe_focus,
            require_anchor_in_focus=self.require_anchor_in_focus,
            focus_blacklist_regex=self.focus_blacklist_regex,
            render_topk_pngs=self.render_topk_pngs,
            assets_dir=self.assets_dir,
            png_dpi=self.png_dpi,
            write_deep_dive=self.write_deep_dive,
            deep_dive_max_bullets=self.deep_dive_max_bullets,
            intra_document_shards=max(1, int(self.intra_document_shards)),
            llm_usage_log_path=self.llm_usage_log_path.resolve() if self.llm_usage_log_path else None,
        )


@dataclass(frozen=True)
class DemocritusAgentRecord:
    """Execution record for one Democritus agent."""

    agent_name: str
    frontier_index: int
    status: str
    started_at: float
    ended_at: float
    outputs: tuple[str, ...] = ()
    log_path: str | None = None
    notes: str = ""


def build_democritus_agentic_workflow(*, include_phase2: bool = True) -> AgenticWorkflow:
    """Build an agentic workflow that mirrors the current Democritus pipeline."""

    artifacts = [
        ArtifactSpec("input_document", "pdf_document", persistent=True),
        ArtifactSpec("root_topics", "topic_seed_set", persistent=True),
        ArtifactSpec("topic_graph", "topic_graph_jsonl", persistent=True),
        ArtifactSpec("causal_questions", "causal_question_set", persistent=True),
        ArtifactSpec("causal_statements", "causal_statement_set", persistent=True),
        ArtifactSpec("relational_triples", "relational_triple_set", persistent=True),
        ArtifactSpec("relational_state", "relational_state_pickle", persistent=True),
        ArtifactSpec("manifold_state", "manifold_state_pickle", persistent=True),
        ArtifactSpec("topos_slice", "topos_slice_bundle", persistent=True),
        ArtifactSpec("manifold_viz", "manifold_visualization_bundle", persistent=True),
    ]
    agents = [
        AgentSpec(
            name="document_intake_agent",
            role="prepare_document_inputs",
            produces=("input_document",),
            capabilities=("copy_pdf", "write_document_manifest"),
        ),
        AgentSpec(
            name="root_topic_discovery_agent",
            role="discover_root_topics",
            consumes=("input_document",),
            produces=("root_topics",),
            attention_from=("document_intake_agent",),
            capabilities=("root_topic_discovery", "openai_compatible_llm_calls", "topic_seed_selection"),
        ),
        AgentSpec(
            name="topic_graph_agent",
            role="build_topic_graph",
            consumes=("root_topics",),
            produces=("topic_graph",),
            attention_from=("root_topic_discovery_agent",),
            capabilities=("topic_expansion", "topic_graph_construction"),
        ),
        AgentSpec(
            name="causal_question_agent",
            role="build_causal_questions",
            consumes=("topic_graph",),
            produces=("causal_questions",),
            attention_from=("topic_graph_agent",),
            capabilities=("question_generation", "topic_path_reasoning"),
        ),
        AgentSpec(
            name="causal_statement_agent",
            role="build_causal_statements",
            consumes=("causal_questions",),
            produces=("causal_statements",),
            attention_from=("causal_question_agent",),
            capabilities=("statement_generation", "causal_language_synthesis"),
        ),
        AgentSpec(
            name="triple_extraction_agent",
            role="extract_relational_triples",
            consumes=("causal_statements",),
            produces=("relational_triples",),
            attention_from=("causal_statement_agent",),
            capabilities=("relational_extraction", "typed_triple_building"),
        ),
        AgentSpec(
            name="manifold_builder_agent",
            role="build_relational_manifold",
            consumes=("relational_triples",),
            produces=("relational_state", "manifold_state"),
            attention_from=("triple_extraction_agent",),
            capabilities=("state_construction", "manifold_refinement", "umap_embedding"),
        ),
        AgentSpec(
            name="topos_slice_agent",
            role="write_topos_slice",
            consumes=("relational_state", "root_topics"),
            produces=("topos_slice",),
            attention_from=("manifold_builder_agent", "root_topic_discovery_agent"),
            capabilities=("topos_export", "slice_metadata"),
        ),
        AgentSpec(
            name="manifold_visualization_agent",
            role="visualize_relational_manifold",
            consumes=("manifold_state",),
            produces=("manifold_viz",),
            attention_from=("manifold_builder_agent",),
            capabilities=("plotting", "manifold_visualization"),
        ),
    ]

    if include_phase2:
        artifacts.extend(
            [
                ArtifactSpec("lcm_sweep", "lcm_sweep_directory", persistent=True),
                ArtifactSpec("lcm_scores", "lcm_scores_csv", persistent=True),
                ArtifactSpec("credibility_report", "credibility_report_bundle", persistent=True),
                ArtifactSpec("executive_summary", "executive_summary_markdown", persistent=True),
            ]
        )
        agents.extend(
            [
                AgentSpec(
                    name="lcm_sweep_agent",
                    role="generate_local_causal_models",
                    consumes=("relational_triples",),
                    produces=("lcm_sweep",),
                    attention_from=("triple_extraction_agent",),
                    capabilities=("local_causal_model_generation", "neighborhood_sweep"),
                ),
                AgentSpec(
                    name="lcm_scoring_agent",
                    role="score_local_causal_models",
                    consumes=("lcm_sweep", "relational_triples"),
                    produces=("lcm_scores",),
                    attention_from=("lcm_sweep_agent", "triple_extraction_agent"),
                    capabilities=("model_scoring", "edge_consistency_scoring"),
                ),
                AgentSpec(
                    name="credibility_bundle_agent",
                    role="build_reports_and_summary",
                    consumes=("lcm_scores", "relational_triples"),
                    produces=("credibility_report", "executive_summary"),
                    attention_from=("lcm_scoring_agent", "triple_extraction_agent"),
                    capabilities=("report_building", "executive_summarization"),
                ),
            ]
        )

    return build_agentic_workflow(
        name="DemocritusCurrentPipelineAgenticWorkflow",
        agents=tuple(agents),
        artifacts=tuple(artifacts),
        metadata={
            "family": "Democritus",
            "semantic_role": "agentic_runtime_blueprint",
            "phase2_enabled": include_phase2,
        },
    )


class DemocritusAgenticRunner:
    """Execute the current Democritus pipeline as an agentic FF2 workflow."""

    def __init__(
        self,
        config: DemocritusAgenticConfig,
        *,
        handlers: dict[str, Callable[[], tuple[str, ...]]] | None = None,
    ) -> None:
        self.config = config.resolved()
        self.workflow = build_democritus_agentic_workflow(include_phase2=self.config.include_phase2)
        self.repo_root = _democritus_repo_root()
        self.outdir = self.config.outdir
        self.logs_dir = self.outdir / "agent_logs"
        self.summary_path = self.outdir / "agent_run_summary.json"
        self._handlers = handlers or self._default_handlers()

    def plan(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(agent.name for agent in frontier) for frontier in self.workflow.parallel_frontiers())

    def run_agent(self, agent_name: str) -> tuple[str, ...]:
        return self._handlers[agent_name]()

    def run(
        self,
        *,
        dry_run: bool = False,
        max_workers: int | None = None,
    ) -> tuple[DemocritusAgentRecord, ...]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        records: list[DemocritusAgentRecord] = []
        for frontier_index, frontier in enumerate(self.workflow.parallel_frontiers()):
            if dry_run:
                now = time.time()
                records.extend(
                    DemocritusAgentRecord(
                        agent_name=agent.name,
                        frontier_index=frontier_index,
                        status="planned",
                        started_at=now,
                        ended_at=now,
                    )
                    for agent in frontier
                )
                continue

            worker_count = max_workers if max_workers is not None else len(frontier)
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
                future_map = {
                    executor.submit(self._execute_agent, agent.name, frontier_index): agent.name
                    for agent in frontier
                }
                for future in as_completed(future_map):
                    records.append(future.result())

        ordered = tuple(sorted(records, key=lambda record: (record.frontier_index, record.agent_name)))
        self.summary_path.write_text(
            json.dumps([asdict(record) for record in ordered], indent=2),
            encoding="utf-8",
        )
        return ordered

    def _execute_agent(self, agent_name: str, frontier_index: int) -> DemocritusAgentRecord:
        started_at = time.time()
        log_path = self.logs_dir / f"{agent_name}.log"
        try:
            outputs = self.run_agent(agent_name)
        except Exception as exc:
            ended_at = time.time()
            record = DemocritusAgentRecord(
                agent_name=agent_name,
                frontier_index=frontier_index,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                log_path=str(log_path),
                notes=str(exc),
            )
            self._append_failure(log_path, exc)
            raise RuntimeError(f"Agent {agent_name!r} failed: {exc}") from exc
        ended_at = time.time()
        return DemocritusAgentRecord(
            agent_name=agent_name,
            frontier_index=frontier_index,
            status="ok",
            started_at=started_at,
            ended_at=ended_at,
            outputs=outputs,
            log_path=str(log_path),
        )

    def _default_handlers(self) -> dict[str, Callable[[], tuple[str, ...]]]:
        handlers: dict[str, Callable[[], tuple[str, ...]]] = {
            "document_intake_agent": self._run_document_intake_agent,
            "root_topic_discovery_agent": self._run_root_topic_discovery_agent,
            "topic_graph_agent": self._run_topic_graph_agent,
            "causal_question_agent": self._run_causal_question_agent,
            "causal_statement_agent": self._run_causal_statement_agent,
            "triple_extraction_agent": self._run_triple_extraction_agent,
            "manifold_builder_agent": self._run_manifold_builder_agent,
            "topos_slice_agent": self._run_topos_slice_agent,
            "manifold_visualization_agent": self._run_manifold_visualization_agent,
        }
        if self.config.include_phase2:
            handlers.update(
                {
                    "lcm_sweep_agent": self._run_lcm_sweep_agent,
                    "lcm_scoring_agent": self._run_lcm_scoring_agent,
                    "credibility_bundle_agent": self._run_credibility_bundle_agent,
                }
            )
        return handlers

    def _append_failure(self, log_path: Path, exc: Exception) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[FF2] Agent failed: {exc}\n")

    def _read_log_tail(self, log_path: Path, *, max_lines: int = 20) -> str:
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def _stage_env(self) -> dict[str, str]:
        base = dict(os.environ)
        pythonpath_parts = [str(self.repo_root), str(_workspace_root())]
        if base.get("PYTHONPATH"):
            pythonpath_parts.append(base["PYTHONPATH"])
        env = {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = base.get("MPLBACKEND", "Agg")
        mpl_config_dir = self.outdir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = base.get("MPLCONFIGDIR", str(mpl_config_dir))
        if self.config.llm_usage_log_path:
            env["CLIFF_LLM_USAGE_PATH"] = str(self.config.llm_usage_log_path)
            env["CLIFF_LLM_USAGE_ROUTE"] = "democritus"
            env["CLIFF_LLM_USAGE_RUN"] = self.config.domain_name
            env["CLIFF_LLM_USAGE_OUTDIR"] = str(self.outdir)
        return {**base, **env}

    def _run_subprocess_agent(
        self,
        agent_name: str,
        cmd: list[str],
        *,
        cwd: Path,
        outputs: tuple[Path, ...],
    ) -> tuple[str, ...]:
        log_path = self.logs_dir / f"{agent_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = self._stage_env()
        env["CLIFF_LLM_USAGE_AGENT"] = agent_name
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("[CMD] " + " ".join(cmd) + "\n\n")
            log_file.write(f"[CWD] {cwd}\n")
            log_file.write(f"[PYTHONPATH] {env.get('PYTHONPATH', '')}\n\n")
            log_file.flush()
            try:
                subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                tail = self._read_log_tail(log_path)
                message = (
                    f"Subprocess agent {agent_name!r} exited with status {exc.returncode}. "
                    f"Log: {log_path}"
                )
                if tail:
                    message += f"\nLast log lines:\n{tail}"
                raise RuntimeError(message) from exc
        return tuple(str(path) for path in outputs)

    def _topics_path(self) -> Path:
        return self.outdir / "configs" / "root_topics.txt"

    def _topic_guidance_path(self) -> Path:
        return self.outdir / "configs" / "document_topic_guide.json"

    def _root_topics_values(self) -> tuple[str, ...]:
        topics_path = self._topics_path()
        if not topics_path.exists():
            return ()
        topics = []
        for line in topics_path.read_text(encoding="utf-8").splitlines():
            topic = line.strip()
            if topic and not topic.startswith("#"):
                topics.append(topic)
        return tuple(topics)

    def _run_document_intake_agent(self) -> tuple[str, ...]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "configs").mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        manifest_path = self.outdir / "document_manifest.json"
        manifest = {
            "input_pdf": str(self.config.input_pdf) if self.config.input_pdf else None,
            "domain_name": self.config.domain_name,
            "root_topic_strategy": self.config.root_topic_strategy,
        }

        if self.config.input_pdf:
            input_copy = self.outdir / "input.pdf"
            shutil.copy2(self.config.input_pdf, input_copy)
            outputs.append(str(input_copy))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        outputs.append(str(manifest_path.resolve()))
        return tuple(outputs)

    def _load_topics_from_path(self, path: Path) -> tuple[str, ...]:
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    def _discover_root_topics_v0_openai(self) -> tuple[str, ...]:
        workspace_root = _workspace_root()
        sys.path.insert(0, str(workspace_root))
        try:
            democ = import_module("FunctorFlow.democritus")
            llm_client = democ.OpenAICompatibleDemocritusClient(
                democ.DemocritusLLMConfig.from_env(),
                usage_log_path=self.config.llm_usage_log_path,
                usage_metadata={
                    "route": "democritus",
                    "run_name": self.config.domain_name,
                    "agent_name": "root_topic_discovery_agent",
                    "outdir": str(self.outdir),
                },
            )
            discovery_config = democ.DemocritusTopicDiscoveryConfig(guidance_mode="plain")
            _, topics = democ.discover_topics_from_pdf(
                self.config.input_pdf,
                llm=llm_client,
                config=discovery_config,
            )
        finally:
            sys.path.pop(0)
        return tuple(topics)

    def _discover_root_topics_summary_guided(self) -> tuple[str, ...]:
        workspace_root = _workspace_root()
        sys.path.insert(0, str(workspace_root))
        try:
            democ = import_module("FunctorFlow.democritus")
            llm_client = democ.OpenAICompatibleDemocritusClient(
                democ.DemocritusLLMConfig.from_env(),
                usage_log_path=self.config.llm_usage_log_path,
                usage_metadata={
                    "route": "democritus",
                    "run_name": self.config.domain_name,
                    "agent_name": "root_topic_discovery_agent",
                    "outdir": str(self.outdir),
                },
            )
            discovery_config = democ.DemocritusTopicDiscoveryConfig(guidance_mode="summary_guided")
            if hasattr(democ, "discover_topics_from_pdf_with_metadata"):
                _, topics, metadata = democ.discover_topics_from_pdf_with_metadata(
                    self.config.input_pdf,
                    llm=llm_client,
                    config=discovery_config,
                )
                guidance = dict(metadata.get("guidance") or {})
                if guidance:
                    self._topic_guidance_path().write_text(json.dumps(guidance, indent=2), encoding="utf-8")
            else:
                _, topics = democ.discover_topics_from_pdf(
                    self.config.input_pdf,
                    llm=llm_client,
                    config=discovery_config,
                )
        finally:
            sys.path.pop(0)
        return tuple(topics)

    def _discover_root_topics_heuristic(self) -> tuple[str, ...]:
        sys.path.insert(0, str(self.repo_root))
        try:
            from pipelines.batch_pipeline import auto_root_topics_from_text, extract_pdf_text
        finally:
            sys.path.pop(0)
        if not self.config.input_pdf:
            raise ValueError("Heuristic topic discovery requires `input_pdf`.")
        text = extract_pdf_text(self.config.input_pdf)
        return tuple(auto_root_topics_from_text(text, n=18))

    def _write_agent_note(self, agent_name: str, message: str) -> None:
        log_path = self.logs_dir / f"{agent_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _discover_root_topics_with_resilience(self) -> tuple[str, ...]:
        if self.config.root_topic_strategy == "heuristic":
            return self._discover_root_topics_heuristic()
        if self.config.root_topic_strategy not in {"v0_openai", "summary_guided"}:
            raise ValueError(
                "Unsupported `root_topic_strategy`. Expected one of "
                "`heuristic`, `summary_guided`, or `v0_openai`."
            )
        strategy_name = self.config.root_topic_strategy
        discover_topics = (
            self._discover_root_topics_summary_guided
            if strategy_name == "summary_guided"
            else self._discover_root_topics_v0_openai
        )

        last_openai_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return discover_topics()
            except Exception as exc:
                last_openai_error = exc
                self._write_agent_note(
                    "root_topic_discovery_agent",
                    f"[FF2] {strategy_name} topic discovery attempt {attempt}/3 failed: {exc}",
                )
                if attempt < 3:
                    time.sleep(1.0)

        if self.config.input_pdf:
            try:
                topics = self._discover_root_topics_heuristic()
                self._write_agent_note(
                    "root_topic_discovery_agent",
                    f"[FF2] Falling back to heuristic topic discovery after {strategy_name} failure.",
                )
                return topics
            except Exception as heuristic_exc:
                self._write_agent_note(
                    "root_topic_discovery_agent",
                    f"[FF2] Heuristic topic discovery fallback failed: {heuristic_exc}",
                )
                raise RuntimeError(
                    f"Root topic discovery failed for both {strategy_name} and heuristic strategies. "
                    f"{strategy_name} error: {last_openai_error}; heuristic error: {heuristic_exc}"
                ) from heuristic_exc

        raise RuntimeError(
            f"Root topic discovery failed via {strategy_name} and no PDF-backed heuristic fallback was available. "
            f"Last error: {last_openai_error}"
        ) from last_openai_error

    def _run_root_topic_discovery_agent(self) -> tuple[str, ...]:
        topics_path = self._topics_path()
        topics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.root_topics:
            topics = self.config.root_topics
        elif self.config.topics_file:
            topics = self._load_topics_from_path(self.config.topics_file)
        elif self.config.auto_topics_from_pdf and self.config.input_pdf:
            topics = self._discover_root_topics_with_resilience()
        else:
            default_topics = _default_topics_path()
            if default_topics.exists():
                topics = self._load_topics_from_path(default_topics)
            else:
                raise ValueError(
                    "Democritus agentic runner needs root topics. Provide `topics_file`, "
                    "`root_topics`, or `input_pdf` with `auto_topics_from_pdf=True`."
                )

        topics_path.write_text("\n".join(topics) + "\n", encoding="utf-8")
        outputs = [str(topics_path.resolve())]
        guidance_path = self._topic_guidance_path()
        if guidance_path.exists():
            outputs.append(str(guidance_path.resolve()))
        return tuple(outputs)

    def _run_topic_graph_agent(self) -> tuple[str, ...]:
        topic_graph = self.outdir / "topic_graph.jsonl"
        topic_list = self.outdir / "topic_list.txt"
        shard_count = max(1, int(self.config.intra_document_shards))
        if shard_count <= 1:
            return self._run_subprocess_agent(
                "topic_graph_agent",
                [
                    sys.executable,
                    "-m",
                    "scripts.topic_graph_builder",
                    "--topics-file",
                    str(self._topics_path()),
                    "--depth-limit",
                    str(self.config.depth_limit),
                    "--max-total-topics",
                    str(self.config.max_total_topics),
                    "--topic-graph",
                    str(topic_graph),
                    "--topic-list",
                    str(topic_list),
                ],
                cwd=self.outdir,
                outputs=(topic_graph, topic_list),
            )

        shard_dir = self.outdir / ".intra_document_shards" / "topic_graph_agent"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_graph_outputs = [
            shard_dir / f"topic_graph_agent_shard_{index:02d}.jsonl"
            for index in range(shard_count)
        ]
        shard_list_outputs = [
            shard_dir / f"topic_graph_agent_shard_{index:02d}.txt"
            for index in range(shard_count)
        ]

        def command_for_shard(index: int) -> list[str]:
            return [
                sys.executable,
                "-m",
                "scripts.topic_graph_builder",
                "--topics-file",
                str(self._topics_path()),
                "--depth-limit",
                str(self.config.depth_limit),
                "--max-total-topics",
                str(self.config.max_total_topics),
                "--topic-graph",
                str(shard_graph_outputs[index]),
                "--topic-list",
                str(shard_list_outputs[index]),
                "--shard-index",
                str(index),
                "--num-shards",
                str(shard_count),
            ]

        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            future_map = {
                executor.submit(
                    self._run_subprocess_agent,
                    f"topic_graph_agent_shard_{index:02d}",
                    command_for_shard(index),
                    cwd=self.outdir,
                    outputs=(shard_graph_outputs[index], shard_list_outputs[index]),
                ): index
                for index in range(shard_count)
            }
            for future in as_completed(future_map):
                future.result()

        self._merge_topic_graph_outputs(shard_graph_outputs, topic_graph, topic_list)
        return (str(topic_graph), str(topic_list))

    def _merge_topic_graph_outputs(
        self,
        shard_outputs: list[Path],
        topic_graph_path: Path,
        topic_list_path: Path,
    ) -> None:
        merged_by_topic: dict[str, dict[str, object]] = {}
        for shard_output in shard_outputs:
            if not shard_output.exists():
                continue
            for line in shard_output.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                record = dict(json.loads(stripped))
                topic = str(record.get("topic") or "").strip()
                if not topic:
                    continue
                depth = int(record.get("depth") or 0)
                existing = merged_by_topic.get(topic)
                if existing is None or depth < int(existing.get("depth") or 0):
                    merged_by_topic[topic] = record

        ordered_records = sorted(
            merged_by_topic.values(),
            key=lambda item: (int(item.get("depth") or 0), str(item.get("topic") or "").lower()),
        )
        topic_graph_path.parent.mkdir(parents=True, exist_ok=True)
        with topic_graph_path.open("w", encoding="utf-8") as handle:
            for record in ordered_records:
                handle.write(json.dumps(record) + "\n")
        topic_list_path.parent.mkdir(parents=True, exist_ok=True)
        with topic_list_path.open("w", encoding="utf-8") as handle:
            for record in ordered_records:
                handle.write(f"{record.get('topic', '')}\t{int(record.get('depth') or 0)}\n")

    def _merge_jsonl_outputs(self, shard_outputs: list[Path], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as merged:
            for shard_output in shard_outputs:
                if not shard_output.exists():
                    continue
                content = shard_output.read_text(encoding="utf-8")
                if not content:
                    continue
                merged.write(content)
                if not content.endswith("\n"):
                    merged.write("\n")

    def _run_sharded_generation_agent(
        self,
        *,
        agent_name: str,
        output_path: Path,
        command_builder: Callable[[int, int, Path], list[str]],
    ) -> tuple[str, ...]:
        shard_count = max(1, int(self.config.intra_document_shards))
        if shard_count <= 1:
            return self._run_subprocess_agent(
                agent_name,
                command_builder(0, 1, output_path),
                cwd=self.outdir,
                outputs=(output_path,),
            )

        shard_dir = self.outdir / ".intra_document_shards" / agent_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_outputs = [
            shard_dir / f"{agent_name}_shard_{index:02d}.jsonl"
            for index in range(shard_count)
        ]

        with ThreadPoolExecutor(max_workers=shard_count) as executor:
            future_map = {
                executor.submit(
                    self._run_subprocess_agent,
                    f"{agent_name}_shard_{index:02d}",
                    command_builder(index, shard_count, shard_outputs[index]),
                    cwd=self.outdir,
                    outputs=(shard_outputs[index],),
                ): index
                for index in range(shard_count)
            }
            for future in as_completed(future_map):
                future.result()

        self._merge_jsonl_outputs(shard_outputs, output_path)
        return (str(output_path),)

    def _run_causal_question_agent(self) -> tuple[str, ...]:
        output_path = self.outdir / "causal_questions.jsonl"
        document_guide_path = self._topic_guidance_path()
        return self._run_sharded_generation_agent(
            agent_name="causal_question_agent",
            output_path=output_path,
            command_builder=lambda shard_index, shard_count, shard_output: [
                sys.executable,
                "-m",
                "scripts.causal_question_builder",
                "--topic-graph",
                str(self.outdir / "topic_graph.jsonl"),
                "--output",
                str(shard_output),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(shard_count),
            ]
            + (
                ["--document-guide", str(document_guide_path)]
                if document_guide_path.exists()
                else []
            ),
        )

    def _run_causal_statement_agent(self) -> tuple[str, ...]:
        output_path = self.outdir / "causal_statements.jsonl"
        document_guide_path = self._topic_guidance_path()
        return self._run_sharded_generation_agent(
            agent_name="causal_statement_agent",
            output_path=output_path,
            command_builder=lambda shard_index, shard_count, shard_output: [
                sys.executable,
                "-m",
                "scripts.causal_statement_builder",
                "--input",
                str(self.outdir / "causal_questions.jsonl"),
                "--output",
                str(shard_output),
                "--statements-per-question",
                str(self.config.statements_per_question),
                "--batch-size",
                str(self.config.statement_batch_size),
                "--max-tokens",
                str(self.config.statement_max_tokens),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(shard_count),
            ]
            + (
                ["--document-guide", str(document_guide_path)]
                if document_guide_path.exists()
                else []
            ),
        )

    def _run_triple_extraction_agent(self) -> tuple[str, ...]:
        output_path = self.outdir / "relational_triples.jsonl"
        return self._run_subprocess_agent(
            "triple_extraction_agent",
            [sys.executable, "-m", "scripts.relational_triple_extractor"],
            cwd=self.outdir,
            outputs=(output_path,),
        )

    def _run_manifold_builder_agent(self) -> tuple[str, ...]:
        relational_state = self.outdir / "relational_state.pkl"
        manifold_state = self.outdir / "manifold_state.pkl"
        return self._run_subprocess_agent(
            "manifold_builder_agent",
            [
                sys.executable,
                "-m",
                "scripts.manifold_builder",
                "--mode",
                self.config.manifold_mode,
            ],
            cwd=self.outdir,
            outputs=(relational_state, manifold_state),
        )

    def _run_topos_slice_agent(self) -> tuple[str, ...]:
        sys.path.insert(0, str(self.repo_root))
        try:
            from scripts.write_topos_slice import write_topos_slice
        finally:
            sys.path.pop(0)
        slice_dir = self.outdir / "topos_slices"
        slice_pkl, meta_json = write_topos_slice(
            rel_state_path=str(self.outdir / "relational_state.pkl"),
            domain_name=self.config.domain_name,
            topic_roots=list(self._root_topics_values()),
            out_dir=str(slice_dir),
        )
        return (slice_pkl, meta_json)

    def _run_manifold_visualization_agent(self) -> tuple[str, ...]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        sys.path.insert(0, str(self.repo_root))
        try:
            from scripts.visualize_manifold import visualize_from_state
        finally:
            sys.path.pop(0)
        viz_dir = self.outdir / "viz"
        state_path = self.outdir / "manifold_state.pkl"
        if not state_path.exists():
            state_path = self.outdir / "relational_state.pkl"
        # Matplotlib pyplot state is not thread-safe across simultaneous document renders.
        # Keep the batch distributed while serializing just the image rendering section.
        with _MATPLOTLIB_RENDER_LOCK:
            visualize_from_state(
                state_path=str(state_path),
                out_dir=str(viz_dir),
                title_prefix=f"{self.config.domain_name} relational manifold",
            )
        return (
            str(viz_dir / "relational_manifold_2d.png"),
            str(viz_dir / "relational_manifold_3d.png"),
        )

    def _run_lcm_sweep_agent(self) -> tuple[str, ...]:
        sweep_dir = self.outdir / "sweep"
        return self._run_subprocess_agent(
            "lcm_sweep_agent",
            [
                sys.executable,
                "-m",
                "scripts.sweep_lcm",
                "--triples",
                str(self.outdir / "relational_triples.jsonl"),
                "--outdir",
                str(sweep_dir),
                "--topk",
                str(self.config.topk),
                "--radii",
                self.config.radii,
                "--maxnodes",
                self.config.maxnodes,
            ],
            cwd=self.outdir,
            outputs=(sweep_dir,),
        )

    def _run_lcm_scoring_agent(self) -> tuple[str, ...]:
        sweep_dir = self.outdir / "sweep"
        scores_path = sweep_dir / "scores.csv"
        cmd = [
            sys.executable,
            "-m",
            "scripts.score_lcms_dir",
            "--indir",
            str(sweep_dir),
            "--triples",
            str(self.outdir / "relational_triples.jsonl"),
            "--out",
            str(scores_path),
            "--lambda-edge",
            str(self.config.lambda_edge),
        ]
        return self._run_subprocess_agent(
            "lcm_scoring_agent",
            cmd,
            cwd=self.outdir,
            outputs=(scores_path,),
        )

    def _run_credibility_bundle_agent(self) -> tuple[str, ...]:
        reports_dir = self.outdir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        name = self.config.domain_name
        cmd = [
            sys.executable,
            "-m",
            "scripts.make_credibility_bundle",
            "--scores",
            str(self.outdir / "sweep" / "scores.csv"),
            "--triples",
            str(self.outdir / "relational_triples.jsonl"),
            "--lcm-dir",
            str(self.outdir / "sweep"),
            "--topk-models",
            str(self.config.topk_models),
            "--topk-claims",
            str(self.config.topk_claims),
            "--alpha",
            str(self.config.alpha),
            "--tier1",
            str(self.config.tier1),
            "--tier2",
            str(self.config.tier2),
            "--outdir",
            str(reports_dir),
            "--name",
            name,
        ]
        if self.config.dedupe_focus:
            cmd.append("--dedupe-focus")
        if self.config.require_anchor_in_focus:
            cmd.append("--require-anchor-in-focus")
        if self.config.anchors.strip():
            cmd += ["--keyword-anchors", self.config.anchors]
        if self.config.title.strip():
            cmd += ["--title", self.config.title]
        if self.config.focus_blacklist_regex.strip():
            cmd += ["--focus-blacklist-regex", self.config.focus_blacklist_regex]
        if self.config.render_topk_pngs:
            cmd += [
                "--render-topk-pngs",
                "--assets-dir",
                self.config.assets_dir,
                "--png-dpi",
                str(self.config.png_dpi),
            ]
        if self.config.write_deep_dive:
            cmd += [
                "--write-deep-dive",
                "--deep-dive-max-bullets",
                str(self.config.deep_dive_max_bullets),
            ]
        return self._run_subprocess_agent(
            "credibility_bundle_agent",
            cmd,
            cwd=self.outdir,
            outputs=(
                reports_dir / f"{name}_credibility_report.md",
                reports_dir / f"{name}_executive_summary.md",
            ),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Democritus as an FF2 agentic workflow.")
    parser.add_argument("--outdir", required=True, help="Output directory for the agentic run.")
    parser.add_argument("--domain-name", default="topics")
    parser.add_argument("--topics-file", default="")
    parser.add_argument("--root-topic", action="append", default=[])
    parser.add_argument("--input-pdf", default="")
    parser.add_argument("--auto-topics-from-pdf", action="store_true")
    parser.add_argument(
        "--root-topic-strategy",
        default="summary_guided",
        choices=["summary_guided", "v0_openai", "heuristic"],
    )
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--depth-limit", type=int, default=3)
    parser.add_argument("--max-total-topics", type=int, default=100)
    parser.add_argument("--manifold-mode", default="full", choices=["full", "lite", "moe"])
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--radii", default="1,2,3")
    parser.add_argument("--maxnodes", default="10,20,30,40,60")
    parser.add_argument("--lambda-edge", type=float, default=0.25)
    parser.add_argument("--topk-models", type=int, default=5)
    parser.add_argument("--topk-claims", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tier1", type=float, default=0.60)
    parser.add_argument("--tier2", type=float, default=0.30)
    parser.add_argument("--anchors", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--dedupe-focus", action="store_true")
    parser.add_argument("--require-anchor-in-focus", action="store_true")
    parser.add_argument("--focus-blacklist-regex", default="")
    parser.add_argument(
        "--render-topk-pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--png-dpi", type=int, default=200)
    parser.add_argument("--write-deep-dive", action="store_true")
    parser.add_argument("--deep-dive-max-bullets", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = DemocritusAgenticConfig(
        outdir=Path(args.outdir),
        domain_name=args.domain_name,
        topics_file=Path(args.topics_file).expanduser() if args.topics_file else None,
        root_topics=tuple(args.root_topic),
        input_pdf=Path(args.input_pdf).expanduser() if args.input_pdf else None,
        auto_topics_from_pdf=args.auto_topics_from_pdf,
        root_topic_strategy=args.root_topic_strategy,
        include_phase2=not args.skip_phase2,
        depth_limit=args.depth_limit,
        max_total_topics=args.max_total_topics,
        manifold_mode=args.manifold_mode,
        topk=args.topk,
        radii=args.radii,
        maxnodes=args.maxnodes,
        lambda_edge=args.lambda_edge,
        topk_models=args.topk_models,
        topk_claims=args.topk_claims,
        alpha=args.alpha,
        tier1=args.tier1,
        tier2=args.tier2,
        anchors=args.anchors,
        title=args.title,
        dedupe_focus=args.dedupe_focus,
        require_anchor_in_focus=args.require_anchor_in_focus,
        focus_blacklist_regex=args.focus_blacklist_regex,
        render_topk_pngs=args.render_topk_pngs,
        assets_dir=args.assets_dir,
        png_dpi=args.png_dpi,
        write_deep_dive=args.write_deep_dive,
        deep_dive_max_bullets=args.deep_dive_max_bullets,
    )
    runner = DemocritusAgenticRunner(config)
    print(json.dumps({"frontiers": runner.plan()}, indent=2))
    records = runner.run(dry_run=args.dry_run)
    print(json.dumps([asdict(record) for record in records], indent=2))


if __name__ == "__main__":
    main()
