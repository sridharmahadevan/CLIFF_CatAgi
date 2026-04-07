"""Agent-native categorical workflow semantics for FunctorFlow v2."""

from __future__ import annotations

from dataclasses import dataclass, field

from .semantic_kernel import Category, Interface, ModelObject, Morphism


def _agent_interfaces() -> tuple[Interface, ...]:
    return (
        Interface("inbox", "artifact_inbox"),
        Interface("outbox", "artifact_outbox"),
        Interface("attention", "attention_context"),
        Interface("memory", "working_memory"),
        Interface("control", "execution_control"),
    )


def _artifact_interfaces(kind: str) -> tuple[Interface, ...]:
    return (
        Interface("payload", kind),
        Interface("provenance", "provenance"),
        Interface("schema", "schema"),
    )


@dataclass(frozen=True)
class ArtifactSpec:
    """Declarative specification for an artifact in an agentic workflow."""

    name: str
    kind: str
    persistent: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    """Declarative specification for an agent in an FF2 workflow."""

    name: str
    role: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    attention_from: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AttentionSpec:
    """A colimit-style attention context for a target agent."""

    name: str
    target_agent: str
    input_artifacts: tuple[str, ...]
    source_agents: tuple[str, ...] = ()
    context_kind: str = "attention_context"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DiffusionSpec:
    """A limit-style consistency state for a target agent."""

    name: str
    target_agent: str
    input_artifacts: tuple[str, ...]
    state_kind: str = "diffusion_state"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactNode:
    """A first-class artifact object."""

    object: ModelObject
    kind: str
    persistent: bool

    @property
    def name(self) -> str:
        return self.object.name

    @property
    def metadata(self) -> dict[str, object]:
        return self.object.metadata


@dataclass(frozen=True)
class AgentNode:
    """A first-class agent object."""

    object: ModelObject
    role: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    attention_from: tuple[str, ...]
    capabilities: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.object.name

    @property
    def metadata(self) -> dict[str, object]:
        return self.object.metadata


@dataclass(frozen=True)
class AttentionColimit:
    """A colimit-style context that a target agent aggregates before acting."""

    object: ArtifactNode
    target_agent: AgentNode
    input_artifacts: tuple[ArtifactNode, ...]
    source_agents: tuple[AgentNode, ...]
    injections: tuple[Morphism, ...]

    @property
    def name(self) -> str:
        return self.object.name


@dataclass(frozen=True)
class DiffusionLimit:
    """A limit-style consistency state that glues multiple artifacts."""

    object: ArtifactNode
    target_agent: AgentNode
    input_artifacts: tuple[ArtifactNode, ...]
    projections: tuple[Morphism, ...]

    @property
    def name(self) -> str:
        return self.object.name


@dataclass(frozen=True)
class AgenticWorkflow:
    """A workflow whose primary semantic subjects are agents and artifacts."""

    object: ModelObject
    agents: tuple[AgentNode, ...]
    artifacts: tuple[ArtifactNode, ...]
    productions: tuple[Morphism, ...]
    consumptions: tuple[Morphism, ...]
    attentions: tuple[AttentionColimit, ...]
    diffusions: tuple[DiffusionLimit, ...]

    def agent_named(self, name: str) -> AgentNode:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(f"No agent named {name!r}.")

    def artifact_named(self, name: str) -> ArtifactNode:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise KeyError(f"No artifact named {name!r}.")

    def producer_of(self, artifact_name: str) -> AgentNode | None:
        for production in self.productions:
            if production.target.name == artifact_name:
                return self.agent_named(production.source.name)
        return None

    def dependencies_of(self, agent_name: str) -> tuple[AgentNode, ...]:
        agent = self.agent_named(agent_name)
        dependencies: dict[str, AgentNode] = {}

        for artifact_name in agent.consumes:
            producer = self.producer_of(artifact_name)
            if producer is not None and producer.name != agent.name:
                dependencies[producer.name] = producer

        for source_name in agent.attention_from:
            if source_name != agent.name:
                dependencies[source_name] = self.agent_named(source_name)

        for attention in self.attentions:
            if attention.target_agent.name != agent.name:
                continue
            for source_agent in attention.source_agents:
                if source_agent.name != agent.name:
                    dependencies[source_agent.name] = source_agent
            for artifact in attention.input_artifacts:
                producer = self.producer_of(artifact.name)
                if producer is not None and producer.name != agent.name:
                    dependencies[producer.name] = producer

        for diffusion in self.diffusions:
            if diffusion.target_agent.name != agent.name:
                continue
            for artifact in diffusion.input_artifacts:
                producer = self.producer_of(artifact.name)
                if producer is not None and producer.name != agent.name:
                    dependencies[producer.name] = producer

        ordered = [candidate for candidate in self.agents if candidate.name in dependencies]
        return tuple(ordered)

    def parallel_frontiers(self) -> tuple[tuple[AgentNode, ...], ...]:
        """Topologically layer agents into executable parallel frontiers."""

        dependency_names = {
            agent.name: {dependency.name for dependency in self.dependencies_of(agent.name)}
            for agent in self.agents
        }
        remaining = {agent.name for agent in self.agents}
        frontiers: list[tuple[AgentNode, ...]] = []

        while remaining:
            frontier = tuple(
                agent
                for agent in self.agents
                if agent.name in remaining and not (dependency_names[agent.name] & remaining)
            )
            if not frontier:
                cycle = {
                    name: sorted(dependency_names[name] & remaining)
                    for name in sorted(remaining)
                }
                raise ValueError(f"Workflow contains a cyclic dependency: {cycle}")
            frontiers.append(frontier)
            for agent in frontier:
                remaining.remove(agent.name)

        return tuple(frontiers)


def _build_artifact_node(spec: ArtifactSpec, *, artifact_category: Category) -> ArtifactNode:
    return ArtifactNode(
        object=ModelObject(
            name=spec.name,
            category=artifact_category,
            interfaces=_artifact_interfaces(spec.kind),
            metadata={
                "semantic_role": "artifact",
                "artifact_kind": spec.kind,
                "persistent": spec.persistent,
                **spec.metadata,
            },
        ),
        kind=spec.kind,
        persistent=spec.persistent,
    )


def _build_agent_node(spec: AgentSpec, *, agent_category: Category) -> AgentNode:
    return AgentNode(
        object=ModelObject(
            name=spec.name,
            category=agent_category,
            interfaces=_agent_interfaces(),
            metadata={
                "semantic_role": "agent",
                "role": spec.role,
                "consumes": spec.consumes,
                "produces": spec.produces,
                "attention_from": spec.attention_from,
                "capabilities": spec.capabilities,
                **spec.metadata,
            },
        ),
        role=spec.role,
        consumes=spec.consumes,
        produces=spec.produces,
        attention_from=spec.attention_from,
        capabilities=spec.capabilities,
    )


def _ensure_unique_names(values: tuple[str, ...], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Duplicate {label} names: {duplicates}")


def build_agentic_workflow(
    *,
    name: str,
    agents: tuple[AgentSpec, ...],
    artifacts: tuple[ArtifactSpec, ...],
    attentions: tuple[AttentionSpec, ...] = (),
    diffusions: tuple[DiffusionSpec, ...] = (),
    metadata: dict[str, object] | None = None,
) -> AgenticWorkflow:
    """Build an FF2 workflow from declarative agent/artifact specifications."""

    agent_names = tuple(agent.name for agent in agents)
    artifact_names = tuple(artifact.name for artifact in artifacts)
    _ensure_unique_names(agent_names, "agent")
    _ensure_unique_names(artifact_names, "artifact")

    agent_category = Category("Agent", metadata={"workflow": name})
    artifact_category = Category("Artifact", metadata={"workflow": name})
    workflow_category = Category("AgenticWorkflow", metadata={"workflow": name})

    agent_nodes = tuple(_build_agent_node(spec, agent_category=agent_category) for spec in agents)
    artifact_nodes = tuple(
        _build_artifact_node(spec, artifact_category=artifact_category) for spec in artifacts
    )
    agents_by_name = {agent.name: agent for agent in agent_nodes}
    artifacts_by_name = {artifact.name: artifact for artifact in artifact_nodes}

    productions: list[Morphism] = []
    consumptions: list[Morphism] = []
    produced_artifacts: dict[str, str] = {}

    for agent in agent_nodes:
        for artifact_name in agent.produces:
            if artifact_name not in artifacts_by_name:
                raise KeyError(f"Agent {agent.name!r} produces unknown artifact {artifact_name!r}.")
            if artifact_name in produced_artifacts:
                raise ValueError(
                    f"Artifact {artifact_name!r} is produced by both "
                    f"{produced_artifacts[artifact_name]!r} and {agent.name!r}."
                )
            produced_artifacts[artifact_name] = agent.name
            productions.append(
                Morphism(
                    name=f"{agent.name}__produces__{artifact_name}",
                    source=agent.object,
                    target=artifacts_by_name[artifact_name].object,
                    metadata={"semantic_role": "agent_production"},
                )
            )
        for artifact_name in agent.consumes:
            if artifact_name not in artifacts_by_name:
                raise KeyError(f"Agent {agent.name!r} consumes unknown artifact {artifact_name!r}.")
            consumptions.append(
                Morphism(
                    name=f"{artifact_name}__feeds__{agent.name}",
                    source=artifacts_by_name[artifact_name].object,
                    target=agent.object,
                    metadata={"semantic_role": "agent_consumption"},
                )
            )

    attention_objects: list[ArtifactNode] = []
    attention_records: list[AttentionColimit] = []
    for spec in attentions:
        if spec.target_agent not in agents_by_name:
            raise KeyError(f"Attention target agent {spec.target_agent!r} is unknown.")
        source_agents = tuple(agents_by_name[name] for name in spec.source_agents)
        input_artifacts = tuple(artifacts_by_name[name] for name in spec.input_artifacts)
        context_node = _build_artifact_node(
            ArtifactSpec(
                name=spec.name,
                kind=spec.context_kind,
                persistent=False,
                metadata={
                    "semantic_role": "attention_context",
                    "target_agent": spec.target_agent,
                    "inputs": spec.input_artifacts,
                    "source_agents": spec.source_agents,
                    **spec.metadata,
                },
            ),
            artifact_category=artifact_category,
        )
        attention_objects.append(context_node)
        injections = tuple(
            Morphism(
                name=f"{artifact.name}__into__{context_node.name}",
                source=artifact.object,
                target=context_node.object,
                metadata={"semantic_role": "attention_injection"},
            )
            for artifact in input_artifacts
        )
        attention_records.append(
            AttentionColimit(
                object=context_node,
                target_agent=agents_by_name[spec.target_agent],
                input_artifacts=input_artifacts,
                source_agents=source_agents,
                injections=injections,
            )
        )

    diffusion_objects: list[ArtifactNode] = []
    diffusion_records: list[DiffusionLimit] = []
    for spec in diffusions:
        if spec.target_agent not in agents_by_name:
            raise KeyError(f"Diffusion target agent {spec.target_agent!r} is unknown.")
        input_artifacts = tuple(artifacts_by_name[name] for name in spec.input_artifacts)
        state_node = _build_artifact_node(
            ArtifactSpec(
                name=spec.name,
                kind=spec.state_kind,
                persistent=False,
                metadata={
                    "semantic_role": "diffusion_state",
                    "target_agent": spec.target_agent,
                    "inputs": spec.input_artifacts,
                    **spec.metadata,
                },
            ),
            artifact_category=artifact_category,
        )
        diffusion_objects.append(state_node)
        projections = tuple(
            Morphism(
                name=f"{state_node.name}__to__{artifact.name}",
                source=state_node.object,
                target=artifact.object,
                metadata={"semantic_role": "diffusion_projection"},
            )
            for artifact in input_artifacts
        )
        diffusion_records.append(
            DiffusionLimit(
                object=state_node,
                target_agent=agents_by_name[spec.target_agent],
                input_artifacts=input_artifacts,
                projections=projections,
            )
        )

    workflow_object = ModelObject(
        name=name,
        category=workflow_category,
        metadata={
            "semantic_role": "agentic_workflow",
            "n_agents": len(agent_nodes),
            "n_artifacts": len(artifact_nodes),
            "n_attentions": len(attention_records),
            "n_diffusions": len(diffusion_records),
            **(metadata or {}),
        },
    )

    return AgenticWorkflow(
        object=workflow_object,
        agents=agent_nodes,
        artifacts=artifact_nodes + tuple(attention_objects) + tuple(diffusion_objects),
        productions=tuple(productions),
        consumptions=tuple(consumptions),
        attentions=tuple(attention_records),
        diffusions=tuple(diffusion_records),
    )
