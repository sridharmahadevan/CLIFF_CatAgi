"""Conscious-workspace semantics layered on top of FF3 agentic workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnconsciousProcess:
    """A background subprocess competing for conscious access."""

    name: str
    source_agent: str
    summary: str = ""
    artifact_refs: tuple[str, ...] = ()
    salience: float = 0.0
    relevance: float = 0.0
    novelty: float = 0.0
    urgency: float = 0.0
    attention_cost: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attention_cost < 1:
            raise ValueError("attention_cost must be at least 1.")


@dataclass(frozen=True)
class AttentionScoreWeights:
    """Weights for competition over conscious access."""

    salience: float = 0.35
    relevance: float = 0.30
    novelty: float = 0.20
    urgency: float = 0.15


@dataclass(frozen=True)
class ConsciousFieldOfView:
    """The limited capacity of the conscious workspace."""

    capacity: int = 7

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1.")


@dataclass(frozen=True)
class BroadcastSelection:
    """A subprocess that won access to the conscious workspace."""

    process: UnconsciousProcess
    score: float


@dataclass(frozen=True)
class ConsciousWorkspaceState:
    """Selected conscious broadcasts plus deferred background processes."""

    field_of_view: ConsciousFieldOfView
    selected: tuple[BroadcastSelection, ...]
    deferred: tuple[UnconsciousProcess, ...]

    @property
    def used_capacity(self) -> int:
        return sum(item.process.attention_cost for item in self.selected)

    @property
    def remaining_capacity(self) -> int:
        return self.field_of_view.capacity - self.used_capacity


@dataclass(frozen=True)
class ConsciousBroadcast:
    """A message elevated into the shared conscious workspace."""

    broadcast_id: str
    source_agent: str
    title: str
    summary: str
    payload: dict[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    audience: str = "global"
    read_broadcast_ids: tuple[str, ...] = ()


class ConsciousBroadcastBoard:
    """Shared broadcast board readable by multiple unconscious processes."""

    def __init__(self) -> None:
        self._broadcasts: list[ConsciousBroadcast] = []
        self._counter = 0

    def publish(
        self,
        *,
        source_agent: str,
        title: str,
        summary: str,
        payload: dict[str, object] | None = None,
        tags: tuple[str, ...] | list[str] = (),
        audience: str = "global",
        read_broadcast_ids: tuple[str, ...] | list[str] = (),
    ) -> ConsciousBroadcast:
        self._counter += 1
        broadcast = ConsciousBroadcast(
            broadcast_id=f"broadcast-{self._counter:04d}",
            source_agent=source_agent,
            title=title,
            summary=summary,
            payload=dict(payload or {}),
            tags=tuple(tags),
            audience=audience,
            read_broadcast_ids=tuple(read_broadcast_ids),
        )
        self._broadcasts.append(broadcast)
        return broadcast

    def broadcasts(self) -> tuple[ConsciousBroadcast, ...]:
        return tuple(self._broadcasts)

    def messages_for_agent(
        self,
        agent_name: str,
        *,
        tag: str | None = None,
    ) -> tuple[ConsciousBroadcast, ...]:
        broadcasts = [
            broadcast
            for broadcast in self._broadcasts
            if broadcast.audience in ("global", str(agent_name))
        ]
        if tag is not None:
            broadcasts = [broadcast for broadcast in broadcasts if tag in broadcast.tags]
        return tuple(broadcasts)


class ConsciousnessFunctor:
    """Map many unconscious processes into a bounded conscious workspace."""

    def __init__(
        self,
        field_of_view: ConsciousFieldOfView | None = None,
        *,
        weights: AttentionScoreWeights | None = None,
    ) -> None:
        self.field_of_view = field_of_view or ConsciousFieldOfView()
        self.weights = weights or AttentionScoreWeights()

    def score(self, process: UnconsciousProcess) -> float:
        """Score a report candidate for conscious access."""

        return (
            self.weights.salience * process.salience
            + self.weights.relevance * process.relevance
            + self.weights.novelty * process.novelty
            + self.weights.urgency * process.urgency
        )

    def competition_for_access(
        self,
        processes: tuple[UnconsciousProcess, ...] | list[UnconsciousProcess],
    ) -> ConsciousWorkspaceState:
        """Select the highest-priority processes that fit in conscious view."""

        ranked = sorted(
            processes,
            key=lambda process: (
                -self.score(process),
                process.attention_cost,
                process.name,
            ),
        )
        selected: list[BroadcastSelection] = []
        deferred: list[UnconsciousProcess] = []
        remaining = self.field_of_view.capacity

        for process in ranked:
            if process.attention_cost <= remaining:
                selected.append(BroadcastSelection(process=process, score=self.score(process)))
                remaining -= process.attention_cost
            else:
                deferred.append(process)

        return ConsciousWorkspaceState(
            field_of_view=self.field_of_view,
            selected=tuple(selected),
            deferred=tuple(deferred),
        )
