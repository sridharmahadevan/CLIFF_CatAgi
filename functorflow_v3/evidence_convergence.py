"""Reusable convergence controller for evidence-gathering workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Generic, Mapping, Protocol, TypeVar

SnapshotT = TypeVar("SnapshotT")


@dataclass(frozen=True)
class EvidenceConvergencePolicy:
    """Policy controlling when a workflow can stop gathering evidence."""

    min_evidence: int
    stability_threshold: float = 1.0
    required_stable_passes: int = 1
    max_evidence: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceConvergenceAssessment(Generic[SnapshotT]):
    """Result of comparing the latest snapshot against prior evidence state."""

    snapshot: SnapshotT
    evidence_count: int
    iteration: int
    similarity: float | None
    comparable: bool
    stability_threshold: float
    stable_passes: int
    required_stable_passes: int
    remaining_stable_passes: int
    min_evidence_remaining: int
    stable: bool
    stop: bool
    stop_trigger: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the assessment."""

        return asdict(self)


class EvidenceConvergenceAdapter(Protocol[SnapshotT]):
    """Workflow-specific convergence semantics."""

    def similarity(
        self,
        previous: SnapshotT,
        current: SnapshotT,
        *,
        policy: EvidenceConvergencePolicy,
    ) -> float:
        """Return a normalized stability score in [0, 1]."""

    def describe(self, snapshot: SnapshotT) -> str:
        """Return a short human-readable summary of the current state."""


class EvidenceConvergenceTracker(Generic[SnapshotT]):
    """Track convergence of evolving evidence snapshots."""

    def __init__(
        self,
        *,
        policy: EvidenceConvergencePolicy,
        adapter: EvidenceConvergenceAdapter[SnapshotT],
    ) -> None:
        self.policy = policy
        self.adapter = adapter
        self._previous_snapshot: SnapshotT | None = None
        self._last_assessment: EvidenceConvergenceAssessment[SnapshotT] | None = None
        self._stable_passes = 0
        self._iteration = 0

    def last_assessment(self) -> EvidenceConvergenceAssessment[SnapshotT]:
        """Return the most recent assessment."""

        if self._last_assessment is None:
            raise RuntimeError("No convergence assessment has been recorded yet.")
        return self._last_assessment

    def assess(
        self,
        snapshot: SnapshotT,
        *,
        evidence_count: int,
    ) -> EvidenceConvergenceAssessment[SnapshotT]:
        self._iteration += 1
        similarity: float | None = None
        comparable = False
        stable = False
        required_stable_passes = max(1, self.policy.required_stable_passes)
        min_evidence_remaining = max(0, self.policy.min_evidence - evidence_count)

        if evidence_count >= self.policy.min_evidence and self._previous_snapshot is not None:
            comparable = True
            similarity = self.adapter.similarity(
                self._previous_snapshot,
                snapshot,
                policy=self.policy,
            )
            stable = similarity >= float(self.policy.stability_threshold)

        if stable:
            self._stable_passes += 1
        else:
            self._stable_passes = 0

        stop = False
        stop_trigger = "pending"
        if evidence_count >= self.policy.min_evidence and self._stable_passes >= required_stable_passes:
            stop = True
            stop_trigger = "stability"
            reason = (
                f"Evidence stabilized after {evidence_count} items "
                f"(similarity={similarity:.3f}); {self.adapter.describe(snapshot)}"
            )
        elif self.policy.max_evidence > 0 and evidence_count >= self.policy.max_evidence:
            stop = True
            stop_trigger = "max_evidence"
            reason = (
                f"Reached max evidence budget of {self.policy.max_evidence}; "
                f"{self.adapter.describe(snapshot)}"
            )
        elif evidence_count < self.policy.min_evidence:
            stop_trigger = "min_evidence"
            remaining = self.policy.min_evidence - evidence_count
            reason = (
                f"Need {remaining} more evidence item(s) before convergence checks; "
                f"{self.adapter.describe(snapshot)}"
            )
        elif similarity is None:
            stop_trigger = "baseline_pending"
            reason = (
                f"Need one more post-floor update before judging stability; "
                f"{self.adapter.describe(snapshot)}"
            )
        else:
            stop_trigger = "stability_pending"
            reason = (
                f"Evidence not yet stable (similarity={similarity:.3f}, "
                f"threshold={self.policy.stability_threshold:.3f}); "
                f"{self.adapter.describe(snapshot)}"
            )

        remaining_stable_passes = 0 if stop else max(0, required_stable_passes - self._stable_passes)

        self._previous_snapshot = snapshot
        assessment = EvidenceConvergenceAssessment(
            snapshot=snapshot,
            evidence_count=evidence_count,
            iteration=self._iteration,
            similarity=similarity,
            comparable=comparable,
            stability_threshold=float(self.policy.stability_threshold),
            stable_passes=self._stable_passes,
            required_stable_passes=required_stable_passes,
            remaining_stable_passes=remaining_stable_passes,
            min_evidence_remaining=min_evidence_remaining,
            stable=stable,
            stop=stop,
            stop_trigger=stop_trigger,
            reason=reason,
        )
        self._last_assessment = assessment
        return assessment
