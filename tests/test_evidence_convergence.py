"""Tests for shared evidence convergence tracking."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

try:
    from functorflow_v3.evidence_convergence import (
        EvidenceConvergenceAdapter,
        EvidenceConvergencePolicy,
        EvidenceConvergenceTracker,
    )
except ModuleNotFoundError:
    from ..functorflow_v3.evidence_convergence import (
        EvidenceConvergenceAdapter,
        EvidenceConvergencePolicy,
        EvidenceConvergenceTracker,
    )


@dataclass(frozen=True)
class FakeSnapshot:
    label: str


class FakeAdapter(EvidenceConvergenceAdapter[FakeSnapshot]):
    def similarity(
        self,
        previous: FakeSnapshot,
        current: FakeSnapshot,
        *,
        policy: EvidenceConvergencePolicy,
    ) -> float:
        del policy
        return 1.0 if previous.label == current.label else 0.0

    def describe(self, snapshot: FakeSnapshot) -> str:
        return snapshot.label


class EvidenceConvergenceTests(unittest.TestCase):
    def test_tracker_stops_after_min_evidence_and_stable_repeat(self) -> None:
        tracker = EvidenceConvergenceTracker(
            policy=EvidenceConvergencePolicy(
                min_evidence=2,
                stability_threshold=1.0,
                required_stable_passes=1,
            ),
            adapter=FakeAdapter(),
        )

        first = tracker.assess(FakeSnapshot("alpha"), evidence_count=1)
        second = tracker.assess(FakeSnapshot("alpha"), evidence_count=2)

        self.assertFalse(first.stop)
        self.assertTrue(second.stop)
        self.assertEqual(second.stable_passes, 1)
        self.assertEqual(first.stop_trigger, "min_evidence")
        self.assertTrue(second.comparable)
        self.assertEqual(second.stop_trigger, "stability")
        self.assertEqual(second.remaining_stable_passes, 0)

    def test_tracker_stops_at_max_evidence_when_never_stable(self) -> None:
        tracker = EvidenceConvergenceTracker(
            policy=EvidenceConvergencePolicy(
                min_evidence=2,
                stability_threshold=1.0,
                required_stable_passes=1,
                max_evidence=3,
            ),
            adapter=FakeAdapter(),
        )

        tracker.assess(FakeSnapshot("alpha"), evidence_count=1)
        tracker.assess(FakeSnapshot("beta"), evidence_count=2)
        final = tracker.assess(FakeSnapshot("gamma"), evidence_count=3)

        self.assertTrue(final.stop)
        self.assertIn("max evidence budget", final.reason.lower())
        self.assertEqual(final.stop_trigger, "max_evidence")
        self.assertEqual(final.remaining_stable_passes, 0)


if __name__ == "__main__":
    unittest.main()
