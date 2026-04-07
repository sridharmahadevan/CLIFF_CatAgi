"""Tests for the FF3 conscious-workspace scaffold."""

from __future__ import annotations

import unittest

try:
    from functorflow_v3 import (
        ConsciousBroadcastBoard,
        ConsciousFieldOfView,
        ConsciousnessFunctor,
        UnconsciousProcess,
        route_ff3_query,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import (
        ConsciousBroadcastBoard,
        ConsciousFieldOfView,
        ConsciousnessFunctor,
        UnconsciousProcess,
        route_ff3_query,
    )


class ConsciousnessTests(unittest.TestCase):
    def test_competition_for_access_respects_capacity_and_priority(self) -> None:
        workspace = ConsciousnessFunctor(ConsciousFieldOfView(capacity=3))

        state = workspace.competition_for_access(
            [
                UnconsciousProcess(
                    name="retrieval_summary",
                    source_agent="retrieval_agent",
                    salience=0.9,
                    relevance=0.9,
                    novelty=0.6,
                    urgency=0.4,
                    attention_cost=2,
                ),
                UnconsciousProcess(
                    name="minor_status_ping",
                    source_agent="status_agent",
                    salience=0.2,
                    relevance=0.2,
                    novelty=0.1,
                    urgency=0.1,
                    attention_cost=1,
                ),
                UnconsciousProcess(
                    name="urgent_contradiction",
                    source_agent="consistency_agent",
                    salience=0.8,
                    relevance=1.0,
                    novelty=0.8,
                    urgency=1.0,
                    attention_cost=1,
                ),
            ]
        )

        self.assertEqual(
            tuple(selection.process.name for selection in state.selected),
            ("urgent_contradiction", "retrieval_summary"),
        )
        self.assertEqual(tuple(process.name for process in state.deferred), ("minor_status_ping",))
        self.assertEqual(state.used_capacity, 3)
        self.assertEqual(state.remaining_capacity, 0)

    def test_route_ff3_query_alias_uses_inherited_router(self) -> None:
        decision = route_ff3_query("Find me 10 recent AMD 10-K filings")

        self.assertEqual(decision.route_name, "basket_rocket_sec")

    def test_conscious_broadcast_board_supports_shared_message_passing(self) -> None:
        board = ConsciousBroadcastBoard()

        board.publish(
            source_agent="intent_interpreter_agent",
            title="Normalized intent",
            summary="Detected a kimchi-focused culinary tour in Seoul.",
            payload={"destination": "Seoul", "food_focus": "kimchi"},
            tags=("intent",),
        )
        board.publish(
            source_agent="budget_guard_agent",
            title="Budget guardrails",
            summary="Set a working cap of $50 per meal.",
            payload={"budget_per_meal": 50},
            tags=("budget",),
        )

        messages = board.messages_for_agent("itinerary_composer_agent")
        budget_messages = board.messages_for_agent("itinerary_composer_agent", tag="budget")

        self.assertEqual(len(messages), 2)
        self.assertEqual(len(budget_messages), 1)
        self.assertEqual(budget_messages[0].payload["budget_per_meal"], 50)

    def test_conscious_broadcast_board_preserves_consumed_broadcast_ids(self) -> None:
        board = ConsciousBroadcastBoard()

        intent = board.publish(
            source_agent="intent_interpreter_agent",
            title="Intent",
            summary="Seoul kimchi trip.",
            payload={"destination": "Seoul"},
            tags=("intent",),
        )
        retrieval = board.publish(
            source_agent="stop_retrieval_agent",
            title="Retrieved stops",
            summary="Found candidate stops.",
            payload={"venues": []},
            tags=("retrieved_stops",),
            read_broadcast_ids=(intent.broadcast_id,),
        )

        self.assertEqual(retrieval.read_broadcast_ids, (intent.broadcast_id,))


if __name__ == "__main__":
    unittest.main()
