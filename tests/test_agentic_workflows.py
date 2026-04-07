"""Tests for FunctorFlow v2 agentic workflow semantics."""

from __future__ import annotations

import unittest

try:
    from functorflow_v3 import build_basket_rocket_workflow, build_democritus_workflow
except ModuleNotFoundError:
    from ..functorflow_v3 import build_basket_rocket_workflow, build_democritus_workflow


class AgenticWorkflowTests(unittest.TestCase):
    def test_democritus_parallel_frontiers_capture_parallel_agents(self) -> None:
        workflow = build_democritus_workflow()

        frontiers = tuple(tuple(agent.name for agent in frontier) for frontier in workflow.parallel_frontiers())

        self.assertEqual(frontiers[0], ("document_collection_agent",))
        self.assertEqual(
            frontiers[1],
            ("document_normalization_agent", "document_index_agent"),
        )
        self.assertIn("relational_triple_extractor_agent", frontiers[2])
        self.assertEqual(frontiers[-1], ("executive_summary_agent",))

    def test_democritus_attention_depends_on_document_collection(self) -> None:
        workflow = build_democritus_workflow()

        dependencies = {agent.name for agent in workflow.dependencies_of("relational_triple_extractor_agent")}

        self.assertIn("document_collection_agent", dependencies)
        self.assertIn("document_normalization_agent", dependencies)
        self.assertIn("document_index_agent", dependencies)

    def test_basket_rocket_parallel_frontiers_capture_parallel_agents(self) -> None:
        workflow = build_basket_rocket_workflow()

        frontiers = tuple(tuple(agent.name for agent in frontier) for frontier in workflow.parallel_frontiers())

        self.assertEqual(frontiers[0], ("filing_collection_agent",))
        self.assertEqual(
            frontiers[1],
            ("filing_chunking_agent", "basket_artifact_builder_agent"),
        )
        self.assertEqual(frontiers[-1], ("workflow_reporting_agent",))

    def test_diffusion_objects_are_added_as_artifacts(self) -> None:
        workflow = build_basket_rocket_workflow()

        artifact_names = {artifact.name for artifact in workflow.artifacts}

        self.assertIn("psr_diffusion", artifact_names)
        self.assertIn("workflow_candidate_diffusion", artifact_names)


if __name__ == "__main__":
    unittest.main()
