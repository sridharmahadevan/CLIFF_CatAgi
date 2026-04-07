"""Canonical FF2 workflow blueprints."""

from __future__ import annotations

from .agentic_workflows import (
    AgentSpec,
    ArtifactSpec,
    AttentionSpec,
    DiffusionSpec,
    AgenticWorkflow,
    build_agentic_workflow,
)


def build_democritus_workflow() -> AgenticWorkflow:
    """Blueprint for an agent-native Democritus pipeline."""

    return build_agentic_workflow(
        name="DemocritusAgenticWorkflow",
        artifacts=(
            ArtifactSpec("document_corpus", "document_bundle"),
            ArtifactSpec("retrieval_manifest", "retrieval_manifest"),
            ArtifactSpec("normalized_documents", "normalized_document_bundle"),
            ArtifactSpec("document_index", "document_index"),
            ArtifactSpec("relational_triples", "relational_triple_set"),
            ArtifactSpec("grounded_triples", "grounded_relational_triple_set"),
            ArtifactSpec("csql_database", "csql_database"),
            ArtifactSpec("topos_state", "topos_state"),
            ArtifactSpec("causal_summary", "causal_summary"),
        ),
        agents=(
            AgentSpec(
                name="document_collection_agent",
                role="collect_public_documents",
                produces=("document_corpus", "retrieval_manifest"),
                capabilities=("web_search", "pdf_acquisition", "provenance_tracking"),
            ),
            AgentSpec(
                name="document_normalization_agent",
                role="normalize_documents",
                consumes=("document_corpus",),
                produces=("normalized_documents",),
                attention_from=("document_collection_agent",),
                capabilities=("ocr_cleanup", "chunking", "canonicalization"),
            ),
            AgentSpec(
                name="document_index_agent",
                role="index_documents",
                consumes=("document_corpus",),
                produces=("document_index",),
                attention_from=("document_collection_agent",),
                capabilities=("indexing", "provenance_alignment"),
            ),
            AgentSpec(
                name="relational_triple_extractor_agent",
                role="extract_relational_triples",
                consumes=("normalized_documents", "document_index"),
                produces=("relational_triples",),
                attention_from=("document_collection_agent",),
                capabilities=("relation_extraction", "evidence_linking"),
            ),
            AgentSpec(
                name="grounding_gluer_agent",
                role="glue_relational_triples",
                consumes=("relational_triples", "retrieval_manifest"),
                produces=("grounded_triples",),
                attention_from=("relational_triple_extractor_agent", "document_collection_agent"),
                capabilities=("gluing", "consistency_checking"),
            ),
            AgentSpec(
                name="csql_builder_agent",
                role="build_csql_database",
                consumes=("grounded_triples",),
                produces=("csql_database",),
                attention_from=("grounding_gluer_agent",),
                capabilities=("schema_construction", "categorical_db_building"),
            ),
            AgentSpec(
                name="topos_reasoner_agent",
                role="perform_topos_reasoning",
                consumes=("csql_database",),
                produces=("topos_state",),
                attention_from=("csql_builder_agent",),
                capabilities=("topos_reasoning", "subobject_analysis"),
            ),
            AgentSpec(
                name="executive_summary_agent",
                role="summarize_causal_state",
                consumes=("topos_state",),
                produces=("causal_summary",),
                attention_from=("topos_reasoner_agent", "document_collection_agent"),
                capabilities=("narrative_synthesis", "causal_reporting"),
            ),
        ),
        attentions=(
            AttentionSpec(
                name="document_normalization_attention",
                target_agent="document_normalization_agent",
                input_artifacts=("document_corpus", "retrieval_manifest"),
                source_agents=("document_collection_agent",),
            ),
            AttentionSpec(
                name="triple_extraction_attention",
                target_agent="relational_triple_extractor_agent",
                input_artifacts=("normalized_documents", "document_index", "retrieval_manifest"),
                source_agents=("document_collection_agent", "document_normalization_agent"),
            ),
            AttentionSpec(
                name="summary_attention",
                target_agent="executive_summary_agent",
                input_artifacts=("topos_state", "retrieval_manifest"),
                source_agents=("topos_reasoner_agent", "document_collection_agent"),
            ),
        ),
        diffusions=(
            DiffusionSpec(
                name="grounded_triple_diffusion",
                target_agent="grounding_gluer_agent",
                input_artifacts=("relational_triples", "retrieval_manifest"),
            ),
            DiffusionSpec(
                name="causal_summary_diffusion",
                target_agent="executive_summary_agent",
                input_artifacts=("topos_state", "retrieval_manifest"),
            ),
        ),
        metadata={"family": "Democritus", "semantic_role": "agentic_blueprint"},
    )


def build_basket_rocket_workflow() -> AgenticWorkflow:
    """Blueprint for an agent-native BASKET/ROCKET pipeline."""

    return build_agentic_workflow(
        name="BasketRocketAgenticWorkflow",
        artifacts=(
            ArtifactSpec("filing_corpus", "filing_bundle"),
            ArtifactSpec("company_context", "company_context_bundle"),
            ArtifactSpec("filing_chunks", "chunked_filings"),
            ArtifactSpec("basket_artifacts", "basket_artifact_bundle"),
            ArtifactSpec("candidate_workflows", "workflow_candidate_set"),
            ArtifactSpec("rocket_rankings", "rocket_ranking_set"),
            ArtifactSpec("psr_models", "predictive_state_model_set"),
            ArtifactSpec("workflow_report", "workflow_report"),
        ),
        agents=(
            AgentSpec(
                name="filing_collection_agent",
                role="collect_10k_filings",
                produces=("filing_corpus", "company_context"),
                capabilities=("sec_retrieval", "metadata_tracking"),
            ),
            AgentSpec(
                name="filing_chunking_agent",
                role="chunk_filings",
                consumes=("filing_corpus",),
                produces=("filing_chunks",),
                attention_from=("filing_collection_agent",),
                capabilities=("sectioning", "chunk_alignment"),
            ),
            AgentSpec(
                name="basket_artifact_builder_agent",
                role="build_basket_artifacts",
                consumes=("filing_corpus", "company_context"),
                produces=("basket_artifacts",),
                attention_from=("filing_collection_agent",),
                capabilities=("artifact_extraction", "statement_grounding"),
            ),
            AgentSpec(
                name="workflow_extraction_agent",
                role="extract_candidate_workflows",
                consumes=("filing_chunks", "basket_artifacts"),
                produces=("candidate_workflows",),
                attention_from=("basket_artifact_builder_agent", "filing_chunking_agent"),
                capabilities=("workflow_extraction", "plan_normalization"),
            ),
            AgentSpec(
                name="rocket_reranking_agent",
                role="rerank_workflows",
                consumes=("candidate_workflows", "company_context"),
                produces=("rocket_rankings",),
                attention_from=("workflow_extraction_agent", "filing_collection_agent"),
                capabilities=("reward_scoring", "plan_reranking"),
            ),
            AgentSpec(
                name="psr_modeling_agent",
                role="construct_predictive_state_models",
                consumes=("rocket_rankings", "basket_artifacts"),
                produces=("psr_models",),
                attention_from=("rocket_reranking_agent", "basket_artifact_builder_agent"),
                capabilities=("psr_construction", "state_abstraction"),
            ),
            AgentSpec(
                name="workflow_reporting_agent",
                role="summarize_agentic_workflow_state",
                consumes=("psr_models", "rocket_rankings"),
                produces=("workflow_report",),
                attention_from=("psr_modeling_agent", "rocket_reranking_agent"),
                capabilities=("report_generation", "decision_support"),
            ),
        ),
        attentions=(
            AttentionSpec(
                name="workflow_extraction_attention",
                target_agent="workflow_extraction_agent",
                input_artifacts=("filing_chunks", "basket_artifacts", "company_context"),
                source_agents=("filing_chunking_agent", "basket_artifact_builder_agent"),
            ),
            AttentionSpec(
                name="psr_attention",
                target_agent="psr_modeling_agent",
                input_artifacts=("rocket_rankings", "basket_artifacts"),
                source_agents=("rocket_reranking_agent", "basket_artifact_builder_agent"),
            ),
        ),
        diffusions=(
            DiffusionSpec(
                name="workflow_candidate_diffusion",
                target_agent="workflow_extraction_agent",
                input_artifacts=("filing_chunks", "basket_artifacts"),
            ),
            DiffusionSpec(
                name="psr_diffusion",
                target_agent="psr_modeling_agent",
                input_artifacts=("rocket_rankings", "basket_artifacts"),
            ),
        ),
        metadata={"family": "BASKET/ROCKET", "semantic_role": "agentic_blueprint"},
    )
