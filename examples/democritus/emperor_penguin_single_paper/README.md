# Democritus Example Bundle

This is a compact, GitHub-friendly export of a saved CLIFF Democritus run.
The heavyweight artifacts from the original run were intentionally excluded:
PDF inputs, sweep outputs, PKL state, SQLite databases, and large report assets.

## Query

- Query: Analyze the PDF article at WaPo_EmperorPenguin.pdf
- Retrieval query: pdf users sridharmahadevan desktop wapo emperorpenguin pdf
- Execution mode: quick
- Retrieval backend: direct_file
- Target documents: 1
- Selected documents: 1
- Included manifold images: 1

## Included Files

- `query_plan.json`: sanitized request and retrieval plan
- `selected_documents.json`: compact metadata for the selected studies
- `batch_stage_summary.json`: aggregated stage timings across the run
- `documents/`: one Markdown executive summary per selected study
- `images/`: a small sample of 2D manifold plots

## Selected Documents

| Rank | Year | Score | Backend | Title | Summary |
| --- | --- | --- | --- | --- | --- |
| 1 | - | 1000.0 | direct_file | WaPo EmperorPenguin | [summary](documents/01_wapo-emperorpenguin.md) |

## Stage Summary

| Agent | Records | Avg Sec | Max Sec | Total Sec |
| --- | --- | --- | --- | --- |
| causal_statement_agent | 1 | 327.766 | 327.766 | 327.766 |
| causal_question_agent | 1 | 171.118 | 171.118 | 171.118 |
| topic_graph_agent | 1 | 56.713 | 56.713 | 56.713 |
| root_topic_discovery_agent | 1 | 23.699 | 23.699 | 23.699 |
| manifold_builder_agent | 1 | 19.905 | 19.905 | 19.905 |
| manifold_visualization_agent | 1 | 1.113 | 1.113 | 1.113 |
| topos_slice_agent | 1 | 0.887 | 0.887 | 0.887 |
| triple_extraction_agent | 1 | 0.247 | 0.247 | 0.247 |

## Representative Claims

### 1. WaPo EmperorPenguin
- Rising ocean temperatures lead to krill moving to deeper waters, which reduces the availability of food for Antarctic fur seals.
- Loss of coastal sea ice reduces the availability of stable breeding platforms for emperor penguins, which leads to breeding failure.
- 2D manifold image: `images/01_wapo-emperorpenguin_manifold_2d.png`
