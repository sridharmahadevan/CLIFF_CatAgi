# Democritus Example Bundle

This is a compact, GitHub-friendly export of a saved CLIFF Democritus run.
The heavyweight artifacts from the original run were intentionally excluded:
PDF inputs, sweep outputs, PKL state, SQLite databases, and large report assets.

## Query

- Query: Analyze 10 recent studies of declining fish population due to ocean temperature warming and synthesize their joint support
- Retrieval query: declining fish population due ocean temperature warming
- Execution mode: deep
- Retrieval backend: scholarly
- Target documents: 10
- Selected documents: 5
- Included manifold images: 3

## Curation

- Source run selected documents: 11
- Public release subset: 5 documents
- This bundle keeps the most on-topic studies from the original saved run and omits obvious drift, corrections, and review-only artifacts.

## Included Files

- `query_plan.json`: sanitized request and retrieval plan
- `selected_documents.json`: compact metadata for the selected studies
- `batch_stage_summary.json`: aggregated stage timings across the run
- `documents/`: one Markdown executive summary per selected study
- `images/`: a small sample of 2D manifold plots

## Selected Documents

| Rank | Year | Score | Backend | Title | Summary |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026 | 10.65 | europe_pmc | Signals From the Southern Edge: Demographic Effects of Ocean Warming on Two Cold-Adapted Seabird Species in the Gulf of Maine. | [summary](documents/01_signals-from-the-southern-edge-demographic-effects-of-ocean-warm.md) |
| 2 | 2025 | 2.45 | europe_pmc | The combined impact of fisheries and climate change on future carbon sequestration by oceanic macrofauna. | [summary](documents/02_the-combined-impact-of-fisheries-and-climate-change-on-future.md) |
| 3 | 2025 | 0.92625 | europe_pmc | Climate-driven shifts in marine habitat explain recent declines of Japanese Chum salmon. | [summary](documents/03_climate-driven-shifts-in-marine-habitat-explain-recent-declines.md) |
| 4 | 2026 | 0.27625 | europe_pmc | Biodiversity changes in Arctic coastal ecosystems under borealization. | [summary](documents/04_biodiversity-changes-in-arctic-coastal-ecosystems-under-borealiz.md) |
| 5 | 2024 | 9.703999999999999 | crossref | Do fishers follow fish displaced by climate warming? | [summary](documents/05_do-fishers-follow-fish-displaced-by-climate-warming.md) |

## Stage Summary

| Agent | Records | Avg Sec | Max Sec | Total Sec |
| --- | --- | --- | --- | --- |
| causal_statement_agent | 5 | 235.435 | 251.048 | 1177.176 |
| causal_question_agent | 5 | 108.261 | 112.92 | 541.306 |
| topic_graph_agent | 5 | 51.901 | 62.362 | 259.505 |
| credibility_bundle_agent | 5 | 13.963 | 14.32 | 69.815 |
| manifold_builder_agent | 5 | 13.093 | 15.29 | 65.464 |
| root_topic_discovery_agent | 5 | 11.916 | 23.432 | 59.581 |
| lcm_scoring_agent | 5 | 3.613 | 4.058 | 18.067 |
| lcm_sweep_agent | 5 | 0.974 | 1.452 | 4.868 |

## Representative Claims

### 1. Signals From the Southern Edge: Demographic Effects of Ocean Warming on Two Cold-Adapted Seabird Species in the Gulf of Maine.
- Ocean warming in the Gulf of Maine causes a shift from high-energy cold-water fish to lower-quality warm-water prey species.
- Ocean warming in the Gulf of Maine causes a shift from high-energy cold-water prey species to lower-quality warm-water prey species, leading to nutritional stress for Atlantic P...
- 2D manifold image: `images/01_signals-from-the-southern-edge-demographic-effects-of-ocean-warm_manifold_2d.png`

### 2. The combined impact of fisheries and climate change on future carbon sequestration by oceanic macrofauna.
- Rising ocean temperature reduces marine macrofauna biomass by altering metabolic rates and size structure.
- This reduction in macrofauna biomass leads to decreased carbon export in ocean ecosystems.
- 2D manifold image: `images/02_the-combined-impact-of-fisheries-and-climate-change-on-future_manifold_2d.png`

### 3. Climate-driven shifts in marine habitat explain recent declines of Japanese Chum salmon.
- Overfishing reduces the population size of Japanese chum salmon, which decreases their resilience to climate-driven habitat shifts.
- Overfishing affects the ability of salmon stocks to recover from declines caused by altered feeding and overwintering habitats due to rising ocean temperatures and marine heatwa...
- 2D manifold image: `images/03_climate-driven-shifts-in-marine-habitat-explain-recent-declines_manifold_2d.png`

### 4. Biodiversity changes in Arctic coastal ecosystems under borealization.
- Climate warming causes the northward expansion of boreal species in Arctic coastal ecosystems studied by the University Centre in Svalbard.
- The expansion of boreal species causes shifts in species composition that alter Arctic coastal biodiversity.

### 5. Do fishers follow fish displaced by climate warming?
- Climate warming causes altered ocean temperature and salinity, which leads to shifted Atlantic cod spawning suitability areas.
- Climate warming causes shifts in cod spawning habitats, which increases catch potential in newly suitable areas.
