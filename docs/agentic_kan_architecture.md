# CLIFF_CatAgi Architecture

## Why This Repo Exists

`CLIFF_CatAgi` is not just another packaging of FunctorFlow.
It is the textbook-facing layer for *Categories for AGI*:
a system that teaches the book by letting users ask natural-language questions,
then answering through demos, synthesis workflows, code snippets, project ideas,
and explicit chapter recommendations.

That educational framing is the main difference from earlier FunctorFlow
releases.

The current acronym expansion makes that architectural choice more explicit:

- `CLIFF`: `Conscious Layer Interface to Functor Flow`

The conscious layer is the user-facing bounded interface.
Functor Flow is the deeper causal and compositional engine beneath it.

## From FunctorFlow v1 To CLIFF_CatAgi

The original `FunctorFlow_v1` release emphasized the core semantic idea:

- attention as a categorical aggregation mechanism
- diffusion as a structured reconciliation mechanism
- workflows as compositional diagrams rather than flat scripts

In `CLIFF_CatAgi`, those ideas become operational and agentic.

Instead of treating a workflow as one model pass, CLIFF treats a query as a
small society of agents:

- retrieval agents gather evidence or runnable artifacts
- normalization agents convert evidence into a common structure
- synthesis agents glue partial views into a coherent object
- dashboard agents surface a bounded conscious view to the user

This is the practical shift from FunctorFlow v1 to the current v3-era CLIFF
design.

## The Kan Extension View

The core intuition is:

- attention is not only token-to-token weighting
- it can be understood as a Kan-style aggregation of information along a map
- diffusion is not only denoising in latent space
- it can be understood as a limit-style reconciliation of multiple partial
  states into a shared object

CLIFF uses that perspective at the workflow level.

### Kan Extension Attention

In the CLIFF framing, an agent often receives information from many upstream
artifacts:

- documents
- retrieved reviews
- filing chunks
- notebook demos
- prior broadcasts in the conscious layer

The agent’s job is to aggregate those inputs into a locally meaningful view.
This is the broad role of Kan-extension-style attention:

- a source space of partial observations
- a relation or incidence structure telling us what should be aggregated
- a target space where the new summary, action, or explanation lives

This is why KET and related transformer demos matter so much in the repo:
they are not isolated model curiosities, but local exemplars of the same
aggregation principle used more broadly in CLIFF.

### Diffusion As Gluing

Multiple agents often produce partially compatible outputs:

- different studies support overlapping claims
- several reviews suggest conflicting usage narratives
- filing worksets imply partially aligned workflow motifs
- multiple demos or code snippets illuminate the same textbook topic

CLIFF treats this as a gluing problem.

The diffusion metaphor here is not merely stochastic generation.
It is a structured reconciliation process:

- collect partial outputs
- align them
- detect incompatibilities
- produce a more coherent object for the next stage

That is why the codebase frequently uses synthesis, corpus, convergence, and
gluing language. The mathematics motivating those moves comes from the same
categorical picture that inspired FunctorFlow from the start.

## Why The Consciousness Layer Matters

CLIFF adds a user-facing interpretation:

- unconscious agents do the heavy distributed work
- the user sees a bounded dashboard or artifact surface
- successful results are promoted into that visible workspace

This is the reason for the name:

- `CLIFF`: Conscious Layer Interface to Functor Flow

The consciousness metaphor is not decorative. It organizes the product design:

- route the query
- let specialized agents do local work
- synthesize the outputs
- show a stable conscious surface
- point the user back to the textbook

Recent work on the GUI and long-running routes has made this more literal in
the product:

- the launcher banner now expands the acronym explicitly
- the GUI surfaces ETA and parallelism as unconscious-layer broadcasts
- long-running routes show best-so-far answers rather than hiding all progress
  until the end

## Textbook-Centric Design

In this repo, every major route should eventually support four linked views:

1. `Read`
   relevant chapter or section in `catagi.pdf`
2. `Run`
   the closest demo or executable workflow
3. `Inspect`
   the key code snippet in PyTorch or Julia
4. `Extend`
   project suggestions or follow-up experiments

That is the product vision crystallized in this release.

## Route-Level Interpretation

The same architecture appears in different forms across routes:

- `course_demo`
  KET, Geometric Transformer, sheaves, causality, and Julia demos
- `democritus`
  multi-document causal synthesis and corpus-level gluing
- `basket_rocket_sec`
  workflow recovery over SEC filings
- `company_similarity`
  cross-company diffusion and manifold comparison
- `product_feedback`
  structured review synthesis with workflows and causal hypotheses
- `culinary_tour`
  a lighter consciousness demo showing message-passing and composed planning

Some routes are closer to the mathematical core than others, but they all
serve the same educational purpose: expose the textbook ideas through action.

There is also an important asymmetry between routes:

- some routes are genuinely lightweight textbook or demo lookups
- some routes are medium-weight structured syntheses
- some routes are deep-research workflows that eventually reduce to building
  causal state from evidence

In practice, `democritus` is now the clearest expression of that causal core,
and `company_similarity` often becomes a Democritus-like problem internally
because it must build local company causal state before it can compare firms.

That observation is not incidental.
It is one of the clearest guides for future optimization work in CLIFF.

## Public Repo Strategy

This repo should be understood as an interface layer, not as a demand that all
supporting systems live in one monolith.

That is why external engines remain optional:

- Democritus
- BASKET/ROCKET
- brand diffusion backends
- the course repo
- FunctorFlow.jl

`CLIFF_CatAgi` is the orchestrator and educational shell around them.

## Recommended Companion Docs

For a stronger public release, the next documentation layer should include:

- an installation matrix: minimal vs full setup
- a route-by-route integration table
- a short tutorial showing how one textbook concept appears in chapter, demo,
  code snippet, and project idea form
- a migration note comparing `FunctorFlow_v1` and `CLIFF_CatAgi`
