# CLIFF_CatAgi

`CLIFF_CatAgi` is a textbook-centric version of CLIFF:
the Consciousness Layer Interface that teaches ideas from *Categories for AGI*
through runnable demos, project suggestions, product/company feedback views,
Democritus synthesis, SEC workflow analysis, and book annotations.

The key product idea is simple:

- every route should point back to the textbook
- textbook chapters should connect to runnable demos
- demos should connect to code snippets and project ideas
- external research engines should be optional integrations, not hidden assumptions

This repo is intended to be the clean public-facing interface layer.

## Core Design

The major architectural shift from the earlier `FunctorFlow_v1` release is that
CLIFF now treats Kan-extension-style attention and diffusion-style gluing as
agentic workflow principles, not only as model-level ideas.

The first public-facing note for that transition is:

- `docs/agentic_kan_architecture.md`

## What This Repo Contains

- `functorflow_v3/`: the current CLIFF package and route logic
- `tests/`: regression coverage for routing, course demos, textbook backstops, and major workflows
- `catagi.pdf`: the textbook artifact used for chapter recommendations

## Supported CLIFF Modes

- `course_demo`: book-guided course demos, project ideas, learning guides, and Julia/PyTorch code snippets
- `democritus`: multi-document synthesis with textbook backstops
- `basket_rocket_sec`: SEC workflow analysis with textbook backstops
- `company_similarity`: cross-company diffusion comparison with textbook backstops
- `product_feedback`: review synthesis with textbook backstops
- `culinary_tour`: consciousness-style itinerary demos, also tied back to the textbook

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run CLIFF:

```bash
python3 -m functorflow_v3.cliff --outdir /tmp/cliff-session
```

## Optional Integrations

`CLIFF_CatAgi` is designed to work even when some supporting repos are absent.
Routes should degrade gracefully and explain what is missing.

Optional sibling or `third_party/` repos:

- `democritus_v2_public` or `Democritus_OpenAI`
- `BASKET`
- `brand_democritus_block_denoise`
- `brand_awareness_democritus`
- `Category-Theory-for-AGI-UMass-CMPSCI-692CT`
- `FunctorFlow.jl`
- `Julia FF`

The resolver module is:

- `functorflow_v3/repo_layout.py`

It supports either:

1. bundling dependencies under `third_party/`
2. keeping them as sibling repos beside `CLIFF_CatAgi`
3. overriding them with environment variables

Environment variables:

- `CLIFF_BOOK_PDF_PATH`
- `CLIFF_DEMOCRITUS_ROOT`
- `CLIFF_DEMOCRITUS_PDF_ROOT`
- `CLIFF_BASKET_ROOT`
- `CLIFF_BRAND_PANEL_ROOT`
- `CLIFF_BRAND_AWARENESS_ROOT`
- `CLIFF_COURSE_REPO_ROOT`
- `CLIFF_JULIA_REPO_ROOT`
- `CLIFF_JULIA_EXAMPLES_ROOT`
- `CLIFF_JULIA_DEPOT_PATH`
- `CLIFF_JULIA_BIN`
- `CLIFF_JULIAUP_BIN`

## Suggested Public Release Strategy

For a first GitHub release, treat this repo as the interface layer and keep the
heavier engines optional:

- CLIFF explains concepts and routes queries
- the textbook provides conceptual grounding
- external repos provide specialized execution backends

That keeps setup lighter and makes the architecture much easier to explain.

## Smoke Tests

```bash
python3 -m unittest tests.test_textbook_backstop tests.test_query_router_agentic tests.test_cliff
```

For broader local verification:

```bash
python3 -m unittest
```
