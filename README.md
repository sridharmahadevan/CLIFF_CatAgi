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

If you are new to the repo, start here first:

- `docs/first_10_minutes.md`

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

## Route Integration Table

| Route | What it does | Works with core repo only? | Optional repos / runtimes |
| --- | --- | --- | --- |
| `course_demo` | Runs textbook-linked demos, recommendations, project ideas, and code snippets | Partly | `Category-Theory-for-AGI-UMass-CMPSCI-692CT`; for Julia paths also `FunctorFlow.jl`, optionally `Julia FF`, and a Julia runtime |
| `democritus` | Finds studies or documents, runs synthesis, and builds corpus-level claims dashboards | No | `Democritus_OpenAI`; OpenAI API access for LLM-backed stages |
| `basket_rocket_sec` | Recovers workflows from SEC filings and builds BASKET/ROCKET-style dashboards | No | `BASKET`, `brand_democritus_block_denoise` |
| `company_similarity` | Compares companies through the diffusion/manifold pipeline and links back to the textbook | No | `brand_democritus_block_denoise` and a Python environment with its dependencies |
| `product_feedback` | Builds product-feedback syntheses, workflows, and causal hypotheses with textbook pointers | Mostly | No extra repo for the basic route; external review sources may still matter depending on retrieval path |
| `culinary_tour` | Demonstrates conscious message-passing through itinerary planning with textbook backstops | Yes | None for the core demo path |

Quick rule of thumb:

- start with `product_feedback` or `culinary_tour` if you want a lightweight first run
- add `course_demo` next if you want the clearest textbook experience
- add `democritus`, `basket_rocket_sec`, and `company_similarity` when you want the full research workflow stack

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

## Install Matrix

Use the same base Python environment for every setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 1. Core Only

Best for:

- router testing
- textbook backstops
- product feedback demos
- culinary tour demos
- basic CLIFF UI checks

Needed:

- this repo
- `catagi.pdf` in the repo root, or `CLIFF_BOOK_PDF_PATH`

Useful checks:

```bash
python3 -m unittest tests.test_textbook_backstop tests.test_query_router_agentic tests.test_cliff
```

### 2. Course Demo Setup

Best for:

- textbook-guided course demos
- project ideas
- learning guides
- PyTorch snippet walkthroughs

Also needed:

- `Category-Theory-for-AGI-UMass-CMPSCI-692CT`

Resolution options:

- sibling repo beside `CLIFF_CatAgi`
- `third_party/Category-Theory-for-AGI-UMass-CMPSCI-692CT`
- `CLIFF_COURSE_REPO_ROOT=/path/to/Category-Theory-for-AGI-UMass-CMPSCI-692CT`

Useful checks:

```bash
python3 -m unittest tests.test_course_demo_agentic tests.test_query_router_agentic
```

### 3. Democritus Setup

Best for:

- multi-document synthesis
- study retrieval and corpus gluing
- CSQL-backed textbook-grounded analysis

Also needed:

- `Democritus_OpenAI`
- OpenAI API access for the LLM-backed steps

Resolution options:

- sibling repo beside `CLIFF_CatAgi`
- `third_party/Democritus_OpenAI`
- `CLIFF_DEMOCRITUS_ROOT=/path/to/Democritus_OpenAI`

Optional seed corpus path:

- `CLIFF_DEMOCRITUS_PDF_ROOT=/path/to/pdf/root`

Useful checks:

```bash
python3 -m unittest tests.test_democritus_agentic tests.test_democritus_query_agentic
```

### 4. BASKET/ROCKET And Company Similarity Setup

Best for:

- SEC workflow recovery
- company diffusion comparisons
- finance-oriented dashboards

Also needed:

- `BASKET`
- `brand_democritus_block_denoise`

Resolution options:

- sibling repos beside `CLIFF_CatAgi`
- `third_party/BASKET`
- `third_party/brand_democritus_block_denoise`
- env vars:
  - `CLIFF_BASKET_ROOT`
  - `CLIFF_BRAND_PANEL_ROOT`
  - `CLIFF_BRAND_PIPELINE_PYTHON` if the company-similarity backend needs a dedicated interpreter

Useful checks:

```bash
python3 -m unittest tests.test_basket_rocket_sec_agentic tests.test_query_router_agentic tests.test_cliff
```

### 5. Julia Setup

Best for:

- Julia KET demos
- Julia causal-semantics demos
- side-by-side Julia/Python educational comparisons

Also needed:

- `FunctorFlow.jl`
- optionally `Julia FF`
- a working Julia runtime

Resolution options:

- sibling repos beside `CLIFF_CatAgi`
- `third_party/FunctorFlow.jl`
- `third_party/Julia FF`
- env vars:
  - `CLIFF_JULIA_REPO_ROOT`
  - `CLIFF_JULIA_EXAMPLES_ROOT`
  - `CLIFF_JULIA_DEPOT_PATH`
  - `CLIFF_JULIA_BIN`
  - `CLIFF_JULIAUP_BIN`

Useful checks:

```bash
python3 -m unittest tests.test_course_demo_agentic
```

### 6. Full Textbook Interface Setup

Best for:

- the full CLIFF_CatAgi vision
- private multi-machine testing before public release

Needed:

- this repo
- `catagi.pdf`
- course repo
- Democritus repo
- BASKET/ROCKET-related repos
- Julia repos if you want both language paths

Recommended smoke queries:

- `Explain the Geometric Transformer on the Sudoku problem`
- `What demo should I use for causality?`
- `Show me the Julia version of KET`
- `How similar is Adobe to Nike?`
- `Give me 5 studies of global warming and synthesize their joint claims`
- `How easy is it to drive a Tesla Model 3?`

## Optional Integrations

`CLIFF_CatAgi` is designed to work even when some supporting repos are absent.
Routes should degrade gracefully and explain what is missing.

Optional sibling or `third_party/` repos:

- `Democritus_OpenAI`
- `BASKET`
- `brand_democritus_block_denoise`
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
- `CLIFF_BRAND_PIPELINE_PYTHON`
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
