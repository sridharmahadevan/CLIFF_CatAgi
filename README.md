# CLIFF_CatAgi

`CLIFF_CatAgi` is a textbook-centric version of CLIFF:
the **Conscious Layer Interface to Functor Flow** for *Categories for AGI*.
It teaches ideas from the book through runnable demos, project suggestions,
product/company feedback views, Democritus synthesis, SEC workflow analysis,
and book annotations.

The key product idea is simple:

- every route should point back to the textbook
- textbook chapters should connect to runnable demos
- demos should connect to code snippets and project ideas
- external research engines should be optional integrations, not hidden assumptions

This repo is intended to be the clean public-facing interface layer.

## Core Product Picture

CLIFF is best understood as a conscious interface sitting on top of a deeper
Functor Flow causal engine.

In practice, many apparently different queries collapse toward the same deeper
pattern:

- retrieve evidence
- build causal state
- synthesize a best-so-far answer
- refine, compare, or visualize that answer

That is why Democritus matters so much in this repo. Multi-study synthesis,
company similarity, and several evidence-heavy routes eventually rely on the
same causal-state-building machinery.

## Runtime Expectations

CLIFF now exposes both execution depth and route latency more explicitly in the
GUI.

For routes that emit LLM usage telemetry, CLIFF also shows token counts and a
rough estimated API cost in the session UI. The cost number is only an
approximation, intended as a budgeting aid for students and researchers rather
than exact billing.

- `Quick answer`
  lightweight routes such as textbook lookup and some filing-oriented lookups
- `Longer analysis`
  routes such as product feedback that do structured evidence synthesis
- `Deep research`
  routes such as `democritus` and `company_similarity`, which may still take
  several minutes even in `quick` mode

`Quick` vs `deep` is therefore not the same as “fast” vs “slow.”
For the deep-research routes, `quick` means an earlier, lighter, best-so-far
answer path, not an instant answer.

## Core Design

The major architectural shift from the earlier `FunctorFlow_v1` release is that
CLIFF now treats Kan-extension-style attention and diffusion-style gluing as
agentic workflow principles, not only as model-level ideas.

The first public-facing note for that transition is:

- `docs/agentic_kan_architecture.md`

The current design note for the next Democritus/CLIFF synthesis milestone is:

- `docs/homotopy_aware_synthesis.md`

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

- start with `culinary_tour`, `course_demo`, or lightweight textbook-guided prompts if you want the fastest first run
- try `product_feedback` next if you want a medium-weight route that still feels interactive
- add `democritus` and `company_similarity` when you want the full deep-research workflow stack
- use `basket_rocket_sec` when you specifically want filing workflow recovery

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

### Headless / Remote CLIFF Sessions

CLIFF can run on a headless server while you use the interface from a browser
on your laptop or desktop.

This is useful when:

- the compute machine has no display attached
- you are SSH'd into a remote GPU or DGX box
- you want CLIFF to use remote compute resources but keep the interface local

The basic pattern is:

1. run CLIFF on the remote machine
2. keep the CLIFF process alive there
3. forward the CLIFF port over SSH
4. open the forwarded URL in a browser on your local machine

Start CLIFF on the remote machine without trying to open a browser:

```bash
python3 -m functorflow_v3.cliff \
  --outdir /tmp/cliff-session \
  --host 127.0.0.1 \
  --port 8765 \
  --no-browser
```

CLIFF will print JSON similar to:

```json
{
  "system_name": "CLIFF",
  "session_url": "http://127.0.0.1:8765/",
  "session_listen_url": "http://127.0.0.1:8765/",
  "session_outdir_root": "/tmp/cliff-session",
  "mode": "interactive_session"
}
```

Keep that remote CLIFF process running. Then, from your local machine, create
an SSH tunnel to the same port:

```bash
ssh -N -L 8765:127.0.0.1:8765 your-remote-host
```

After the tunnel is up, open the CLIFF session locally:

```bash
open http://127.0.0.1:8765/
```

Or open the same URL manually in your browser:

```text
http://127.0.0.1:8765/
```

If your SSH path uses a jump host, tunnel through it explicitly:

```bash
ssh -N -J your-login-host -L 8765:127.0.0.1:8765 your-remote-host
```

#### Notes

- `--no-browser` only disables automatic browser launching on the remote host. It does not open a browser on your laptop for you.
- `--host 127.0.0.1` is usually the safest choice for SSH tunneling because CLIFF only listens on the remote loopback interface.
- `--port 8765` is just an example; any unused port is fine as long as the CLIFF process and the SSH tunnel use the same one.
- `session_listen_url` tells you where CLIFF is actually listening on the remote machine.
- `session_url` is the URL CLIFF wants to advertise to the user. By default it matches the listen URL.

#### Reverse Proxy Or External Tunnel

If you expose CLIFF through a reverse proxy, Tailscale funnel, cloud tunnel, or
some other externally reachable URL, pass that public address so the printed
session link matches what users should open:

```bash
python3 -m functorflow_v3.cliff \
  --outdir /tmp/cliff-session \
  --host 127.0.0.1 \
  --port 8765 \
  --no-browser \
  --public-url https://your-public-cliff-url.example.com/
```

#### Quick Troubleshooting

- If `curl -I http://127.0.0.1:8765/` on your local machine says `Couldn't connect to server`, the SSH tunnel is not up or is pointed at the wrong host.
- If `curl -I` returns `501 Unsupported method ('HEAD')`, that is usually fine. The lightweight server may not implement `HEAD`; try `open http://127.0.0.1:8765/` or `curl http://127.0.0.1:8765/` instead.
- If the browser still cannot load CLIFF, make sure the remote CLIFF process is still running and that the local tunnel command is still active.

### Saved Runs And Session Restore

CLIFF now treats the `--outdir` tree as a reusable run workspace, not just a
temporary dump folder.

- each submitted query gets its own run folder under the chosen `--outdir`
- CLIFF writes a `cliff_run_record.json` alongside the route artifacts for that
  run
- when you open CLIFF again, the launcher rescans the current run root, the
  default archive root at `~/Downloads/CLIFF_runs_archive`, and any extra paths
  listed in `CLIFF_RUN_ARCHIVE_ROOTS`
- saved runs show up in the archived-runs panel, where you can reopen the old
  artifact or queue the same query again as a rerun

In practice, this means you can stop and restart CLIFF without losing the runs
you already completed, and you can use older runs as checkpoints for later
comparison or follow-up work.

Export a compact public Democritus example bundle from a saved run:

```bash
python3 -m functorflow_v3.democritus_example_export \
  --run-dir /path/to/democritus \
  --output-dir examples/democritus/my_saved_run \
  --document-ranks 2,3,4,5,6 \
  --copy-manifold-images 3 \
  --force
```

This keeps the GitHub-facing artifact small by exporting query metadata,
selected-document summaries, stage timing summaries, and a few representative
images while excluding the heavyweight PDFs, sweep outputs, PKL state, SQLite
bundles, and large report assets from the original run. Use
`--document-ranks` when you want a curated public subset rather than the full
retrieval set from the saved run.

Export a compact public BASKET/ROCKET example bundle from legacy company artifacts:

```bash
python3 -m functorflow_v3.basket_rocket_example_export \
  --company adobe \
  --extractor-dir ../BASKET/outputs/tenk_rawpdf_fullpanel_monitored \
  --reranking-dir ../BASKET/outputs/rocket_fullpanel_financial_real \
  --company-viz-dir ../BASKET/outputs/rocket_company_viz_financial_real \
  --psr-company-dir ../BASKET/outputs/psr_rocket_variant_comparison_20260322/companies \
  --diffusion-dir ../brand_democritus_block_denoise_complete/outputs/adobe/temporal_denoiser/infer \
  --radar-dir ../brand_democritus_block_denoise_complete/outputs/adobe/survival_radar \
  --output-dir examples/basket_rocket/adobe_financial_reranking \
  --force
```

This keeps the GitHub-facing artifact small by exporting a company-level
snapshot: sanitized extractor metadata, reranking summaries, representative
changed statements, GitHub-renderable visualization summaries, and selected
diffusion/radar dashboards while excluding the raw PDFs, full JSONL panels,
and bulky intermediate outputs.

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
- single-document analysis from a direct article or PDF URL
- single-document analysis from a local uploaded PDF path
- CSQL-backed textbook-grounded analysis
- building the causal substrate that several heavier CLIFF routes depend on

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

### Writing Better Democritus Queries

Democritus works best when the query is narrow enough that retrieval and
synthesis can converge on one shared evidence question.

In practice, the strongest Democritus prompts usually specify:

- a concrete phenomenon or intervention
- a concrete outcome
- a population, system, or setting
- optionally a study type such as randomized trials, observational studies,
  systematic reviews, or meta-analyses

This general pattern works well:

```text
Analyze N recent [study type] on X effect on Y in Z, and synthesize their joint support.
```

Good examples:

```text
Analyze 5 recent randomized trials of semaglutide for weight loss in adults with obesity and synthesize their joint support.
```

```text
Analyze 5 recent studies on how rising ocean temperatures affect wild fish population abundance and distribution, and synthesize their joint support.
```

```text
Analyze 5 recent studies on the effect of rising ocean temperatures on marine fisheries yields, and synthesize their joint support.
```

Less effective broad prompts often mix several adjacent topics into one run. For
example, a prompt about fish populations, coral mortality, aquaculture, human
pathogens, and fisheries economics may all live in the same broad climate/ocean
neighborhood while still failing to form one coherent evidence corpus.

If Democritus shows topic drift, fragmented topic partitions, or many unrelated
singleton partitions, try narrowing one or more of these axes:

- intervention or exposure
- outcome
- organism or population
- geography or environment
- study type

Examples of useful rewrites:

- instead of `Analyze 5 recent studies of the impact of fish populations of rising ocean temperatures...`
- try `Analyze 5 recent studies on how rising ocean temperatures affect wild fish population abundance and distribution...`
- instead of `Analyze 5 recent studies of GLP-1 and synthesize their joint support`
- try `Analyze 5 recent randomized trials of GLP-1 receptor agonists for weight loss in adults with obesity...`

One practical rule of thumb:

- if the query could reasonably retrieve papers from several neighboring subfields, it is probably too broad for a clean first Democritus run

### 4. BASKET/ROCKET And Company Similarity Setup

Best for:

- SEC workflow recovery
- company diffusion comparisons
- finance-oriented dashboards
- testing the deeper route stack that eventually leans on Democritus-style
  causal-state construction

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

## Current UX Notes

- The launcher banner now expands `CLIFF` as `Conscious Layer Interface to Functor Flow`.
- `Democritus` quick mode is intended to return a useful provisional answer
  sooner and then improve it as more evidence is processed.
- `Company similarity` now reports ETA, parallelism, and inner Democritus build
  stages, but it remains the slowest major route and should still be treated as
  deep research.

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
