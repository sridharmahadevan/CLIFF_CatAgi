# First 10 Minutes With CLIFF_CatAgi

This guide is for a first-time user who wants to understand what CLIFF does
without reading the whole codebase.

## 1. What CLIFF_CatAgi Is

Think of CLIFF as a textbook interface.

More precisely, CLIFF now frames itself as the **Conscious Layer Interface to
Functor Flow**.

You ask a question in natural language, and CLIFF tries to respond through one
or more of these layers:

- a pointer to the right section of `catagi.pdf`
- a runnable demo
- a synthesis dashboard
- a code snippet
- a project suggestion

The big idea is that the repo is not only code.
It is a teaching surface for *Categories for AGI*.

It is also useful to keep one architectural fact in mind from the start:
many of the heavier routes eventually reduce to building causal state from
evidence, which is why Democritus sits so centrally in the system.

## 2. The Fastest Useful Setup

If you just want to get oriented, start with the core setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python3 -m functorflow_v3.cliff --outdir /tmp/cliff-session
```

This gives you the CLIFF interface and the routes that do not require every
optional backend to be installed.

If you launch the full CLIFF GUI, you will now also see:

- the acronym expansion in the banner
- `Quick` vs `Deep` execution depth
- a latency guide that distinguishes lightweight routes from deep-research
  routes

One practical detail is easy to miss: the `--outdir` folder is also where CLIFF
stores per-query run folders and run records. When you come back in a later
session, CLIFF rescans those saved runs, restores them into the launcher, and
lets you reopen or rerun earlier work from the archived-runs list.

## 3. How To Think About The Repo

Do not start by reading everything in `functorflow_v3/`.

A better mental model is:

- `cliff.py`
  starts the user-facing interface
- `query_router_agentic.py`
  decides which route should answer your query
- route modules such as `course_demo_agentic.py` or `product_feedback_query_agentic.py`
  do the specialized work
- `textbook_backstop.py`
  points the result back to the book

That is enough to understand the product flow.

## 4. Your First Queries

Try one query from each style of interaction.

### Learn A Concept

```text
Explain how the Kan Extension Transformer works
```

What to expect:

- a book-guided learning response
- code snippets
- a recommended demo

### Run A Demo

```text
Explain the Geometric Transformer on the Sudoku problem
```

What to expect:

- a pointer to the relevant chapter
- a runnable course demo
- a result dashboard

### Explore A Topic

```text
What demo should I use for causality?
```

What to expect:

- several demo suggestions
- guidance on where to start in the textbook

### Get A Project Idea

```text
I would like a project suggestion that applies the Kan Extension Transformer
```

What to expect:

- book chapters to read
- starter demo
- project ideas
- deliverables and stretch goals

### See A Non-Course Route

```text
How easy is it to drive a Tesla Model 3?
```

What to expect:

- product-feedback synthesis
- usage workflows
- textbook pointers that connect the route back to the book

### Try A Deep-Research Route

```text
Analyze 10 recent studies on red wine and synthesize what they jointly support
```

What to expect:

- a slower route than textbook lookup or product feedback
- a provisional answer that improves as more evidence is processed
- telemetry such as ETA, parallelism, and stage labels

```text
How similar is Adobe to Walmart?
```

What to expect:

- the slowest major route in the current system
- two company builds plus a cross-company comparison
- inner Democritus-style stages such as causal statements, triple recovery, and
  yearly atlas construction

### How To Ask Democritus Better Questions

If you are using Democritus for the first time, the most common mistake is to
ask for a corpus that is too broad.

Democritus usually works best when the query clearly names:

- one intervention, exposure, or phenomenon
- one main outcome
- one population, system, or setting
- optionally one study type

This pattern is a good default:

```text
Analyze N recent [study type] on X effect on Y in Z, and synthesize their joint support.
```

Better Democritus-style prompts:

```text
Analyze 5 recent randomized trials of semaglutide for weight loss in adults with obesity and synthesize their joint support.
```

```text
Analyze 5 recent studies on how rising ocean temperatures affect wild fish population abundance and distribution, and synthesize their joint support.
```

Broader prompts often drift into neighboring topics. For example, a query about
ocean temperatures and fish populations can easily wander into coral systems,
aquaculture, pathogens, or fisheries economics unless the prompt narrows the
outcome or system more explicitly.

Signs the query is too broad:

- topic partitions are mostly singleton partitions
- many partitions are about adjacent but different topics
- the run keeps surfacing institution names, locations, or side domains instead
  of one shared evidence theme

If that happens, rewrite the query by tightening one of these axes:

- intervention or exposure
- outcome
- population or organism
- environment or setting
- study type

That usually improves Democritus more than simply increasing the document count.

## 5. What “Success” Looks Like

In your first session, success is not “install every backend.”

Success is:

- you can launch CLIFF
- at least one route answers correctly
- the result points back to the textbook
- you can see how the repo is trying to teach, not just compute
- you can tell the difference between lightweight routes and deep-research
  routes
- you can reopen a saved run or rerun it after restarting the session

If that is working, the overall design is already visible.

## 6. If You Want The Most Impressive Path

The strongest first impression usually comes from these three queries:

1. `Explain the Geometric Transformer on the Sudoku problem`
2. `What demo should I use for causality?`
3. `Show me the Julia version of KET`

Together, they show:

- textbook grounding
- recommendation mode
- Python and Julia explanation paths

If you want to understand the causal core instead of the fastest path, add:

4. `Analyze 10 recent studies of global warming and synthesize what they jointly support`

## 7. If Something Is Missing

Some routes depend on external repos.
That is normal in this version.

If a route says a dependency is unavailable, check the install matrix in
`README.md` and the resolver settings in `functorflow_v3/repo_layout.py`.

You do not need every external backend to understand CLIFF’s design.

## 8. What To Read Next

After the first 10 minutes, the best next documents are:

1. `README.md`
   for setup profiles and integration options
2. `docs/agentic_kan_architecture.md`
   for the conceptual shift from FunctorFlow v1 to the current CLIFF design
3. `functorflow_v3/query_router_agentic.py`
   for the top-level routing logic

## 9. What To Ignore At First

For a first pass, you can safely ignore:

- most of the tests
- the full Democritus internals
- the BASKET/ROCKET details
- implementation-specific path plumbing

But it is worth knowing that if you later want to optimize the system,
Democritus and the causal-state-building path are where many of the biggest
performance gains will come from.
