# First 10 Minutes With CLIFF_CatAgi

This guide is for a first-time user who wants to understand what CLIFF does
without reading the whole codebase.

## 1. What CLIFF_CatAgi Is

Think of CLIFF as a textbook interface.

You ask a question in natural language, and CLIFF tries to respond through one
or more of these layers:

- a pointer to the right section of `catagi.pdf`
- a runnable demo
- a synthesis dashboard
- a code snippet
- a project suggestion

The big idea is that the repo is not only code.
It is a teaching surface for *Categories for AGI*.

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

## 5. What “Success” Looks Like

In your first session, success is not “install every backend.”

Success is:

- you can launch CLIFF
- at least one route answers correctly
- the result points back to the textbook
- you can see how the repo is trying to teach, not just compute

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

Those matter later.
They are not the best entry point for understanding the product.
