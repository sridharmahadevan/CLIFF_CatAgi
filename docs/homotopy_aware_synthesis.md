# Homotopy-Aware Synthesis In CLIFF And Democritus

## Why This Matters

CLIFF and Democritus do not fail only because retrieval can miss documents.
They also fail because language is highly redundant.

The same underlying idea can appear as:

- a paraphrase
- a weaker or stronger formulation
- a more local or more global statement
- a causal restatement in a different vocabulary
- a change of framing rather than a change of meaning

For multi-document synthesis, this means a naive similarity pipeline is not
enough. The system needs to recover when two linguistic paths are deformable
into the same semantic object, and when they are not.

That is the motivating intuition for a homotopy-aware design.

## Explicit Design Goal

CLIFF and Democritus should aim to recover **equivalence classes of meaning**
under linguistic deformation, not only ranked lists of similar statements.

Operationally, the system should:

- collapse harmless wording drift into shared claim backbones
- preserve meaningful refinements rather than flattening everything into one summary
- separate same-direction reformulations from genuine disagreement
- express uncertainty when the system cannot yet tell whether two statements belong to the same class

The product goal is simple:

- less paraphrase spam
- less false consensus
- fewer long runs that end in gibberish
- more stable backbone claims and clearer disagreement surfaces

## Current Repo Signals

Some parts of this goal already exist in embryo.

- `democritus_corpus_synthesis.py` already distinguishes `equivalence_classes`
  from `disagreements`
- the rendered synthesis already speaks of `backbone claim` and
  `homotopically equivalent same-direction relations`
- the newer query-ingress work in `democritus_query_agentic.py` adds early
  clarification checkpoints to avoid obvious retrieval drift
- `course_demo_agentic.py` now handles some explanation-style queries before
  they fall through to deep evidence acquisition

So this note is not proposing a new philosophical layer unrelated to the code.
It is formalizing a direction that is already present in the current synthesis
and routing work.

## System Principle

At every stage, prefer:

- invariants over surface wording
- weak equivalence tests over plain similarity
- explicit relation typing over forced merging
- bounded clarification over expensive ungrounded runs

The important shift is that the system should not ask only:

- `How similar are these two claims?`

It should ask:

- `Are these claims the same backbone under allowable deformation?`
- `Is one a refinement of the other?`
- `Do they differ only by context, granularity, or emphasis?`
- `Are they truly in polarity conflict?`
- `Is the relation between them still unresolved?`

## Proposed Semantic Objects

To make this concrete, Democritus should move toward a typed claim graph with
explicit relation semantics.

### 1. Claim Surface

The raw extracted statement as it appears in a document.

Useful fields:

- original sentence or clause
- document id / run id
- local topic / domain
- extraction confidence

### 2. Claim Backbone

A normalized semantic representative for a family of near-equivalent claim
surfaces.

Useful fields:

- canonical subject
- canonical relation
- canonical object
- polarity
- scope or domain tags
- abstraction level

### 3. Relation Edge

A typed edge between claim surfaces or backbones.

Initial relation types should include:

- `weak_equivalence`
- `refinement`
- `same_direction_variant`
- `context_shift`
- `polarity_conflict`
- `unresolved`

### 4. Equivalence Class

A connected component of claims that are safe to treat as one backbone for
display and synthesis.

### 5. Disagreement Surface

A component where the system has evidence for opposed polarity or genuinely
incompatible causal direction.

## Weak Equivalence

The central technical object is not strict equality but **weak equivalence**.

A weak-equivalence test should be conservative.
It should succeed when two claims preserve the same semantic invariant under
rewording, but fail when the change may matter causally.

Examples of invariants:

- same subject/object backbone after normalization
- same causal direction
- same sign or polarity
- same domain scope
- compatible quantification or population scope

Examples of non-invariants:

- changing benefit to harm
- changing correlation to intervention
- changing population or regime
- changing local finding to universal claim

This is where a homotopy viewpoint helps:
many paths through wording space should count as the same object, but only if
they preserve the semantic structure we care about.

## Where This Fits In The Current Pipeline

The implementation path should attach to stages that already exist.

### Query ingress

Files:

- `functorflow_v3/query_router_agentic.py`
- `functorflow_v3/democritus_query_agentic.py`
- `functorflow_v3/course_demo_agentic.py`

Near-term responsibility:

- identify explanation-style prompts that should route to course guidance or a
  clarification checkpoint
- identify ambiguous retrieval terms before a long run starts
- avoid starting Democritus when the user has not actually specified an
  evidence-acquisition task

### Per-document extraction

Files:

- `functorflow_v3/democritus_agentic.py`
- `functorflow_v3/democritus_batch_agentic.py`

Near-term responsibility:

- preserve richer claim metadata needed for equivalence testing
- include scope, polarity, domain, and possibly quantifier cues in the
  extracted structured records
- keep enough provenance to trace every backbone claim back to its surfaces

### Cross-document synthesis

Files:

- `functorflow_v3/democritus_corpus_synthesis.py`
- `functorflow_v3/csql_bundle.py`

Near-term responsibility:

- replace ad hoc coalescing with explicit typed relation construction
- build equivalence classes from relation edges rather than only lexical
  normalization
- keep `same-direction variant` separate from `refinement`
- keep unresolved edges visible rather than silently merging them

### Conscious layer / product surface

Files:

- `functorflow_v3/cliff.py`
- `functorflow_v3/dashboard_query_launcher.py`

Near-term responsibility:

- surface clarification checkpoints as first-class outcomes
- show backbone claims, refinement chains, and disagreement surfaces explicitly
- make “not enough structure yet to merge these claims” a valid user-facing
  state

## Implementation Plan

This should be done incrementally.

### Phase 1. Tighten ingress and avoid bad long runs

Goal:

- reduce gibberish by refusing underspecified or wrongly-routed deep runs

Tasks:

- expand query clarification in `democritus_query_agentic.py`
- keep explanation-style course queries out of Democritus
- add more ambiguity patterns beyond the initial `inflation` case
- add a lightweight “evidence-acquisition readiness” score before launching a
  long Democritus run

Success signal:

- fewer deep runs started on queries that should have been clarified or routed
  elsewhere

### Phase 2. Promote typed claim relations

Goal:

- stop treating all near matches as one kind of merge problem

Tasks:

- extend claim records with polarity, scope, and abstraction cues
- add typed relation scoring between claims:
  `weak_equivalence`, `refinement`, `polarity_conflict`, `unresolved`
- persist those relation types in the CSQL / synthesis bundle layer

Success signal:

- the system can say why two claims were merged, separated, or left unresolved

### Phase 3. Build backbone-first synthesis

Goal:

- make the synthesis output centered on backbone claims, not on raw extracted
  sentences

Tasks:

- define a `ClaimBackbone` object in synthesis
- aggregate support at the backbone level
- attach surface variants and refinements beneath each backbone
- only place claims in disagreements when polarity conflict is explicit

Success signal:

- synthesis pages show fewer duplicate claims and clearer organization

### Phase 4. Add uncertainty-preserving merge control

Goal:

- allow the system to stop before over-merging

Tasks:

- add an `unresolved` relation bucket
- surface uncertain merges in dashboards
- use thresholds that favor under-merging over hallucinated consensus

Success signal:

- fewer false equivalences and clearer explanation of uncertainty

### Phase 5. Homotopy-aware evaluation

Goal:

- test the actual design goal, not only extraction coverage

Tasks:

- build regression sets with:
  paraphrases, refinements, context shifts, and true contradictions
- add tests that measure whether claims land in the right relation buckets
- evaluate false merge rate and false split rate

Success signal:

- the repo can track progress on equivalence recovery directly

## Suggested Internal Abstractions

These names are intentionally close to the current synthesis vocabulary.

- `ClaimSurface`
- `ClaimBackbone`
- `ClaimRelationEdge`
- `WeakEquivalence`
- `RefinementEdge`
- `DisagreementEdge`
- `UnresolvedEdge`
- `EquivalenceClass`

The point is not naming for its own sake.
It is to make the homotopy-aware design explicit in code, tests, and product
surfaces.

## Product Consequence

If this design goal is taken seriously, CLIFF should increasingly behave like a
system that asks:

- `What is the invariant semantic structure here?`

rather than a system that asks only:

- `What text looks similar enough to merge?`

That is the practical interpretation of categorical homotopy for this repo.

## Recommended Next Steps

The next concrete implementation work should be:

1. keep extending clarification-first ingress in `democritus_query_agentic.py`
2. define a first typed relation schema in `democritus_corpus_synthesis.py`
3. add regression tests that distinguish paraphrase, refinement, and conflict
4. only then refactor the synthesis dashboards around backbone claims

This order should reduce wasted long runs first, then improve the semantic
quality of the runs that remain.
