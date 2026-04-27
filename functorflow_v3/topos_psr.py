"""Topos-style predictive state representations for product-review episodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ReviewContextSpec:
    """A local context in which histories and tests are interpreted."""

    context_id: str
    label: str
    action_set: frozenset[str]
    predicate_roots: frozenset[str]
    parents: tuple[str, ...] = ()
    description: str = ""


_ALL_ACTIONS = frozenset(
    {
        "research",
        "order",
        "deliver",
        "unbox",
        "open",
        "assemble",
        "configure",
        "wear",
        "run",
        "drive",
        "charge",
        "sit",
        "taste",
        "eat",
        "share",
        "store",
        "clean",
        "wash",
        "reconfigure",
        "return",
        "recommend",
    }
)

_ROOT_CONTEXT_SPECS: tuple[ReviewContextSpec, ...] = (
    ReviewContextSpec(
        context_id="review",
        label="Review",
        action_set=_ALL_ACTIONS,
        predicate_roots=frozenset({"*"}),
        parents=(),
        description="Global review context that sees the full episode.",
    ),
    ReviewContextSpec(
        context_id="post_purchase",
        label="Post Purchase",
        action_set=frozenset(
            {
                "order",
                "deliver",
                "unbox",
                "open",
                "assemble",
                "configure",
                "wear",
                "run",
                "drive",
                "charge",
                "sit",
                "taste",
                "eat",
                "share",
                "store",
                "clean",
                "wash",
                "reconfigure",
                "return",
                "recommend",
            }
        ),
        predicate_roots=frozenset(
            {
                "assembly",
                "instructions",
                "comfort",
                "fit",
                "value",
                "recommend",
                "return_risk",
                "returned",
                "seat_depth",
                "cushion_stability",
                "ease_of_use",
                "taste",
                "flavor",
                "texture",
                "sweetness",
                "aroma",
            }
        ),
        parents=("review",),
        description="The customer journey after ordering or receiving the product.",
    ),
    ReviewContextSpec(
        context_id="use",
        label="Use",
        action_set=frozenset({"configure", "wear", "run", "drive", "sit", "open", "taste", "eat", "share", "store", "clean", "wash", "reconfigure"}),
        predicate_roots=frozenset(
            {
                "comfort",
                "cushioning",
                "support",
                "durability",
                "traction",
                "taste",
                "flavor",
                "texture",
                "sweetness",
                "aroma",
                "seat_depth",
                "cushion_stability",
                "ease_of_use",
                "recommend",
                "return_risk",
                "returned",
            }
        ),
        parents=("review",),
        description="Local use-time context shared by product interaction, consumption, cleaning, and reconfiguration.",
    ),
    ReviewContextSpec(
        context_id="taste",
        label="Taste",
        action_set=frozenset({"open", "taste", "eat", "share", "store", "return", "recommend"}),
        predicate_roots=frozenset({"taste", "flavor", "texture", "sweetness", "aroma", "value", "recommend", "return_risk", "returned"}),
        parents=("use",),
        description="Consumption and flavor context for food and beverage products.",
    ),
    ReviewContextSpec(
        context_id="fit",
        label="Fit",
        action_set=frozenset({"configure", "wear", "run", "return", "recommend"}),
        predicate_roots=frozenset({"fit", "heel_slip", "stability", "ease_of_use", "return_risk", "returned", "recommend"}),
        parents=("use",),
        description="Sizing, stability, and fit-driven decision context.",
    ),
    ReviewContextSpec(
        context_id="run",
        label="Run",
        action_set=frozenset({"wear", "run", "clean", "return", "recommend"}),
        predicate_roots=frozenset(
            {"comfort", "cushioning", "support", "traction", "stability", "heel_slip", "fit", "recommend", "return_risk", "returned"}
        ),
        parents=("use",),
        description="Running and daily-trainer context for shoes and related products.",
    ),
    ReviewContextSpec(
        context_id="assemble",
        label="Assemble",
        action_set=frozenset({"deliver", "unbox", "assemble", "configure", "sit", "recommend", "return"}),
        predicate_roots=frozenset({"assembly", "instructions", "comfort", "seat_depth", "cushion_stability", "recommend", "return_risk", "returned"}),
        parents=("post_purchase",),
        description="Assembly and setup context for furniture and similar products.",
    ),
    ReviewContextSpec(
        context_id="sit",
        label="Sit",
        action_set=frozenset({"deliver", "assemble", "configure", "sit", "clean", "wash", "reconfigure", "recommend", "return"}),
        predicate_roots=frozenset({"comfort", "seat_depth", "cushion_stability", "durability", "value", "recommend", "return_risk", "returned"}),
        parents=("use",),
        description="Post-setup sitting and lounging context for furniture.",
    ),
    ReviewContextSpec(
        context_id="durability",
        label="Durability",
        action_set=frozenset({"wear", "run", "sit", "clean", "wash", "reconfigure", "return", "recommend"}),
        predicate_roots=frozenset({"durability", "quality", "washable", "cleanability", "recommend", "return_risk", "returned"}),
        parents=("use",),
        description="Long-run wear, maintenance, and product longevity context.",
    ),
    ReviewContextSpec(
        context_id="decision",
        label="Decision",
        action_set=frozenset({"return", "recommend"}),
        predicate_roots=frozenset({"recommend", "return_risk", "returned", "value", "fit", "comfort", "durability"}),
        parents=("review",),
        description="Recommendation and return decision context.",
    ),
)


def _clean_actions(actions: Iterable[object]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for action in actions:
        value = str(action).strip().lower()
        if not value:
            continue
        if cleaned and cleaned[-1] == value:
            continue
        cleaned.append(value)
    return tuple(cleaned)


def _predicate_root(predicate: str) -> str:
    value = predicate.strip().lower()
    if not value:
        return value
    special_roots = {
        "fit_tight": "fit",
        "fit_loose": "fit",
        "fit_true_to_size": "fit",
        "heel_slip": "heel_slip",
        "stability_negative": "stability",
        "assembly_difficult": "assembly",
        "assembly_easy": "assembly",
        "instructions_confusing": "instructions",
        "recommend_positive": "recommend",
        "return_risk_positive": "return_risk",
        "returned": "returned",
        "cushioning_positive": "cushioning",
        "cushioning_negative": "cushioning",
        "support_positive": "support",
        "support_negative": "support",
        "quality_positive": "quality",
        "quality_negative": "quality",
        "washable_positive": "washable",
        "cleanability_positive": "cleanability",
        "taste_positive": "taste",
        "taste_negative": "taste",
    }
    if value in special_roots:
        return special_roots[value]
    if "_" not in value:
        return value
    prefix, suffix = value.rsplit("_", 1)
    if suffix in {"positive", "negative", "mixed"}:
        return prefix
    return value


def _canonical_predicates(event: dict[str, object]) -> tuple[str, ...]:
    lowered = str(event.get("text") or "").lower()
    predicates: set[str] = set()
    for aspect, polarity_value in dict(event.get("aspect_polarities") or {}).items():
        aspect_name = str(aspect).strip().lower()
        polarity = str(polarity_value).strip().lower()
        if not aspect_name or not polarity:
            continue
        predicates.add(f"{aspect_name}_{polarity}")
        if aspect_name == "fit":
            if any(token in lowered for token in ("tight", "too tight", "narrow", "too small", "runs small")):
                predicates.add("fit_tight")
            if any(token in lowered for token in ("loose", "too loose", "too big", "runs large")):
                predicates.add("fit_loose")
            if any(token in lowered for token in ("heel slip", "slips off", "slipped on turns")):
                predicates.add("heel_slip")
                predicates.add("stability_negative")
            if polarity == "positive" and any(token in lowered for token in ("true to size", "fits well", "good fit", "perfect fit")):
                predicates.add("fit_true_to_size")
        elif aspect_name == "comfort":
            if polarity == "positive":
                predicates.add("comfort_positive")
            elif polarity == "negative":
                predicates.add("comfort_negative")
            if "cushion" in lowered:
                predicates.add(f"cushioning_{polarity}")
            if "support" in lowered:
                predicates.add(f"support_{polarity}")
        elif aspect_name == "durability":
            predicates.add(f"durability_{polarity}")
            if any(token in lowered for token in ("quality", "well made", "poor quality", "cheaply made", "cheap")):
                predicates.add(f"quality_{polarity}")
        elif aspect_name == "traction":
            predicates.add(f"traction_{polarity}")
            if polarity == "negative":
                predicates.add("stability_negative")
        elif aspect_name == "assembly":
            if polarity == "negative" or any(token in lowered for token in ("difficult assembly", "hard to assemble", "assembly took", "tedious assembly")):
                predicates.add("assembly_difficult")
            if polarity == "positive":
                predicates.add("assembly_easy")
            if any(token in lowered for token in ("instructions confusing", "confusing instructions", "manual unclear", "instructions were confusing")):
                predicates.add("instructions_confusing")
        elif aspect_name == "ease_of_use" and polarity == "positive":
            predicates.add("ease_of_use_positive")
        elif aspect_name == "taste":
            predicates.add(f"taste_{polarity}")
        elif aspect_name == "seat_depth":
            predicates.add(f"seat_depth_{polarity}")
        elif aspect_name == "cushion_stability":
            predicates.add(f"cushion_stability_{polarity}")
        elif aspect_name == "value":
            predicates.add(f"value_{polarity}")
        elif aspect_name == "style":
            predicates.add(f"style_{polarity}")
    if bool(event.get("recommendation_signal")):
        predicates.add("recommend_positive")
    if bool(event.get("return_risk_signal")):
        predicates.add("return_risk_positive")
    if bool(event.get("returned")):
        predicates.add("returned")
        predicates.add("return_risk_positive")
    return tuple(sorted(predicates))


def _contains_subsequence(sequence: tuple[str, ...], motif: tuple[str, ...], *, start_index: int = 0) -> bool:
    if not motif:
        return True
    if not sequence or len(motif) > len(sequence):
        return False
    max_start = len(sequence) - len(motif)
    for index in range(max(0, start_index), max_start + 1):
        if sequence[index : index + len(motif)] == motif:
            return True
    return False


def _iter_prefixes(sequence: tuple[str, ...], *, max_history_length: int) -> Iterable[tuple[str, ...]]:
    yield ()
    limit = max(0, min(len(sequence), max_history_length))
    for size in range(1, limit + 1):
        yield sequence[:size]


def _iter_action_tests(sequence: tuple[str, ...], *, max_test_length: int) -> Iterable[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    limit = max(1, min(len(sequence), max_test_length))
    for start in range(len(sequence)):
        for size in range(1, limit + 1):
            stop = start + size
            if stop > len(sequence):
                break
            motif = sequence[start:stop]
            if motif in seen:
                continue
            seen.add(motif)
            yield motif


def build_review_episodes(
    normalized_events: Iterable[dict[str, object]],
    usage_workflows: dict[str, object],
    *,
    product_name: str,
    brand_name: str = "",
) -> list[dict[str, object]]:
    """Collapse normalized feedback plus workflow extraction into review episodes."""

    workflow_rows = {
        str(row.get("feedback_id") or ""): dict(row)
        for row in list(usage_workflows.get("workflows") or [])
        if str(row.get("feedback_id") or "").strip()
    }
    domain = str(usage_workflows.get("usage_family") or "generic").strip().lower() or "generic"
    episodes: list[dict[str, object]] = []
    for index, event in enumerate(normalized_events, start=1):
        feedback_id = str(event.get("feedback_id") or f"feedback_{index:04d}")
        workflow_row = workflow_rows.get(feedback_id, {})
        selected_workflow = dict(workflow_row.get("selected_workflow") or {})
        actions = _clean_actions(selected_workflow.get("workflow_stages") or workflow_row.get("base_workflow_stages") or ())
        predicates = _canonical_predicates(event)
        contexts = {"review"}
        if any(action in {"order", "deliver", "unbox", "open", "assemble", "configure"} for action in actions):
            contexts.add("post_purchase")
        if any(action in {"configure", "wear", "run", "drive", "sit", "open", "taste", "eat", "share", "store", "clean", "wash", "reconfigure"} for action in actions):
            contexts.add("use")
        if "run" in actions:
            contexts.add("run")
        if "assemble" in actions:
            contexts.add("assemble")
        if "sit" in actions:
            contexts.add("sit")
        if any(action in {"taste", "eat"} for action in actions):
            contexts.add("taste")
        if "return" in actions or "recommend" in actions or any(
            predicate in {"recommend_positive", "return_risk_positive", "returned"} for predicate in predicates
        ):
            contexts.add("decision")
        predicate_roots = {_predicate_root(predicate) for predicate in predicates}
        if predicate_roots & {"fit", "heel_slip", "stability", "ease_of_use"}:
            contexts.add("fit")
        if predicate_roots & {"taste", "flavor", "texture", "sweetness", "aroma"}:
            contexts.add("taste")
        if predicate_roots & {"durability", "quality", "washable", "cleanability"}:
            contexts.add("durability")
        episodes.append(
            {
                "episode_id": feedback_id,
                "product": product_name,
                "brand": brand_name,
                "domain": domain,
                "title": str(event.get("title") or ""),
                "text": str(event.get("text") or ""),
                "contexts": sorted(contexts),
                "events": list(actions),
                "attributes": list(predicates),
                "outcome": "recommend" if "recommend_positive" in predicates else ("return_risk" if "return_risk_positive" in predicates else "mixed"),
                "sentiment": str(event.get("sentiment") or ""),
            }
        )
    return episodes


def _episode_view_for_context(
    episode: dict[str, object],
    context: ReviewContextSpec,
) -> dict[str, object] | None:
    actions = tuple(str(action) for action in episode.get("events") or ())
    predicates = tuple(str(predicate) for predicate in episode.get("attributes") or ())
    projected_actions = tuple(action for action in actions if action in context.action_set)
    visible_predicates = tuple(
        predicate
        for predicate in predicates
        if "*" in context.predicate_roots or _predicate_root(predicate) in context.predicate_roots
    )
    if not projected_actions and not visible_predicates:
        return None
    return {
        "episode_id": str(episode.get("episode_id") or ""),
        "projected_actions": projected_actions,
        "visible_predicates": visible_predicates,
    }


def _contexts_for_bundle(episodes: Iterable[dict[str, object]]) -> tuple[ReviewContextSpec, ...]:
    observed_contexts = {
        str(context_id)
        for episode in episodes
        for context_id in list(episode.get("contexts") or [])
        if str(context_id).strip()
    }
    specs = [spec for spec in _ROOT_CONTEXT_SPECS if spec.context_id in observed_contexts or spec.context_id == "review"]
    return tuple(specs)


def _history_rows_for_context(
    views: list[dict[str, object]],
    *,
    context_id: str,
    max_history_length: int,
    min_history_support: int,
    max_histories_per_context: int,
) -> list[dict[str, object]]:
    support: Counter[tuple[str, ...]] = Counter()
    for view in views:
        prefixes = set(_iter_prefixes(tuple(view["projected_actions"]), max_history_length=max_history_length))
        support.update(prefixes)
    ranked = sorted(
        [
            (history, count)
            for history, count in support.items()
            if history == () or count >= min_history_support
        ],
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    if ranked and ranked[0][0] != ():
        ranked = [(((), support.get((), len(views))))] + ranked
    if not ranked:
        ranked = [(((), len(views)))]
    rows = []
    for history, count in ranked[:max_histories_per_context]:
        signature = "__".join(history) if history else "epsilon"
        rows.append(
            {
                "history_id": f"{context_id}::history::{signature}",
                "signature": signature,
                "actions": list(history),
                "support": int(count),
            }
        )
    return rows


def _test_rows_for_context(
    views: list[dict[str, object]],
    *,
    context_id: str,
    max_test_length: int,
    min_test_support: int,
    max_tests_per_context: int,
) -> list[dict[str, object]]:
    attribute_support: Counter[str] = Counter()
    workflow_support: Counter[tuple[str, ...]] = Counter()
    workflow_attribute_support: Counter[tuple[tuple[str, ...], str]] = Counter()
    for view in views:
        predicates = set(str(predicate) for predicate in view["visible_predicates"])
        motifs = set(_iter_action_tests(tuple(view["projected_actions"]), max_test_length=max_test_length))
        attribute_support.update(predicates)
        workflow_support.update(motifs)
        for motif in motifs:
            for predicate in predicates:
                workflow_attribute_support[(motif, predicate)] += 1

    attr_limit = max(4, max_tests_per_context // 4)
    joint_limit = max(6, max_tests_per_context // 2)
    workflow_limit = max(4, max_tests_per_context - attr_limit - joint_limit)

    rows: list[dict[str, object]] = []
    for predicate, count in sorted(
        [(predicate, count) for predicate, count in attribute_support.items() if count >= min_test_support],
        key=lambda item: (-item[1], item[0]),
    )[:attr_limit]:
        rows.append(
            {
                "test_id": f"{context_id}::attribute::{predicate}",
                "signature": predicate,
                "test_kind": "attribute",
                "actions": [],
                "predicate": predicate,
                "support": int(count),
            }
        )
    for (motif, predicate), count in sorted(
        [((motif, predicate), count) for (motif, predicate), count in workflow_attribute_support.items() if count >= min_test_support],
        key=lambda item: (-item[1], len(item[0][0]), item[0][0], item[0][1]),
    )[:joint_limit]:
        motif_signature = "__".join(motif)
        rows.append(
            {
                "test_id": f"{context_id}::workflow_attribute::{motif_signature}::{predicate}",
                "signature": f"{motif_signature}=>{predicate}",
                "test_kind": "workflow_attribute",
                "actions": list(motif),
                "predicate": predicate,
                "support": int(count),
            }
        )
    for motif, count in sorted(
        [(motif, count) for motif, count in workflow_support.items() if count >= min_test_support],
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )[:workflow_limit]:
        motif_signature = "__".join(motif)
        rows.append(
            {
                "test_id": f"{context_id}::workflow::{motif_signature}",
                "signature": motif_signature,
                "test_kind": "workflow",
                "actions": list(motif),
                "predicate": "",
                "support": int(count),
            }
        )
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        signature = str(row["test_id"])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped[:max_tests_per_context]


def _test_occurs(
    *,
    history: tuple[str, ...],
    test_row: dict[str, object],
    projected_actions: tuple[str, ...],
    visible_predicates: set[str],
) -> bool:
    if history and projected_actions[: len(history)] != history:
        return False
    start_index = len(history)
    kind = str(test_row["test_kind"])
    if kind == "attribute":
        return str(test_row["predicate"]) in visible_predicates
    motif = tuple(str(action) for action in test_row.get("actions") or ())
    if kind == "workflow":
        return _contains_subsequence(projected_actions, motif, start_index=start_index)
    if kind == "workflow_attribute":
        return _contains_subsequence(projected_actions, motif, start_index=start_index) and str(test_row["predicate"]) in visible_predicates
    return False


def _matrix_rank_fallback(matrix: list[list[float]], *, tolerance: float = 1e-9) -> int:
    rows = [list(map(float, row)) for row in matrix if row]
    if not rows:
        return 0
    n_rows = len(rows)
    n_cols = len(rows[0])
    rank = 0
    pivot_row = 0
    for col in range(n_cols):
        pivot = None
        for row_index in range(pivot_row, n_rows):
            if abs(rows[row_index][col]) > tolerance:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][col]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for row_index in range(n_rows):
            if row_index == pivot_row:
                continue
            factor = rows[row_index][col]
            if abs(factor) <= tolerance:
                continue
            rows[row_index] = [
                current - factor * pivot_current
                for current, pivot_current in zip(rows[row_index], rows[pivot_row])
            ]
        rank += 1
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    return rank


def _svd_summary(matrix: list[list[float]]) -> dict[str, object]:
    if not matrix or not matrix[0]:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    if np is None:
        return {
            "rank": _matrix_rank_fallback(matrix),
            "singular_values": [],
            "energy_captured_top3": 0.0,
        }
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    _, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    if singular_values.size == 0:
        return {"rank": 0, "singular_values": [], "energy_captured_top3": 0.0}
    tolerance = max(1e-9, float(singular_values[0]) * 1e-6)
    rank = int(sum(float(value) > tolerance for value in singular_values))
    squared = singular_values ** 2
    total_energy = float(np.sum(squared))
    top_energy = float(np.sum(squared[:3])) if total_energy > 0.0 else 0.0
    return {
        "rank": rank,
        "singular_values": [round(float(value), 6) for value in singular_values[:10]],
        "energy_captured_top3": round(top_energy / total_energy, 6) if total_energy > 0.0 else 0.0,
    }


def build_topos_psr_bundle(
    normalized_events: Iterable[dict[str, object]],
    usage_workflows: dict[str, object],
    *,
    product_name: str,
    brand_name: str = "",
    max_history_length: int = 3,
    max_test_length: int = 3,
    min_history_support: int = 1,
    min_test_support: int = 1,
    max_histories_per_context: int = 12,
    max_tests_per_context: int = 18,
    glue_tolerance: float = 0.20,
) -> dict[str, object]:
    """Build a presheaf-style family of local Hankel matrices from review episodes."""

    episodes = build_review_episodes(
        normalized_events,
        usage_workflows,
        product_name=product_name,
        brand_name=brand_name,
    )
    context_specs = _contexts_for_bundle(episodes)
    context_views: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        for context in context_specs:
            view = _episode_view_for_context(episode, context)
            if view is None:
                continue
            context_views[context.context_id].append(view)

    local_hankel_family: list[dict[str, object]] = []
    matrix_lookup: dict[str, dict[str, object]] = {}
    for context in context_specs:
        views = context_views.get(context.context_id, [])
        if not views:
            continue
        histories = _history_rows_for_context(
            views,
            context_id=context.context_id,
            max_history_length=max_history_length,
            min_history_support=min_history_support,
            max_histories_per_context=max_histories_per_context,
        )
        tests = _test_rows_for_context(
            views,
            context_id=context.context_id,
            max_test_length=max_test_length,
            min_test_support=min_test_support,
            max_tests_per_context=max_tests_per_context,
        )
        history_tuples = [tuple(str(action) for action in row["actions"]) for row in histories]
        matrix: list[list[float]] = []
        entries: list[dict[str, object]] = []
        for history_row, history_tuple in zip(histories, history_tuples):
            matching_views = [
                view
                for view in views
                if not history_tuple or tuple(view["projected_actions"])[: len(history_tuple)] == history_tuple
            ]
            denominator = len(matching_views)
            row_values: list[float] = []
            for test_row in tests:
                if denominator <= 0:
                    probability = 0.0
                    numerator = 0
                else:
                    numerator = sum(
                        1
                        for view in matching_views
                        if _test_occurs(
                            history=history_tuple,
                            test_row=test_row,
                            projected_actions=tuple(view["projected_actions"]),
                            visible_predicates=set(str(predicate) for predicate in view["visible_predicates"]),
                        )
                    )
                    probability = float(numerator) / float(denominator)
                row_values.append(round(probability, 6))
                entries.append(
                    {
                        "history_id": history_row["history_id"],
                        "test_id": test_row["test_id"],
                        "history_signature": history_row["signature"],
                        "test_signature": test_row["signature"],
                        "probability": round(probability, 6),
                        "matches": int(numerator),
                        "history_support": int(denominator),
                    }
                )
            matrix.append(row_values)
        svd = _svd_summary(matrix)
        payload = {
            "context_id": context.context_id,
            "context_label": context.label,
            "description": context.description,
            "parents": list(context.parents),
            "n_episode_views": len(views),
            "histories": histories,
            "tests": tests,
            "matrix": matrix,
            "entries": entries,
            "svd": svd,
        }
        local_hankel_family.append(payload)
        matrix_lookup[context.context_id] = payload

    restriction_diagnostics: list[dict[str, object]] = []
    for context in context_specs:
        child = matrix_lookup.get(context.context_id)
        if child is None:
            continue
        child_history = {str(row["signature"]): row for row in child["histories"]}
        child_tests = {str(row["signature"]): row for row in child["tests"]}
        child_entry = {
            (str(row["history_signature"]), str(row["test_signature"])): float(row["probability"])
            for row in child["entries"]
        }
        for parent_id in context.parents:
            parent = matrix_lookup.get(parent_id)
            if parent is None:
                continue
            parent_history = {str(row["signature"]): row for row in parent["histories"]}
            parent_tests = {str(row["signature"]): row for row in parent["tests"]}
            overlap_histories = sorted(set(parent_history) & set(child_history))
            overlap_tests = sorted(set(parent_tests) & set(child_tests))
            diffs = [
                abs(
                    child_entry.get((history_signature, test_signature), 0.0)
                    - float(
                        next(
                            (
                                row["probability"]
                                for row in parent["entries"]
                                if str(row["history_signature"]) == history_signature
                                and str(row["test_signature"]) == test_signature
                            ),
                            0.0,
                        )
                    )
                )
                for history_signature in overlap_histories
                for test_signature in overlap_tests
            ]
            mean_diff = float(sum(diffs) / len(diffs)) if diffs else 0.0
            max_diff = max(diffs) if diffs else 0.0
            restriction_diagnostics.append(
                {
                    "source_context": parent_id,
                    "target_context": context.context_id,
                    "shared_histories": overlap_histories,
                    "shared_tests": overlap_tests,
                    "mean_abs_gap": round(mean_diff, 6),
                    "max_abs_gap": round(max_diff, 6),
                    "compatible": bool(max_diff <= glue_tolerance),
                    "restriction_rule": "restrict histories and tests to signatures supported in the target context",
                }
            )

    ranks = [int(row["svd"]["rank"]) for row in local_hankel_family]
    summary = {
        "product_name": product_name,
        "brand_name": brand_name,
        "domain": str(usage_workflows.get("usage_family") or "generic"),
        "n_episodes": len(episodes),
        "n_contexts": len(local_hankel_family),
        "context_ids": [row["context_id"] for row in local_hankel_family],
        "n_restriction_checks": len(restriction_diagnostics),
        "n_compatible_restrictions": sum(1 for row in restriction_diagnostics if bool(row["compatible"])),
        "mean_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
        "max_rank": max(ranks) if ranks else 0,
        "config": {
            "max_history_length": max_history_length,
            "max_test_length": max_test_length,
            "min_history_support": min_history_support,
            "min_test_support": min_test_support,
            "max_histories_per_context": max_histories_per_context,
            "max_tests_per_context": max_tests_per_context,
            "glue_tolerance": glue_tolerance,
        },
    }
    return {
        "episodes": episodes,
        "contexts": [
            {
                "context_id": context.context_id,
                "label": context.label,
                "parents": list(context.parents),
                "description": context.description,
            }
            for context in context_specs
            if context.context_id in summary["context_ids"]
        ],
        "local_hankel_family": local_hankel_family,
        "restriction_diagnostics": restriction_diagnostics,
        "summary": summary,
    }
