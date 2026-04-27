"""Counterfactual assertions over Democritus causal claim artifacts."""

from __future__ import annotations

from collections import Counter
import json
import sqlite3
from pathlib import Path

from .causal_homotopy import normalize_claim_text, normalize_relation, relation_polarity


COUNTERFACTUAL_SCHEMA_VERSION = "cliff.democritus_counterfactuals.v1"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _label(value: object, *, fallback: str = "unknown") -> str:
    text = " ".join(str(value or "").split()).strip()
    return text or fallback


def _counterfactual_text(subj: str, rel: str, obj: str, polarity: str) -> str:
    normalized_rel = normalize_relation(rel)
    if polarity == "negative":
        return (
            f"If `{subj}` were increased or introduced, Democritus would expect `{obj}` to be lower or less likely, "
            f"because the source claim states that `{subj}` {normalized_rel} `{obj}`. Conversely, weakening "
            f"`{subj}` would be expected to remove that protective or suppressive pressure."
        )
    if normalized_rel in {"causes", "affects", "supports"}:
        return (
            f"If `{subj}` were absent or reduced, Democritus would expect the asserted effect on `{obj}` to attenuate; "
            f"if `{subj}` were strengthened, `{obj}` would be expected to become more likely under this local claim."
        )
    return (
        f"If `{subj}` were reduced or removed, Democritus would expect `{obj}` to decrease or become less likely; "
        f"if `{subj}` were strengthened, `{obj}` would be expected to increase under this local claim."
    )


def _intervention_label(subj: str, polarity: str) -> str:
    if polarity == "negative":
        return f"introduce or increase `{subj}`"
    return f"reduce or remove `{subj}`"


def _outcome_shift(obj: str, polarity: str) -> str:
    if polarity == "negative":
        return f"`{obj}` becomes higher or less suppressed when the cause is removed"
    return f"`{obj}` becomes lower or less likely when the cause is removed"


def _counterfactual_label_for_gluing_state(gluing_state: str) -> str:
    return {
        "obstructed": "obstructed",
        "regime_sensitive": "regime-sensitive",
        "multi_regime_glued": "stable",
        "single_regime": "local-only",
    }.get(gluing_state, "local-only")


def _load_csql_regime_surfaces(csql_sqlite_path: Path | None) -> dict[tuple[str, str], dict[str, object]]:
    if csql_sqlite_path is None or not csql_sqlite_path.exists():
        return {}

    def decode_json_array(value: object) -> list[str]:
        if value is None:
            return []
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            return []
        return [str(item) for item in payload] if isinstance(payload, list) else []

    connection = sqlite3.connect(str(csql_sqlite_path))
    try:
        rows = connection.execute(
            """
            SELECT
                canonical_subj,
                canonical_obj,
                regime_variant_count,
                regime_count,
                canonical_relation_count,
                polarity_count,
                total_document_support,
                max_regime_support,
                regimes_json,
                canonical_relations_json,
                gluing_state
            FROM regime_gluing_surfaces
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()
    return {
        (str(row[0]), str(row[1])): {
            "canonical_subj": str(row[0]),
            "canonical_obj": str(row[1]),
            "regime_variant_count": int(row[2] or 0),
            "regime_count": int(row[3] or 0),
            "canonical_relation_count": int(row[4] or 0),
            "polarity_count": int(row[5] or 0),
            "total_document_support": int(row[6] or 0),
            "max_regime_support": int(row[7] or 0),
            "regimes": decode_json_array(row[8]),
            "canonical_relations": decode_json_array(row[9]),
            "gluing_state": str(row[10] or "single_regime"),
        }
        for row in rows
    }


def _load_topos_contexts(topos_psr_path: Path | None) -> dict[str, object]:
    if topos_psr_path is None or not topos_psr_path.exists():
        return {}
    try:
        payload = json.loads(topos_psr_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    contexts = {
        str(row.get("context_id") or ""): dict(row)
        for row in list(dict(payload).get("contexts") or [])
        if isinstance(row, dict)
    }
    restriction_by_target: dict[str, list[dict[str, object]]] = {}
    for row in list(dict(payload).get("restriction_diagnostics") or []):
        if not isinstance(row, dict):
            continue
        restriction_by_target.setdefault(str(row.get("target_context") or ""), []).append(row)
    return {"contexts": contexts, "restriction_by_target": restriction_by_target}


def _topos_label_for_domain(domain: str, topos_contexts: dict[str, object]) -> str:
    contexts = dict(topos_contexts.get("contexts") or {})
    restrictions = dict(topos_contexts.get("restriction_by_target") or {})
    domain_id = normalize_claim_text(domain).replace(" ", "_")
    context_id = f"domain::{domain_id}"
    if context_id not in contexts:
        return "topos-unlocalized"
    target_restrictions = [dict(row) for row in list(restrictions.get(context_id) or [])]
    if any(not bool(row.get("compatible")) for row in target_restrictions):
        return "topos-tense"
    if target_restrictions:
        return "topos-stable"
    return "topos-local"


def enrich_democritus_counterfactual_payload(
    payload: dict[str, object],
    *,
    csql_sqlite_path: Path | None = None,
    topos_psr_path: Path | None = None,
) -> dict[str, object]:
    """Attach CSQL regime-gluing and Topos PSR labels to counterfactual rows."""

    regime_surfaces = _load_csql_regime_surfaces(csql_sqlite_path)
    topos_contexts = _load_topos_contexts(topos_psr_path)
    enriched_rows: list[dict[str, object]] = []
    for row in list(payload.get("counterfactuals") or []):
        item = dict(row)
        key = (str(item.get("canonical_subj") or ""), str(item.get("canonical_obj") or ""))
        surface = regime_surfaces.get(key)
        if surface is not None:
            gluing_state = str(surface.get("gluing_state") or "single_regime")
            item["gluing_label"] = _counterfactual_label_for_gluing_state(gluing_state)
            item["gluing_state"] = gluing_state
            item["regime_gluing_support"] = dict(surface)
        else:
            item["gluing_label"] = "local-only"
            item["gluing_state"] = "single_regime"
            item["regime_gluing_support"] = {}
        item["topos_label"] = _topos_label_for_domain(str(item.get("domain") or ""), topos_contexts)
        enriched_rows.append(item)

    label_counts = Counter(str(row.get("gluing_label") or "unknown") for row in enriched_rows)
    topos_counts = Counter(str(row.get("topos_label") or "unknown") for row in enriched_rows)
    summary = dict(payload.get("summary") or {})
    summary["gluing_label_counts"] = dict(label_counts)
    summary["topos_label_counts"] = dict(topos_counts)
    summary["csql_regime_surface_count"] = len(regime_surfaces)
    summary["label_semantics"] = (
        "Counterfactual labels are assigned from CSQL regime-gluing surfaces when the same canonical "
        "cause/outcome appears across regimes; Topos labels summarize whether the claim domain has a compatible local PSR context."
    )
    enriched = dict(payload)
    enriched["summary"] = summary
    enriched["counterfactuals"] = enriched_rows
    return enriched


def build_democritus_counterfactuals_from_triples(
    triples: list[dict[str, object]],
    *,
    domain_name: str = "",
    limit: int = 24,
) -> dict[str, object]:
    """Build local counterfactual assertions from extracted Democritus triples."""

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, triple in enumerate(triples, start=1):
        subj = _label(triple.get("subj") or triple.get("src") or triple.get("source"), fallback=f"cause {index}")
        rel = _label(triple.get("rel") or triple.get("relation"), fallback="affects")
        obj = _label(triple.get("obj") or triple.get("dst") or triple.get("target"), fallback=f"outcome {index}")
        canonical_subj = normalize_claim_text(subj)
        canonical_rel = normalize_relation(rel)
        canonical_obj = normalize_claim_text(obj)
        polarity = relation_polarity(rel)
        key = (canonical_subj, canonical_rel, canonical_obj, str(triple.get("domain") or ""))
        if key in seen:
            continue
        seen.add(key)
        statement = _label(triple.get("statement"), fallback=f"{subj} {rel} {obj}.")
        domain = _label(triple.get("domain") or triple.get("topic") or domain_name, fallback=domain_name or "document")
        rows.append(
            {
                "counterfactual_id": f"cf_{len(rows) + 1:04d}",
                "source": "democritus_relational_triple",
                "topic": _label(triple.get("topic"), fallback=domain),
                "path": list(triple.get("path") or []),
                "question": _label(triple.get("question"), fallback=""),
                "statement": statement,
                "subj": subj,
                "rel": rel,
                "obj": obj,
                "canonical_subj": canonical_subj,
                "canonical_rel": canonical_rel,
                "canonical_polarity": polarity,
                "canonical_obj": canonical_obj,
                "domain": domain,
                "intervention": _intervention_label(subj, polarity),
                "expected_shift": _outcome_shift(obj, polarity),
                "counterfactual": _counterfactual_text(subj, rel, obj, polarity),
                "support_tier": "document-supported",
                "evidence_status": "derived_from_causal_claim",
                "gluing_label": "local-only",
                "gluing_state": "single_regime",
                "topos_label": "topos-unlabeled",
                "semantics": "local_intervention_counterfactual_v1",
            }
        )
        if len(rows) >= limit:
            break

    polarity_counts = Counter(str(row["canonical_polarity"]) for row in rows)
    domain_counts = Counter(str(row["domain"]) for row in rows)
    return {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "summary": {
            "domain_name": domain_name,
            "triple_count": len(triples),
            "counterfactual_count": len(rows),
            "polarity_counts": dict(polarity_counts),
            "domain_counts": dict(domain_counts),
            "semantics": (
                "Local intervention-style counterfactual assertions derived from Democritus causal triples. "
                "These are logical consequences of extracted claims, not independently identified causal effects."
            ),
        },
        "counterfactuals": rows,
    }


def build_democritus_counterfactuals_from_jsonl(
    triples_path: Path,
    *,
    domain_name: str = "",
    limit: int = 24,
) -> dict[str, object]:
    """Read a Democritus triples JSONL file and build counterfactual assertions."""

    return build_democritus_counterfactuals_from_triples(
        _read_jsonl(triples_path),
        domain_name=domain_name,
        limit=limit,
    )


def write_democritus_counterfactual_artifacts(
    triples_path: Path,
    *,
    outdir: Path,
    domain_name: str = "",
    limit: int = 24,
    csql_sqlite_path: Path | None = None,
    topos_psr_path: Path | None = None,
) -> dict[str, Path]:
    """Materialize JSON and Markdown counterfactual artifacts for one Democritus run."""

    outdir.mkdir(parents=True, exist_ok=True)
    payload = build_democritus_counterfactuals_from_jsonl(
        triples_path,
        domain_name=domain_name,
        limit=limit,
    )
    payload = enrich_democritus_counterfactual_payload(
        payload,
        csql_sqlite_path=csql_sqlite_path,
        topos_psr_path=topos_psr_path,
    )
    json_path = outdir / "democritus_counterfactuals.json"
    markdown_path = outdir / "democritus_counterfactuals.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": markdown_path}


def _markdown_report(payload: dict[str, object]) -> str:
    summary = dict(payload.get("summary") or {})
    lines = [
        "# Democritus Counterfactual Assertions",
        "",
        f"- counterfactuals: {int(summary.get('counterfactual_count', 0) or 0)}",
        f"- source triples: {int(summary.get('triple_count', 0) or 0)}",
        "",
        "These assertions are derived from extracted document claims. They are not independently identified causal effects.",
        "",
    ]
    for row in list(payload.get("counterfactuals") or []):
        item = dict(row)
        lines.extend(
            [
                f"## {item.get('subj', 'cause')} -> {item.get('obj', 'outcome')}",
                "",
                f"- statement: {item.get('statement', '')}",
                f"- intervention: {item.get('intervention', '')}",
                f"- expected shift: {item.get('expected_shift', '')}",
                f"- support: {item.get('support_tier', '')}",
                f"- gluing label: {item.get('gluing_label', 'local-only')}",
                f"- topos label: {item.get('topos_label', 'topos-unlabeled')}",
                "",
                str(item.get("counterfactual") or ""),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
