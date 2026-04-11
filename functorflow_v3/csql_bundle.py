"""Batch-level CSQL-style bundle generation for Democritus outputs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .causal_homotopy import normalize_claim_text, normalize_relation, relation_polarity


@dataclass(frozen=True)
class BatchCSQLBundleResult:
    """Materialized CSQL-style bundle paths."""

    sqlite_path: Path
    summary_path: Path


def _iter_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(dict(json.loads(stripped)))
            except json.JSONDecodeError:
                # Partial incremental writes can leave a trailing truncated line
                # while triple extraction is still in flight. Skip that fragment and
                # pick it up on the next synthesis refresh.
                continue
    return rows


def build_batch_csql_bundle(
    *,
    batch_outdir: Path,
    records: list[dict[str, object]],
    pdf_dir: Path,
) -> BatchCSQLBundleResult:
    """Materialize a simple cross-document CSQL-style SQLite bundle."""

    csql_dir = batch_outdir / "csql"
    csql_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = csql_dir / "democritus_csql.sqlite"
    summary_path = csql_dir / "democritus_csql_summary.json"
    if sqlite_path.exists():
        sqlite_path.unlink()

    connection = sqlite3.connect(str(sqlite_path))
    try:
        _create_schema(connection)
        _populate_bundle(connection, records=records, pdf_dir=pdf_dir)
        connection.commit()
        summary = _compute_summary(connection, pdf_dir=pdf_dir)
    finally:
        connection.close()

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return BatchCSQLBundleResult(sqlite_path=sqlite_path, summary_path=summary_path)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            run_name TEXT NOT NULL UNIQUE,
            pdf_path TEXT NOT NULL,
            domain_name TEXT,
            triples_path TEXT NOT NULL
        );

        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE relations (
            relation_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE domains (
            domain_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE claims (
            claim_id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            path_json TEXT NOT NULL,
            question TEXT NOT NULL,
            statement TEXT NOT NULL,
            surface_form TEXT NOT NULL,
            canonical_subj TEXT NOT NULL,
            canonical_rel TEXT NOT NULL,
            canonical_polarity TEXT NOT NULL,
            canonical_obj TEXT NOT NULL,
            canonical_domain TEXT NOT NULL,
            subj_entity_id INTEGER NOT NULL,
            rel_relation_id INTEGER NOT NULL,
            obj_entity_id INTEGER NOT NULL,
            domain_id INTEGER NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(document_id),
            FOREIGN KEY(subj_entity_id) REFERENCES entities(entity_id),
            FOREIGN KEY(rel_relation_id) REFERENCES relations(relation_id),
            FOREIGN KEY(obj_entity_id) REFERENCES entities(entity_id),
            FOREIGN KEY(domain_id) REFERENCES domains(domain_id)
        );

        CREATE VIEW aggregated_edges AS
        SELECT
            subj.name AS subj,
            rel.name AS rel,
            obj.name AS obj,
            dom.name AS domain,
            c.canonical_subj AS canonical_subj,
            c.canonical_rel AS canonical_rel,
            c.canonical_obj AS canonical_obj,
            c.canonical_domain AS canonical_domain,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT c.document_id) AS document_support
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        GROUP BY subj.name, rel.name, obj.name, dom.name, c.canonical_subj, c.canonical_rel, c.canonical_obj, c.canonical_domain;

        CREATE VIEW homotopy_localized_claims AS
        WITH exact_surfaces AS (
            SELECT
                canonical_subj,
                canonical_rel,
                canonical_obj,
                surface_form,
                COUNT(DISTINCT document_id) AS exact_document_support
            FROM claims
            GROUP BY canonical_subj, canonical_rel, canonical_obj, surface_form
        )
        SELECT
            c.canonical_subj AS canonical_subj,
            c.canonical_rel AS canonical_rel,
            c.canonical_obj AS canonical_obj,
            MIN(c.canonical_domain) AS canonical_domain,
            MIN(c.statement) AS statement,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT c.document_id) AS document_support,
            COUNT(DISTINCT c.surface_form) AS surface_form_count,
            COUNT(DISTINCT subj.name || '||' || rel.name || '||' || obj.name || '||' || dom.name) AS variant_count,
            COUNT(DISTINCT dom.name) AS domain_alias_count,
            (
                SELECT json_group_array(run_name)
                FROM (
                    SELECT DISTINCT d2.run_name AS run_name
                    FROM claims c2
                    JOIN documents d2 ON d2.document_id = c2.document_id
                    WHERE c2.canonical_subj = c.canonical_subj
                      AND c2.canonical_rel = c.canonical_rel
                      AND c2.canonical_obj = c.canonical_obj
                    ORDER BY d2.run_name
                )
            ) AS supporting_runs_json,
            (
                SELECT json_group_array(surface_form)
                FROM (
                    SELECT DISTINCT c2.surface_form AS surface_form
                    FROM claims c2
                    WHERE c2.canonical_subj = c.canonical_subj
                      AND c2.canonical_rel = c.canonical_rel
                      AND c2.canonical_obj = c.canonical_obj
                    ORDER BY c2.surface_form
                )
            ) AS surface_forms_json,
            (
                SELECT json_group_array(domain_name)
                FROM (
                    SELECT DISTINCT c2.canonical_domain AS domain_name
                    FROM claims c2
                    WHERE c2.canonical_subj = c.canonical_subj
                      AND c2.canonical_rel = c.canonical_rel
                      AND c2.canonical_obj = c.canonical_obj
                    ORDER BY c2.canonical_domain
                )
            ) AS domain_aliases_json,
            MAX(es.exact_document_support) AS exact_document_support_max
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        JOIN documents d ON d.document_id = c.document_id
        JOIN exact_surfaces es
          ON es.canonical_subj = c.canonical_subj
         AND es.canonical_rel = c.canonical_rel
         AND es.canonical_obj = c.canonical_obj
         AND es.surface_form = c.surface_form
        GROUP BY c.canonical_subj, c.canonical_rel, c.canonical_obj;

        CREATE VIEW regime_localized_claims AS
        SELECT
            c.canonical_subj AS canonical_subj,
            c.canonical_rel AS canonical_rel,
            c.canonical_polarity AS canonical_polarity,
            c.canonical_obj AS canonical_obj,
            c.canonical_domain AS canonical_domain,
            MIN(c.statement) AS statement,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT c.document_id) AS document_support,
            COUNT(DISTINCT c.surface_form) AS surface_form_count,
            COUNT(DISTINCT subj.name || '||' || rel.name || '||' || obj.name || '||' || dom.name) AS variant_count,
            (
                SELECT json_group_array(run_name)
                FROM (
                    SELECT DISTINCT d2.run_name AS run_name
                    FROM claims c2
                    JOIN documents d2 ON d2.document_id = c2.document_id
                    WHERE c2.canonical_subj = c.canonical_subj
                      AND c2.canonical_rel = c.canonical_rel
                      AND c2.canonical_polarity = c.canonical_polarity
                      AND c2.canonical_obj = c.canonical_obj
                      AND c2.canonical_domain = c.canonical_domain
                    ORDER BY d2.run_name
                )
            ) AS supporting_runs_json,
            (
                SELECT json_group_array(surface_form)
                FROM (
                    SELECT DISTINCT c2.surface_form AS surface_form
                    FROM claims c2
                    WHERE c2.canonical_subj = c.canonical_subj
                      AND c2.canonical_rel = c.canonical_rel
                      AND c2.canonical_polarity = c.canonical_polarity
                      AND c2.canonical_obj = c.canonical_obj
                      AND c2.canonical_domain = c.canonical_domain
                    ORDER BY c2.surface_form
                )
            ) AS surface_forms_json
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        JOIN documents d ON d.document_id = c.document_id
        GROUP BY c.canonical_subj, c.canonical_rel, c.canonical_polarity, c.canonical_obj, c.canonical_domain;

        CREATE VIEW regime_gluing_surfaces AS
        WITH regime_surface_base AS (
            SELECT
                canonical_subj,
                canonical_obj,
                COUNT(*) AS regime_variant_count,
                COUNT(DISTINCT canonical_domain) AS regime_count,
                COUNT(DISTINCT canonical_rel) AS canonical_relation_count,
                COUNT(DISTINCT canonical_polarity) AS polarity_count,
                SUM(document_support) AS total_document_support,
                MAX(document_support) AS max_regime_support
            FROM regime_localized_claims
            GROUP BY canonical_subj, canonical_obj
        ),
        regime_names AS (
            SELECT
                canonical_subj,
                canonical_obj,
                json_group_array(canonical_domain) AS regimes_json
            FROM (
                SELECT DISTINCT canonical_subj, canonical_obj, canonical_domain
                FROM regime_localized_claims
                ORDER BY canonical_subj, canonical_obj, canonical_domain
            )
            GROUP BY canonical_subj, canonical_obj
        ),
        relation_names AS (
            SELECT
                canonical_subj,
                canonical_obj,
                json_group_array(canonical_rel) AS canonical_relations_json
            FROM (
                SELECT DISTINCT canonical_subj, canonical_obj, canonical_rel
                FROM regime_localized_claims
                ORDER BY canonical_subj, canonical_obj, canonical_rel
            )
            GROUP BY canonical_subj, canonical_obj
        )
        SELECT
            b.canonical_subj,
            b.canonical_obj,
            b.regime_variant_count,
            b.regime_count,
            b.canonical_relation_count,
            b.polarity_count,
            b.total_document_support,
            b.max_regime_support,
            COALESCE(n.regimes_json, json('[]')) AS regimes_json,
            COALESCE(r.canonical_relations_json, json('[]')) AS canonical_relations_json,
            CASE
                WHEN b.polarity_count > 1 THEN 'obstructed'
                WHEN b.regime_count > 1 AND b.canonical_relation_count > 1 THEN 'regime_sensitive'
                WHEN b.regime_count > 1 THEN 'multi_regime_glued'
                ELSE 'single_regime'
            END AS gluing_state
        FROM regime_surface_base b
        LEFT JOIN regime_names n
          ON n.canonical_subj = b.canonical_subj
         AND n.canonical_obj = b.canonical_obj
        LEFT JOIN relation_names r
          ON r.canonical_subj = b.canonical_subj
         AND r.canonical_obj = b.canonical_obj
        WHERE
            b.regime_count > 1
            OR b.polarity_count > 1
            OR b.canonical_relation_count > 1;

        CREATE VIEW document_claim_support AS
        SELECT
            d.run_name,
            d.pdf_path,
            COUNT(*) AS claim_count,
            COUNT(DISTINCT c.domain_id) AS domain_count
        FROM documents d
        LEFT JOIN claims c ON c.document_id = d.document_id
        GROUP BY d.document_id;
        """
    )


def _populate_bundle(
    connection: sqlite3.Connection,
    *,
    records: list[dict[str, object]],
    pdf_dir: Path,
) -> None:
    entity_ids: dict[str, int] = {}
    relation_ids: dict[str, int] = {}
    domain_ids: dict[str, int] = {}

    def intern(table: str, cache: dict[str, int], value: str) -> int:
        normalized = value.strip()
        if normalized in cache:
            return cache[normalized]
        cursor = connection.execute(f"INSERT INTO {table} (name) VALUES (?)", (normalized,))
        cache[normalized] = int(cursor.lastrowid)
        return cache[normalized]

    for record in records:
        run_name = str(record["run_name"])
        triples_path = Path(str(record["triples_path"]))
        pdf_path = record.get("pdf_path") or str(pdf_dir / f"{run_name}.pdf")
        cursor = connection.execute(
            "INSERT INTO documents (run_name, pdf_path, domain_name, triples_path) VALUES (?, ?, ?, ?)",
            (run_name, str(pdf_path), run_name, str(triples_path)),
        )
        document_id = int(cursor.lastrowid)
        for triple in _iter_jsonl_rows(triples_path):
            subj = str(triple.get("subj") or "")
            rel = str(triple.get("rel") or "")
            obj = str(triple.get("obj") or "")
            domain = str(triple.get("domain") or "")
            surface_form = " ".join(f"{subj} {rel} {obj}".split()).strip()
            subj_id = intern("entities", entity_ids, subj)
            rel_id = intern("relations", relation_ids, rel)
            obj_id = intern("entities", entity_ids, obj)
            domain_id = intern("domains", domain_ids, domain)
            connection.execute(
                """
                INSERT INTO claims (
                    document_id, topic, path_json, question, statement,
                    surface_form, canonical_subj, canonical_rel, canonical_polarity, canonical_obj, canonical_domain,
                    subj_entity_id, rel_relation_id, obj_entity_id, domain_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    str(triple.get("topic") or ""),
                    json.dumps(triple.get("path") or []),
                    str(triple.get("question") or ""),
                    str(triple.get("statement") or ""),
                    surface_form,
                    normalize_claim_text(subj),
                    normalize_relation(rel),
                    relation_polarity(rel),
                    normalize_claim_text(obj),
                    normalize_claim_text(domain),
                    subj_id,
                    rel_id,
                    obj_id,
                    domain_id,
                ),
            )


def _compute_summary(connection: sqlite3.Connection, *, pdf_dir: Path) -> dict[str, object]:
    def scalar(query: str) -> int:
        row = connection.execute(query).fetchone()
        return int(row[0] or 0) if row else 0

    def decode_json_array(value: object) -> list[str]:
        if value is None:
            return []
        text = str(value).strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [item for item in text.split(",") if item]
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload if str(item).strip()]

    top_edges = [
        {
            "subj": row[0],
            "rel": row[1],
            "obj": row[2],
            "domain": row[3],
            "canonical_subj": row[4],
            "canonical_rel": row[5],
            "canonical_obj": row[6],
            "canonical_domain": row[7],
            "claim_count": row[8],
            "document_support": row[9],
        }
        for row in connection.execute(
            """
            SELECT
                subj,
                rel,
                obj,
                domain,
                canonical_subj,
                canonical_rel,
                canonical_obj,
                canonical_domain,
                claim_count,
                document_support
            FROM aggregated_edges
            ORDER BY document_support DESC, claim_count DESC, subj, rel, obj
            LIMIT 20
            """
        ).fetchall()
    ]
    top_localized = [
        {
            "canonical_subj": row[0],
            "canonical_rel": row[1],
            "canonical_obj": row[2],
            "canonical_domain": row[3],
            "statement": row[4],
            "claim_count": row[5],
            "document_support": row[6],
            "surface_form_count": row[7],
            "variant_count": row[8],
            "domain_alias_count": row[9],
            "supporting_runs": decode_json_array(row[10]),
            "surface_forms": decode_json_array(row[11]),
            "domain_aliases": decode_json_array(row[12]),
            "exact_document_support_max": row[13],
        }
        for row in connection.execute(
            """
            SELECT
                canonical_subj,
                canonical_rel,
                canonical_obj,
                canonical_domain,
                statement,
                claim_count,
                document_support,
                surface_form_count,
                variant_count,
                domain_alias_count,
                supporting_runs_json,
                surface_forms_json,
                domain_aliases_json,
                exact_document_support_max
            FROM homotopy_localized_claims
            ORDER BY document_support DESC, surface_form_count DESC, claim_count DESC, canonical_subj, canonical_obj
            LIMIT 20
            """
        ).fetchall()
    ]
    top_regime_gluing = [
        {
            "canonical_subj": row[0],
            "canonical_obj": row[1],
            "regime_variant_count": row[2],
            "regime_count": row[3],
            "canonical_relation_count": row[4],
            "polarity_count": row[5],
            "total_document_support": row[6],
            "max_regime_support": row[7],
            "regimes": decode_json_array(row[8]),
            "canonical_relations": decode_json_array(row[9]),
            "gluing_state": row[10],
        }
        for row in connection.execute(
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
            ORDER BY
                CASE gluing_state
                    WHEN 'obstructed' THEN 0
                    WHEN 'regime_sensitive' THEN 1
                    WHEN 'multi_regime_glued' THEN 2
                    ELSE 3
                END,
                total_document_support DESC,
                canonical_subj,
                canonical_obj
            LIMIT 20
            """
        ).fetchall()
    ]
    per_document = [
        {
            "run_name": row[0],
            "pdf_path": row[1],
            "claim_count": row[2],
            "domain_count": row[3],
        }
        for row in connection.execute(
            """
            SELECT run_name, pdf_path, claim_count, domain_count
            FROM document_claim_support
            ORDER BY run_name
            """
        ).fetchall()
    ]
    return {
        "pdf_dir": str(pdf_dir),
        "n_documents": scalar("SELECT COUNT(*) FROM documents"),
        "n_claims": scalar("SELECT COUNT(*) FROM claims"),
        "n_entities": scalar("SELECT COUNT(*) FROM entities"),
        "n_relations": scalar("SELECT COUNT(*) FROM relations"),
        "n_domains": scalar("SELECT COUNT(*) FROM domains"),
        "n_homotopy_localized_claims": scalar("SELECT COUNT(*) FROM homotopy_localized_claims"),
        "n_regime_gluing_surfaces": scalar("SELECT COUNT(*) FROM regime_gluing_surfaces"),
        "top_aggregated_edges": top_edges,
        "top_homotopy_localized_claims": top_localized,
        "top_regime_gluing_surfaces": top_regime_gluing,
        "per_document_claim_support": per_document,
    }
