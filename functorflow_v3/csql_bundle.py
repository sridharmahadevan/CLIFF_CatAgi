"""Batch-level CSQL-style bundle generation for Democritus outputs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


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
            COUNT(*) AS claim_count,
            COUNT(DISTINCT c.document_id) AS document_support
        FROM claims c
        JOIN entities subj ON subj.entity_id = c.subj_entity_id
        JOIN relations rel ON rel.relation_id = c.rel_relation_id
        JOIN entities obj ON obj.entity_id = c.obj_entity_id
        JOIN domains dom ON dom.domain_id = c.domain_id
        GROUP BY subj.name, rel.name, obj.name, dom.name;

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
            subj_id = intern("entities", entity_ids, str(triple.get("subj") or ""))
            rel_id = intern("relations", relation_ids, str(triple.get("rel") or ""))
            obj_id = intern("entities", entity_ids, str(triple.get("obj") or ""))
            domain_id = intern("domains", domain_ids, str(triple.get("domain") or ""))
            connection.execute(
                """
                INSERT INTO claims (
                    document_id, topic, path_json, question, statement,
                    subj_entity_id, rel_relation_id, obj_entity_id, domain_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    str(triple.get("topic") or ""),
                    json.dumps(triple.get("path") or []),
                    str(triple.get("question") or ""),
                    str(triple.get("statement") or ""),
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

    top_edges = [
        {
            "subj": row[0],
            "rel": row[1],
            "obj": row[2],
            "domain": row[3],
            "claim_count": row[4],
            "document_support": row[5],
        }
        for row in connection.execute(
            """
            SELECT subj, rel, obj, domain, claim_count, document_support
            FROM aggregated_edges
            ORDER BY document_support DESC, claim_count DESC, subj, rel, obj
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
        "top_aggregated_edges": top_edges,
        "per_document_claim_support": per_document,
    }
