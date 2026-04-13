"""Export compact public example bundles from Democritus runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportedDocument:
    """Public-facing summary for one selected document."""

    rank: int
    run_name: str
    title: str
    year: str
    score: float | None
    retrieval_backend: str
    identifier: str
    url: str
    download_url: str
    abstract: str
    evidence: tuple[str, ...]
    executive_summary_text: str
    summary_filename: str
    top_claims: tuple[str, ...]
    manifold_image_source: Path | None
    manifold_image_filename: str | None
    root_topics: tuple[str, ...]
    triple_count: int
    repeated_statement_count: int
    summary_is_synthetic: bool


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slugify(text: str, *, max_parts: int = 10, max_length: int = 64) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return "item"
    slug = "-".join(tokens[:max_parts])
    return slug[:max_length].rstrip("-") or "item"


def _extract_top_tier1_claims(markdown_text: str, *, limit: int = 3) -> tuple[str, ...]:
    lines = markdown_text.splitlines()
    claims: list[str] = []
    in_tier1 = False
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line == "## Tier 1 Claims":
            in_tier1 = True
            index += 1
            continue
        if in_tier1 and line.startswith("## "):
            break
        if in_tier1 and re.match(r"^\*\*\d+\.", line):
            candidate = ""
            probe = index + 1
            while probe < len(lines) and not lines[probe].strip():
                probe += 1
            if probe < len(lines) and lines[probe].lstrip().startswith(">"):
                candidate = lines[probe].lstrip()[1:].strip()
            if not candidate:
                candidate = re.sub(r"^\*\*\d+\.\s*", "", line)
                candidate = re.sub(r"\*\*$", "", candidate).strip()
            if candidate:
                claims.append(candidate)
        index += 1
        if len(claims) >= limit:
            break
    return tuple(claims)


def _summarize_batch_agents(batch_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, dict[str, Any]] = {}
    for record in batch_records:
        agent_record = dict(record.get("agent_record") or {})
        agent_name = str(agent_record.get("agent_name") or "").strip()
        if not agent_name:
            continue
        row = by_agent.setdefault(
            agent_name,
            {
                "agent_name": agent_name,
                "completed_records": 0,
                "ok_records": 0,
                "error_records": 0,
                "document_count": 0,
                "total_seconds": 0.0,
                "max_seconds": 0.0,
                "documents": set(),
            },
        )
        row["completed_records"] += 1
        status = str(agent_record.get("status") or "").strip().lower()
        if status == "ok":
            row["ok_records"] += 1
        elif status:
            row["error_records"] += 1
        run_name = str(record.get("run_name") or "").strip()
        if run_name:
            row["documents"].add(run_name)
        started = agent_record.get("started_at")
        ended = agent_record.get("ended_at")
        if isinstance(started, (int, float)) and isinstance(ended, (int, float)) and ended >= started:
            duration = float(ended - started)
            row["total_seconds"] += duration
            row["max_seconds"] = max(float(row["max_seconds"]), duration)

    rows: list[dict[str, Any]] = []
    for row in by_agent.values():
        completed = int(row["completed_records"])
        total_seconds = float(row["total_seconds"])
        rows.append(
            {
                "agent_name": row["agent_name"],
                "completed_records": completed,
                "ok_records": int(row["ok_records"]),
                "error_records": int(row["error_records"]),
                "document_count": len(row["documents"]),
                "avg_seconds": round(total_seconds / completed, 3) if completed else 0.0,
                "max_seconds": round(float(row["max_seconds"]), 3),
                "total_seconds": round(total_seconds, 3),
            }
        )
    rows.sort(key=lambda item: (-item["total_seconds"], item["agent_name"]))
    return rows


def _find_single_file(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _redact_local_paths(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        name = Path(raw).name
        return name or raw

    return re.sub(r"/[^\s)>\"]+", _replace, text)


def _extract_text_from_html(html_text: str) -> str:
    body = html_text
    article_match = re.search(r"<article[^>]*>(.*?)</article>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if article_match:
        body = article_match.group(1)
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</p\s*>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</li\s*>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", " ", body)
    body = unescape(body)
    return _redact_local_paths(_normalize_space(body.replace("\xa0", " ")))


def _html_has_real_content(html_text: str) -> bool:
    extracted = _extract_text_from_html(html_text)
    if not extracted:
        return False
    lowered = extracted.lower()
    if "artifact not available yet." in lowered:
        return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _statement_bucket(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("emperor penguin", "penguin", "sea ice", "breeding")):
        return "penguin"
    if any(token in lowered for token in ("krill", "fur seal", "prey")):
        return "food_web"
    return "other"


def _summarize_run_artifacts(run_root: Path, *, max_topics: int = 6, max_claims: int = 5) -> dict[str, Any]:
    root_topics_path = run_root / "configs" / "root_topics.txt"
    root_topics = tuple(
        line.strip()
        for line in _read_text(root_topics_path).splitlines()
        if line.strip()
    )[:max_topics]

    causal_statements_path = run_root / "causal_statements.jsonl"
    statement_counter: Counter[str] = Counter()
    statement_example: dict[str, str] = {}
    for row in _load_jsonl(causal_statements_path):
        for statement in row.get("statements") or []:
            raw = _normalize_space(str(statement))
            if not raw:
                continue
            key = raw.lower()
            statement_counter[key] += 1
            statement_example.setdefault(key, raw)
    repeated_statements: list[tuple[str, int]] = []
    chosen_keys: set[str] = set()
    chosen_buckets: set[str] = set()
    ranked_items = list(statement_counter.most_common())
    for key, count in ranked_items:
        bucket = _statement_bucket(statement_example[key])
        if bucket in {"penguin", "food_web"} and bucket not in chosen_buckets:
            repeated_statements.append((statement_example[key], count))
            chosen_keys.add(key)
            chosen_buckets.add(bucket)
        if len(repeated_statements) >= max_claims:
            break
    for key, count in ranked_items:
        if key in chosen_keys:
            continue
        if count < 2 and repeated_statements:
            break
        repeated_statements.append((statement_example[key], count))
        chosen_keys.add(key)
        if len(repeated_statements) >= max_claims:
            break
    if not repeated_statements:
        for key, count in statement_counter.most_common(max_claims):
            repeated_statements.append((statement_example[key], count))

    relational_triples_path = run_root / "relational_triples.jsonl"
    triple_rows = _load_jsonl(relational_triples_path)
    triple_count = len(triple_rows)

    return {
        "root_topics": root_topics,
        "repeated_statements": tuple(repeated_statements),
        "triple_count": triple_count,
        "repeated_statement_count": sum(count for _, count in repeated_statements),
    }


def _build_synthetic_summary_markdown(*, title: str, run_root: Path, summary: dict[str, Any]) -> str:
    lines = [
        f"This public example was synthesized from saved run artifacts for **{title}** because the archived executive-summary file was not available in Markdown form.",
        "",
    ]
    root_topics = list(summary.get("root_topics") or [])
    if root_topics:
        lines.extend(["## Root Topics", ""])
        for topic in root_topics:
            lines.append(f"- {topic}")
        lines.append("")

    repeated_statements = list(summary.get("repeated_statements") or [])
    if repeated_statements:
        lines.extend(["## Repeated Causal Statements", ""])
        for statement, count in repeated_statements:
            suffix = f" ({count} supporting variants)" if count > 1 else ""
            lines.append(f"- {statement}{suffix}")
        lines.append("")

    lines.extend(
        [
            "## Run Diagnostics",
            "",
            f"- Relational triples extracted: {int(summary.get('triple_count') or 0)}",
            f"- Root topics recovered: {len(root_topics)}",
            f"- Saved run directory: `{run_root.name}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip()


def _infer_run_names(
    selected_documents: list[dict[str, Any]],
    *,
    batch_records: list[dict[str, Any]],
    batch_outdir: Path,
) -> list[str]:
    inferred: list[str] = []
    seen: set[str] = set()
    for record in batch_records:
        run_name = str(record.get("run_name") or "").strip()
        if run_name and run_name not in seen:
            inferred.append(run_name)
            seen.add(run_name)
    if not inferred:
        for child in sorted(batch_outdir.iterdir()) if batch_outdir.exists() else []:
            if child.is_dir() and child.name != "csql":
                inferred.append(child.name)
    run_names: list[str] = []
    for index, document in enumerate(selected_documents):
        explicit = str(document.get("run_name") or "").strip()
        if explicit:
            run_names.append(explicit)
        elif index < len(inferred):
            run_names.append(inferred[index])
        else:
            run_names.append(f"document_{index + 1:02d}")
    return run_names


def _build_document_markdown(document: ExportedDocument) -> str:
    lines = [
        f"# {document.title}",
        "",
        f"- Rank: {document.rank}",
        f"- Year: {document.year or 'unknown'}",
        f"- Retrieval backend: {document.retrieval_backend or 'unknown'}",
        f"- Score: {document.score if document.score is not None else 'n/a'}",
    ]
    if document.identifier:
        lines.append(f"- Identifier: {document.identifier}")
    if document.url:
        lines.append(f"- URL: {document.url}")
    if document.download_url:
        lines.append(f"- Download URL: {document.download_url}")
    if document.top_claims:
        lines.extend(["", "## Top Tier 1 Claims", ""])
        for claim in document.top_claims:
            lines.append(f"- {claim}")
    if document.root_topics and not document.summary_is_synthetic:
        lines.extend(["", "## Root Topics", ""])
        for topic in document.root_topics:
            lines.append(f"- {topic}")
    if document.abstract:
        lines.extend(["", "## Abstract Snapshot", "", document.abstract.strip()])
    lines.extend(["", "## CLIFF Executive Summary", ""])
    if document.executive_summary_text.strip():
        lines.append(document.executive_summary_text.strip())
    else:
        lines.append("_No executive summary was available in the saved run; this page preserves the document metadata only._")
    if document.triple_count and not document.summary_is_synthetic:
        lines.extend(
            [
                "",
                "## Run Diagnostics",
                "",
                f"- Relational triples extracted: {document.triple_count}",
                f"- Repeated top-claim support count: {document.repeated_statement_count}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _sanitize_display_value(value: str) -> str:
    return _redact_local_paths(_normalize_space(value))


def _sanitize_markdown_text(value: str) -> str:
    return _redact_local_paths(value).strip()


def _sanitize_query_plan(query_plan: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(query_plan)
    for key in ("query", "normalized_query", "base_query", "retrieval_query", "retrieval_refinement"):
        if key in sanitized and isinstance(sanitized[key], str):
            sanitized[key] = _sanitize_display_value(sanitized[key])
    sanitized["direct_document_paths"] = [Path(str(item)).name for item in query_plan.get("direct_document_paths") or []]
    sanitized["direct_document_directories"] = [
        Path(str(item)).name for item in query_plan.get("direct_document_directories") or []
    ]
    return sanitized


def _build_readme(
    *,
    query_plan: dict[str, Any],
    execution_mode: str,
    retrieval_backend: str,
    selected_documents: list[dict[str, Any]],
    exported_documents: list[ExportedDocument],
    stage_summary: list[dict[str, Any]],
    included_image_count: int,
    source_selected_document_count: int,
) -> str:
    query = _sanitize_display_value(str(query_plan.get("query") or ""))
    retrieval_query = _sanitize_display_value(str(query_plan.get("retrieval_query") or ""))
    lines = [
        "# Democritus Example Bundle",
        "",
        "This is a compact, GitHub-friendly export of a saved CLIFF Democritus run.",
        "The heavyweight artifacts from the original run were intentionally excluded:",
        "PDF inputs, sweep outputs, PKL state, SQLite databases, and large report assets.",
        "",
        "## Query",
        "",
        f"- Query: {query}",
        f"- Retrieval query: {retrieval_query}",
        f"- Execution mode: {execution_mode}",
        f"- Retrieval backend: {retrieval_backend}",
        f"- Target documents: {query_plan.get('target_documents')}",
        f"- Selected documents: {len(selected_documents)}",
        f"- Included manifold images: {included_image_count}",
        "",
    ]
    if len(selected_documents) != source_selected_document_count:
        lines.extend(
            [
                "## Curation",
                "",
                f"- Source run selected documents: {source_selected_document_count}",
                f"- Public release subset: {len(selected_documents)} documents",
                "- This bundle keeps the most on-topic studies from the original saved run and omits obvious drift, corrections, and review-only artifacts.",
                "",
            ]
        )
    lines.extend(
        [
        "## Included Files",
        "",
        "- `query_plan.json`: sanitized request and retrieval plan",
        "- `selected_documents.json`: compact metadata for the selected studies",
        "- `batch_stage_summary.json`: aggregated stage timings across the run",
        "- `documents/`: one Markdown executive summary per selected study",
        "- `images/`: a small sample of 2D manifold plots",
        "",
        "## Selected Documents",
        "",
        "| Rank | Year | Score | Backend | Title | Summary |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for document in exported_documents:
        summary_link = f"[summary](documents/{document.summary_filename})"
        lines.append(
            f"| {document.rank} | {document.year or '-'} | "
            f"{document.score if document.score is not None else '-'} | "
            f"{document.retrieval_backend or '-'} | {document.title} | {summary_link} |"
        )

    lines.extend(["", "## Stage Summary", "", "| Agent | Records | Avg Sec | Max Sec | Total Sec |", "| --- | --- | --- | --- | --- |"])
    for row in stage_summary[:8]:
        lines.append(
            f"| {row['agent_name']} | {row['completed_records']} | {row['avg_seconds']} | "
            f"{row['max_seconds']} | {row['total_seconds']} |"
        )

    lines.extend(["", "## Representative Claims", ""])
    for document in exported_documents[:5]:
        lines.append(f"### {document.rank}. {document.title}")
        if document.top_claims:
            for claim in document.top_claims[:2]:
                lines.append(f"- {claim}")
        else:
            lines.append("- No Tier 1 claims were available in the exported summary.")
        if document.manifold_image_filename:
            lines.append(f"- 2D manifold image: `images/{document.manifold_image_filename}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_democritus_example(
    run_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
    copy_manifold_images: int = 3,
    top_claims_per_document: int = 3,
    document_ranks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Export a compact public bundle from a saved Democritus run."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    query_run_summary_path = run_dir / "query_run_summary.json"
    if not query_run_summary_path.exists():
        raise FileNotFoundError(f"Expected query summary at {query_run_summary_path}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    images_dir = output_dir / "images"
    documents_dir.mkdir()
    images_dir.mkdir()

    query_summary = _load_json(query_run_summary_path)
    query_plan = _sanitize_query_plan(dict(query_summary.get("query_plan") or {}))
    source_selected_documents = list(query_summary.get("selected_documents") or [])
    selected_documents = source_selected_documents
    if document_ranks:
        allowed = set(document_ranks)
        selected_documents = [
            document for index, document in enumerate(source_selected_documents, start=1) if index in allowed
        ]
    batch_outdir = Path(str(query_summary.get("batch_outdir") or run_dir / "democritus_runs")).resolve()
    if not batch_outdir.exists():
        archive_batch_outdir = run_dir / "democritus_runs"
        if archive_batch_outdir.exists():
            batch_outdir = archive_batch_outdir.resolve()
    batch_summary_path = batch_outdir / "batch_agent_run_summary.json"
    if batch_summary_path.exists():
        batch_records = list(_load_json(batch_summary_path))
    else:
        batch_records = list(query_summary.get("batch_records") or [])
    all_inferred_run_names = _infer_run_names(
        source_selected_documents,
        batch_records=batch_records,
        batch_outdir=batch_outdir,
    )
    if document_ranks:
        inferred_run_names = [
            run_name for index, run_name in enumerate(all_inferred_run_names, start=1) if index in set(document_ranks)
        ]
    else:
        inferred_run_names = all_inferred_run_names
    if document_ranks:
        selected_run_names = set(inferred_run_names)
        stage_summary = _summarize_batch_agents(
            [record for record in batch_records if str(record.get("run_name") or "").strip() in selected_run_names]
        )
    else:
        stage_summary = _summarize_batch_agents(batch_records)

    exported_documents: list[ExportedDocument] = []
    for index, document in enumerate(selected_documents, start=1):
        run_name = inferred_run_names[index - 1]
        title = str(document.get("title") or run_name).strip()
        run_root = batch_outdir / run_name
        executive_summary_path = _find_single_file("reports/*_executive_summary.md", run_root)
        executive_summary_text = _read_text(executive_summary_path)
        if not executive_summary_text:
            executive_summary_html_path = _find_single_file("reports/*_executive_summary.html", run_root)
            executive_summary_html = _read_text(executive_summary_html_path)
            if _html_has_real_content(executive_summary_html):
                executive_summary_text = _extract_text_from_html(executive_summary_html)

        top_claims = _extract_top_tier1_claims(executive_summary_text, limit=top_claims_per_document)
        synthetic_summary = _summarize_run_artifacts(run_root, max_claims=max(top_claims_per_document, 5))
        if not executive_summary_text.strip():
            executive_summary_text = _build_synthetic_summary_markdown(
                title=title,
                run_root=run_root,
                summary=synthetic_summary,
            )
            summary_is_synthetic = True
        else:
            summary_is_synthetic = False
        if not top_claims:
            top_claims = tuple(
                statement for statement, _count in list(synthetic_summary.get("repeated_statements") or [])[:top_claims_per_document]
            )
        slug = _slugify(title)
        summary_filename = f"{index:02d}_{slug}.md"
        manifold_image_source = None
        manifold_image_filename = None
        if index <= max(copy_manifold_images, 0):
            image_path = _find_single_file("viz/relational_manifold_2d.png", run_root)
            if image_path and image_path.exists():
                manifold_image_source = image_path
                manifold_image_filename = f"{index:02d}_{slug}_manifold_2d.png"

        exported = ExportedDocument(
            rank=index,
            run_name=run_name,
            title=_sanitize_display_value(title),
            year=str(document.get("year") or ""),
            score=float(document["score"]) if isinstance(document.get("score"), (int, float)) else None,
            retrieval_backend=str(document.get("retrieval_backend") or ""),
            identifier=_sanitize_display_value(str(document.get("identifier") or "")),
            url=str(document.get("url") or ""),
            download_url=str(document.get("download_url") or ""),
            abstract=_sanitize_display_value(str(document.get("abstract") or "").strip()),
            evidence=tuple(str(item) for item in (document.get("evidence") or [])),
            executive_summary_text=_sanitize_markdown_text(executive_summary_text),
            summary_filename=summary_filename,
            top_claims=tuple(_sanitize_display_value(claim) for claim in top_claims),
            manifold_image_source=manifold_image_source,
            manifold_image_filename=manifold_image_filename,
            root_topics=tuple(_sanitize_display_value(topic) for topic in (synthetic_summary.get("root_topics") or ())),
            triple_count=int(synthetic_summary.get("triple_count") or 0),
            repeated_statement_count=int(synthetic_summary.get("repeated_statement_count") or 0),
            summary_is_synthetic=summary_is_synthetic,
        )
        exported_documents.append(exported)

        (documents_dir / summary_filename).write_text(_build_document_markdown(exported), encoding="utf-8")
        if manifold_image_source and manifold_image_filename:
            shutil.copy2(manifold_image_source, images_dir / manifold_image_filename)

    selected_document_rows = []
    for document in exported_documents:
        selected_document_rows.append(
            {
                "rank": document.rank,
                "run_name": document.run_name,
                "title": document.title,
                "year": document.year,
                "score": document.score,
                "retrieval_backend": document.retrieval_backend,
                "identifier": document.identifier,
                "url": document.url,
                "download_url": document.download_url,
                "evidence": list(document.evidence),
                "abstract": document.abstract,
                "summary_path": f"documents/{document.summary_filename}",
                "top_tier1_claims": list(document.top_claims),
                "root_topics": list(document.root_topics),
                "triple_count": document.triple_count,
                "manifold_image_path": (
                    f"images/{document.manifold_image_filename}" if document.manifold_image_filename else None
                ),
            }
        )

    execution_mode = str(query_summary.get("execution_mode") or "")
    retrieval_backend = str(query_summary.get("retrieval_backend") or "")
    included_image_count = sum(1 for document in exported_documents if document.manifold_image_filename)
    readme_text = _build_readme(
        query_plan=query_plan,
        execution_mode=execution_mode,
        retrieval_backend=retrieval_backend,
        selected_documents=selected_documents,
        exported_documents=exported_documents,
        stage_summary=stage_summary,
        included_image_count=included_image_count,
        source_selected_document_count=len(source_selected_documents),
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    _write_json(output_dir / "query_plan.json", query_plan)
    _write_json(output_dir / "selected_documents.json", selected_document_rows)
    _write_json(output_dir / "batch_stage_summary.json", stage_summary)

    manifest = {
        "bundle_version": 1,
        "route": "democritus",
        "query": query_plan.get("query"),
        "retrieval_query": query_plan.get("retrieval_query"),
        "execution_mode": execution_mode,
        "retrieval_backend": retrieval_backend,
        "target_documents": query_plan.get("target_documents"),
        "selected_document_count": len(selected_document_rows),
        "source_selected_document_count": len(source_selected_documents),
        "included_image_count": included_image_count,
        "documents_dir": "documents",
        "images_dir": "images",
        "document_ranks": list(document_ranks) if document_ranks else [],
        "files": [
            "README.md",
            "query_plan.json",
            "selected_documents.json",
            "batch_stage_summary.json",
            "example_manifest.json",
        ],
    }
    _write_json(output_dir / "example_manifest.json", manifest)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a compact public Democritus example bundle.")
    parser.add_argument("--run-dir", required=True, help="Path to the saved Democritus run directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the compact example bundle should be written.")
    parser.add_argument(
        "--copy-manifold-images",
        type=int,
        default=3,
        help="Number of selected-document 2D manifold images to include.",
    )
    parser.add_argument(
        "--top-claims-per-document",
        type=int,
        default=3,
        help="Number of Tier 1 claims to surface per document in the exported metadata.",
    )
    parser.add_argument(
        "--document-ranks",
        default="",
        help="Comma-separated 1-based selected-document ranks to keep from the saved run.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it already exists.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifest = export_democritus_example(
        Path(args.run_dir),
        Path(args.output_dir),
        force=bool(args.force),
        copy_manifold_images=int(args.copy_manifold_images),
        top_claims_per_document=int(args.top_claims_per_document),
        document_ranks=tuple(
            int(part.strip())
            for part in str(args.document_ranks).split(",")
            if part.strip()
        )
        or None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
