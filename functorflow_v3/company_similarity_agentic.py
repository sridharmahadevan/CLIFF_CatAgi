"""Natural-language company similarity runner built on temporal diffusion outputs."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .democritus_query_agentic import _resolve_sec_user_agent
from .repo_layout import repo_root, resolve_brand_panel_root
from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html


def _slugify(value: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "company"


def _normalize_alias(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9& ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _infer_ticker_from_aliases(brand: str, slug: str, aliases: list[str]) -> str:
    brand_norm = _normalize_alias(brand)
    slug_norm = _normalize_alias(slug.replace("_", " "))
    for alias in aliases:
        token = alias.strip().lower()
        if not token or token in {brand_norm, slug_norm}:
            continue
        if re.fullmatch(r"[a-z]{1,5}", token):
            return token
    return ""


def _is_matchable_company_alias(alias: str) -> bool:
    normalized = str(alias or "").strip().lower()
    if len(normalized) >= 3:
        return True
    return any(char.isdigit() for char in normalized)


def _safe_read_json(path: Path) -> dict[str, object]:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _resolve_brand_workspace_root() -> Path:
    candidate = resolve_brand_panel_root()
    if (candidate / "configs" / "company_batch_registry.csv").exists():
        return candidate
    sibling = resolve_brand_panel_root()
    if (sibling / "configs" / "company_batch_registry.csv").exists():
        return sibling.resolve()
    return candidate


def _portable_output_path(raw_path: str, *, brand_root: Path, fallback: Path) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return fallback.resolve()
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    parts = candidate.parts
    if "outputs" in parts:
        outputs_index = parts.index("outputs")
        translated = brand_root / Path(*parts[outputs_index:])
        return translated.resolve()
    return fallback.resolve()


@dataclass(frozen=True)
class CompanySimilarityQueryPlan:
    query: str
    company_a: str
    company_b: str
    company_a_slug: str
    company_b_slug: str
    intent: str = "company_similarity"


@dataclass(frozen=True)
class _CompanyRecord:
    brand: str
    slug: str
    aliases: tuple[str, ...]
    ticker: str = ""
    index_url: str = ""
    outdir: Path | None = None
    existing_combined_dir: Path | None = None


@dataclass(frozen=True)
class CompanySimilarityRunResult:
    query_plan: CompanySimilarityQueryPlan
    route_outdir: Path
    analysis_dir: Path
    summary_path: Path
    artifact_path: Path | None
    company_a_manifest_path: Path | None
    company_b_manifest_path: Path | None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _company_similarity_parallelism_capacity(stage_state: dict[str, dict[str, object]]) -> float:
    company_builds_seen = 0
    for stage_key, stage in stage_state.items():
        if not stage_key.endswith("_analysis") or stage_key == "functor_analysis":
            continue
        status = str(stage.get("status") or "").lower()
        if status in {"active", "complete"}:
            company_builds_seen += 1
    return 2.0 if company_builds_seen >= 2 else 1.0


def _company_similarity_stage_concurrency(stage_state: dict[str, dict[str, object]], *, at_time: float) -> float:
    concurrent = 0.0
    for stage in stage_state.values():
        started = float(stage.get("started_at_epoch") or 0.0)
        ended = float(stage.get("ended_at_epoch") or 0.0)
        if started <= 0.0:
            continue
        if started <= at_time and (ended <= 0.0 or at_time < ended):
            concurrent += 1.0
    return max(1.0, concurrent) if concurrent > 0 else 1.0


def _company_similarity_peak_parallelism(stage_state: dict[str, dict[str, object]], *, now: float) -> float:
    event_times: set[float] = set()
    for stage in stage_state.values():
        started = float(stage.get("started_at_epoch") or 0.0)
        ended = float(stage.get("ended_at_epoch") or 0.0)
        if started > 0.0:
            event_times.add(started)
        if ended > 0.0:
            event_times.add(max(started, ended - 1e-6))
    if not event_times:
        return 1.0
    peak = 1.0
    for event_time in sorted(event_times):
        peak = max(peak, _company_similarity_stage_concurrency(stage_state, at_time=min(event_time, now)))
    return peak


def _company_similarity_current_stage_label(
    active_stages: list[dict[str, object]],
    pending_stages: list[dict[str, object]],
    *,
    status: str,
) -> str:
    if active_stages:
        labels = [str(stage.get("label") or stage.get("stage_key") or "stage") for stage in active_stages]
        return " + ".join(labels[:2])
    if pending_stages:
        return str(pending_stages[0].get("label") or pending_stages[0].get("stage_key") or "warming up")
    if str(status).lower() == "complete":
        return "Complete"
    return "Warming up"


def _company_similarity_mode_profile(execution_mode: str) -> dict[str, int | str]:
    normalized_mode = "deep" if str(execution_mode).strip().lower() == "deep" else "quick"
    current_year = time.localtime().tm_year
    if normalized_mode == "deep":
        return {
            "execution_mode": normalized_mode,
            "year_start": 2002,
            "year_end": current_year,
            "jobs": 1,
            "llm_jobs": 1,
            "filings_profile": "fast",
            "epochs": 3,
            "batch_size": 4,
        }
    return {
        "execution_mode": normalized_mode,
        "year_start": max(2002, current_year - 2),
        "year_end": current_year,
        "jobs": 3,
        "llm_jobs": 3,
        "filings_profile": "fast",
        "epochs": 1,
        "batch_size": 6,
        "skip_visualization": 1,
        "skip_branch_visuals": 1,
    }


def looks_like_company_similarity_query(query: str) -> bool:
    normalized = _normalize_alias(query)
    if not normalized:
        return False
    similarity_markers = (
        "similar",
        "similarity",
        "compare",
        "comparison",
        "versus",
        " vs ",
    )
    if not any(marker in normalized for marker in similarity_markers):
        return False
    return " to " in normalized or " and " in normalized or " vs " in normalized or " versus " in normalized


def _load_company_registry(brand_root: Path) -> dict[str, _CompanyRecord]:
    records: dict[str, _CompanyRecord] = {}
    registry_path = brand_root / "configs" / "company_batch_registry.csv"
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                brand = str(row.get("brand", "")).strip()
                if not brand:
                    continue
                slug = _slugify(brand)
                aliases = [_normalize_alias(brand), _normalize_alias(slug.replace("_", " "))]
                alias_field = str(row.get("company_aliases", "")).strip()
                if alias_field:
                    aliases.extend(_normalize_alias(item.replace("_", " ")) for item in alias_field.split(","))
                ticker = str(row.get("edgar_ticker", "")).strip().lower()
                if not ticker:
                    ticker = _infer_ticker_from_aliases(brand, slug, aliases)
                if len(ticker) >= 3:
                    aliases.append(ticker)
                outdir_raw = str(row.get("outdir", "")).strip()
                default_outdir = (brand_root / "outputs" / slug).resolve()
                outdir = _portable_output_path(outdir_raw, brand_root=brand_root, fallback=default_outdir)
                combined_raw = str(row.get("existing_combined_dir", "")).strip()
                default_combined = default_outdir / f"runs_{slug}_financial_filings" / f"atlas_{slug}_financial_combined"
                combined_dir = (
                    _portable_output_path(combined_raw, brand_root=brand_root, fallback=default_combined)
                    if combined_raw
                    else None
                )
                records[slug] = _CompanyRecord(
                    brand=brand,
                    slug=slug,
                    aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                    ticker=ticker,
                    index_url=str(row.get("index_url", "")).strip(),
                    outdir=outdir,
                    existing_combined_dir=combined_dir if combined_dir else None,
                )
    outputs_root = brand_root / "outputs"
    if outputs_root.exists():
        for manifest_path in sorted(outputs_root.glob("*/add_company_analysis_manifest.json")):
            payload = _safe_read_json(manifest_path)
            brand = str(payload.get("brand", "")).strip() or manifest_path.parent.name.replace("_", " ").title()
            slug = str(payload.get("brand_slug", "")).strip() or _slugify(brand)
            existing = records.get(slug)
            aliases = list(existing.aliases) if existing else []
            aliases.extend((_normalize_alias(brand), _normalize_alias(slug.replace("_", " ")), _normalize_alias(manifest_path.parent.name.replace("_", " "))))
            if existing and existing.ticker and len(existing.ticker) >= 3:
                aliases.append(existing.ticker)
            records[slug] = _CompanyRecord(
                brand=brand,
                slug=slug,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                ticker=existing.ticker if existing else "",
                index_url=existing.index_url if existing else "",
                outdir=manifest_path.parent.resolve(),
                existing_combined_dir=existing.existing_combined_dir if existing else None,
            )
    preset_path = brand_root / "configs" / "company_universe_presets.json"
    if preset_path.exists():
        payload = _safe_read_json(preset_path)
        presets = payload.get("presets", {})
        if isinstance(presets, dict):
            for preset in presets.values():
                companies = preset.get("companies", [])
                if not isinstance(companies, list):
                    continue
                for item in companies:
                    if not isinstance(item, dict):
                        continue
                    brand = str(item.get("brand", "")).strip()
                    if not brand:
                        continue
                    slug = _slugify(brand)
                    ticker = str(item.get("ticker", "")).strip().lower()
                    existing = records.get(slug)
                    aliases = list(existing.aliases) if existing else []
                    aliases.extend(
                        (
                            _normalize_alias(brand),
                            _normalize_alias(slug.replace("_", " ")),
                        )
                    )
                    if ticker and len(ticker) >= 1:
                        aliases.append(ticker)
                    records[slug] = _CompanyRecord(
                        brand=existing.brand if existing else brand,
                        slug=slug,
                        aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
                        ticker=existing.ticker if existing and existing.ticker else ticker,
                        index_url=existing.index_url if existing else "",
                        outdir=existing.outdir if existing else (brand_root / "outputs" / slug).resolve(),
                        existing_combined_dir=existing.existing_combined_dir if existing else None,
                    )
    return records


def _extract_companies_from_query(query: str, records: dict[str, _CompanyRecord]) -> tuple[_CompanyRecord, _CompanyRecord]:
    normalized = f" {_normalize_alias(query)} "
    matches: list[tuple[int, int, _CompanyRecord]] = []
    for record in records.values():
        best_position: int | None = None
        best_length = -1
        for alias in record.aliases:
            alias = alias.strip()
            if not alias or not _is_matchable_company_alias(alias):
                continue
            pattern = f" {alias} "
            position = normalized.find(pattern)
            if position < 0:
                continue
            if best_position is None or position < best_position or (position == best_position and len(alias) > best_length):
                best_position = position
                best_length = len(alias)
        if best_position is not None:
            matches.append((best_position, best_length, record))
    matches.sort(key=lambda item: (item[0], -item[1], item[2].brand.lower()))
    ordered: list[_CompanyRecord] = []
    seen = set()
    for _, _, record in matches:
        if record.slug in seen:
            continue
        seen.add(record.slug)
        ordered.append(record)
    if len(ordered) < 2:
        raise ValueError(
            "Could not confidently identify two companies in the similarity query. "
            "Try phrasing it like 'How similar is Adobe to Nike?' or 'Compare Adobe and Nike.'"
        )
    return ordered[0], ordered[1]


def interpret_company_similarity_query(query: str) -> CompanySimilarityQueryPlan:
    brand_root = _resolve_brand_workspace_root()
    records = _load_company_registry(brand_root)
    company_a, company_b = _extract_companies_from_query(query, records)
    return CompanySimilarityQueryPlan(
        query=" ".join(query.split()),
        company_a=company_a.brand,
        company_b=company_b.brand,
        company_a_slug=company_a.slug,
        company_b_slug=company_b.slug,
    )


def _find_company_record(plan: CompanySimilarityQueryPlan, brand_root: Path, slug: str) -> _CompanyRecord:
    records = _load_company_registry(brand_root)
    record = records.get(slug)
    if record is not None:
        return record
    brand = plan.company_a if slug == plan.company_a_slug else plan.company_b
    return _CompanyRecord(
        brand=brand,
        slug=slug,
        aliases=(_normalize_alias(brand), _normalize_alias(slug.replace("_", " "))),
        outdir=(brand_root / "outputs" / slug).resolve(),
    )


def _combined_dir_candidates(record: _CompanyRecord) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if record.existing_combined_dir is not None:
        candidates.append(record.existing_combined_dir)
    if record.outdir is not None:
        candidates.append(record.outdir / f"runs_{record.slug}_financial_filings" / f"atlas_{record.slug}_financial_combined")
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _manifest_path(record: _CompanyRecord) -> Path:
    base = record.outdir or (_resolve_brand_workspace_root() / "outputs" / record.slug)
    return base / "add_company_analysis_manifest.json"


def _existing_local_filing_manifest(record: _CompanyRecord) -> Path | None:
    if record.outdir is None:
        return None
    filings_outdir = record.outdir / f"runs_{record.slug}_financial_filings"
    candidates = (
        filings_outdir / f"{record.slug}_sec_edgar_manifest.csv",
        filings_outdir / f"{record.slug}_sec_edgar_manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _existing_combined_dir(record: _CompanyRecord) -> Path | None:
    for candidate in _combined_dir_candidates(record):
        if candidate.exists():
            return candidate
    manifest_path = _manifest_path(record)
    if manifest_path.exists():
        payload = _safe_read_json(manifest_path)
        raw = str(payload.get("combined_dir", "")).strip()
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate.resolve()
    return None


def _run_command(command: list[str], *, cwd: Path, stream_label: str = "") -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    log_prefix = f"[company_similarity][{stream_label}]" if stream_label else "[company_similarity]"
    print(f"{log_prefix} command: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    stdout_stream = process.stdout or io.StringIO("")
    with stdout_stream:
        for raw_line in stdout_stream:
            line = raw_line.rstrip("\n")
            print(f"{log_prefix} {line}" if stream_label else line, flush=True)
            if not line.strip():
                continue
            tail.append(f"{log_prefix} {line}" if stream_label else line)
            if len(tail) > 20:
                tail.pop(0)
    return_code = process.wait()
    if return_code != 0:
        details = "\n".join(tail).strip() or "No stdout/stderr was captured."
        raise RuntimeError(
            "Company similarity backend command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"CWD: {cwd}\n"
            f"Exit code: {return_code}\n"
            f"Last output:\n{details}"
        )


def _python_has_modules(python_executable: str, modules: tuple[str, ...], *, cwd: Path | None = None) -> bool:
    probe = [
        python_executable,
        "-c",
        "import importlib.util, sys; "
        + "; ".join(
            f"assert importlib.util.find_spec({module!r}) is not None, {module!r}" for module in modules
        )
        + "; print(sys.executable)",
    ]
    try:
        subprocess.run(
            probe,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
        )
        return True
    except Exception:
        return False


def _select_python_for_brand_pipeline() -> str:
    env_override = os.environ.get("CLIFF_BRAND_PIPELINE_PYTHON", "").strip()
    workspace_root = repo_root().parents[0]
    brand_root = resolve_brand_panel_root()
    required_modules = (
        "brand_democritus_block_denoise",
        "pandas",
        "pyarrow",
        "matplotlib",
        "tqdm",
        "umap",
    )
    candidates = [
        env_override,
        sys.executable,
        str((brand_root / ".venv" / "bin" / "python")),
        str((repo_root() / ".venv" / "bin" / "python")),
        "/opt/homebrew/bin/python3",
        "python3",
    ]
    for candidate in dict.fromkeys(item for item in candidates if item):
        if candidate != "python3" and not Path(candidate).exists():
            continue
        if _python_has_modules(candidate, required_modules, cwd=workspace_root):
            return candidate
    raise RuntimeError(
        "Could not find a Python interpreter with the required brand diffusion pipeline dependencies "
        "(brand_democritus_block_denoise, pandas, pyarrow, matplotlib, tqdm, umap). "
        "If needed, set CLIFF_BRAND_PIPELINE_PYTHON to the correct interpreter."
    )


def _ensure_company_analysis(
    *,
    record: _CompanyRecord,
    sec_user_agent: str,
    workspace_root: Path,
    python_executable: str,
    execution_mode: str,
) -> tuple[Path, Path | None]:
    combined_dir = _existing_combined_dir(record)
    manifest_path = _manifest_path(record)
    if combined_dir is not None:
        return combined_dir, manifest_path if manifest_path.exists() else None
    if record.outdir is None:
        raise ValueError(f"No output directory is configured for {record.brand}.")
    if not record.index_url and not record.ticker:
        raise ValueError(
            f"No retrieval metadata is configured for {record.brand}. "
            "Add an index URL or EDGAR ticker to the company registry before requesting this comparison."
        )
    existing_filing_manifest = _existing_local_filing_manifest(record)
    profile = _company_similarity_mode_profile(execution_mode)
    command = [
        python_executable,
        "-m",
        "brand_democritus_block_denoise.add_company_analysis",
        "--brand",
        record.brand,
        "--company-aliases",
        ",".join(dict.fromkeys(alias.replace(" ", "_") for alias in record.aliases if alias)),
        "--outdir",
        str(record.outdir),
        "--jobs",
        str(profile["jobs"]),
        "--llm-jobs",
        str(profile["llm_jobs"]),
        "--filings-profile",
        str(profile["filings_profile"]),
        "--basis-mode",
        "motif",
        "--block-size",
        "3",
        "--epochs",
        str(profile["epochs"]),
        "--batch-size",
        str(profile["batch_size"]),
        "--device",
        "cpu",
        "--year-start",
        str(profile["year_start"]),
        "--year-end",
        str(profile["year_end"]),
    ]
    if existing_filing_manifest is not None:
        command.extend(["--manifest", str(existing_filing_manifest)])
    elif record.ticker:
        resolved_sec_user_agent = _resolve_sec_user_agent(sec_user_agent)
        command.extend(["--edgar-ticker", record.ticker])
        command.extend(["--sec-user-agent", resolved_sec_user_agent])
    elif record.index_url:
        command.extend(["--index-url", record.index_url])
    if int(profile.get("skip_branch_visuals", 0)):
        command.append("--skip-visuals")
    _run_command(command, cwd=workspace_root, stream_label=record.brand)
    combined_dir = _existing_combined_dir(record)
    if combined_dir is None:
        raise FileNotFoundError(f"Expected combined atlas output for {record.brand} after company analysis completed.")
    return combined_dir, manifest_path if manifest_path.exists() else None


def _ensure_company_analyses(
    *,
    records: tuple[_CompanyRecord, ...],
    sec_user_agent: str,
    workspace_root: Path,
    python_executable: str,
    execution_mode: str,
    log: Callable[[str], None] | None = None,
    on_record_complete: Callable[[_CompanyRecord, Path], None] | None = None,
) -> dict[str, tuple[Path, Path | None]]:
    results: dict[str, tuple[Path, Path | None]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(records)), thread_name_prefix="company-similarity") as executor:
        futures = {}
        for record in records:
            if log is not None:
                log(f"ensuring company analysis for {record.brand}")
            future = executor.submit(
                _ensure_company_analysis,
                record=record,
                sec_user_agent=sec_user_agent,
                workspace_root=workspace_root,
                python_executable=python_executable,
                execution_mode=execution_mode,
            )
            futures[future] = record
        for future in as_completed(futures):
            record = futures[future]
            combined_dir, manifest_path = future.result()
            results[record.slug] = (combined_dir, manifest_path)
            if on_record_complete is not None:
                on_record_complete(record, combined_dir)
            if log is not None:
                log(f"company analysis ready for {record.brand}: {combined_dir}")
    return results


def _preflight_company_similarity_backend(records: tuple[_CompanyRecord, ...]) -> None:
    missing_cached_outputs = [record for record in records if _existing_combined_dir(record) is None]
    if not missing_cached_outputs:
        return
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    brands = ", ".join(record.brand for record in missing_cached_outputs)
    expected_dirs = ", ".join(
        str((record.outdir or (_resolve_brand_workspace_root() / "outputs" / record.slug)) / f"runs_{record.slug}_financial_filings" / f"atlas_{record.slug}_financial_combined")
        for record in missing_cached_outputs
    )
    raise RuntimeError(
        "Company similarity needs either cached combined atlases or an OpenAI-compatible API key for a fresh Democritus build. "
        f"No cached combined atlas was found for: {brands}. "
        "Set OPENAI_API_KEY in the environment, or place the prebuilt atlas directories at: "
        f"{expected_dirs}."
    )


def _render_company_similarity_performance_html(payload: dict[str, object]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    timing = dict(payload.get("timing") or {})
    stage_rows = []
    for stage in list(payload.get("stages") or []):
        if not isinstance(stage, dict):
            continue
        stage_rows.append(
            "<tr>"
            f"<td>{esc(stage.get('label') or stage.get('stage_key') or '')}</td>"
            f"<td>{esc(stage.get('status') or '')}</td>"
            f"<td>{esc(stage.get('duration_human') or '')}</td>"
            f"<td>{esc(stage.get('started_at_local') or '')}</td>"
            f"<td>{esc(stage.get('ended_at_local') or '')}</td>"
            "</tr>"
        )
    slowest_rows = []
    for stage in list(payload.get("slowest_stages") or []):
        if not isinstance(stage, dict):
            continue
        slowest_rows.append(
            "<tr>"
            f"<td>{esc(stage.get('label') or stage.get('stage_key') or '')}</td>"
            f"<td>{esc(stage.get('duration_human') or '')}</td>"
            f"<td>{esc(stage.get('duration_seconds') or '')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Company Similarity Performance</title>
    <style>
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        background: #f7f1e5;
        color: #1f2320;
      }}
      main {{
        max-width: 1080px;
        margin: 36px auto;
        padding: 0 18px 40px;
      }}
      .card {{
        background: rgba(255, 252, 246, 0.96);
        border: 1px solid #d5c8af;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 18px 48px rgba(42, 31, 12, 0.10);
        margin-bottom: 18px;
      }}
      .eyebrow {{
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: #9a5a12;
      }}
      h1, h2 {{
        margin: 0 0 12px 0;
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .metric {{
        border: 1px solid #d5c8af;
        border-radius: 18px;
        padding: 14px;
        background: #fffaf0;
      }}
      .metric span {{
        display: block;
        color: #5d685f;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .metric strong {{
        display: block;
        margin-top: 6px;
        font-size: 22px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
      }}
      th, td {{
        border: 1px solid #e5dcc9;
        padding: 8px 10px;
        text-align: left;
        font-size: 14px;
      }}
      th {{
        background: #f8f1e4;
      }}
      .muted {{
        color: #5d685f;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Company Similarity</p>
        <h1>Performance Telemetry</h1>
        <p class="muted">{esc(payload.get("note") or "")}</p>
        <div class="metrics">
          <div class="metric"><span>Status</span><strong>{esc(payload.get("status") or "")}</strong></div>
          <div class="metric"><span>Elapsed</span><strong>{esc(timing.get("elapsed_human") or "")}</strong></div>
          <div class="metric"><span>Current Stage</span><strong>{esc(timing.get("current_stage") or "warming up")}</strong></div>
          <div class="metric"><span>Completed Work</span><strong>{esc(timing.get("completed_work_human") or "")}</strong></div>
          <div class="metric"><span>Observed Parallelism</span><strong>{esc(timing.get("observed_parallelism") or 1.0)}</strong></div>
          <div class="metric"><span>Peak Parallelism</span><strong>{esc(timing.get("peak_parallelism") or 1.0)}</strong></div>
          <div class="metric"><span>ETA</span><strong>{esc(timing.get("eta_human") or "")}</strong></div>
        </div>
      </section>
      <section class="card">
        <p class="eyebrow">Stage Timing</p>
        <table>
          <thead><tr><th>Stage</th><th>Status</th><th>Duration</th><th>Started</th><th>Ended</th></tr></thead>
          <tbody>{''.join(stage_rows) if stage_rows else "<tr><td colspan='5'>No stage data yet.</td></tr>"}</tbody>
        </table>
      </section>
      <section class="card">
        <p class="eyebrow">Slowest Stages</p>
        <table>
          <thead><tr><th>Stage</th><th>Duration</th><th>Seconds</th></tr></thead>
          <tbody>{''.join(slowest_rows) if slowest_rows else "<tr><td colspan='3'>No completed stages yet.</td></tr>"}</tbody>
        </table>
      </section>
    </main>
  </body>
</html>
"""


def _write_html_report(
    *,
    output_path: Path,
    analysis_dir: Path,
    plan: CompanySimilarityQueryPlan,
    summary_markdown: str,
    manifest: dict[str, object],
    performance_payload: dict[str, object] | None = None,
) -> Path:
    artifact_path = output_path
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(plan.query, route_name="company_similarity"),
    )
    dashboard_png = analysis_dir / "cross_company_functors_dashboard.png"
    summary_html = html.escape(summary_markdown)
    relative_analysis_dir = analysis_dir.relative_to(artifact_path.parent)
    image_markup = (
        f'<figure><img src="{html.escape((relative_analysis_dir / "cross_company_functors_dashboard.png").as_posix())}" alt="Cross-company similarity dashboard" /></figure>'
        if dashboard_png.exists()
        else "<p class=\"muted\">Visualization image not available yet.</p>"
    )
    mean_cosine = manifest.get("mean_yearly_cosine_similarity", "n/a")
    mean_js = manifest.get("mean_yearly_js_divergence", "n/a")
    mean_defect = manifest.get("mean_relative_naturality_defect", "n/a")
    performance_markup = ""
    if performance_payload:
        timing = dict(performance_payload.get("timing") or {})
        performance_markup = f"""
      <section class="card">
        <p class="eyebrow">Performance</p>
        <div class="metrics">
          <div class="metric"><span class="muted">Elapsed</span><strong>{html.escape(str(timing.get("elapsed_human") or "n/a"))}</strong></div>
          <div class="metric"><span class="muted">Completed work</span><strong>{html.escape(str(timing.get("completed_work_human") or "n/a"))}</strong></div>
          <div class="metric"><span class="muted">Observed parallelism</span><strong>{html.escape(str(timing.get("observed_parallelism") or 1.0))}</strong></div>
          <div class="metric"><span class="muted">ETA</span><strong>{html.escape(str(timing.get("eta_human") or "n/a"))}</strong></div>
        </div>
        <p class="muted">
          Performance details:
          <a href="company_similarity_performance.html">performance dashboard</a>,
          <a href="company_similarity_telemetry.json">telemetry json</a>.
        </p>
      </section>
"""
    artifact_path.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(plan.company_a)} vs {html.escape(plan.company_b)} similarity</title>
    <style>
      :root {{
        --ink: #1f2320;
        --muted: #5d685f;
        --paper: #f6f1e6;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #9a5a12;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top, rgba(154, 90, 18, 0.10), transparent 26%),
          linear-gradient(180deg, #fcf8f1 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 1120px;
        margin: 40px auto;
        padding: 0 18px 48px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 18px 48px rgba(42, 31, 12, 0.10);
        margin-bottom: 20px;
      }}
      .eyebrow {{
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--accent);
      }}
      h1 {{
        margin: 0 0 12px 0;
        font-size: clamp(30px, 4vw, 50px);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .metric {{
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px;
      }}
      .metric strong {{
        display: block;
        font-size: 24px;
        margin-top: 4px;
      }}
      .muted {{
        color: var(--muted);
      }}
      .textbook-list {{
        padding-left: 20px;
        display: grid;
        gap: 10px;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
      img {{
        width: 100%;
        border-radius: 18px;
        border: 1px solid var(--line);
      }}
      pre {{
        white-space: pre-wrap;
        background: #fbf7ef;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px;
        color: var(--muted);
        line-height: 1.5;
      }}
      a {{
        color: #7d4306;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Company Similarity</p>
        <h1>{html.escape(plan.company_a)} vs {html.escape(plan.company_b)}</h1>
        <p class="muted">CLIFF resolved the query as a cross-company temporal diffusion comparison over yearly 10-K causal atlases.</p>
        <div class="metrics">
          <div class="metric"><span class="muted">Mean yearly cosine</span><strong>{html.escape(str(mean_cosine))}</strong></div>
          <div class="metric"><span class="muted">Mean yearly JS divergence</span><strong>{html.escape(str(mean_js))}</strong></div>
          <div class="metric"><span class="muted">Mean relative naturality defect</span><strong>{html.escape(str(mean_defect))}</strong></div>
        </div>
      </section>
      <section class="card">
        {textbook_html}
      </section>
      <section class="card">
        <p class="eyebrow">Dashboard</p>
        {image_markup}
      </section>
      {performance_markup}
      <section class="card">
        <p class="eyebrow">Summary</p>
        <pre>{summary_html}</pre>
        <p class="muted">
          Raw outputs:
          <a href="{html.escape((relative_analysis_dir / 'cross_company_functors_summary.md').as_posix())}">summary markdown</a>,
          <a href="{html.escape((relative_analysis_dir / 'cross_company_year_metrics.csv').as_posix())}">year metrics</a>,
          <a href="{html.escape((relative_analysis_dir / 'cross_company_top_shared_edges.csv').as_posix())}">shared edges</a>.
        </p>
      </section>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return artifact_path


class CompanySimilarityAgenticRunner:
    _telemetry_heartbeat_interval_seconds = 2.0

    def __init__(self, query: str, outdir: Path, *, sec_user_agent: str = "", execution_mode: str = "quick") -> None:
        self.query = " ".join(query.split())
        self.outdir = outdir.resolve()
        self.sec_user_agent = sec_user_agent
        self.execution_mode = "deep" if str(execution_mode).strip().lower() == "deep" else "quick"
        if not self.query:
            raise ValueError("A non-empty company similarity query is required.")

    def _log(self, message: str) -> None:
        print(f"[company_similarity] {message}", flush=True)

    def _build_telemetry_payload(
        self,
        *,
        plan: CompanySimilarityQueryPlan,
        started_at: float,
        stage_state: dict[str, dict[str, object]],
        status: str,
        note: str,
    ) -> dict[str, object]:
        now = time.time()
        completed_stages = []
        active_stages = []
        pending_stages = []
        for stage_key, stage in stage_state.items():
            started = float(stage.get("started_at_epoch") or 0.0)
            ended = float(stage.get("ended_at_epoch") or 0.0)
            duration_seconds = max((ended or now) - started, 0.0) if started else 0.0
            stage_payload = {
                "stage_key": stage_key,
                "label": stage.get("label") or stage_key,
                "status": stage.get("status") or "pending",
                "started_at_epoch": round(started, 6) if started else 0.0,
                "ended_at_epoch": round(ended, 6) if ended else 0.0,
                "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)) if started else "",
                "ended_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended)) if ended else "",
                "duration_seconds": round(duration_seconds, 3),
                "duration_human": _format_duration(duration_seconds),
            }
            if stage_payload["status"] == "complete":
                completed_stages.append(stage_payload)
            elif stage_payload["status"] == "active":
                active_stages.append(stage_payload)
            else:
                pending_stages.append(stage_payload)
        slowest_stages = sorted(
            completed_stages,
            key=lambda item: float(item.get("duration_seconds") or 0.0),
            reverse=True,
        )[:5]
        remaining_seconds = 0.0
        for stage in active_stages:
            remaining_seconds += max(float(stage.get("duration_seconds") or 0.0) * 0.6, 30.0)
        for stage in pending_stages:
            label = str(stage.get("label") or "").lower()
            if "build" in label:
                remaining_seconds += 8.0 * 60.0
            elif "functor" in label:
                remaining_seconds += 4.0 * 60.0
            elif "visual" in label:
                remaining_seconds += 90.0
            else:
                remaining_seconds += 30.0
        completed_work_seconds = sum(float(stage.get("duration_seconds") or 0.0) for stage in completed_stages)
        active_work_seconds = sum(float(stage.get("duration_seconds") or 0.0) for stage in active_stages)
        elapsed_seconds = max(now - started_at, 0.0)
        observed_work_seconds = completed_work_seconds + active_work_seconds
        observed_parallelism = 1.0
        if elapsed_seconds > 0:
            observed_parallelism = max(
                1.0,
                min(
                    _company_similarity_parallelism_capacity(stage_state),
                    observed_work_seconds / elapsed_seconds,
                ),
            )
        current_stage_label = _company_similarity_current_stage_label(
            active_stages,
            pending_stages,
            status=status,
        )
        peak_parallelism = max(
            observed_parallelism,
            _company_similarity_peak_parallelism(stage_state, now=now),
        )
        return {
            "query_plan": asdict(plan),
            "execution_mode": self.execution_mode,
            "status": status,
            "note": note,
            "started_at_epoch": round(started_at, 6),
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
            "updated_at_epoch": round(now, 6),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_human": _format_duration(elapsed_seconds),
            "timing": {
                "elapsed_seconds": round(elapsed_seconds, 3),
                "elapsed_human": _format_duration(elapsed_seconds),
                "completed_work_seconds": round(completed_work_seconds, 3),
                "completed_work_human": _format_duration(completed_work_seconds),
                "observed_work_seconds": round(observed_work_seconds, 3),
                "observed_work_human": _format_duration(observed_work_seconds),
                "observed_parallelism": round(observed_parallelism, 3),
                "peak_parallelism": round(peak_parallelism, 3),
                "current_stage": current_stage_label,
                "eta_seconds": round(remaining_seconds, 3),
                "eta_human": _format_duration(remaining_seconds) if remaining_seconds > 0 else "complete",
                "eta_ready": bool(completed_stages or active_stages),
            },
            "stages": [
                *completed_stages,
                *active_stages,
                *pending_stages,
            ],
            "slowest_stages": slowest_stages,
        }

    def _write_telemetry(
        self,
        *,
        telemetry_path: Path,
        performance_dashboard_path: Path,
        plan: CompanySimilarityQueryPlan,
        started_at: float,
        stage_state: dict[str, dict[str, object]],
        status: str,
        note: str,
    ) -> dict[str, object]:
        payload = self._build_telemetry_payload(
            plan=plan,
            started_at=started_at,
            stage_state=stage_state,
            status=status,
            note=note,
        )
        telemetry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        performance_dashboard_path.write_text(
            _render_company_similarity_performance_html(payload),
            encoding="utf-8",
        )
        return payload

    def _start_telemetry_heartbeat(
        self,
        *,
        telemetry_path: Path,
        performance_dashboard_path: Path,
        plan: CompanySimilarityQueryPlan,
        started_at: float,
        stage_state: dict[str, dict[str, object]],
        heartbeat_state: dict[str, str],
    ) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()

        def _pump() -> None:
            interval = max(float(self._telemetry_heartbeat_interval_seconds), 0.1)
            while not stop_event.wait(interval):
                try:
                    self._write_telemetry(
                        telemetry_path=telemetry_path,
                        performance_dashboard_path=performance_dashboard_path,
                        plan=plan,
                        started_at=started_at,
                        stage_state=stage_state,
                        status=heartbeat_state.get("status", "running"),
                        note=heartbeat_state.get("note", ""),
                    )
                except Exception:
                    # Telemetry is diagnostic; keep the main run alive if a refresh fails.
                    pass

        thread = threading.Thread(
            target=_pump,
            name="company-similarity-telemetry-heartbeat",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def run(self) -> CompanySimilarityRunResult:
        self.outdir.mkdir(parents=True, exist_ok=True)
        brand_root = _resolve_brand_workspace_root()
        workspace_root = repo_root().parents[0]
        pipeline_python = _select_python_for_brand_pipeline()
        plan = interpret_company_similarity_query(self.query)
        mode_profile = _company_similarity_mode_profile(self.execution_mode)
        started_at = time.time()
        telemetry_path = self.outdir / "company_similarity_telemetry.json"
        performance_dashboard_path = self.outdir / "company_similarity_performance.html"
        stage_state: dict[str, dict[str, object]] = {
            "query": {"label": "Resolve query", "status": "complete", "started_at_epoch": started_at, "ended_at_epoch": started_at},
            "company_a_analysis": {"label": f"{plan.company_a} build", "status": "pending", "started_at_epoch": 0.0, "ended_at_epoch": 0.0},
            "company_b_analysis": {"label": f"{plan.company_b} build", "status": "pending", "started_at_epoch": 0.0, "ended_at_epoch": 0.0},
            "functor_analysis": {"label": "Cross-company functor comparison", "status": "pending", "started_at_epoch": 0.0, "ended_at_epoch": 0.0},
            "visualization": {"label": "Visualization", "status": "pending", "started_at_epoch": 0.0, "ended_at_epoch": 0.0},
        }
        heartbeat_state = {
            "status": "running",
            "note": (
                "CLIFF is resolving the two company branches and preparing the cross-company comparison "
                f"in {self.execution_mode} mode."
            ),
        }
        self._log(f"resolved query to {plan.company_a} vs {plan.company_b}")
        self._write_telemetry(
            telemetry_path=telemetry_path,
            performance_dashboard_path=performance_dashboard_path,
            plan=plan,
            started_at=started_at,
            stage_state=stage_state,
            status=heartbeat_state["status"],
            note=heartbeat_state["note"],
        )
        heartbeat_stop, heartbeat_thread = self._start_telemetry_heartbeat(
            telemetry_path=telemetry_path,
            performance_dashboard_path=performance_dashboard_path,
            plan=plan,
            started_at=started_at,
            stage_state=stage_state,
            heartbeat_state=heartbeat_state,
        )
        try:
            record_a = _find_company_record(plan, brand_root, plan.company_a_slug)
            record_b = _find_company_record(plan, brand_root, plan.company_b_slug)
            _preflight_company_similarity_backend((record_a, record_b))
            self._log(f"starting parallel company analysis for {record_a.brand} and {record_b.brand}")
            stage_state["company_a_analysis"]["status"] = "active"
            stage_state["company_a_analysis"]["started_at_epoch"] = time.time()
            stage_state["company_b_analysis"]["status"] = "active"
            stage_state["company_b_analysis"]["started_at_epoch"] = time.time()
            heartbeat_state["note"] = f"Running {record_a.brand} and {record_b.brand} analysis in parallel."
            self._write_telemetry(
                telemetry_path=telemetry_path,
                performance_dashboard_path=performance_dashboard_path,
                plan=plan,
                started_at=started_at,
                stage_state=stage_state,
                status=heartbeat_state["status"],
                note=heartbeat_state["note"],
            )
            analysis_results = _ensure_company_analyses(
                records=(record_a, record_b),
                sec_user_agent=self.sec_user_agent,
                workspace_root=workspace_root,
                python_executable=pipeline_python,
                execution_mode=self.execution_mode,
                log=self._log,
                on_record_complete=lambda record, combined_dir: (
                    stage_state.__setitem__(
                        "company_a_analysis" if record.slug == record_a.slug else "company_b_analysis",
                        {
                            **stage_state["company_a_analysis" if record.slug == record_a.slug else "company_b_analysis"],
                            "status": "complete",
                            "ended_at_epoch": time.time(),
                        },
                    ),
                    heartbeat_state.__setitem__(
                        "note",
                        f"{record.brand} analysis finished; waiting for the remaining branch and comparison steps.",
                    ),
                    self._write_telemetry(
                        telemetry_path=telemetry_path,
                        performance_dashboard_path=performance_dashboard_path,
                        plan=plan,
                        started_at=started_at,
                        stage_state=stage_state,
                        status=heartbeat_state["status"],
                        note=heartbeat_state["note"],
                    ),
                ),
            )
            combined_a, manifest_a = analysis_results[record_a.slug]
            combined_b, manifest_b = analysis_results[record_b.slug]

            analysis_dir = self.outdir / f"{plan.company_a_slug}_vs_{plan.company_b_slug}_functors"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.outdir / "company_similarity_summary.json"
            self._log(f"running cross-company functor analysis in {analysis_dir}")
            stage_state["functor_analysis"]["status"] = "active"
            stage_state["functor_analysis"]["started_at_epoch"] = time.time()
            heartbeat_state["note"] = "Both company branches are ready; running the cross-company functor comparison."
            self._write_telemetry(
                telemetry_path=telemetry_path,
                performance_dashboard_path=performance_dashboard_path,
                plan=plan,
                started_at=started_at,
                stage_state=stage_state,
                status=heartbeat_state["status"],
                note=heartbeat_state["note"],
            )

            _run_command(
                [
                    pipeline_python,
                    "-m",
                    "brand_democritus_block_denoise.cross_company_functors",
                    "--company-a-dir",
                    str(combined_a),
                    "--company-b-dir",
                    str(combined_b),
                    "--company-a-name",
                    plan.company_a_slug,
                    "--company-b-name",
                    plan.company_b_slug,
                    "--outdir",
                    str(analysis_dir),
                    "--min-year",
                    str(mode_profile["year_start"]),
                    "--max-year",
                    str(mode_profile["year_end"]),
                ],
                cwd=workspace_root,
            )
            self._log("cross-company functor analysis completed")
            stage_state["functor_analysis"]["status"] = "complete"
            stage_state["functor_analysis"]["ended_at_epoch"] = time.time()
            if int(mode_profile.get("skip_visualization", 0)):
                stage_state["visualization"]["status"] = "complete"
                stage_state["visualization"]["started_at_epoch"] = time.time()
                stage_state["visualization"]["ended_at_epoch"] = stage_state["visualization"]["started_at_epoch"]
                self._log("quick mode: skipping cross-company visualization subprocess")
            else:
                stage_state["visualization"]["status"] = "active"
                stage_state["visualization"]["started_at_epoch"] = time.time()
                heartbeat_state["note"] = "Functor comparison is complete; rendering the visualization and summary outputs."
                self._write_telemetry(
                    telemetry_path=telemetry_path,
                    performance_dashboard_path=performance_dashboard_path,
                    plan=plan,
                    started_at=started_at,
                    stage_state=stage_state,
                    status=heartbeat_state["status"],
                    note=heartbeat_state["note"],
                )
                _run_command(
                    [
                        pipeline_python,
                        "-m",
                        "brand_democritus_block_denoise.visualize_cross_company_functors",
                        "--analysis-dir",
                        str(analysis_dir),
                        "--outdir",
                        str(analysis_dir),
                    ],
                    cwd=workspace_root,
                )
                self._log("cross-company visualization completed")
                stage_state["visualization"]["status"] = "complete"
                stage_state["visualization"]["ended_at_epoch"] = time.time()

            manifest = _safe_read_json(analysis_dir / "cross_company_functors_manifest.json")
            year_metrics_path = analysis_dir / "cross_company_year_metrics.csv"
            naturality_path = analysis_dir / "naturality_defects.csv"
            if year_metrics_path.exists():
                with year_metrics_path.open("r", encoding="utf-8", newline="") as handle:
                    year_rows = [dict(row) for row in csv.DictReader(handle)]
                cosine_values = [float(row["cosine_similarity"]) for row in year_rows if str(row.get("cosine_similarity", "")).strip()]
                js_values = [float(row["js_divergence"]) for row in year_rows if str(row.get("js_divergence", "")).strip()]
                manifest["mean_yearly_cosine_similarity"] = (
                    round(sum(cosine_values) / len(cosine_values), 4) if cosine_values else math.nan
                )
                manifest["mean_yearly_js_divergence"] = (
                    round(sum(js_values) / len(js_values), 4) if js_values else math.nan
                )
            if naturality_path.exists():
                with naturality_path.open("r", encoding="utf-8", newline="") as handle:
                    defect_rows = [dict(row) for row in csv.DictReader(handle)]
                defect_values = [
                    float(row["naturality_defect_rel"])
                    for row in defect_rows
                    if str(row.get("naturality_defect_rel", "")).strip()
                ]
                manifest["mean_relative_naturality_defect"] = (
                    round(sum(defect_values) / len(defect_values), 4) if defect_values else math.nan
                )
            summary_markdown = (analysis_dir / "cross_company_functors_summary.md").read_text(encoding="utf-8")
            heartbeat_state["status"] = "complete"
            heartbeat_state["note"] = "The company similarity run completed and its performance telemetry has been finalized."
            performance_payload = self._write_telemetry(
                telemetry_path=telemetry_path,
                performance_dashboard_path=performance_dashboard_path,
                plan=plan,
                started_at=started_at,
                stage_state=stage_state,
                status=heartbeat_state["status"],
                note=heartbeat_state["note"],
            )
            artifact_path = _write_html_report(
                output_path=self.outdir / "company_similarity_dashboard.html",
                analysis_dir=analysis_dir,
                plan=plan,
                summary_markdown=summary_markdown,
                manifest=manifest,
                performance_payload=performance_payload,
            )
            payload = {
                "query_plan": asdict(plan),
                "execution_mode": self.execution_mode,
                "year_window": {
                    "start": mode_profile["year_start"],
                    "end": mode_profile["year_end"],
                },
                "analysis_dir": str(analysis_dir),
                "artifact_path": str(artifact_path),
                "company_a_manifest_path": str(manifest_a) if manifest_a else None,
                "company_b_manifest_path": str(manifest_b) if manifest_b else None,
                "cross_company_manifest_path": str(analysis_dir / "cross_company_functors_manifest.json"),
                "summary_markdown_path": str(analysis_dir / "cross_company_functors_summary.md"),
                "dashboard_png_path": str(analysis_dir / "cross_company_functors_dashboard.png"),
                "performance_dashboard_path": str(performance_dashboard_path),
                "telemetry_path": str(telemetry_path),
            }
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._log(f"company similarity dashboard ready: {artifact_path}")
            return CompanySimilarityRunResult(
                query_plan=plan,
                route_outdir=self.outdir,
                analysis_dir=analysis_dir,
                summary_path=summary_path,
                artifact_path=artifact_path,
                company_a_manifest_path=manifest_a,
                company_b_manifest_path=manifest_b,
            )
        except Exception as exc:
            heartbeat_state["status"] = "failed"
            heartbeat_state["note"] = f"Company similarity run failed: {exc}"
            self._write_telemetry(
                telemetry_path=telemetry_path,
                performance_dashboard_path=performance_dashboard_path,
                plan=plan,
                started_at=started_at,
                stage_state=stage_state,
                status=heartbeat_state["status"],
                note=heartbeat_state["note"],
            )
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
