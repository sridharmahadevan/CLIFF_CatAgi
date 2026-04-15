"""Helpers for aggregating CLIFF LLM usage records."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping


_WRITE_LOCK = threading.Lock()


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def llm_usage_path_from_env() -> Path | None:
    raw_path = str(os.getenv("CLIFF_LLM_USAGE_PATH") or "").strip()
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


def llm_usage_metadata_from_env() -> dict[str, object]:
    metadata: dict[str, object] = {}
    for env_name, field_name in (
        ("CLIFF_LLM_USAGE_ROUTE", "route"),
        ("CLIFF_LLM_USAGE_RUN", "run_name"),
        ("CLIFF_LLM_USAGE_AGENT", "agent_name"),
        ("CLIFF_LLM_USAGE_OUTDIR", "outdir"),
    ):
        value = str(os.getenv(env_name) or "").strip()
        if value:
            metadata[field_name] = value
    return metadata


def extract_openai_usage(payload: Mapping[str, object] | None) -> dict[str, object]:
    data = dict(payload or {})
    usage = dict(data.get("usage") or {})
    prompt_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens")) or (prompt_tokens + completion_tokens)
    return {
        "model": str(data.get("model") or "").strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def append_llm_usage_row(
    path: Path | str | None = None,
    *,
    usage: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    resolved_path: Path | None
    if path is None:
        resolved_path = llm_usage_path_from_env()
    else:
        resolved_path = Path(path).expanduser().resolve()
    if resolved_path is None:
        return

    merged_metadata = {**llm_usage_metadata_from_env(), **dict(metadata or {})}
    usage_payload = dict(usage or {})
    prompt_tokens = _safe_int(usage_payload.get("prompt_tokens"))
    completion_tokens = _safe_int(usage_payload.get("completion_tokens"))
    total_tokens = _safe_int(usage_payload.get("total_tokens")) or (prompt_tokens + completion_tokens)
    row = {
        "timestamp_epoch": round(time.time(), 6),
        "route": str(merged_metadata.get("route") or "").strip(),
        "run_name": str(merged_metadata.get("run_name") or "").strip(),
        "agent_name": str(merged_metadata.get("agent_name") or "").strip(),
        "outdir": str(merged_metadata.get("outdir") or "").strip(),
        "provider": str(merged_metadata.get("provider") or "").strip(),
        "client": str(merged_metadata.get("client") or "").strip(),
        "request_kind": str(merged_metadata.get("request_kind") or "").strip(),
        "model": str(usage_payload.get("model") or merged_metadata.get("model") or "").strip(),
        "prompt_chars": _safe_int(merged_metadata.get("prompt_chars")),
        "response_chars": _safe_int(merged_metadata.get("response_chars")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with resolved_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_llm_usage(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "request_count": 0,
            "requests_with_usage": 0,
            "requests_missing_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "avg_total_tokens_per_request": 0.0,
            "by_agent": [],
            "by_model": [],
        }

    request_count = 0
    requests_with_usage = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    by_agent: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {
            "agent_name": "",
            "requests": 0,
            "requests_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    by_model: dict[str, dict[str, int | str]] = defaultdict(
        lambda: {
            "model": "",
            "requests": 0,
            "requests_with_usage": 0,
            "total_tokens": 0,
        }
    )

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = dict(json.loads(stripped))
            except json.JSONDecodeError:
                continue
            request_count += 1
            row_prompt_tokens = _safe_int(row.get("prompt_tokens"))
            row_completion_tokens = _safe_int(row.get("completion_tokens"))
            row_total_tokens = _safe_int(row.get("total_tokens"))
            if row_total_tokens > 0 or row_prompt_tokens > 0 or row_completion_tokens > 0:
                requests_with_usage += 1
            prompt_tokens += row_prompt_tokens
            completion_tokens += row_completion_tokens
            total_tokens += row_total_tokens or (row_prompt_tokens + row_completion_tokens)

            agent_name = str(row.get("agent_name") or "unknown")
            agent_bucket = by_agent[agent_name]
            agent_bucket["agent_name"] = agent_name
            agent_bucket["requests"] = int(agent_bucket["requests"]) + 1
            if row_total_tokens > 0 or row_prompt_tokens > 0 or row_completion_tokens > 0:
                agent_bucket["requests_with_usage"] = int(agent_bucket["requests_with_usage"]) + 1
            agent_bucket["prompt_tokens"] = int(agent_bucket["prompt_tokens"]) + row_prompt_tokens
            agent_bucket["completion_tokens"] = int(agent_bucket["completion_tokens"]) + row_completion_tokens
            agent_bucket["total_tokens"] = int(agent_bucket["total_tokens"]) + (
                row_total_tokens or (row_prompt_tokens + row_completion_tokens)
            )

            model_name = str(row.get("model") or "unknown")
            model_bucket = by_model[model_name]
            model_bucket["model"] = model_name
            model_bucket["requests"] = int(model_bucket["requests"]) + 1
            if row_total_tokens > 0 or row_prompt_tokens > 0 or row_completion_tokens > 0:
                model_bucket["requests_with_usage"] = int(model_bucket["requests_with_usage"]) + 1
            model_bucket["total_tokens"] = int(model_bucket["total_tokens"]) + (
                row_total_tokens or (row_prompt_tokens + row_completion_tokens)
            )

    sorted_agents = sorted(
        by_agent.values(),
        key=lambda item: (
            int(item["total_tokens"]),
            int(item["requests"]),
            str(item["agent_name"]),
        ),
        reverse=True,
    )
    sorted_models = sorted(
        by_model.values(),
        key=lambda item: (
            int(item["total_tokens"]),
            int(item["requests"]),
            str(item["model"]),
        ),
        reverse=True,
    )
    return {
        "available": True,
        "path": str(path),
        "request_count": request_count,
        "requests_with_usage": requests_with_usage,
        "requests_missing_usage": max(request_count - requests_with_usage, 0),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_total_tokens_per_request": round(total_tokens / requests_with_usage, 1) if requests_with_usage else 0.0,
        "by_agent": list(sorted_agents[:8]),
        "by_model": list(sorted_models[:5]),
    }
