"""Helpers for aggregating CLIFF LLM usage records."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from typing import Mapping


_WRITE_LOCK = threading.Lock()
_LLM_TOKEN_BUDGET_ENV = "CLIFF_LLM_TOKEN_BUDGET"
_PROMPT_TOKEN_ESTIMATE_DIVISOR = 3
_PROMPT_TOKEN_ESTIMATE_OVERHEAD = 32


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


def llm_token_budget_from_env() -> int | None:
    raw_budget = str(os.getenv(_LLM_TOKEN_BUDGET_ENV) or "").strip()
    if not raw_budget:
        return None
    try:
        budget = int(raw_budget)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


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


class LLMTokenBudgetExceededError(RuntimeError):
    """Raised when a CLIFF run exhausts its configured LLM token budget."""

    def __init__(
        self,
        *,
        budget_tokens: int,
        spent_tokens: int,
        requested_completion_tokens: int | None = None,
        estimated_prompt_tokens: int | None = None,
    ) -> None:
        remaining_tokens = max(0, budget_tokens - spent_tokens)
        detail = f"CLIFF exhausted its LLM token budget ({spent_tokens:,} used of {budget_tokens:,})."
        if estimated_prompt_tokens is not None:
            detail += f" Estimated prompt cost: {estimated_prompt_tokens:,} tokens."
        if requested_completion_tokens is not None:
            detail += f" Requested up to {requested_completion_tokens:,} more completion tokens with only {remaining_tokens:,} remaining."
        else:
            detail += f" Remaining budget: {remaining_tokens:,} tokens."
        super().__init__(detail)
        self.budget_tokens = int(budget_tokens)
        self.spent_tokens = int(spent_tokens)
        self.remaining_tokens = remaining_tokens
        self.requested_completion_tokens = requested_completion_tokens
        self.estimated_prompt_tokens = estimated_prompt_tokens


def _resolved_usage_path(path: Path | str | None) -> Path | None:
    if path is None:
        return llm_usage_path_from_env()
    return Path(path).expanduser().resolve()


def llm_token_budget_status(
    path: Path | str | None = None,
    *,
    budget_tokens: int | None = None,
) -> dict[str, int | bool]:
    resolved_path = _resolved_usage_path(path)
    resolved_budget = int(budget_tokens) if budget_tokens is not None else llm_token_budget_from_env()
    if resolved_budget is None or resolved_budget <= 0:
        return {}
    spent_tokens = 0
    if resolved_path is not None and resolved_path.exists():
        spent_tokens = int(summarize_llm_usage(resolved_path).get("total_tokens") or 0)
    remaining_tokens = max(0, int(resolved_budget) - spent_tokens)
    return {
        "budget_tokens": int(resolved_budget),
        "spent_tokens": spent_tokens,
        "remaining_tokens": remaining_tokens,
        "exhausted": spent_tokens >= int(resolved_budget),
        "over_budget": spent_tokens > int(resolved_budget),
    }


def estimate_prompt_tokens(*, prompt_chars: int | None = None, prompt_text: str | None = None) -> int:
    chars = _safe_int(prompt_chars)
    if chars <= 0 and prompt_text:
        chars = len(str(prompt_text))
    if chars <= 0:
        return 0
    return ((chars + _PROMPT_TOKEN_ESTIMATE_DIVISOR - 1) // _PROMPT_TOKEN_ESTIMATE_DIVISOR) + _PROMPT_TOKEN_ESTIMATE_OVERHEAD


def enforce_llm_token_budget(
    path: Path | str | None = None,
    *,
    requested_completion_tokens: int | None = None,
    estimated_prompt_tokens: int | None = None,
    prompt_chars: int | None = None,
    budget_tokens: int | None = None,
) -> dict[str, int | bool]:
    status = llm_token_budget_status(path, budget_tokens=budget_tokens)
    if not status:
        return {}
    remaining_tokens = int(status.get("remaining_tokens") or 0)
    reserved_prompt_tokens = max(
        0,
        int(estimated_prompt_tokens)
        if estimated_prompt_tokens is not None
        else estimate_prompt_tokens(prompt_chars=prompt_chars),
    )
    if remaining_tokens <= 0 or remaining_tokens <= reserved_prompt_tokens:
        raise LLMTokenBudgetExceededError(
            budget_tokens=int(status["budget_tokens"]),
            spent_tokens=int(status["spent_tokens"]),
            requested_completion_tokens=requested_completion_tokens,
            estimated_prompt_tokens=reserved_prompt_tokens or None,
        )
    allowed_completion_tokens = max(0, remaining_tokens - reserved_prompt_tokens)
    if requested_completion_tokens is not None and requested_completion_tokens > 0:
        allowed_completion_tokens = min(int(requested_completion_tokens), allowed_completion_tokens)
    if requested_completion_tokens is not None and allowed_completion_tokens <= 0:
        raise LLMTokenBudgetExceededError(
            budget_tokens=int(status["budget_tokens"]),
            spent_tokens=int(status["spent_tokens"]),
            requested_completion_tokens=requested_completion_tokens,
            estimated_prompt_tokens=reserved_prompt_tokens or None,
        )
    return {
        **status,
        "estimated_prompt_tokens": reserved_prompt_tokens,
        "allowed_completion_tokens": allowed_completion_tokens,
    }


def raise_if_over_llm_token_budget(
    path: Path | str | None = None,
    *,
    budget_tokens: int | None = None,
) -> None:
    status = llm_token_budget_status(path, budget_tokens=budget_tokens)
    if status and bool(status.get("over_budget")):
        raise LLMTokenBudgetExceededError(
            budget_tokens=int(status["budget_tokens"]),
            spent_tokens=int(status["spent_tokens"]),
        )


@contextmanager
def scoped_llm_token_budget_env(budget_tokens: int | None):
    previous = os.environ.get(_LLM_TOKEN_BUDGET_ENV)
    if budget_tokens is None or int(budget_tokens) <= 0:
        os.environ.pop(_LLM_TOKEN_BUDGET_ENV, None)
    else:
        os.environ[_LLM_TOKEN_BUDGET_ENV] = str(int(budget_tokens))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_LLM_TOKEN_BUDGET_ENV, None)
        else:
            os.environ[_LLM_TOKEN_BUDGET_ENV] = previous


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
