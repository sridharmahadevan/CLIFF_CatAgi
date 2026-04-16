"""Regression tests for prompt-aware LLM token budget enforcement."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from CLIFF_CatAgi.functorflow_v3.llm_usage import (
    LLMTokenBudgetExceededError,
    append_llm_usage_row,
    enforce_llm_token_budget,
)
from Democritus_OpenAI.llms import openai_client as public_openai_client


class _Headers:
    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.headers = _Headers()

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class LLMTokenBudgetGuardTests(unittest.TestCase):
    def test_enforce_budget_blocks_when_estimated_prompt_alone_exhausts_remaining_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_usage.jsonl"
            append_llm_usage_row(
                log_path,
                usage={
                    "model": "gpt-4.1-mini",
                    "prompt_tokens": 4900,
                    "completion_tokens": 0,
                    "total_tokens": 4900,
                },
            )
            with self.assertRaises(LLMTokenBudgetExceededError):
                enforce_llm_token_budget(
                    log_path,
                    budget_tokens=5000,
                    requested_completion_tokens=256,
                    prompt_chars=900,
                )

    def test_public_openai_client_reserves_prompt_budget_before_clamping_completion_budget(self) -> None:
        response_payload = {
            "model": "gpt-4.1-mini",
            "usage": {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90},
            "choices": [{"message": {"content": "question alpha"}}],
        }
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            del timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeHTTPResponse(response_payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_usage.jsonl"
            append_llm_usage_row(
                log_path,
                usage={
                    "model": "gpt-4.1-mini",
                    "prompt_tokens": 4700,
                    "completion_tokens": 0,
                    "total_tokens": 4700,
                },
            )
            client = public_openai_client.OpenAIChatClient(
                api_key="test-key",
                max_tokens=400,
                usage_log_path=log_path,
            )
            prompt = "x" * 600
            with mock.patch.dict("os.environ", {"CLIFF_LLM_TOKEN_BUDGET": "5000"}, clear=False):
                with mock.patch.object(public_openai_client, "requests", None):
                    with mock.patch.object(public_openai_client, "urlopen", side_effect=fake_urlopen):
                        response = client.ask(prompt)

        self.assertEqual(response, "question alpha")
        self.assertEqual(captured["payload"]["max_tokens"], 68)


if __name__ == "__main__":
    unittest.main()
