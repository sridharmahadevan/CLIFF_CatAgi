"""Tests for shared LLM usage tracking helpers and client emitters."""

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

from CLIFF_CatAgi.functorflow_v3.llm_usage import append_llm_usage_row, summarize_llm_usage
from Democritus_OpenAI.llms import openai_client as public_openai_client
from FunctorFlow.democritus import DemocritusLLMConfig, OpenAICompatibleDemocritusClient


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


class LLMUsageTests(unittest.TestCase):
    def test_append_and_summarize_llm_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_usage.jsonl"
            append_llm_usage_row(
                log_path,
                usage={
                    "model": "gpt-4.1-mini",
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
                metadata={
                    "route": "democritus",
                    "run_name": "run_alpha",
                    "agent_name": "root_topic_discovery_agent",
                },
            )
            append_llm_usage_row(
                log_path,
                usage={
                    "model": "gpt-4.1",
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                },
                metadata={
                    "route": "democritus",
                    "run_name": "run_alpha",
                    "agent_name": "causal_question_agent",
                },
            )

            summary = summarize_llm_usage(log_path)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["requests_with_usage"], 2)
        self.assertEqual(summary["prompt_tokens"], 200)
        self.assertEqual(summary["completion_tokens"], 50)
        self.assertEqual(summary["total_tokens"], 250)
        self.assertEqual(summary["by_agent"][0]["agent_name"], "root_topic_discovery_agent")
        self.assertEqual(summary["by_model"][0]["model"], "gpt-4.1-mini")

    def test_functorflow_client_writes_usage_rows(self) -> None:
        payload = {
            "model": "gpt-4.1-mini",
            "usage": {"prompt_tokens": 90, "completion_tokens": 15, "total_tokens": 105},
            "choices": [{"message": {"content": "topic alpha"}}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_usage.jsonl"
            client = OpenAICompatibleDemocritusClient(
                DemocritusLLMConfig(model="gpt-4.1-mini"),
                api_key="test-key",
                usage_log_path=log_path,
                usage_metadata={
                    "route": "democritus",
                    "run_name": "run_alpha",
                    "agent_name": "root_topic_discovery_agent",
                },
            )
            with mock.patch(
                "FunctorFlow.democritus.urllib.request.urlopen",
                return_value=_FakeHTTPResponse(payload),
            ):
                response = client.ask("Find root topics")

            summary = summarize_llm_usage(log_path)

        self.assertEqual(response, "topic alpha")
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["total_tokens"], 105)
        self.assertEqual(summary["by_agent"][0]["agent_name"], "root_topic_discovery_agent")

    def test_public_openai_client_writes_usage_rows_via_urllib_fallback(self) -> None:
        payload = {
            "model": "gpt-4.1-mini",
            "usage": {"prompt_tokens": 70, "completion_tokens": 10, "total_tokens": 80},
            "choices": [{"message": {"content": "question alpha"}}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "llm_usage.jsonl"
            client = public_openai_client.OpenAIChatClient(
                api_key="test-key",
                usage_log_path=log_path,
                usage_metadata={
                    "route": "democritus",
                    "run_name": "run_alpha",
                    "agent_name": "causal_question_agent",
                },
            )
            with mock.patch.object(public_openai_client, "requests", None):
                with mock.patch.object(public_openai_client, "urlopen", return_value=_FakeHTTPResponse(payload)):
                    response = client.ask("Generate causal questions")

            summary = summarize_llm_usage(log_path)

        self.assertEqual(response, "question alpha")
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["total_tokens"], 80)
        self.assertEqual(summary["by_agent"][0]["agent_name"], "causal_question_agent")
