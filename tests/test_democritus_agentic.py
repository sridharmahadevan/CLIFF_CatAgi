"""Tests for the agentic Democritus runner."""

from __future__ import annotations

import json
import tempfile
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest import mock

try:
    from functorflow_v3 import democritus_agentic as democritus_agentic_module
    from functorflow_v3 import (
        DemocritusAgenticConfig,
        DemocritusAgenticRunner,
        build_democritus_agentic_workflow,
    )
except ModuleNotFoundError:
    from ..functorflow_v3 import democritus_agentic as democritus_agentic_module
    from ..functorflow_v3 import (
        DemocritusAgenticConfig,
        DemocritusAgenticRunner,
        build_democritus_agentic_workflow,
    )


class DemocritusAgenticTests(unittest.TestCase):
    def test_pipeline_frontiers_reflect_parallel_restructure(self) -> None:
        workflow = build_democritus_agentic_workflow(include_phase2=True)

        frontiers = tuple(tuple(agent.name for agent in frontier) for frontier in workflow.parallel_frontiers())

        self.assertEqual(frontiers[0], ("document_intake_agent",))
        self.assertEqual(frontiers[1], ("root_topic_discovery_agent",))
        self.assertEqual(frontiers[2], ("topic_graph_agent",))
        self.assertEqual(set(frontiers[6]), {"lcm_sweep_agent", "manifold_builder_agent"})
        self.assertEqual(
            set(frontiers[7]),
            {"lcm_scoring_agent", "manifold_visualization_agent", "topos_slice_agent"},
        )
        self.assertEqual(frontiers[-1], ("credibility_bundle_agent",))

    def test_runner_plan_matches_workflow_frontiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=Path(tmpdir),
                    root_topics=("causal mechanisms", "evidence aggregation"),
                    include_phase2=False,
                )
            )
            self.assertEqual(
                runner.plan()[:7],
                (
                    ("document_intake_agent",),
                    ("root_topic_discovery_agent",),
                    ("topic_graph_agent",),
                    ("causal_question_agent",),
                    ("causal_statement_agent",),
                    ("triple_extraction_agent",),
                    ("manifold_builder_agent",),
                ),
            )
            self.assertEqual(set(runner.plan()[7]), {"manifold_visualization_agent", "topos_slice_agent"})

    def test_root_topic_discovery_agent_writes_root_topics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    root_topics=("topic one", "topic two"),
                    include_phase2=False,
                )
            )

            runner.run_agent("document_intake_agent")
            outputs = runner.run_agent("root_topic_discovery_agent")

            topics_path = (outdir / "configs" / "root_topics.txt").resolve()
            self.assertIn(str(topics_path), outputs)
            self.assertEqual(topics_path.read_text(encoding="utf-8"), "topic one\ntopic two\n")

    def test_root_topic_discovery_agent_falls_back_to_heuristic_after_openai_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            input_pdf = outdir / "input.pdf"
            input_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    input_pdf=input_pdf,
                    auto_topics_from_pdf=True,
                    root_topic_strategy="v0_openai",
                    include_phase2=False,
                )
            )

            attempts = {"openai": 0, "heuristic": 0}

            def fail_openai() -> tuple[str, ...]:
                attempts["openai"] += 1
                raise RuntimeError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac")

            def succeed_heuristic() -> tuple[str, ...]:
                attempts["heuristic"] += 1
                return ("resveratrol", "red wine")

            runner._discover_root_topics_v0_openai = fail_openai  # type: ignore[assignment]
            runner._discover_root_topics_heuristic = succeed_heuristic  # type: ignore[assignment]

            with mock.patch.object(democritus_agentic_module.time, "sleep", return_value=None):
                outputs = runner.run_agent("root_topic_discovery_agent")

            topics_path = (outdir / "configs" / "root_topics.txt").resolve()
            log_path = outdir / "agent_logs" / "root_topic_discovery_agent.log"

            self.assertEqual(attempts["openai"], 3)
            self.assertEqual(attempts["heuristic"], 1)
            self.assertIn(str(topics_path), outputs)
            self.assertEqual(topics_path.read_text(encoding="utf-8"), "resveratrol\nred wine\n")
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("attempt 1/3 failed", log_text)
            self.assertIn("attempt 3/3 failed", log_text)
            self.assertIn("Falling back to heuristic topic discovery", log_text)

    def test_dry_run_records_all_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=Path(tmpdir),
                    root_topics=("root topic",),
                    include_phase2=False,
                )
            )
            records = runner.run(dry_run=True)

            self.assertTrue(records)
            self.assertTrue(all(record.status == "planned" for record in records))

    def test_subprocess_failure_includes_log_path_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    root_topics=("root topic",),
                    include_phase2=False,
                )
            )

            with self.assertRaises(RuntimeError) as ctx:
                runner._run_subprocess_agent(
                    "topic_graph_agent",
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('topic graph exploded'); sys.exit(3)",
                    ],
                    cwd=outdir,
                    outputs=(),
                )

            message = str(ctx.exception)
            self.assertIn("status 3", message)
            self.assertIn("topic_graph_agent.log", message)
            self.assertIn("topic graph exploded", message)

    def test_stage_env_forces_python_unbuffered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=Path(tmpdir),
                    root_topics=("root topic",),
                    include_phase2=False,
                )
            )

            env = runner._stage_env()

            self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")

    def test_causal_question_agent_can_shard_within_one_document_and_merge_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    root_topics=("root topic",),
                    include_phase2=False,
                    intra_document_shards=3,
                )
            )
            (outdir / "topic_graph.jsonl").write_text(
                "\n".join(
                    [
                        '{"topic":"root topic","parent":null,"depth":0}',
                        '{"topic":"child one","parent":"root topic","depth":1}',
                        '{"topic":"child two","parent":"root topic","depth":1}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calls: list[tuple[str, list[str]]] = []
            original = runner._run_subprocess_agent

            def fake_run(agent_name, cmd, *, cwd, outputs):
                del cwd
                calls.append((agent_name, cmd))
                shard_output = Path(outputs[0])
                shard_output.parent.mkdir(parents=True, exist_ok=True)
                shard_output.write_text(
                    json.dumps({"agent_name": agent_name, "cmd": cmd}) + "\n",
                    encoding="utf-8",
                )
                return tuple(str(path) for path in outputs)

            runner._run_subprocess_agent = fake_run  # type: ignore[assignment]
            try:
                outputs = runner._run_causal_question_agent()
            finally:
                runner._run_subprocess_agent = original  # type: ignore[assignment]

            merged_path = outdir / "causal_questions.jsonl"
            self.assertEqual(Path(outputs[0]).resolve(), merged_path.resolve())
            self.assertEqual(len(calls), 3)
            merged_lines = merged_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(merged_lines), 3)
            self.assertTrue(all("--num-shards" in " ".join(cmd) for _, cmd in calls))

    def test_topic_graph_agent_can_shard_within_one_document_and_merge_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    root_topics=("root topic one", "root topic two"),
                    include_phase2=False,
                    intra_document_shards=2,
                )
            )
            runner.run_agent("root_topic_discovery_agent")

            calls: list[tuple[str, list[str]]] = []
            original = runner._run_subprocess_agent

            def fake_run(agent_name, cmd, *, cwd, outputs):
                del cwd
                calls.append((agent_name, cmd))
                graph_output = Path(outputs[0])
                list_output = Path(outputs[1])
                graph_output.parent.mkdir(parents=True, exist_ok=True)
                list_output.parent.mkdir(parents=True, exist_ok=True)
                shard_index = cmd[cmd.index("--shard-index") + 1]
                topic = f"topic_{shard_index}"
                graph_output.write_text(
                    json.dumps({"topic": topic, "parent": None, "depth": 0}) + "\n",
                    encoding="utf-8",
                )
                list_output.write_text(f"{topic}\t0\n", encoding="utf-8")
                return tuple(str(path) for path in outputs)

            runner._run_subprocess_agent = fake_run  # type: ignore[assignment]
            try:
                outputs = runner._run_topic_graph_agent()
            finally:
                runner._run_subprocess_agent = original  # type: ignore[assignment]

            merged_graph = outdir / "topic_graph.jsonl"
            merged_list = outdir / "topic_list.txt"
            self.assertEqual(Path(outputs[0]).resolve(), merged_graph.resolve())
            self.assertEqual(Path(outputs[1]).resolve(), merged_list.resolve())
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("--num-shards" in " ".join(cmd) for _, cmd in calls))
            merged_topics = {json.loads(line)["topic"] for line in merged_graph.read_text(encoding="utf-8").splitlines()}
            self.assertEqual(merged_topics, {"topic_0", "topic_1"})

    def test_causal_statement_agent_passes_statement_budget_to_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            runner = DemocritusAgenticRunner(
                DemocritusAgenticConfig(
                    outdir=outdir,
                    root_topics=("root topic",),
                    include_phase2=False,
                    intra_document_shards=2,
                    statements_per_question=1,
                )
            )
            (outdir / "causal_questions.jsonl").write_text(
                json.dumps({"topic": "root topic", "path": ["root topic"], "questions": ["What increases warming?"]}) + "\n",
                encoding="utf-8",
            )

            calls: list[tuple[str, list[str]]] = []
            original = runner._run_subprocess_agent

            def fake_run(agent_name, cmd, *, cwd, outputs):
                del cwd
                calls.append((agent_name, cmd))
                output = Path(outputs[0])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "topic": "root topic",
                            "path": ["root topic"],
                            "question": "What increases warming?",
                            "statements": ["Emissions increase warming."],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return tuple(str(path) for path in outputs)

            runner._run_subprocess_agent = fake_run  # type: ignore[assignment]
            try:
                outputs = runner._run_causal_statement_agent()
            finally:
                runner._run_subprocess_agent = original  # type: ignore[assignment]

            merged_path = outdir / "causal_statements.jsonl"
            self.assertEqual(Path(outputs[0]).resolve(), merged_path.resolve())
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("--statements-per-question" in " ".join(cmd) for _, cmd in calls))
            for _, cmd in calls:
                arg_index = cmd.index("--statements-per-question")
                self.assertEqual(cmd[arg_index + 1], "1")

    def test_manifold_visualization_agent_serializes_matplotlib_rendering(self) -> None:
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.use = lambda *args, **kwargs: None
        fake_module = ModuleType("scripts.visualize_manifold")
        events: list[tuple[str, float]] = []
        active = {"count": 0, "max": 0}
        guard = threading.Lock()

        def fake_visualize_from_state(*, state_path: str, out_dir: str, title_prefix: str) -> None:
            del state_path, title_prefix
            with guard:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
                events.append(("start", time.time()))
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "relational_manifold_2d.png").write_bytes(b"fakepng")
            (Path(out_dir) / "relational_manifold_3d.png").write_bytes(b"fakepng")
            time.sleep(0.02)
            with guard:
                active["count"] -= 1
                events.append(("end", time.time()))

        fake_module.visualize_from_state = fake_visualize_from_state
        original_matplotlib = sys.modules.get("matplotlib")
        original_module = sys.modules.get("scripts.visualize_manifold")
        sys.modules["matplotlib"] = fake_matplotlib
        sys.modules["scripts.visualize_manifold"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)
                (outdir / "manifold_state.pkl").write_bytes(b"placeholder")
                runner = DemocritusAgenticRunner(
                    DemocritusAgenticConfig(
                        outdir=outdir,
                        root_topics=("root topic",),
                        include_phase2=False,
                    )
                )

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(runner._run_manifold_visualization_agent)
                        for _ in range(2)
                    ]
                    for future in futures:
                        future.result()

            self.assertEqual(active["max"], 1)
            self.assertEqual(len(events), 4)
        finally:
            if original_matplotlib is None:
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = original_matplotlib
            if original_module is None:
                sys.modules.pop("scripts.visualize_manifold", None)
            else:
                sys.modules["scripts.visualize_manifold"] = original_module


if __name__ == "__main__":
    unittest.main()
