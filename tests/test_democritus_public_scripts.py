"""Regression tests for the public Democritus helper scripts."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType


class DemocritusPublicScriptTests(unittest.TestCase):
    def _with_public_script_module(self, module_name: str):
        candidates = (
            Path(__file__).resolve().parents[2] / "Democritus_OpenAI",
            Path("/Users/sridharmahadevan/Documents/GitHub/Democritus_OpenAI"),
            Path("/Users/sridharmahadevan/Documents/Playground/Democritus_OpenAI"),
        )
        repo_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            added = True
        else:
            added = False

        try:
            fake_tqdm = ModuleType("tqdm")
            fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable
            original_tqdm = sys.modules.get("tqdm")
            sys.modules["tqdm"] = fake_tqdm
            fake_requests = ModuleType("requests")
            fake_requests.post = lambda *args, **kwargs: None
            original_requests = sys.modules.get("requests")
            sys.modules["requests"] = fake_requests
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            return module, repo_root, added, original_tqdm, original_requests
        finally:
            pass

    def _cleanup_public_script_module(
        self,
        *,
        repo_root: Path,
        added: bool,
        original_tqdm,
        original_requests,
    ) -> None:
        if original_tqdm is None:
            sys.modules.pop("tqdm", None)
        else:
            sys.modules["tqdm"] = original_tqdm
        if original_requests is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = original_requests
        if added:
            sys.path.remove(str(repo_root))

    def test_causal_statement_builder_falls_back_to_single_prompt_calls(self) -> None:
        module, repo_root, added, original_tqdm, original_requests = self._with_public_script_module(
            "scripts.causal_statement_builder"
        )
        try:
            class FakeLLM:
                def ask_batch(self, prompts):
                    raise RuntimeError("400 Client Error: Bad Request")

                def ask(self, prompt):
                    if "bad question" in prompt:
                        raise RuntimeError("still bad")
                    return "Rising temperature causes glacier melt."

            original_factory = module.make_llm_client
            module.make_llm_client = lambda **kwargs: FakeLLM()

            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    Path("causal_questions.jsonl").write_text(
                        "\n".join(
                            [
                                json.dumps(
                                    {
                                        "topic": "climate",
                                        "path": ["climate"],
                                        "questions": ["good question", "bad question"],
                                    }
                                )
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    module.main()
                    output = Path("causal_statements.jsonl").read_text(encoding="utf-8").splitlines()
                finally:
                    os.chdir(cwd)
                    module.make_llm_client = original_factory

            self.assertEqual(len(output), 1)
            record = json.loads(output[0])
            self.assertEqual(record["question"], "good question")
            self.assertEqual(record["statements"], ["Rising temperature causes glacier melt."])
        finally:
            self._cleanup_public_script_module(
                repo_root=repo_root,
                added=added,
                original_tqdm=original_tqdm,
                original_requests=original_requests,
            )

    def test_causal_statement_builder_honors_batch_and_token_budget(self) -> None:
        module, repo_root, added, original_tqdm, original_requests = self._with_public_script_module(
            "scripts.causal_statement_builder"
        )
        try:
            captured: dict[str, object] = {}

            class FakeLLM:
                def ask_batch(self, prompts):
                    return ["Warming increases drought risk." for _ in prompts]

                def ask(self, prompt):
                    return "Warming increases drought risk."

            original_factory = module.make_llm_client

            def fake_factory(**kwargs):
                captured.update(kwargs)
                return FakeLLM()

            module.make_llm_client = fake_factory

            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    Path("causal_questions.jsonl").write_text(
                        json.dumps(
                            {
                                "topic": "climate",
                                "path": ["climate"],
                                "questions": ["How does warming affect drought?"],
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    module.main(statements_per_question=1, batch_size=32, max_tokens=72)
                finally:
                    os.chdir(cwd)
                    module.make_llm_client = original_factory

            self.assertEqual(captured["max_tokens"], 72)
            self.assertEqual(captured["max_batch_size"], 32)
        finally:
            self._cleanup_public_script_module(
                repo_root=repo_root,
                added=added,
                original_tqdm=original_tqdm,
                original_requests=original_requests,
            )

    def test_causal_question_builder_falls_back_to_single_prompt_calls(self) -> None:
        module, repo_root, added, original_tqdm, original_requests = self._with_public_script_module(
            "scripts.causal_question_builder"
        )
        try:
            class FakeLLM:
                def ask_batch(self, prompts):
                    raise RuntimeError("400 Client Error: Bad Request")

                def ask(self, prompt):
                    if "bad topic" in prompt:
                        raise RuntimeError("still bad")
                    return "How does warming affect coral bleaching?\nWhat increases climate migration?"

            original_factory = module.make_llm_client
            module.make_llm_client = lambda **kwargs: FakeLLM()

            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    Path("topic_graph.jsonl").write_text(
                        "\n".join(
                            [
                                json.dumps({"topic": "good topic", "parent": None, "depth": 0}),
                                json.dumps({"topic": "bad topic", "parent": None, "depth": 0}),
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    module.main(topic_graph_path="topic_graph.jsonl")
                    output = Path("causal_questions.jsonl").read_text(encoding="utf-8").splitlines()
                finally:
                    os.chdir(cwd)
                    module.make_llm_client = original_factory

            self.assertEqual(len(output), 1)
            record = json.loads(output[0])
            self.assertEqual(record["topic"], "good topic")
            self.assertEqual(
                record["questions"],
                ["How does warming affect coral bleaching?", "What increases climate migration?"],
            )
        finally:
            self._cleanup_public_script_module(
                repo_root=repo_root,
                added=added,
                original_tqdm=original_tqdm,
                original_requests=original_requests,
            )

    def test_topic_graph_builder_falls_back_to_single_prompt_calls(self) -> None:
        module, repo_root, added, original_tqdm, original_requests = self._with_public_script_module(
            "scripts.topic_graph_builder"
        )
        try:
            class FakeLLM:
                def ask_batch(self, prompts):
                    raise RuntimeError("400 Client Error: Bad Request")

                def ask(self, prompt):
                    if "bad root" in prompt:
                        raise RuntimeError("still bad")
                    return "ice sheet loss\nsea level rise"

            original_factory = module.make_llm_client
            module.make_llm_client = lambda **kwargs: FakeLLM()

            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    Path("root_topics.txt").write_text("good root\nbad root\n", encoding="utf-8")
                    module.main(
                        topics_file="root_topics.txt",
                        depth_limit=1,
                        max_total_topics=10,
                        topic_graph_path="topic_graph.jsonl",
                        topic_list_path="topic_list.txt",
                    )
                    output = Path("topic_graph.jsonl").read_text(encoding="utf-8").splitlines()
                finally:
                    os.chdir(cwd)
                    module.make_llm_client = original_factory

            records = [json.loads(line) for line in output]
            topics = {record["topic"] for record in records}
            self.assertIn("good root", topics)
            self.assertIn("bad root", topics)
            self.assertIn("ice sheet loss", topics)
            self.assertIn("sea level rise", topics)
        finally:
            self._cleanup_public_script_module(
                repo_root=repo_root,
                added=added,
                original_tqdm=original_tqdm,
                original_requests=original_requests,
            )

    def test_make_credibility_bundle_writes_tier_only_executive_summary(self) -> None:
        try:
            import pandas  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("pandas is required for make_credibility_bundle")
        original_networkx = sys.modules.get("networkx")
        original_matplotlib = sys.modules.get("matplotlib")
        original_pyplot = sys.modules.get("matplotlib.pyplot")
        fake_networkx = ModuleType("networkx")
        fake_matplotlib = ModuleType("matplotlib")
        fake_pyplot = ModuleType("matplotlib.pyplot")
        sys.modules["networkx"] = fake_networkx
        sys.modules["matplotlib"] = fake_matplotlib
        sys.modules["matplotlib.pyplot"] = fake_pyplot
        module, repo_root, added, original_tqdm, original_requests = self._with_public_script_module(
            "scripts.make_credibility_bundle"
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                scores_path = root / "scores.csv"
                triples_path = root / "relational_triples.jsonl"
                lcm_dir = root / "sweep"
                outdir = root / "reports"
                lcm_dir.mkdir(parents=True, exist_ok=True)
                scores_path.write_text(
                    "\n".join(
                        [
                            "file,focus,score,n_nodes,n_edges",
                            "model_a.json,appetite suppression pathway,0.9,3,2",
                            "model_b.json,adherence reinforcement,0.5,2,1",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                triples_path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "subj": "GLP-1 agonists",
                                    "rel": "increase",
                                    "obj": "satiety",
                                    "statement": "GLP-1 agonists increase satiety and reduce caloric intake.",
                                }
                            ),
                            json.dumps(
                                {
                                    "subj": "satiety",
                                    "rel": "supports",
                                    "obj": "adherence",
                                    "statement": "Higher satiety supports dietary adherence over time.",
                                }
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (lcm_dir / "model_a.json").write_text(
                    json.dumps(
                        {
                            "focus": "appetite suppression pathway",
                            "nodes": ["GLP-1 agonists", "satiety", "adherence"],
                            "edges": [
                                {"src": "GLP-1 agonists", "rel": "increase", "dst": "satiety"},
                                {"src": "satiety", "rel": "supports", "dst": "adherence"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                (lcm_dir / "model_b.json").write_text(
                    json.dumps(
                        {
                            "focus": "adherence reinforcement",
                            "nodes": ["GLP-1 agonists", "satiety"],
                            "edges": [
                                {"src": "GLP-1 agonists", "rel": "increase", "dst": "satiety"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                original_argv = sys.argv
                sys.argv = [
                    "make_credibility_bundle.py",
                    "--scores",
                    str(scores_path),
                    "--triples",
                    str(triples_path),
                    "--lcm-dir",
                    str(lcm_dir),
                    "--outdir",
                    str(outdir),
                    "--name",
                    "glp1_demo",
                    "--topk-models",
                    "2",
                    "--max-per-tier",
                    "5",
                ]
                try:
                    with redirect_stdout(StringIO()):
                        module.main()
                finally:
                    sys.argv = original_argv

                summary_md = (outdir / "glp1_demo_executive_summary.md").read_text(encoding="utf-8")

            self.assertIn("## Tier 1 Claims", summary_md)
            self.assertIn("## Tier 2 Claims", summary_md)
            self.assertNotIn("# Executive", summary_md)
            self.assertNotIn("Models used", summary_md)
            self.assertNotIn("How to read credibility", summary_md)
            self.assertNotIn("Notes and caveats", summary_md)
        finally:
            self._cleanup_public_script_module(
                repo_root=repo_root,
                added=added,
                original_tqdm=original_tqdm,
                original_requests=original_requests,
            )
            if original_networkx is None:
                sys.modules.pop("networkx", None)
            else:
                sys.modules["networkx"] = original_networkx
            if original_matplotlib is None:
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = original_matplotlib
            if original_pyplot is None:
                sys.modules.pop("matplotlib.pyplot", None)
            else:
                sys.modules["matplotlib.pyplot"] = original_pyplot


if __name__ == "__main__":
    unittest.main()
