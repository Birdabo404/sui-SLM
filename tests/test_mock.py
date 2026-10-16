#!/usr/bin/env python3
"""Import and mock-run tests. No Ollama, no live Chroma index."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from shared.prompt import ABSTAIN, SYSTEM, system_for_mode, user_msg
from rag.query_hybrid import dense_scores, rrf_fuse, tokenize
from app.engine import (
    BASE_MODEL,
    CSV_FIELDS,
    THESIS_CONFIGS,
    WEAK_DENSE,
    answer_question,
    build_messages,
    config_id,
    git_commit,
    is_ollama_process,
    is_weak,
    log_answer,
    model_name_for,
    ollama_chat,
    uses_rag,
)
from shared.ollama_api import THINK, chat_result
from shared.frozen import load_frozen
from eval.run_eval import (
    EVAL_FIELDS,
    OVERRIDE_REFUSED_ON_TEST,
    SWEEP_REFUSED_ON_TEST,
    as_bool,
    load_questions,
    refuse_sweep_on_test,
    refuse_test_overrides,
    run_eval,
    sweep_abstain,
)


class PromptTests(unittest.TestCase):
    def test_system_text(self):
        self.assertIn("CCIS Linux lab assistant", SYSTEM)
        self.assertIn("Think step by step", SYSTEM)
        self.assertIn("don't know", SYSTEM)
        self.assertIn("Never invent", SYSTEM)

    def test_shared_with_build_sft(self):
        import kd.build_sft as build_sft

        self.assertIs(build_sft.SYSTEM, SYSTEM)
        self.assertIs(build_sft.ABSTAIN, ABSTAIN)

    def test_mode_prompts(self):
        self.assertEqual(system_for_mode("full"), SYSTEM)
        self.assertNotIn("Think step by step", system_for_mode("rag"))
        self.assertNotIn("excerpts", system_for_mode("base"))
        self.assertNotIn("excerpts", system_for_mode("cot"))
        with self.assertRaises(ValueError):
            system_for_mode("nope")

    def test_user_msg_numbering(self):
        text = user_msg("how do I use tar", [
            {"title": "tar (tldr common)", "text": "tar -xzf"},
            {"title": "gzip(1)", "text": "gzip files"},
        ])
        self.assertIn("[1] tar (tldr common)", text)
        self.assertIn("[2] gzip(1)", text)
        self.assertIn("Question: how do I use tar", text)


class ConfigTests(unittest.TestCase):
    def test_five_thesis_configs(self):
        ids = [config_id(mk, mode) for mk, mode in THESIS_CONFIGS]
        self.assertEqual(
            ids,
            ["base", "base+CoT", "base+RAG", "base+RAG+CoT", "distilled+RAG+CoT"],
        )

    def test_model_and_mode_are_separate(self):
        self.assertEqual(BASE_MODEL, "thesis-base-q4km")
        self.assertEqual(model_name_for("base"), "thesis-base-q4km")
        self.assertEqual(model_name_for("base", base_model="qwen3.5:2b", distilled_model=""), "qwen3.5:2b")
        self.assertEqual(model_name_for("distilled", base_model="qwen3.5:2b", distilled_model=""), "")
        self.assertEqual(
            model_name_for("distilled", base_model="qwen3.5:2b", distilled_model="student:latest"),
            "student:latest",
        )

    def test_unknown_config_parts(self):
        with self.assertRaises(ValueError):
            config_id("other", "base")
        with self.assertRaises(ValueError):
            config_id("base", "nope")


class HybridRankingTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Extract a tar.gz with tar -xzf"),
                         ["extract", "a", "tar.gz", "with", "tar", "xzf"])

    def test_dense_scores_from_chroma_distance(self):
        query_result = {"ids": [["a", "b"]], "distances": [[0.2, 0.7]]}
        sim = dense_scores(col=None, qemb=[[1.0, 0.0]], hit_ids=["a", "b"], query_result=query_result)
        self.assertAlmostEqual(sim["a"], 0.8)
        self.assertAlmostEqual(sim["b"], 0.3)

    def test_dense_scores_fills_missing_from_embeddings(self):
        class Col:
            def get(self, ids, include):
                return {"ids": ids, "embeddings": [[1.0, 0.0]]}

        query_result = {"ids": [["a"]], "distances": [[0.1]]}
        sim = dense_scores(Col(), [[1.0, 0.0]], ["a", "c"], query_result)
        self.assertAlmostEqual(sim["a"], 0.9)
        self.assertAlmostEqual(sim["c"], 1.0)

    def test_dense_scores_accepts_numpy_arrays(self):
        import numpy as np

        class Col:
            def get(self, ids, include):
                return {
                    "ids": np.array(ids),
                    "embeddings": np.array([[0.0, 1.0]], dtype=np.float32),
                }

        query_result = {
            "ids": np.array([["a"]]),
            "distances": np.array([[0.25]], dtype=np.float32),
        }
        sim = dense_scores(Col(), np.array([[1.0, 0.0]]), ["a", "c"], query_result)
        self.assertAlmostEqual(sim["a"], 0.75, places=5)
        self.assertAlmostEqual(sim["c"], 0.0, places=5)

    def test_dense_scores_against_chromadb(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            self.skipTest("chromadb not installed")
        settings = Settings(anonymized_telemetry=False)
        self.assertFalse(settings.anonymized_telemetry)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = chromadb.PersistentClient(path=tmp.name, settings=settings)
        col = client.create_collection("linux_lab", metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["a", "c"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["aa", "cc"],
            metadatas=[{"k": 1}, {"k": 2}],
        )
        qemb = [[1.0, 0.0]]
        r = col.query(query_embeddings=qemb, n_results=1)
        embs = col.get(ids=["c"], include=["embeddings"]).get("embeddings")
        self.assertIsNotNone(embs)
        sim = dense_scores(col, qemb, ["a", "c"], r)
        self.assertGreater(sim["a"], 0.99)
        self.assertAlmostEqual(sim["c"], 0.0, places=5)
        del col, client
        reopened = chromadb.PersistentClient(path=tmp.name, settings=settings)
        col2 = reopened.get_collection("linux_lab")
        self.assertEqual(col2.count(), 2)
        sim2 = dense_scores(col2, qemb, ["a", "c"], col2.query(query_embeddings=qemb, n_results=1))
        self.assertGreater(sim2["a"], 0.99)
        self.assertAlmostEqual(sim2["c"], 0.0, places=5)

    def test_rrf_matches_original_formula(self):
        lanes = [["a", "b", "c"], ["b", "a"], ["a"]]
        fused = rrf_fuse(lanes)
        self.assertAlmostEqual(fused["a"], 1 / 60 + 1 / 61 + 1 / 60)
        self.assertAlmostEqual(fused["b"], 1 / 61 + 1 / 60)
        self.assertAlmostEqual(fused["c"], 1 / 62)
        ranked = sorted(fused, key=lambda i: -fused[i])
        self.assertEqual(ranked[0], "a")

    def test_weak_threshold(self):
        self.assertTrue(is_weak([]))
        self.assertTrue(is_weak([{"rrf": 1 / 60}]))
        self.assertTrue(is_weak([{"dense": 0.10, "rrf": 1 / 60}]))
        self.assertTrue(is_weak([{"dense": WEAK_DENSE - 0.01, "rrf": 0.05}]))
        self.assertFalse(is_weak([{"dense": WEAK_DENSE, "rrf": 1 / 60}]))
        self.assertFalse(is_weak([{"dense": 0.55, "rrf": 1 / 60}]))
        # BM25 can rank an exact-token hit first with a low cosine; a later hit may still be strong.
        self.assertFalse(is_weak([
            {"dense": 0.12, "rrf": 1 / 60},
            {"dense": 0.55, "rrf": 1 / 61},
        ]))
        self.assertTrue(is_weak([
            {"dense": 0.12, "rrf": 1 / 60},
            {"dense": 0.20, "rrf": 1 / 61},
        ]))


class OllamaRamTests(unittest.TestCase):
    def test_is_ollama_process(self):
        self.assertTrue(is_ollama_process("ollama", ["/usr/bin/ollama", "serve"]))
        self.assertTrue(is_ollama_process("ollama", ["/usr/bin/ollama", "runner", "--model", "qwen"]))
        self.assertTrue(is_ollama_process(
            "ollama_llama_server",
            ["/tmp/ollama/runners/cpu/ollama_llama_server"],
        ))
        self.assertFalse(is_ollama_process("python3", ["streamlit", "run", "app/app.py"]))
        self.assertFalse(is_ollama_process("bash", ["install.sh"]))
        self.assertFalse(is_ollama_process("python3", ["python3", "/tmp/ollama/notes.py"]))
        self.assertFalse(is_ollama_process("grep", ["grep", "-n", "ollama"]))

    def test_ollama_rss_sums_matching_processes(self):
        class Proc:
            def __init__(self, name, cmdline, rss):
                self.info = {"name": name, "cmdline": cmdline}
                self._rss = rss

            def memory_info(self):
                return type("m", (), {"rss": self._rss})()

        procs = [
            Proc("ollama", ["/usr/bin/ollama", "serve"], 80 * 1024 * 1024),
            Proc("ollama", ["/usr/bin/ollama", "runner"], 1200 * 1024 * 1024),
            Proc("python3", ["python3", "/tmp/ollama/app.py"], 400 * 1024 * 1024),
            Proc("python3", ["streamlit", "run", "app/app.py"], 400 * 1024 * 1024),
        ]
        with patch("app.engine.psutil.process_iter", return_value=procs):
            from app.engine import ollama_rss_bytes

            self.assertEqual(ollama_rss_bytes(), (80 + 1200) * 1024 * 1024)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, question, k=5, tldr_lane=True):
        self.calls.append({"question": question, "k": k, "tldr_lane": tldr_lane})
        return list(self.hits)[:k]


class MockRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "answers.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _chat(self, messages, model):
        self.last_messages = messages
        self.last_model = model
        return chat_result("MOCK ANSWER from " + messages[0]["content"][:24])

    def test_base_skips_retrieval(self):
        retriever = FakeRetriever([{"title": "x", "text": "y", "rrf": 0.05, "source": "tldr", "path": "x.md"}])
        result = answer_question(
            "how do I list files",
            "base",
            "qwen3.5:2b",
            retriever=retriever,
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        self.assertEqual(retriever.calls, [])
        self.assertFalse(uses_rag("base"))
        self.assertIn("MOCK ANSWER", result["text"])
        self.assertNotIn("excerpts", self.last_messages[0]["content"])
        self.assertEqual(self.last_model, "qwen3.5:2b")
        self.assertEqual(result["config"], "base")

    def test_cot_asks_for_steps(self):
        result = answer_question(
            "how do I list files",
            "cot",
            "qwen3.5:2b",
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        self.assertIn("reasoning steps", self.last_messages[1]["content"])
        self.assertIn("MOCK ANSWER", result["text"])
        self.assertEqual(result["config"], "base+CoT")

    def test_full_weak_retrieval_abstains_without_ollama(self):
        retriever = FakeRetriever([{
            "id": "weak", "title": "unrelated", "text": "nothing",
            "source": "tldr", "path": "x.md", "rrf": 1 / 60, "dense": 0.10,
        }])
        called = []

        def boom(messages, model):
            called.append(1)
            raise AssertionError("Ollama should not be called on weak retrieval")

        result = answer_question(
            "how do I time travel",
            "full",
            "qwen3.5:2b",
            retriever=retriever,
            chat_fn=boom,
            log_path=self.log,
            model_key="base",
        )
        self.assertEqual(called, [])
        self.assertTrue(result["weak"])
        self.assertEqual(result["text"], ABSTAIN)
        self.assertEqual(retriever.calls[0]["tldr_lane"], True)
        self.assertEqual(result["config"], "base+RAG+CoT")

    def test_full_strong_retrieval_calls_chat(self):
        hits = [{
            "id": "tar#0", "title": "tar (tldr common)", "text": "tar -xzf archive.tar.gz",
            "source": "tldr", "path": "tldr/common/tar.md",
            "rrf": 1 / 60 + 1 / 60 + 1 / 60, "dense": 0.55,
        }]
        retriever = FakeRetriever(hits)
        result = answer_question(
            "how do I extract a tar.gz",
            "full",
            "student-gguf",
            retriever=retriever,
            chat_fn=self._chat,
            log_path=self.log,
            model_key="distilled",
        )
        self.assertFalse(result["weak"])
        self.assertIn("MOCK ANSWER", result["text"])
        self.assertEqual(self.last_model, "student-gguf")
        self.assertEqual(self.last_messages[0]["content"], SYSTEM)
        self.assertIn("[1] tar (tldr common)", self.last_messages[1]["content"])
        self.assertIn("Think step by step", self.last_messages[0]["content"])
        self.assertEqual(result["config"], "distilled+RAG+CoT")

    def test_rag_only_has_no_step_request_in_system(self):
        hits = [{
            "id": "apt#0", "title": "apt (tldr common)", "text": "sudo apt install",
            "source": "tldr", "path": "tldr/common/apt.md", "rrf": 0.05, "dense": 0.52,
        }]
        result = answer_question(
            "install a package",
            "rag",
            "qwen3.5:2b",
            retriever=FakeRetriever(hits),
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        self.assertNotIn("Think step by step", self.last_messages[0]["content"])
        self.assertIn("Do not write a step list", self.last_messages[1]["content"])
        self.assertEqual(result["config"], "base+RAG")

    def test_csv_log_ram_model_mode(self):
        answer_question(
            "hello",
            "base",
            "qwen3.5:2b",
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        with open(self.log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
        self.assertEqual(rows[0]["mode"], "base")
        self.assertEqual(rows[0]["model"], "qwen3.5:2b")
        self.assertEqual(rows[0]["config"], "base")
        self.assertEqual(rows[0]["n_chunks"], "0")
        self.assertGreaterEqual(float(rows[0]["latency_s"]), 0)
        self.assertGreater(float(rows[0]["app_rss_mb"]), 0)
        self.assertGreaterEqual(float(rows[0]["ollama_rss_mb"]), 0)
        self.assertGreaterEqual(float(rows[0]["total_mb"]), float(rows[0]["app_rss_mb"]))
        self.assertGreater(float(rows[0]["system_used_mb"]), 0)
        self.assertTrue(rows[0]["git_commit"])
        self.assertEqual(rows[0]["threshold"], "0.30")
        self.assertEqual(rows[0]["k"], "5")
        self.assertEqual(rows[0]["think"], "false")
        self.assertEqual(rows[0]["think_leak"], "0")
        self.assertTrue(rows[0]["raw_output"].startswith("MOCK ANSWER from "))
        self.assertEqual(rows[0]["model_digest"], "")

    def test_csv_rotates_on_header_mismatch(self):
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("old,columns\n1,2\n")
        log_answer({"question": "q", "mode": "base", "model": "qwen3.5:2b"}, path=self.log)
        self.assertTrue(self.log.exists())
        with open(self.log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
        self.assertEqual(rows[0]["question"], "q")
        backups = list(self.log.parent.glob("answers.*.csv"))
        self.assertEqual(len(backups), 1)
        with open(backups[0], encoding="utf-8") as f:
            self.assertEqual(f.readline().strip(), "old,columns")

    def test_csv_appends_when_header_matches(self):
        log_answer({"question": "one", "mode": "base", "model": "m"}, path=self.log)
        log_answer({"question": "two", "mode": "cot", "model": "m"}, path=self.log)
        with open(self.log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["question"] for r in rows], ["one", "two"])
        self.assertEqual(list(self.log.parent.glob("answers.*.csv")), [])

    def test_build_messages_unknown_mode(self):
        with self.assertRaises(ValueError):
            build_messages("q", [], "nope")

    def test_ollama_chat_sends_temperature_and_seed(self):
        fake = MagicMock()
        fake.json.return_value = {"message": {"content": "hi"}}
        with patch("app.engine.requests.post", return_value=fake) as post:
            out = ollama_chat(
                [{"role": "user", "content": "q"}],
                "qwen3.5:2b",
                temperature=0,
                seed=42,
            )
        self.assertEqual(out["text"], "hi")
        self.assertEqual(out["think"], "false")
        self.assertEqual(out["think_leak"], 0)
        self.assertEqual(out["raw_output"], "hi")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["think"], False)
        self.assertIs(body["think"], THINK)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 42)
        self.assertEqual(body["model"], "qwen3.5:2b")
        self.assertNotIn("think", body["options"])


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.questions = self.dir / "q.jsonl"
        rows = [
            {"id": "q1", "question": "how do I list files", "answerable": True,
             "gold_answer": "ls", "source": "tldr ls"},
            {"id": "q2", "question": "how do I time travel", "answerable": False,
             "gold_answer": "", "source": ""},
        ]
        with open(self.questions, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.out = self.dir / "eval.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_questions(self):
        rows = load_questions(self.questions)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "q1")

    def test_run_eval_skips_empty_distilled_and_logs_four_configs(self):
        hits = [{
            "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md", "rrf": 0.05, "dense": 0.55,
        }]
        seen = []

        def chat(messages, model):
            seen.append(model)
            return chat_result("mock")

        n, skipped, _sweep = run_eval(
            self.questions,
            self.out,
            distilled_model="",
            retriever=FakeRetriever(hits),
            chat_fn=chat,
            split="dev",
            sweep=False,
        )
        self.assertEqual(skipped, ["distilled+RAG+CoT"])
        self.assertEqual(n, 8)
        with open(self.out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        self.assertEqual(list(rows[0].keys()), EVAL_FIELDS)
        configs = sorted({r["config"] for r in rows})
        self.assertEqual(configs, ["base", "base+CoT", "base+RAG", "base+RAG+CoT"])
        self.assertTrue(all(r["model"] == "thesis-base-q4km" for r in rows))
        self.assertTrue(all(r["git_commit"] for r in rows))
        self.assertTrue(all(r["question_id"] in ("q1", "q2") for r in rows))
        self.assertTrue(all(r["split"] == "dev" for r in rows))
        self.assertTrue(all(r["threshold"] == "0.30" for r in rows))
        self.assertTrue(all(r["k"] == "5" for r in rows))
        self.assertTrue(all(r["think"] == "false" for r in rows))
        self.assertTrue(all(r["think_leak"] == "0" for r in rows))
        self.assertTrue(all("raw_output" in r for r in rows))
        self.assertTrue(all(r["model_digest"] == "" for r in rows))
        rag_rows = [r for r in rows if r["mode"] in ("rag", "full")]
        self.assertTrue(all("ls (tldr common)" in r["cited_sources"] for r in rag_rows))
        self.assertTrue(all("app_rss_mb" in r and r["app_rss_mb"] for r in rows))

    def test_run_eval_includes_distilled_when_named(self):
        hits = [{
            "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md", "rrf": 0.05, "dense": 0.55,
        }]
        n, skipped, _sweep = run_eval(
            self.questions,
            self.out,
            distilled_model="student:latest",
            retriever=FakeRetriever(hits),
            chat_fn=lambda messages, model: chat_result(model),
            split="dev",
            sweep=False,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(n, 10)
        with open(self.out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        distilled = [r for r in rows if r["config"] == "distilled+RAG+CoT"]
        self.assertEqual(len(distilled), 2)
        self.assertTrue(all(r["model"] == "student:latest" for r in distilled))
        self.assertTrue(all(r["mode"] == "full" for r in distilled))

    def test_threshold_sweep_precision_recall(self):
        class Retriever:
            def retrieve(self, question, k=5, tldr_lane=True):
                if "time travel" in question:
                    return [{"dense": 0.10}]
                return [{"dense": 0.12}, {"dense": 0.55}]

        rows = sweep_abstain(
            load_questions(self.questions),
            Retriever(),
            ks=(5,),
            thresholds=(0.20, 0.30, 0.60),
            split="dev",
        )
        by_t = {r["threshold"]: r for r in rows}
        self.assertEqual(by_t["0.30"]["abstain_on_answerable"], 0)
        self.assertEqual(by_t["0.30"]["abstain_on_unanswerable"], 1)
        self.assertEqual(by_t["0.30"]["abstain_precision"], "1.000")
        self.assertEqual(by_t["0.30"]["abstain_recall"], "1.000")
        self.assertEqual(by_t["0.30"]["split"], "dev")
        self.assertEqual(by_t["0.30"]["k"], 5)
        self.assertEqual(by_t["0.60"]["abstain_on_answerable"], 1)
        self.assertEqual(by_t["0.20"]["abstain_on_unanswerable"], 1)
        self.assertEqual(by_t["0.20"]["abstain_on_answerable"], 0)

    def test_topk_sweep_uses_max_cosine_in_prefix(self):
        class Retriever:
            def retrieve(self, question, k=5, tldr_lane=True):
                return [{"dense": 0.12}, {"dense": 0.55}][:k]

        rows = sweep_abstain(
            [{"id": "q", "question": "q", "answerable": True,
              "gold_answer": "x", "source": "x"}],
            Retriever(),
            ks=(1, 2),
            thresholds=(0.30,),
            split="dev",
        )
        by_k = {r["k"]: r for r in rows}
        self.assertEqual(by_k[1]["abstain_on_answerable"], 1)
        self.assertEqual(by_k[2]["abstain_on_answerable"], 0)

    def test_sweep_refused_on_test_split(self):
        with self.assertRaises(SystemExit) as cm:
            refuse_sweep_on_test("test", sweep=True)
        self.assertEqual(str(cm.exception), SWEEP_REFUSED_ON_TEST)
        self.assertEqual(refuse_sweep_on_test("test", sweep=False), "test")
        self.assertEqual(refuse_sweep_on_test("dev", sweep=True), "dev")
        with self.assertRaises(SystemExit) as cm:
            run_eval(
                self.questions,
                self.out,
                distilled_model="student:latest",
                retriever=FakeRetriever([{"dense": 0.55}]),
                chat_fn=lambda messages, model: chat_result("x"),
                split="test",
                sweep=True,
            )
        self.assertIn("held out", str(cm.exception))
        self.assertFalse(self.out.exists())

    def test_test_split_refuses_threshold_k_frozen_overrides(self):
        hits = FakeRetriever([{
            "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md", "dense": 0.55,
        }])
        frozen = self.dir / "frozen.json"
        frozen.write_text('{"threshold": 0.42, "k": 8}\n', encoding="utf-8")
        cases = [
            {"k": 8},
            {"threshold": 0.42},
            {"frozen_path": frozen},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(SystemExit) as cm:
                    run_eval(
                        self.questions,
                        self.out,
                        distilled_model="student:latest",
                        retriever=hits,
                        chat_fn=lambda messages, model: chat_result("x"),
                        split="test",
                        sweep=False,
                        **kwargs,
                    )
                self.assertEqual(str(cm.exception), OVERRIDE_REFUSED_ON_TEST)
        self.assertFalse(self.out.exists())
        extras = [
            ["-k", "8"],
            ["--threshold", "0.42"],
            ["--frozen", str(frozen)],
        ]
        for extra in extras:
            with self.subTest(cli=extra):
                r = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "eval" / "run_eval.py"),
                        "--split", "test",
                        "--no-sweep",
                        "--questions", str(self.questions),
                        "--out", str(self.out),
                        *extra,
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("cannot be overridden", (r.stderr + r.stdout))
        refuse_test_overrides("dev", k=8, threshold=0.42, frozen_path=frozen)
        with self.assertRaises(SystemExit) as cm:
            refuse_test_overrides("test", k=8)
        self.assertEqual(str(cm.exception), OVERRIDE_REFUSED_ON_TEST)

    def test_cli_refuses_sweep_on_test_split(self):
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval" / "run_eval.py"),
                "--split", "test",
                "--questions", str(self.questions),
                "--out", str(self.out),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("held out", (r.stderr + r.stdout))
        self.assertFalse(self.out.exists())

    def test_eval_records_split_threshold_k_from_frozen_file(self):
        frozen = self.dir / "frozen.json"
        frozen.write_text('{"threshold": 0.42, "k": 8}\n', encoding="utf-8")
        n, skipped, sweep_rows = run_eval(
            self.questions,
            self.out,
            distilled_model="student:latest",
            retriever=FakeRetriever([{
                "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
                "source": "tldr", "path": "tldr/common/ls.md", "dense": 0.55,
            }]),
            chat_fn=lambda messages, model: chat_result("x"),
            split="dev",
            sweep=False,
            frozen_path=frozen,
        )
        self.assertEqual(n, 10)
        self.assertEqual(skipped, [])
        self.assertEqual(sweep_rows, [])
        with open(self.out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0].keys()), EVAL_FIELDS)
        self.assertTrue(all(r["split"] == "dev" for r in rows))
        self.assertTrue(all(r["threshold"] == "0.42" for r in rows))
        self.assertTrue(all(r["k"] == "8" for r in rows))

    def test_dev_and_test_question_files(self):
        dev = load_questions(ROOT / "eval" / "dev.jsonl")
        test = load_questions(ROOT / "eval" / "test.jsonl")
        tuning = [r for r in dev if r.get("set") == "tuning"]
        unans = [r for r in dev if r.get("set") == "unanswerable"]
        self.assertEqual(len(tuning), 24)
        self.assertGreaterEqual(len(unans), 1)
        self.assertTrue(all(as_bool(r["answerable"]) for r in tuning))
        self.assertTrue(all(not as_bool(r["answerable"]) for r in unans))
        self.assertTrue(all(not as_bool(r["answerable"]) for r in test if r.get("set") == "unanswerable"))
        overlap = {r["question"] for r in dev} & {r["question"] for r in test}
        self.assertEqual(overlap, set())
        first = (ROOT / "eval" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(first.startswith("#"))
        self.assertIn("NOT the thesis test set", first)
        self.assertTrue(all(r.get("placeholder") is True for r in test))

    def test_load_frozen_file(self):
        frozen = load_frozen()
        self.assertEqual(frozen["threshold"], 0.30)
        self.assertEqual(frozen["k"], 5)

    def test_sweep_file_records_split_threshold_k(self):
        hits = [{
            "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md", "dense": 0.55,
        }]
        n, skipped, sweep_rows = run_eval(
            self.questions,
            self.out,
            configs=[("base", "rag")],
            retriever=FakeRetriever(hits),
            chat_fn=lambda messages, model: chat_result("x"),
            split="dev",
            sweep=True,
        )
        self.assertEqual(n, 2)
        self.assertEqual(skipped, [])
        self.assertTrue(sweep_rows)
        sweep_path = self.out.with_suffix(".sweep.csv")
        with open(sweep_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(sweep_rows))
        self.assertTrue(all(r["split"] == "dev" for r in rows))
        self.assertTrue(all(r["threshold"] for r in rows))
        self.assertTrue(all(r["k"] for r in rows))
        self.assertTrue(any(r["k"] == "5" and r["threshold"] == "0.30" for r in rows))


class SnapshotResolveTests(unittest.TestCase):
    def test_picks_newest_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(Path(td) / "snapshot-2026-10-11")
            os.makedirs(Path(td) / "snapshot-2026-10-12")
            script = f'''
            SCRIPT_DIR="{td}"
            . "{ROOT}/corpus/resolve_snap.sh"
            printf "%s" "$SNAP"
            '''
            out = subprocess.check_output(["bash", "-c", script], text=True)
            self.assertTrue(out.endswith("snapshot-2026-10-12"))

    def test_honors_existing_snap(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(Path(td) / "snapshot-2026-10-11")
            os.makedirs(Path(td) / "snapshot-2026-10-12")
            chosen = str(Path(td) / "snapshot-2026-10-11")
            script = f'''
            SCRIPT_DIR="{td}"
            SNAP="{chosen}"
            . "{ROOT}/corpus/resolve_snap.sh"
            printf "%s" "$SNAP"
            '''
            out = subprocess.check_output(["bash", "-c", script], text=True)
            self.assertEqual(out, chosen)

    def test_fails_when_none(self):
        with tempfile.TemporaryDirectory() as td:
            script = f'''
            SCRIPT_DIR="{td}"
            . "{ROOT}/corpus/resolve_snap.sh"
            '''
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("No snapshot-", r.stderr)

    def test_git_commit_looks_like_sha(self):
        sha = git_commit()
        self.assertGreaterEqual(len(sha), 7)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha.lower()))


class InstallScriptTests(unittest.TestCase):
    def _fake_python(self, folder, version_line):
        fake = Path(folder) / "python3"
        fake.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = --version ]; then echo {version_line}; exit 0; fi\n'
            'echo "unexpected: $*" >&2\n'
            "exit 2\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = folder + os.pathsep + env.get("PATH", "")
        return env

    def test_accepts_supported_python_and_prints_it_first(self):
        with tempfile.TemporaryDirectory() as td:
            env = self._fake_python(td, "Python 3.14.4")
            r = subprocess.run(
                ["bash", str(ROOT / "install.sh"), "--check-python"],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[0], "Python 3.14.4")

    def test_rejects_unsupported_python(self):
        with tempfile.TemporaryDirectory() as td:
            env = self._fake_python(td, "Python 3.11.9")
            r = subprocess.run(
                ["bash", str(ROOT / "install.sh"), "--check-python"],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(r.returncode, 0)
        combined = r.stderr + r.stdout
        self.assertIn("Python 3.11.9", combined)
        self.assertIn("3.12, 3.13, and 3.14", combined)
        self.assertTrue(combined.startswith("Python 3.11.9") or r.stdout.startswith("Python 3.11.9"))


class ImportTests(unittest.TestCase):
    def test_engine_and_prompt_import(self):
        import app.engine
        import shared.prompt
        import rag.query_hybrid

        self.assertTrue(hasattr(app.engine, "answer_question"))
        self.assertTrue(hasattr(shared.prompt, "SYSTEM"))
        self.assertTrue(hasattr(rag.query_hybrid, "retrieve"))

    def test_app_py_imports_with_stub_streamlit(self):
        try:
            import streamlit  # noqa: F401
        except ImportError:
            st = MagicMock()
            st.cache_resource = lambda fn: fn
            sys.modules["streamlit"] = st
        import importlib
        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        mod = importlib.import_module("app.app")
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "load_retriever"))
        self.assertEqual(os.environ.get("ANONYMIZED_TELEMETRY"), "False")

    def test_chroma_client_sets_telemetry_env(self):
        import rag.chroma_client  # noqa: F401

        self.assertEqual(os.environ.get("ANONYMIZED_TELEMETRY"), "False")


if __name__ == "__main__":
    unittest.main()
