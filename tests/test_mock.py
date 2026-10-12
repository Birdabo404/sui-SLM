#!/usr/bin/env python3
"""Import and mock-run tests. No Ollama, no live Chroma index."""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from shared.prompt import ABSTAIN, SYSTEM, system_for_mode, user_msg
from rag.query_hybrid import rrf_fuse, tokenize
from app.engine import (
    WEAK_RRF,
    answer_question,
    build_messages,
    is_weak,
    uses_rag,
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


class HybridRankingTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Extract a tar.gz with tar -xzf"),
                         ["extract", "a", "tar.gz", "with", "tar", "xzf"])

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
        self.assertTrue(is_weak([{"rrf": WEAK_RRF / 2}]))
        self.assertFalse(is_weak([{"rrf": 1 / 60}]))


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
        return "MOCK ANSWER from " + messages[0]["content"][:24]

    def test_base_skips_retrieval(self):
        retriever = FakeRetriever([{"title": "x", "text": "y", "rrf": 0.05, "source": "tldr", "path": "x.md"}])
        result = answer_question(
            "how do I list files",
            "base",
            "qwen3.5:2b",
            retriever=retriever,
            chat_fn=self._chat,
            log_path=self.log,
        )
        self.assertEqual(retriever.calls, [])
        self.assertFalse(uses_rag("base"))
        self.assertIn("MOCK ANSWER", result["text"])
        self.assertNotIn("excerpts", self.last_messages[0]["content"])
        self.assertEqual(self.last_model, "qwen3.5:2b")

    def test_cot_asks_for_steps(self):
        result = answer_question(
            "how do I list files",
            "cot",
            "qwen3.5:2b",
            chat_fn=self._chat,
            log_path=self.log,
        )
        self.assertIn("reasoning steps", self.last_messages[1]["content"])
        self.assertIn("MOCK ANSWER", result["text"])

    def test_full_weak_retrieval_abstains_without_ollama(self):
        retriever = FakeRetriever([{
            "id": "weak", "title": "unrelated", "text": "nothing",
            "source": "tldr", "path": "x.md", "rrf": WEAK_RRF / 2,
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
        )
        self.assertEqual(called, [])
        self.assertTrue(result["weak"])
        self.assertEqual(result["text"], ABSTAIN)
        self.assertEqual(retriever.calls[0]["tldr_lane"], True)

    def test_full_strong_retrieval_calls_chat(self):
        hits = [{
            "id": "tar#0", "title": "tar (tldr common)", "text": "tar -xzf archive.tar.gz",
            "source": "tldr", "path": "tldr/common/tar.md", "rrf": 1 / 60 + 1 / 60 + 1 / 60,
        }]
        retriever = FakeRetriever(hits)
        result = answer_question(
            "how do I extract a tar.gz",
            "full",
            "student-gguf",
            retriever=retriever,
            chat_fn=self._chat,
            log_path=self.log,
        )
        self.assertFalse(result["weak"])
        self.assertIn("MOCK ANSWER", result["text"])
        self.assertEqual(self.last_model, "student-gguf")
        self.assertEqual(self.last_messages[0]["content"], SYSTEM)
        self.assertIn("[1] tar (tldr common)", self.last_messages[1]["content"])
        self.assertIn("Think step by step", self.last_messages[0]["content"])

    def test_rag_only_has_no_step_request_in_system(self):
        hits = [{
            "id": "apt#0", "title": "apt (tldr common)", "text": "sudo apt install",
            "source": "tldr", "path": "tldr/common/apt.md", "rrf": 0.05,
        }]
        answer_question(
            "install a package",
            "rag",
            "qwen3.5:2b",
            retriever=FakeRetriever(hits),
            chat_fn=self._chat,
            log_path=self.log,
        )
        self.assertNotIn("Think step by step", self.last_messages[0]["content"])
        self.assertIn("Do not write a step list", self.last_messages[1]["content"])

    def test_csv_log(self):
        answer_question(
            "hello",
            "base",
            "qwen3.5:2b",
            chat_fn=self._chat,
            log_path=self.log,
        )
        with open(self.log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "base")
        self.assertEqual(rows[0]["model"], "qwen3.5:2b")
        self.assertEqual(rows[0]["n_chunks"], "0")
        self.assertGreaterEqual(float(rows[0]["latency_s"]), 0)
        self.assertGreater(float(rows[0]["app_rss_peak_mb"]), 0)

    def test_build_messages_unknown_mode(self):
        with self.assertRaises(ValueError):
            build_messages("q", [], "nope")


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


if __name__ == "__main__":
    unittest.main()
