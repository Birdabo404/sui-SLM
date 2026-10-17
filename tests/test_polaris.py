#!/usr/bin/env python3
"""Polaris UI helpers. No live Ollama."""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.prompt import ABSTAIN
from app.engine import WEAK_DENSE, answer_question, is_weak, model_digest, ollama_reachable
from app.polaris import (
    EXPORT_FIELDS,
    POLARIS,
    best_match_label,
    build_run_record,
    card_head_html,
    default_compare_modes,
    evidence_rows_html,
    feature,
    idk_card_html,
    index_chunk_count,
    max_dense,
    metrics_html,
    pair_compare_modes,
    records_to_csv,
    records_to_json,
    retrieval_bar_html,
    run_compare_pair,
    source_chips_html,
    source_key,
    split_code_blocks,
    split_reasoning,
    style_path,
)
class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, question, k=5, tldr_lane=True):
        self.calls.append({"question": question, "k": k, "tldr_lane": tldr_lane})
        return list(self.hits)[:k]


class FlagTests(unittest.TestCase):
    def test_flags_are_a_dict(self):
        self.assertIsInstance(POLARIS, dict)
        self.assertIn("enabled", POLARIS)
        self.assertIn("compare", POLARIS)
        self.assertIn("evidence_panel", POLARIS)
        self.assertIn("run_records", POLARIS)

    def test_feature_respects_master_switch(self):
        old = POLARIS["enabled"]
        try:
            POLARIS["enabled"] = False
            self.assertFalse(feature("compare"))
            self.assertFalse(feature("enabled"))
            POLARIS["enabled"] = True
            self.assertTrue(feature("compare"))
        finally:
            POLARIS["enabled"] = old


class RunRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "answers.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _chat(self, messages, model):
        return "use `tar -xzf file.tar.gz`"

    def test_record_from_answer_question(self):
        hits = [{
            "id": "tar#0",
            "title": "tar (tldr common)",
            "text": "tar -xzf archive.tar.gz " * 20,
            "source": "tldr",
            "path": "tldr/common/tar.md",
            "rrf": 0.05,
            "dense": 0.74,
            "bm25": 6.2,
        }]
        result = answer_question(
            "extract a tar.gz",
            "full",
            "qwen3.5:2b",
            retriever=FakeRetriever(hits),
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        rec = build_run_record(result, index_path="rag/rag-index-v2")
        self.assertEqual(rec["question"], "extract a tar.gz")
        self.assertEqual(rec["mode"], "full")
        self.assertEqual(rec["config"], "base+RAG+CoT")
        self.assertEqual(rec["model"], "qwen3.5:2b")
        self.assertEqual(rec["answer"], "use `tar -xzf file.tar.gz`")
        self.assertFalse(rec["abstained"])
        self.assertAlmostEqual(rec["retrieval_max_dense"], 0.74)
        self.assertEqual(rec["index_path"], "rag/rag-index-v2")
        self.assertEqual(rec["threshold"], WEAK_DENSE)
        self.assertEqual(rec["k"], 5)
        self.assertTrue(rec["git_commit"])
        self.assertTrue(rec["ts"])
        self.assertEqual(len(rec["hits"]), 1)
        self.assertEqual(rec["hits"][0]["rank"], 1)
        self.assertEqual(rec["hits"][0]["source"], "tldr")
        self.assertAlmostEqual(rec["hits"][0]["dense"], 0.74)
        self.assertAlmostEqual(rec["hits"][0]["rrf"], 0.05)
        self.assertAlmostEqual(rec["hits"][0]["bm25"], 6.2)
        self.assertLessEqual(len(rec["hits"][0]["snippet"]), 200)
        self.assertGreaterEqual(rec["total_mb"], rec["app_rss_mb"])
        self.assertEqual(rec["docs"], ["tar (tldr common)"])

    def test_idk_reuses_is_weak_and_abstain(self):
        hits = [{
            "id": "weak", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md",
            "rrf": 1 / 60, "dense": 0.10, "bm25": 1.2,
        }]
        result = answer_question(
            "what's the lab wi-fi password?",
            "rag",
            "qwen3.5:2b",
            retriever=FakeRetriever(hits),
            chat_fn=lambda messages, model: (_ for _ in ()).throw(AssertionError("no ollama")),
            log_path=self.log,
            model_key="base",
        )
        self.assertTrue(is_weak(result["hits"]))
        self.assertTrue(result["weak"])
        self.assertEqual(result["text"], ABSTAIN)
        rec = build_run_record(result, index_path="rag/rag-index-v2")
        self.assertTrue(rec["abstained"])
        self.assertEqual(rec["answer"], ABSTAIN)
        html = idk_card_html(rec)
        self.assertIn("no fix on this one", html)
        self.assertIn("best match:", html)
        self.assertIn("retr low", html)

    def test_base_record_has_no_retrieval_max(self):
        result = answer_question(
            "hello",
            "base",
            "qwen3.5:2b",
            chat_fn=self._chat,
            log_path=self.log,
            model_key="base",
        )
        rec = build_run_record(result, index_path="rag/rag-index-v2")
        self.assertIsNone(rec["retrieval_max_dense"])
        self.assertFalse(rec["abstained"])
        self.assertEqual(rec["hits"], [])

    def test_export_reads_records_not_log_csv(self):
        rec = build_run_record(
            {
                "ts": "2026-10-17T00:00:00+00:00",
                "question": "q",
                "config": "base",
                "mode": "base",
                "model": "qwen3.5:2b",
                "text": "hi",
                "latency_s": 1.25,
                "app_rss_mb": 100,
                "ollama_rss_mb": 200,
                "total_mb": 300,
                "hits": [],
                "weak": False,
                "threshold": 0.30,
                "k": 5,
                "git_commit": "abc1234",
            },
            index_path="/tmp/index",
        )
        csv_text = records_to_csv([rec])
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(list(rows[0].keys()), list(EXPORT_FIELDS))
        self.assertEqual(rows[0]["question"], "q")
        self.assertEqual(rows[0]["total_mb"], "300")
        self.assertNotIn("answers.csv", csv_text)
        payload = json.loads(records_to_json([rec]))
        self.assertEqual(payload[0]["answer"], "hi")
        self.assertEqual(payload[0]["index_path"], "/tmp/index")


class CompareTests(unittest.TestCase):
    def test_defaults_base_vs_full(self):
        self.assertEqual(default_compare_modes(), ("base", "full"))

    def test_pair_must_differ(self):
        self.assertEqual(pair_compare_modes("base", "base"), ("base", "cot"))
        self.assertEqual(pair_compare_modes("full", "full"), ("full", "base"))
        self.assertEqual(pair_compare_modes("base", "full"), ("base", "full"))

    def test_runs_sequentially_never_nested(self):
        order = []
        inflight = []

        def answer_fn(question, mode, **kwargs):
            self.assertEqual(inflight, [])
            inflight.append(mode)
            order.append(mode)
            inflight.pop()
            return {"mode": mode, "question": question}

        left, right = run_compare_pair("q", "base", "full", answer_fn, model="qwen3.5:2b")
        self.assertEqual(order, ["base", "full"])
        self.assertEqual(left["mode"], "base")
        self.assertEqual(right["mode"], "full")


class HtmlTests(unittest.TestCase):
    def test_card_head_and_retrieval_off(self):
        self.assertIn("[ BASE ]", card_head_html("base"))
        self.assertIn("MODEL ONLY", card_head_html("base"))
        self.assertIn("[ FULL ]", card_head_html("full"))
        self.assertIn("retrieval", retrieval_bar_html(None, 0.30, retrieval_off=True))
        self.assertIn("off", retrieval_bar_html(None, 0.30, retrieval_off=True))
        html = retrieval_bar_html(0.74, 0.30)
        self.assertIn("retr ok", html)
        self.assertIn("width:74.0%", html)
        self.assertIn("left:30.0%", html)
        low = retrieval_bar_html(0.22, 0.30)
        self.assertIn("retr low", low)

    def test_chips_and_evidence(self):
        hits = [
            {
                "rank": 1, "source": "tldr", "title": "tar (tldr common)",
                "path": "tldr/common/tar.md", "dense": 0.74, "rrf": 0.05,
                "bm25": 6.2, "snippet": "tar -xzf archive",
            },
            {
                "rank": 2, "source": "man", "title": "tar(1)",
                "path": "man1/tar.1", "dense": 0.51, "rrf": 0.03,
                "bm25": 2.1, "snippet": "tar — tape archive",
            },
        ]
        chips = source_chips_html(hits)
        self.assertIn("TLDR", chips)
        self.assertIn("MAN", chips)
        ev = evidence_rows_html(hits)
        self.assertIn("cos 0.74", ev)
        self.assertIn("rrf 0.0500", ev)
        self.assertIn("bm25 6.200", ev)
        self.assertIn("tldr/common/tar.md", ev)
        off = source_chips_html([], retrieval_off=True)
        self.assertIn("retrieval off", off)

    def test_metrics_footer_reads_record(self):
        rec = {
            "latency_s": 1.8,
            "app_rss_mb": 420,
            "ollama_rss_mb": 2100,
            "total_mb": 2520,
            "docs": ["a", "b", "c"],
            "mode": "full",
            "abstained": False,
            "answer": "use `tar`",
        }
        html = metrics_html(rec)
        self.assertIn("1.8", html)
        self.assertIn("420", html)
        self.assertIn("2100", html)
        self.assertIn("2520", html)
        self.assertIn("docs <b>3</b>", html)
        self.assertNotIn("tok", html)

    def test_source_key_and_best_match(self):
        self.assertEqual(source_key("tldr"), "TLDR")
        self.assertEqual(source_key("man-pages"), "MAN")
        self.assertEqual(
            best_match_label([{"source": "tldr", "title": "ls(1)", "dense": 0.2}]),
            "tldr ls(1)",
        )


class HelperTests(unittest.TestCase):
    def test_split_reasoning_and_code(self):
        reason, answer = split_reasoning("Steps:\n1. x\n\nAnswer: use tar")
        self.assertIn("Steps:", reason)
        self.assertEqual(answer, "use tar")
        parts = list(split_code_blocks("do this\n```bash\ntar -xzf a.tar.gz\n```\ndone"))
        kinds = [k for k, _ in parts]
        self.assertIn("code", kinds)
        self.assertEqual(parts[1][1], "tar -xzf a.tar.gz")

    def test_max_dense(self):
        self.assertIsNone(max_dense([]))
        self.assertAlmostEqual(max_dense([{"dense": 0.1}, {"dense": 0.4}]), 0.4)

    def test_index_chunk_count(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(index_chunk_count(td), 0)
            (Path(td) / "chunks.jsonl").write_text('{"id":1}\n\n{"id":2}\n', encoding="utf-8")
            self.assertEqual(index_chunk_count(td), 2)

    def test_style_ships_offline(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn("/app/static/Inter-Regular.woff2", css)
        self.assertNotIn("fonts.googleapis.com", css)
        self.assertNotIn("cdn", css.lower())


class DigestTests(unittest.TestCase):
    def test_show_quant(self):
        fake = MagicMock()
        fake.ok = True
        fake.json.return_value = {
            "details": {"parameter_size": "2.0B", "quantization_level": "Q4_K_M"}
        }
        with patch("app.engine.requests.post", return_value=fake):
            self.assertEqual(model_digest("qwen3.5:2b"), "2.0B Q4_K_M")

    def test_blank_on_failure(self):
        with patch("app.engine.requests.post", side_effect=OSError("down")), patch(
            "app.engine.requests.get", side_effect=OSError("down")
        ):
            self.assertEqual(model_digest("qwen3.5:2b"), "")
        with patch("app.engine.requests.get", side_effect=OSError("down")):
            self.assertFalse(ollama_reachable())

    def test_tags_fallback(self):
        show = MagicMock()
        show.ok = False
        tags = MagicMock()
        tags.ok = True
        tags.json.return_value = {
            "models": [{
                "name": "qwen3.5:2b",
                "details": {"quantization_level": "Q8_0"},
            }]
        }
        with patch("app.engine.requests.post", return_value=show), patch(
            "app.engine.requests.get", return_value=tags
        ):
            self.assertEqual(model_digest("qwen3.5:2b"), "Q8_0")


class ImportPolarisTests(unittest.TestCase):
    def test_app_import_with_stub(self):
        st = MagicMock()
        st.cache_resource = lambda fn: fn
        st.cache_data = lambda *a, **k: (lambda fn: fn)
        with patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "False"}):
            import importlib
            import sys

            sys.modules["streamlit"] = st
            if "app.app" in sys.modules:
                del sys.modules["app.app"]
            mod = importlib.import_module("app.app")
        self.assertTrue(hasattr(mod, "run_polaris"))
        self.assertTrue(hasattr(mod, "run_legacy"))
        self.assertEqual(os.environ.get("ANONYMIZED_TELEMETRY"), "False")


if __name__ == "__main__":
    unittest.main()
