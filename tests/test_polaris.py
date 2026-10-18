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
    best_match_html,
    best_match_label,
    build_run_record,
    card_head_html,
    card_html,
    debug_rows_html,
    default_compare_modes,
    doc_label,
    evidence_block_html,
    evidence_panel_html,
    evidence_rows_html,
    empty_hero_html,
    evidence_summary_text,
    format_total_ram,
    header_band_html,
    idk_card_html,
    index_chunk_count,
    injected_css,
    max_dense,
    metrics_html,
    orbit_style_path,
    pair_compare_modes,
    polaris_on,
    ram_split_html,
    records_to_csv,
    records_to_json,
    retrieval_bar_html,
    source_chips_html,
    source_key,
    split_code_blocks,
    split_reasoning,
    style_path,
    waiting_html,
)
class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, question, k=5, tldr_lane=True):
        self.calls.append({"question": question, "k": k, "tldr_lane": tldr_lane})
        return list(self.hits)[:k]


class FlagTests(unittest.TestCase):
    def test_polaris_enabled_env(self):
        with patch.dict(os.environ, {"POLARIS_ENABLED": "0"}):
            self.assertFalse(polaris_on())
        with patch.dict(os.environ, {"POLARIS_ENABLED": "false"}):
            self.assertFalse(polaris_on())
        with patch.dict(os.environ, {"POLARIS_ENABLED": "1"}):
            self.assertTrue(polaris_on())
        with patch.dict(os.environ, {"POLARIS_ENABLED": "yes"}):
            self.assertTrue(polaris_on())

    def test_polaris_enabled_defaults_on(self):
        env = {k: v for k, v in os.environ.items() if k != "POLARIS_ENABLED"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(polaris_on())


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

    def test_docs_keep_source_only_hits(self):
        rec = build_run_record(
            {
                "ts": "t",
                "question": "q",
                "config": "base+RAG",
                "mode": "rag",
                "model": "qwen3.5:2b",
                "text": "x",
                "latency_s": 0.1,
                "app_rss_mb": 1,
                "ollama_rss_mb": 2,
                "total_mb": 3,
                "hits": [
                    {"source": "tldr", "title": "", "path": "", "text": "a", "dense": 0.4},
                    {"source": "", "title": "", "path": "man1/ls.1", "text": "b", "dense": 0.5},
                ],
                "weak": False,
                "threshold": 0.30,
                "k": 5,
                "git_commit": "abc",
            },
            index_path="rag/rag-index-v2",
        )
        self.assertEqual(rec["docs"], ["tldr", "man1/ls.1"])
        self.assertEqual(doc_label({"source": "man", "title": "", "path": ""}), "man")

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
        self.assertIn("idk-body", html)
        self.assertNotIn('class="idk"', html)
        self.assertIn("polaris-idk-flag", html)
        self.assertIn(ABSTAIN.splitlines()[0], html)
        self.assertIn("Please ask the lab technician", html)
        card = card_html(rec)
        self.assertIn("idk-card", card)
        self.assertIn("retr low", card)
        self.assertIn("best match:", card)
        self.assertIn("best match: tldr ls", card)
        self.assertNotIn("cosine 0.10 of 0.30 needed", card)
        self.assertIn("1 candidate checked", card)
        self.assertIn("abstain", card)
        self.assertNotIn("1 source · abstain", card.replace(" · ", " · "))
        self.assertIn("Evidence · 1 source", card)
        self.assertNotIn(" open", card.split("ev-box", 1)[1][:20])

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
        self.assertEqual(rows[0]["total_mb"], "300.0")
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

    def test_run_compare_clears_waiting_before_right_starts(self):
        st = MagicMock()
        st.cache_resource = lambda fn: fn
        st.cache_data = lambda *a, **k: (lambda fn: fn)
        sys.modules["streamlit"] = st
        import importlib

        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        app_mod = importlib.import_module("app.app")

        events = []

        class Slot:
            def markdown(self, *a, **k):
                events.append("waiting")

            def empty(self):
                events.append("clear_wait")

        class Col:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def empty(self):
                events.append("placeholder")
                return Slot()

        class Box:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def answer(question, mode, *a, **k):
            events.append(("start", mode))
            return {
                "mode": mode,
                "answer": "x",
                "abstained": False,
                "hits": [],
                "docs": [],
                "latency_s": 0.1,
                "app_rss_mb": 1,
                "ollama_rss_mb": 1,
                "total_mb": 2,
                "retrieval_max_dense": None,
                "threshold": 0.3,
            }

        def render(rec, key, align_slots=False):
            events.append(("render", rec["mode"]))

        with patch.object(app_mod.st, "container", return_value=Box()), patch.object(
            app_mod.st, "columns", return_value=(Col(), Col())
        ), patch.object(app_mod, "_answer_record", side_effect=answer), patch.object(
            app_mod, "_render_card", side_effect=render
        ):
            left, right = app_mod._run_compare("q", "base", "full", "m", None, "idx")

        self.assertEqual(left["mode"], "base")
        self.assertEqual(right["mode"], "full")
        self.assertLess(events.index(("start", "base")), events.index("clear_wait"))
        self.assertLess(events.index("clear_wait"), events.index(("start", "full")))
        self.assertEqual(
            [e for e in events if isinstance(e, tuple) and e[0] == "start"],
            [("start", "base"), ("start", "full")],
        )


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
        self.assertIn("left:74.0%", html)
        self.assertIn(">0.30<", html)
        self.assertIn(">0.74<", html)
        self.assertIn(">0<", html)
        self.assertIn(">1<", html)
        self.assertNotIn("threshold <b>", html)
        low = retrieval_bar_html(0.22, 0.30)
        self.assertIn("retr low", low)
        self.assertIn("width:22.0%", low)
        self.assertIn("left:22.0%", low)
        self.assertIn("left:30.0%", low)

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
        rec = {
            "mode": "full",
            "hits": hits,
            "docs": ["tar (tldr common)", "tar(1)"],
            "threshold": 0.30,
            "app_rss_mb": 420,
            "ollama_rss_mb": 2100,
            "total_mb": 2520,
            "latency_s": 1.8,
            "abstained": False,
            "answer": "use tar",
            "retrieval_max_dense": 0.74,
        }
        chips = source_chips_html(hits)
        self.assertIn("TLDR", chips)
        self.assertIn("MAN", chips)
        self.assertEqual(evidence_summary_text(rec), "Evidence · 2 sources · tldr tar")
        ev = evidence_rows_html(hits)
        self.assertIn("cosine 0.74", ev)
        self.assertNotIn("rrf", ev)
        self.assertNotIn("bm25", ev)
        self.assertNotIn("tldr/common/tar.md", ev)
        panel = evidence_panel_html(rec)
        self.assertIn("best match:", panel)
        self.assertIn("cosine 0.74 of 0.30", panel)
        self.assertNotIn("rrf 0.0500", panel.split("<details", 1)[0])
        self.assertNotIn("tldr/common/tar.md", panel.split("<details", 1)[0])
        self.assertIn("app <b>420</b> MB", panel)
        self.assertIn("ollama <b>2100</b> MB", panel)
        dbg = debug_rows_html(hits)
        self.assertIn("rrf 0.0500", dbg)
        self.assertIn("bm25 6.200", dbg)
        self.assertIn("tldr/common/tar.md", dbg)
        block = evidence_block_html(rec)
        self.assertIn("<summary>", block)
        self.assertIn("Evidence · 2 sources", block)
        self.assertIn("tldr", block)
        self.assertNotIn(" open", block.split("ev-box", 1)[1][:20])
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
        self.assertIn("2.5 GB", html)
        self.assertIn("total", html)
        self.assertIn("3 sources", html)
        self.assertNotIn("app <b>420</b>", html)
        self.assertNotIn("2100", html)
        self.assertNotIn("docs", html)
        self.assertNotIn("tok", html)
        self.assertNotIn("copy", html)
        self.assertEqual(format_total_ram(2520), "2.5 GB")
        split = ram_split_html(rec)
        self.assertIn("app <b>420</b> MB", split)
        self.assertIn("ollama <b>2100</b> MB", split)
        idk = dict(rec)
        idk["abstained"] = True
        idk["docs"] = ["a"]
        idk["hits"] = [{"rank": 1, "source": "tldr", "dense": 0.2}]
        idk["mode"] = "rag"
        idk_html = metrics_html(idk)
        self.assertIn("1 candidate checked", idk_html)
        self.assertIn("abstain", idk_html)
        self.assertNotIn("1 source", idk_html)

    def test_source_key_and_best_match(self):
        self.assertEqual(source_key("tldr"), "TLDR")
        self.assertEqual(source_key("man-pages"), "MAN")
        self.assertEqual(
            best_match_label([{"source": "tldr", "title": "ls(1)", "dense": 0.2}]),
            "tldr ls(1)",
        )
        html = best_match_html({
            "hits": [{"source": "tldr", "title": "ls - List files", "dense": 0.22}],
            "threshold": 0.30,
        })
        self.assertIn("tldr", html)
        self.assertIn("ls - List files", html)
        self.assertNotIn("0.22", html)
        self.assertNotIn("cosine", html)

    def test_waiting_is_solid_muted_not_dashed(self):
        html = waiting_html("full", "BASE")
        self.assertIn("waiting-card", html)
        self.assertIn("slot-empty", html)
        self.assertIn("waiting · runs after BASE finishes", html)
        self.assertNotIn("dashed", html)
        self.assertIn("retr off", html)
        self.assertIn("slot-line", html)
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(".card.waiting-card{border:var(--hair)", css.replace(" ", ""))
        self.assertIn("1px dashed var(--c-slate)", css)

    def test_compare_base_vs_full_slots(self):
        question = "how do I extract a .tar.gz file?"
        base = {
            "mode": "base",
            "question": question,
            "answer": "Use tar with -xzf.",
            "hits": [],
            "docs": [],
            "threshold": 0.30,
            "retrieval_max_dense": None,
            "latency_s": 1.2,
            "app_rss_mb": 400,
            "ollama_rss_mb": 1800,
            "total_mb": 2200,
            "abstained": False,
        }
        full = {
            "mode": "full",
            "question": question,
            "answer": "Use tar.\n```bash\ntar -xzf file.tar.gz\n```",
            "hits": [{
                "rank": 1, "source": "tldr", "title": "tar (tldr common)",
                "path": "tldr/common/tar.md", "dense": 0.74, "rrf": 0.05,
                "bm25": 6.2, "snippet": "tar -xzf",
            }],
            "docs": ["tar (tldr common)"],
            "threshold": 0.30,
            "retrieval_max_dense": 0.74,
            "latency_s": 1.8,
            "app_rss_mb": 420,
            "ollama_rss_mb": 2100,
            "total_mb": 2520,
            "abstained": False,
        }
        left = card_html(base, align_slots=True)
        right = card_html(full, align_slots=True)
        self.assertIn("[ BASE ]", left)
        self.assertIn("MODEL ONLY", left)
        self.assertIn("[ FULL ]", right)
        self.assertIn("RAG + COT", right)
        self.assertIn("slot-empty ev", left)
        self.assertIn("retr off", left)
        self.assertIn("Evidence · 1 source", right)
        self.assertIn("retr ok", right)
        self.assertIn("slot-empty best", left)
        self.assertIn("slot-empty best", right)
        self.assertNotIn("idk-card", left)
        self.assertNotIn("idk-card", right)
        wait = waiting_html("full", "BASE")
        self.assertIn("waiting-card", wait)
        self.assertIn("slot-empty ev", wait)
        self.assertIn("[ FULL ]", wait)
        self.assertNotIn("dashed", wait)
        css = style_path().read_text(encoding="utf-8")
        compact = css.replace(" ", "").replace("\n", "")
        self.assertIn(".slot-empty.ev,.slot-empty.best{", compact)
        self.assertIn(".slot-empty.ev::after,.slot-empty.best::after{", compact)
        self.assertIn("height:1px;background:var(--c-slate-dim)", compact)


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
        self.assertIn("--fs-base:15px", css)
        self.assertIn("1px dashed var(--c-slate)", css)
        self.assertNotIn('content:"▸', css)
        self.assertIn("@keyframes star-float", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn(".idk-star svg{width:20px;height:20px;fill:var(--c-slate-dim);animation:none}", css.replace("\n", ""))
        self.assertNotIn(".waiting{border:1px dashed", css.replace(" ", ""))
        orbit = orbit_style_path().read_text(encoding="utf-8")
        self.assertIn("@keyframes star-orbit", orbit)
        self.assertIn("@keyframes star-orbit-slim", orbit)
        self.assertNotIn("cdn", orbit.lower())
        all_css = injected_css()
        self.assertIn("star-orbit 7.2s", all_css)
        band = header_band_html(slim=True)
        self.assertIn('class="orbit"', band)
        self.assertIn('class="float"', band)
        self.assertIn("polaris", band)
        hero = empty_hero_html()
        self.assertNotIn('class="orbit"', hero)


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
