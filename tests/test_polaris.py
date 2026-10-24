#!/usr/bin/env python3
"""Polaris UI helpers. No live Ollama."""
import csv
import http.client
import importlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.prompt import ABSTAIN
from app.engine import (
    OllamaInfo,
    WEAK_DENSE,
    answer_question,
    build_messages,
    config_id,
    is_weak,
    model_digest,
    ollama_info,
    ollama_chat_body,
    ollama_reachable,
    warm_model,
)
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
    STILL_WORKING,
    CancellableChat,
    ChatCancelled,
    compare_compact_html,
    default_compare_model_keys,
    model_key_choices,
    model_mode_choices,
    mode_from_name,
    mode_name,
    pair_compare_config_modes,
    pair_compare_model_keys,
    preview_text,
    running_card_html,
    ENGLISH_NOTE,
    MODE_HINTS,
    composer_note_html,
    connection_html,
    flags_html,
    mode_menu_css,
    network_online,
    SKY_LEVELS,
    side_logo_html,
    sky_html,
    sky_level,
    star_defs,
    stopped_html,
    thinking_html,
    thinking_label,
)


def _cache_stub(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


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


class SessionStartupTests(unittest.TestCase):
    def test_compare_defaults_off_and_git_is_read_once(self):
        app_mod = importlib.import_module("app.app")
        fake_st = MagicMock()
        fake_st.session_state = {}
        with patch.object(app_mod, "st", fake_st), patch.object(
            app_mod,
            "git_commit",
            return_value="abc1234",
        ) as commit:
            app_mod._init_session()
            app_mod._init_session()
            self.assertEqual(app_mod._session_git_commit(), "abc1234")
        self.assertFalse(fake_st.session_state["compare"])
        commit.assert_called_once_with()

    def test_failed_warmup_retries_and_success_is_cached(self):
        app_mod = importlib.import_module("app.app")
        warmed = set()
        with patch.object(app_mod, "_warmed_models", return_value=warmed), patch.object(
            app_mod,
            "warm_model",
            side_effect=[False, True],
        ) as warm:
            self.assertFalse(app_mod._warm_model_at_startup("model", "30m"))
            self.assertTrue(app_mod._warm_model_at_startup("model", "30m"))
            self.assertTrue(app_mod._warm_model_at_startup("model", "30m"))
        self.assertEqual(warm.call_count, 2)


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
        self.assertIn("No fix on this one", html)
        self.assertIn("idk-body", html)
        self.assertNotIn('class="idk"', html)
        self.assertIn("polaris-idk-flag", html)
        self.assertIn("I don&#x27;t know.", html)
        self.assertNotIn("Steps:", html)
        self.assertIn("<code>man &lt;command&gt;</code>", html)
        self.assertIn("Please ask the lab technician", html)
        card = card_html(rec)
        self.assertIn("idk-card", card)
        self.assertIn("retr low", card)
        self.assertIn("Best match:", card)
        self.assertIn("<code>tldr ls</code>", card)
        self.assertNotIn("cosine 0.10 of 0.30 needed", card)
        self.assertIn("<b>1</b> candidate checked", card)
        self.assertIn("abstained", card)
        self.assertNotIn(" · ", card.split('class="metrics"', 1)[1])
        self.assertIn('<span class="ev-title">Evidence</span> <span class="count">1 source</span>', card)
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
                "context_turns": 2,
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
        self.assertEqual(rows[0]["context_turns"], "2")
        self.assertEqual(rows[0]["total_mb"], "300.0")
        self.assertNotIn("answers.csv", csv_text)
        payload = json.loads(records_to_json([rec]))
        self.assertEqual(payload[0]["answer"], "hi")
        self.assertEqual(payload[0]["context_turns"], 2)
        self.assertEqual(payload[0]["index_path"], "/tmp/index")


class CompareTests(unittest.TestCase):
    def test_defaults_base_vs_full(self):
        self.assertEqual(default_compare_modes(), ("base", "full"))

    def test_pair_must_differ(self):
        self.assertEqual(pair_compare_modes("base", "base"), ("base", "cot"))
        self.assertEqual(pair_compare_modes("full", "full"), ("full", "base"))
        self.assertEqual(pair_compare_modes("base", "full"), ("base", "full"))

    def test_distilled_model_choices_are_hidden_when_unset(self):
        self.assertEqual(model_key_choices(""), ("base",))
        self.assertEqual(default_compare_model_keys(""), ("base", "base"))
        self.assertEqual(
            pair_compare_model_keys("base", "distilled", ""),
            ("base", "base"),
        )

    def test_distilled_compare_allows_the_same_mode(self):
        self.assertEqual(
            default_compare_model_keys("student:latest"),
            ("base", "distilled"),
        )
        self.assertEqual(
            pair_compare_model_keys("base", "distilled", "student:latest"),
            ("base", "distilled"),
        )
        self.assertEqual(
            pair_compare_config_modes("full", "full", "base", "distilled"),
            ("full", "full"),
        )

    def test_distilled_exposes_only_full_mode(self):
        self.assertEqual(model_mode_choices("base"), ("base", "cot", "rag", "full"))
        self.assertEqual(model_mode_choices("distilled"), ("full",))
        self.assertEqual(
            pair_compare_config_modes("base", "rag", "base", "distilled"),
            ("base", "full"),
        )
        self.assertEqual(
            pair_compare_config_modes("base", "cot", "distilled", "distilled"),
            ("full", "full"),
        )

    def test_run_compare_clears_waiting_before_right_starts(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        st.session_state.messages = [{"role": "user", "content": "q"}]
        sys.modules["streamlit"] = st

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

        def answer(question, mode, model, *a, **k):
            events.append(("start", mode, model, k.get("model_key")))
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
            events.append(("render", rec["mode"], key))

        with patch.object(app_mod.st, "container", return_value=Box()), patch.object(
            app_mod.st, "columns", return_value=(Col(), Col())
        ), patch.object(app_mod, "_answer_record", side_effect=answer), patch.object(
            app_mod, "_render_card", side_effect=render
        ), patch.object(app_mod, "DISTILLED_MODEL", "student:latest"):
            left, right = app_mod._run_compare(
                "q",
                "base",
                "full",
                "qwen3.5:2b",
                None,
                "idx",
                left_model_key="base",
                right_model_key="distilled",
            )

        self.assertEqual(left["mode"], "base")
        self.assertEqual(right["mode"], "full")
        left_start = ("start", "base", "qwen3.5:2b", "base")
        right_start = ("start", "full", "student:latest", "distilled")
        self.assertLess(events.index(left_start), events.index("clear_wait"))
        self.assertLess(events.index("clear_wait"), events.index(right_start))
        self.assertEqual(
            [e for e in events if isinstance(e, tuple) and e[0] == "start"],
            [left_start, right_start],
        )
        self.assertIn(("render", "base", "answer-h1-l"), events)
        self.assertIn(("render", "full", "answer-h1-r"), events)


class HtmlTests(unittest.TestCase):
    def test_card_head_and_retrieval_off(self):
        self.assertIn('<span class="mode">BASE</span>', card_head_html("base"))
        self.assertIn("Plain model, no documents", card_head_html("base"))
        self.assertIn('<span class="mode">+RAG+CoT</span>', card_head_html("full"))
        self.assertNotIn("[", card_head_html("full"))
        self.assertIn("No retrieval", retrieval_bar_html(None, 0.30, retrieval_off=True))
        self.assertIn(
            "No retrieval in BASE",
            retrieval_bar_html(None, 0.30, retrieval_off=True, mode="base"),
        )
        self.assertNotIn("retrieval off", retrieval_bar_html(None, 0.30, retrieval_off=True))
        html = retrieval_bar_html(0.74, 0.30)
        self.assertIn("retr ok", html)
        self.assertIn("width:74.0%", html)
        self.assertIn("left:30.0%", html)
        self.assertIn("<b>0.74</b> clears 0.30", html)
        self.assertIn('role="meter"', html)
        self.assertNotIn('class="scale"', html)
        self.assertNotIn('class="knob"', html)
        low = retrieval_bar_html(0.22, 0.30)
        self.assertIn("retr low", low)
        self.assertIn("width:22.0%", low)
        self.assertIn("left:30.0%", low)
        self.assertIn("<b>0.22</b> below 0.30", low)

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
        self.assertEqual(evidence_summary_text(rec), "Evidence, 2 sources, best tldr tar")
        ev = evidence_rows_html(hits)
        self.assertIn("cosine 0.74", ev)
        self.assertNotIn("rrf", ev)
        self.assertNotIn("bm25", ev)
        self.assertNotIn("tldr/common/tar.md", ev)
        panel = evidence_panel_html(rec)
        self.assertIn("Best match", panel)
        self.assertIn("cosine 0.74, threshold 0.30", panel)
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
        self.assertIn('<span class="count">2 sources</span>', block)
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
        self.assertIn("<b>3</b> sources", html)
        self.assertNotIn(" · ", html)
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
        self.assertIn("<b>1</b> candidate checked", idk_html)
        self.assertIn("abstained", idk_html)
        self.assertNotIn("source", idk_html)

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
        self.assertIn("Starts when BASE finishes", html)
        self.assertIn("thinking queued", html)
        self.assertNotIn("dashed", html)
        self.assertIn("retr off", html)
        self.assertIn("No sources yet", html)
        self.assertIn("No retrieval yet", html)
        self.assertNotIn("slot-line", html)
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(".card.waiting-card{border:1pxsolidvar(--c-line)", css.replace(" ", ""))
        self.assertIn(".thinking.queued .dots i{animation:none", css)
        self.assertIn("1px dashed var(--c-idk)", css)

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
        self.assertIn('<span class="mode">BASE</span>', left)
        self.assertIn("Plain model, no documents", left)
        self.assertIn('<span class="mode">+RAG+CoT</span>', right)
        self.assertIn("Retrieved lab docs, step-by-step prompt", right)
        self.assertIn("slot-empty ev", left)
        self.assertIn("No sources in BASE", left)
        self.assertIn("No retrieval in BASE", left)
        self.assertIn("retr off", left)
        self.assertIn('<span class="count">1 source</span>', right)
        self.assertIn("retr ok", right)
        self.assertIn("slot-empty best", left)
        self.assertIn("slot-empty best", right)
        self.assertNotIn("idk-card", left)
        self.assertNotIn("idk-card", right)
        wait = waiting_html("full", "BASE")
        self.assertIn("waiting-card", wait)
        self.assertIn("slot-empty ev", wait)
        self.assertIn("No sources yet", wait)
        self.assertIn("No retrieval yet", wait)
        self.assertIn('<span class="mode">+RAG+CoT</span>', wait)
        self.assertNotIn("dashed", wait)
        css = style_path().read_text(encoding="utf-8")
        compact = css.replace(" ", "").replace("\n", "")
        self.assertIn(".slot-empty{min-height:24px", compact)
        self.assertNotIn(".slot-empty.ev::after", compact)
        self.assertIn(".ev-box.best-src{", compact)
        self.assertNotIn("var(--c-accent)", compact.split(".ev-box{", 1)[1].split(".retr{", 1)[0])


class LoaderAndNamingTests(unittest.TestCase):
    def test_thinking_loader_is_css_only_constellation(self):
        html = thinking_html("Searching the lab docs")
        self.assertIn('role="status"', html)
        self.assertEqual(html.count("<i></i>"), 9)
        self.assertIn("Searching the lab docs", html)
        self.assertIn(STILL_WORKING, html)
        self.assertNotIn("<script", html)
        css = style_path().read_text(encoding="utf-8")
        block = css.split("/* ---------- loader", 1)[1].split("/* ---------- sidebar", 1)[0]
        for prop in ("width:", "height:", "top:", "left:", "margin"):
            for frames in block.split("@keyframes")[1:]:
                body = frames.split("}}", 1)[0]
                self.assertNotIn(prop, body, f"loader keyframes animate {prop}")

    def test_thinking_label_follows_mode(self):
        self.assertEqual(thinking_label("full"), "Generating an answer from the lab docs")
        self.assertEqual(thinking_label("rag"), "Generating an answer from the lab docs")
        self.assertEqual(thinking_label("cot"), "Generating an answer")
        self.assertEqual(thinking_label("base"), "Generating an answer")
        self.assertEqual(thinking_label("full", loading_index=True), "Loading the lab docs index")
        self.assertIn("Stopped", stopped_html())
        card = running_card_html("cot", thinking_label("cot"))
        self.assertIn("running-card", card)
        self.assertIn('<span class="mode">+CoT</span>', card)

    def test_loader_text_is_neutral(self):
        banned = (
            "think", "reason", "grepping", "man page", "untarring", "piping",
            "charting", "stargazing", "checking permissions",
        )
        css = style_path().read_text(encoding="utf-8")
        for mode in ("base", "cot", "rag", "full"):
            label = thinking_label(mode)
            html = thinking_html(label)
            text = re.sub(r"<[^>]+>", " ", html).lower()
            for word in banned:
                self.assertNotIn(word, text, f"{mode} loader says {word!r}")
            self.assertEqual(html.count('class="thinking-label"'), 1)
            self.assertIn(f'<span class="thinking-label">{label}…</span>', html)
            self.assertNotIn('aria-hidden="true"><span class="thinking-label"', html)
        for mode in ("base", "cot"):
            self.assertNotRegex(thinking_label(mode).lower(), "doc|search|retriev|index")
        self.assertNotIn("polaris-cycle", css)
        self.assertNotIn("polaris-shimmer", css)
        self.assertNotIn(".thinking-label > span", css)

    def test_base_label_says_plain_model_no_documents(self):
        head = card_head_html("base")
        self.assertIn('<span class="mode">BASE</span><span class="sub">Plain model, no documents</span>', head)
        self.assertEqual(mode_name("base"), "BASE")
        names = {mode: mode_name(mode) for mode in ("base", "cot", "rag", "full")}
        self.assertEqual(names, {"base": "BASE", "cot": "+CoT", "rag": "+RAG", "full": "+RAG+CoT"})
        for mode, name in names.items():
            self.assertEqual(config_id("base", mode), "base" if name == "BASE" else "base" + name)
        for mode in ("base", "cot"):
            self.assertIn("no documents", MODE_HINTS[mode])
        for mode in ("rag", "full"):
            self.assertIn("Retrieved lab docs", MODE_HINTS[mode])
        for hint in MODE_HINTS.values():
            self.assertNotRegex(hint.lower(), "think|reason")
        menu = mode_menu_css("qwen3.5:2b", ["base", "cot", "rag", "full"])
        self.assertIn('nth-child(1)::after{content:"Plain model, no documents"}', menu)

    def test_ambient_motion_pauses_while_generating(self):
        self.assertIn("polaris-generating", flags_html(1, generating=True))
        self.assertNotIn("polaris-generating", flags_html(1))
        self.assertEqual(
            flags_html(2, daylight=True, compare=True, generating=True),
            '<div class="polaris-flags polaris-sky-2 polaris-daylight polaris-compare polaris-generating"></div>',
        )
        css = style_path().read_text(encoding="utf-8")
        rules = [r for r in css.split("}") if ":root:has(.polaris-generating)" in r]
        self.assertEqual(len(rules), 1)
        selector, body = rules[0].rsplit("{", 1)
        selector = selector.rsplit("*/", 1)[-1].strip()
        self.assertEqual(body, "animation-play-state:paused!important")
        # Only the star and the sky: the loader and the one-shot entrances must keep running.
        self.assertEqual(selector, ":root:has(.polaris-generating) :is(.band *,.sky)")

    def test_english_only_note_sits_in_the_composer(self):
        self.assertEqual(ENGLISH_NOTE, "English questions only")
        html = composer_note_html(connection_html(False, True))
        self.assertTrue(html.startswith('<div class="composer-conn"><span class="conn offline"'))
        self.assertIn('<span class="composer-note">English questions only</span>', html)
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(".composer-note{", css)

    def test_mode_names_round_trip(self):
        for mode in ("base", "cot", "rag", "full"):
            self.assertEqual(mode_from_name(mode_name(mode)), mode)
        self.assertEqual(mode_from_name("nope", "rag"), "rag")

    def test_connection_pill_carries_text_not_only_color(self):
        self.assertIn(">Offline<", connection_html(False, True))
        self.assertIn("conn offline", connection_html(False, True))
        self.assertIn(">Online<", connection_html(True, True))
        down = connection_html(True, False)
        self.assertIn("conn down", down)
        self.assertIn("Ollama stopped", down)
        self.assertIn("ollama serve", down)

    def test_network_online_reads_routes_without_sending(self):
        header = "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "route").write_text(header + "eth0\t00000000\t0102A8C0\t0003\t0\t0\t100\t00000000\n")
            self.assertTrue(network_online(td))
            Path(td, "route").write_text(header + "eth0\t0002A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\n")
            self.assertFalse(network_online(td))
            Path(td, "route").write_text(header)
            Path(td, "ipv6_route").write_text(
                "00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
                "fe800000000000000000000000000001 00000400 00000001 00000000 00000003 wlan0\n"
            )
            self.assertTrue(network_online(td))

    def test_mode_menu_css_names_model_and_hints(self):
        css = mode_menu_css('qwen"3.5:2b', ["base", "cot", "rag", "full"])
        self.assertIn('--polaris-model:"qwen\\"3.5:2b"', css)
        self.assertIn('nth-child(4)::after{content:"Retrieved lab docs, step-by-step prompt"}', css)
        self.assertTrue(css.startswith("<style>") and css.endswith("</style>"))

    def test_sidebar_logo_and_star_ids(self):
        idle = side_logo_html()
        self.assertIn("band side calm wrap-on", idle)
        self.assertNotIn("is-wrapping", idle)
        self.assertIn("is-wrapping", side_logo_html(play=True))
        self.assertIn('id="polaris-star-lux-side"', idle)
        self.assertIn("url(#polaris-star-lux-side)", idle)
        self.assertIn('id="polaris-star-lux-hero"', empty_hero_html())
        self.assertIn('id="polaris-star-lux-slim"', header_band_html(slim=True))
        self.assertNotEqual(star_defs("a"), star_defs("b"))

    def test_html_avoids_inline_custom_properties(self):
        # Streamlit's sanitizer strips inline custom properties, which would void any CSS that reads them.
        for html in (thinking_html("Searching the lab docs"), side_logo_html(True), empty_hero_html(), stopped_html(), sky_html()):
            self.assertNotIn('style="--', html)

    def test_sky_is_one_raw_html_line_that_fades_per_question(self):
        html = sky_html()
        self.assertNotIn("\n", html)
        self.assertNotIn("<pre", html)
        self.assertTrue(html.startswith('<div class="sky" aria-hidden="true"><div class="sky-field">'))
        self.assertEqual(html.count("<b>"), 7)
        self.assertIs(sky_html(), html)
        self.assertEqual([sky_level(n) for n in (-1, 0, 1, 2, 3, 4, 9)], [0, 0, 1, 2, 3, 4, 4])
        css = style_path().read_text(encoding="utf-8")
        for level, opacity in enumerate(SKY_LEVELS[1:], 1):
            rule = f':root:has(.polaris-sky-{level}) [data-testid="stMarkdownContainer"] .sky{{opacity:{opacity:g}'
            self.assertIn(rule.replace("opacity:0.", "opacity:."), css)
        self.assertIn("visibility:hidden", css.split(".polaris-sky-4", 1)[1].split("}", 1)[0])

    def test_status_widget_stays_hidden(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn('[data-testid="stStatusWidget"]{display:none!important}', css)

    def test_compare_preview_skips_reasoning_and_fences(self):
        text = "Think first.\n\nAnswer: Run this:\n```bash\ntar -xzf a.tar.gz\n```\nThen `ls` it."
        self.assertEqual(preview_text(text), "Run this: Then ls it.")
        html = compare_compact_html({"mode": "base", "answer": "Use `tar`."}, {"mode": "cot", "answer": text})
        self.assertNotIn("```", html)
        self.assertNotIn("Think first", html)


class _Interrupted(Exception):
    pass


class _SlowOllama(BaseHTTPRequestHandler):
    delay = 0.0
    bodies = []
    request_started = threading.Event()
    reply = {"message": {"role": "assistant", "content": "use tar -xzf"}}

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        type(self).bodies.append(json.loads(self.rfile.read(n)))
        type(self).request_started.set()
        time.sleep(type(self).delay)
        out = json.dumps(type(self).reply).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except OSError:
            pass

    def log_message(self, *a):
        pass


class StopTests(unittest.TestCase):
    def setUp(self):
        _SlowOllama.bodies = []
        _SlowOllama.request_started.clear()
        _SlowOllama.reply = {
            "message": {"role": "assistant", "content": "use tar -xzf"}
        }
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowOllama)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_sends_temperature_and_seed_in_the_engine_body(self):
        _SlowOllama.delay = 0
        chat = CancellableChat(self.host)
        with tempfile.TemporaryDirectory() as td:
            out = answer_question(
                "hi",
                "base",
                "qwen3.5:2b",
                chat_fn=chat,
                log_path=Path(td) / "answers.csv",
                temperature=0,
                seed=42,
                keep_alive="30m",
                log_commit=chat.commit,
            )
        self.assertEqual(out["text"], "use tar -xzf")
        self.assertEqual(
            _SlowOllama.bodies[0],
            ollama_chat_body(
                build_messages("hi", [], "base"),
                "qwen3.5:2b",
                temperature=0,
                seed=42,
                keep_alive="30m",
            ),
        )

    def test_warm_model_makes_a_real_non_streaming_call(self):
        _SlowOllama.delay = 0
        self.assertTrue(warm_model("qwen3.5:2b", host=self.host, keep_alive="45m"))
        self.assertEqual(len(_SlowOllama.bodies), 1)
        body = _SlowOllama.bodies[0]
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["keep_alive"], "45m")
        self.assertFalse(body["stream"])

    def test_answer_returns_and_logs_ollama_stats(self):
        _SlowOllama.reply = {
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 40,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 100_000_000,
        }
        chat = CancellableChat(self.host)
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            result = answer_question(
                "hi",
                "base",
                "m",
                chat_fn=chat,
                log_path=log,
                log_commit=chat.commit,
                model_info=OllamaInfo(
                    version="0.12.3",
                    model_digest="2.0B Q4_K_M",
                ),
            )
            with open(log, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
        self.assertEqual(result["eval_count"], 20)
        self.assertEqual(result["eval_duration"], 2_000_000_000)
        self.assertEqual(result["prompt_eval_count"], 40)
        self.assertEqual(result["prompt_eval_duration"], 500_000_000)
        self.assertEqual(result["load_duration"], 100_000_000)
        self.assertEqual(result["tok_s"], 10.0)
        self.assertEqual(result["gen_s"], 2.0)
        self.assertEqual(result["prompt_s"], 0.5)
        self.assertEqual(result["load_s"], 0.1)
        self.assertEqual(result["ollama_version"], "0.12.3")
        self.assertEqual(result["model_digest"], "2.0B Q4_K_M")
        record = build_run_record(result)
        self.assertEqual(record["tok_s"], 10.0)
        self.assertEqual(record["eval_count"], 20)
        self.assertEqual(record["ollama_version"], "0.12.3")
        self.assertEqual(row["tok_s"], "10.0000")
        self.assertEqual(row["gen_s"], "2.0000")
        self.assertEqual(row["prompt_s"], "0.5000")
        self.assertEqual(row["load_s"], "0.1000")
        self.assertEqual(row["ollama_version"], "0.12.3")
        self.assertEqual(row["model_digest"], "2.0B Q4_K_M")

    def test_cancel_aborts_a_slow_answer(self):
        _SlowOllama.delay = 5
        chat = CancellableChat(self.host)
        box = {}

        def run():
            try:
                box["out"] = chat([{"role": "user", "content": "hi"}], "m")
            except BaseException as e:
                box["err"] = e

        worker = threading.Thread(target=run)
        t0 = time.perf_counter()
        worker.start()
        self.assertTrue(_SlowOllama.request_started.wait(2))
        chat.cancel()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.perf_counter() - t0, 2)
        self.assertIsInstance(box.get("err"), ChatCancelled)

    def test_cancelled_slow_answer_writes_no_csv(self):
        _SlowOllama.delay = 5
        chat = CancellableChat(self.host)
        box = {}
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"

            def run():
                try:
                    box["out"] = answer_question(
                        "hi",
                        "base",
                        "m",
                        chat_fn=chat,
                        log_path=log,
                        log_commit=chat.commit,
                    )
                except BaseException as e:
                    box["err"] = e

            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(_SlowOllama.request_started.wait(2))
            chat.cancel()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(box.get("err"), ChatCancelled)
            self.assertFalse(log.exists())

    def test_cancelled_weak_rag_abstain_writes_no_csv(self):
        chat = CancellableChat(self.host)
        chat.cancel()
        hits = [{
            "id": "weak",
            "title": "unrelated",
            "text": "nothing",
            "source": "tldr",
            "path": "x.md",
            "rrf": 1 / 60,
            "dense": 0.10,
        }]
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            with self.assertRaises(ChatCancelled):
                answer_question(
                    "where is the password?",
                    "full",
                    "m",
                    retriever=FakeRetriever(hits),
                    chat_fn=chat,
                    log_path=log,
                    log_commit=chat.commit,
                )
            self.assertFalse(log.exists())
        self.assertEqual(_SlowOllama.bodies, [])

    def test_cancel_closes_after_socket_shutdown(self):
        events = []

        class Sock:
            def shutdown(self, how):
                events.append(("shutdown", how))

        class Conn:
            sock = Sock()

            def close(self):
                events.append(("close", None))

        chat = CancellableChat(self.host)
        chat._conn = Conn()
        chat.cancel()
        self.assertEqual(events, [("shutdown", socket.SHUT_RDWR), ("close", None)])

    def test_cancel_while_connecting_sends_no_request(self):
        connected = threading.Event()
        release = threading.Event()
        calls = []

        class Conn:
            sock = None

            def __init__(self, *args, **kwargs):
                pass

            def connect(self):
                connected.set()
                release.wait(2)

            def request(self, *args, **kwargs):
                calls.append("request")

            def close(self):
                calls.append("close")

        chat = CancellableChat(self.host)
        box = {}

        def run():
            try:
                chat([{"role": "user", "content": "hi"}], "m")
            except BaseException as e:
                box["err"] = e

        with patch("app.polaris.http.client.HTTPConnection", Conn):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(connected.wait(2))
            chat.cancel()
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("err"), ChatCancelled)
        self.assertNotIn("request", calls)
        self.assertIn("close", calls)

    def test_cancel_before_request_cannot_reconnect(self):
        request_entered = threading.Event()
        release = threading.Event()
        calls = []

        class Sock:
            def shutdown(self, how):
                calls.append(("shutdown", how))

        class Conn:
            auto_open = 1
            sock = None

            def __init__(self, *args, **kwargs):
                pass

            def connect(self):
                self.sock = Sock()

            def request(self, *args, **kwargs):
                request_entered.set()
                release.wait(2)
                if self.sock is None and self.auto_open:
                    calls.append(("reconnect", None))
                    self.connect()
                if self.sock is None:
                    raise http.client.NotConnected()

            def close(self):
                self.sock = None
                calls.append(("close", None))

        chat = CancellableChat(self.host)
        box = {}

        def run():
            try:
                chat([{"role": "user", "content": "hi"}], "m")
            except BaseException as e:
                box["err"] = e

        with patch("app.polaris.http.client.HTTPConnection", Conn):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(request_entered.wait(2))
            chat.cancel()
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(box.get("err"), ChatCancelled)
        self.assertNotIn(("reconnect", None), calls)

    def test_unreachable_ollama_is_a_requests_error(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
        with self.assertRaises(requests.RequestException):
            CancellableChat(dead, timeout=2)([{"role": "user", "content": "hi"}], "m")

    def test_interrupt_while_waiting_cancels_the_request(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        sys.modules["streamlit"] = st

        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        app_mod = importlib.import_module("app.app")
        seen = {}

        def work(chat):
            seen["chat"] = chat
            while not chat.cancelled:
                time.sleep(0.05)
            raise ChatCancelled()

        beats = {"n": 0}

        def heartbeat():
            beats["n"] += 1
            if beats["n"] == 2:
                raise _Interrupted("rerun requested")

        with self.assertRaises(_Interrupted):
            app_mod._run_stoppable(work, heartbeat, tick=0.05)
        self.assertTrue(seen["chat"].cancelled)
        self.assertEqual(app_mod._run_stoppable(lambda chat: 42, heartbeat), 42)

    def test_answer_record_catches_chat_cancelled(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        sys.modules["streamlit"] = st

        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        app_mod = importlib.import_module("app.app")
        slot = MagicMock()
        with patch.object(app_mod, "_run_stoppable", side_effect=ChatCancelled):
            rec = app_mod._answer_record(
                "q",
                "base",
                "m",
                None,
                "idx",
                slot=slot,
            )
        self.assertIsNone(rec)
        self.assertIs(st.session_state.last_cancelled, True)
        slot.empty.assert_called_once()

    def test_loader_is_sent_once_while_the_answer_runs(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        st.session_state = MagicMock()
        st.session_state.run_records = []
        sys.modules["streamlit"] = st
        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        app_mod = importlib.import_module("app.app")
        slot = MagicMock()

        def slow_answer(question, mode, model, **kw):
            time.sleep(0.7)
            return {"mode": mode, "text": "use tar", "hits": [], "weak": False, "latency_s": 0.7}

        with patch.object(app_mod, "answer_question", side_effect=slow_answer):
            rec = app_mod._answer_record("q", "base", "m", None, "idx", slot=slot)
        self.assertEqual(rec["answer"], "use tar")
        self.assertEqual(slot.markdown.call_count, 1)
        self.assertIn("Generating an answer…", slot.markdown.call_args.args[0])
        stop_checks = [c for c in st.session_state.get.call_args_list if c.args == ("stop-gen",)]
        self.assertGreaterEqual(len(stop_checks), 2)


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
        self.assertIn("/app/static/UbuntuSans-Variable-Latin.woff2", css)
        self.assertTrue((style_path().parent / "static" / "UbuntuSans-Variable-Latin.woff2").is_file())
        self.assertTrue((style_path().parent / "static" / "UbuntuSans-UFL.txt").is_file())
        self.assertNotIn("Inter", css)
        self.assertNotIn("fonts.googleapis.com", css)
        self.assertNotIn("cdn", css.lower())
        self.assertIn("--fs-body:16px", css)
        self.assertIn("1px dashed var(--c-idk)", css)
        self.assertNotIn('content:"▸', css)
        self.assertNotIn("transition:all", css.replace(" ", ""))
        self.assertIn("@keyframes star-breathe", css)
        self.assertIn("@keyframes polaris-breathe", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn(".idk-star svg{width:18px;height:18px;fill:none;stroke:var(--c-idk);stroke-width:1.5;animation:none}", css.replace("\n", ""))
        self.assertNotIn(".waiting{border:1px dashed", css.replace(" ", ""))
        orbit = orbit_style_path().read_text(encoding="utf-8")
        self.assertIn("@keyframes star-wrap-path", orbit)
        self.assertIn("@keyframes star-depth-ink", orbit)
        self.assertIn("offset-path:path(", orbit.replace(" ", ""))
        self.assertNotIn("cdn", orbit.lower())
        all_css = injected_css()
        self.assertIn("star-wrap-path 4.4s linear", all_css)
        self.assertIn("@keyframes star-wrap-path", all_css)
        self.assertIn("@keyframes star-wrap-body", all_css)
        self.assertIn("@keyframes star-depth-ink", all_css)
        self.assertIn("@keyframes star-depth-base", all_css)
        self.assertIn("@keyframes star-halo", all_css)
        self.assertIn("@keyframes star-trail", all_css)
        self.assertIn("@keyframes star-trail-spark", all_css)
        self.assertIn("prefers-reduced-motion", all_css)
        idle = header_band_html(animate=True)
        self.assertIn("wrap-on", idle)
        self.assertIn("is-wrapping", idle)
        self.assertIn('class="base"', idle)
        self.assertIn('class="ink"', idle)
        self.assertIn("polaris", idle)
        self.assertNotIn("slim", idle)
        band = header_band_html(slim=True)
        self.assertIn("band slim rest", band)
        self.assertNotIn("wrap-on", band)
        self.assertNotIn("is-wrapping", band)
        self.assertIn("mark rest", band)
        self.assertIn('class="base"', band)
        self.assertIn('class="ink"', band)
        hero = empty_hero_html()
        self.assertNotIn('class="orbit"', hero)
        self.assertEqual(hero.count('class="band'), 1)
        self.assertIn("wrap-on", hero)
        self.assertIn("Offline help for the Linux lab", hero)

    def test_star_motion_v3_occlusion_contract(self):
        idle = header_band_html(animate=True)
        self.assertEqual(idle.count('class="base"'), 1)
        self.assertEqual(idle.count('class="ink"'), 1)
        self.assertIn('<span class="base" aria-hidden="true">polaris</span>', idle)
        self.assertIn('<span class="ink">polaris</span>', idle)
        self.assertGreaterEqual(idle.count("<i></i>"), 10)
        css = injected_css()
        self.assertIn("@keyframes star-depth-ink", css)
        self.assertIn("@keyframes star-depth-base", css)
        self.assertIn("40.0227%", css)
        self.assertIn("--t-behind", css)
        self.assertIn("--t-front", css)
        self.assertIn("--t-switch", css)
        self.assertIn("offset-path:path(", css.replace("\n", "").replace(" ", ""))
        rest = header_band_html(slim=True)
        self.assertIn("rest", rest)
        self.assertNotIn("is-wrapping", rest)
        self.assertNotIn("wrap-on", rest)
        compact = css.replace(" ", "").replace("\n", "")
        self.assertIn("prefers-reduced-motion:reduce", compact)
        self.assertIn(".band.wrap-on.mark.is-wrapping.rig", compact)


class DigestTests(unittest.TestCase):
    digest = "a" * 64

    def test_show_digest(self):
        show = MagicMock()
        show.ok = True
        show.json.return_value = {"digest": self.digest}
        with patch("app.engine.requests.post", return_value=show), patch(
            "app.engine.requests.get"
        ) as get:
            self.assertEqual(model_digest("qwen3.5:2b"), self.digest)
        get.assert_not_called()

    def test_blank_on_failure(self):
        with patch("app.engine.requests.post", side_effect=OSError("down")), patch(
            "app.engine.requests.get", side_effect=OSError("down")
        ):
            self.assertEqual(model_digest("qwen3.5:2b"), "")
        with patch("app.engine.requests.get", side_effect=OSError("down")):
            self.assertFalse(ollama_reachable())

    def test_tags_fallback_requires_an_exact_model(self):
        show = MagicMock()
        show.ok = True
        show.json.return_value = {
            "details": {"parameter_size": "2.0B", "quantization_level": "Q4_K_M"}
        }
        tags = MagicMock()
        tags.ok = True
        tags.json.return_value = {
            "models": [
                {"name": "qwen3.5:7b", "digest": "wrong"},
                {"name": "qwen3.5:2b", "digest": self.digest},
            ]
        }
        with patch("app.engine.requests.post", return_value=show), patch(
            "app.engine.requests.get", return_value=tags
        ):
            self.assertEqual(model_digest("qwen3.5:2b"), self.digest)

    def test_info_combines_version_and_model_digest(self):
        version = MagicMock()
        version.ok = True
        version.json.return_value = {"version": "0.12.3"}
        show = MagicMock()
        show.ok = True
        show.json.return_value = {"details": {"parameter_size": "2.0B"}}
        tags = MagicMock()
        tags.ok = True
        tags.json.return_value = {
            "models": [{"name": "qwen3.5:2b", "digest": self.digest}]
        }
        with patch("app.engine.requests.get", side_effect=[version, tags]), patch(
            "app.engine.requests.post",
            return_value=show,
        ) as post:
            info = ollama_info("qwen3.5:2b")
        self.assertEqual(info.version, "0.12.3")
        self.assertEqual(info.model_digest, self.digest)
        self.assertEqual(post.call_args.kwargs["json"], {"name": "qwen3.5:2b"})


class ImportPolarisTests(unittest.TestCase):
    def test_app_import_with_stub(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        with patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "False"}):
            sys.modules["streamlit"] = st
            if "app.app" in sys.modules:
                del sys.modules["app.app"]
            mod = importlib.import_module("app.app")
        self.assertTrue(hasattr(mod, "run_polaris"))
        self.assertTrue(hasattr(mod, "run_legacy"))
        self.assertEqual(os.environ.get("ANONYMIZED_TELEMETRY"), "False")


if __name__ == "__main__":
    unittest.main()
