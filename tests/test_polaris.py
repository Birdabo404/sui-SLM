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
from app.streaming import LiveThroughput, OllamaStreamError, retrieval_chart
from app.models import ComparePick, ModelOption, compare_picks, default_compare_ids, model_options, modes_for, pair_compare_ids
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
    MONITOR_SWITCH,
    RAM_TITLE,
    RateTrace,
    config_summary_html,
    finished_runs,
    hits_from_chart,
    last_runs,
    model_choice_label,
    model_status_html,
    models_table_html,
    monitor_html,
    rail_art_html,
    ram_html,
    rate_svg,
    score_table_html,
    live_phase_label,
    EXPORT_FIELDS,
    best_match_html,
    best_match_label,
    build_run_record,
    card_head_html,
    cube_frame,
    cube_html,
    cut_off_html,
    debug_rows_html,
    empty_slot_html,
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
    source_chips_html,
    source_key,
    style_path,
    CUBE_COLS,
    CUBE_FRAMES,
    CUBE_ROWS,
    waiting_html,
    STILL_WORKING,
    CancellableChat,
    ChatCancelled,
    compare_compact_html,
    mode_from_name,
    mode_name,
    preview_text,
    ENGLISH_NOTE,
    MODE_HINTS,
    composer_note_html,
    connection_html,
    flags_html,
    network_online,
    SKY_LEVELS,
    side_logo_html,
    sky_html,
    sky_level,
    star_defs,
    stopped_html,
    thinking_html,
    thinking_label,
    DELETE_ASK,
    HISTORY_EMPTY,
    answer_stats_html,
    compare_model_tags,
    compare_summary_html,
    history_error_text,
    history_note_html,
    plain_label,
    records_in,
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
        ) as commit, patch.object(app_mod, "_recent_chat_ids", return_value=["b", "a"]) as recent:
            app_mod._init_session()
            app_mod._init_session()
            self.assertEqual(app_mod._session_git_commit(), "abc1234")
        self.assertFalse(fake_st.session_state["compare"])
        self.assertEqual(fake_st.session_state["tabs"], ["b", "a"])
        self.assertEqual(fake_st.session_state["model_pick"], app_mod.BASE_MODEL)
        recent.assert_called_once_with()
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
        card = html + evidence_block_html(rec) + best_match_html(rec) + metrics_html(rec)
        self.assertNotIn("retr", card)
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
        options = model_options("qwen3.5:2b", "")
        self.assertEqual(default_compare_ids(options, False), ("qwen3.5:2b|base", "qwen3.5:2b|full"))

    def test_pair_must_differ(self):
        self.assertEqual(pair_compare_modes("base", "base"), ("base", "cot"))
        self.assertEqual(pair_compare_modes("full", "full"), ("full", "base"))
        self.assertEqual(pair_compare_modes("base", "full"), ("base", "full"))

    def test_student_side_is_the_default_only_when_configured(self):
        options = model_options("qwen3.5:2b", "student:latest")
        self.assertEqual(default_compare_ids(options, True), ("qwen3.5:2b|full", "student:latest|full"))
        self.assertEqual(default_compare_ids(model_options("qwen3.5:2b", ""), False)[1], "qwen3.5:2b|full")

    def test_pair_of_the_same_config_moves_the_right_side(self):
        options = model_options("qwen3.5:2b", "")
        left, right = pair_compare_ids("qwen3.5:2b|full", "qwen3.5:2b|full", options, False)
        self.assertEqual(left, "qwen3.5:2b|full")
        self.assertNotEqual(right, left)
        self.assertEqual(
            pair_compare_ids("qwen3.5:2b|full", "thesis-student|full", options, False),
            ("qwen3.5:2b|full", "thesis-student|full"),
        )

    def test_student_offers_only_full_mode(self):
        options = model_options("qwen3.5:2b", "")
        by_key = {o.key: o for o in options}
        self.assertEqual(modes_for(by_key["base"]), ("base", "cot", "rag", "full"))
        self.assertEqual(modes_for(by_key["distilled"]), ("full",))
        ids = [p.id for p in compare_picks(options)]
        self.assertIn("thesis-student|full", ids)
        self.assertNotIn("thesis-student|base", ids)

    def test_model_labels_say_whether_ollama_has_it(self):
        base, q4, student = model_options("qwen3.5:2b", "")
        names = frozenset({"qwen3.5:2b", "thesis-student:latest"})
        self.assertEqual(model_choice_label(base, names), "qwen3.5:2b")
        self.assertEqual(model_choice_label(student, names), "thesis-student")
        self.assertEqual(model_choice_label(q4, names), "thesis-base-q4km · not installed")
        # Unknown (Ollama down) is not the same as missing.
        self.assertEqual(model_choice_label(q4, None), "thesis-base-q4km")

        self.assertIn('<span class="ok">installed</span> · KD student · runs +RAG+CoT only',
                      model_status_html(student, names, True))
        missing = model_status_html(q4, names, True)
        self.assertIn('class="side-note model-st err"', missing)
        self.assertIn("<code>ollama create thesis-base-q4km -f Modelfile-base</code>", missing)
        self.assertIn("can't check, Ollama is not running", model_status_html(q4, None, False))

        table = models_table_html((base, q4, student), names, True)
        self.assertEqual(table.count('<li class="installed">'), 2)
        self.assertEqual(table.count('<li class="not-installed">'), 1)
        self.assertEqual(models_table_html((base, q4, student), None, False).count('<li class="unknown">'), 3)

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

        class View:
            def __init__(self, key, model):
                self.key, self.model = key, model

            def finish(self, rec, align_slots=False):
                events.append(("render", rec["mode"], self.key, align_slots))
                events.append(("tag", self.key, self.model))

        def answer(question, mode, option, *a, **k):
            events.append(("start", mode, option.name, option.key, k.get("side")))
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
            }, View(k["key"], k.get("tag", ""))

        with patch.object(app_mod.st, "container", return_value=Box()), patch.object(
            app_mod.st, "columns", return_value=(Col(), Col())
        ), patch.object(app_mod, "_answer_record", side_effect=answer):
            base = ModelOption("base", "qwen3.5:2b", "default base", "ollama pull qwen3.5:2b")
            student = ModelOption("distilled", "student:latest", "KD student", "ollama create student:latest -f Modelfile")
            monitor = MagicMock()
            (left, right), unanswered = app_mod._run_compare(
                "q", ComparePick(base, "base"), ComparePick(student, "full"), None, "idx", monitor=monitor
            )

        self.assertEqual(unanswered, (None, None))
        self.assertEqual(left["mode"], "base")
        self.assertEqual(right["mode"], "full")
        left_start = ("start", "base", "qwen3.5:2b", "base", "L")
        right_start = ("start", "full", "student:latest", "distilled", "R")
        monitor.plan.assert_called_once_with([("L", "qwen3.5:2b", "base"), ("R", "student:latest", "full")])
        self.assertLess(events.index(left_start), events.index("clear_wait"))
        self.assertLess(events.index("clear_wait"), events.index(right_start))
        self.assertEqual(
            [e for e in events if isinstance(e, tuple) and e[0] == "start"],
            [left_start, right_start],
        )
        self.assertIn(("render", "base", "answer-h1-l", True), events)
        self.assertIn(("render", "full", "answer-h1-r", True), events)
        # Different models on each side, so both heads name theirs.
        self.assertIn(("tag", "answer-h1-l", "qwen3.5:2b"), events)
        self.assertIn(("tag", "answer-h1-r", "student:latest"), events)


class HtmlTests(unittest.TestCase):
    def test_card_head_and_retrieval_off(self):
        self.assertIn('<span class="mode">BASE</span>', card_head_html("base"))
        self.assertIn("Plain model, no documents", card_head_html("base"))
        self.assertIn('<span class="mode">+RAG+CoT</span>', card_head_html("full"))
        self.assertNotIn("[", card_head_html("full"))

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
        self.assertIn("<b>2.5 GB</b> peak RSS app+Ollama", html)
        self.assertIn("app 420 MB, Ollama 2100 MB", html)
        self.assertIn("<b>3</b> sources", html)
        self.assertNotIn(" · ", html)
        self.assertNotIn("app <b>420</b>", html)
        self.assertNotIn("2100", re.sub(r"<[^>]+>", "", html))
        self.assertNotIn("docs", html)
        # No timing fields: speed reads as unavailable rather than disappearing or showing a 0.
        self.assertIn("tok/s unavailable", html)
        self.assertNotIn("<b>0.0</b> tok/s", html)
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

    def test_ram_says_what_was_measured(self):
        self.assertIn("RAM not measured", ram_html({"total_mb": 0}))
        self.assertIn("RAM not measured", ram_html({}))
        self.assertNotIn("<b>0", ram_html({"total_mb": 0}))
        app_only = ram_html({"total_mb": 300, "app_rss_mb": 300, "ollama_rss_mb": 0})
        self.assertIn("peak RSS, app only", app_only)
        self.assertIn("No Ollama process was visible", app_only)
        both = ram_html({"total_mb": 2520, "app_rss_mb": 420, "ollama_rss_mb": 2100})
        self.assertIn("peak RSS app+Ollama", both)
        self.assertTrue(RAM_TITLE.startswith("Peak resident memory (RSS)"))

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
        self.assertIn('<span class="cube still"', html)
        self.assertIn(".cube.still .cube-reel{animation:none", css)
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
        # The pieces a compare card is drawn from (app._AnswerView.finish with align_slots).
        left = card_head_html("base") + empty_slot_html("ev", "No sources in BASE") + metrics_html(base)
        right = card_head_html("full") + evidence_block_html(full) + metrics_html(full)
        self.assertIn('<span class="mode">BASE</span>', left)
        self.assertIn("Plain model, no documents", left)
        self.assertIn('<span class="mode">+RAG+CoT</span>', right)
        self.assertIn("Retrieved lab docs, step-by-step prompt", right)
        self.assertIn("slot-empty ev", left)
        self.assertIn("No sources in BASE", left)
        self.assertIn('<span class="count">1 source</span>', right)
        # Scores and the rate live in the rail's monitor now, never inside an answer card.
        for card in (left, right):
            self.assertNotIn("retr", card)
            self.assertNotIn('class="spark"', card)
        self.assertIn("slot-empty best", empty_slot_html("best"))
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
        self.assertNotIn("var(--c-accent)", compact.split(".ev-box{", 1)[1].split(".metrics{", 1)[0])


class LoaderAndNamingTests(unittest.TestCase):
    def test_thinking_loader_is_a_css_only_cube_reel(self):
        html = thinking_html("Searching the lab docs")
        self.assertIn('role="status"', html)
        self.assertIn('<span class="cube" aria-hidden="true">', html)
        reel = html.split('<span class="cube-reel">', 1)[1].split("</span>", 1)[0]
        lines = reel.split("<br>")
        self.assertEqual(len(lines), CUBE_FRAMES * CUBE_ROWS)
        self.assertTrue(all(len(line) == CUBE_COLS for line in lines))
        self.assertNotIn("\n", html)
        self.assertIn("Searching the lab docs", html)
        self.assertIn(STILL_WORKING, html)
        self.assertNotIn("<script", html)
        css = style_path().read_text(encoding="utf-8")
        block = css.split("/* ---------- loader", 1)[1].split("/* ---------- while an answer generates", 1)[0]
        for prop in ("width:", "height:", "top:", "left:", "margin"):
            for frames in block.split("@keyframes")[1:]:
                body = frames.split("}}", 1)[0]
                self.assertNotIn(prop, body, f"loader keyframes animate {prop}")

    def test_cube_turns_a_seamless_quarter_turn_and_rests_under_reduced_motion(self):
        frames = [cube_frame(t) for t in range(CUBE_FRAMES)]
        self.assertEqual(len({tuple(f) for f in frames}), CUBE_FRAMES)
        self.assertEqual(cube_frame(CUBE_FRAMES), frames[0])
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(f"animation:polaris-cube 1.2s steps({CUBE_FRAMES}) infinite", css)
        reduced = [block.split("\n}", 1)[0] for block in css.split("@media (prefers-reduced-motion:reduce){")[1:]]
        self.assertTrue(any(".cube-reel{animation:none" in block for block in reduced))
        self.assertIn('class="cube still"', cube_html(still=True))

    def test_thinking_label_follows_mode(self):
        self.assertEqual(thinking_label("full"), "Generating an answer from the lab docs")
        self.assertEqual(thinking_label("rag"), "Generating an answer from the lab docs")
        self.assertEqual(thinking_label("cot"), "Generating an answer")
        self.assertEqual(thinking_label("base"), "Generating an answer")
        self.assertEqual(thinking_label("full", loading_index=True), "Loading the lab docs index")
        self.assertIn("Stopped", stopped_html())

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

    def test_ambient_motion_pauses_while_generating(self):
        self.assertIn("polaris-generating", flags_html(1, generating=True))
        self.assertNotIn("polaris-generating", flags_html(1))
        self.assertEqual(
            flags_html(2, daylight=True, compare=True, generating=True),
            '<div class="polaris-flags polaris-sky-2 polaris-daylight polaris-compare polaris-generating"></div>',
        )
        css = style_path().read_text(encoding="utf-8")
        rules = [r for r in css.split("}") if ":root:has(.polaris-generating)" in r and "play-state" in r]
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
        self.assertEqual(preview_text(text, "cot"), "Run this: Then ls it.")
        html = compare_compact_html({"mode": "base", "answer": "Use `tar`."}, {"mode": "cot", "answer": text})
        self.assertNotIn("```", html)
        self.assertNotIn("Think first", html)


class AnswerStatsTests(unittest.TestCase):
    TIMED = {
        "mode": "full",
        "answer": "Use `tar -xzf`.",
        "latency_s": 3.24,
        "tok_s": 18.43,
        "gen_s": 2.46,
        "prompt_s": 0.36,
        "load_s": 0.18,
        "eval_count": 45,
        "total_mb": 2520,
        "docs": ["a"],
        "abstained": False,
    }

    def test_time_and_speed_chips_lead_the_metrics_row(self):
        html = metrics_html(self.TIMED)
        self.assertTrue(html.startswith('<div class="metrics"><span class="stats">'))
        self.assertIn(
            '<span class="stat" title="Question to answer: model load 0.18 s, prompt 0.36 s, generation 2.46 s">'
            '<span class="sr">Response time </span><b>3.2</b> s</span>',
            html,
        )
        self.assertIn(
            '<span class="stat" title="45 tokens in 2.46 s of generation">'
            '<span class="sr">Speed </span><b>18.4</b> tok/s</span>',
            html,
        )
        self.assertLess(html.index("tok/s"), html.index("peak RSS"))

    def test_speed_chip_says_unavailable_without_generation_timing(self):
        for missing in ({"tok_s": None}, {"tok_s": ""}):
            html = metrics_html({**self.TIMED, **missing, "gen_s": None, "prompt_s": None, "load_s": None})
            self.assertIn("<b>3.2</b> s", html)
            self.assertIn('<span class="sr">Speed </span>tok/s unavailable</span>', html)
            self.assertNotRegex(html, r"<b>[\d.]+</b> tok/s")
            self.assertIn('title="Question to answer"', html)
        # An abstained answer never reaches the model: its time shows, and its speed says nothing was generated.
        idk = answer_stats_html({"latency_s": 0.4, "abstained": True})
        self.assertEqual(idk.count('class="stat"'), 1)
        self.assertIn("tok/s not generated", idk)
        self.assertNotIn("unavailable", idk)

    def test_speed_shows_ollamas_own_counts_beside_tok_s(self):
        html = answer_stats_html(self.TIMED)
        self.assertIn('<span class="stat calc">45 tok / 2.46 s</span>', html)
        self.assertNotIn("calc", answer_stats_html({**self.TIMED, "eval_count": None}))

    def test_stat_numbers_keep_their_width(self):
        css = style_path().read_text(encoding="utf-8")
        rule = css.split(".stat b{", 1)[1].split("}", 1)[0]
        self.assertIn("font-variant-numeric:tabular-nums", rule)
        self.assertIn(".sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)", css)


class LiveStreamHtmlTests(unittest.TestCase):
    HITS = [
        {"rank": 1, "source": "tldr", "title": "tar", "path": "tldr/common/tar.md", "dense": 0.7412, "rrf": 0.0328,
         "bm25": 7.013},
        {"rank": 2, "source": "man-pages", "title": "tar(1)", "path": "man/tar.1", "dense": 0.21, "rrf": 0.0161,
         "bm25": None},
    ]

    DONE = {"mode": "full", "model": "qwen3.5:2b", "answer": "Use tar.", "latency_s": 6.7, "tok_s": 15.2,
            "gen_s": 5.45, "eval_count": 83, "total_mb": 512, "docs": ["tar"], "abstained": False}

    def test_score_table_prints_raw_scores_and_bars_only_cosine(self):
        html = score_table_html(self.HITS, 0.30)
        self.assertEqual(html.count('class="bar"'), 2)
        self.assertIn('style="width:74.1%"', html)
        self.assertEqual(html.count('style="left:30.0%"'), 2)
        self.assertIn('<li class="ok">', html)
        self.assertIn('<li class="low">', html)
        self.assertIn('<span class="sr">cosine </span>.741', html)
        self.assertIn('<span class="sr">rrf </span>.0328', html)
        self.assertIn('<span class="sr">bm25 </span>7.01', html)
        self.assertIn('<span class="sr">bm25 </span>—', html)
        self.assertNotRegex(html.lower(), "confidence|probab|%</")

    def test_live_chart_rows_map_from_the_backend_contract(self):
        chart = retrieval_chart(self.HITS, 0.30)
        hits = hits_from_chart(chart)
        self.assertEqual([(h["rank"], h["dense"], h["rrf"], h["bm25"]) for h in hits],
                         [(1, 0.7412, 0.0328, 7.013), (2, 0.21, 0.0161, None)])
        self.assertIn('title="tldr/common/tar.md"', score_table_html(hits, chart.dense_threshold))

    def test_rate_trace_counts_characters_from_the_first_text(self):
        trace = RateTrace(every=0.5)
        trace.add(LiveThroughput(2.0, 1, 10, 0.5, 5.0))
        trace.add(LiveThroughput(2.3, 4, 40, 1.7, 17.4))
        self.assertEqual(trace.samples, [])
        trace.add(LiveThroughput(2.5, 6, 70, 2.4, 28.0))
        trace.add(LiveThroughput(3.0, 9, 100, 3.0, 33.3))
        trace.finish()
        self.assertEqual(trace.first_s, 2.0)
        self.assertEqual([round(r, 1) for _, r in trace.samples], [120.0, 60.0])
        self.assertEqual(trace.frozen(), {"samples": tuple(trace.samples), "first_s": 2.0})

    def test_rate_chart_draws_left_solid_and_right_dashed_on_shared_axes(self):
        left = ((1.0, 10.0), (1.5, 40.0))
        right = ((3.0, 20.0), (3.5, 20.0), (4.0, 80.0))
        svg = rate_svg([("l", left), ("r", right)])
        self.assertEqual(svg.count('class="line l"'), 1)
        self.assertEqual(svg.count('class="line r"'), 1)
        self.assertEqual(svg.count('class="area"'), 1)
        self.assertIn("peak 80", svg)
        self.assertEqual(rate_svg([("l", ((1.0, 5.0),))]), "")

    def test_streaming_monitor_never_shows_a_token_rate(self):
        trace = RateTrace(every=0.5)
        trace.add(LiveThroughput(1.0, 1, 12, 1.0, 12.0))
        run = {"side": "", "model": "qwen3.5:2b", "mode": "base", "state": "streaming", "live": trace.last,
               "samples": (), "first_s": 1.0}
        html = monitor_html([run], streaming=True)
        self.assertIn('<span class="mon-tag live">streaming</span>', html)
        self.assertIn("<b>12</b> chars", html)
        self.assertIn("<b>1</b> chunks", html)
        self.assertIn("<b>—</b> chars/s", html)
        self.assertIn("tok/s comes from Ollama when the answer ends", html)
        self.assertNotRegex(html, r"<b>[\d.,]+</b> tok")
        self.assertIn("not tokens", html)
        waiting = monitor_html([{**run, "live": None}], streaming=True)
        self.assertIn("waiting for the first text", waiting)

    def test_finished_monitor_keeps_ollamas_tok_s_and_the_trace(self):
        run = {"side": "", "model": "qwen3.5:2b", "mode": "full", "state": "done", "record": self.DONE,
               "samples": ((1.2, 30.0), (1.7, 68.0)), "first_s": 1.2, "hits": self.HITS, "threshold": 0.30}
        html = monitor_html([run])
        self.assertIn('<span class="mon-tag">last answer</span>', html)
        self.assertIn("<b>6.7</b> s", html)
        self.assertIn("<b>15.2</b> tok/s", html)
        self.assertIn("first text <b>1.2</b> s · peak <b>68</b>", html)
        self.assertIn('class="spark"', html)
        self.assertIn("best cos <b>0.741</b> clears 0.30", html)

    def test_restored_answer_says_why_it_has_no_trace(self):
        html = monitor_html([{"side": "", "model": "m", "mode": "full", "state": "done", "record": self.DONE,
                              "samples": (), "first_s": None, "hits": [], "threshold": 0.3}])
        self.assertIn("only recorded while an answer streams in this session", html)
        self.assertIn("<b>15.2</b> tok/s", html)
        self.assertNotIn('class="spark"', html)
        self.assertIn("No answer in this chat yet", monitor_html([]))

    def test_monitor_tells_compare_sides_apart(self):
        left = {"side": "L", "model": "qwen3.5:2b", "mode": "base", "state": "done",
                "record": {**self.DONE, "mode": "base"}, "samples": ((0.5, 10.0), (1.0, 20.0)), "first_s": 0.5}
        right = {"side": "R", "model": "thesis-student", "mode": "full", "state": "waiting", "samples": ()}
        html = monitor_html([left, right], streaming=True)
        self.assertIn('<span class="side l">L</span><span class="m">qwen3.5:2b</span><span class="md">BASE</span>', html)
        self.assertIn('<span class="side r">R</span><span class="m">thesis-student</span><span class="md">+RAG+CoT</span>', html)
        self.assertIn("starts when the left answer finishes", html)
        self.assertIn("No retrieval in BASE", monitor_html([left]))
        stopped = monitor_html([{**right, "state": "stopped"}])
        self.assertIn("stopped, no stats", stopped)
        self.assertNotRegex(stopped, r"<b>0(\.0)?</b> tok/s")

    def test_scores_view_names_below_threshold_and_shared_hits(self):
        low = [dict(self.HITS[1], rank=1)]
        run = {"side": "", "mode": "rag", "state": "done", "record": {"abstained": True, "latency_s": 0.3},
               "samples": (), "hits": low, "threshold": 0.30}
        self.assertIn("below 0.30</span><span class=\"unit\">model not called</span>", monitor_html([run]))
        left = {"side": "L", "mode": "rag", "model": "a", "state": "done", "record": self.DONE, "samples": (),
                "hits": self.HITS, "threshold": 0.3}
        right = {**left, "side": "R", "mode": "full"}
        self.assertIn("Both sides retrieved the same chunks.", monitor_html([left, right]))
        plain = {**left, "side": "R", "mode": "base", "hits": []}
        self.assertIn("R (BASE) retrieves nothing.", monitor_html([left, plain]))

    def test_monitor_switch_is_uncontrolled_and_both_views_ship(self):
        # React would turn a checked radio into a controlled input that ignores clicks.
        self.assertNotIn("checked", MONITOR_SWITCH)
        self.assertIn('role="radiogroup"', MONITOR_SWITCH)
        self.assertIn('<label for="mon-v-scores">Scores</label>', MONITOR_SWITCH)
        html = monitor_html([])
        self.assertIn("mon-view mon-rate", html)
        self.assertIn("mon-view mon-scores", html)

    def test_runs_come_from_the_last_answer_with_its_session_trace(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "record": self.DONE},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "compare": ({**self.DONE, "mode": "base"}, None),
             "compare_unanswered": (None, {"mode": "full", "model": "thesis-student", "state": "cancelled"})},
        ]
        traces = {"answer-h3-l": {"samples": ((0.1, 5.0), (0.6, 9.0)), "first_s": 0.1}}
        left, right = last_runs(messages, traces)
        self.assertEqual((left["side"], left["state"], left["samples"]), ("L", "done", ((0.1, 5.0), (0.6, 9.0))))
        self.assertEqual((right["side"], right["state"], right["model"]), ("R", "stopped", "thesis-student"))
        (single,) = finished_runs(messages[1], 1, {})
        self.assertEqual((single["side"], single["samples"], single["first_s"]), ("", (), None))
        self.assertEqual(last_runs(messages[:1], {}), [])

    def test_rail_art_is_static_text_hidden_from_readers(self):
        html = rail_art_html()
        self.assertTrue(html.startswith('<div class="rail-art" aria-hidden="true">'))
        self.assertEqual(html.count("<span"), 9)
        self.assertEqual(html.count('class="pole"'), 1)
        self.assertEqual(html.count('class="dip"'), 6)
        self.assertNotRegex(html, "<(img|svg|script|style)")
        self.assertNotIn("style=", html)
        rows = re.findall(r'<span class="(?:hi|mid|lo)">(.*?)</span>(?=<span class="(?:hi|mid|lo)">|</div>)', html)
        self.assertEqual([len(re.sub(r"<[^>]+>", "", r)) for r in rows], [44] * 9)
        self.assertIs(type(html), str)
        self.assertEqual(rail_art_html(), html)

    def test_phase_labels_name_only_work_that_runs(self):
        self.assertEqual(live_phase_label("full"), "Searching the lab docs")
        self.assertEqual(live_phase_label("full", retrieved=True), "Waiting for the model")
        self.assertIn("not called", live_phase_label("rag", retrieved=True, weak=True))
        for mode in ("base", "cot"):
            self.assertEqual(live_phase_label(mode), "Waiting for the model")


class CompareLabelTests(unittest.TestCase):
    def test_card_head_names_a_model_only_when_asked(self):
        self.assertEqual(
            card_head_html("full"),
            '<div class="card-head"><span class="mode">+RAG+CoT</span>'
            '<span class="sub">Retrieved lab docs, step-by-step prompt</span></div>',
        )
        tagged = card_head_html("full", "student<1>")
        self.assertIn('<span class="mode">+RAG+CoT</span><span class="model">student&lt;1&gt;</span>', tagged)
        self.assertIn('<span class="model">student:latest</span>', waiting_html("full", "the left answer", "student:latest"))
        self.assertIn("Starts when the left answer finishes", waiting_html("full", "the left answer"))

    def test_model_tags_only_when_the_sides_differ(self):
        base = {"mode": "full", "model": "qwen3.5:2b", "answer": "a"}
        student = {"mode": "full", "model": "student:latest", "answer": "b"}
        self.assertEqual(compare_model_tags(base, student), ("qwen3.5:2b", "student:latest"))
        self.assertEqual(compare_model_tags(base, {**base, "mode": "base"}), ("", ""))
        self.assertEqual(compare_model_tags(base, None), ("", ""))

    def test_compact_compare_rows_carry_models_and_stats(self):
        left = {"mode": "full", "model": "qwen3.5:2b", "answer": "Use tar.", "latency_s": 3.0, "tok_s": 18.3}
        right = {"mode": "full", "model": "student:latest", "answer": "Use tar too.", "latency_s": 2.1, "tok_s": 26.4}
        html = compare_compact_html(left, right)
        self.assertEqual(html.count('<div class="cmp-top">'), 2)
        self.assertIn('<span class="model">qwen3.5:2b</span>', html)
        self.assertIn('<span class="model">student:latest</span>', html)
        self.assertIn("<b>26.4</b> tok/s", html)
        same = compare_compact_html({**left, "mode": "base"}, left)
        self.assertNotIn('class="model"', same)

    def test_composer_line_names_the_model_once_or_per_side(self):
        self.assertEqual(
            compare_summary_html("qwen3.5:2b", "base", "qwen3.5:2b", "full"),
            '<p class="composer-cmp"><span class="m">qwen3.5:2b</span> <b>BASE</b>'
            '<span class="vs">vs</span><b>+RAG+CoT</b></p>',
        )
        self.assertEqual(
            compare_summary_html("qwen3.5:2b", "full", "student:latest", "full"),
            '<p class="composer-cmp"><span class="m">qwen3.5:2b</span> <b>+RAG+CoT</b><span class="vs">vs</span>'
            '<span class="m">student:latest</span> <b>+RAG+CoT</b></p>',
        )

    def test_composer_line_outside_compare_names_the_picked_model(self):
        self.assertEqual(
            config_summary_html("thesis-student", "full"),
            '<p class="composer-cmp"><span class="m">thesis-student</span> <b>+RAG+CoT</b></p>',
        )


class HistoryHelperTests(unittest.TestCase):
    def test_saved_question_shows_as_typed(self):
        label = plain_label("what is  $HOME vs *.tar.gz?\n`ls -a` [docs](x) :smile: #1")
        self.assertEqual(
            label,
            "what is \\$HOME vs \\*\\.tar\\.gz? \\`ls \\-a\\` \\[docs\\]\\(x\\) :\u200bsmile:\u200b \\#1",
        )
        self.assertNotIn(":smile:", label)

    def test_records_come_back_in_answer_order(self):
        a, b, c = {"mode": "base"}, {"mode": "cot"}, {"mode": "full"}
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "record": a},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "compare": [b, None]},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "error": "Could not reach Ollama"},
            {"role": "user", "content": "q4"},
            {"role": "assistant", "compare": [c, a]},
        ]
        self.assertEqual(records_in(messages), [a, b, c, a])

    def test_history_notes_and_errors(self):
        self.assertEqual(history_note_html(HISTORY_EMPTY), '<p class="side-note">Chats you start are saved on this machine.</p>')
        error = history_error_text(Path.home() / ".local" / "share" / "sui-slm")
        self.assertEqual(
            error,
            "Couldn't open chat history in ~/.local/share/sui-slm, so chats aren't being saved. "
            "Check that the folder is writable.",
        )
        self.assertIn('class="side-note err"', history_note_html(error, error=True))
        self.assertIn("/srv/x", history_error_text("/srv/x"))
        self.assertEqual(DELETE_ASK, "Delete this chat? This can't be undone.")


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
                release.wait(5)

            def request(self, *args, **kwargs):
                calls.append("request")

            def close(self):
                calls.append("close")
                release.set()

        chat = CancellableChat(self.host)
        box = {}

        def run():
            try:
                chat([{"role": "user", "content": "hi"}], "m")
            except BaseException as e:
                box["err"] = e

        with patch("app.streaming.http.client.HTTPConnection", Conn):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(connected.wait(2))
            chat.cancel()
            worker.join(1)
            release.set()
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

        with patch("app.streaming.http.client.HTTPConnection", Conn):
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

    def _app_with_mock_st(self):
        st = MagicMock()
        st.cache_resource = _cache_stub
        st.cache_data = _cache_stub
        st.session_state = MagicMock()
        st.session_state.run_records = []
        st.write_stream.side_effect = lambda gen: "".join(gen)
        sys.modules["streamlit"] = st
        if "app.app" in sys.modules:
            del sys.modules["app.app"]
        app_mod = importlib.import_module("app.app")
        return st, app_mod

    def _answer(self, app_mod, slot):
        """_answer_record for one plain answer, with a real monitor on a mock slot; returns (record, monitor run)."""
        self.monitor = app_mod._Monitor(MagicMock())
        self.monitor.plan([("", "m", "base")])
        option = ModelOption("base", "m", "default base", "ollama pull m")
        rec, _ = app_mod._answer_record("q", "base", option, None, "idx", slot, key="answer-1", monitor=self.monitor)
        return rec, self.monitor.runs[0]

    def test_interrupt_before_the_first_chunk_cancels_and_closes_the_stream(self):
        st, app_mod = self._app_with_mock_st()
        answer = _BlockingAnswer()
        beats = {"n": 0}

        def get(key, *default):
            if key == "stop-gen":
                beats["n"] += 1
                if beats["n"] == 2:
                    raise _Interrupted("rerun requested")
            return None

        st.session_state.get.side_effect = get
        slot = MagicMock()
        with patch.object(app_mod, "stream_answer_question", return_value=answer), \
                patch.object(app_mod, "_model_info", return_value=None):
            with self.assertRaises(_Interrupted):
                self._answer(app_mod, slot)
        self.assertTrue(answer.cancelled.is_set())
        self.assertTrue(answer.closed)
        self.assertFalse(answer.worker_alive_after(1.0))
        slot.error.assert_not_called()

    def test_answer_record_catches_chat_cancelled(self):
        st, app_mod = self._app_with_mock_st()
        slot = MagicMock()
        answer = _BlockingAnswer(raise_now=True)
        with patch.object(app_mod, "stream_answer_question", return_value=answer), \
                patch.object(app_mod, "_model_info", return_value=None):
            rec, run = self._answer(app_mod, slot)
        self.assertIsNone(rec)
        self.assertEqual(run["state"], "stopped")
        self.assertIs(st.session_state.last_cancelled, True)
        slot.empty.assert_called_once()
        self.assertTrue(answer.closed)

    def test_a_stream_error_is_shown_as_ollama_error_not_a_connection_failure(self):
        st, app_mod = self._app_with_mock_st()
        slot = MagicMock()
        answer = _BlockingAnswer(error=OllamaStreamError("model 'm' not found"))
        with patch.object(app_mod, "stream_answer_question", return_value=answer), \
                patch.object(app_mod, "_model_info", return_value=None):
            rec, run = self._answer(app_mod, slot)
        self.assertIsNone(rec)
        self.assertEqual(run["state"], "error")
        message = slot.error.call_args.args[0]
        self.assertIn("model 'm' not found", message)
        self.assertNotIn("Could not reach", message)
        self.assertTrue(answer.closed)

    def test_loader_is_sent_once_while_waiting_for_the_first_chunk(self):
        st, app_mod = self._app_with_mock_st()
        slot = MagicMock()
        answer = _BlockingAnswer(text="use tar", delay=0.7, result={"mode": "base", "text": "use tar", "hits": [],
                                                                      "weak": False, "latency_s": 0.7})
        with patch.object(app_mod, "stream_answer_question", return_value=answer), \
                patch.object(app_mod, "_model_info", return_value=None):
            rec, run = self._answer(app_mod, slot)
        self.assertEqual(rec["answer"], "use tar")
        self.assertEqual((run["state"], run["record"]), ("done", rec))
        drawn = self.monitor.slot.markdown.call_args.args[0]
        self.assertIn('<span class="mon-tag">last answer</span>', drawn)
        status = st.empty.return_value
        self.assertEqual(status.markdown.call_count, 1)
        self.assertIn("Waiting for the model…", status.markdown.call_args.args[0])
        stop_checks = [c for c in st.session_state.get.call_args_list if c.args == ("stop-gen",)]
        self.assertGreaterEqual(len(stop_checks), 2)
        self.assertTrue(answer.closed)

    def test_text_reaches_write_stream_before_the_answer_finishes(self):
        st, app_mod = self._app_with_mock_st()
        release = threading.Event()
        events = __import__("queue").Queue()

        def produce():
            events.put(("text", "Use "))
            release.wait(3)
            events.put(("text", "`tar`."))
            events.put(("done", None))

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        gen = app_mod._stream_text(events, lambda *a: None, heartbeat=lambda: None, tick=0.05, every=0.05)
        self.assertEqual(next(gen), "Use ")
        self.assertFalse(release.is_set())
        self.assertTrue(worker.is_alive())
        release.set()
        self.assertEqual("".join(gen), "`tar`.")

    def test_chunks_that_arrive_together_are_drawn_once(self):
        st, app_mod = self._app_with_mock_st()
        events = __import__("queue").Queue()
        seen = []
        for piece in ("a", "b", "c"):
            events.put(("progress", piece))
            events.put(("text", piece))
        events.put(("done", None))
        out = list(app_mod._stream_text(events, lambda kind, value: seen.append(kind), heartbeat=lambda: None, every=10))
        self.assertEqual(out, ["abc"])
        self.assertEqual(seen, ["progress"] * 3)


class _BlockingAnswer:
    """A StreamingAnswer stand-in: blocks until cancelled, raises, or yields `text` after `delay`."""

    def __init__(self, text=None, delay=0.0, result=None, raise_now=False, error=None):
        self.text, self.delay, self._result = text, delay, result
        self.raise_now, self.error = raise_now, error
        self.cancelled = threading.Event()
        self.closed = False
        self._done = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.error is not None:
                raise self.error
            if self.raise_now:
                raise ChatCancelled()
            if self.text is not None:
                time.sleep(self.delay)
                text, self.text = self.text, None
                return text
            if self._result is not None:
                raise StopIteration
            self.cancelled.wait(5)
            raise ChatCancelled()
        except BaseException:
            self._done.set()
            raise

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("streaming answer has not completed")
        return self._result

    def worker_alive_after(self, seconds):
        return not self._done.wait(seconds)

    def cancel(self):
        self.cancelled.set()

    def close(self):
        self.closed = True
        self.cancelled.set()


class HelperTests(unittest.TestCase):
    def test_abstain_card_reads_the_answer_with_the_one_parser(self):
        html = idk_card_html({"answer": ABSTAIN})
        self.assertNotIn("I checked the excerpts", html)
        self.assertIn("I don&#x27;t know.", html)

    def test_cut_off_only_when_ollama_says_length(self):
        self.assertIn("cut off at a length limit", cut_off_html({"done_reason": "length"}))
        for reason in ("stop", None, ""):
            self.assertEqual(cut_off_html({"done_reason": reason}), "")
        self.assertIn("cut off", metrics_html({"done_reason": "length", "mode": "base"}))
        self.assertEqual(build_run_record({"done_reason": "length", "mode": "base"})["done_reason"], "length")
        self.assertIsNone(build_run_record({"mode": "base"})["done_reason"])
        self.assertIn("done_reason", EXPORT_FIELDS)

    def test_max_dense(self):
        self.assertIsNone(max_dense([]))
        self.assertAlmostEqual(max_dense([{"dense": 0.1}, {"dense": 0.4}]), 0.4)

    def test_index_chunk_count(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(index_chunk_count(td), 0)
            (Path(td) / "chunks.jsonl").write_text('{"id":1}\n\n{"id":2}\n', encoding="utf-8")
            self.assertEqual(index_chunk_count(td), 2)

    def test_every_static_url_is_served_from_the_entry_scripts_static_dir(self):
        # Streamlit serves /app/static/ from <main script dir>/static, and the launcher runs streamlit_app.py.
        try:
            from streamlit.file_util import get_app_static_dir
        except ImportError:
            self.skipTest("streamlit not installed")
        static = Path(get_app_static_dir(str(ROOT / "streamlit_app.py")))
        css = style_path().read_text(encoding="utf-8") + orbit_style_path().read_text(encoding="utf-8")
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        names = set(re.findall(r"/app/static/([\w.-]+)", css)) | set(re.findall(r'"app/static/([\w.-]+)"', config))
        self.assertGreaterEqual(len(names), 3)
        for name in sorted(names):
            self.assertTrue((static / name).is_file(), f"{name} is not served from {static}")
            self.assertEqual(os.path.commonpath([os.path.abspath(static / name), str(static)]), str(static))

    def test_style_ships_offline(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn("/app/static/UbuntuSans-Variable-Latin.woff2", css)
        self.assertTrue((style_path().parent / "static" / "UbuntuSans-Variable-Latin.woff2").is_file())
        self.assertTrue((style_path().parent / "static" / "UbuntuSans-UFL.txt").is_file())
        self.assertNotIn("Inter", css)
        self.assertNotIn("fonts.googleapis.com", css)
        self.assertNotIn("cdn", css.lower())
        self.assertIn("--fs-body:14.5px", css)
        self.assertIn("1px dashed var(--c-idk)", css)
        self.assertNotIn('content:"▸', css)
        self.assertNotIn("transition:all", css.replace(" ", ""))
        self.assertIn("@keyframes star-breathe", css)
        self.assertIn("@keyframes polaris-breathe", css)
        self.assertIn("@keyframes polaris-cube", css)
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
