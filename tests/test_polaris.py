#!/usr/bin/env python3
"""Polaris UI helpers. No live Ollama."""
import csv
import http.client
import importlib
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
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
from app import identity
from app.streaming import LiveThroughput, retrieval_chart
from tests.backend_fake import PlainStreamError
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
from app.answer_format import parse_answer
from app.polaris import (
    AMBIGUOUS_K,
    AMBIGUOUS_SPREAD,
    AMBIGUOUS_TOP,
    ambiguity_hint_html,
    corpus_label,
    excerpt_label,
    source_line_html,
    COPY_FEEDBACK_ID,
    FOLLOW_ID,
    MONITOR_SWITCH,
    TABS_SCRIPT_ID,
    RAM_TITLE,
    REASONING_LABEL,
    REASONING_TITLE,
    THEMES,
    RateTrace,
    config_summary_html,
    copy_feedback_html,
    copy_feedback_path,
    finished_runs,
    follow_html,
    follow_path,
    tabs_script_html,
    tabs_script_path,
    hits_from_chart,
    last_runs,
    model_choice_label,
    model_status_html,
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
    CUBE_PITCH,
    CUBE_RAMP,
    CUBE_ROWS,
    CUBE_SECONDS,
    _CUBE_DISTANCE,
    _CUBE_FACE_LIST,
    _CUBE_FIT,
    _cube_orientation,
    _cube_render,
    _mat_apply,
    waiting_html,
    STILL_WORKING,
    CancellableChat,
    ChatCancelled,
    mode_from_name,
    mode_name,
    ENGLISH_NOTE,
    MODE_HINTS,
    composer_note_html,
    connection_html,
    flags_html,
    network_online,
    side_logo_html,
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
        self.assertIn('<details class="dbg"><summary>Raw scores</summary>', panel)
        self.assertNotIn("Debug", panel)
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
        reel = html.split('<span class="cube-reel">', 1)[1].split('</span></span><span class="thinking-text">', 1)[0]
        blocks = reel.split('<span class="cube-frame">')[1:]
        self.assertEqual(len(blocks), CUBE_FRAMES)
        for t, block in enumerate(blocks):
            # The dimmest third of the ramp is inked .d, the brightest .b, the middle third the cube's own ink.
            for ink, run in re.findall(r'<span class="([db])">([^<]*)</span>', block):
                thirds = {CUBE_RAMP.index(ch) * 3 // len(CUBE_RAMP) for ch in run}
                self.assertEqual(thirds, {0 if ink == "d" else 2}, run)
            frame = re.sub(r"</?span[^>]*>", "", block).split("<br>")
            self.assertEqual(frame, cube_frame(t))
            self.assertEqual(len(frame), CUBE_ROWS)
            self.assertTrue(all(len(line) == CUBE_COLS for line in frame))
            self.assertLessEqual(set("".join(frame)), set(CUBE_RAMP + " "))
        self.assertNotIn("\n", html)
        self.assertIn("Searching the lab docs", html)
        self.assertIn(STILL_WORKING, html)
        self.assertNotIn("<script", html)
        css = style_path().read_text(encoding="utf-8")
        block = css.split("/* ---------- loader", 1)[1].split("/* ---------- while an answer generates", 1)[0]
        for prop in ("width:", "height:", "top:", "left:", "margin"):
            for keyframes in block.split("@keyframes")[1:]:
                body = keyframes.split("}}", 1)[0]
                self.assertNotIn(prop, body, f"loader keyframes animate {prop}")
        # A fixed grid in any monospace font: whole-pixel lines, a width in ch, no ligatures merging runs of a glyph.
        self.assertIn(f"width:{CUBE_COLS}ch;height:{CUBE_ROWS * 4}px", block)
        self.assertIn("font:600 5px/4px var(--f-mono)", block)
        self.assertIn('font-variant-ligatures:none;font-feature-settings:"liga" 0,"calt" 0', block)
        self.assertIn(f".cube-frame{{display:block;height:{CUBE_PITCH * 4}px}}", block)
        self.assertIn("color:var(--c-text-3)}\n.cube .d{color:color-mix(in oklch,var(--c-text-3) 50%,transparent)}", block)
        self.assertIn(".cube .b{color:var(--c-text)}", block)
        # An inline indicator: no side over 48px (a 0.6em advance at 5px is 3px).
        self.assertLessEqual(CUBE_ROWS * 4, 48)
        self.assertLessEqual(CUBE_COLS * 3, 48)
        self.assertNotIn("\u2588", "".join(cube_frame(t)[r] for t in range(CUBE_FRAMES) for r in range(CUBE_ROWS)))

    def test_cube_is_projected_and_loops_with_no_seam(self):
        frames = [cube_frame(t) for t in range(CUBE_FRAMES)]
        # One step past either end of the loop draws the frame at the other end: the wrap is just another step.
        self.assertEqual(_cube_render(CUBE_FRAMES, _CUBE_FIT), frames[0])
        self.assertEqual(_cube_render(-1, _CUBE_FIT), frames[-1])
        self.assertGreaterEqual(len({tuple(frame) for frame in frames}), CUBE_FRAMES - 2)

        def cells(frame):
            return {(r, c): ch for r, line in enumerate(frame) for c, ch in enumerate(line) if ch != " "}

        sizes = [len(cells(frame)) for frame in frames]
        for t, frame in enumerate(frames):
            drawn = cells(frame)
            # Two or three faces, each its own character and its own ink, and the cube fills most of the box.
            glyphs = set(drawn.values())
            self.assertIn(len(glyphs), (2, 3), frame)
            self.assertEqual(len({CUBE_RAMP.index(ch) * 3 // len(CUBE_RAMP) for ch in glyphs}), len(glyphs), frame)
            # No specks: every cell touches another of its face, at least at a corner.
            for (r, c), ch in drawn.items():
                self.assertTrue(any(drawn.get((r + dr, c + dc)) == ch for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                    if dr or dc), (t, r, c))
            rows = {r for r, _ in drawn}
            cols = {c for _, c in drawn}
            self.assertGreaterEqual(max(rows) - min(rows) + 1, CUBE_ROWS - 3, frame)
            self.assertGreaterEqual(max(cols) - min(cols) + 1, CUBE_COLS // 2, frame)
            # Smooth: the outline barely changes from one frame to the next and the cube doesn't drift.
            after = cells(frames[(t + 1) % CUBE_FRAMES])
            self.assertLessEqual(len(set(drawn) ^ set(after)), len(drawn) // 6, (frame, frames[(t + 1) % CUBE_FRAMES]))
            centre = sum(c for _, c in drawn) / len(drawn), sum(r for r, _ in drawn) / len(drawn)
            moved = sum(c for _, c in after) / len(after), sum(r for r, _ in after) / len(after)
            self.assertLess(abs(centre[0] - moved[0]) + abs(centre[1] - moved[1]), 1.0)
        self.assertLess(max(sizes) - min(sizes), min(sizes) // 4)

    def test_cube_tumbles_about_more_than_one_axis(self):
        def transpose(m):
            return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))

        def step_axis(t):
            a, b = _cube_orientation(t), _cube_orientation(t + 1)
            r = tuple(tuple(sum(b[i][k] * a[j][k] for k in range(3)) for j in range(3)) for i in range(3))
            axis = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
            norm = math.sqrt(sum(x * x for x in axis))
            return tuple(x / norm for x in axis)

        axes = [step_axis(t) for t in range(CUBE_FRAMES)]
        spread = max(math.degrees(math.acos(max(-1.0, min(1.0, sum(p * q for p, q in zip(a, b))))))
                     for a in axes for b in axes)
        # Spinning about one fixed axis would keep every step's axis the same; this one swings through 50 degrees.
        self.assertGreater(spread, 45)
        seen, hidden, tops = set(), set(), set()
        for t in range(CUBE_FRAMES):
            m = _cube_orientation(t)
            for face, (normal, corners) in enumerate(_CUBE_FACE_LIST):
                turned, first = _mat_apply(m, normal), _mat_apply(m, corners[0])
                facing = sum(a * b for a, b in zip(turned, (first[0], first[1], first[2] + _CUBE_DISTANCE))) < 0
                (seen if facing else hidden).add(face)
            tops.add(max(range(6), key=lambda f: _mat_apply(m, _CUBE_FACE_LIST[f][0])[1]))
            # Never face-on: the view always meets a second face at a real angle, so it reads as a solid.
            view = sorted((abs(c) for c in _mat_apply(transpose(m), (0, 0, -1))), reverse=True)
            self.assertGreater(view[1], 0.5 * view[0], t)
        self.assertEqual(seen, set(range(6)))
        self.assertEqual(hidden, set(range(6)))
        self.assertGreaterEqual(len(tops), 4)

    def test_cube_rests_under_reduced_motion_and_keeps_its_place_when_relabelled(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(f"animation:polaris-cube {CUBE_SECONDS:g}s steps({CUBE_FRAMES}) infinite", css)
        reduced = [block.split("\n}", 1)[0] for block in css.split("@media (prefers-reduced-motion:reduce){")[1:]]
        self.assertTrue(any(".cube-reel{animation:none" in block for block in reduced))
        still = cube_html(still=True)
        self.assertIn('class="cube still"', still)
        self.assertEqual(still.count("<br>"), CUBE_ROWS - 1)
        self.assertIn("<br>".join(cube_frame(0)), re.sub(r'</?span(?: class="[db]")?>', "", still))
        self.assertNotIn("animation-delay", thinking_html("Waiting for the model"))
        later = thinking_html("Waiting for the model", elapsed=5.0)
        self.assertIn('<span class="cube-reel" style="animation-delay:-1.00s">', later)
        self.assertIn('class="thinking-note" style="animation-delay:10.0s"', later)
        noted = thinking_html("Writing its reasoning", elapsed=20.0, note="The answer shows from its Answer: line.")
        self.assertIn('<span class="thinking-note now">The answer shows from its Answer: line.</span>', noted)
        self.assertNotIn(STILL_WORKING, noted)
        self.assertIn(".thinking-note.now{opacity:1;animation:none}", css)

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
        self.assertIn("polaris-generating", flags_html("polaris", generating=True))
        self.assertNotIn("polaris-generating", flags_html("polaris"))
        self.assertEqual(
            flags_html("light", generating=True),
            '<div class="polaris-flags polaris-theme-light polaris-calm polaris-generating"></div>',
        )
        css = style_path().read_text(encoding="utf-8")
        rules = [r for r in css.split("}") if ":root:has(.polaris-generating)" in r and "play-state" in r]
        self.assertEqual(len(rules), 1)
        selector, body = rules[0].rsplit("{", 1)
        selector = selector.rsplit("*/", 1)[-1].strip()
        self.assertEqual(body, "animation-play-state:paused!important")
        # Only the star: the loader and the one-shot entrances must keep running.
        self.assertEqual(selector, ":root:has(.polaris-generating) .band *")

    def test_compare_answers_in_the_same_chat_shell(self):
        css = style_path().read_text(encoding="utf-8")
        rules = [r.rsplit("{", 1) for r in css.split("}") if "{" in r]
        widths = [body for sel, body in rules if "stMainBlockContainer" in sel and "max-width" in body]
        self.assertEqual(widths, ["max-width:var(--main-max);padding:calc(56px + var(--s-4)) var(--s-5) 0"])
        self.assertEqual(re.findall(r"--main-max:([^;]+);", css), ["880px"])
        # two columns stack before either is squeezed, and a code box scrolls inside its column
        self.assertIn('.st-key-cmp,[class*="st-key-cmp-hist"]{container-type:inline-size}', css)
        self.assertIn("@container (max-width:620px)", css)

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
        for html in (thinking_html("Searching the lab docs"), side_logo_html(True), empty_hero_html(), stopped_html()):
            self.assertNotIn('style="--', html)

    def test_no_theme_draws_a_starfield_behind_the_chat(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertNotRegex(css, r"\.sky\b|polaris-sky")
        for theme in THEMES:
            self.assertNotIn("sky", flags_html(theme))
        self.assertNotIn("sky", (ROOT / "app" / "app.py").read_text(encoding="utf-8"))

    def test_light_and_dark_paint_every_baseweb_portal_in_their_own_colours(self):
        # The portals carry .streamlit/config.toml's dark base on their wrappers, and Streamlit 1.45's select menu is
        # ul[data-testid=stSelectboxVirtualDropdown], not ul[role=listbox]: Light opened a dark model menu.
        rules = []
        for block in re.sub(r"/\*.*?\*/", "", style_path().read_text(encoding="utf-8"), flags=re.DOTALL).split("}"):
            if "{" in block:
                selectors, body = block.rsplit("{", 1)
                rules += [(s, body) for s in re.split(r",(?![^(]*\))", " ".join(selectors.split()))]
        portal = [(s.strip(), body) for s, body in rules if '[data-baseweb="popover"]' in s or '[data-baseweb="tooltip"]' in s]
        for part in ('[data-testid="stSelectboxVirtualDropdown"]', '[data-baseweb="menu"]', '[data-baseweb="tooltip"] > div',
                     '[data-baseweb="popover"]:not([data-testid="stPopoverBody"])'):
            painted = [s for s, body in portal if part in s and "background:" in body]
            self.assertTrue(painted, part)
        # Polaris keeps the portals it had: every rule that paints one (the option rows aside) is for Light and Dark only.
        for s, body in portal:
            if "background" in body and 'li[role="option"]' not in s:
                self.assertTrue(s.startswith(":root:has(.polaris-calm) "), s)

    def test_status_widget_stays_hidden(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn('[data-testid="stStatusWidget"]{display:none!important}', css)


class IdentityTests(unittest.TestCase):
    """The rule-based identity response: who-are-you questions get fixed text before retrieval; a question that only
    mentions a model or a maker goes to the model."""

    ASKS = (
        "Who are you?", "who are you, Polaris?", "Hey Polaris, what are you?", "hello, who am I talking to",
        "What model are you running?", "which model are you based on", "what model is this", "What LLM are you?",
        "What is your name", "What\u2019s your name?", "who made you", "Who trained you?", "who built you?",
        "which company created you", "who is your creator", "are you ChatGPT?", "Are you Qwen?",
        "introduce yourself", "please tell me about yourself", "what are you running on",
    )
    NEAR_MISSES = (
        "what model is my CPU", "who made ls", "who made linux", "who made the sudo command",
        "what model is this laptop", "what is the model of my wifi card", "how do I check what model my GPU is",
        "which model should I use", "what are you doing", "what can you do", "are you able to extract tar files",
        "who are the lab admins", "what is your name for the ls command", "tell me about yourself and ls",
        "who made you do that", "",
    )

    def test_identity_questions_match_in_any_case_and_punctuation(self):
        for question in self.ASKS:
            with self.subTest(question=question):
                self.assertTrue(identity.asks_identity(question))

    def test_questions_that_only_mention_a_model_or_a_maker_go_to_the_model(self):
        for question in self.NEAR_MISSES:
            with self.subTest(question=question):
                self.assertFalse(identity.asks_identity(question))

    def test_the_fixed_response_names_the_model_that_runs(self):
        self.assertEqual(
            identity.identity_response("thesis-student", "full"),
            "I'm Polaris, an offline Linux lab assistant. I run Thesis-Student, a Qwen3.5-2B model fine-tuned through "
            "knowledge distillation by Clyde Heinrich Lontoc and John Bert P. Cruz of Caraga State University.",
        )
        self.assertIn("I run my-student, a Qwen3.5-2B model", identity.identity_response("my-student", "full"))

    def test_plain_mode_and_the_stock_model_say_stock(self):
        stock = ("I'm Polaris, an offline Linux lab assistant. Right now I'm running the stock Qwen3.5-2B model with no "
                 "fine-tuning, set up by Clyde Heinrich Lontoc and John Bert P. Cruz of Caraga State University.")
        for model, mode in (("qwen3.5:2b", "base"), ("qwen3.5:2b", "full"), ("qwen3.5:2b", "cot"),
                            ("thesis-student", "base")):
            with self.subTest(model=model, mode=mode):
                self.assertEqual(identity.identity_response(model, mode), stock)

    def test_the_q4km_base_says_it_is_the_stock_model_quantised_in_every_mode(self):
        quantised = ("I'm Polaris, an offline Linux lab assistant. Right now I'm running the stock Qwen3.5-2B model "
                     "(Q4_K_M quantised, no fine-tuning), set up by Clyde Heinrich Lontoc and John Bert P. Cruz of Caraga "
                     "State University.")
        for mode in ("base", "cot", "rag", "full"):
            with self.subTest(mode=mode):
                text = identity.identity_response("thesis-base-q4km", mode)
                self.assertEqual(text, quantised)
                self.assertNotIn("fine-tuned", text)

    def test_only_the_app_uses_it(self):
        for path in (ROOT / "shared", ROOT / "rag", ROOT / "eval", ROOT / "kd"):
            for source in path.rglob("*.py"):
                with self.subTest(source=source.name):
                    self.assertNotRegex(source.read_text(encoding="utf-8"), r"app\.identity|asks_identity|identity_response")
        self.assertNotRegex((ROOT / "app" / "engine.py").read_text(encoding="utf-8"), r"identity")


class CopyFeedbackTests(unittest.TestCase):
    """The code boxes' copy button shows a check and "Copied" for 1.5 s after a click, the same in both themes."""

    def test_the_script_marks_any_copy_button_for_1_5_s_then_restores_it(self):
        js = copy_feedback_path().read_text(encoding="utf-8")
        self.assertIn("const COPIED_MS = 1500;", js)
        self.assertIn("const BUTTON = '[data-testid=\"stCodeCopyButton\"]';", js)
        # one listener on the document, in the capture phase, so boxes drawn later and in compare get it too
        self.assertIn('document.addEventListener("click"', js)
        self.assertIn("}, true);", js)
        self.assertIn("event.target.closest(BUTTON)", js)
        self.assertIn('button.setAttribute("data-copied", "");', js)
        self.assertIn('button.removeAttribute("data-copied");', js)
        self.assertIn("clearTimeout(button.polarisCopied);", js)
        self.assertIn('button.title = "Copy to clipboard";', js)
        self.assertIn('live.setAttribute("aria-live", "polite");', js)

    def test_the_loader_adds_each_script_to_the_page_once(self):
        for html, script_id, path in ((copy_feedback_html(), COPY_FEEDBACK_ID, copy_feedback_path()),
                                      (follow_html(), FOLLOW_ID, follow_path()),
                                      (tabs_script_html(), TABS_SCRIPT_ID, tabs_script_path())):
            with self.subTest(script=path.name):
                self.assertTrue(html.startswith("<script>") and html.endswith("</script>"))
                self.assertEqual(html.count("</script>"), 1)
                self.assertIn("window.parent.document", html)
                self.assertIn(f"if(doc.getElementById({json.dumps(script_id)}))return;", html)
                body = html.split("s.textContent=", 1)[1].split(";doc.head.appendChild", 1)[0]
                self.assertEqual(json.loads(body.replace("<\\/", "</")), path.read_text(encoding="utf-8"))

    def test_the_copied_state_is_one_rule_set_for_both_themes(self):
        css = style_path().read_text(encoding="utf-8")
        copied = [line for line in css.splitlines() if "[data-copied]" in line]
        self.assertTrue(copied)
        self.assertFalse([line for line in copied if "daylight" in line])
        rule = css.split('[data-testid="stCodeCopyButton"][data-copied]{', 1)[1].split("}", 1)[0]
        for ink in ("color:var(--term-cmd)", "background:var(--term-bg)", "var(--term-line)"):
            self.assertIn(ink, rule)
        self.assertIn('[data-testid="stCodeCopyButton"][data-copied] svg{display:none}', css)
        self.assertIn('[data-testid="stCodeCopyButton"][data-copied]::after{content:"Copied"}', css)
        self.assertIn("mask:var(--check)", css)
        self.assertIn("M3 8.5l3 3 7-7", css)
        self.assertIn('[data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-page-scripts){display:none}', css)


# Drives follow.js in Node against the few DOM pieces it touches: Streamlit's main section (scrollTop clamps to
# the content and a change queues a scroll event, as in a browser), the dock, the marker a send raises, and one
# frame per requestAnimationFrame.
FOLLOW_HARNESS = r"""
const factory = require(process.argv[1]);
let frames = [], scrolls = [], observer = null, marker = false, button = null;
const main = {
  _t: 0, _h: 800, clientHeight: 800, handler: null,
  get scrollHeight() { return this._h; },
  get scrollTop() { return this._t; },
  set scrollTop(v) {
    const t = Math.max(0, Math.min(v, this._h - this.clientHeight));
    if (t !== this._t) { this._t = t; scrolls.push(1); }
  },
  addEventListener(type, f) { this.handler = f; },
  removeEventListener() { this.handler = null; },
  getBoundingClientRect() { return { left: 200, width: 800, top: 0 }; },
};
const dock = { getBoundingClientRect() { return { left: 240, width: 720, top: 700 }; } };
const doc = {
  body: { appendChild(b) { button = b; } },
  scrollingElement: null,
  createElement() { return { style: {}, hidden: false, addEventListener(type, f) { this.onclick = f; } }; },
  querySelector(sel) {
    if (sel === ".polaris-generating") return marker ? {} : null;
    if (sel === '[data-testid="stMain"]') return main;
    if (sel === ".st-key-composer-dock") return dock;
    return null;
  },
};
const win = {
  innerHeight: 800,
  requestAnimationFrame(f) { frames.push(f); },
  addEventListener() {},
  MutationObserver: class { constructor(cb) { observer = cb; } observe() {} },
};
function settle() {
  while (frames.length || scrolls.length) {
    const f = frames; frames = []; f.forEach((g) => g());
    const s = scrolls; scrolls = []; s.forEach(() => main.handler && main.handler());
  }
}
function grow(px) { main._h += px; observer(); settle(); }
function user(top) { main.scrollTop = top; settle(); }
const gap = () => main._h - main._t - main.clientHeight;
const out = {};
factory(win, doc); settle();
grow(1200); out.no_send = main._t;
marker = true; grow(40); out.send_gap = gap();
grow(300); grow(300); out.stream_gap = gap();
main._h -= 200; main.scrollTop = main._t; settle(); grow(250); out.after_shrink_gap = gap();
const before = main._t; user(before - 300); const held = main._t;
grow(300); grow(300); out.held = main._t === held; out.held_gap = gap(); out.jump_shown = !button.hidden;
out.jump_left = button.style.left; out.jump_bottom = button.style.bottom;
user(main._t + gap() - 60); grow(300); out.resumed_gap = gap(); out.jump_hidden_again = button.hidden;
user(main._t - 30); grow(300); out.small_up_held = gap() > 0;
button.onclick(); settle(); out.clicked_gap = gap();
grow(100); out.clicked_stream_gap = gap();
marker = false; grow(80); out.last_batch_gap = gap();
grow(500); out.after_end_gap = gap(); out.jump_after_end = !button.hidden;
console.log(JSON.stringify(out));
"""


class FollowTests(unittest.TestCase):
    """While a sent question is answered the thread follows the stream; the reader scrolling up stops it."""

    def test_the_script_follows_streamlits_main_section_only_while_a_send_is_answered(self):
        js = follow_path().read_text(encoding="utf-8")
        self.assertIn("const SCROLLER = '[data-testid=\"stMain\"]';", js)
        self.assertIn('const LIVE = ".polaris-generating";', js)
        self.assertIn("const NEAR_PX = 80;", js)
        self.assertNotIn("window.scroll", js)
        self.assertNotIn("scrollIntoView", js)

    def test_the_jump_button_is_fixed_compact_and_themed(self):
        css = style_path().read_text(encoding="utf-8")
        rule = css.split(".polaris-jump{", 1)[1].split("}", 1)[0]
        for part in ("position:fixed", "font:400 var(--fs-ui)", "background:var(--c-bg)", "var(--c-line-strong)"):
            self.assertIn(part, rule)
        self.assertIn(".polaris-jump[hidden]{display:none}", css)
        self.assertNotIn("daylight", rule)

    def test_the_view_follows_a_send_and_lets_the_reader_scroll_away(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not installed")
        run = subprocess.run([node, "-e", FOLLOW_HARNESS, str(follow_path())], capture_output=True, text=True,
                             timeout=30, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        out = json.loads(run.stdout)
        # Content that grows without a send (a tab switch, Settings, the system check) leaves the view alone.
        self.assertEqual(out["no_send"], 0)
        # The send jumps to the bottom, every token keeps it there, and a redraw that shrinks the page doesn't count.
        self.assertEqual((out["send_gap"], out["stream_gap"], out["after_shrink_gap"]), (0, 0, 0))
        # Scrolled up mid-stream: the view stays put and the button shows over the dock.
        self.assertTrue(out["held"])
        self.assertGreater(out["held_gap"], 80)
        self.assertTrue(out["jump_shown"])
        self.assertEqual((out["jump_left"], out["jump_bottom"]), ("600px", "108px"))
        # Back within 80 px of the bottom, it follows again; any move up, however small, stops it.
        self.assertEqual(out["resumed_gap"], 0)
        self.assertTrue(out["jump_hidden_again"])
        self.assertTrue(out["small_up_held"])
        self.assertEqual((out["clicked_gap"], out["clicked_stream_gap"]), (0, 0))
        # The answer's last change still lands in view; after that the page is the reader's again.
        self.assertEqual(out["last_batch_gap"], 0)
        self.assertEqual(out["after_end_gap"], 500)
        self.assertFalse(out["jump_after_end"])


# Drives tabs.js in Node against a strip of equal 88 px tabs, 2 px apart, starting 100 px from the left. scrollLeft
# clamps to the content and a change queues a scroll event, as in a browser; one frame per requestAnimationFrame.
TABS_HARNESS = r"""
const factory = require(process.argv[1]);
let frames = [], observer = null, resize = null, n = 5, active = 4, attrs = {};
const buttons = [], LEFT = 100, TAB = 88, GAP = 2;
const strip = {
  _l: 0, clientWidth: 600, on: {},
  get scrollWidth() { return Math.max(this.clientWidth, n * (TAB + GAP) - GAP); },
  get scrollLeft() { return Math.min(this._l, this.scrollWidth - this.clientWidth); },
  set scrollLeft(v) {
    const l = Math.max(0, Math.min(v, this.scrollWidth - this.clientWidth));
    if (l !== this._l) { this._l = l; frames.push(() => this.on.scroll()); }
  },
  scrollBy(o) { this.scrollLeft = this.scrollLeft + o.left; },
  addEventListener(type, f) { this.on[type] = f; },
  setAttribute(k) { attrs[k] = 1; },
  removeAttribute(k) { delete attrs[k]; },
  getBoundingClientRect() { return { left: LEFT, right: LEFT + this.clientWidth, top: 10, height: 38 }; },
  querySelector() { return active === null ? null : { classList: ["x", "st-key-tab-on-c" + active], parentElement: tab(active) }; },
  querySelectorAll() { return { length: n }; },
};
function tab(i) {
  return { getBoundingClientRect() { const l = LEFT + i * (TAB + GAP) - strip.scrollLeft; return { left: l, right: l + TAB }; } };
}
const doc = {
  body: { appendChild(b) { buttons.push(b); } },
  createElement() { return { style: {}, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; }, addEventListener(t, f) { this.onclick = f; } }; },
  querySelector(sel) { return sel === ".st-key-tabs" ? strip : null; },
};
const win = {
  requestAnimationFrame(f) { frames.push(f); },
  addEventListener(type, f) { if (type === "resize") resize = f; },
  MutationObserver: class { constructor(cb) { observer = cb; } observe() {} },
};
function settle() { while (frames.length) { const f = frames; frames = []; f.forEach((g) => g()); } }
function redraw() { observer(); settle(); }
function shown(i) { const r = tab(i).getBoundingClientRect(); return r.left >= LEFT && r.right <= LEFT + strip.clientWidth; }
const state = () => ({ scroll: strip.scrollLeft, left: !buttons[0].hidden, right: !buttons[1].hidden, marks: Object.keys(attrs).sort() });
const out = {};
factory(win, doc); settle();
out.fit = state();
n = 25; active = 24; redraw();
out.added = state(); out.added_shown = shown(24);
out.chevrons = [buttons[0].style.top, buttons[0].style.left, buttons[1].style.left, buttons[0].attrs["aria-label"]];
strip.scrollLeft = 0; settle(); out.reader = state();
redraw(); out.kept = strip.scrollLeft;
active = 3; redraw(); out.select_shown = strip.scrollLeft;
active = 20; redraw(); out.select_far = shown(20);
out.select_far_clear = tab(20).getBoundingClientRect().right <= LEFT + strip.clientWidth - 32;
strip.scrollLeft = 0; settle();
const wheel = { deltaY: 100, deltaX: 0, preventDefault() { this.stopped = true; } };
strip.on.wheel(wheel); settle(); out.wheel = [strip.scrollLeft, !!wheel.stopped];
const sideways = { deltaY: 0, deltaX: 60, preventDefault() { this.stopped = true; } };
strip.on.wheel(sideways); out.sideways = [strip.scrollLeft, !!sideways.stopped];
strip.scrollLeft = 0; settle(); buttons[1].onclick(); settle(); out.click = strip.scrollLeft;
buttons[0].onclick(); settle(); out.click_back = strip.scrollLeft;
strip.scrollLeft = 1400; settle(); active = 2; redraw(); strip.scrollLeft = 1400; settle();
strip.clientWidth = 300; resize(); settle(); out.resized = shown(2);
n = 3; active = 2; strip.clientWidth = 600; redraw(); out.few = state();
console.log(JSON.stringify(out));
"""


class TabStripTests(unittest.TestCase):
    """Tabs shrink like a browser's to a floor, then the strip scrolls and the open tab is kept in view."""

    def setUp(self):
        self.css = style_path().read_text(encoding="utf-8")

    def rule(self, selector):
        return self.css.split(selector + "{", 1)[1].split("}", 1)[0]

    def test_tabs_shrink_evenly_to_a_floor_and_never_hide_or_wrap(self):
        tab = self.rule('.st-key-tabs > [data-testid="stVerticalBlockBorderWrapper"]')
        self.assertIn("flex:0 1 208px", tab)
        floor = int(re.search(r"min-width:(\d+)px", tab).group(1))
        self.assertTrue(72 <= floor <= 96, floor)
        on = self.rule('.st-key-tabs > [data-testid="stVerticalBlockBorderWrapper"]:has(> [class*="st-key-tab-on-"])')
        self.assertGreater(int(re.search(r"min-width:(\d+)px", on).group(1)), floor)
        strip = self.rule(".st-key-tabs")
        for part in ("flex-wrap:nowrap", "overflow:auto hidden", "scrollbar-width:none"):
            self.assertIn(part, strip)
        self.assertIn(".st-key-tabs::-webkit-scrollbar{display:none}", self.css)
        self.assertFalse([line for line in self.css.splitlines()
                          if '.st-key-tabs > [data-testid="stVerticalBlockBorderWrapper"]' in line and "display:none" in line])

    def test_new_chat_follows_the_strip_and_history_keeps_the_right_end(self):
        self.assertIn(".st-key-tabbar > :first-child{flex:0 1 auto;width:auto!important;min-width:0}", self.css)
        self.assertIn("flex:none", self.rule(".st-key-tabbar > :nth-child(n+2)"))
        self.assertIn(".st-key-tabbar > :has(> .st-key-tab-history){margin-left:auto}", self.css)

    def test_narrow_background_tabs_show_their_close_only_on_hover_or_focus(self):
        narrow = self.css.split("@container tab (max-width:139px){", 1)[1].split("\n}", 1)[0]
        self.assertIn('[class*="st-key-tab-off-"] [class*="st-key-tab-close-"] button{opacity:0', narrow)
        self.assertIn('[class*="st-key-tab-off-"]:is(:hover,:focus-within) [class*="st-key-tab-close-"] button{opacity:1}',
                      narrow)
        self.assertNotIn("st-key-tab-on-", narrow)
        self.assertIn('@media (hover:none){[class*="st-key-tab-off-"] [class*="st-key-tab-close-"] button{opacity:1!important}}',
                      self.css)

    def test_the_scroll_chevrons_are_small_fixed_and_themed(self):
        chevron = self.rule(".polaris-tab-scroll")
        for part in ("position:fixed", "width:24px", "height:24px", "background:var(--c-bg)", "color:var(--c-text-2)"):
            self.assertIn(part, chevron)
        self.assertIn(".polaris-tab-scroll[hidden]{display:none}", self.css)
        self.assertNotIn("daylight", chevron)

    def test_the_script_scrolls_only_the_strip(self):
        js = tabs_script_path().read_text(encoding="utf-8")
        self.assertIn('const STRIP = ".st-key-tabs";', js)
        # scrollIntoView would scroll the thread too.
        self.assertNotIn("scrollIntoView", js)
        self.assertNotIn("window.scroll", js)

    def test_the_strip_keeps_the_open_tab_in_view_and_scrolls_like_a_browsers(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not installed")
        run = subprocess.run([node, "-e", TABS_HARNESS, str(tabs_script_path())], capture_output=True, text=True,
                             timeout=30, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        out = json.loads(run.stdout)
        nothing = {"scroll": 0, "left": False, "right": False, "marks": []}
        self.assertEqual(out["fit"], nothing)
        # A new tab past the right end is scrolled in.
        self.assertTrue(out["added_shown"])
        self.assertEqual({k: out["added"][k] for k in ("left", "right", "marks")},
                         {"left": True, "right": False, "marks": ["data-more-left"]})
        self.assertEqual(out["chevrons"], ["17px", "100px", "676px", "Scroll tabs left"])
        # The reader's own scrolling stands until the open tab changes.
        self.assertEqual(out["reader"], {"scroll": 0, "left": False, "right": True, "marks": ["data-more-right"]})
        self.assertEqual((out["kept"], out["select_shown"]), (0, 0))
        # One out of sight is scrolled in clear of the chevron over that end.
        self.assertTrue(out["select_far"] and out["select_far_clear"])
        self.assertEqual(out["wheel"], [100, True])
        self.assertEqual(out["sideways"], [100, False])
        self.assertEqual((out["click"], out["click_back"]), (420, 0))
        self.assertTrue(out["resized"])
        self.assertEqual(out["few"], nothing)


class NavGridTests(unittest.TestCase):
    """Bar and rail controls share one height, icon size and gap so they line up in every theme."""

    def test_nav_buttons_share_height_icon_size_and_gap(self):
        css = style_path().read_text(encoding="utf-8")
        nav = ':is(.st-key-new-chat,.st-key-tab-history,[class*="st-key-settings"],[class*="st-key-syscheck-open"],.st-key-stop-rail)'
        self.assertIn(nav + " button{gap:6px;height:30px;min-height:30px}", css)
        self.assertIn(nav + ' [data-testid="stIconMaterial"]{font-size:18px}', css)
        square = css.split(':is(.st-key-new-chat,[class*="st-key-syscheck-open"]) button[data-testid]{', 1)[1].split("}", 1)[0]
        self.assertIn("width:30px", square)
        rail = ':is([data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]) button'
        self.assertIn(rail + "{display:flex;align-items:center;justify-content:center;width:30px;height:30px;padding:0}", css)
        self.assertIn(rail + " svg{width:18px;height:18px;margin:0}", css)
        self.assertIn('[class*="st-key-rail-foot"] > *{height:30px}', css)


class ReasoningStyleTests(unittest.TestCase):
    """The chain of thought opens by default above the answer and reads as secondary, never as a numbered step."""

    def test_the_reasoning_is_labelled_as_chain_of_thought(self):
        self.assertEqual(REASONING_TITLE, "Reasoning (chain of thought)")
        self.assertIn("chain of thought", REASONING_LABEL)

    def test_the_reasoning_is_muted_compact_and_indented_in_both_themes(self):
        css = style_path().read_text(encoding="utf-8")
        scope = ':is([class*="st-key-answer"],[class*="st-key-live-"]) [data-testid="stExpanderDetails"]'
        details = css.split(scope + "{", 1)[1].split("}", 1)[0]
        self.assertIn("border-left:2px solid var(--c-line)", details)
        self.assertIn("margin-left:var(--s-2)", details)
        text = css.split(scope + " :is(p,li){", 1)[1].split("}", 1)[0]
        self.assertIn("color:var(--c-text-2)", text)
        self.assertIn("font-size:var(--fs-reason)", text)
        size = float(re.search(r"--fs-reason:([\d.]+)px", css).group(1))
        self.assertTrue(13 <= size <= 15, size)
        self.assertLess(size, float(re.search(r"--fs-body:([\d.]+)px", css).group(1)))
        self.assertIn(scope + " li::marker{color:var(--c-text-3)", css)
        self.assertIn(scope + " code{background:transparent;color:var(--c-text-2)", css)
        # Light and Dark redefine the same tokens rather than restyling the reasoning
        for theme in ("light", "dark"):
            tokens = css.split(f":root:has(.polaris-theme-{theme}){{", 1)[1].split("}", 1)[0]
            for token in ("--c-text-2:", "--c-text-3:", "--c-line:"):
                self.assertIn(token, tokens)
        self.assertNotIn("stExpanderDetails", "".join(line for line in css.splitlines() if "polaris-theme-" in line))


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


class _Interrupted(BaseException):
    """Like Streamlit's StopException and RerunException, a BaseException, so no `except Exception` swallows it."""


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
        engine_body = ollama_chat_body(
            build_messages("hi", [], "base"),
            "qwen3.5:2b",
            temperature=0,
            seed=42,
            keep_alive="30m",
        )
        # The app's transport adds only its num_predict cap to what app.engine builds.
        self.assertEqual(_SlowOllama.bodies[0],
                         {**engine_body, "options": {**engine_body["options"], "num_predict": 512}})

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
        slot.markdown.assert_not_called()

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

    def _failed_card(self, error):
        st, app_mod = self._app_with_mock_st()
        slot = MagicMock()
        answer = _BlockingAnswer(error=error)
        with patch.object(app_mod, "stream_answer_question", return_value=answer), \
                patch.object(app_mod, "_model_info", return_value=None), patch("app.issues.backend", None):
            rec, run = self._answer(app_mod, slot)
        self.assertIsNone(rec)
        self.assertEqual(run["state"], "error")
        self.assertTrue(answer.closed)
        slot.error.assert_not_called()
        (html,) = slot.markdown.call_args.args
        return html, st.session_state.last_issue

    def test_a_model_not_found_in_the_stream_names_the_command_not_a_connection_failure(self):
        html, issue = self._failed_card(PlainStreamError("model 'm' not found"))
        self.assertEqual(issue.kind, "ollama_model_missing")
        self.assertIn("m isn&#x27;t installed", html)
        self.assertIn("<code>ollama pull m</code>", html)
        self.assertNotIn("isn&#x27;t running", html)

    def test_another_stream_error_says_ollama_returned_an_error(self):
        html, issue = self._failed_card(PlainStreamError("Ollama returned invalid NDJSON"))
        self.assertEqual(issue.kind, "ollama_response_error")
        self.assertIn("Ollama returned invalid NDJSON", html)

    def test_any_other_exception_is_a_card_not_a_traceback(self):
        html, issue = self._failed_card(KeyError("eval_count"))
        self.assertEqual(issue.kind, "unexpected")
        self.assertIn("KeyError while answering", html)
        self.assertNotIn("Traceback", html)

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
        # the loader is the first block's slot, which the answer's first block then takes
        first_block = st.container.return_value.empty.return_value
        loaders = [c.args[0] for c in first_block.markdown.call_args_list if 'class="thinking"' in c.args[0]]
        self.assertEqual(len(loaders), 1)
        self.assertIn("Waiting for the model…", loaders[0])
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
        self.assertIn("answer cut off at the length limit", cut_off_html({"done_reason": "length"}))
        for reason in ("stop", None, ""):
            self.assertEqual(cut_off_html({"done_reason": reason}), "")
        self.assertIn("cut off", metrics_html({"done_reason": "length", "mode": "base"}))
        self.assertEqual(build_run_record({"done_reason": "length", "mode": "base"})["done_reason"], "length")
        self.assertIsNone(build_run_record({"mode": "base"})["done_reason"])
        self.assertIn("done_reason", EXPORT_FIELDS)
        self.assertIn("token cap", cut_off_html({"done_reason": "length"}))

    def test_an_abstention_says_why_under_the_unchanged_text(self):
        html = idk_card_html({"answer": ABSTAIN}, "No documents were retrieved for this question.")
        self.assertIn('<p class="idk-why">No documents were retrieved for this question.</p>', html)
        self.assertNotIn("idk-why", idk_card_html({"answer": ABSTAIN}))

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
        # Polaris keeps Ubuntu Sans; Inter is bundled for Light and Dark only, at regular and medium.
        self.assertIn('@font-face{font-family:"Inter";src:url("/app/static/Inter-Variable-Latin.woff2") format("woff2");'
                      "font-weight:400 500;", css)
        self.assertTrue((style_path().parent / "static" / "Inter-Variable-Latin.woff2").is_file())
        self.assertTrue((style_path().parent / "static" / "Inter-OFL.txt").is_file())
        self.assertNotIn("Inter", css.split(":root{", 1)[1].split("}", 1)[0])
        self.assertIn('--f-sans:"Inter"', css.split(":root:has(.polaris-calm){", 1)[1].split("}", 1)[0])
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


# What the real index returned for "how do i edit files inside the terminal?" (top 5, dense 0.413-0.447).
EDIT_HITS = [
    {"rank": 1, "source": "tldr", "title": "nmtui-edit (tldr linux)", "path": "tldr/linux/nmtui-edit.md", "dense": 0.447},
    {"rank": 2, "source": "tldr", "title": "vim (tldr common)", "path": "tldr/common/vim.md", "dense": 0.441},
    {"rank": 3, "source": "tldr", "title": "git-unlock (tldr common)", "path": "tldr/common/git-unlock.md", "dense": 0.430},
    {"rank": 4, "source": "tldr", "title": "nano (tldr common)", "path": "tldr/common/nano.md", "dense": 0.422},
    {"rank": 5, "source": "ubuntu-server-docs", "title": "Basic commands", "path": "docs/tutorial/basic.md", "dense": 0.413},
]
EDIT_ANSWER = "Answer: Open the file with `vim notes.txt` [2].\n\nSource: [2]"


class SourceTitleTests(unittest.TestCase):
    """A cited [n] reads as the excerpt's name and the corpus it came from, as does each Evidence row."""

    def test_each_corpus_names_its_excerpts_the_way_a_reader_knows_them(self):
        cases = (
            ({"source": "tldr", "title": "nmtui-edit (tldr linux)"}, "nmtui-edit · tldr"),
            ({"source": "tldr", "title": "vim (tldr common)"}, "vim · tldr"),
            ({"source": "command-manpages", "title": "nano(1)"}, "nano(1) · man page"),
            ({"source": "man-pages", "title": "crontab(5)"}, "crontab(5) · man page"),
            ({"source": "ubuntu-docs", "title": "Edit text files"}, "Edit text files · Ubuntu help"),
            ({"source": "gnome-user-docs", "title": "Wi-Fi"}, "Wi-Fi · GNOME help"),
            ({"source": "ubuntu-server-docs", "title": "Basic commands"}, "Basic commands · Ubuntu Server docs"),
            ({"source": "arch-wiki", "title": "Nano"}, "Nano · Arch Wiki"),
            ({"title": "", "path": "tldr/common/tar.md"}, "tar"),
            ({}, ""),
        )
        for hit, label in cases:
            with self.subTest(hit=hit):
                self.assertEqual(excerpt_label(hit), label)
        self.assertEqual(corpus_label("some-new-corpus"), source_key("some-new-corpus").lower())

    def test_the_source_line_names_each_cited_excerpt_and_its_origin(self):
        parsed = parse_answer("Answer: Use `nano notes.txt` [4], or vim [2].\n\nSource: [2] [4]", "full", n_hits=5)
        line = source_line_html(parsed, "full", EDIT_HITS)
        self.assertEqual(line, '<p class="src-line">Source '
                               '<b>[2]</b> <span class="n">vim</span><span class="origin"> · tldr</span> · '
                               '<b>[4]</b> <span class="n">nano</span><span class="origin"> · tldr</span></p>')
        self.assertNotIn("(tldr", line)

    def test_a_number_past_the_retrieved_excerpts_is_muted_not_a_crash(self):
        parsed = parse_answer("Answer: Use vim [7].\n\nSource: [7]", "full", n_hits=5)
        line = source_line_html(parsed, "full", EDIT_HITS)
        self.assertIn('<span class="miss">removed [7], not among the 5 retrieved</span>', line)
        # Even a [n] that reaches the line with no excerpt behind it is shown muted, without a name.
        line = source_line_html(parsed.__class__(**{**parsed.__dict__, "cited": (2, 9)}), "full", EDIT_HITS)
        self.assertIn('<b>[2]</b> <span class="n">vim</span>', line)
        self.assertIn('<span class="miss">[9]</span>', line)
        self.assertIn('<span class="miss">[9]</span>', source_line_html(parsed.__class__(**{**parsed.__dict__, "cited": (9,)}), "full", []))

    def test_titles_are_escaped(self):
        parsed = parse_answer("Answer: x [1].\n\nSource: [1]", "full", n_hits=1)
        line = source_line_html(parsed, "full", [{"source": "tldr", "title": "<b>x</b>"}])
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", line)

    def test_evidence_rows_use_the_same_names(self):
        rows = evidence_rows_html(EDIT_HITS)
        self.assertIn('<span class="title">nmtui-edit<span class="origin"> · tldr</span></span>', rows)
        self.assertIn('<span class="title">Basic commands<span class="origin"> · Ubuntu Server docs</span></span>', rows)
        self.assertNotIn("(tldr", rows)

    def test_origins_and_misses_are_muted_in_every_theme(self):
        css = style_path().read_text(encoding="utf-8")
        self.assertIn(".src-line .miss{color:var(--c-text-3);opacity:.75}", css)
        self.assertIn(".ev .origin{color:var(--c-text-3)}", css)
        self.assertFalse([line for line in css.splitlines() if ("src-line ." in line or ".ev .origin" in line)
                          and "daylight" in line])


class AmbiguityHintTests(unittest.TestCase):
    """One muted line when the search found several near-equal weak matches; display only."""

    def record(self, **kw):
        return {"mode": "full", "hits": EDIT_HITS, "abstained": False, "answer": EDIT_ANSWER, **kw}

    def test_the_constants_are_the_documented_ones(self):
        self.assertEqual((AMBIGUOUS_K, AMBIGUOUS_SPREAD, AMBIGUOUS_TOP), (5, 0.05, 0.50))

    def test_near_equal_weak_matches_suggest_naming_the_top_retrieved_tool(self):
        self.assertEqual(ambiguity_hint_html(self.record()),
                         "<p class=\"amb-hint\">Search wasn't sure about this one. Naming the tool helps, e.g. "
                         "“how to use nmtui-edit”.</p>")

    def test_the_example_is_a_command_when_one_was_retrieved_else_a_generic_tool(self):
        man = [dict(h, source="command-manpages", title="nano(1)") if h["rank"] == 1 else h for h in EDIT_HITS]
        self.assertIn("“how to use nano”", ambiguity_hint_html(self.record(hits=man)))
        docs = [dict(h, source="ubuntu-docs", title=f"Page {h['rank']}") for h in EDIT_HITS]
        self.assertIn("“how to use tar”", ambiguity_hint_html(self.record(hits=docs)))

    def test_it_needs_a_tight_cluster_and_no_confident_match(self):
        def scored(top, bottom):
            step = (top - bottom) / 4
            return [dict(h, dense=round(top - i * step, 4)) for i, h in enumerate(EDIT_HITS)]
        self.assertTrue(ambiguity_hint_html(self.record(hits=scored(0.449, 0.401))))
        self.assertEqual(ambiguity_hint_html(self.record(hits=scored(0.45, 0.40))), "")  # spread is 0.05
        self.assertEqual(ambiguity_hint_html(self.record(hits=scored(0.50, 0.47))), "")  # top reaches 0.50
        self.assertEqual(ambiguity_hint_html(self.record(hits=EDIT_HITS[:4])), "")  # fewer than five scores
        self.assertEqual(ambiguity_hint_html(self.record(hits=[dict(h, dense=None) for h in EDIT_HITS])), "")

    def test_it_never_shows_on_abstentions_plain_modes_or_missing_hits(self):
        self.assertEqual(ambiguity_hint_html(self.record(abstained=True)), "")
        self.assertEqual(ambiguity_hint_html(self.record(answer=ABSTAIN)), "")
        self.assertEqual(ambiguity_hint_html(self.record(answer="Answer: I do not know.")), "")
        for mode in ("base", "cot"):
            self.assertEqual(ambiguity_hint_html(self.record(mode=mode)), "")
        self.assertEqual(ambiguity_hint_html(self.record(hits=[])), "")
        self.assertEqual(ambiguity_hint_html({}), "")

    def test_the_hint_is_compact_and_muted_in_every_theme(self):
        css = style_path().read_text(encoding="utf-8")
        rule = css.split("p.amb-hint{", 1)[1].split("}", 1)[0]
        self.assertIn("font:400 var(--fs-ui)", rule)
        self.assertIn("color:var(--c-text-3)", rule)
        self.assertNotIn("daylight", css.split("p.amb-hint{", 1)[0].rsplit("\n", 1)[1])


if __name__ == "__main__":
    unittest.main()
