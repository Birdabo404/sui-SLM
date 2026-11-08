#!/usr/bin/env python3
"""The Polaris run loop under Streamlit's AppTest. No live Ollama: answer_question is replaced."""
import csv
import json
import os
import sqlite3
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

import app.engine as engine
import app.polaris as polaris
import app.streaming as streaming
from app.answer_format import app_messages
from app.chat_memory import ChatStore
from app.engine import OllamaInfo
from shared.prompt import ABSTAIN

REAL_STREAM = streaming.stream_answer_question

# Imported while discovery loads modules, before other tests swap a MagicMock into sys.modules["streamlit"].
try:
    import streamlit
    from streamlit.logger import set_log_level
    from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
    from streamlit.runtime.scriptrunner_utils.exceptions import StopException
    from streamlit.testing.v1 import AppTest
except ImportError:
    streamlit = None

APP = str(ROOT / "app" / "app.py")


HIT = {"source": "tldr", "title": "tar", "path": "tldr/tar.md", "dense": 0.7, "rrf": 0.03, "bm25": 7.0, "text": "tar"}


def _fake_answer(calls, extra=None):
    def answer(question, mode, model, retriever=None, chat_fn=None, **kw):
        calls.append({"question": question, "mode": mode, "retriever": retriever})
        if extra is not None:
            extra.append({"model": model, "model_key": kw.get("model_key"), "context": kw.get("context")})
        return {
            "text": "Use `tar -xzf file.tar.gz`.",
            "hits": [dict(HIT)] if retriever is not None else [],
            "weak": False,
            "latency_s": 1.5,
            "config": "base",
            "mode": mode,
            "model": model,
            "model_key": kw.get("model_key", "base"),
            "question": question,
            "threshold": 0.3,
            "k": 5,
            "app_rss_mb": 100.0,
            "ollama_rss_mb": 900.0,
            "total_mb": 1000.0,
            "ts": "2026-10-24T00:00:00+00:00",
            "git_commit": "abc1234",
            "eval_count": 22,
            "eval_duration": 1_200_000_000,
            "tok_s": 18.33,
            "gen_s": 1.2,
            "prompt_s": 0.2,
            "load_s": 0.05,
        }

    return answer


class _Retriever:
    def __init__(self, index_dir):
        self.index_dir = index_dir

    def retrieve(self, query, k=5, tldr_lane=True):
        return [dict(HIT)]


def _ndjson(pieces, done_reason="stop"):
    return [
        *({"message": {"role": "assistant", "content": piece}, "done": False} for piece in pieces),
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": done_reason, "eval_count": 20,
         "eval_duration": 2_000_000_000, "prompt_eval_count": 40, "prompt_eval_duration": 500_000_000,
         "load_duration": 100_000_000},
    ]


class _NdjsonOllama(BaseHTTPRequestHandler):
    """/api/chat streaming `events`: by default two text deltas and Ollama's terminal record, `pause` seconds apart."""

    bodies = []
    events = _ndjson(["Use ", "`tar -xzf`."])
    pause = 0.0

    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        type(self).bodies.append(json.loads(self.rfile.read(size)))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in type(self).events:
            time.sleep(type(self).pause)
            self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
            self.wfile.flush()

    def log_message(self, *args):
        pass


class _FakeStream:
    """Stands in for app.streaming.StreamingAnswer. Iterating calls whatever app.engine.answer_question each test
    patched in, then hands its text out in pieces with the same retrieval and progress callbacks the real one makes,
    so a fake answer that raises raises mid-stream."""

    made = []

    def __init__(self, question, mode, model, *, retriever=None, on_retrieval=None, on_progress=None, **kw):
        self.question, self.mode, self.model, self.retriever = question, mode, model, retriever
        self.on_retrieval, self.on_progress, self.kw = on_retrieval, on_progress, kw
        self.closed = self.cancelled = False
        self.pieces = []
        self._result = None
        self._gen = None
        type(self).made.append(self)

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("streaming answer has not completed")
        return self._result

    def __iter__(self):
        return self

    def __next__(self):
        if self._gen is None:
            self._gen = self._run()
        return next(self._gen)

    def _run(self):
        result = engine.answer_question(
            self.question,
            self.mode,
            self.model,
            retriever=self.retriever,
            model_key=self.kw.get("model_key", "base"),
            context=self.kw.get("context"),
        )
        if engine.uses_rag(self.mode) and self.on_retrieval is not None:
            self.on_retrieval(streaming.retrieval_chart(result["hits"], result.get("threshold", 0.3), result["weak"]))
        text = result["text"]
        cut = [0, len(text) // 3, 2 * len(text) // 3, len(text)]
        for i, (a, b) in enumerate(zip(cut, cut[1:]), 1):
            if self.on_progress is not None:
                self.on_progress(streaming.LiveThroughput(0.5 * i, i, b, 2.0, b / (0.5 * i)))
            self.pieces.append(text[a:b])
            yield text[a:b]
        self._result = result

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class PolarisRunLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if streamlit is None:
            raise unittest.SkipTest("streamlit not installed")
        # Other test modules leave a MagicMock in sys.modules["streamlit"]; the script AppTest runs needs the real
        # package, and importing it a second time fails, so the copy captured above goes back in.
        cls._saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = streamlit
        streamlit.config.set_option("logger.level", "error")
        set_log_level("error")
        cls.streamlit = streamlit
        cls.AppTest = AppTest

    @classmethod
    def tearDownClass(cls):
        if cls._saved is not None:
            sys.modules["streamlit"] = cls._saved

    def setUp(self):
        self.calls = []
        self.seen = []
        self.flags = MagicMock(wraps=polaris.flags_html)
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        self.store = ChatStore(Path(data.name) / "sui-slm")
        _FakeStream.made = []
        patches = [
            patch.dict(os.environ, {"XDG_DATA_HOME": data.name}),
            patch("app.engine.answer_question", side_effect=_fake_answer(self.calls, self.seen)),
            patch("app.streaming.stream_answer_question", _FakeStream),
            patch("app.engine.ollama_reachable", return_value=True),
            patch("app.engine.ollama_info", return_value=OllamaInfo(version="0.12.0", model_digest="sha256:abc")),
            patch("app.engine.warm_model", return_value=True),
            patch("app.models.installed_names", return_value=frozenset({"qwen3.5:2b"})),
            patch("app.polaris.flags_html", self.flags),
            patch.object(self.streamlit, "rerun", side_effect=AssertionError("st.rerun() called")),
            # AppTest 1.45 reads a single-select st.pills with nothing picked as a multiselect and fails.
            patch.object(self.streamlit, "pills", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.rerun = self.streamlit.rerun
        # AppTest runs every test in one process, so a cached /api/tags answer would leak from one test to the next.
        self.streamlit.cache_data.clear()
        self.at = self.AppTest.from_file(APP, default_timeout=20)
        self.at.session_state["mode"] = "base"

    def _html(self):
        return "\n".join(m.value for m in self.at.markdown if not m.value.startswith("<style>"))

    def _settings_html(self):
        """Open the Settings dialog and return what it shows."""
        self.at.button(key="settings").click().run()
        self.assertFalse(self.at.exception)
        return self._html()

    def _flags(self):
        (flags,) = [m.value for m in self.at.markdown if m.value.startswith('<div class="polaris-flags')]
        return flags.split("<style>", 1)[0]

    def _generating_args(self):
        return [c.kwargs.get("generating") for c in self.flags.call_args_list]

    def _ask(self, question, where="dock"):
        self.at.chat_input(key=f"chat-{where}").set_value(question).run()
        self.assertFalse(self.at.exception)

    def test_landing_question_is_answered_in_the_run_it_arrives_in(self):
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertIn('class="hero"', self._html())
        self.assertEqual(self._generating_args(), [False])

        self.flags.reset_mock()
        self.at.chat_input(key="chat-landing").set_value("extract a .tar.gz file").run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        html = self._html()
        self.assertNotIn('class="hero"', html)
        self.assertIn("band slim", html)
        self.assertEqual(self.calls, [{"question": "extract a .tar.gz file", "mode": "base", "retriever": None}])
        # Paused for the answer, released in the same run once it lands.
        self.assertEqual(self._generating_args(), [True, False])
        self.assertEqual(self._flags(), '<div class="polaris-flags polaris-sky-1"></div>')
        self.assertIn('<span class="mode">BASE</span><span class="sub">Plain model, no documents</span>', html)
        self.assertIn("English questions only", html)
        self.assertNotIn('<span class="k">Answers</span>', html)
        settings = self._settings_html()
        self.assertIn('<span class="k">Answers</span><span class="v">1</span>', settings)
        self.assertIn('<span class="k">Warm-up</span><span class="v">ready</span>', settings)

    def test_second_question_refreshes_the_sidebar_without_duplicate_keys(self):
        self.at.run()
        self.at.chat_input(key="chat-landing").set_value("extract a .tar.gz file").run()
        self.at.chat_input(key="chat-dock").set_value("make a file readable only by me").run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(len(self.calls), 2)
        html = self._html()
        self.assertEqual(html.count('class="q"'), 2)
        self.assertEqual(self._flags(), '<div class="polaris-flags polaris-sky-2"></div>')
        self.assertIn('<span class="k">Answers</span><span class="v">2</span>', self._settings_html())

        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self._generating_args()[-1], False)

    def test_new_chat_returns_to_the_landing_in_one_run(self):
        self.at.run()
        self.at.chat_input(key="chat-landing").set_value("extract a .tar.gz file").run()
        self.at.button(key="new-chat").click().run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertIn('class="hero"', self._html())
        self.assertEqual(self.at.session_state["messages"], [])
        self.assertIn('<span class="k">Answers</span><span class="v">0</span>', self._settings_html())

    def test_a_turn_that_raises_still_clears_stop_and_releases_the_sky(self):
        self.at.run()
        self.flags.reset_mock()
        with patch("app.engine.answer_question", side_effect=RuntimeError("engine failed")):
            self.at.chat_input(key="chat-landing").set_value("extract a .tar.gz file").run()
        self.assertIn("engine failed", self.at.exception[0].message)
        self.assertEqual(self._generating_args(), [True, False])
        self.assertNotIn("stop-gen", [b.key for b in self.at.button])

    # ---------- answer stats ----------

    def test_each_answer_shows_time_and_speed_chips(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        html = self._html()
        self.assertIn('<span class="sr">Response time </span><b>1.5</b> s', html)
        self.assertIn('<span class="sr">Speed </span><b>18.3</b> tok/s', html)
        self.assertIn('title="22 tokens in 1.20 s of generation"', html)

    # ---------- streaming ----------

    def _speed_shown(self, html, tok_s, tokens, gen_s, latency="1.5"):
        self.assertIn(f'<span class="sr">Response time </span><b>{latency}</b> s', html)
        self.assertIn(f'<span class="sr">Speed </span><b>{tok_s}</b> tok/s', html)
        self.assertIn(f'<span class="stat calc">{tokens} tok / {gen_s} s</span>', html)

    def test_a_streamed_answer_keeps_ollamas_tok_s_after_it_ends_on_rerun_and_from_history(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        (stream,) = _FakeStream.made
        self.assertEqual(stream.pieces, ["Use `tar ", "-xzf file", ".tar.gz`."])
        self.assertTrue(stream.closed)
        html = self._html()
        # The rail's monitor now shows the finished answer: Ollama's numbers, and the trace this session recorded.
        self.assertNotIn("tok/s comes from Ollama when the answer ends", html)
        self.assertNotIn("mon-tag live", html)
        self.assertIn('<span class="mon-tag">last answer</span>', html)
        self._speed_shown(html, "18.3", 22, "1.20")
        self.assertIn("first text <b>0.5</b> s", html)
        self.assertNotIn('class="spark"', html.split('<div class="mon">', 1)[0])
        self.assertIn("Use `tar -xzf file.tar.gz`.", [m.value for m in self.at.markdown])

        self.at.run()
        self._speed_shown(self._html(), "18.3", 22, "1.20")
        self.assertEqual(len(self.calls), 1)

        cid = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertFalse(self.at.exception)
        html = self._html()
        self._speed_shown(html, "18.3", 22, "1.20")
        # The stream trace was this session's measurement, so a restored chat doesn't pretend to have one.
        self.assertNotIn("first text", html)
        self.assertIn("only recorded while an answer streams in this session", html)

    def test_both_compare_sides_keep_their_own_tok_s(self):
        answer = _fake_answer(self.calls)

        def per_side(question, mode, model, **kw):
            result = answer(question, mode, model, **kw)
            if mode == "cot":
                result.update(tok_s=9.75, eval_count=39, eval_duration=4_000_000_000, gen_s=4.0, latency_s=4.6)
            return result

        self._compare_base_vs_cot()
        with patch("app.engine.answer_question", side_effect=per_side):
            self._ask("extract a .tar.gz file", "landing")
        self.assertEqual(len(_FakeStream.made), 2)
        self.assertTrue(all(s.closed for s in _FakeStream.made))

        def both(html):
            self._speed_shown(html, "18.3", 22, "1.20")
            self._speed_shown(html, "9.8", 39, "4.00", latency="4.6")

        both(self._html())
        self.at.run()
        both(self._html())
        # An older compare turn collapses to one row per side; each still shows its own speed.
        self._ask("and a .zip?")
        both(self._html().split('class="q"', 2)[1])
        cid = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        both(self._html())

    def test_an_abstained_answer_says_nothing_was_generated(self):
        answer = _fake_answer(self.calls)

        def weak(question, mode, model, **kw):
            result = answer(question, mode, model, **kw)
            result.update(
                text=ABSTAIN,
                weak=True,
                hits=[dict(HIT, dense=0.12)],
                tok_s=None, eval_count=None, eval_duration=None, gen_s=None, prompt_s=None, load_s=None,
                latency_s=0.4,
            )
            return result

        rag = MagicMock(HybridRetriever=_Retriever)
        self.at.session_state["mode"] = "full"
        with patch("app.engine.index_ok", return_value=True), patch.dict(sys.modules, {"rag.query_hybrid": rag}), \
                patch("app.engine.answer_question", side_effect=weak):
            self.at.run()
            self._ask("what's the lab wi-fi password?", "landing")
        html = self._html()
        self.assertIn('<span class="sr">Response time </span><b>0.4</b> s', html)
        self.assertIn("tok/s not generated", html)
        self.assertNotRegex(html, r"<b>[\d.]+</b> tok/s")
        self.assertIn("best cos <b>0.120</b> below 0.30</span><span class=\"unit\">model not called</span>", html)
        self.assertIn('<span class="sr">rrf </span>.0300', html)

    def test_base_shows_no_retrieval_scores(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        html = self._html()
        self.assertIn("No retrieval in BASE", html)
        self.assertNotIn("rchart", html)
        self.assertNotIn("cosine", html)

    def test_a_restored_answer_without_timing_says_tok_s_unavailable(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        stored = list(self.store.load_chat(cid).messages)
        for field in ("tok_s", "eval_count", "eval_duration", "gen_s"):
            stored[1]["record"].pop(field)
        self.store.save_chat(cid, stored)
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        html = self._html()
        self.assertIn('<span class="sr">Response time </span><b>1.5</b> s', html)
        self.assertIn("tok/s unavailable", html)
        self.assertNotRegex(html, r"<b>[\d.]+</b> tok/s")

    def _ollama(self, events=None):
        _NdjsonOllama.bodies = []
        _NdjsonOllama.events = events or _ndjson(["Use ", "`tar -xzf`."])
        server = ThreadingHTTPServer(("127.0.0.1", 0), _NdjsonOllama)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def _ask_real(self, question, events=None, mode="base", dense=0.7):
        host = self._ollama(events)
        retriever = MagicMock(HybridRetriever=lambda index_dir: MagicMock(
            retrieve=lambda *a, **k: [dict(HIT, dense=dense)]))
        self.streamlit.cache_resource.clear()
        self.at.session_state["mode"] = mode
        with patch("app.streaming.stream_answer_question", REAL_STREAM), patch("app.streaming.OLLAMA_HOST", host), \
                patch("app.engine.LOG_CSV", Path(tempfile.mkdtemp()) / "answers.csv"), \
                patch("app.engine.index_ok", return_value=True), patch.dict(sys.modules, {"rag.query_hybrid": retriever}):
            self.at.run()
            self._ask(question, "landing")
        self.assertFalse(self.at.exception)

    def test_a_numbered_draft_before_answer_stays_under_show_reasoning(self):
        draft = ("Reasoning:\n1. The excerpts show `chmod 600` gives the owner read and write.\n"
                 "2. `ls -l` shows the result.\n3. A third draft line, longer than the answer.\n\nAnswer:\n"
                 "To make a file readable only by you:\n1. Remove group and other access:\n"
                 "```bash\nchmod 600 <path/to/file>\n```\n2. Check it:\n```bash\nls -l <path/to/file>\n```\nSource: [1]")
        self._ask_real("make a file readable only by me", _ndjson([draft[i:i + 7] for i in range(0, len(draft), 7)]),
                       mode="full")
        (body,) = _NdjsonOllama.bodies
        self.assertEqual(sorted(body["options"]), ["num_ctx", "temperature"])
        self.assertEqual((body["think"], body["options"]["temperature"]), (False, 0.2))
        (reasoning,) = self.at.expander
        self.assertEqual(reasoning.label, "Show reasoning")
        self.assertIn("The excerpts show", reasoning.markdown[0].value)
        answer = [m.value for m in self.at.markdown if m not in reasoning.markdown]
        self.assertFalse([m for m in answer if "excerpts" in m or "draft line" in m])
        self.assertEqual([c.value for c in self.at.code], ["chmod 600 <path/to/file>", "ls -l <path/to/file>"])
        self.assertEqual(self.store.load_chat(self._chat_id()).messages[1]["record"]["done_reason"], "stop")
        self.assertNotIn("cut off", self._html())

    def _record_deltas(self):
        """Every element the runs send from here on, in order, as (delta path, element type, text)."""
        seen = []

        def hook(msg):
            if msg.WhichOneof("type") == "delta" and msg.delta.WhichOneof("type") == "new_element":
                element = msg.delta.new_element
                kind = element.WhichOneof("type")
                text = element.markdown.body if kind == "markdown" else element.code.code_text if kind == "code" else ""
                seen.append((tuple(msg.metadata.delta_path), kind, text))

        ForwardMsgQueue.on_before_enqueue_msg(hook)
        self.addCleanup(ForwardMsgQueue.on_before_enqueue_msg, None)
        return seen

    @staticmethod
    def _loader_lifecycles(deltas):
        """Per answer card, in send order: its loader going on (with its label) and off, and its reasoning and answer
        blocks being drawn, with repeats folded. An answer view's children are the head (0), reasoning (1), blocks
        (2), loader (3) and footer (4)."""
        cards = sorted({path[:-1] for path, kind, text in deltas
                        if kind == "markdown" and text.startswith('<div class="card-head">')})
        lifecycles = []
        for card in cards:
            n = len(card)
            events, labels = [], []
            for path, kind, text in deltas:
                if path[:n] != card or len(path) == n:
                    continue
                child = path[n]
                if child == 3 and kind == "markdown" and 'class="thinking" role="status"' in text:
                    event = "loader"
                    labels.append(text.split('class="thinking-label">', 1)[1].split("…<", 1)[0])
                elif child == 3 and kind == "empty" and labels:
                    event = "off"
                elif child == 1 and kind == "markdown":
                    event = "reasoning"
                elif child == 2 and kind in ("markdown", "code") and text.strip():
                    event = "answer"
                else:
                    continue
                if not events or events[-1] != event:
                    events.append(event)
            lifecycles.append((events, list(dict.fromkeys(labels))))
        return lifecycles

    def _paced(self, text, *cuts, pause=0.15):
        """`text` streamed in pieces cut at `cuts`, `pause` seconds apart, so each piece is its own redraw."""
        _NdjsonOllama.pause = pause
        self.addCleanup(setattr, _NdjsonOllama, "pause", 0.0)
        bounds = (0, *cuts, len(text))
        return _ndjson([text[a:b] for a, b in zip(bounds, bounds[1:])])

    def test_the_cube_stays_through_the_reasoning_and_goes_when_the_answer_starts(self):
        deltas = self._record_deltas()
        cut = DRAFT.index("Answer:")
        self._ask_real("make a file readable only by me", self._paced(DRAFT, 30, 80, cut, cut + 40, cut + 90),
                       mode="cot")
        self.assertEqual(len(_NdjsonOllama.bodies), 1)
        ((events, labels),) = self._loader_lifecycles(deltas)
        self.assertEqual(events, ["loader", "reasoning", "loader", "reasoning", "off", "answer"])
        self.assertEqual(labels, ["Waiting for the model", polaris.REASONING_LABEL])
        self.assertNotIn('class="cube"', self._html())

    def test_a_cot_reply_without_an_answer_line_keeps_the_cube_until_it_ends(self):
        deltas = self._record_deltas()
        text = ("Reasoning:\n1. The excerpts mention `df -h` for free space per filesystem.\n2. `du -sh` sums one "
                "directory.\n\nRun `df -h` to see free space on every mounted filesystem.")
        self._ask_real("how much disk space is free", self._paced(text, 40, 90, 120), mode="cot")
        ((events, labels),) = self._loader_lifecycles(deltas)
        # Every streamed piece is held as reasoning; the finished reply then shows its last paragraph as the answer.
        self.assertEqual(events[-2:], ["off", "answer"])
        self.assertNotIn("answer", events[:-1])
        self.assertEqual(labels[-1], polaris.REASONING_LABEL)
        (reasoning,) = self.at.expander
        self.assertIn("du -sh", reasoning.markdown[0].value)
        self.assertIn("Run `df -h` to see free space on every mounted filesystem.", [m.value for m in self.at.markdown])

    def test_both_compare_sides_keep_the_cube_through_reasoning_and_drop_it_at_the_answer(self):
        deltas = self._record_deltas()
        cut = DRAFT.index("Answer:")
        host = self._ollama(self._paced(DRAFT, 30, 80, cut, cut + 40, cut + 90))
        index, rag = self._full_mode()
        self.streamlit.cache_resource.clear()
        with patch("app.streaming.stream_answer_question", REAL_STREAM), patch("app.streaming.OLLAMA_HOST", host), \
                patch("app.engine.LOG_CSV", Path(tempfile.mkdtemp()) / "answers.csv"), index, rag:
            self.at.run()
            self.at.toggle(key="compare").set_value(True).run()
            self.at.selectbox(key="cmp-left-config").set_value("qwen3.5:2b|cot").run()
            self.at.selectbox(key="cmp-right-config").set_value("qwen3.5:2b|full").run()
            del deltas[:]
            self._ask("make a file readable only by me", "landing")
        self.assertEqual(len(_NdjsonOllama.bodies), 2)
        left, right = self._loader_lifecycles(deltas)
        for events, labels in (left, right):
            self.assertEqual(events, ["loader", "reasoning", "loader", "reasoning", "off", "answer"])
            self.assertEqual(labels[-1], polaris.REASONING_LABEL)
        self.assertEqual(right[1][0], "Searching the lab docs")
        self.assertNotIn('class="cube"', self._html())

    def test_an_answer_cut_at_a_length_limit_says_so(self):
        self._ask_real("extract a .tar.gz file", _ndjson(["Use ", "`tar -x"], done_reason="length"))
        self.assertEqual(len(_NdjsonOllama.bodies), 1)
        self.assertIn("cut off at a length limit", self._html())

    def test_a_weak_retrieval_never_calls_the_model(self):
        deltas = self._record_deltas()
        self._ask_real("what's the lab wi-fi password?", mode="full", dense=0.1)
        self.assertEqual(_NdjsonOllama.bodies, [])
        ((events, labels),) = self._loader_lifecycles(deltas)
        self.assertEqual((events, labels), (["loader", "off", "answer"], ["Searching the lab docs"]))
        html = self._html()
        self.assertIn("No fix on this one", html)
        self.assertNotIn('class="cube"', html)
        self.assertEqual(self.at.code, [])

    def test_the_real_stream_renders_and_logs_through_the_app(self):
        host = self._ollama()
        log = Path(tempfile.mkdtemp()) / "answers.csv"
        with patch("app.streaming.stream_answer_question", REAL_STREAM), patch("app.streaming.OLLAMA_HOST", host), \
                patch("app.engine.LOG_CSV", log):
            self.at.run()
            self._ask("extract a .tar.gz file", "landing")
        self.assertFalse(self.at.exception)
        (body,) = _NdjsonOllama.bodies
        self.assertEqual((body["stream"], body["think"], body["options"]["num_ctx"]), (True, False, 4096))
        self.assertIn("Use `tar -xzf`.", [m.value for m in self.at.markdown])
        html = self._html()
        # 20 tokens over 2 s of eval_duration, exactly as Ollama's terminal record says.
        self._speed_shown(html, "10.0", 20, "2.00", latency=html.split('Response time </span><b>', 1)[1].split("<", 1)[0])
        (row,) = list(csv.DictReader(log.open(encoding="utf-8")))
        self.assertEqual(row["tok_s"], "10.0000")
        self.assertEqual(self.store.load_chat(self._chat_id()).messages[1]["record"]["tok_s"], 10.0)

    # ---------- chat history ----------

    def _rows(self):
        return [b for b in self.at.button if b.key and b.key.startswith("hist-open-")]

    def _chat_id(self, at=None):
        return (at or self.at).session_state["chat"]["id"]

    def _stored(self, chat_id):
        return [m.get("content") or m.get("record", {}).get("answer") for m in self.store.load_chat(chat_id).messages]

    def test_first_question_saves_a_chat_listed_as_current(self):
        self.at.run()
        self.assertIn("Chats you start are saved on this machine.", self._html())
        self.assertEqual(self.store.list_chats(), [])

        self._ask("extract a .tar.gz file", "landing")
        self.rerun.assert_not_called()
        (chat,) = self.store.list_chats()
        self.assertEqual((chat.title, chat.turn_count), ("extract a .tar.gz file", 1))
        stored = self.store.load_chat(chat.chat_id).messages
        self.assertEqual([m["role"] for m in stored], ["user", "assistant"])
        self.assertEqual(stored[1]["record"]["answer"], "Use `tar -xzf file.tar.gz`.")
        self.assertEqual(self._chat_id(), chat.chat_id)
        # Listed in the run that asked it, as the open chat; its row is disabled so a click can't stop an answer.
        (row,) = self._rows()
        self.assertEqual(row.key, f"hist-open-{chat.chat_id}")
        self.assertTrue(row.disabled)
        self.assertNotIn("Chats you start are saved", self._html())

    def test_new_chat_keeps_history_and_switching_restores_a_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.assertIsNone(self._chat_id())
        self._ask("make a file readable only by me", "landing")
        self._ask("and for a folder?")
        second = self._chat_id()
        self.assertNotEqual(first, second)
        self.assertEqual([c.chat_id for c in self.store.list_chats()], [second, first])
        self.assertEqual(self.store.list_chats()[0].turn_count, 2)

        self.at.button(key=f"hist-open-{first}").click().run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(self._chat_id(), first)
        self.assertEqual(self._html().count('class="q"'), 1)
        self.assertIn(">extract a .tar.gz file<", self._html())
        self.assertEqual(len(self.at.session_state["run_records"]), 1)
        self.assertTrue(self.at.button(key=f"hist-open-{first}").disabled)
        self.assertFalse(self.at.button(key=f"hist-open-{second}").disabled)
        self.assertIn('<span class="k">Answers</span><span class="v">1</span>', self._settings_html())

        # The restored chat carries on: the next turn sees the earlier one and is saved back to the same chat.
        self._ask("what about .zip?")
        self.assertEqual(self.seen[-1]["context"].turns[0].question, "extract a .tar.gz file")
        self.assertEqual(self.store.load_chat(first).messages[2]["content"], "what about .zip?")
        self.assertEqual(self.store.list_chats()[0].chat_id, first)

    def test_delete_asks_first_and_cancel_keeps_the_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        self.at.button(key=f"hist-del-{cid}").click().run()
        self.assertIn(polaris.history_note_html(polaris.DELETE_ASK), self._html())
        self.assertEqual(len(self.store.list_chats()), 1)
        self.at.button(key=f"hist-no-{cid}").click().run()
        self.assertNotIn("Delete this chat?", self._html())
        self.assertEqual(len(self.store.list_chats()), 1)
        self.assertEqual(self._chat_id(), cid)

        self.at.button(key=f"hist-del-{cid}").click().run()
        self.at.button(key=f"hist-yes-{cid}").click().run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(self.store.list_chats(), [])
        self.assertIsNone(self._chat_id())
        self.assertIn('class="hero"', self._html())
        self.assertIn("Chats you start are saved on this machine.", self._html())

    def test_deleting_another_chat_leaves_the_open_one(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self._ask("make a file readable only by me", "landing")
        second = self._chat_id()
        self.at.button(key=f"hist-del-{first}").click().run()
        self.at.button(key=f"hist-yes-{first}").click().run()
        self.assertEqual([c.chat_id for c in self.store.list_chats()], [second])
        self.assertEqual(self._chat_id(), second)
        self.assertIn(">make a file readable only by me<", self._html())

    def test_a_chat_deleted_elsewhere_continues_in_a_new_one(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        gone = self._chat_id()
        self.store.delete_chat(gone)
        self._ask("and a .zip?")
        (chat,) = self.store.list_chats()
        self.assertNotEqual(chat.chat_id, gone)
        self.assertEqual(self._chat_id(), chat.chat_id)
        self.assertEqual(chat.turn_count, 2)

    def test_two_tabs_on_one_chat_keep_each_others_turns(self):
        answer = "Use `tar -xzf file.tar.gz`."
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        other = self.AppTest.from_file(APP, default_timeout=20)
        other.session_state["mode"] = "base"
        other.run()
        other.button(key=f"hist-open-{cid}").click().run()
        other.chat_input(key="chat-dock").set_value("and a .zip?").run()
        self.assertFalse(other.exception)
        # This tab never saw the other tab's turn; its next turn is added after it, not written over it.
        self._ask("and a .tar.xz?")
        self.assertEqual(
            self._stored(cid),
            ["extract a .tar.gz file", answer, "and a .zip?", answer, "and a .tar.xz?", answer],
        )
        self.assertEqual(self.store.list_chats()[0].turn_count, 3)
        other.chat_input(key="chat-dock").set_value("and a .7z?").run()
        self.assertEqual(self._stored(cid)[-2:], ["and a .7z?", answer])
        self.assertEqual(len(self._stored(cid)), 8)

    def test_an_answer_whose_chat_is_deleted_mid_turn_moves_to_a_new_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        gone = self._chat_id()
        answer = _fake_answer(self.calls)

        def delete_then_answer(*args, **kw):
            self.store.delete_chat(gone)
            return answer(*args, **kw)

        with patch("app.engine.answer_question", side_effect=delete_then_answer):
            self._ask("and a .zip?")
        (chat,) = self.store.list_chats()
        self.assertNotEqual(chat.chat_id, gone)
        self.assertEqual(self._chat_id(), chat.chat_id)
        self.assertEqual(chat.turn_count, 2)
        self.assertEqual(
            self._stored(chat.chat_id),
            ["extract a .tar.gz file", "Use `tar -xzf file.tar.gz`.", "and a .zip?", "Use `tar -xzf file.tar.gz`."],
        )
        # Later turns go to the replacement chat.
        self._ask("and a .7z?")
        self.assertEqual(self.store.list_chats()[0].turn_count, 3)

    def test_a_turn_that_raises_keeps_its_question_in_an_existing_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        with patch("app.engine.answer_question", side_effect=RuntimeError("engine failed")):
            self.at.chat_input(key="chat-dock").set_value("and a .zip?").run()
        self.assertIn("engine failed", self.at.exception[0].message)
        self.assertEqual(self._stored(cid), ["extract a .tar.gz file", "Use `tar -xzf file.tar.gz`.", "and a .zip?"])

    def test_turns_missed_while_history_is_unwritable_are_stored_with_the_next_one(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        with patch.object(ChatStore, "append_messages", side_effect=sqlite3.OperationalError("database is locked")):
            self._ask("and a .zip?")
        self.assertEqual(len(self._stored(cid)), 2)
        # The sidebar is drawn before the turn ends, so the note shows from the next run.
        self.at.run()
        self.assertIn("Couldn&#x27;t open chat history in", self._html())
        self._ask("and a .7z?")
        self.assertEqual(self._stored(cid)[2::2], ["and a .zip?", "and a .7z?"])
        self.assertEqual(len(self._stored(cid)), 6)
        self.at.run()
        self.assertNotIn("Couldn&#x27;t open chat history in", self._html())

    def test_a_stopped_turn_is_saved_with_its_question(self):
        self.at.run()
        with patch("app.engine.answer_question", side_effect=polaris.ChatCancelled()):
            self._ask("extract a .tar.gz file", "landing")
        (chat,) = self.store.list_chats()
        self.assertEqual(
            list(self.store.load_chat(chat.chat_id).messages),
            [{"role": "user", "content": "extract a .tar.gz file"}],
        )
        self.at.run()
        self.assertIn("Stopped. The partial answer was not kept.", self._html())

    def test_answers_still_work_when_history_cannot_be_written(self):
        with patch.object(ChatStore, "_connect", side_effect=sqlite3.OperationalError("disk I/O error")):
            self.at.run()
            self._ask("extract a .tar.gz file", "landing")
        self.rerun.assert_not_called()
        self.assertEqual(len(self.calls), 1)
        html = self._html()
        self.assertIn("<b>1.5</b> s", html)
        self.assertIn("Couldn&#x27;t open chat history in", html)
        self.assertIsNone(self._chat_id())
        self.assertEqual(self._rows(), [])

    def test_the_theme_toggle_says_light_mode_and_keeps_the_stored_key(self):
        self.at.run()
        toggle = self.at.toggle(key="daylight")
        self.assertEqual(toggle.label, "Light mode")

        def flags():
            return next(m.value for m in self.at.markdown if m.value.startswith('<div class="polaris-flags'))

        self.assertNotIn("polaris-daylight", flags())
        toggle.set_value(True).run()
        self.assertTrue(self.at.session_state["daylight"])
        self.assertIn("polaris-daylight", flags())
        shown = [t.label for t in self.at.toggle] + [m.value for m in self.at.markdown if "<style" not in m.value]
        self.assertFalse(any("Daylight" in s for s in shown))

    # ---------- compare ----------

    def test_compare_starts_off_with_one_picker_per_side(self):
        self.at.run()
        self.assertFalse(self.at.toggle(key="compare").value)
        self.assertEqual(self.at.toggle(key="compare").label, "Compare two configs")
        self.assertNotIn("cmp-left-config", [s.key for s in self.at.selectbox])

        self.at.toggle(key="compare").set_value(True).run()
        left, right = self.at.selectbox(key="cmp-left-config"), self.at.selectbox(key="cmp-right-config")
        self.assertEqual(
            left.options[:4],
            ["qwen3.5:2b · BASE", "qwen3.5:2b · +CoT", "qwen3.5:2b · +RAG", "qwen3.5:2b · +RAG+CoT"],
        )
        # The other models are offered as they are on this machine: listed, and marked when Ollama lacks them.
        self.assertIn("thesis-base-q4km · +RAG · not installed", left.options)
        self.assertEqual(left.options[-1], "thesis-student · +RAG+CoT · not installed")
        self.assertNotIn("thesis-student · BASE · not installed", left.options)
        self.assertEqual((left.value, right.value), ("qwen3.5:2b|base", "qwen3.5:2b|full"))
        self.assertNotIn("qwen3.5:2b · BASE", right.options)
        self.assertIn('<span class="m">qwen3.5:2b</span> <b>BASE</b><span class="vs">vs</span><b>+RAG+CoT</b>', self._html())

        right.set_value("qwen3.5:2b|cot").run()
        self.assertEqual(self.at.session_state["cmp_right_pick"], "qwen3.5:2b|cot")
        self._ask("extract a .tar.gz file", "landing")
        self.assertEqual([c["mode"] for c in self.calls], ["base", "cot"])
        self.assertNotIn('class="model"', self._html())
        # Still the same picks on the next run, so the browser never falls back to a default.
        self.at.run()
        self.assertEqual(self.at.selectbox(key="cmp-right-config").value, "qwen3.5:2b|cot")

    def test_compare_across_models_names_each_side(self):
        rag = MagicMock(HybridRetriever=_Retriever)
        with patch("app.engine.DISTILLED_MODEL", "student:latest"), patch("app.engine.index_ok", return_value=True), \
                patch.dict(sys.modules, {"rag.query_hybrid": rag}), \
                patch("app.models.installed_names", return_value=frozenset({"qwen3.5:2b", "student:latest"})):
            self.at.run()
            self.at.toggle(key="compare").set_value(True).run()
            left, right = self.at.selectbox(key="cmp-left-config"), self.at.selectbox(key="cmp-right-config")
            self.assertEqual(left.options[-1], "student:latest · +RAG+CoT")
            self.assertEqual((left.value, right.value), ("qwen3.5:2b|full", "student:latest|full"))
            self._ask("extract a .tar.gz file", "landing")
        self.assertFalse(self.at.exception)
        self.assertEqual(
            [(s["model"], s["model_key"]) for s in self.seen],
            [("qwen3.5:2b", "base"), ("student:latest", "distilled")],
        )
        html = self._html()
        self.assertIn('<span class="mode">+RAG+CoT</span><span class="model">qwen3.5:2b</span>', html)
        self.assertIn('<span class="mode">+RAG+CoT</span><span class="model">student:latest</span>', html)
        self.assertIn(
            '<span class="m">qwen3.5:2b</span> <b>+RAG+CoT</b><span class="vs">vs</span>'
            '<span class="m">student:latest</span> <b>+RAG+CoT</b>',
            html,
        )
        (chat,) = self.store.list_chats()
        left_rec, right_rec = self.store.load_chat(chat.chat_id).messages[1]["compare"]
        self.assertEqual((left_rec["model"], right_rec["model"]), ("qwen3.5:2b", "student:latest"))

    def test_a_compare_side_that_fails_shows_its_error_live_on_rerun_and_from_history(self):
        answer = _fake_answer(self.calls)

        def left_fails(question, mode, model, **kw):
            if mode == "base":
                raise requests.ConnectionError("connection refused")
            return answer(question, mode, model, **kw)

        self.at.run()
        self.at.toggle(key="compare").set_value(True).run()
        self.at.selectbox(key="cmp-right-config").set_value("qwen3.5:2b|cot").run()
        with patch("app.engine.answer_question", side_effect=left_fails):
            self._ask("extract a .tar.gz file", "landing")
        self.assertEqual([c["mode"] for c in self.calls], ["cot"])
        self.assertNotIn("last_error", self.at.session_state)

        def shown():
            (error,) = self.at.error
            self.assertIn("Could not reach Ollama", error.value)
            self.assertIn("connection refused", error.value)
            html = self._html()
            # The failed side keeps its config head, so it's clear which one failed.
            self.assertIn('<span class="mode">BASE</span>', html)
            self.assertIn('<span class="mode">+CoT</span>', html)
            self.assertIn("Use `tar -xzf file.tar.gz`.", [m.value for m in self.at.markdown])

        shown()
        cid = self._chat_id()
        (reply,) = self.store.load_chat(cid).messages[1:]
        self.assertIsNone(reply["compare"][0])
        self.assertEqual(reply["compare"][1]["mode"], "cot")
        failed, ok = reply["compare_unanswered"]
        self.assertEqual((failed["state"], failed["mode"], failed["model"], ok), ("error", "base", "qwen3.5:2b", None))
        self.assertIn("connection refused", failed["error"])

        self.at.run()
        shown()
        self.at.button(key="new-chat").click().run()
        self.assertEqual(self.at.error, [])
        self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertFalse(self.at.exception)
        shown()

    def _compare_base_vs_cot(self):
        self.at.run()
        self.at.toggle(key="compare").set_value(True).run()
        self.at.selectbox(key="cmp-right-config").set_value("qwen3.5:2b|cot").run()

    def _answers_left_then_raises(self, error):
        answer = _fake_answer(self.calls)

        def answer_left_only(question, mode, model, **kw):
            if mode == "cot":
                raise error
            return answer(question, mode, model, **kw)

        return answer_left_only

    def _left_answer_and_right_cancelled_shown(self):
        html = self._html()
        self.assertIn("Use `tar -xzf file.tar.gz`.", [m.value for m in self.at.markdown])
        self.assertIn('<span class="mode">+CoT</span>', html)
        self.assertEqual(html.count(polaris.cancelled_side_html()), 1)
        self.assertNotIn("Stopped. The partial answer was not kept.", html)
        self.assertEqual(self.at.error, [])

    def _left_answer_and_right_cancelled_stored(self):
        (reply,) = self.store.load_chat(self._chat_id()).messages[1:]
        self.assertEqual(reply["compare"][0]["mode"], "base")
        self.assertIsNone(reply["compare"][1])
        self.assertEqual(reply["compare_unanswered"], [None, {"mode": "cot", "model": "qwen3.5:2b", "state": "cancelled"}])

    def test_a_cancelled_right_side_shows_cancelled_live_on_rerun_and_from_history(self):
        self._compare_base_vs_cot()
        with patch("app.engine.answer_question", side_effect=self._answers_left_then_raises(polaris.ChatCancelled())):
            self._ask("extract a .tar.gz file", "landing")
        self._left_answer_and_right_cancelled_shown()
        self._left_answer_and_right_cancelled_stored()
        self.at.run()
        self._left_answer_and_right_cancelled_shown()
        cid = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertFalse(self.at.exception)
        self._left_answer_and_right_cancelled_shown()

    def test_a_click_that_ends_the_run_mid_compare_keeps_the_left_answer(self):
        # A Stop click, or any other click, ends the run at its next interrupt check by raising a Streamlit control
        # exception, as StopException does here, rather than ChatCancelled.
        self._compare_base_vs_cot()
        with patch("app.engine.answer_question", side_effect=self._answers_left_then_raises(StopException())):
            self._ask("extract a .tar.gz file", "landing")
        self._left_answer_and_right_cancelled_stored()
        self.at.run()
        self._left_answer_and_right_cancelled_shown()

    def test_the_left_answer_is_in_the_thread_while_the_right_side_runs(self):
        # A click starts the next run straight away, while this one is still unwinding, and that run draws the
        # thread as it is then.
        self._compare_base_vs_cot()
        thread = self.at.session_state["messages"]
        answer = _fake_answer(self.calls)
        during_right = []

        def answer_and_look(question, mode, model, **kw):
            if mode == "cot":
                during_right.extend(dict(m) for m in thread)
            return answer(question, mode, model, **kw)

        with patch("app.engine.answer_question", side_effect=answer_and_look):
            self._ask("extract a .tar.gz file", "landing")
        _, reply = during_right
        self.assertEqual((reply["compare"][0]["mode"], reply["compare"][1]), ("base", None))
        self.assertEqual(reply["compare_unanswered"], (None, {"mode": "cot", "model": "qwen3.5:2b", "state": "cancelled"}))
        (stored,) = self.store.load_chat(self._chat_id()).messages[1:]
        self.assertEqual([rec["mode"] for rec in stored["compare"]], ["base", "cot"])
        self.assertNotIn("compare_unanswered", stored)

    def test_a_compare_cancelled_on_the_left_never_starts_the_right_and_is_kept_as_stopped(self):
        self._compare_base_vs_cot()
        with patch("app.engine.answer_question", side_effect=polaris.ChatCancelled()) as answer:
            self._ask("extract a .tar.gz file", "landing")
        self.assertEqual(answer.call_count, 1)
        # Both sides read Cancelled under their own heads; with no answer on either, the turn is kept as stopped.
        html = self._html()
        self.assertEqual(html.count(polaris.cancelled_side_html()), 2)
        self.assertIn('<span class="mode">BASE</span>', html)
        self.assertIn('<span class="mode">+CoT</span>', html)
        self.assertEqual(
            list(self.store.load_chat(self._chat_id()).messages),
            [{"role": "user", "content": "extract a .tar.gz file"}],
        )
        self.at.run()
        self.assertIn("Stopped. The partial answer was not kept.", self._html())


    # ---------- model picker ----------

    def test_the_picker_lists_each_model_and_keeps_the_default(self):
        self.at.run()
        picker = self.at.selectbox(key="model-select")
        self.assertEqual(picker.value, "qwen3.5:2b")
        self.assertEqual(
            picker.options,
            ["qwen3.5:2b", "thesis-base-q4km · not installed", "thesis-student · not installed"],
        )
        self.assertEqual(self.at.radio(key="mode-pick").options, ["BASE", "+CoT", "+RAG", "+RAG+CoT"])
        self.assertIn('<span class="ok">installed</span> · default base', self._html())

    def test_a_missing_model_says_how_to_create_it_and_nothing_is_called(self):
        warm = self.enterContext(patch("app.engine.warm_model", return_value=True))
        self.at.run()
        self.at.selectbox(key="model-select").set_value("thesis-base-q4km").run()
        self.assertEqual(self.at.session_state["model_pick"], "thesis-base-q4km")
        html = self._html()
        self.assertIn("not installed", html)
        self.assertIn("ollama create thesis-base-q4km -f Modelfile-base", html)
        self._ask("extract a .tar.gz file", "landing")
        (error,) = self.at.error
        self.assertIn("thesis-base-q4km isn't installed in Ollama on this machine", error.value)
        self.assertIn("`ollama create thesis-base-q4km -f Modelfile-base`", error.value)
        # Never answered by another model in its place, never warmed, never pulled.
        self.assertEqual((self.calls, _FakeStream.made), ([], []))
        self.assertNotIn("thesis-base-q4km", [c.args[0] for c in warm.call_args_list])
        (_, reply) = self.store.load_chat(self._chat_id()).messages
        self.assertIn("isn't installed", reply["error"])

    def test_a_model_created_after_the_page_loaded_is_found_on_ask(self):
        self.at.run()
        self.at.selectbox(key="model-select").set_value("thesis-base-q4km").run()
        with patch("app.models.installed_names", return_value=frozenset({"qwen3.5:2b", "thesis-base-q4km:latest"})):
            self._ask("extract a .tar.gz file", "landing")
        self.assertEqual(self.at.error, [])
        self.assertEqual([(s["model"], s["model_key"]) for s in self.seen], [("thesis-base-q4km", "base")])

    def test_the_student_runs_full_only_under_its_real_name(self):
        rag = MagicMock(HybridRetriever=_Retriever)
        with patch("app.models.installed_names", return_value=frozenset({"qwen3.5:2b", "thesis-student:latest"})), \
                patch("app.engine.index_ok", return_value=True), patch.dict(sys.modules, {"rag.query_hybrid": rag}):
            self.at.run()
            self.at.selectbox(key="model-select").set_value("thesis-student").run()
            self.assertEqual(self.at.radio(key="mode-pick").options, ["+RAG+CoT"])
            self.assertIn("runs +RAG+CoT only", self._html())
            self._ask("extract a .tar.gz file", "landing")
        self.assertFalse(self.at.exception)
        self.assertEqual([c["mode"] for c in self.calls], ["full"])
        self.assertEqual([(s["model"], s["model_key"]) for s in self.seen], [("thesis-student", "distilled")])
        # The base model is still what a new session starts on.
        self.assertEqual(engine.BASE_MODEL, "qwen3.5:2b")

    def test_the_env_override_names_the_student(self):
        with patch("app.engine.DISTILLED_MODEL", "my-student:q4"):
            self.at.run()
        self.assertEqual(
            self.at.selectbox(key="model-select").options,
            ["qwen3.5:2b", "thesis-base-q4km · not installed", "my-student:q4 · not installed"],
        )

    def test_without_ollama_the_picker_says_it_cannot_check(self):
        with patch("app.engine.ollama_reachable", return_value=False):
            self.at.run()
        self.assertIn("can't check, Ollama is not running", self._html())
        self.assertEqual(self.at.selectbox(key="model-select").options[1], "thesis-base-q4km")

    # ---------- tabs ----------

    def _tab_keys(self):
        return [b.key for b in self.at.button if b.key and b.key.startswith("tab-open-")]

    def test_each_chat_gets_a_tab_and_switching_restores_it(self):
        self.at.run()
        self.assertEqual(self._tab_keys(), ["tab-open-draft"])
        self.assertNotIn("tab-close-draft", [b.key for b in self.at.button])
        self._ask("extract a .tar.gz file", "landing")
        first = self._chat_id()
        self.assertEqual(self._tab_keys(), [f"tab-open-{first}"])
        self.assertTrue(self.at.button(key=f"tab-open-{first}").disabled)

        self.at.button(key="new-chat").click().run()
        self.assertEqual(self._tab_keys(), [f"tab-open-{first}", "tab-open-draft"])
        self.assertIn('class="hero"', self._html())
        self._ask("make a file readable only by me", "landing")
        second = self._chat_id()
        self.assertEqual(self._tab_keys(), [f"tab-open-{first}", f"tab-open-{second}"])

        self.at.button(key=f"tab-open-{first}").click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self._chat_id(), first)
        self.assertIn(">extract a .tar.gz file<", self._html())
        self.assertNotIn(">make a file readable only by me<", self._html())
        self.assertEqual(len(self.calls), 2)

    def test_closing_the_open_tab_moves_to_its_neighbour_and_keeps_the_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self._ask("make a file readable only by me", "landing")
        second = self._chat_id()

        self.at.button(key=f"tab-close-{second}").click().run()
        self.assertFalse(self.at.exception)
        self.assertEqual(self._chat_id(), first)
        self.assertEqual(self._tab_keys(), [f"tab-open-{first}"])
        self.assertEqual(len(self.store.list_chats()), 2)
        # Still in History, and opening it from there brings its tab back.
        self.at.button(key=f"hist-open-{second}").click().run()
        self.assertEqual(self._tab_keys(), [f"tab-open-{first}", f"tab-open-{second}"])

        self.at.button(key=f"tab-close-{first}").click().run()
        self.at.button(key=f"tab-close-{second}").click().run()
        self.assertIsNone(self._chat_id())
        self.assertIn('class="hero"', self._html())
        self.assertEqual(self._tab_keys(), ["tab-open-draft"])
        self.assertEqual(len(self.store.list_chats()), 2)

    def test_closing_a_background_tab_leaves_the_open_chat_alone(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self._ask("make a file readable only by me", "landing")
        second = self._chat_id()
        self.at.button(key=f"tab-close-{first}").click().run()
        self.assertEqual(self._chat_id(), second)
        self.assertIn(">make a file readable only by me<", self._html())
        self.assertEqual(self._tab_keys(), [f"tab-open-{second}"])

    def _seed(self, n):
        ids = []
        for i in range(n):
            cid = self.store.new_chat().chat_id
            self.store.save_chat(cid, [{"role": "user", "content": f"question {i}"}])
            ids.append(cid)
        return ids

    def test_a_new_session_opens_recent_chats_and_tabs_stay_capped(self):
        ids = self._seed(9)
        self.at.run()
        # The five most recent, oldest first, plus the new-chat tab.
        self.assertEqual(self._tab_keys(), [f"tab-open-{cid}" for cid in ids[4:]] + ["tab-open-draft"])
        self.assertEqual(len(self._rows()), 9)
        self.at.button(key=f"hist-open-{ids[0]}").click().run()
        self.assertFalse(self.at.exception)
        keys = self._tab_keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[-1], f"tab-open-{ids[0]}")
        self.assertTrue(self.at.button(key=f"tab-open-{ids[0]}").disabled)
        self.assertNotIn("tab-open-draft", keys)

        for cid in ids[1:4]:
            self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertEqual(len(self._tab_keys()), 6)
        self.assertEqual(self._chat_id(), ids[3])

    def test_deleting_the_open_chat_closes_its_tab(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        cid = self._chat_id()
        self.at.button(key=f"hist-del-{cid}").click().run()
        self.at.button(key=f"hist-yes-{cid}").click().run()
        self.assertFalse(self.at.exception)
        self.assertIsNone(self._chat_id())
        self.assertEqual(self._tab_keys(), ["tab-open-draft"])
        self.assertEqual(self.store.list_chats(), [])
        self.assertIn('class="hero"', self._html())

    def test_a_tab_for_a_chat_deleted_elsewhere_disappears(self):
        ids = self._seed(2)
        self.at.run()
        self.store.delete_chat(ids[0])
        self.at.run()
        self.assertEqual(self._tab_keys(), [f"tab-open-{ids[1]}", "tab-open-draft"])

    # ---------- settings ----------

    def test_settings_shows_the_run_settings_and_every_model(self):
        self.at.run()
        self.assertNotIn('class="dlg-h"', self._html())
        html = self._settings_html()
        for k, v in (
            ("Model", "qwen3.5:2b"),
            ("Ollama", "running 0.12.0"),
            ("Keep alive", "30m"),
            ("Context", "4,096 tokens"),
            ("IDK threshold", "cos 0.30"),
            ("Top-k", "5"),
            ("Answers", "0"),
        ):
            self.assertIn(f'<span class="k">{k}</span><span class="v">{v}</span>', html)
        self.assertIn("ollama create thesis-student -f Modelfile", html)
        self.assertIn("ollama create thesis-base-q4km -f Modelfile-base", html)
        self.assertIn(polaris.RATE_NOTE.split(".")[0], html)
        # A run without the click closes it again.
        self.at.run()
        self.assertNotIn('class="dlg-h"', self._html())

    # ---------- monitor ----------

    def _monitor(self):
        (mon,) = [m.value for m in self.at.markdown if m.value.startswith('<div class="mon">')]
        return mon

    def test_the_monitor_waits_then_shows_the_last_answer(self):
        self.at.run()
        mon = self._monitor()
        self.assertIn("No answer in this chat yet", mon)
        self.assertNotRegex(mon, r"<b>[\d.]+</b>")
        self._ask("extract a .tar.gz file", "landing")
        mon = self._monitor()
        self.assertIn("18.3", mon)
        self.assertIn('class="spark"', mon)
        self.assertIn("No retrieval in BASE", mon)

    def test_the_monitor_keeps_each_compare_side(self):
        answer = _fake_answer(self.calls)

        def per_side(question, mode, model, **kw):
            result = answer(question, mode, model, **kw)
            if mode == "cot":
                result.update(tok_s=9.75, eval_count=39, eval_duration=4_000_000_000, gen_s=4.0, latency_s=4.6)
            return result

        self._compare_base_vs_cot()
        with patch("app.engine.answer_question", side_effect=per_side):
            self._ask("extract a .tar.gz file", "landing")
        mon = self._monitor()
        self.assertIn("18.3", mon)
        self.assertIn("9.8", mon)
        self.assertIn('class="line r"', mon)

    # Mocked model below: these check the app's plumbing (parse, render, memory), not what a real model writes.

    def _answers(self, texts, hits=None):
        answer = _fake_answer(self.calls, self.seen)

        def scripted(question, mode, model, **kw):
            result = answer(question, mode, model, **kw)
            result["text"] = texts(question, mode)
            if kw.get("retriever") is not None and hits is not None:
                result["hits"] = [dict(h) for h in hits]
            return result

        return patch("app.engine.answer_question", side_effect=scripted)

    def _full_mode(self):
        rag = MagicMock(HybridRetriever=_Retriever)
        self.at.session_state["mode"] = "full"
        return [patch("app.engine.index_ok", return_value=True), patch.dict(sys.modules, {"rag.query_hybrid": rag})]

    def _how_to_shown(self):
        self.assertEqual([c.value for c in self.at.code], list(SUDO_COMMANDS))
        self.assertTrue(all(c.proto.language == "bash" for c in self.at.code))
        shown = [m.value for m in self.at.markdown]
        for line in ("1. Create the user and set a password:", "2. Add the user to the `sudo` group:",
                     "3. Log in as the user and check:"):
            self.assertIn(line, shown)
        (reasoning,) = self.at.expander
        self.assertEqual(reasoning.label, "Show reasoning")
        self.assertFalse(reasoning.proto.expanded)
        self.assertEqual([m.value for m in reasoning.markdown],
                         ["- adduser creates the account.\n- usermod -aG appends the sudo group."])
        html = self._html()
        self.assertIn('<p class="src-line">Source <b>[1]</b> adduser · <b>[2]</b> usermod</p>', html)
        self.assertNotIn("{{", html + "".join(c.value for c in self.at.code))
        self.assertNotIn("Reasoning:", html)

    def test_a_how_to_answer_shows_numbered_steps_and_one_code_box_per_command(self):
        texts = lambda question, mode: SUDO_FULL  # noqa: E731
        patches = self._full_mode()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with self._answers(texts, SUDO_HITS):
            self.at.run()
            self._ask(SUDO_QUESTION, "landing")
        self._how_to_shown()

        self.at.run()
        self._how_to_shown()

        cid = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        self.assertFalse(self.at.exception)
        self._how_to_shown()

    def test_explain_step_2_asks_about_the_usermod_step(self):
        texts = lambda question, mode: SUDO_FULL if question == SUDO_QUESTION else "It appends sudo [2]."  # noqa: E731
        patches = self._full_mode()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with self._answers(texts, SUDO_HITS):
            self.at.run()
            self._ask(SUDO_QUESTION, "landing")
            self._ask("explain step 2")
        first, follow = _FakeStream.made
        self.assertEqual(first.kw["messages_fn"](SUDO_QUESTION, SUDO_HITS, "full"),
                         app_messages(SUDO_QUESTION, SUDO_HITS, "full"))

        context = follow.kw["context"]
        self.assertIn("step 2 means step 2 of your previous answer", context.step_note)
        self.assertIn("sudo usermod -aG sudo <username>", context.step_note)
        self.assertNotIn("adduser", context.step_note)
        self.assertNotIn("appends the sudo group", context.step_note)
        self.assertIn("usermod -aG sudo username", context.retrieval_query)
        system, *history, user = context.apply(follow.kw["messages_fn"]("explain step 2", SUDO_HITS, "full"))
        self.assertTrue(user["content"].endswith(context.step_note))
        self.assertIn("Documentation excerpts:", user["content"])
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertNotIn("Reasoning", history[1]["content"])
        self.assertNotIn("[1]", history[1]["content"])
        self.assertIn("```bash\nsudo usermod -aG sudo <username>\n```", history[1]["content"])

    def test_base_says_it_had_no_documents_and_drops_a_citation_it_invented(self):
        with self._answers(lambda question, mode: "Replace <username> and run `id <username>` [1].\n\nSource: [1]"):
            self.at.run()
            self._ask("check a user's groups", "landing")
        html = self._html()
        self.assertIn('<p class="src-line none">No documents in BASE, so nothing to cite · removed [1] the model '
                      "wrote</p>", html)
        self.assertIn("Replace &lt;username> and run `id <username>`.", [m.value for m in self.at.markdown])
        self.assertEqual(self.at.expander, [])

    def test_compare_folds_reasoning_on_the_side_that_wrote_it(self):
        texts = lambda question, mode: SUDO_FULL if mode == "cot" else "Use `sudo adduser <username>`."  # noqa: E731
        self._compare_base_vs_cot()
        with self._answers(texts):
            self._ask(SUDO_QUESTION, "landing")

        (reasoning,) = self.at.expander
        self.assertEqual(reasoning.label, "Show reasoning")
        self.assertFalse(reasoning.proto.expanded)
        self.assertEqual([c.value for c in self.at.code], list(SUDO_COMMANDS))
        self.assertIn("No documents in +CoT", self._html())

        def compact():
            # A replayed compare is one preview line per side, taken from the answer, never the reasoning.
            html = self._html()
            self.assertIn('<p class="preview">Create the account, then add it to the sudo group.', html)
            self.assertNotIn("adduser creates the account", html)
            self.assertEqual(self.at.expander, [])

        self.at.run()
        compact()
        cid = self._chat_id()
        self.at.button(key="new-chat").click().run()
        self.at.button(key=f"hist-open-{cid}").click().run()
        compact()

    def _sudo_compare(self):
        texts = lambda question, mode: SUDO_FULL if mode == "cot" else BASE_STEPS  # noqa: E731
        self._compare_base_vs_cot()
        with self._answers(texts):
            self._ask(SUDO_QUESTION, "landing")
        return texts

    def test_a_compare_follow_up_gives_each_side_its_own_step(self):
        texts = self._sudo_compare()
        with self._answers(texts):
            self._ask("explain step 2")
        left, right = _FakeStream.made[-2:]

        def note(stream, mode):
            return stream.kw["messages_fn"]("explain step 2", [], mode)[1]["content"]

        self.assertIn("sudo passwd <username>", note(left, "base"))
        self.assertNotIn("usermod", note(left, "base"))
        self.assertIn("sudo usermod -aG sudo <username>", note(right, "cot"))
        self.assertNotIn("passwd", note(right, "cot"))

    def test_one_answer_after_a_compare_gets_both_steps_labelled(self):
        texts = self._sudo_compare()
        self.at.toggle(key="compare").set_value(False).run()
        with self._answers(texts):
            self._ask("explain step 2")
        (stream,) = _FakeStream.made[2:]
        user = stream.kw["messages_fn"]("explain step 2", [], "base")[1]["content"]
        self.assertIn("Left answer (qwen3.5:2b · BASE): step 2 was shown as:\n2. Set a password:", user)
        self.assertIn("Right answer (qwen3.5:2b · +CoT): step 2 was shown as:\n2. Add the user", user)
        self.assertIn("Answer for each separately, starting with its label.", user)

    def test_inline_commands_get_boxes_and_trailing_ones_are_not_given_a_step(self):
        text = ("1. Create the user with `sudo adduser {{username}}`.\n2. Give it sudo.\n3. Check it.\n\n"
                "```bash\nsudo usermod -aG sudo {{username}}\n```\n\n```bash\ngroups {{username}}\n```")
        with self._answers(lambda question, mode: text):
            self.at.run()
            self._ask(SUDO_QUESTION, "landing")
            self._ask("explain step 2")
        self.assertEqual([c.value for c in self.at.code][:3], list(SUDO_COMMANDS))
        shown = [m.value for m in self.at.markdown]
        self.assertIn("1. Create the user with `sudo adduser <username>`.", shown)
        self.assertIn('<p class="cmd-loose">The model listed these 2 commands after its steps without saying which '
                      "step each belongs to.</p>", shown)
        note = _FakeStream.made[-1].kw["context"].step_note
        self.assertIn("step 2 was shown as:\n2. Give it sudo.\nAfter its steps it also listed these commands", note)
        self.assertIn("```bash\nsudo usermod -aG sudo <username>\n```", note)
        self.assertNotIn("adduser", note)


DRAFT = ("Reasoning:\n1. The excerpts show `chmod 600` gives the owner read and write.\n"
         "2. `ls -l` shows the result.\n3. A third draft line, longer than the answer.\n\nAnswer:\n"
         "To make a file readable only by you:\n1. Remove group and other access:\n"
         "```bash\nchmod 600 <path/to/file>\n```\n2. Check it:\n```bash\nls -l <path/to/file>\n```\nSource: [1]")
SUDO_QUESTION = "How do I add a new user on Ubuntu and give them sudo access?"
SUDO_COMMANDS = ("sudo adduser <username>", "sudo usermod -aG sudo <username>", "groups <username>")
SUDO_HITS = [dict(HIT, title="adduser", path="tldr/adduser.md"), dict(HIT, title="usermod", path="tldr/usermod.md")]
BASE_STEPS = (
    "1. Create the user:\n```bash\nsudo adduser {{username}}\n```\n"
    "2. Set a password:\n```bash\nsudo passwd {{username}}\n```"
)
SUDO_FULL = (
    "Reasoning:\n- adduser creates the account.\n- usermod -aG appends the sudo group.\n\nAnswer:\n"
    "Create the account, then add it to the sudo group.\n\n"
    "1. Create the user and set a password:\n   ```bash\n   sudo adduser {{username}}\n   ```\n"
    "1. Add the user to the `sudo` group:\n   ```bash\n   sudo usermod -aG sudo {{username}}\n   ```\n"
    "1. Log in as the user and check:\n   ```bash\n   groups {{username}}\n   ```\n\nSource: [1], [2]"
)


if __name__ == "__main__":
    unittest.main()
