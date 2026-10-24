#!/usr/bin/env python3
"""The Polaris run loop under Streamlit's AppTest. No live Ollama: answer_question is replaced."""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.polaris as polaris
from app.chat_memory import ChatStore
from app.engine import OllamaInfo

# Imported while discovery loads modules, before other tests swap a MagicMock into sys.modules["streamlit"].
try:
    import streamlit
    from streamlit.logger import set_log_level
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
        patches = [
            patch.dict(os.environ, {"XDG_DATA_HOME": data.name}),
            patch("app.engine.answer_question", side_effect=_fake_answer(self.calls, self.seen)),
            patch("app.engine.ollama_reachable", return_value=True),
            patch("app.engine.ollama_info", return_value=OllamaInfo(version="0.12.0", model_digest="sha256:abc")),
            patch("app.engine.warm_model", return_value=True),
            patch("app.polaris.flags_html", self.flags),
            patch.object(self.streamlit, "rerun", side_effect=AssertionError("st.rerun() called")),
            # AppTest 1.45 reads a single-select st.pills with nothing picked as a multiselect and fails.
            patch.object(self.streamlit, "pills", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.rerun = self.streamlit.rerun
        self.at = self.AppTest.from_file(APP, default_timeout=20)
        self.at.session_state["mode"] = "base"

    def _html(self):
        return "\n".join(m.value for m in self.at.markdown if not m.value.startswith("<style>"))

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
        self.assertIn('<span class="k">Answers</span><span class="v">1</span>', html)
        self.assertIn('<span class="k">Warm-up</span><span class="v">ready</span>', html)
        self.assertIn("English questions only", html)

    def test_second_question_refreshes_the_sidebar_without_duplicate_keys(self):
        self.at.run()
        self.at.chat_input(key="chat-landing").set_value("extract a .tar.gz file").run()
        self.at.chat_input(key="chat-dock").set_value("make a file readable only by me").run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(len(self.calls), 2)
        html = self._html()
        self.assertIn('<span class="k">Answers</span><span class="v">2</span>', html)
        self.assertEqual(html.count('class="q"'), 2)
        self.assertEqual(self._flags(), '<div class="polaris-flags polaris-sky-2"></div>')

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
        html = self._html()
        self.assertIn('class="hero"', html)
        self.assertIn('<span class="k">Answers</span><span class="v">0</span>', html)
        self.assertEqual(self.at.session_state["messages"], [])

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
        self.assertIn('<span class="k">Answers</span><span class="v">1</span>', self._html())
        self.assertTrue(self.at.button(key=f"hist-open-{first}").disabled)
        self.assertFalse(self.at.button(key=f"hist-open-{second}").disabled)

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
        self.assertIn("Stopped before an answer came back.", self._html())

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

    # ---------- compare ----------

    def test_compare_starts_off_with_one_picker_per_side(self):
        self.at.run()
        self.assertFalse(self.at.toggle(key="compare").value)
        self.assertEqual(self.at.toggle(key="compare").label, "Compare two configs")
        self.assertNotIn("cmp-left-config", [s.key for s in self.at.selectbox])

        self.at.toggle(key="compare").set_value(True).run()
        left, right = self.at.selectbox(key="cmp-left-config"), self.at.selectbox(key="cmp-right-config")
        self.assertEqual(left.options, ["BASE", "+CoT", "+RAG", "+RAG+CoT"])
        self.assertEqual((left.value, right.value), ("base", "base+RAG+CoT"))
        self.assertNotIn("BASE", right.options)
        self.assertIn('<span class="m">qwen3.5:2b</span> <b>BASE</b><span class="vs">vs</span><b>+RAG+CoT</b>', self._html())

        right.set_value("base+CoT").run()
        self.assertEqual(self.at.session_state["cmp_right"], "cot")
        self._ask("extract a .tar.gz file", "landing")
        self.assertEqual([c["mode"] for c in self.calls], ["base", "cot"])
        self.assertNotIn('class="model"', self._html())
        # Still the same picks on the next run, so the browser never falls back to a default.
        self.at.run()
        self.assertEqual(self.at.selectbox(key="cmp-right-config").value, "base+CoT")

    def test_compare_across_models_names_each_side(self):
        rag = MagicMock(HybridRetriever=_Retriever)
        with patch("app.engine.DISTILLED_MODEL", "student:latest"), patch("app.engine.index_ok", return_value=True), \
                patch.dict(sys.modules, {"rag.query_hybrid": rag}):
            self.at.run()
            self.at.toggle(key="compare").set_value(True).run()
            left, right = self.at.selectbox(key="cmp-left-config"), self.at.selectbox(key="cmp-right-config")
            self.assertEqual(left.options[-1], "distilled+RAG+CoT")
            self.assertEqual((left.value, right.value), ("base+RAG+CoT", "distilled+RAG+CoT"))
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


if __name__ == "__main__":
    unittest.main()
