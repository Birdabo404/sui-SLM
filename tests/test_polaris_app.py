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


def _fake_answer(calls, extra=None):
    def answer(question, mode, model, retriever=None, chat_fn=None, **kw):
        calls.append({"question": question, "mode": mode, "retriever": retriever})
        if extra is not None:
            extra.append({"model": model, "model_key": kw.get("model_key"), "context": kw.get("context")})
        return {
            "text": "Use `tar -xzf file.tar.gz`.",
            "hits": [],
            "weak": False,
            "latency_s": 1.5,
            "config": "base",
            "mode": mode,
            "model": model,
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
        self.assertEqual(self.at.session_state["chat_id"], chat.chat_id)
        # Listed in the run that asked it, as the open chat; its row is disabled so a click can't stop an answer.
        (row,) = self._rows()
        self.assertEqual(row.key, f"hist-open-{chat.chat_id}")
        self.assertTrue(row.disabled)
        self.assertNotIn("Chats you start are saved", self._html())

    def test_new_chat_keeps_history_and_switching_restores_a_chat(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self.at.session_state["chat_id"]
        self.at.button(key="new-chat").click().run()
        self.assertIsNone(self.at.session_state["chat_id"])
        self._ask("make a file readable only by me", "landing")
        self._ask("and for a folder?")
        second = self.at.session_state["chat_id"]
        self.assertNotEqual(first, second)
        self.assertEqual([c.chat_id for c in self.store.list_chats()], [second, first])
        self.assertEqual(self.store.list_chats()[0].turn_count, 2)

        self.at.button(key=f"hist-open-{first}").click().run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(self.at.session_state["chat_id"], first)
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
        cid = self.at.session_state["chat_id"]
        self.at.button(key=f"hist-del-{cid}").click().run()
        self.assertIn(polaris.history_note_html(polaris.DELETE_ASK), self._html())
        self.assertEqual(len(self.store.list_chats()), 1)
        self.at.button(key=f"hist-no-{cid}").click().run()
        self.assertNotIn("Delete this chat?", self._html())
        self.assertEqual(len(self.store.list_chats()), 1)
        self.assertEqual(self.at.session_state["chat_id"], cid)

        self.at.button(key=f"hist-del-{cid}").click().run()
        self.at.button(key=f"hist-yes-{cid}").click().run()
        self.assertFalse(self.at.exception)
        self.rerun.assert_not_called()
        self.assertEqual(self.store.list_chats(), [])
        self.assertIsNone(self.at.session_state["chat_id"])
        self.assertIn('class="hero"', self._html())
        self.assertIn("Chats you start are saved on this machine.", self._html())

    def test_deleting_another_chat_leaves_the_open_one(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        first = self.at.session_state["chat_id"]
        self.at.button(key="new-chat").click().run()
        self._ask("make a file readable only by me", "landing")
        second = self.at.session_state["chat_id"]
        self.at.button(key=f"hist-del-{first}").click().run()
        self.at.button(key=f"hist-yes-{first}").click().run()
        self.assertEqual([c.chat_id for c in self.store.list_chats()], [second])
        self.assertEqual(self.at.session_state["chat_id"], second)
        self.assertIn(">make a file readable only by me<", self._html())

    def test_a_chat_deleted_elsewhere_continues_in_a_new_one(self):
        self.at.run()
        self._ask("extract a .tar.gz file", "landing")
        gone = self.at.session_state["chat_id"]
        self.store.delete_chat(gone)
        self._ask("and a .zip?")
        (chat,) = self.store.list_chats()
        self.assertNotEqual(chat.chat_id, gone)
        self.assertEqual(self.at.session_state["chat_id"], chat.chat_id)
        self.assertEqual(chat.turn_count, 2)

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
        self.assertIsNone(self.at.session_state["chat_id"])
        self.assertEqual(self._rows(), [])


if __name__ == "__main__":
    unittest.main()
