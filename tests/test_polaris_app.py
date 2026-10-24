#!/usr/bin/env python3
"""The Polaris run loop under Streamlit's AppTest. No live Ollama: answer_question is replaced."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.polaris as polaris
from app.engine import OllamaInfo

# Imported while discovery loads modules, before other tests swap a MagicMock into sys.modules["streamlit"].
try:
    import streamlit
    from streamlit.logger import set_log_level
    from streamlit.testing.v1 import AppTest
except ImportError:
    streamlit = None

APP = str(ROOT / "app" / "app.py")


def _fake_answer(calls):
    def answer(question, mode, model, retriever=None, chat_fn=None, **kw):
        calls.append({"question": question, "mode": mode, "retriever": retriever})
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
        self.flags = MagicMock(wraps=polaris.flags_html)
        patches = [
            patch("app.engine.answer_question", side_effect=_fake_answer(self.calls)),
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


if __name__ == "__main__":
    unittest.main()
