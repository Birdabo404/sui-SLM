#!/usr/bin/env python3
"""Which card each failure gets. The backend's own error types are stood in for by local classes with the same names
and bases, so the cards are right before and after those types exist."""
import sys
import tempfile
import unittest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.issues import (
    Issue,
    checks_html,
    checks_summary,
    hf_cached,
    index_missing,
    issue_for_error,
    issue_from_dict,
    issue_html,
    offline_checks,
)
from app.streaming import OllamaStreamError

HOST = "http://127.0.0.1:11434"


class OllamaUnavailable(requests.ConnectionError):
    pass


class OllamaTimedOut(requests.Timeout):
    pass


class OllamaDisconnected(requests.ConnectionError):
    pass


class OllamaModelMissing(requests.RequestException):
    pass


class AppRetrievalError(RuntimeError):
    pass


class IndexLoadError(AppRetrievalError):
    pass


class EmbeddingModelError(AppRetrievalError):
    pass


class RetrievalQueryError(AppRetrievalError):
    pass


def kind(error, **kw):
    return issue_for_error(error, host=HOST, **kw).kind


def wrapped(error, cause):
    """`raise error from cause`, the way app.streaming wraps socket errors."""
    try:
        raise error from cause
    except type(error) as e:
        return e


class ClassifyTodayTests(unittest.TestCase):
    """The plain requests and socket errors app.streaming raises now."""

    def test_refused_before_any_text_is_ollama_down(self):
        issue = issue_for_error(requests.ConnectionError("[Errno 111] Connection refused"), host=HOST)
        self.assertEqual((issue.kind, issue.fix), ("ollama_down", "ollama serve"))
        self.assertIn(HOST, issue.detail)
        self.assertNotIn("Errno", issue.text())

    def test_a_socket_timeout_wrapped_as_a_connection_error_is_a_timeout(self):
        self.assertEqual(kind(wrapped(requests.ConnectionError("timed out"), TimeoutError("timed out"))), "timeout")
        self.assertEqual(kind(wrapped(requests.ConnectionError("x"), TimeoutError())), "timeout")
        self.assertEqual(kind(wrapped(requests.ConnectionError("x"), requests.ReadTimeout())), "timeout")

    def test_a_connection_lost_after_text_arrived_is_a_drop(self):
        issue = issue_for_error(requests.ConnectionError("Connection reset by peer"), host=HOST, partial_chars=1234)
        self.assertEqual(issue.kind, "dropped")
        self.assertIn("after 1,234 characters. The partial answer was not kept.", issue.detail)

    def test_a_stream_without_its_terminal_record_is_a_drop(self):
        self.assertEqual(kind(OllamaStreamError("Ollama stream ended without a terminal event")), "dropped")

    def test_a_404_or_an_in_stream_not_found_is_the_missing_model_with_its_command(self):
        for error in (requests.HTTPError('Ollama returned HTTP 404: {"error":"model \\"x\\" not found, try pulling it first"}'),
                      OllamaStreamError("model 'thesis-student' not found")):
            issue = issue_for_error(error, host=HOST, model="thesis-student",
                                    create="ollama create thesis-student -f Modelfile")
            self.assertEqual((issue.kind, issue.fix), ("model_missing", "ollama create thesis-student -f Modelfile"))
            self.assertEqual(issue.title, "thesis-student isn't installed")
            self.assertIn("kd/README.md", issue.hint)

    def test_a_pulled_model_says_it_downloads_once(self):
        issue = issue_for_error(requests.HTTPError("Ollama returned HTTP 404: model not found"), host=HOST,
                                model="qwen3.5:2b", create="ollama pull qwen3.5:2b")
        self.assertEqual(issue.fix, "ollama pull qwen3.5:2b")
        self.assertIn("works offline", issue.hint)

    def test_another_http_error_shows_its_first_line_only(self):
        issue = issue_for_error(requests.HTTPError("Ollama returned HTTP 500: out of memory\nstack..."), host=HOST)
        self.assertEqual((issue.kind, issue.detail), ("ollama_error", "Ollama returned HTTP 500: out of memory"))

    def test_the_rag_index_not_loaded_runtime_error_is_the_broken_index(self):
        self.assertEqual(kind(RuntimeError("RAG index is not loaded"), index_dir="/x"), "index_broken")

    def test_an_error_while_searching_is_a_retrieval_failure(self):
        self.assertEqual(kind(ValueError("bad"), retrieving=True), "retrieval_failed")
        # A transport error is still about Ollama, whatever stage it came from.
        self.assertEqual(kind(requests.ConnectionError("refused"), retrieving=True), "ollama_down")

    def test_anything_else_names_its_type_and_hides_its_message(self):
        issue = issue_for_error(KeyError("secret internals"), host=HOST)
        self.assertEqual(issue.kind, "unexpected")
        self.assertIn("KeyError", issue.detail)
        self.assertNotIn("secret internals", issue.text())


class ClassifyBackendTypesTests(unittest.TestCase):
    """The same cards for the backend's named error types."""

    def test_each_named_type_gets_its_own_card(self):
        for error, expected in (
            (OllamaUnavailable("x"), "ollama_down"),
            (OllamaTimedOut("x"), "timeout"),
            (OllamaDisconnected("x"), "dropped"),
            (OllamaModelMissing("x"), "model_missing"),
            (IndexLoadError("x"), "index_broken"),
            (EmbeddingModelError("x"), "embedding_missing"),
            (RetrievalQueryError("x"), "retrieval_failed"),
        ):
            with self.subTest(type(error).__name__):
                self.assertEqual(kind(error), expected)

    def test_a_disconnect_is_a_drop_even_before_text_and_unavailable_never_is(self):
        self.assertEqual(kind(OllamaDisconnected("x"), partial_chars=0), "dropped")
        self.assertEqual(kind(OllamaUnavailable("x"), partial_chars=0), "ollama_down")

    def test_a_subclass_of_a_named_type_is_named_too(self):
        class SlowFirstToken(OllamaTimedOut):
            pass

        self.assertEqual(kind(SlowFirstToken("x")), "timeout")


class CardTests(unittest.TestCase):
    def test_the_card_escapes_and_shows_the_command_as_code(self):
        html = issue_html(Issue("x", "A <b> title", "detail & more", fix="ollama pull a<b>", hint="then <i>"))
        self.assertIn('role="alert"', html)
        self.assertIn("A &lt;b&gt; title", html)
        self.assertIn("<code>ollama pull a&lt;b&gt;</code>", html)
        self.assertNotIn("<i>", html)

    def test_compact_and_without_a_command(self):
        html = issue_html(Issue("x", "T", "D"), compact=True)
        self.assertIn('class="issue compact"', html)
        self.assertNotIn("<code>", html)

    def test_a_stored_issue_round_trips_and_an_old_error_string_keeps_its_first_paragraph(self):
        issue = index_missing("/tmp/idx")
        self.assertEqual(issue_from_dict(issue.as_dict()), issue)
        old = issue_from_dict(None, "Ollama stopped this answer with an error.\n\nHTTPError: Ollama returned HTTP 500")
        self.assertEqual((old.title, old.detail), ("That answer failed", "Ollama stopped this answer with an error."))
        self.assertEqual(issue_from_dict({}, "").detail, "")

    def test_text_reads_as_one_message(self):
        self.assertEqual(Issue("k", "T", "D.", fix="c", hint="H.").text(), "T. D. Run `c`. H.")


class OfflineCheckTests(unittest.TestCase):
    def _checks(self, **kw):
        args = {"host": HOST, "ollama_up": True, "model": "qwen3.5:2b", "installed": True,
                "create": "ollama pull qwen3.5:2b", "index_dir": "/idx", "index_present": True, "chunks": 1200,
                "embed_cached": True, "online": False}
        args.update(kw)
        return offline_checks(**args)

    def test_a_ready_machine_needs_no_network(self):
        checks = self._checks()
        self.assertEqual(checks_summary(checks), "Ready to answer offline")
        self.assertEqual([c.label for c in checks], ["Ollama", "Model", "Lab docs index", "Embedding model", "Network"])
        self.assertIsNone(checks[-1].ok)
        self.assertEqual(checks[-1].value, "offline, not needed")
        self.assertNotIn("ck-fix", checks_html(checks))

    def test_each_failing_check_names_its_fix(self):
        checks = self._checks(installed=False, index_present=False, embed_cached=False)
        self.assertEqual(checks_summary(checks), "Needs attention: model, lab docs index, embedding model")
        self.assertEqual([c.fix for c in checks if c.ok is False],
                         ["ollama pull qwen3.5:2b", checks[2].fix, "bash install.sh"])
        self.assertIn("rag/build_index.py", checks[2].fix)

    def test_with_ollama_down_the_model_cannot_be_checked(self):
        checks = self._checks(ollama_up=False, installed=None)
        self.assertEqual((checks[0].ok, checks[0].fix), (False, "ollama serve"))
        self.assertIsNone(checks[1].ok)

    def test_the_embedding_model_is_found_in_the_hugging_face_cache_without_loading_it(self):
        with tempfile.TemporaryDirectory() as hub:
            env = {"HF_HUB_CACHE": hub}
            self.assertFalse(hf_cached(environ=env))
            snap = Path(hub) / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            self.assertFalse(hf_cached(environ=env))
            (snap / "config.json").write_text("{}")
            self.assertTrue(hf_cached(environ=env))
            self.assertTrue(hf_cached(environ={"SENTENCE_TRANSFORMERS_HOME": hub, "HF_HOME": "/nonexistent"}))


if __name__ == "__main__":
    unittest.main()
