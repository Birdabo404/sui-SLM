#!/usr/bin/env python3
"""Which card each failure gets, both with app/backend.py (PR #20) present, stood in for by tests/backend_fake.py
with the same classes, codes and messages, and without it, where plain requests and socket errors are classified."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import issues
from app.issues import (
    Issue,
    abstain_note,
    answer_status,
    checks_html,
    checks_summary,
    hf_cached,
    index_missing,
    issue_for_error,
    issue_from_dict,
    issue_html,
    load_index,
    model_missing,
    offline_checks,
    ollama_down,
)
from tests import backend_fake
from tests.backend_fake import PlainStreamError

fake = backend_fake.module()
HOST = "http://127.0.0.1:11434"


def kind(error, **kw):
    return issue_for_error(error, host=HOST, **kw).kind


def wrapped(error, cause):
    """`raise error from cause`, the way app.streaming wraps socket errors."""
    try:
        raise error from cause
    except type(error) as e:
        return e


class WithoutBackend(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(issues, "backend", None)
        patcher.start()
        self.addCleanup(patcher.stop)


class WithBackend(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(issues, "backend", fake)
        patcher.start()
        self.addCleanup(patcher.stop)


class ImportTests(unittest.TestCase):
    """`from app import backend` at import time, with app/backend.py importable or not, whichever this branch has."""

    def _reload_with(self, module):
        package = sys.modules["app"]
        had, before = hasattr(package, "backend"), getattr(package, "backend", None)
        if had:
            delattr(package, "backend")
        if module is not None:
            package.backend = module

        def restore():
            if hasattr(package, "backend"):
                delattr(package, "backend")
            if had:
                package.backend = before
            importlib.reload(issues)

        self.addCleanup(restore)
        self.enterContext(patch.dict(sys.modules, {"app.backend": module}))
        return importlib.reload(issues)

    def test_without_app_backend_the_module_still_imports_and_uses_its_own_classification(self):
        reloaded = self._reload_with(None)
        self.assertIsNone(reloaded.backend)
        self.assertEqual(reloaded.issue_for_error(requests.ConnectionError("refused"), host=HOST).kind,
                         "ollama_unavailable")

    def test_with_app_backend_importable_the_module_picks_it_up(self):
        self.assertIs(self._reload_with(fake).backend, fake)


class ClassifyWithoutBackendTests(WithoutBackend):
    """The plain requests and socket errors app.streaming raises before PR #20."""

    def test_refused_before_any_text_is_ollama_unavailable(self):
        issue = issue_for_error(requests.ConnectionError("[Errno 111] Connection refused"), host=HOST)
        self.assertEqual((issue.kind, issue.fix), ("ollama_unavailable", "ollama serve"))
        self.assertIn(HOST, issue.detail)
        self.assertNotIn("Errno", issue.text())

    def test_a_socket_timeout_wrapped_as_a_connection_error_is_a_timeout(self):
        self.assertEqual(kind(wrapped(requests.ConnectionError("timed out"), TimeoutError("timed out"))),
                         "ollama_timeout")
        self.assertEqual(kind(wrapped(requests.ConnectionError("x"), TimeoutError())), "ollama_timeout")
        self.assertEqual(kind(wrapped(requests.ConnectionError("x"), requests.ReadTimeout())), "ollama_timeout")

    def test_a_connection_lost_after_text_arrived_is_a_disconnect(self):
        issue = issue_for_error(requests.ConnectionError("Connection reset by peer"), host=HOST, partial_chars=1234)
        self.assertEqual(issue.kind, "ollama_disconnected")
        self.assertIn("after 1,234 characters. The partial answer was not kept.", issue.detail)

    def test_a_stream_without_its_terminal_record_is_a_disconnect(self):
        self.assertEqual(kind(PlainStreamError("Ollama stream ended without a terminal event")),
                         "ollama_disconnected")

    def test_a_404_or_an_in_stream_not_found_is_the_missing_model_with_its_command(self):
        for error in (requests.HTTPError('Ollama returned HTTP 404: {"error":"model \\"x\\" not found"}'),
                      PlainStreamError("model 'thesis-student' not found")):
            issue = issue_for_error(error, host=HOST, model="thesis-student",
                                    create="ollama create thesis-student -f Modelfile")
            self.assertEqual((issue.kind, issue.fix),
                             ("ollama_model_missing", "ollama create thesis-student -f Modelfile"))
            self.assertEqual(issue.title, "thesis-student isn't installed")
            self.assertIn("kd/README.md", issue.hint)

    def test_a_pulled_model_says_it_downloads_once(self):
        issue = issue_for_error(requests.HTTPError("Ollama returned HTTP 404: model not found"), host=HOST,
                                model="qwen3.5:2b", create="ollama pull qwen3.5:2b")
        self.assertEqual(issue.fix, "ollama pull qwen3.5:2b")
        self.assertIn("works offline", issue.hint)

    def test_another_http_error_shows_its_first_line_only(self):
        issue = issue_for_error(requests.HTTPError("Ollama returned HTTP 500: out of memory\nstack..."), host=HOST)
        self.assertEqual((issue.kind, issue.detail),
                         ("ollama_response_error", "Ollama returned HTTP 500: out of memory"))

    def test_the_rag_index_not_loaded_runtime_error_is_an_unreadable_index(self):
        self.assertEqual(kind(RuntimeError("RAG index is not loaded"), index_dir="/x"), "index_corrupt")

    def test_a_load_failure_is_an_unreadable_index_or_the_missing_embedding_model(self):
        self.assertEqual(kind(ValueError("bad sqlite"), stage="load", index_dir="/x"), "index_corrupt")
        self.assertEqual(kind(OSError("We couldn't connect to huggingface.co"), stage="load"),
                         "embedding_model_missing")

    def test_an_error_while_searching_is_a_failed_query(self):
        self.assertEqual(kind(ValueError("bad"), stage="retrieve"), "index_query_failed")
        # A transport error is still about Ollama, whatever stage it came from.
        self.assertEqual(kind(requests.ConnectionError("refused"), stage="retrieve"), "ollama_unavailable")

    def test_anything_else_names_its_type_and_hides_its_message(self):
        issue = issue_for_error(KeyError("secret internals"), host=HOST)
        self.assertEqual(issue.kind, "unexpected")
        self.assertIn("KeyError", issue.detail)
        self.assertNotIn("secret internals", issue.text())

    def test_the_index_loads_through_the_factory_directly(self):
        self.assertEqual(load_index("/idx", lambda path: ("loaded", path)), ("loaded", "/idx"))


class ClassifyWithBackendTests(WithBackend):
    """app.backend's typed errors: a title for the code, its user_message exactly, never its detail."""

    CODES = (
        (fake.OllamaUnavailable, "ollama_unavailable"),
        (fake.OllamaModelMissing, "ollama_model_missing"),
        (fake.EmbeddingModelError, "embedding_model_missing"),
        (fake.IndexMissingError, "index_missing"),
        (fake.IndexLoadError, "index_corrupt"),
        (fake.RetrievalQueryError, "index_query_failed"),
        (fake.OllamaTimedOut, "ollama_timeout"),
        (fake.OllamaDisconnected, "ollama_disconnected"),
        (fake.OllamaResponseError, "ollama_response_error"),
    )

    def test_every_code_gets_its_own_title_and_the_user_message_as_written(self):
        titles = set()
        for cls, code in self.CODES:
            with self.subTest(code):
                error = cls("internal detail: /home/x/secret.sqlite3")
                issue = issue_for_error(error, host=HOST, model="qwen3.5:2b", create="ollama pull qwen3.5:2b")
                self.assertEqual(issue.kind, code)
                self.assertEqual(issue.detail, error.user_message)
                self.assertNotIn("secret", issue.text() + issue_html(issue))
                self.assertNotIn("Traceback", issue_html(issue))
                titles.add(issue.title)
        self.assertEqual(len(titles), len(self.CODES))

    def test_the_stage_does_not_override_a_typed_error(self):
        self.assertEqual(kind(fake.OllamaUnavailable("x"), stage="load"), "ollama_unavailable")
        self.assertEqual(kind(fake.IndexLoadError("x"), stage="answer"), "index_corrupt")

    def test_the_missing_model_adds_the_exact_command_for_the_selected_model(self):
        for create, hint in (("ollama create thesis-student -f Modelfile", "kd/README.md"),
                             ("ollama pull qwen3.5:2b", "works offline")):
            issue = issue_for_error(fake.OllamaModelMissing("404"), host=HOST, model=create.split()[2], create=create)
            self.assertEqual(issue.fix, create)
            self.assertIn(hint, issue.hint)
            self.assertEqual(issue.title, f"{create.split()[2]} isn't installed")

    def test_index_cards_add_the_build_command_and_others_add_nothing_that_contradicts(self):
        self.assertIn("rag/build_index.py", issue_for_error(fake.IndexMissingError(), host=HOST).fix)
        self.assertIn("rag/build_index.py", issue_for_error(fake.IndexLoadError(), host=HOST).fix)
        self.assertIn("rag/build_index.py", issue_for_error(fake.RetrievalQueryError(), host=HOST).fix)
        for cls in (fake.OllamaTimedOut, fake.OllamaResponseError, fake.EmbeddingModelError):
            issue = issue_for_error(cls(), host=HOST)
            self.assertEqual((issue.fix, issue.hint), ("", ""), cls.__name__)

    def test_a_disconnect_says_how_much_was_discarded(self):
        issue = issue_for_error(fake.OllamaDisconnected("eof"), host=HOST, partial_chars=4321)
        self.assertIn("Discard the partial answer", issue.detail)
        self.assertEqual(issue.hint, "The partial answer (4,321 characters) was discarded.")

    def test_untyped_errors_still_fall_back_to_the_local_classification(self):
        self.assertEqual(kind(KeyError("x")), "unexpected")
        self.assertEqual(kind(requests.ConnectionError("refused")), "ollama_unavailable")

    def test_pre_checks_use_the_backend_messages(self):
        self.assertEqual(ollama_down(HOST).detail, fake.OllamaUnavailable().user_message)
        self.assertEqual(index_missing("/x").detail, fake.IndexMissingError().user_message)
        issue = model_missing("thesis-student", "ollama create thesis-student -f Modelfile")
        self.assertEqual((issue.detail, issue.fix),
                         (fake.OllamaModelMissing().user_message, "ollama create thesis-student -f Modelfile"))

    def test_the_index_loads_through_the_backend_loader_so_failures_arrive_typed(self):
        def broken(path):
            raise ValueError("no such table: embeddings")

        with tempfile.TemporaryDirectory() as idx:
            loaded = load_index(idx, lambda path: ("ok", path))
            self.assertIsInstance(loaded, fake.SafeRetriever)
            with self.assertRaises(fake.IndexLoadError):
                load_index(idx, broken)
        with self.assertRaises(fake.IndexMissingError):
            load_index("/nonexistent/index", broken)

    def test_backticked_commands_in_the_message_render_as_code(self):
        html = issue_html(issue_for_error(fake.OllamaUnavailable(), host=HOST))
        self.assertIn("Start it with <code>ollama serve</code>, then try again.", html)


class AbstainStatusTests(unittest.TestCase):
    def test_the_backend_status_wins_over_the_local_reading(self):
        class Chart:
            status = fake.AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL

        self.assertEqual(answer_status(Chart(), [{"x": 1}], weak=False), "abstain_empty_retrieval")
        self.assertEqual(answer_status({"status": "complete"}, [], weak=True), "complete")

    def test_without_a_status_it_comes_from_the_hits_and_the_threshold_decision(self):
        self.assertEqual(answer_status({}, [{"x": 1}], weak=False), "complete")
        self.assertEqual(answer_status({}, [{"x": 1}], weak=True), "abstain_weak_retrieval")
        self.assertEqual(answer_status(object(), [], weak=True), "abstain_empty_retrieval")

    def test_the_note_is_the_status_message_with_the_backend_and_the_apps_words_without(self):
        weak = {"status": "abstain_weak_retrieval", "threshold": 0.3, "hits": [{}]}
        empty = {"status": "abstain_empty_retrieval", "hits": []}
        with patch.object(issues, "backend", fake):
            self.assertEqual(abstain_note(weak), fake.AnswerStatus("abstain_weak_retrieval").user_message)
            self.assertEqual(abstain_note(empty), fake.AnswerStatus("abstain_empty_retrieval").user_message)
            self.assertEqual(abstain_note({"status": "complete"}), "")
        with patch.object(issues, "backend", None):
            self.assertEqual(abstain_note(weak), "No retrieved chunk reached cosine 0.30, so the model wasn't asked.")
            self.assertIn("Nothing was retrieved", abstain_note(empty))
            self.assertIn("No retrieved chunk", abstain_note({"hits": [{}]}))


class CardTests(WithoutBackend):
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
