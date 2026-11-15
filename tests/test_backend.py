#!/usr/bin/env python3
"""App backend failure boundaries. No live Ollama, Chroma, or model cache."""
import unittest

from app.backend import (
    AnswerStatus,
    EmbeddingModelError,
    IndexMissingError,
    IndexLoadError,
    OllamaDisconnected,
    OllamaModelMissing,
    OllamaResponseError,
    OllamaTimedOut,
    OllamaUnavailable,
    RetrievalQueryError,
    SafeRetriever,
    UserFacingBackendError,
    answer_status,
    load_app_retriever,
)


class _BrokenRetriever:
    def retrieve(self, question, k=5, tldr_lane=True):
        raise RuntimeError("sqlite internals that must not reach the UI")


class BackendBoundaryTests(unittest.TestCase):
    def test_error_contract_has_stable_codes_and_user_messages(self):
        cases = (
            (
                OllamaUnavailable,
                "ollama_unavailable",
                "Ollama is not running or could not be reached. "
                "Start it with `ollama serve`, then try again.",
            ),
            (
                OllamaModelMissing,
                "ollama_model_missing",
                "The selected Ollama model is not installed. "
                "Install or create the model in Ollama, then try again.",
            ),
            (
                EmbeddingModelError,
                "embedding_model_missing",
                "The embedding model is not available in the local cache. "
                "Run `bash install.sh` once while online, then try again.",
            ),
            (
                IndexMissingError,
                "index_missing",
                "The local search index is missing. Build the index, then try again.",
            ),
            (
                IndexLoadError,
                "index_corrupt",
                "The local search index could not be opened because it is incomplete, "
                "corrupt, or incompatible. Rebuild the index, then try again.",
            ),
            (
                RetrievalQueryError,
                "index_query_failed",
                "The local search index failed while searching. "
                "Rebuild the index, then try again.",
            ),
            (
                OllamaTimedOut,
                "ollama_timeout",
                "Ollama took too long to answer, so the request was stopped. "
                "Try again, or restart Ollama if it keeps happening.",
            ),
            (
                OllamaDisconnected,
                "ollama_disconnected",
                "Ollama disconnected before the answer finished. "
                "Discard the partial answer, then try again.",
            ),
            (
                OllamaResponseError,
                "ollama_response_error",
                "Ollama returned a response the app could not use. "
                "Try again, or restart Ollama if the problem continues.",
            ),
        )
        for error_type, code, message in cases:
            with self.subTest(error_type=error_type.__name__):
                error = error_type("private implementation detail")
                self.assertEqual(error.code, code)
                self.assertEqual(error.user_message, message)
                self.assertEqual(str(error), message)
                self.assertNotIn("private implementation detail", str(error))
                with self.assertRaises(UserFacingBackendError):
                    raise error

    def test_answer_status_distinguishes_complete_empty_and_weak_retrieval(self):
        complete = answer_status([{"dense": 0.8}], weak=False)
        empty = answer_status([], weak=True)
        empty_without_weak_flag = answer_status([], weak=False)
        weak = answer_status([{"dense": 0.1}], weak=True)

        self.assertIs(complete, AnswerStatus.COMPLETE)
        self.assertIs(empty, AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL)
        self.assertIs(empty_without_weak_flag, AnswerStatus.ABSTAIN_EMPTY_RETRIEVAL)
        self.assertIs(weak, AnswerStatus.ABSTAIN_WEAK_RETRIEVAL)
        self.assertEqual(
            empty.user_message,
            "No documents were retrieved for this question. "
            "Ask a more specific question or rebuild the index if this keeps happening.",
        )
        self.assertEqual(
            weak.user_message,
            "The retrieved documents were too weak to support an answer. "
            "Rephrase the question or ask the lab technician.",
        )

    def test_missing_index_is_classified_before_loading(self):
        called = False

        def factory(_index_dir):
            nonlocal called
            called = True

        with self.assertRaises(IndexMissingError):
            load_app_retriever(
                "/tmp/missing-index",
                factory=factory,
                is_dir=lambda _path: False,
            )
        self.assertFalse(called)

    def test_corrupt_index_is_classified_without_exposing_details(self):
        def corrupt(_index_dir):
            raise ValueError("database disk image is malformed")

        with self.assertRaises(IndexLoadError) as caught:
            load_app_retriever("/tmp/index", factory=corrupt, is_dir=lambda _path: True)
        self.assertNotIn("malformed", str(caught.exception))

    def test_missing_embedding_cache_is_distinct_from_corrupt_index(self):
        def missing(_index_dir):
            raise OSError(
                "sentence_transformers could not find config.json in cached files "
                "while local_files_only is enabled"
            )

        with self.assertRaises(EmbeddingModelError) as caught:
            load_app_retriever("/tmp/index", factory=missing, is_dir=lambda _path: True)
        self.assertNotIn("config.json", str(caught.exception))

    def test_query_failure_is_wrapped_without_exposing_details(self):
        retriever = SafeRetriever(_BrokenRetriever())
        with self.assertRaises(RetrievalQueryError) as caught:
            retriever.retrieve("question")
        self.assertNotIn("sqlite", str(caught.exception))

    def test_successful_retriever_keeps_arguments_and_results(self):
        calls = []

        class Retriever:
            def retrieve(self, question, k=5, tldr_lane=True):
                calls.append((question, k, tldr_lane))
                return [{"dense": 0.1}]

        retriever = load_app_retriever(
            "/tmp/index",
            factory=lambda _path: Retriever(),
            is_dir=lambda _path: True,
        )
        self.assertEqual(retriever.retrieve("question", k=3, tldr_lane=False), [{"dense": 0.1}])
        self.assertEqual(calls, [("question", 3, False)])


if __name__ == "__main__":
    unittest.main()
