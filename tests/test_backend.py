#!/usr/bin/env python3
"""App backend failure boundaries. No live Ollama, Chroma, or model cache."""
import unittest

from app.backend import (
    EmbeddingModelError,
    IndexLoadError,
    RetrievalQueryError,
    SafeRetriever,
    load_app_retriever,
)


class _BrokenRetriever:
    def retrieve(self, question, k=5, tldr_lane=True):
        raise RuntimeError("sqlite internals that must not reach the UI")


class BackendBoundaryTests(unittest.TestCase):
    def test_corrupt_index_is_classified_without_exposing_details(self):
        def corrupt(_index_dir):
            raise ValueError("database disk image is malformed")

        with self.assertRaises(IndexLoadError) as caught:
            load_app_retriever("/tmp/index", factory=corrupt)
        self.assertNotIn("malformed", str(caught.exception))

    def test_missing_embedding_cache_is_distinct_from_corrupt_index(self):
        def missing(_index_dir):
            raise OSError(
                "sentence_transformers could not find config.json in cached files "
                "while local_files_only is enabled"
            )

        with self.assertRaises(EmbeddingModelError) as caught:
            load_app_retriever("/tmp/index", factory=missing)
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

        retriever = load_app_retriever("/tmp/index", factory=lambda _path: Retriever())
        self.assertEqual(retriever.retrieve("question", k=3, tldr_lane=False), [{"dense": 0.1}])
        self.assertEqual(calls, [("question", 3, False)])


if __name__ == "__main__":
    unittest.main()
