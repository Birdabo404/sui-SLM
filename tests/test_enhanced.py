#!/usr/bin/env python3
"""Phase 1 contracts for opt-in Enhanced retrieval and grounding."""
import unittest

from app.grounding import GroundingModelError, score
from app.hybrid import (
    METHOD_LABEL,
    RerankerModelError,
    merge_rrf,
    retrieve_enhanced,
)


class _Reranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = ()

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = tuple(pairs)
        return self.scores


class _GroundingScorer:
    def __init__(self, result):
        self.result = result
        self.pairs = ()

    def predict(self, pairs):
        self.pairs = tuple(pairs)
        return self.result


class EnhancedRetrievalTests(unittest.TestCase):
    def test_rrf_merge_reuses_existing_semantics_and_preserves_dense_score(self):
        dense = (
            {"id": "a", "text": "dense a", "dense": 0.71},
            {"id": "b", "text": "dense b", "dense": 0.42},
        )
        keyword = (
            {"id": "b", "text": "keyword b", "dense": 99},
            {"id": "c", "text": "keyword c"},
        )
        merged = merge_rrf(dense, keyword)

        self.assertEqual([hit["id"] for hit in merged], ["b", "a", "c"])
        self.assertEqual(merged[0]["dense"], 0.42)
        self.assertEqual(merged[0]["retrieval_methods"], ("dense", "keyword"))
        self.assertAlmostEqual(merged[0]["rrf"], 1 / 61 + 1 / 60)

    def test_enhanced_contract_requests_twenty_reranks_and_keeps_five(self):
        calls = []
        dense = tuple(
            {"id": f"d{i}", "text": f"dense {i}", "dense": i / 100}
            for i in range(20)
        )
        keyword = tuple(
            {"id": f"k{i}", "text": f"keyword {i}", "dense": i / 200}
            for i in range(20)
        )

        def search(name, hits):
            def run(question, limit):
                calls.append((name, question, limit))
                return hits[:limit]

            return run

        reranker = _Reranker(range(40))
        result = retrieve_enhanced(
            "question",
            dense_search=search("dense", dense),
            keyword_search=search("keyword", keyword),
            reranker=reranker,
        )

        self.assertEqual(calls, [("dense", "question", 20), ("keyword", "question", 20)])
        self.assertEqual(len(reranker.pairs), 40)
        self.assertEqual(len(result.hits), 5)
        self.assertEqual(result.method, METHOD_LABEL)
        self.assertEqual(result.dense_gate_similarity, 0.19)
        self.assertTrue(all("rerank" in hit for hit in result.hits))

    def test_enhanced_contract_requires_original_dense_scores_for_the_gate(self):
        with self.assertRaisesRegex(ValueError, "original dense similarity"):
            retrieve_enhanced(
                "question",
                dense_search=lambda question, limit: (),
                keyword_search=lambda question, limit: ({"id": "keyword", "text": "text"},),
                reranker=_Reranker([1.0]),
            )

    def test_reranker_failure_is_typed_and_hides_internal_detail(self):
        class BrokenReranker:
            def predict(self, pairs, show_progress_bar=False):
                raise RuntimeError("private tensor failure")

        with self.assertRaises(RerankerModelError) as caught:
            retrieve_enhanced(
                "question",
                dense_search=lambda question, limit: ({"id": "a", "text": "a", "dense": 0.8},),
                keyword_search=lambda question, limit: (),
                reranker=BrokenReranker(),
            )
        self.assertNotIn("private tensor", str(caught.exception))


class GroundingTests(unittest.TestCase):
    def test_score_returns_consistency_and_label_information(self):
        scorer = _GroundingScorer([0.8])
        result = score("Supported answer.", ("first excerpt", "second excerpt"), scorer=scorer)

        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.labels[0].label, "consistent")
        self.assertEqual(result.labels[0].text, "Supported answer.")
        self.assertEqual(
            scorer.pairs,
            (("first excerpt\n\nsecond excerpt", "Supported answer."),),
        )

    def test_grounding_failure_is_typed_and_hides_internal_detail(self):
        class BrokenScorer:
            def predict(self, pairs):
                raise RuntimeError("private model failure")

        with self.assertRaises(GroundingModelError) as caught:
            score("answer", ("excerpt",), scorer=BrokenScorer())
        self.assertNotIn("private model", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
