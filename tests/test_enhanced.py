#!/usr/bin/env python3
"""Contracts for opt-in Enhanced retrieval and grounding."""
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import grounding
from app.backend import SafeRetriever
from app.engine import is_weak
from app.grounding import (
    FLAN_TOKENIZER_REVISION,
    GROUNDING_REVISION,
    GroundingModelError,
    cache_grounding_assets,
    score,
)
from app.hybrid import (
    METHOD_LABEL,
    RERANKER_REVISION,
    RerankerModelError,
    cache_reranker_model,
    merge_rrf,
    retrieve_enhanced,
    retrieve_hybrid,
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


class _Encoded:
    def tolist(self):
        return [[1.0, 0.0]]


class _EmbeddingModel:
    def encode(self, questions, normalize_embeddings=True):
        return _Encoded()


class _Bm25:
    def get_scores(self, tokens):
        return [0.1, 4.0, 3.0]


class _Collection:
    def query(self, **kwargs):
        return {"ids": [["dense"]], "distances": [[0.05]]}

    def get(self, ids, include):
        vectors = {
            "keyword": [0.8, 0.0],
            "other": [0.7, 0.0],
        }
        return {
            "ids": ids,
            "embeddings": [vectors[hit_id] for hit_id in ids],
        }


class _HybridRetriever:
    exclude_distro = None
    model = _EmbeddingModel()
    bm25 = _Bm25()
    col = _Collection()
    rows = (
        {"id": "dense", "text": "dense text"},
        {"id": "keyword", "text": "keyword text"},
        {"id": "other", "text": "other text"},
    )

    def retrieve(self, question, k=5, tldr_lane=True):
        return [{"id": "frozen", "text": "frozen text", "dense": 0.29}]


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

    def test_real_hybrid_retriever_keeps_the_frozen_dense_abstain_decision(self):
        question = "same question"
        retriever = SafeRetriever(_HybridRetriever())
        frozen_hits = retriever.retrieve(question, k=5, tldr_lane=True)
        result = retrieve_hybrid(
            question,
            retriever,
            reranker=_Reranker([1.0, 0.9, 0.8]),
        )

        self.assertEqual(result.weak, is_weak(frozen_hits, threshold=0.30))
        self.assertTrue(result.weak)
        self.assertEqual(result.dense_gate_similarity, 0.29)
        self.assertGreater(max(hit["dense"] for hit in result.hits), 0.30)
        self.assertEqual(result.gate_hits, tuple(frozen_hits))

    def test_frozen_answer_path_never_imports_enhanced_modules(self):
        probe = """
import sys
import tempfile
from app.engine import answer_question

class Retriever:
    def retrieve(self, question, k=5, tldr_lane=True):
        return [{
            "id": "frozen",
            "title": "Frozen hit",
            "path": "frozen.txt",
            "text": "Frozen evaluated excerpt.",
            "dense": 0.9,
        }]

with tempfile.NamedTemporaryFile(suffix=".csv") as log_file:
    answer_question(
        "question",
        "rag",
        "model",
        retriever=Retriever(),
        chat_fn=lambda messages, model: "answer",
        log_path=log_file.name,
    )
assert "app.hybrid" not in sys.modules
assert "app.grounding" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


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

    def test_grounding_cache_uses_pinned_hhem_and_flan_revisions(self):
        with patch(
            "app.grounding._snapshot_download",
            side_effect=("/cache/hhem", "/cache/flan"),
        ) as download:
            paths = cache_grounding_assets(local_files_only=False)

        self.assertEqual(paths, ("/cache/hhem", "/cache/flan"))
        self.assertEqual(download.call_args_list[0].args[1], GROUNDING_REVISION)
        self.assertEqual(download.call_args_list[1].args[1], FLAN_TOKENIZER_REVISION)
        self.assertIn(
            "modeling_hhem_v2.py",
            download.call_args_list[0].kwargs["allow_patterns"],
        )
        self.assertIn(
            "tokenizer.json",
            download.call_args_list[1].kwargs["allow_patterns"],
        )

    def test_grounding_loader_redirects_foundation_to_pinned_local_tokenizer(self):
        config = SimpleNamespace(foundation="google/flan-t5-base")
        model = SimpleNamespace(to=lambda device: ("loaded", device))
        auto_config = SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
        calls = []

        def load_model(*args, **kwargs):
            calls.append((args, kwargs))
            return model

        transformers = SimpleNamespace(
            AutoConfig=auto_config,
            AutoModelForSequenceClassification=SimpleNamespace(
                from_pretrained=load_model,
            ),
        )
        grounding._load_scorer.cache_clear()
        with (
            patch(
                "app.grounding.cache_grounding_assets",
                return_value=("/cache/hhem", "/cache/flan"),
            ),
            patch(
                "app.grounding.importlib.import_module",
                return_value=transformers,
            ),
        ):
            loaded = grounding._load_scorer(
                grounding.GROUNDING_MODEL,
                GROUNDING_REVISION,
                FLAN_TOKENIZER_REVISION,
                True,
            )

        self.assertEqual(config.foundation, "/cache/flan")
        self.assertEqual(calls[0][0], ("/cache/hhem",))
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(loaded, ("loaded", "cpu"))

    def test_missing_grounding_cache_raises_typed_plain_error(self):
        with patch(
            "app.grounding.importlib.import_module",
            side_effect=RuntimeError("private hub error"),
        ):
            with self.assertRaises(GroundingModelError) as caught:
                cache_grounding_assets(local_files_only=True)
        self.assertNotIn("private hub", str(caught.exception))
        self.assertEqual(caught.exception.code, "grounding_model_unavailable")


class OfflineRerankerTests(unittest.TestCase):
    def test_missing_reranker_cache_raises_typed_plain_error(self):
        with patch(
            "app.hybrid.importlib.import_module",
            side_effect=RuntimeError("private hub error"),
        ):
            with self.assertRaises(RerankerModelError) as caught:
                cache_reranker_model(local_files_only=True)
        self.assertNotIn("private hub", str(caught.exception))
        self.assertEqual(caught.exception.code, "reranker_model_unavailable")

    def test_reranker_cache_uses_immutable_revision(self):
        snapshot_download = Mock(return_value="/cache/reranker")
        hub = SimpleNamespace(snapshot_download=snapshot_download)
        with patch("app.hybrid.importlib.import_module", return_value=hub):
            path = cache_reranker_model(local_files_only=False)

        self.assertEqual(path, "/cache/reranker")
        self.assertEqual(
            snapshot_download.call_args.kwargs["revision"],
            RERANKER_REVISION,
        )
        self.assertNotIn(
            "pytorch_model.bin",
            snapshot_download.call_args.kwargs["allow_patterns"],
        )


if __name__ == "__main__":
    unittest.main()
