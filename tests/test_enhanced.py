#!/usr/bin/env python3
"""Contracts for opt-in Enhanced grounding."""
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import grounding
from app.grounding import (
    FLAN_TOKENIZER_REVISION,
    GROUNDING_REVISION,
    GroundingModelError,
    cache_grounding_assets,
    score,
)


class _GroundingScorer:
    def __init__(self, result):
        self.result = result
        self.pairs = ()

    def predict(self, pairs):
        self.pairs = tuple(pairs)
        return self.result


class GroundingTests(unittest.TestCase):
    def test_frozen_answer_path_never_imports_grounding(self):
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
assert "app.grounding" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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


if __name__ == "__main__":
    unittest.main()
