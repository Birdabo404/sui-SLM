#!/usr/bin/env python3
"""Mocked Ollama HTTP tests for think=false and matched-model /api/show checks."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.engine import THESIS_CONFIGS, answer_question, build_messages, ollama_chat
from eval.run_eval import check_matched_models, run_eval
from shared.ollama_api import (
    REQUIRED_QUANT,
    THINK,
    assert_matched_models,
    inspect_chat_response,
    matched_model_diffs,
    model_fields,
    ollama_show,
    strip_think_leaks,
)
from shared.prompt import SYSTEM, SYSTEM_BASE, system_for_mode


TEMPLATE = "{{ if .System }}system{{ .System }}{{ end }}{{ .Prompt }}"


def show_payload(quant=REQUIRED_QUANT, template=TEMPLATE, system="", digest="sha256:aaa"):
    return {
        "template": template,
        "system": system,
        "digest": digest,
        "details": {"quantization_level": quant, "format": "gguf"},
        "model_info": {"general.file_type": quant},
    }


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k=5, tldr_lane=True):
        return list(self.hits)[:k]


class ThinkParseTests(unittest.TestCase):
    def test_strip_closed_and_open_think_tags(self):
        cleaned, leaked = strip_think_leaks("<think>secret</think>\nls -l")
        self.assertEqual(cleaned, "ls -l")
        self.assertTrue(leaked)
        cleaned, leaked = strip_think_leaks("<think>never closed\nstill going")
        self.assertEqual(cleaned, "")
        self.assertTrue(leaked)
        cleaned, leaked = strip_think_leaks("just an answer")
        self.assertEqual(cleaned, "just an answer")
        self.assertFalse(leaked)

    def test_inspect_flags_thinking_field(self):
        out = inspect_chat_response(
            {"message": {"content": "ls -l", "thinking": "I should list files"}},
            think=False,
        )
        self.assertEqual(out["text"], "ls -l")
        self.assertEqual(out["raw_output"], "ls -l")
        self.assertEqual(out["thinking"], "I should list files")
        self.assertEqual(out["think"], "false")
        self.assertEqual(out["think_leak"], 1)

    def test_inspect_strips_tags_in_content(self):
        out = inspect_chat_response(
            {"message": {"content": "<think>nope</think>\napt install vim"}},
            think=False,
        )
        self.assertEqual(out["text"], "apt install vim")
        self.assertIn("<think>", out["raw_output"])
        self.assertEqual(out["think_leak"], 1)
        self.assertEqual(out["think"], "false")


class OllamaChatHttpTests(unittest.TestCase):
    def test_chat_posts_think_false_and_strips_leak(self):
        fake = MagicMock()
        fake.json.return_value = {
            "message": {
                "content": "<think>internal</think>\nuse `ls`",
                "thinking": "",
            }
        }
        with patch("app.engine.requests.post", return_value=fake) as post:
            out = ollama_chat(
                [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": "q"}],
                "qwen3.5:2b",
            )
        body = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["think"], False)
        self.assertIs(body["think"], THINK)
        self.assertNotIn("think", body["options"])
        self.assertEqual(out["text"], "use `ls`")
        self.assertEqual(out["think_leak"], 1)
        self.assertEqual(out["think"], "false")
        self.assertIn("<think>", out["raw_output"])

    def test_every_thesis_mode_sends_shared_system(self):
        hits = [{"title": "ls", "text": "ls -l"}]
        for _model_key, mode in THESIS_CONFIGS:
            msgs = build_messages("how do I list files", hits, mode)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[0]["content"], system_for_mode(mode))
        self.assertFalse(THINK)


class AnswerLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "answers.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_logs_think_and_raw_and_strips_leak(self):
        def chat(messages, model):
            return "<think>hidden</think>\nfinal answer"

        result = answer_question(
            "how do I list files",
            "base",
            "qwen3.5:2b",
            chat_fn=chat,
            log_path=self.log,
            model_key="base",
        )
        self.assertEqual(result["text"], "final answer")
        self.assertEqual(result["think"], "false")
        self.assertEqual(result["think_leak"], 1)
        self.assertIn("<think>", result["raw_output"])
        with open(self.log, newline="", encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["think"], "false")
        self.assertEqual(row["think_leak"], "1")
        self.assertIn("<think>", row["raw_output"])
        self.assertEqual(row["answer"], "final answer")

    def test_abstain_logs_think_without_raw_model_output(self):
        retriever = FakeRetriever([{
            "id": "weak", "title": "unrelated", "text": "nothing",
            "source": "tldr", "path": "x.md", "rrf": 1 / 60, "dense": 0.10,
        }])
        result = answer_question(
            "how do I time travel",
            "full",
            "qwen3.5:2b",
            retriever=retriever,
            chat_fn=lambda messages, model: "should not run",
            log_path=self.log,
            model_key="base",
        )
        self.assertEqual(result["think"], "false")
        self.assertEqual(result["raw_output"], "")
        self.assertEqual(result["think_leak"], 0)
        with open(self.log, newline="", encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["think"], "false")
        self.assertEqual(row["raw_output"], "")


class MatchedModelTests(unittest.TestCase):
    def test_same_quant_template_empty_system_ok(self):
        base = model_fields(show_payload(digest="sha256:base"), name="qwen3.5:2b")
        distilled = model_fields(show_payload(digest="sha256:student"), name="student:latest")
        self.assertEqual(matched_model_diffs(base, distilled), [])
        assert_matched_models(base, distilled)

    def test_shared_system_on_both_ok(self):
        payload = show_payload(system=SYSTEM)
        base = model_fields(payload, name="qwen3.5:2b")
        distilled = model_fields(payload, name="student:latest")
        self.assertEqual(matched_model_diffs(base, distilled), [])

    def test_quant_mismatch_exits_with_diff(self):
        base = model_fields(show_payload(quant="Q8_0"), name="qwen3.5:2b")
        distilled = model_fields(show_payload(), name="student:latest")
        diffs = matched_model_diffs(base, distilled)
        self.assertTrue(any("Q8_0" in d and "Q4_K_M" in d for d in diffs))
        with self.assertRaises(SystemExit) as cm:
            assert_matched_models(base, distilled)
        self.assertIn("Matched-model check failed", str(cm.exception))
        self.assertIn("Q8_0", str(cm.exception))

    def test_template_mismatch_exits(self):
        base = model_fields(show_payload(template="TEMPLATE-A"), name="qwen3.5:2b")
        distilled = model_fields(show_payload(template="TEMPLATE-B"), name="student:latest")
        with self.assertRaises(SystemExit) as cm:
            assert_matched_models(base, distilled)
        self.assertIn("chat templates differ", str(cm.exception))

    def test_system_not_from_shared_exits(self):
        base = model_fields(show_payload(system="You are a pirate."), name="qwen3.5:2b")
        distilled = model_fields(show_payload(system="You are a pirate."), name="student:latest")
        with self.assertRaises(SystemExit) as cm:
            assert_matched_models(base, distilled)
        msg = str(cm.exception)
        self.assertIn("shared/prompt.py", msg)
        self.assertIn("pirate", msg)

    def test_file_type_15_counts_as_q4_k_m(self):
        payload = {
            "template": TEMPLATE,
            "system": "",
            "digest": "sha256:x",
            "details": {"format": "gguf"},
            "model_info": {"general.file_type": 15},
        }
        info = model_fields(payload, name="qwen3.5:2b")
        self.assertEqual(info["quant"], REQUIRED_QUANT)

    def test_show_http_and_tags_digest_fallback(self):
        show_resp = MagicMock()
        show_resp.json.return_value = {
            "template": TEMPLATE,
            "system": "",
            "details": {"quantization_level": "Q4_K_M"},
        }
        tags_resp = MagicMock()
        tags_resp.json.return_value = {
            "models": [{"name": "qwen3.5:2b", "digest": "sha256:from-tags"}]
        }

        def post(url, json=None, timeout=None):
            self.assertTrue(url.endswith("/api/show"))
            self.assertEqual(json, {"model": "qwen3.5:2b"})
            return show_resp

        def get(url, timeout=None):
            self.assertTrue(url.endswith("/api/tags"))
            return tags_resp

        with patch("shared.ollama_api.requests.post", side_effect=post):
            with patch("shared.ollama_api.requests.get", side_effect=get):
                data = ollama_show("qwen3.5:2b", host="http://127.0.0.1:11434")
        self.assertEqual(data["digest"], "sha256:from-tags")

    def test_show_http_failure_exits(self):
        import requests as req_mod

        with patch("shared.ollama_api.requests.post", side_effect=req_mod.ConnectionError("down")):
            with self.assertRaises(SystemExit) as cm:
                ollama_show("qwen3.5:2b")
        self.assertIn("/api/show failed", str(cm.exception))
        self.assertIn("qwen3.5:2b", str(cm.exception))


class EvalShowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.questions = self.dir / "q.jsonl"
        rows = [
            {"id": "q1", "question": "how do I list files", "answerable": True,
             "gold_answer": "ls", "source": "tldr ls"},
        ]
        with open(self.questions, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.out = self.dir / "eval.csv"
        self.hits = [{
            "id": "ls#0", "title": "ls (tldr common)", "text": "ls -l",
            "source": "tldr", "path": "tldr/common/ls.md", "rrf": 0.05, "dense": 0.55,
        }]

    def tearDown(self):
        self.tmp.cleanup()

    def test_eval_logs_digest_from_show(self):
        shown = []

        def show(name):
            shown.append(name)
            digest = "sha256:base" if name == "qwen3.5:2b" else "sha256:student"
            return show_payload(digest=digest)

        n, skipped, _sweep = run_eval(
            self.questions,
            self.out,
            distilled_model="student:latest",
            retriever=FakeRetriever(self.hits),
            chat_fn=lambda messages, model: "ok",
            split="dev",
            sweep=False,
            show_fn=show,
        )
        self.assertEqual(n, 5)
        self.assertEqual(skipped, [])
        self.assertEqual(shown, ["qwen3.5:2b", "student:latest"])
        with open(self.out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["think"] == "false" for r in rows))
        base_rows = [r for r in rows if r["model"] == "qwen3.5:2b"]
        dist_rows = [r for r in rows if r["model"] == "student:latest"]
        self.assertTrue(all(r["model_digest"] == "sha256:base" for r in base_rows))
        self.assertTrue(all(r["model_digest"] == "sha256:student" for r in dist_rows))
        self.assertTrue(all(r["raw_output"] == "ok" for r in rows))

    def test_eval_exits_on_quant_mismatch_before_rows(self):
        def show(name):
            if name == "qwen3.5:2b":
                return show_payload(quant="Q8_0", digest="sha256:base")
            return show_payload(digest="sha256:student")

        with self.assertRaises(SystemExit) as cm:
            run_eval(
                self.questions,
                self.out,
                distilled_model="student:latest",
                retriever=FakeRetriever(self.hits),
                chat_fn=lambda messages, model: "ok",
                split="dev",
                sweep=False,
                show_fn=show,
            )
        self.assertIn("Q8_0", str(cm.exception))
        self.assertFalse(self.out.exists())

    def test_injected_chat_skips_live_show(self):
        with patch("eval.run_eval.ollama_show") as show:
            n, skipped, _sweep = run_eval(
                self.questions,
                self.out,
                distilled_model="student:latest",
                retriever=FakeRetriever(self.hits),
                chat_fn=lambda messages, model: "ok",
                split="dev",
                sweep=False,
            )
        show.assert_not_called()
        self.assertEqual(n, 5)
        self.assertEqual(skipped, [])

    def test_check_matched_models_uses_show_fn(self):
        seen = []

        def show(name):
            seen.append(name)
            return show_payload(digest=f"sha256:{name}")

        digests = check_matched_models(
            THESIS_CONFIGS,
            "qwen3.5:2b",
            "student:latest",
            show_fn=show,
        )
        self.assertEqual(seen, ["qwen3.5:2b", "student:latest"])
        self.assertEqual(digests["qwen3.5:2b"], "sha256:qwen3.5:2b")
        self.assertEqual(digests["student:latest"], "sha256:student:latest")


if __name__ == "__main__":
    unittest.main()
