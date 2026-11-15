#!/usr/bin/env python3
"""DEV-only Enhanced comparison tests without Ollama or model downloads."""
import tempfile
import unittest
from pathlib import Path

from scripts.compare_enhanced import (
    DEFAULT_REPORTED_PATH,
    DEV_PATH,
    answer_correct,
    check_reported_questions,
    gold_source_matches,
    load_jsonl,
    mode_summary,
    normalized_question,
    placeholder_reported_paths,
    reciprocal_rank,
    refuse_reported_overlap,
    validate_dev_rows,
    write_csv,
    write_summary,
)


class DevBoundaryTests(unittest.TestCase):
    def test_committed_dev_is_valid_and_separate_from_reported_test(self):
        dev = load_jsonl(DEV_PATH)

        validate_dev_rows(dev)
        check_reported_questions(dev, (DEFAULT_REPORTED_PATH,))

        self.assertEqual(len(dev), 32)
        self.assertTrue(all(row["set"] in {"tuning", "unanswerable"} for row in dev))

    def test_overlap_guard_rejects_ids_and_normalized_questions(self):
        dev = [
            {
                "id": "dev-one",
                "question": "How do I list files?",
            }
        ]

        with self.assertRaisesRegex(ValueError, "DEV overlaps"):
            refuse_reported_overlap(
                dev,
                [{"id": "dev-one", "question": "different"}],
            )
        with self.assertRaisesRegex(ValueError, "matches dev-one"):
            refuse_reported_overlap(
                dev,
                [{"id": "test-one", "question": " HOW do I list files "}],
            )

    def test_example_questions_are_detected_as_dev_overlap(self):
        dev = load_jsonl(DEV_PATH)
        example = DEV_PATH.parent / "questions.example.jsonl"

        with self.assertRaisesRegex(ValueError, "DEV overlaps"):
            check_reported_questions(dev, (example,))

    def test_placeholder_reported_set_is_identified(self):
        self.assertEqual(
            placeholder_reported_paths((DEFAULT_REPORTED_PATH,)),
            [str(DEFAULT_REPORTED_PATH)],
        )

    def test_dev_validator_refuses_held_out_or_placeholder_rows(self):
        base = {
            "id": "bad",
            "set": "held-out",
            "question": "question",
            "answerable": True,
            "gold_answer": "answer",
            "source": "source",
        }
        with self.assertRaisesRegex(ValueError, "non-DEV"):
            validate_dev_rows([base])

        placeholder = {**base, "set": "tuning", "placeholder": True}
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate_dev_rows([placeholder])

    def test_question_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(
            normalized_question("  How do I list files?! "),
            "how do i list files",
        )


class MetricTests(unittest.TestCase):
    def test_tldr_gold_source_matches_path_or_title(self):
        by_path = {
            "source": "tldr",
            "path": "tldr/pages/common/tar.md",
            "title": "Unrelated",
        }
        by_title = {
            "source": "tldr",
            "path": "other.md",
            "title": "tar (tldr common)",
        }

        self.assertTrue(gold_source_matches("tldr tar", by_path))
        self.assertTrue(gold_source_matches("tldr tar", by_title))
        self.assertFalse(gold_source_matches("tldr ssh", by_path))

    def test_reciprocal_rank_uses_first_gold_source_hit(self):
        hits = (
            {"source": "tldr", "path": "wrong.md", "title": "wrong"},
            {"source": "tldr", "path": "tar.md", "title": "tar"},
            {"source": "tldr", "path": "tar.md", "title": "tar"},
        )

        self.assertEqual(reciprocal_rank(hits, "tldr tar"), 0.5)
        self.assertEqual(reciprocal_rank(hits, "tldr ssh"), 0.0)

    def test_answer_score_uses_gold_command_and_abstention(self):
        answerable = {
            "answerable": True,
            "gold_answer": "sudo apt install package",
        }
        unanswerable = {"answerable": False, "gold_answer": ""}

        self.assertTrue(
            answer_correct(
                answerable,
                "Run `sudo apt install package` in the terminal.",
                False,
            )
        )
        self.assertFalse(answer_correct(answerable, "Use apt.", False))
        self.assertTrue(answer_correct(unanswerable, "abstained", True))
        self.assertFalse(answer_correct(unanswerable, "answered", False))

    def test_summary_calculates_retrieval_answer_and_resource_metrics(self):
        rows = [
            {
                "retrieval_hit": "1",
                "reciprocal_rank": "1.0",
                "abstained": "0",
                "answer_correct": "1",
                "answerable": "1",
                "grounding_score": "",
                "latency_s": "1.0",
                "total_rss_mb": "400",
            },
            {
                "retrieval_hit": "0",
                "reciprocal_rank": "0.0",
                "abstained": "1",
                "answer_correct": "1",
                "answerable": "0",
                "grounding_score": "",
                "latency_s": "3.0",
                "total_rss_mb": "500",
            },
        ]

        summary = mode_summary(rows)

        self.assertEqual(summary["hit_at_5"], 0.5)
        self.assertEqual(summary["mrr"], 0.5)
        self.assertEqual(summary["answer_accuracy"], 1.0)
        self.assertEqual(summary["answerable_accuracy"], 1.0)
        self.assertEqual(summary["abstain_rate"], 0.5)
        self.assertEqual(summary["median_latency_s"], 2.0)
        self.assertEqual(summary["peak_total_rss_mb"], 500.0)


class OutputTests(unittest.TestCase):
    def test_csv_and_markdown_outputs_are_written(self):
        row = {
            "mode": "frozen",
            "question_id": "q",
            "question": "question",
            "answerable": 1,
            "gold_source": "tldr tar",
            "gold_answer": "tar -xzf archive.tar.gz",
            "retrieval_hit": 1,
            "reciprocal_rank": "1.000000",
            "abstained": 0,
            "answer": "",
            "answer_correct": "",
            "grounding_score": "",
            "latency_s": "0.1",
            "app_rss_mb": "100",
            "ollama_rss_mb": "0",
            "total_rss_mb": "100",
            "system_used_mb": "1000",
        }
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "comparison.csv"
            markdown_path = csv_path.with_suffix(".md")

            write_csv(csv_path, [row])
            write_summary(
                markdown_path,
                [row],
                retrieval_only=True,
                reported_paths=(DEFAULT_REPORTED_PATH,),
            )

            self.assertIn("question_id", csv_path.read_text(encoding="utf-8"))
            summary = markdown_path.read_text(encoding="utf-8")
            self.assertIn("frozen", summary)
            self.assertIn("not run", summary)
            self.assertIn("placeholder questions only", summary)


if __name__ == "__main__":
    unittest.main()
