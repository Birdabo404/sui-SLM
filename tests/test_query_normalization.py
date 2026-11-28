"""Rule-based query normalization and its persisted opt-in setting."""

import json
import tempfile
import unittest
from pathlib import Path

from app.prefs import (
    NORMALIZE_ENV,
    normalization_env_override,
    save_normalization,
    saved_normalization,
)
from app.query_normalization import normalize_query, normalize_retrieval_query


class QueryNormalizationTests(unittest.TestCase):
    def test_filler_heavy_question(self):
        question = (
            "Hello, could you please tell me how to install Python "
            "on my Ubuntu 24.04 laptop? Thanks!"
        )
        self.assertEqual(
            normalize_query(question),
            "how to install Python on Ubuntu 24.04",
        )

    def test_preserves_command_flags_path_and_glob_byte_for_byte(self):
        command = "sudo find /var/log -type f -path '/tmp/please/*.log'"
        question = f"Would you please show me how to run {command} on my computer, thanks?"
        normalized = normalize_query(question)
        self.assertEqual(normalized, f"how to run {command}")
        self.assertIn(command, normalized)

    def test_already_terse_query_is_unchanged(self):
        query = "sudo apt install python3"
        self.assertEqual(normalize_query(query), query)

    def test_empty_result_falls_back_to_original(self):
        self.assertEqual(normalize_query("Please?"), "Please?")

    def test_normalizes_current_question_after_conversation_context(self):
        question = "Could you please show me how to run chmod +x script?"
        contextual = f"make a script executable {question}"
        self.assertEqual(
            normalize_retrieval_query(question, contextual),
            "make a script executable how to run chmod +x script",
        )


class NormalizationPreferenceTests(unittest.TestCase):
    def test_defaults_off_and_persists_a_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(saved_normalization(root, {}))
            self.assertTrue(save_normalization(True, root))
            self.assertTrue(saved_normalization(root, {}))
            self.assertEqual(
                json.loads((root / "prefs.json").read_text(encoding="utf-8")),
                {"normalize_question_before_search": True},
            )

    def test_environment_value_overrides_saved_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_normalization(True, root)
            self.assertFalse(saved_normalization(root, {NORMALIZE_ENV: "off"}))
            self.assertTrue(saved_normalization(root, {NORMALIZE_ENV: "YES"}))
            self.assertIsNone(normalization_env_override({NORMALIZE_ENV: "invalid"}))


if __name__ == "__main__":
    unittest.main()
