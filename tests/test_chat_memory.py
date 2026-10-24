#!/usr/bin/env python3
"""App-only multi-turn context tests."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.chat_memory import (
    ChatContext,
    ChatStore,
    Turn,
    build_chat_context,
    chat_data_dir,
)
from app.engine import answer_question
from shared.prompt import ABSTAIN


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, question, k=5, tldr_lane=True):
        self.calls.append((question, k, tldr_lane))
        return self.hits[:k]


class ChatContextTests(unittest.TestCase):
    def test_query_ignores_a_cancelled_previous_turn(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "record": {"answer": "first answer"}},
            {"role": "user", "content": "stopped question"},
        ]
        context = build_chat_context(messages, "new question")
        self.assertEqual(context.retrieval_query, "new question")

    def test_completed_turns_are_inserted_after_the_system_message(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "record": {"answer": "first answer"}},
            {"role": "user", "content": "failed question"},
            {"role": "assistant", "error": "failed"},
        ]
        context = build_chat_context(messages, "current")
        self.assertEqual(context.retrieval_query, "current")
        current = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current"},
        ]
        self.assertEqual(
            context.apply(current),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "current"},
            ],
        )

    def test_context_drops_oldest_whole_turns_to_fit_num_ctx(self):
        context = ChatContext(
            retrieval_query="latest current",
            turns=(
                Turn("old", "x" * 20_000),
                Turn("latest", "short"),
            ),
        )
        fitted, count = context.apply_with_count([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current"},
        ])
        self.assertEqual(
            fitted,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "latest"},
                {"role": "assistant", "content": "short"},
                {"role": "user", "content": "current"},
            ],
        )
        self.assertEqual(count, 1)

    def test_weak_retrieval_uses_combined_query_and_frozen_policy(self):
        retriever = FakeRetriever([{
            "id": "weak",
            "title": "unrelated",
            "text": "nothing",
            "source": "tldr",
            "path": "x.md",
            "rrf": 1 / 60,
            "dense": 0.10,
        }])
        context = build_chat_context(
            [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "record": {"answer": "previous answer"}},
            ],
            "current question",
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "answers.csv"
            result = answer_question(
                "current question",
                "full",
                "m",
                retriever=retriever,
                chat_fn=lambda messages, model: self.fail("chat should not run"),
                log_path=log,
                context=context,
            )
            with open(log, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
        self.assertEqual(
            retriever.calls,
            [("previous question current question", 5, True)],
        )
        self.assertTrue(result["weak"])
        self.assertEqual(result["text"], ABSTAIN)
        self.assertEqual(result["context_turns"], 1)
        self.assertEqual(row["context_turns"], "1")


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.store = ChatStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_xdg_data_home_is_respected(self):
        self.assertEqual(
            chat_data_dir(
                {"XDG_DATA_HOME": "/tmp/xdg"},
                home="/home/student",
            ),
            Path("/tmp/xdg/sui-slm"),
        )
        self.assertEqual(
            chat_data_dir({}, home="/home/student"),
            Path("/home/student/.local/share/sui-slm"),
        )
        self.assertEqual(
            chat_data_dir(
                {"XDG_DATA_HOME": "relative"},
                home="/home/student",
            ),
            Path("/home/student/.local/share/sui-slm"),
        )

    def test_new_save_list_load_delete_round_trip(self):
        with patch(
            "app.chat_memory._now",
            side_effect=(
                "2026-10-24T09:00:00+00:00",
                "2026-10-24T09:01:00+00:00",
            ),
        ):
            chat = self.store.new_chat()
            saved = self.store.save_chat(
                chat.chat_id,
                [
                    {"role": "user", "content": "How do I list files?"},
                    {
                        "role": "assistant",
                        "record": {"answer": "Run `ls`.", "mode": "full"},
                    },
                ],
            )
        self.assertEqual(saved.title, "How do I list files?")
        self.assertEqual(saved.created_at, "2026-10-24T09:00:00+00:00")
        self.assertEqual(saved.updated_at, "2026-10-24T09:01:00+00:00")
        summaries = self.store.list_chats()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].chat_id, chat.chat_id)
        self.assertEqual(summaries[0].turn_count, 1)
        loaded = ChatStore(self.root).load_chat(chat.chat_id)
        self.assertEqual(loaded, saved)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)
        self.store.delete_chat(chat.chat_id)
        self.store.delete_chat(chat.chat_id)
        self.assertEqual(self.store.list_chats(), [])

    def test_list_is_newest_first(self):
        with patch(
            "app.chat_memory._now",
            side_effect=(
                "2026-10-24T09:00:00+00:00",
                "2026-10-24T10:00:00+00:00",
            ),
        ):
            first = self.store.new_chat()
            second = self.store.new_chat()
        self.assertEqual(
            [chat.chat_id for chat in self.store.list_chats()],
            [second.chat_id, first.chat_id],
        )

    def test_ids_cannot_escape_the_store(self):
        for chat_id in ("", "../outside", "a/b", "A" * 32):
            with self.subTest(chat_id=chat_id):
                with self.assertRaises(ValueError):
                    self.store.load_chat(chat_id)
                with self.assertRaises(ValueError):
                    self.store.save_chat(chat_id, [])
                with self.assertRaises(ValueError):
                    self.store.delete_chat(chat_id)

    def test_missing_chat_load_and_save_fail(self):
        chat_id = "0" * 32
        with self.assertRaises(KeyError):
            self.store.load_chat(chat_id)
        with self.assertRaises(KeyError):
            self.store.save_chat(chat_id, [])
        self.store.delete_chat(chat_id)


if __name__ == "__main__":
    unittest.main()
