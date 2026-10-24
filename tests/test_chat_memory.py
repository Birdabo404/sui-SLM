#!/usr/bin/env python3
"""App-only multi-turn context tests."""
import tempfile
import unittest
from pathlib import Path

from app.chat_memory import ChatContext, Turn, build_chat_context
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
    def test_query_uses_the_previous_user_question(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "record": {"answer": "first answer"}},
            {"role": "user", "content": "stopped question"},
        ]
        context = build_chat_context(messages, "new question")
        self.assertEqual(
            context.retrieval_query,
            "stopped question new question",
        )

    def test_completed_turns_are_inserted_after_the_system_message(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "record": {"answer": "first answer"}},
            {"role": "user", "content": "failed question"},
            {"role": "assistant", "error": "failed"},
        ]
        context = build_chat_context(messages, "current")
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
        fitted = context.apply([
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
            [{"role": "user", "content": "previous question"}],
            "current question",
        )
        with tempfile.TemporaryDirectory() as td:
            result = answer_question(
                "current question",
                "full",
                "m",
                retriever=retriever,
                chat_fn=lambda messages, model: self.fail("chat should not run"),
                log_path=Path(td) / "answers.csv",
                context=context,
            )
        self.assertEqual(
            retriever.calls,
            [("previous question current question", 5, True)],
        )
        self.assertTrue(result["weak"])
        self.assertEqual(result["text"], ABSTAIN)


if __name__ == "__main__":
    unittest.main()
