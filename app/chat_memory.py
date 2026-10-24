"""App-only conversation context. Persistence is added separately."""
from dataclasses import dataclass

from app.engine import NUM_CTX

ANSWER_RESERVE = 1024
MESSAGE_OVERHEAD = 4


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass(frozen=True)
class ChatContext:
    retrieval_query: str
    turns: tuple[Turn, ...]
    retrieval_turns: int = 0

    def apply(self, messages):
        applied, _count = self.apply_with_count(messages)
        return applied

    def apply_with_count(self, messages):
        remaining = NUM_CTX - ANSWER_RESERVE - _message_tokens(messages)
        kept = []
        for turn in reversed(self.turns):
            pair = (
                {"role": "user", "content": turn.question},
                {"role": "assistant", "content": turn.answer},
            )
            cost = _message_tokens(pair)
            if cost > remaining:
                break
            kept.append(pair)
            remaining -= cost
        history = [message for pair in reversed(kept) for message in pair]
        applied = [messages[0], *history, *messages[1:]]
        return applied, max(self.retrieval_turns, len(kept))


def build_chat_context(messages, question):
    previous = _previous_completed_question(messages)
    query = f"{previous} {question}" if previous else question
    turns = _completed_turns(messages)
    if not previous and not turns:
        return None
    return ChatContext(
        retrieval_query=query,
        turns=turns,
        retrieval_turns=int(bool(previous)),
    )


def _previous_completed_question(messages):
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        assistant = next(
            (
                candidate
                for candidate in messages[index + 1:]
                if candidate.get("role") == "assistant"
            ),
            None,
        )
        if assistant is None or "error" in assistant:
            return ""
        if any(key in assistant for key in ("record", "compare", "content")):
            return str(message.get("content") or "")
        return ""
    return ""


def _completed_turns(messages):
    turns = []
    question = None
    for message in messages:
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            continue
        if message.get("role") != "assistant" or question is None:
            continue
        record = message.get("record")
        if record:
            turns.append(Turn(question=question, answer=str(record.get("answer") or "")))
        elif "content" in message and not message.get("error"):
            turns.append(Turn(question=question, answer=str(message.get("content") or "")))
        question = None
    return tuple(turns)


def _message_tokens(messages):
    return sum(
        len(str(message.get("content") or "").encode("utf-8")) + MESSAGE_OVERHEAD
        for message in messages
    )
