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

    def apply(self, messages):
        if not self.turns:
            return list(messages)
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
        return [messages[0], *history, *messages[1:]]


def build_chat_context(messages, question):
    previous = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    query = f"{previous} {question}" if previous else question
    turns = _completed_turns(messages)
    if not previous and not turns:
        return None
    return ChatContext(retrieval_query=query, turns=turns)


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
