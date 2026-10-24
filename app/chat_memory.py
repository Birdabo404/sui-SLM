"""App conversation context and local chat persistence."""
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.engine import NUM_CTX

ANSWER_RESERVE = 1024
MESSAGE_OVERHEAD = 4
_CHAT_ID = re.compile(r"^[0-9a-f]{32}$")
_DATABASE = "chats.sqlite3"


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


@dataclass(frozen=True)
class ChatSummary:
    chat_id: str
    title: str
    updated_at: str
    turn_count: int


@dataclass(frozen=True)
class StoredChat:
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: tuple[dict, ...]


def chat_data_dir(environ=None, home=None):
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    configured = environ.get("XDG_DATA_HOME")
    if configured and Path(configured).is_absolute():
        return Path(configured) / "sui-slm"
    return home / ".local" / "share" / "sui-slm"


class ChatStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else chat_data_dir()
        self.path = self.root / _DATABASE

    def new_chat(self):
        chat_id = uuid.uuid4().hex
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO chats
                    (id, title, created_at, updated_at, messages, turn_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, "New chat", now, now, "[]", 0),
            )
        return StoredChat(chat_id, "New chat", now, now, ())

    def list_chats(self):
        with closing(self._connect()) as db:
            rows = db.execute(
                """
                SELECT id, title, updated_at, turn_count
                FROM chats
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [
            ChatSummary(
                chat_id=row["id"],
                title=row["title"],
                updated_at=row["updated_at"],
                turn_count=row["turn_count"],
            )
            for row in rows
        ]

    def load_chat(self, chat_id):
        self._validate_id(chat_id)
        with closing(self._connect()) as db:
            row = db.execute(
                """
                SELECT id, title, created_at, updated_at, messages
                FROM chats
                WHERE id = ?
                """,
                (chat_id,),
            ).fetchone()
        if row is None:
            raise KeyError(chat_id)
        messages = json.loads(row["messages"])
        return StoredChat(
            chat_id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=tuple(messages),
        )

    def save_chat(self, chat_id, messages, title=None):
        self._validate_id(chat_id)
        messages = list(messages)
        payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        updated_at = _now()
        derived_title = title or _chat_title(messages)
        with closing(self._connect()) as db, db:
            current = db.execute(
                "SELECT title, created_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
            if current is None:
                raise KeyError(chat_id)
            chat_title = derived_title or current["title"]
            changed = db.execute(
                """
                UPDATE chats
                SET title = ?, updated_at = ?, messages = ?, turn_count = ?
                WHERE id = ?
                """,
                (
                    chat_title,
                    updated_at,
                    payload,
                    sum(message.get("role") == "user" for message in messages),
                    chat_id,
                ),
            )
            if changed.rowcount != 1:
                raise KeyError(chat_id)
        return StoredChat(
            chat_id=chat_id,
            title=chat_title,
            created_at=current["created_at"],
            updated_at=updated_at,
            messages=tuple(messages),
        )

    def append_messages(self, chat_id, messages):
        """Add messages to the end of a chat. The read and the write share one write transaction, so another tab
        appending to the same chat can't drop these messages or have its own dropped."""
        self._validate_id(chat_id)
        added = list(messages)
        updated_at = _now()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT title, created_at, messages FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
            if current is None:
                raise KeyError(chat_id)
            messages = [*json.loads(current["messages"]), *added]
            chat_title = _chat_title(messages) or current["title"]
            db.execute(
                """
                UPDATE chats
                SET title = ?, updated_at = ?, messages = ?, turn_count = ?
                WHERE id = ?
                """,
                (
                    chat_title,
                    updated_at,
                    json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    sum(message.get("role") == "user" for message in messages),
                    chat_id,
                ),
            )
        return StoredChat(
            chat_id=chat_id,
            title=chat_title,
            created_at=current["created_at"],
            updated_at=updated_at,
            messages=tuple(messages),
        )

    def delete_chat(self, chat_id):
        self._validate_id(chat_id)
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                messages TEXT NOT NULL,
                turn_count INTEGER NOT NULL
            )
            """
        )
        db.commit()
        self.path.chmod(0o600)
        return db

    @staticmethod
    def _validate_id(chat_id):
        if not _CHAT_ID.fullmatch(str(chat_id or "")):
            raise ValueError("invalid chat id")


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


def _chat_title(messages):
    for message in messages:
        if message.get("role") == "user":
            title = " ".join(str(message.get("content") or "").split())
            if len(title) > 80:
                return title[:79] + "…"
            return title
    return ""


def _now():
    return datetime.now(timezone.utc).isoformat()
