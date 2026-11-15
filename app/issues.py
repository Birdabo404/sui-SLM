"""What went wrong, in words a student can act on: one small card per failure instead of a traceback.

`issue_for_error` is the one place the UI turns an exception into a message. It reads the exception's class names
as well as its base classes, so the backend's own error types (OllamaUnavailable, OllamaTimedOut,
OllamaDisconnected, OllamaModelMissing, IndexLoadError, EmbeddingModelError, RetrievalQueryError) are named
correctly once they exist, and the plain requests / socket errors raised today are named the same way.
"""
import os
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import requests

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_BUILD = "python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index"


@dataclass(frozen=True)
class Issue:
    kind: str
    title: str
    detail: str
    fix: str = ""  # one command that fixes it, shown as code
    hint: str = ""  # the step after the command, or the step when there is no command

    def text(self):
        """The same message as plain text, for chat history and the chat memory that skips failed turns."""
        parts = [self.title + ".", self.detail]
        if self.fix:
            parts.append(f"Run `{self.fix}`.")
        if self.hint:
            parts.append(self.hint)
        return " ".join(p for p in parts if p)

    def as_dict(self):
        return asdict(self)


def issue_from_dict(data, fallback=""):
    """The stored issue of a failed turn, or one built from its plain error text (chats saved before issues)."""
    if isinstance(data, dict) and data.get("title"):
        fields = {k: str(data.get(k) or "") for k in ("kind", "title", "detail", "fix", "hint")}
        return Issue(**fields)
    # Older chats kept one string: a sentence for the student, then the raw exception after a blank line.
    first = str(fallback or "").strip().split("\n\n")[0].replace("`", "")
    return Issue("failed", "That answer failed", first)


def ollama_down(host):
    return Issue(
        "ollama_down",
        "Ollama isn't running",
        f"Nothing answered at {host}, so no model can reply.",
        fix="ollama serve",
        hint="Start it in a terminal (or open the app with sui-slm, which starts it), then ask again.",
    )


def model_missing(name, create):
    built = create.startswith("ollama create")
    return Issue(
        "model_missing",
        f"{name} isn't installed",
        "Ollama is running but doesn't have this model, so nothing was asked and no other model stood in.",
        fix=create,
        hint=("Run it where the KD export's Modelfile is (kd/README.md), then ask again." if built
              else "It downloads once; after that it works offline. Then ask again."),
    )


def index_missing(index_dir):
    return Issue(
        "index_missing",
        "The lab docs index isn't there",
        f"No index at {_short(index_dir)}, so +RAG and +RAG+CoT can't look anything up. BASE and +CoT still work.",
        fix=INDEX_BUILD,
        hint="See Build the index in the README, or set RAG_INDEX to an index you already have.",
    )


def index_broken(index_dir, error=None):
    names = _class_names(error)
    if "EmbeddingModelError" in names or _mentions_embedding(error):
        return Issue(
            "embedding_missing",
            "The embedding model isn't on this machine",
            f"The index needs {EMBED_MODEL} from the local Hugging Face cache, and the app never downloads.",
            fix="bash install.sh",
            hint="install.sh caches it (needs the network once). BASE and +CoT still work.",
        )
    return Issue(
        "index_broken",
        "The lab docs index couldn't be opened",
        f"{_short(index_dir)} is there but didn't load. An index built with chromadb 0.6.3 can't be read by 1.0.8.",
        fix=INDEX_BUILD,
        hint="Rebuild it rather than copying an old chroma/ folder. BASE and +CoT still work.",
    )


def retrieval_failed():
    return Issue(
        "retrieval_failed",
        "Looking up the lab docs failed",
        "The index is loaded, but this search raised an error, so the model wasn't asked.",
        hint="Ask again. If it repeats, restart the app; the details are in the terminal running it.",
    )


def issue_for_error(error, *, host, partial_chars=0, index_dir="", model="", create="", retrieving=False):
    """The card for an exception raised while answering a question to `model`, which `create` installs.
    partial_chars is how much answer text had arrived; retrieving is True while the lab docs were being searched."""
    names = _class_names(error)
    message = str(error or "")
    lowered = message.lower()
    got = f" after {partial_chars:,} characters" if partial_chars else ""
    dropped_note = " The partial answer was not kept." if partial_chars else ""
    if "OllamaModelMissing" in names or (isinstance(error, requests.RequestException) and "not found" in lowered
                                         and ("404" in message or "model" in lowered)):
        return model_missing(model or "The model", create or "ollama list")
    if "OllamaTimedOut" in names or _timed_out(error):
        return Issue(
            "timeout",
            "Ollama took too long",
            f"No reply arrived within the time limit{got}.{dropped_note}",
            hint="A slow CPU or a model still loading can do this. Ask again; if it repeats, restart Ollama.",
        )
    if "OllamaDisconnected" in names or (partial_chars and isinstance(error, requests.ConnectionError)) or (
            "ended without a terminal event" in lowered):
        return Issue(
            "dropped",
            "The answer stopped mid-stream",
            f"Ollama closed the connection{got}.{dropped_note}",
            fix="ollama ps",
            hint="Check that Ollama is still running and has the model loaded, then ask again.",
        )
    if "OllamaUnavailable" in names or isinstance(error, requests.ConnectionError):
        return ollama_down(host)
    if names & {"IndexLoadError", "EmbeddingModelError"} or "RAG index is not loaded" in message:
        return index_broken(index_dir, error)
    if "RetrievalQueryError" in names or (retrieving and not isinstance(error, requests.RequestException)):
        return retrieval_failed()
    if isinstance(error, requests.RequestException):
        return Issue(
            "ollama_error",
            "Ollama returned an error",
            _first_line(message) or "It sent a reply the app couldn't read.",
            hint="Ask again. If it repeats, restart Ollama.",
        )
    return Issue(
        "unexpected",
        "Something in the app failed",
        f"{type(error).__name__} while answering.{dropped_note}",
        hint="Ask again. If it repeats, restart the app; the details are in the terminal running it.",
    )


def issue_html(issue, compact=False):
    fix = f'<p class="issue-fix"><code>{escape(issue.fix)}</code></p>' if issue.fix else ""
    hint = f'<p class="issue-hint">{escape(issue.hint)}</p>' if issue.hint else ""
    klass = "issue compact" if compact else "issue"
    return (
        f'<div class="{klass}" role="alert" data-kind="{escape(issue.kind)}">'
        f'<p class="issue-t">{escape(issue.title)}</p>'
        f'<p class="issue-d">{escape(issue.detail)}</p>'
        f"{fix}{hint}</div>"
    )


def _class_names(error):
    return {c.__name__ for c in type(error).__mro__} if error is not None else set()


def _causes(error):
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _timed_out(error):
    for e in _causes(error):
        if isinstance(e, (requests.Timeout, TimeoutError)):
            return True
    return "timed out" in str(error or "").lower()


def _mentions_embedding(error):
    text = " ".join(f"{type(e).__name__} {e}" for e in _causes(error)).lower()
    return any(w in text for w in ("sentence_transformers", "sentence-transformers", "huggingface", "minilm",
                                   "local_files_only", "offline mode"))


def _first_line(text, limit=160):
    line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _short(path):
    shown, home = str(path), str(Path.home())
    if shown == home or shown.startswith(home + os.sep):
        return "~" + shown[len(home):]
    return shown
