"""What went wrong, in words a student can act on: one small card per failure instead of a traceback.

`issue_for_error` is the one place the UI turns an exception into a card. With app/backend.py present, its typed
errors are shown by code with their `user_message` exactly (docs/errors.md); `detail` and the cause never are.
Without it, plain requests and socket errors are classified here and get the same cards under the same codes.
"""
import os
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import requests

try:
    from app import backend
except ImportError:  # branches that predate app/backend.py
    backend = None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_BUILD = "python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index"
INDEX_TRIM_HINT = "Then trim it into rag-index-v2 (README, Build the index), or set RAG_INDEX to an index you have."
COMPLETE = "complete"
ABSTAIN_EMPTY = "abstain_empty_retrieval"
ABSTAIN_WEAK = "abstain_weak_retrieval"

TITLES = {
    "ollama_unavailable": "Ollama isn't running",
    "ollama_model_missing": "{model} isn't installed",
    "embedding_model_missing": "The embedding model isn't on this machine",
    "index_missing": "The lab docs index isn't there",
    "index_corrupt": "The lab docs index couldn't be opened",
    "index_query_failed": "Looking up the lab docs failed",
    "ollama_timeout": "Ollama took too long",
    "ollama_disconnected": "The answer stopped mid-stream",
    "ollama_response_error": "Ollama returned an error",
    "grounding_model_unavailable": "The grounding model isn't on this machine",
}


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


def backend_issue(error, *, model="", create="", partial_chars=0):
    """The card for one of app.backend's typed errors: a title for its code and its user_message as written, with a
    next step added only where the message leaves room for it."""
    code = str(getattr(error, "code", "") or "backend_error")
    fix = hint = ""
    if code == "ollama_unavailable":
        hint = "Or open the app with sui-slm, which starts it."
    elif code == "ollama_model_missing":
        fix, hint = create, _model_hint(create)
    elif code == "index_missing":
        fix, hint = INDEX_BUILD, INDEX_TRIM_HINT
    elif code in ("index_corrupt", "index_query_failed"):
        fix = INDEX_BUILD
        if code == "index_corrupt":
            hint = "chromadb 1.0.8 can't read an index built with 0.6.3; don't copy an old chroma/ folder forward."
    elif code == "ollama_disconnected" and partial_chars:
        hint = f"The partial answer ({partial_chars:,} characters) was discarded."
    title = TITLES.get(code, "The backend couldn't finish this answer").format(model=model or "The model")
    return Issue(code, title, str(error.user_message), fix=fix, hint=hint)


def ollama_down(host):
    if backend is not None:
        return backend_issue(backend.OllamaUnavailable())
    return Issue(
        "ollama_unavailable",
        TITLES["ollama_unavailable"],
        f"Nothing answered at {host}, so no model can reply.",
        fix="ollama serve",
        hint="Start it in a terminal (or open the app with sui-slm, which starts it), then ask again.",
    )


def model_missing(name, create):
    if backend is not None:
        return backend_issue(backend.OllamaModelMissing(), model=name, create=create)
    return Issue(
        "ollama_model_missing",
        TITLES["ollama_model_missing"].format(model=name),
        "Ollama is running but doesn't have this model, so nothing was asked and no other model stood in.",
        fix=create,
        hint=_model_hint(create),
    )


def index_missing(index_dir):
    if backend is not None:
        return backend_issue(backend.IndexMissingError())
    return Issue(
        "index_missing",
        TITLES["index_missing"],
        f"No index at {_short(index_dir)}, so +RAG and +RAG+CoT can't look anything up. BASE and +CoT still work.",
        fix=INDEX_BUILD,
        hint=INDEX_TRIM_HINT,
    )


def load_index(index_dir, factory):
    """The retriever for index_dir, through app.backend's loader when it exists, so a failure arrives typed."""
    if backend is None:
        return factory(index_dir)
    return backend.load_app_retriever(index_dir, factory=factory)


def issue_for_error(error, *, host, stage="answer", partial_chars=0, index_dir="", model="", create=""):
    """The card for an exception raised while loading the index (stage "load"), searching it ("retrieve"),
    answering ("answer") with `model`, which `create` installs, or scoring a finished answer's grounding ("ground").
    partial_chars is how much answer text had arrived; that text is never kept."""
    if backend is not None and isinstance(error, backend.UserFacingBackendError):
        return backend_issue(error, model=model, create=create, partial_chars=partial_chars)
    if stage == "ground":
        return Issue(
            "grounding_failed",
            "The grounding check failed",
            "The answer above is unchanged; only its grounding score is missing.",
            hint="Turn grounding off in Settings, or ask again. The details are in the terminal running the app.",
        )
    message = str(error or "")
    lowered = message.lower()
    if stage == "load":
        return _index_unreadable(index_dir, error)
    got = f" after {partial_chars:,} characters" if partial_chars else ""
    dropped_note = " The partial answer was not kept." if partial_chars else ""
    if isinstance(error, requests.RequestException) and "not found" in lowered and (
            "404" in message or "model" in lowered):
        return model_missing(model or "The model", create or "ollama list")
    if _timed_out(error):
        return Issue(
            "ollama_timeout",
            TITLES["ollama_timeout"],
            f"No reply arrived within the time limit{got}.{dropped_note}",
            hint="A slow CPU or a model still loading can do this. Ask again; if it repeats, restart Ollama.",
        )
    if (partial_chars and isinstance(error, requests.ConnectionError)) or "ended without a terminal event" in lowered:
        return Issue(
            "ollama_disconnected",
            TITLES["ollama_disconnected"],
            f"Ollama closed the connection{got}.{dropped_note}",
            fix="ollama ps",
            hint="Check that Ollama is still running and has the model loaded, then ask again.",
        )
    if isinstance(error, requests.ConnectionError):
        return ollama_down(host)
    if "RAG index is not loaded" in message:
        return _index_unreadable(index_dir, error)
    if stage == "retrieve" and not isinstance(error, requests.RequestException):
        return Issue(
            "index_query_failed",
            TITLES["index_query_failed"],
            "The index is loaded, but this search raised an error, so the model wasn't asked.",
            hint="Ask again. If it repeats, restart the app; the details are in the terminal running it.",
        )
    if isinstance(error, requests.RequestException):
        return Issue(
            "ollama_response_error",
            TITLES["ollama_response_error"],
            _first_line(message) or "It sent a reply the app couldn't read.",
            hint="Ask again. If it repeats, restart Ollama.",
        )
    return Issue(
        "unexpected",
        "Something in the app failed",
        f"{type(error).__name__} while answering.{dropped_note}",
        hint="Ask again. If it repeats, restart the app; the details are in the terminal running it.",
    )


def _index_unreadable(index_dir, error):
    if _mentions_embedding(error):
        return Issue(
            "embedding_model_missing",
            TITLES["embedding_model_missing"],
            f"The index needs {EMBED_MODEL} from the local Hugging Face cache, and the app never downloads.",
            fix="bash install.sh",
            hint="install.sh caches it (needs the network once). BASE and +CoT still work.",
        )
    return Issue(
        "index_corrupt",
        TITLES["index_corrupt"],
        f"{_short(index_dir)} is there but didn't load. An index built with chromadb 0.6.3 can't be read by 1.0.8.",
        fix=INDEX_BUILD,
        hint="Rebuild it rather than copying an old chroma/ folder. BASE and +CoT still work.",
    )


def _model_hint(create):
    if create.startswith("ollama create"):
        return "Run it where the KD export's Modelfile is (kd/README.md), then ask again."
    return "It downloads once; after that it works offline. Then ask again."


# ---------- abstentions: a status, not an error ----------

def status_value(status):
    return str(getattr(status, "value", status) or "")


def answer_status(chart_or_result, hits=None, weak=False):
    """The answer's status: the backend's own (RetrievalChart.status, or result["status"]) when it gives one, else
    the same three values read from the hits and the threshold decision, which stay the app's."""
    given = chart_or_result.get("status") if isinstance(chart_or_result, dict) else getattr(
        chart_or_result, "status", None)
    if given:
        return status_value(given)
    if not weak:
        return COMPLETE
    return ABSTAIN_WEAK if hits else ABSTAIN_EMPTY


def abstain_note(record):
    """The line under an abstention saying why: the backend status's user_message, or the app's own words."""
    status = record.get("status") or (ABSTAIN_WEAK if record.get("hits") else ABSTAIN_EMPTY)
    if status == COMPLETE:
        return ""
    if backend is not None:
        try:
            return backend.AnswerStatus(status).user_message
        except ValueError:
            pass
    if status == ABSTAIN_EMPTY:
        return "Nothing was retrieved for this question, so the model wasn't asked."
    threshold = record.get("threshold")
    gate = f"cosine {threshold:.2f}" if isinstance(threshold, (int, float)) and threshold else "the threshold"
    return f"No retrieved chunk reached {gate}, so the model wasn't asked."


def issue_html(issue, compact=False):
    fix = f'<p class="issue-fix"><code>{escape(issue.fix)}</code></p>' if issue.fix else ""
    hint = f'<p class="issue-hint">{escape(issue.hint)}</p>' if issue.hint else ""
    klass = "issue compact" if compact else "issue"
    return (
        f'<div class="{klass}" role="alert" data-kind="{escape(issue.kind)}">'
        f'<p class="issue-t">{escape(issue.title)}</p>'
        f'<p class="issue-d">{_code_spans(escape(issue.detail))}</p>'
        f"{fix}{hint}</div>"
    )


def _code_spans(text):
    """`command` in a message (the backend writes them that way) as code; the words are unchanged."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


# ---------- the offline start check ----------

@dataclass(frozen=True)
class Check:
    label: str
    ok: bool | None  # None: not needed or can't be known
    value: str
    fix: str = ""


def hf_cached(repo_id=EMBED_MODEL, environ=None):
    """True when a snapshot of repo_id is in the local Hugging Face cache, where sentence-transformers loads it from
    offline. Looks at the folders only; imports and downloads nothing."""
    env = os.environ if environ is None else environ
    hubs = [env.get("SENTENCE_TRANSFORMERS_HOME"),
            env.get("HF_HUB_CACHE") or str(Path(env.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")]
    for hub in filter(None, hubs):
        folder = Path(hub) / ("models--" + repo_id.replace("/", "--")) / "snapshots"
        try:
            if any((snap / "config.json").is_file() for snap in folder.iterdir()):
                return True
        except OSError:
            continue
    return False


def offline_checks(*, host, ollama_up, model, installed, create, index_dir, index_present, chunks, embed_cached,
                   online, extra=()):
    """Everything an answer needs from this machine, in the order it's needed, with `extra` checks before the
    network. installed is None when Ollama can't be asked."""
    checks = [Check("Ollama", ollama_up, f"running at {host}" if ollama_up else f"not running at {host}",
                    "" if ollama_up else "ollama serve")]
    if installed is None:
        checks.append(Check("Model", None, f"{model} · can't check until Ollama runs"))
    else:
        checks.append(Check("Model", installed, f"{model} installed" if installed else f"{model} missing",
                            "" if installed else create))
    checks.append(Check("Lab docs index", index_present,
                        f"{chunks:,} chunks · {_short(index_dir)}" if index_present else f"missing · {_short(index_dir)}",
                        "" if index_present else INDEX_BUILD))
    checks.append(Check("Embedding model", embed_cached, "MiniLM cached locally" if embed_cached else "not cached",
                        "" if embed_cached else "bash install.sh"))
    checks.extend(extra)
    checks.append(Check("Network", None, "online, not used" if online else "offline, not needed"))
    return checks


def checks_summary(checks):
    failing = [c for c in checks if c.ok is False]
    if not failing:
        return "Ready to answer offline"
    names = ", ".join(c.label.lower() for c in failing)
    return f"Needs attention: {names}"


def checks_html(checks):
    rows = []
    for c in checks:
        state = {True: "ok", False: "bad", None: "na"}[c.ok]
        mark = {True: "ok", False: "fix", None: "–"}[c.ok]
        fix = f'<code class="ck-fix">{escape(c.fix)}</code>' if c.fix else ""
        rows.append(f'<li class="{state}"><span class="ck-m">{mark}</span><span class="ck-l">{escape(c.label)}</span>'
                    f'<span class="ck-v">{escape(c.value)}</span>{fix}</li>')
    ready = all(c.ok is not False for c in checks)
    return (f'<p class="ck-sum {"ok" if ready else "bad"}">{escape(checks_summary(checks))}</p>'
            '<ul class="checks">' + "".join(rows) + "</ul>")


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
