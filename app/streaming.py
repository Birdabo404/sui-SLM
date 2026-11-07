"""App-only Ollama streaming and the frontend event contract.

`StreamingAnswer` is an iterator of text deltas suitable for
`st.write_stream`. Retrieval and progress callbacks expose chart-ready data,
while `result` contains Ollama's authoritative terminal timing and token
counts after the iterator is exhausted.
"""
import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import requests

from app.engine import (
    K,
    OLLAMA_HOST,
    WEAK_DENSE,
    OllamaChatResult,
    OllamaInfo,
    RamSampler,
    _finalize_answer,
    build_messages,
    is_weak,
    ollama_chat_body,
    parse_ollama_chat_response,
    uses_rag,
)
from shared.prompt import ABSTAIN


class ChatCancelled(Exception):
    """The user stopped the answer before Ollama committed a result."""


class OllamaStreamError(requests.RequestException):
    """Ollama returned malformed or incomplete streaming data."""


@dataclass(frozen=True)
class LiveThroughput:
    """A transport-level live estimate.

    Ollama chooses NDJSON chunk boundaries, so chunks and characters are not
    model tokens. Terminal `eval_count / eval_duration` remains authoritative.
    """

    elapsed_s: float
    chunk_count: int
    character_count: int
    chunks_per_s: float
    characters_per_s: float


@dataclass(frozen=True)
class RetrievalScore:
    """Raw per-hit scores for a retrieval chart; no cross-metric normalization."""

    rank: int
    label: str
    path: str
    dense_cosine: float | None
    rrf_rank_score: float | None
    bm25_raw: float | None


@dataclass(frozen=True)
class RetrievalChart:
    rows: tuple[RetrievalScore, ...]
    dense_threshold: float
    weak: bool
    notes: tuple[str, ...] = (
        "dense_cosine is the MiniLM cosine used by the abstain threshold",
        "rrf_rank_score is a reciprocal-rank fusion score, not confidence",
        "bm25_raw is query-relative and unbounded; it is not comparable to cosine or RRF",
    )


def _optional_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def retrieval_chart(hits, threshold, weak=False):
    rows = []
    for rank, hit in enumerate(hits or (), 1):
        label = str(hit.get("title") or hit.get("path") or hit.get("source") or f"hit {rank}")
        rows.append(
            RetrievalScore(
                rank=rank,
                label=label,
                path=str(hit.get("path") or ""),
                dense_cosine=_optional_float(hit.get("dense")),
                rrf_rank_score=_optional_float(hit.get("rrf")),
                bm25_raw=_optional_float(hit.get("bm25")),
            )
        )
    return RetrievalChart(tuple(rows), float(threshold), bool(weak))


def _terminal_result(text, payload):
    def count(name):
        value = payload.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise OllamaStreamError(f"Ollama stream field {name!r} must be an integer")
        return value

    return OllamaChatResult(
        text=text,
        eval_count=count("eval_count"),
        eval_duration=count("eval_duration"),
        prompt_eval_count=count("prompt_eval_count"),
        prompt_eval_duration=count("prompt_eval_duration"),
        load_duration=count("load_duration"),
    )


class CancellableChat:
    """Cancellable nonstreaming and streaming `/api/chat` transport."""

    def __init__(self, host=None, timeout=300, temperature=0.2, seed=None, keep_alive=None):
        self._url = urlsplit((host or OLLAMA_HOST).rstrip("/") + "/api/chat")
        self._timeout = timeout
        self._temperature = temperature
        self._seed = seed
        self._keep_alive = keep_alive
        self._lock = threading.Lock()
        self._conn = None
        self._committed = False
        self.cancelled = False

    def configure(self, *, temperature, seed, keep_alive):
        with self._lock:
            self._temperature = temperature
            self._seed = seed
            self._keep_alive = keep_alive

    def _body(self, messages, model, *, stream):
        return json.dumps(
            ollama_chat_body(
                messages,
                model,
                temperature=self._temperature,
                seed=self._seed,
                keep_alive=self._keep_alive,
                stream=stream,
            )
        ).encode("utf-8")

    @contextmanager
    def _response(self, body):
        self.raise_if_cancelled()
        conn = http.client.HTTPConnection(self._url.hostname, self._url.port or 80, timeout=self._timeout)
        conn.auto_open = 0
        with self._lock:
            if self.cancelled:
                conn.close()
                raise ChatCancelled()
            self._conn = conn
        try:
            conn.connect()
        except (OSError, http.client.HTTPException) as e:
            with self._lock:
                if self._conn is conn:
                    self._conn = None
            conn.close()
            if self.cancelled:
                raise ChatCancelled() from e
            raise requests.ConnectionError(str(e)) from e
        with self._lock:
            if self.cancelled:
                self._conn = None
                conn.close()
                raise ChatCancelled()
        try:
            try:
                conn.request("POST", self._url.path, body=body, headers={"Content-Type": "application/json"})
                response = conn.getresponse()
            except (OSError, ValueError, http.client.HTTPException) as e:
                if self.cancelled:
                    raise ChatCancelled() from e
                raise requests.ConnectionError(str(e)) from e
            yield response
        finally:
            with self._lock:
                if self._conn is conn:
                    self._conn = None
            conn.close()
        self.raise_if_cancelled()

    def _raise_for_status(self, response):
        if response.status < 400:
            return
        try:
            detail = response.read(4096).decode("utf-8", errors="replace").strip()
        except (OSError, ValueError, http.client.HTTPException) as e:
            if self.cancelled:
                raise ChatCancelled() from e
            raise requests.ConnectionError(str(e)) from e
        suffix = f": {detail}" if detail else ""
        raise requests.HTTPError(f"Ollama returned HTTP {response.status}{suffix}")

    def __call__(self, messages, model):
        with self._response(self._body(messages, model, stream=False)) as response:
            self._raise_for_status(response)
            try:
                data = response.read()
            except (OSError, ValueError, http.client.HTTPException) as e:
                if self.cancelled:
                    raise ChatCancelled() from e
                raise requests.ConnectionError(str(e)) from e
        return parse_ollama_chat_response(json.loads(data))

    def stream(self, messages, model):
        """Yield text deltas and return one terminal `OllamaChatResult`."""
        pieces = []
        with self._response(self._body(messages, model, stream=True)) as response:
            self._raise_for_status(response)
            while True:
                try:
                    raw = response.readline()
                except (OSError, ValueError, http.client.HTTPException) as e:
                    if self.cancelled:
                        raise ChatCancelled() from e
                    raise requests.ConnectionError(str(e)) from e
                if not raw:
                    break
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise OllamaStreamError("Ollama returned invalid NDJSON") from e
                if not isinstance(payload, dict):
                    raise OllamaStreamError("Ollama stream event must be a JSON object")
                if payload.get("error"):
                    raise OllamaStreamError(str(payload["error"]))
                message = payload.get("message")
                if message is not None:
                    if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                        raise OllamaStreamError("Ollama stream message content must be text")
                    content = message.get("content", "")
                    if content:
                        pieces.append(content)
                        self.raise_if_cancelled()
                        yield content
                done = payload.get("done")
                if done is True:
                    return _terminal_result("".join(pieces), payload)
                if done is not False and done is not None:
                    raise OllamaStreamError("Ollama stream field 'done' must be boolean")
        raise OllamaStreamError("Ollama stream ended without a terminal event")

    def raise_if_cancelled(self):
        if self.cancelled:
            raise ChatCancelled()

    @contextmanager
    def commit(self):
        with self._lock:
            if self.cancelled:
                raise ChatCancelled()
            if self._committed:
                raise RuntimeError("chat result already committed")
            self._committed = True
        yield

    def cancel(self):
        with self._lock:
            if self._committed:
                return
            self.cancelled = True
            conn = self._conn
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if conn is not None:
            conn.close()


class StreamingAnswer:
    """Single-use text iterator and terminal result for the Streamlit app."""

    def __init__(
        self,
        question,
        mode,
        model,
        *,
        retriever=None,
        chat=None,
        k=None,
        threshold=None,
        log_path=None,
        model_key="base",
        temperature=0.2,
        seed=None,
        extra_log=None,
        log_fields=None,
        keep_alive=None,
        git_sha=None,
        model_info=None,
        context=None,
        on_retrieval: Callable[[RetrievalChart], None] | None = None,
        on_progress: Callable[[LiveThroughput], None] | None = None,
    ):
        self.question = question
        self.mode = mode
        self.model = model
        self.retriever = retriever
        self.chat = chat or CancellableChat()
        self.k = K if k is None else k
        self.threshold = WEAK_DENSE if threshold is None else threshold
        self.log_path = log_path
        self.model_key = model_key
        self.temperature = temperature
        self.seed = seed
        self.extra_log = extra_log
        self.log_fields = log_fields
        self.keep_alive = keep_alive
        self.git_sha = git_sha
        self.model_info = model_info
        self.context = context
        self.on_retrieval = on_retrieval
        self.on_progress = on_progress
        self.retrieval = None
        self._result = None
        self._iterator = None

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("streaming answer has not completed")
        return self._result

    def cancel(self):
        self.chat.cancel()

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._iterate()
        return next(self._iterator)

    def close(self):
        self.chat.cancel()
        if self._iterator is not None:
            self._iterator.close()

    def _iterate(self):
        self.chat.configure(
            temperature=self.temperature,
            seed=self.seed,
            keep_alive=self.keep_alive,
        )
        started_at = time.perf_counter()
        sampler = RamSampler()
        sampler.start()
        sampler_stopped = False
        transport = None
        complete = False
        hits = []
        weak = False
        context_turns = 0
        try:
            self.chat.raise_if_cancelled()
            if uses_rag(self.mode):
                if self.retriever is None:
                    raise RuntimeError("RAG index is not loaded")
                query = self.question if self.context is None else self.context.retrieval_query
                hits = self.retriever.retrieve(query, k=self.k, tldr_lane=True)
                weak = is_weak(hits, threshold=self.threshold)
                self.chat.raise_if_cancelled()
            messages = build_messages(self.question, hits, self.mode)
            if self.context is not None:
                messages, context_turns = self.context.apply_with_count(messages)
            self.retrieval = retrieval_chart(hits, self.threshold, weak)
            self.chat.raise_if_cancelled()
            if self.on_retrieval is not None:
                self.on_retrieval(self.retrieval)
                self.chat.raise_if_cancelled()

            if weak:
                self.chat.raise_if_cancelled()
                ollama_result = OllamaChatResult(text=ABSTAIN)
                yield ABSTAIN
                self.chat.raise_if_cancelled()
            else:
                transport = self.chat.stream(messages, self.model)
                generation_started = time.perf_counter()
                chunk_count = 0
                character_count = 0
                while True:
                    try:
                        delta = next(transport)
                    except StopIteration as stopped:
                        ollama_result = stopped.value
                        transport = None
                        break
                    self.chat.raise_if_cancelled()
                    chunk_count += 1
                    character_count += len(delta)
                    elapsed = max(time.perf_counter() - generation_started, 1e-9)
                    if self.on_progress is not None:
                        self.on_progress(
                            LiveThroughput(
                                elapsed_s=elapsed,
                                chunk_count=chunk_count,
                                character_count=character_count,
                                chunks_per_s=chunk_count / elapsed,
                                characters_per_s=character_count / elapsed,
                            )
                        )
                        self.chat.raise_if_cancelled()
                    self.chat.raise_if_cancelled()
                    yield delta

            ram = sampler.stop()
            sampler_stopped = True
            self._result = _finalize_answer(
                ollama_result,
                question=self.question,
                mode=self.mode,
                model=self.model,
                model_key=self.model_key,
                hits=hits,
                weak=weak,
                context_turns=context_turns,
                started_at=started_at,
                ram=ram,
                threshold=self.threshold,
                k=self.k,
                log_path=self.log_path,
                extra_log=self.extra_log,
                log_fields=self.log_fields,
                log_commit=self.chat.commit,
                git_sha=self.git_sha,
                model_info=self.model_info or OllamaInfo(),
            )
            complete = True
        finally:
            if not complete:
                self.chat.cancel()
                if transport is not None:
                    transport.close()
            if not sampler_stopped:
                sampler.stop()


def stream_answer_question(question, mode, model, **kwargs):
    """Return a `st.write_stream`-compatible app answer iterator."""
    return StreamingAnswer(question, mode, model, **kwargs)
