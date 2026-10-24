"""Chat turn logic used by the Streamlit app and eval. Ollama is localhost only."""
import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

from shared.frozen import format_threshold, load_frozen
from shared.prompt import ABSTAIN, MODES, system_for_mode, user_msg

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
DISTILLED_MODEL = os.environ.get("OLLAMA_DISTILLED_MODEL", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_INDEX = os.environ.get("RAG_INDEX", str(ROOT / "rag" / "rag-index-v2"))
LOG_CSV = Path(os.environ.get("LOG_CSV", str(ROOT / "logs" / "answers.csv")))
_FROZEN = load_frozen()
K = _FROZEN["k"]
EVAL_SEED = 42
NUM_CTX = 4096
# MiniLM cosine: abstain if none of the top-k fused hits reach this.
# RRF is a rank score, not relevance (any rank-0 hit scores 1/60).
# 0.30 and k=5 in shared/frozen.json are provisional defaults, not a
# tuned result. Replace them after --split dev on the real index and
# commit the sweep CSV.
WEAK_DENSE = _FROZEN["threshold"]
RSS_SAMPLE_S = 0.25
MODE_LABELS = {
    "base": "BASE",
    "cot": "COT",
    "rag": "RAG",
    "full": "FULL",
}
MODE_SUBS = {
    "base": "MODEL ONLY",
    "cot": "STEP-BY-STEP",
    "rag": "RETRIEVAL",
    "full": "RAG + COT",
}
MODE_SUFFIX = {
    "base": "",
    "cot": "+CoT",
    "rag": "+RAG",
    "full": "+RAG+CoT",
}
MODEL_KEYS = ("base", "distilled")
# Five thesis configs: base model in four prompt/retrieval settings, plus distilled with RAG+CoT.
THESIS_CONFIGS = (
    ("base", "base"),
    ("base", "cot"),
    ("base", "rag"),
    ("base", "full"),
    ("distilled", "full"),
)


def mode_choices():
    """UI mode keys, in THESIS_CONFIGS order. A new mode in both maps is a new entry."""
    seen = []
    for _model_key, mode in THESIS_CONFIGS:
        if mode not in MODE_LABELS or mode in seen:
            continue
        seen.append(mode)
    return seen
CSV_FIELDS = [
    "ts", "git_commit", "config", "model", "mode", "latency_s",
    "app_rss_mb", "ollama_rss_mb", "total_mb", "system_used_mb",
    "n_chunks", "weak_retrieval", "threshold", "k",
    "tok_s", "gen_s", "prompt_s", "load_s", "ollama_version", "model_digest",
    "question",
]


@dataclass(frozen=True)
class OllamaChatResult:
    text: str
    eval_count: int | None = None
    eval_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    load_duration: int | None = None


@dataclass(frozen=True)
class OllamaInfo:
    version: str = ""
    model_digest: str = ""


def uses_rag(mode):
    return mode in ("rag", "full")


def is_weak(hits, threshold=None):
    threshold = WEAK_DENSE if threshold is None else threshold
    if not hits:
        return True
    sims = [float(h["dense"]) for h in hits if h.get("dense") is not None]
    if len(sims) == 0:
        return True
    return max(sims) < threshold


def index_ok(path):
    p = Path(path)
    return (p / "chunks.jsonl").is_file() and (p / "chroma").is_dir()


def config_id(model_key, mode):
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model key: {model_key}")
    if mode not in MODE_SUFFIX:
        raise ValueError(f"unknown mode: {mode}")
    suffix = MODE_SUFFIX[mode]
    if model_key == "base" and mode == "base":
        return "base"
    return f"{model_key}{suffix}"


def model_name_for(model_key, base_model=None, distilled_model=None):
    base_model = BASE_MODEL if base_model is None else base_model
    distilled_model = DISTILLED_MODEL if distilled_model is None else distilled_model
    if model_key == "base":
        return base_model
    if model_key == "distilled":
        return (distilled_model or "").strip()
    raise ValueError(f"unknown model key: {model_key}")


def git_commit(repo=None):
    repo = ROOT if repo is None else repo
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def is_ollama_process(name, cmdline):
    """True for the ollama binary and its runner processes, not a random cmdline substring."""
    args = list(cmdline or [])
    exe = Path(args[0]).name.lower() if args else (name or "").lower()
    if exe in ("ollama", "ollama.exe"):
        return True
    return exe.startswith("ollama_")


def ollama_rss_bytes():
    total = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if is_ollama_process(p.info.get("name"), p.info.get("cmdline")):
                total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


def format_cited_sources(hits):
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] {h.get('title', '')} ({h.get('path', '')})")
    return "; ".join(parts)


def build_messages(question, hits, mode):
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    system = system_for_mode(mode)
    if mode == "base":
        user = f"Question: {question}"
    elif mode == "cot":
        user = (
            f"Question: {question}\n\n"
            "Write 2 to 5 short reasoning steps, then the answer."
        )
    elif mode == "rag":
        user = user_msg(question, hits)
        user += "\n\nGive a short answer that cites the excerpt you used. Do not write a step list."
    elif mode == "full":
        user = user_msg(question, hits)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def keep_alive_setting(environ=None):
    environ = os.environ if environ is None else environ
    return environ.get("OLLAMA_KEEP_ALIVE") or DEFAULT_KEEP_ALIVE


def ollama_chat_body(messages, model, temperature=0.2, seed=None, keep_alive=None):
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": options,
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    return body


def parse_ollama_chat_response(payload):
    message = payload["message"]
    return OllamaChatResult(
        text=message["content"],
        eval_count=payload.get("eval_count"),
        eval_duration=payload.get("eval_duration"),
        prompt_eval_count=payload.get("prompt_eval_count"),
        prompt_eval_duration=payload.get("prompt_eval_duration"),
        load_duration=payload.get("load_duration"),
    )


def ollama_chat_result(
    messages,
    model,
    host=None,
    temperature=0.2,
    seed=None,
    keep_alive=None,
):
    host = host or OLLAMA_HOST
    url = host.rstrip("/") + "/api/chat"
    body = ollama_chat_body(
        messages,
        model,
        temperature=temperature,
        seed=seed,
        keep_alive=keep_alive,
    )
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    return parse_ollama_chat_response(r.json())


def ollama_chat(messages, model, host=None, temperature=0.2, seed=None, keep_alive=None):
    return ollama_chat_result(
        messages,
        model,
        host=host,
        temperature=temperature,
        seed=seed,
        keep_alive=keep_alive,
    ).text


def warm_model(model, host=None, keep_alive=None):
    try:
        ollama_chat(
            [],
            model,
            host=host,
            keep_alive=keep_alive or keep_alive_setting(),
        )
        return True
    except requests.RequestException:
        return False


def model_digest(model, host=None, timeout=1.5):
    """Exact Ollama model digest, or blank when the model cannot be identified."""
    if not model:
        return ""
    base = (host or OLLAMA_HOST).rstrip("/")
    try:
        r = requests.post(base + "/api/show", json={"name": model}, timeout=timeout)
        if r.ok:
            digest = str((r.json() or {}).get("digest") or "").strip()
            if digest:
                return digest
    except Exception:
        pass
    try:
        r = requests.get(base + "/api/tags", timeout=timeout)
        if not r.ok:
            return ""
        for item in (r.json() or {}).get("models") or []:
            names = {str(item.get(field) or "") for field in ("name", "model")}
            if model not in names and (
                ":" in model or f"{model}:latest" not in names
            ):
                continue
            digest = str(item.get("digest") or "").strip()
            if digest:
                return digest
    except Exception:
        return ""
    return ""


def ollama_version(host=None, timeout=1.5):
    base = (host or OLLAMA_HOST).rstrip("/")
    try:
        r = requests.get(base + "/api/version", timeout=timeout)
        if r.ok:
            return str((r.json() or {}).get("version") or "")
    except requests.RequestException:
        pass
    return ""


def ollama_info(model, host=None, timeout=1.5):
    return OllamaInfo(
        version=ollama_version(host=host, timeout=timeout),
        model_digest=model_digest(model, host=host, timeout=timeout),
    )


def ollama_reachable(host=None, timeout=1.0):
    """True if /api/tags answers. False on any failure; never raises."""
    base = (host or OLLAMA_HOST).rstrip("/")
    try:
        r = requests.get(base + "/api/tags", timeout=timeout)
        return bool(r.ok)
    except Exception:
        return False


class RamSampler:
    """Peak RSS of this process, Ollama serve/runner processes, and system used memory."""

    def __init__(self):
        self.proc = psutil.Process()
        self.peak_app = 0
        self.peak_ollama = 0
        self.peak_total = 0
        self.peak_system = 0
        self._stop = threading.Event()
        self._t = None

    def _sample(self):
        app = self.proc.memory_info().rss
        ollama = ollama_rss_bytes()
        system = psutil.virtual_memory().used
        total = app + ollama
        if app > self.peak_app:
            self.peak_app = app
        if ollama > self.peak_ollama:
            self.peak_ollama = ollama
        if total > self.peak_total:
            self.peak_total = total
        if system > self.peak_system:
            self.peak_system = system

    def start(self):
        self.peak_app = self.peak_ollama = self.peak_total = self.peak_system = 0
        self._sample()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(RSS_SAMPLE_S):
            self._sample()

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1)
        self._sample()
        return {
            "app_rss_mb": self.peak_app / (1024 * 1024),
            "ollama_rss_mb": self.peak_ollama / (1024 * 1024),
            "total_mb": self.peak_total / (1024 * 1024),
            "system_used_mb": self.peak_system / (1024 * 1024),
        }


def csv_header(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            return None


def rotate_if_header_mismatch(path, fields):
    dest = Path(path)
    existing = csv_header(dest)
    if existing is None or existing == list(fields):
        return dest, existing is None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated = dest.with_name(f"{dest.stem}.{stamp}{dest.suffix}")
    dest.replace(rotated)
    return dest, True


def log_answer(row, path=None, fields=None):
    dest = Path(path) if path else LOG_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields) if fields else list(CSV_FIELDS)
    dest, new = rotate_if_header_mismatch(dest, fields)
    with open(dest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def _chat_result(value):
    if isinstance(value, OllamaChatResult):
        return value
    if isinstance(value, str):
        return OllamaChatResult(text=value)
    raise TypeError("chat function must return str or OllamaChatResult")


def _seconds(nanoseconds):
    if nanoseconds is None:
        return None
    return nanoseconds / 1_000_000_000


def _tokens_per_second(count, duration):
    if count is None or not duration:
        return None
    return count / duration * 1_000_000_000


def _csv_metric(value):
    return "" if value is None else f"{value:.4f}"


def answer_question(
    question,
    mode,
    model,
    retriever=None,
    chat_fn=None,
    k=None,
    threshold=None,
    log_path=None,
    model_key="base",
    temperature=0.2,
    seed=None,
    extra_log=None,
    log_fields=None,
    log_commit=None,
    keep_alive=None,
    git_sha=None,
    model_info=None,
    context=None,
):
    k = K if k is None else k
    threshold = WEAK_DENSE if threshold is None else threshold
    if chat_fn is None:
        def chat_fn(messages, model_name):
            return ollama_chat_result(
                messages,
                model_name,
                temperature=temperature,
                seed=seed,
                keep_alive=keep_alive,
            )
    else:
        configure = getattr(chat_fn, "configure", None)
        if configure is not None:
            configure(
                temperature=temperature,
                seed=seed,
                keep_alive=keep_alive,
            )

    t0 = time.perf_counter()
    sampler = RamSampler()
    sampler.start()
    hits = []
    weak = False
    result = OllamaChatResult(text="")
    try:
        if uses_rag(mode):
            if retriever is None:
                raise RuntimeError("RAG index is not loaded")
            retrieval_query = question if context is None else context.retrieval_query
            hits = retriever.retrieve(retrieval_query, k=k, tldr_lane=True)
            weak = is_weak(hits, threshold=threshold)
            if weak:
                result = OllamaChatResult(text=ABSTAIN)
            else:
                messages = build_messages(question, hits, mode)
                if context is not None:
                    messages = context.apply(messages)
                result = _chat_result(chat_fn(messages, model))
        else:
            messages = build_messages(question, hits, mode)
            if context is not None:
                messages = context.apply(messages)
            result = _chat_result(chat_fn(messages, model))
    finally:
        ram = sampler.stop()
    text = result.text
    latency = time.perf_counter() - t0
    gen_s = _seconds(result.eval_duration)
    prompt_s = _seconds(result.prompt_eval_duration)
    load_s = _seconds(result.load_duration)
    tok_s = _tokens_per_second(result.eval_count, result.eval_duration)
    model_info = model_info or OllamaInfo()
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit() if git_sha is None else git_sha,
        "config": config_id(model_key, mode),
        "model": model,
        "model_key": model_key,
        "mode": mode,
        "latency_s": f"{latency:.4f}",
        "app_rss_mb": f"{ram['app_rss_mb']:.1f}",
        "ollama_rss_mb": f"{ram['ollama_rss_mb']:.1f}",
        "total_mb": f"{ram['total_mb']:.1f}",
        "system_used_mb": f"{ram['system_used_mb']:.1f}",
        "n_chunks": len(hits),
        "weak_retrieval": int(weak) if uses_rag(mode) else "",
        "threshold": format_threshold(threshold),
        "k": int(k),
        "tok_s": _csv_metric(tok_s),
        "gen_s": _csv_metric(gen_s),
        "prompt_s": _csv_metric(prompt_s),
        "load_s": _csv_metric(load_s),
        "ollama_version": model_info.version,
        "model_digest": model_info.model_digest,
        "question": question.replace("\n", " "),
        "cited_sources": format_cited_sources(hits),
        "answer": text,
    }
    if extra_log:
        row.update(extra_log)
    if log_commit is None:
        log_answer(row, path=log_path, fields=log_fields)
    else:
        with log_commit():
            log_answer(row, path=log_path, fields=log_fields)
    return {
        "text": text,
        "hits": hits,
        "weak": weak,
        "latency_s": latency,
        "config": row["config"],
        "cited_sources": row["cited_sources"],
        "ts": row["ts"],
        "git_commit": row["git_commit"],
        "model": model,
        "model_key": model_key,
        "mode": mode,
        "question": question,
        "threshold": float(threshold),
        "k": int(k),
        "eval_count": result.eval_count,
        "eval_duration": result.eval_duration,
        "prompt_eval_count": result.prompt_eval_count,
        "prompt_eval_duration": result.prompt_eval_duration,
        "load_duration": result.load_duration,
        "tok_s": tok_s,
        "gen_s": gen_s,
        "prompt_s": prompt_s,
        "load_s": load_s,
        "ollama_version": model_info.version,
        "model_digest": model_info.model_digest,
        **ram,
    }
