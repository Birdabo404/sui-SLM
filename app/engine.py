"""Chat turn logic used by the Streamlit app and eval. Ollama is localhost only."""
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

from shared.frozen import format_threshold, load_frozen
from shared.paths import DEFAULT_INDEX, INDEX_NAME, ROOT
from shared.prompt import ABSTAIN, MODES, system_for_mode, user_msg

BASE_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
DISTILLED_MODEL = os.environ.get("OLLAMA_DISTILLED_MODEL", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
LOG_CSV = Path(os.environ.get("LOG_CSV", str(ROOT / "logs" / "answers.csv")))
_FROZEN = load_frozen()
K = _FROZEN["k"]
EVAL_SEED = 42
# MiniLM cosine: abstain if none of the top-k fused hits reach this.
# RRF is a rank score, not relevance (any rank-0 hit scores 1/60).
# 0.30 and k=5 in shared/frozen.json are provisional defaults, not a
# tuned result. Replace them after --split dev on the real index and
# commit the sweep CSV.
WEAK_DENSE = _FROZEN["threshold"]
RSS_SAMPLE_S = 0.25
MODE_LABELS = {
    "base": "base",
    "cot": "CoT",
    "rag": "RAG",
    "full": "RAG+CoT",
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
CSV_FIELDS = [
    "ts", "git_commit", "config", "model", "mode", "latency_s",
    "app_rss_mb", "ollama_rss_mb", "total_mb", "system_used_mb",
    "n_chunks", "weak_retrieval", "threshold", "k", "question",
]


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


def index_ok(path, allow_ubuntu_server=None):
    """True if the index files exist. Exits if leftover Server-docs chunks are present.

    Set INCLUDE_UBUNTU_SERVER=1 or allow_ubuntu_server=True to opt in.
    """
    p = Path(path)
    chunks = p / "chunks.jsonl"
    if not (chunks.is_file() and (p / "chroma").is_dir()):
        return False
    if allow_ubuntu_server is None:
        allow_ubuntu_server = os.environ.get("INCLUDE_UBUNTU_SERVER", "0") == "1"
    if allow_ubuntu_server:
        return True
    with open(chunks, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "ubuntu-server-docs":
                raise SystemExit(
                    f"{chunks} contains source=ubuntu-server-docs. "
                    "That tree is unlicensed and not in the default corpus. "
                    f"Rebuild with: python rag/build_index.py --out rag/{INDEX_NAME} "
                    "or set INCLUDE_UBUNTU_SERVER=1 to opt in."
                )
    return True


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


def ollama_chat(messages, model, host=None, temperature=0.2, seed=None):
    host = host or OLLAMA_HOST
    url = host.rstrip("/") + "/api/chat"
    options = {"temperature": temperature, "num_ctx": 4096}
    if seed is not None:
        options["seed"] = seed
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": options,
    }
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


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
):
    k = K if k is None else k
    threshold = WEAK_DENSE if threshold is None else threshold
    if chat_fn is None:
        def chat_fn(messages, model_name):
            return ollama_chat(
                messages, model_name, temperature=temperature, seed=seed
            )

    t0 = time.perf_counter()
    sampler = RamSampler()
    sampler.start()
    hits = []
    weak = False
    text = ""
    try:
        if uses_rag(mode):
            if retriever is None:
                raise RuntimeError("RAG index is not loaded")
            hits = retriever.retrieve(question, k=k, tldr_lane=True)
            weak = is_weak(hits, threshold=threshold)
            if weak:
                text = ABSTAIN
            else:
                text = chat_fn(build_messages(question, hits, mode), model)
        else:
            text = chat_fn(build_messages(question, hits, mode), model)
    finally:
        ram = sampler.stop()
    latency = time.perf_counter() - t0
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": config_id(model_key, mode),
        "model": model,
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
        "question": question.replace("\n", " "),
        "cited_sources": format_cited_sources(hits),
        "answer": text,
    }
    if extra_log:
        row.update(extra_log)
    log_answer(row, path=log_path, fields=log_fields)
    return {
        "text": text,
        "hits": hits,
        "weak": weak,
        "latency_s": latency,
        "config": row["config"],
        "cited_sources": row["cited_sources"],
        **ram,
    }
