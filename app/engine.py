"""Chat turn logic used by the Streamlit app. Ollama is localhost only."""
import csv
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

from shared.prompt import ABSTAIN, MODES, system_for_mode, user_msg

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_INDEX = os.environ.get("RAG_INDEX", str(ROOT / "rag" / "rag-index-v2"))
LOG_CSV = Path(os.environ.get("LOG_CSV", str(ROOT / "logs" / "answers.csv")))
K = 5
# RRF of a rank-0 hit in a single lane is 1/60. Below 1/75 means it was not in the top 15 of any lane.
WEAK_RRF = 1.0 / 75
MODE_LABELS = {
    "base": "base",
    "cot": "CoT only",
    "rag": "RAG only",
    "full": "full",
}


def uses_rag(mode):
    return mode in ("rag", "full")


def is_weak(hits):
    if not hits:
        return True
    return float(hits[0].get("rrf", 0)) < WEAK_RRF


def index_ok(path):
    p = Path(path)
    return (p / "chunks.jsonl").is_file() and (p / "chroma").is_dir()


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
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ollama_chat(messages, model, host=None):
    host = host or OLLAMA_HOST
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


class RamSampler:
    def __init__(self):
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self.peak = self.proc.memory_info().rss
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.05):
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1)
        rss = self.proc.memory_info().rss
        if rss > self.peak:
            self.peak = rss
        return self.peak


def log_answer(row, path=None):
    dest = Path(path) if path else LOG_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ts", "mode", "model", "latency_s", "app_rss_peak_mb",
        "system_used_mb", "n_chunks", "weak_retrieval", "question",
    ]
    new = not dest.exists()
    with open(dest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def answer_question(question, mode, model, retriever=None, chat_fn=None, k=K, log_path=None):
    chat_fn = chat_fn or ollama_chat
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
            weak = is_weak(hits)
            if weak:
                text = ABSTAIN
            else:
                text = chat_fn(build_messages(question, hits, mode), model)
        else:
            text = chat_fn(build_messages(question, hits, mode), model)
    finally:
        peak = sampler.stop()
    latency = time.perf_counter() - t0
    log_answer(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "model": model,
            "latency_s": f"{latency:.4f}",
            "app_rss_peak_mb": f"{peak / (1024 * 1024):.1f}",
            "system_used_mb": f"{psutil.virtual_memory().used / (1024 * 1024):.1f}",
            "n_chunks": len(hits),
            "weak_retrieval": int(weak) if uses_rag(mode) else "",
            "question": question.replace("\n", " "),
        },
        path=log_path,
    )
    return {
        "text": text,
        "hits": hits,
        "weak": weak,
        "latency_s": latency,
        "rss_mb": peak / (1024 * 1024),
    }
