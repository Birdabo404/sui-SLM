#!/usr/bin/env python3
"""Run the five thesis configs over a JSONL question file. Temperature 0, fixed seed.

JSONL fields: id, question, answerable, gold_answer, source
  python eval/run_eval.py --questions eval/questions.example.jsonl --out logs/eval.csv
  python eval/run_eval.py --questions eval/questions.jsonl --distilled student:latest
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine import (
    BASE_MODEL,
    DEFAULT_INDEX,
    DISTILLED_MODEL,
    EVAL_SEED,
    THESIS_CONFIGS,
    WEAK_DENSE,
    answer_question,
    config_id,
    git_commit,
    index_ok,
    is_weak,
    model_name_for,
    uses_rag,
)

EVAL_FIELDS = [
    "ts", "git_commit", "config", "model", "mode",
    "question_id", "question", "answerable", "gold_answer", "gold_source",
    "answer", "cited_sources", "n_chunks", "weak_retrieval",
    "latency_s", "app_rss_mb", "ollama_rss_mb", "total_mb", "system_used_mb",
]
SWEEP_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes")


def sweep_abstain_thresholds(questions, retriever, k=5, thresholds=SWEEP_THRESHOLDS):
    """Abstain precision/recall vs gold answerable labels. Retrieval only, no Ollama."""
    scored = []
    for q in questions:
        hits = retriever.retrieve(q["question"], k=k, tldr_lane=True)
        scored.append((as_bool(q["answerable"]), hits))
    rows = []
    n_ans = sum(1 for a, _ in scored if a)
    n_un = sum(1 for a, _ in scored if not a)
    for t in thresholds:
        tp = fp = fn = 0
        for answerable, hits in scored:
            abstain = is_weak(hits, threshold=t)
            if abstain and not answerable:
                tp += 1
            elif abstain and answerable:
                fp += 1
            elif (not abstain) and (not answerable):
                fn += 1
        prec = (tp / (tp + fp)) if (tp + fp) else None
        rec = (tp / (tp + fn)) if (tp + fn) else None
        rows.append({
            "threshold": t,
            "default": int(t == WEAK_DENSE),
            "n_answerable": n_ans,
            "n_unanswerable": n_un,
            "abstain_on_answerable": fp,
            "abstain_on_unanswerable": tp,
            "missed_unanswerable": fn,
            "abstain_precision": "" if prec is None else f"{prec:.3f}",
            "abstain_recall": "" if rec is None else f"{rec:.3f}",
        })
    return rows


def print_sweep(rows):
    print("threshold  default  abstain_prec  abstain_rec  abstain_ans  abstain_unans")
    for r in rows:
        mark = "*" if r["default"] else " "
        print(
            f"{r['threshold']:.2f}{mark:^8}"
            f"{r['abstain_precision']:>12}  {r['abstain_recall']:>11}  "
            f"{r['abstain_on_answerable']:>4}/{r['n_answerable']:<4}  "
            f"{r['abstain_on_unanswerable']:>4}/{r['n_unanswerable']}"
        )


def load_questions(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for key in ("id", "question", "answerable", "gold_answer", "source"):
                if key not in r:
                    raise ValueError(f"{path} is missing field {key}: {line[:120]}")
            rows.append(r)
    return rows


def load_retriever(index_dir):
    # Same lazy load as the app: MiniLM/Chroma only when a RAG config actually runs.
    from rag.query_hybrid import HybridRetriever

    return HybridRetriever(index_dir)


def run_eval(
    questions_path,
    out_path,
    configs=None,
    base_model=None,
    distilled_model=None,
    index_dir=None,
    retriever=None,
    chat_fn=None,
    seed=EVAL_SEED,
    temperature=0,
    k=5,
    sweep=True,
):
    base_model = BASE_MODEL if base_model is None else base_model
    distilled_model = DISTILLED_MODEL if distilled_model is None else distilled_model
    index_dir = DEFAULT_INDEX if index_dir is None else index_dir
    configs = list(THESIS_CONFIGS if configs is None else configs)
    questions = load_questions(questions_path)
    commit = git_commit()

    need_rag = any(uses_rag(mode) for _, mode in configs)
    if need_rag and retriever is None:
        if not index_ok(index_dir):
            raise SystemExit(f"No RAG index at {index_dir}")
        retriever = load_retriever(index_dir)

    n = 0
    skipped = []
    for model_key, mode in configs:
        model = model_name_for(model_key, base_model=base_model, distilled_model=distilled_model)
        cid = config_id(model_key, mode)
        if not model:
            skipped.append(cid)
            print(f"skip {cid}: distilled model name is empty (set --distilled or OLLAMA_DISTILLED_MODEL)")
            continue
        rag = retriever if uses_rag(mode) else None
        for q in questions:
            extra = {
                "git_commit": commit,
                "question_id": q["id"],
                "answerable": q["answerable"],
                "gold_answer": q["gold_answer"],
                "gold_source": q["source"],
            }
            answer_question(
                q["question"],
                mode,
                model,
                retriever=rag,
                chat_fn=chat_fn,
                k=k,
                log_path=out_path,
                model_key=model_key,
                temperature=temperature,
                seed=seed,
                extra_log=extra,
                log_fields=EVAL_FIELDS,
            )
            n += 1
            print(f"{cid}\t{q['id']}\t{n}")
    print(f"wrote {n} rows -> {out_path}  git {commit or '?'}")
    sweep_rows = []
    if sweep and retriever is not None:
        sweep_rows = sweep_abstain_thresholds(questions, retriever, k=k)
        print_sweep(sweep_rows)
        sweep_path = Path(out_path).with_suffix(".sweep.csv")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()) if sweep_rows else [])
            if sweep_rows:
                w.writeheader()
                w.writerows(sweep_rows)
        print(f"wrote threshold sweep -> {sweep_path}")
    return n, skipped, sweep_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval.csv"))
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--distilled", default=DISTILLED_MODEL, help="Ollama name of the distilled model")
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--configs",
        default="",
        help="comma-separated config ids (default: all five thesis configs)",
    )
    ap.add_argument("--no-sweep", action="store_true", help="skip the abstain threshold sweep")
    a = ap.parse_args()
    configs = THESIS_CONFIGS
    if a.configs.strip():
        wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
        by_id = {config_id(mk, mode): (mk, mode) for mk, mode in THESIS_CONFIGS}
        extra = {
            config_id("distilled", "base"): ("distilled", "base"),
            config_id("distilled", "cot"): ("distilled", "cot"),
            config_id("distilled", "rag"): ("distilled", "rag"),
        }
        by_id.update(extra)
        configs = []
        for cid in wanted:
            if cid not in by_id:
                raise SystemExit(f"unknown config {cid}; known: {', '.join(by_id)}")
            configs.append(by_id[cid])
    run_eval(
        a.questions,
        a.out,
        configs=configs,
        base_model=a.base_model,
        distilled_model=a.distilled,
        index_dir=a.index,
        seed=a.seed,
        temperature=a.temperature,
        k=a.k,
        sweep=not a.no_sweep,
    )


if __name__ == "__main__":
    main()
