#!/usr/bin/env python3
"""Run the five thesis configs over a JSONL question file. Temperature 0, fixed seed.

JSONL fields: id, question, answerable, gold_answer, source
Optional field: set (tuning | unanswerable | held-out)

  python eval/run_eval.py --split dev --out logs/eval-dev.csv
  python eval/run_eval.py --split test --no-sweep --questions eval/your-heldout.jsonl --out logs/eval-test.csv --distilled student:latest

The threshold/top-k sweep is allowed on --split dev only. --split test is held out
and refuses unless you pass --no-sweep. --split test also rejects --threshold, -k,
and --frozen overrides. Threshold and k live in shared/frozen.json (provisional
defaults until replaced after a real --split dev sweep).
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
    answer_question,
    config_id,
    git_commit,
    index_ok,
    is_weak,
    model_name_for,
    uses_rag,
)
from shared.frozen import PATH as FROZEN_PATH
from shared.frozen import format_threshold, load_frozen

EVAL_FIELDS = [
    "ts", "git_commit", "split", "threshold", "k", "config", "model", "mode",
    "question_id", "question", "answerable", "gold_answer", "gold_source",
    "answer", "cited_sources", "n_chunks", "weak_retrieval",
    "latency_s", "app_rss_mb", "ollama_rss_mb", "total_mb", "system_used_mb",
]
SWEEP_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SWEEP_KS = (3, 5, 8, 10)
SPLITS = ("dev", "test")
SWEEP_REFUSED_ON_TEST = (
    "Threshold/top-k sweep is only allowed on --split dev. "
    "The test split is held out. Pass --no-sweep to score the test set "
    "with the committed threshold and k from shared/frozen.json."
)
OVERRIDE_REFUSED_ON_TEST = (
    "On --split test, --threshold, -k, and --frozen cannot be overridden. "
    "Use the committed values in shared/frozen.json. Tune on --split dev, "
    "then write the chosen values into that file and commit the sweep CSV."
)


def as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes")


def questions_path_for_split(split, questions=None):
    if questions:
        return str(questions)
    return str(ROOT / "eval" / f"{split}.jsonl")


def refuse_sweep_on_test(split, sweep):
    split = (split or "").strip().lower()
    if split not in SPLITS:
        raise SystemExit(f"--split must be dev or test, not {split!r}")
    if sweep and split == "test":
        raise SystemExit(SWEEP_REFUSED_ON_TEST)
    return split


def refuse_test_overrides(split, k=None, threshold=None, frozen_path=None):
    if (split or "").strip().lower() != "test":
        return
    if k is not None or threshold is not None or frozen_path is not None:
        raise SystemExit(OVERRIDE_REFUSED_ON_TEST)


def sweep_abstain(
    questions,
    retriever,
    ks=None,
    thresholds=None,
    split="dev",
    frozen_threshold=None,
    frozen_k=None,
):
    """Abstain precision/recall vs gold labels. Retrieval only, no Ollama.

    Sweeps both MiniLM cosine threshold and top-k. Call only on the dev split.
    """
    ks = tuple(SWEEP_KS if ks is None else ks)
    thresholds = tuple(SWEEP_THRESHOLDS if thresholds is None else thresholds)
    frozen = load_frozen()
    frozen_threshold = frozen["threshold"] if frozen_threshold is None else frozen_threshold
    frozen_k = frozen["k"] if frozen_k is None else frozen_k
    max_k = max(ks)
    scored = []
    for q in questions:
        hits = retriever.retrieve(q["question"], k=max_k, tldr_lane=True)
        scored.append((as_bool(q["answerable"]), hits))
    rows = []
    n_ans = sum(1 for a, _ in scored if a)
    n_un = sum(1 for a, _ in scored if not a)
    for k in ks:
        for t in thresholds:
            tp = fp = fn = 0
            for answerable, hits in scored:
                abstain = is_weak(hits[:k], threshold=t)
                if abstain and not answerable:
                    tp += 1
                elif abstain and answerable:
                    fp += 1
                elif (not abstain) and (not answerable):
                    fn += 1
            prec = (tp / (tp + fp)) if (tp + fp) else None
            rec = (tp / (tp + fn)) if (tp + fn) else None
            rows.append({
                "split": split,
                "k": int(k),
                "threshold": format_threshold(t),
                "default": int(t == frozen_threshold and k == frozen_k),
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
    print("split  k  threshold  default  abstain_prec  abstain_rec  abstain_ans  abstain_unans")
    for r in rows:
        mark = "*" if r["default"] else " "
        print(
            f"{r['split']:<5} {r['k']:<2} {r['threshold']:>9}{mark:^8}"
            f"{r['abstain_precision']:>12}  {r['abstain_recall']:>11}  "
            f"{r['abstain_on_answerable']:>4}/{r['n_answerable']:<4}  "
            f"{r['abstain_on_unanswerable']:>4}/{r['n_unanswerable']}"
        )


def load_questions(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
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
    k=None,
    threshold=None,
    split=None,
    sweep=True,
    frozen_path=None,
):
    refuse_test_overrides(split, k=k, threshold=threshold, frozen_path=frozen_path)
    split = refuse_sweep_on_test(split, sweep)
    frozen = load_frozen(frozen_path)
    threshold = frozen["threshold"] if threshold is None else threshold
    k = frozen["k"] if k is None else k
    base_model = BASE_MODEL if base_model is None else base_model
    distilled_model = DISTILLED_MODEL if distilled_model is None else distilled_model
    index_dir = DEFAULT_INDEX if index_dir is None else index_dir
    configs = list(THESIS_CONFIGS if configs is None else configs)
    questions = load_questions(questions_path)
    commit = git_commit()
    threshold_s = format_threshold(threshold)
    k_i = int(k)

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
                "split": split,
                "threshold": threshold_s,
                "k": k_i,
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
                k=k_i,
                threshold=threshold,
                log_path=out_path,
                model_key=model_key,
                temperature=temperature,
                seed=seed,
                extra_log=extra,
                log_fields=EVAL_FIELDS,
            )
            n += 1
            print(f"{cid}\t{q['id']}\t{n}")
    print(
        f"wrote {n} rows -> {out_path}  git {commit or '?'}  "
        f"split={split} threshold={threshold_s} k={k_i}"
    )
    sweep_rows = []
    if sweep and retriever is not None:
        sweep_rows = sweep_abstain(
            questions,
            retriever,
            split=split,
            frozen_threshold=threshold,
            frozen_k=k_i,
        )
        print_sweep(sweep_rows)
        sweep_path = Path(out_path).with_suffix(".sweep.csv")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()) if sweep_rows else [])
            if sweep_rows:
                w.writeheader()
                w.writerows(sweep_rows)
        print(f"wrote threshold/top-k sweep -> {sweep_path}")
    return n, skipped, sweep_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        choices=SPLITS,
        required=True,
        help="dev = 24 tuning questions plus unanswerable (sweep allowed); "
             "test = held-out set (sweep and threshold/k/--frozen overrides refused)",
    )
    ap.add_argument(
        "--questions",
        default="",
        help="JSONL path (default: eval/<split>.jsonl)",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--distilled", default=DISTILLED_MODEL, help="Ollama name of the distilled model")
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("-k", type=int, default=None, help="override k; refused on --split test")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override MiniLM cosine threshold; refused on --split test",
    )
    ap.add_argument(
        "--frozen",
        default=None,
        help=f"JSON file with threshold and k (default: {FROZEN_PATH}); refused on --split test",
    )
    ap.add_argument(
        "--configs",
        default="",
        help="comma-separated config ids (default: all five thesis configs)",
    )
    ap.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the abstain threshold/top-k sweep (required for --split test)",
    )
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
    out = a.out.strip() or str(ROOT / "logs" / f"eval-{a.split}.csv")
    run_eval(
        questions_path_for_split(a.split, a.questions.strip() or None),
        out,
        configs=configs,
        base_model=a.base_model,
        distilled_model=a.distilled,
        index_dir=a.index,
        seed=a.seed,
        temperature=a.temperature,
        k=a.k,
        threshold=a.threshold,
        split=a.split,
        sweep=not a.no_sweep,
        frozen_path=a.frozen,
    )


if __name__ == "__main__":
    main()
