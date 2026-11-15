#!/usr/bin/env python3
"""Compare frozen answers with optional grounding on the DEV questions only.

Retrieval-only (no Ollama):
  python scripts/compare_enhanced.py --retrieval-only

Full app-path comparison with the student model:
  python scripts/compare_enhanced.py --model thesis-student

The script never imports or changes the eval harness. It reads ``eval/dev.jsonl``
and refuses to run if a DEV id or normalized question overlaps any row in the
reported-question files.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend import UserFacingBackendError
from app.engine import (
    DEFAULT_INDEX,
    EVAL_SEED,
    RamSampler,
    build_messages,
    is_weak,
    ollama_chat_result,
)
from app.grounding import score as grounding_score
from rag.query_hybrid import HybridRetriever
from shared.frozen import load_frozen
from shared.prompt import ABSTAIN


DEV_PATH = ROOT / "eval" / "dev.jsonl"
DEFAULT_REPORTED_PATH = ROOT / "eval" / "test.jsonl"
DEFAULT_OUT = ROOT / "logs" / "compare-enhanced-dev.csv"
APP_TEMPERATURE = 0.2
APP_K = 5
APP_THRESHOLD = 0.30
MODES = ("frozen", "grounding")
CSV_FIELDS = (
    "mode",
    "index_snapshot",
    "question_id",
    "question",
    "answerable",
    "gold_source",
    "gold_answer",
    "retrieved_sources",
    "retrieved_titles",
    "retrieval_hit",
    "reciprocal_rank",
    "abstained",
    "answer",
    "answer_correct",
    "grounding_score",
    "latency_s",
    "app_rss_mb",
    "ollama_rss_mb",
    "total_rss_mb",
    "system_used_mb",
)
TOKEN_RE = re.compile(r"[a-z0-9@:/._+*-]+")


def load_jsonl(path):
    """Read non-comment JSONL rows from ``path``."""

    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
    return rows


def normalized_question(value):
    """Normalize a question for conservative cross-split overlap checks."""

    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def validate_dev_rows(rows):
    """Reject anything that is not the committed tuning/unanswerable DEV set."""

    if not rows:
        raise ValueError("The DEV question set is empty.")
    required = {"id", "set", "question", "answerable", "gold_answer", "source"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"DEV row {row.get('id', '<unknown>')} is missing "
                f"{', '.join(sorted(missing))}."
            )
        if row["set"] not in {"tuning", "unanswerable"}:
            raise ValueError(f"DEV row {row['id']} has non-DEV set {row['set']!r}.")
        if row.get("placeholder"):
            raise ValueError(f"DEV row {row['id']} is marked as a placeholder.")


def refuse_reported_overlap(dev_rows, reported_rows):
    """Raise when DEV shares an id or normalized question with reported rows."""

    dev_ids = {str(row["id"]).strip() for row in dev_rows}
    dev_questions = {
        normalized_question(row["question"]): str(row["id"])
        for row in dev_rows
    }
    collisions = []
    for row in reported_rows:
        reported_id = str(row.get("id", "")).strip()
        question = normalized_question(row.get("question", ""))
        if reported_id and reported_id in dev_ids:
            collisions.append(f"id {reported_id}")
        if question and question in dev_questions:
            collisions.append(
                f"question {reported_id or '<unknown>'} matches "
                f"{dev_questions[question]}"
            )
    if collisions:
        detail = "; ".join(sorted(set(collisions)))
        raise ValueError(f"DEV overlaps reported eval/test questions: {detail}")


def check_reported_questions(dev_rows, paths):
    """Load all reported sets and enforce the no-overlap boundary."""

    reported = []
    for path in paths:
        question_path = Path(path)
        if not question_path.is_file():
            raise ValueError(f"Reported question file is missing: {question_path}")
        reported.extend(load_jsonl(question_path))
    if not reported:
        raise ValueError("At least one reported eval/test question is required.")
    refuse_reported_overlap(dev_rows, reported)


def placeholder_reported_paths(paths):
    """Return reported files that contain only placeholder rows."""

    placeholders = []
    for path in paths:
        rows = load_jsonl(path)
        if rows and all(bool(row.get("placeholder")) for row in rows):
            placeholders.append(str(Path(path)))
    return placeholders


def gold_source_matches(gold_source, hit):
    """Return whether a retrieved hit belongs to the DEV gold source."""

    gold = str(gold_source).strip().casefold()
    if not gold:
        return False
    source, _, name = gold.partition(" ")
    hit_source = str(hit.get("source", "")).strip().casefold()
    path = str(hit.get("path", "")).split("#", 1)[0]
    path_stem = Path(path).stem.casefold()
    title = normalized_question(hit.get("title", ""))
    normalized_name = normalized_question(name)
    if source == "tldr":
        return hit_source == "tldr" and (
            path_stem == name
            or title == normalized_name
            or title.startswith(f"{normalized_name} ")
        )
    haystack = normalized_question(
        " ".join((hit_source, path, str(hit.get("title", ""))))
    )
    return normalized_question(gold) in haystack


def reciprocal_rank(hits, gold_source):
    """Return reciprocal rank of the first gold-source hit, or zero."""

    for rank, hit in enumerate(hits, 1):
        if gold_source_matches(gold_source, hit):
            return 1.0 / rank
    return 0.0


def answer_tokens(value):
    """Tokenize commands and prose for the documented answer score."""

    return TOKEN_RE.findall(str(value).casefold())


def contains_token_sequence(tokens, expected):
    if not expected or len(expected) > len(tokens):
        return False
    width = len(expected)
    return any(tokens[index : index + width] == expected for index in range(len(tokens) - width + 1))


def answer_correct(question, answer, abstained):
    """Score answer correctness with normalized gold-command containment.

    The repository's eval runner records gold answers but has no answer-scoring
    function. This comparison therefore uses an explicit local rule: answerable
    rows must contain the normalized gold command token sequence, while
    unanswerable rows are correct only when the app abstains.
    """

    if not bool(question["answerable"]):
        return bool(abstained)
    if abstained:
        return False
    return contains_token_sequence(
        answer_tokens(answer),
        answer_tokens(question["gold_answer"]),
    )


def retrieve(question, retriever):
    """Retrieve with the frozen path and return its abstention decision."""

    hits = tuple(
        retriever.retrieve(
            question,
            k=APP_K,
            tldr_lane=True,
        )
    )
    return hits, is_weak(hits, threshold=APP_THRESHOLD)


def generate_answer(question, hits, weak, model, host):
    """Generate one app-path answer with the unchanged app request settings."""

    if weak:
        return ABSTAIN
    messages = build_messages(question, hits, "full")
    return ollama_chat_result(
        messages,
        model,
        host=host,
        temperature=APP_TEMPERATURE,
        seed=EVAL_SEED,
        keep_alive="30m",
    ).text


def compare_mode(
    mode,
    questions,
    retriever,
    *,
    retrieval_only,
    model,
    host,
    answer_fn=generate_answer,
    grounding_fn=grounding_score,
):
    """Run one comparison mode and return detailed CSV rows."""

    rows = []
    index_metadata = getattr(retriever.col, "metadata", {}) or {}
    index_snapshot = str(index_metadata.get("snapshot", ""))
    for question in questions:
        sampler = RamSampler()
        sampler.start()
        started = time.perf_counter()
        try:
            hits, weak = retrieve(question["question"], retriever)
            answer = ""
            correct = ""
            consistency = ""
            if not retrieval_only:
                answer = answer_fn(
                    question["question"],
                    hits,
                    weak,
                    model,
                    host,
                )
                correct = int(answer_correct(question, answer, weak))
                if mode == "grounding" and not weak:
                    excerpts = tuple(str(hit.get("text", "")) for hit in hits)
                    consistency = grounding_fn(answer, excerpts).score
        finally:
            ram = sampler.stop()
        rank = reciprocal_rank(hits, question["source"])
        rows.append(
            {
                "mode": mode,
                "index_snapshot": index_snapshot,
                "question_id": question["id"],
                "question": question["question"],
                "answerable": int(bool(question["answerable"])),
                "gold_source": question["source"],
                "gold_answer": question["gold_answer"],
                "retrieved_sources": " | ".join(
                    f"{hit.get('source', '')}:{hit.get('path', '')}"
                    for hit in hits
                ),
                "retrieved_titles": " | ".join(
                    str(hit.get("title", "")) for hit in hits
                ),
                "retrieval_hit": (
                    int(rank > 0) if question["source"].strip() else ""
                ),
                "reciprocal_rank": (
                    f"{rank:.6f}" if question["source"].strip() else ""
                ),
                "abstained": int(weak),
                "answer": answer,
                "answer_correct": correct,
                "grounding_score": (
                    f"{float(consistency):.6f}" if consistency != "" else ""
                ),
                "latency_s": f"{time.perf_counter() - started:.6f}",
                "app_rss_mb": f"{ram['app_rss_mb']:.1f}",
                "ollama_rss_mb": f"{ram['ollama_rss_mb']:.1f}",
                "total_rss_mb": f"{ram['total_mb']:.1f}",
                "system_used_mb": f"{ram['system_used_mb']:.1f}",
            }
        )
    return rows


def mode_summary(rows):
    """Aggregate one mode's detailed rows."""

    retrieval_rows = [row for row in rows if row["retrieval_hit"] != ""]
    answer_rows = [row for row in rows if row["answer_correct"] != ""]
    answerable_rows = [
        row
        for row in answer_rows
        if int(row["answerable"]) == 1
    ]
    grounding_values = [
        float(row["grounding_score"])
        for row in rows
        if row["grounding_score"] != ""
    ]
    return {
        "questions": len(rows),
        "hit_at_5": (
            sum(int(row["retrieval_hit"]) for row in retrieval_rows)
            / len(retrieval_rows)
            if retrieval_rows
            else None
        ),
        "mrr": (
            sum(float(row["reciprocal_rank"]) for row in retrieval_rows)
            / len(retrieval_rows)
            if retrieval_rows
            else None
        ),
        "abstain_rate": sum(int(row["abstained"]) for row in rows) / len(rows),
        "answer_accuracy": (
            sum(int(row["answer_correct"]) for row in answer_rows)
            / len(answer_rows)
            if answer_rows
            else None
        ),
        "answerable_accuracy": (
            sum(int(row["answer_correct"]) for row in answerable_rows)
            / len(answerable_rows)
            if answerable_rows
            else None
        ),
        "median_latency_s": statistics.median(
            float(row["latency_s"]) for row in rows
        ),
        "peak_total_rss_mb": max(float(row["total_rss_mb"]) for row in rows),
        "mean_grounding_score": (
            statistics.fmean(grounding_values) if grounding_values else None
        ),
    }


def format_metric(value):
    return "not run" if value is None else f"{value:.3f}"


def write_summary(
    path,
    rows,
    *,
    retrieval_only,
    reported_paths,
    index_path="",
):
    """Write the compact Markdown evidence summary."""

    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)
    lines = [
        "# Grounding DEV comparison",
        "",
        f"- Questions: `{DEV_PATH.relative_to(ROOT)}` only.",
        f"- Index: `{index_path or '<injected test index>'}`.",
        f"- Index snapshot metadata: `{rows[0].get('index_snapshot', '') or 'unknown'}`.",
        "- Retrieval metrics use answerable rows with non-empty gold sources.",
        (
            "- Answer score: not run (`--retrieval-only`)."
            if retrieval_only
            else (
                "- Answer score: normalized gold-command token containment; "
                "unanswerable rows are correct only on abstention. "
                "`eval/run_eval.py` has no answer-scoring function to reuse."
            )
        ),
        "- Overlap check passed against: "
        + ", ".join(str(Path(item)) for item in reported_paths)
        + ".",
    ]
    placeholder_paths = placeholder_reported_paths(reported_paths)
    if placeholder_paths:
        lines.append(
            "- Warning: the overlap input contains placeholder questions only: "
            + ", ".join(placeholder_paths)
            + ". Rerun with the real reported test JSONL before relying on "
            "split isolation."
        )
    lines.extend(
        (
            "",
            "| Mode | hit@5 | MRR | answer accuracy | answerable accuracy | "
            "abstain rate | median latency (s) | peak app+Ollama RSS (MB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for mode in MODES:
        if mode not in grouped:
            continue
        summary = mode_summary(grouped[mode])
        lines.append(
            f"| {mode} | {format_metric(summary['hit_at_5'])} | "
            f"{format_metric(summary['mrr'])} | "
            f"{format_metric(summary['answer_accuracy'])} | "
            f"{format_metric(summary['answerable_accuracy'])} | "
            f"{format_metric(summary['abstain_rate'])} | "
            f"{summary['median_latency_s']:.3f} | "
            f"{summary['peak_total_rss_mb']:.1f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare frozen answers with optional grounding on DEV only.",
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument(
        "--reported-questions",
        action="append",
        default=[],
        help=(
            "Reported eval/test JSONL used for the overlap guard. Repeat for "
            "multiple files. Defaults to eval/test.jsonl."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_DISTILLED_MODEL") or "thesis-student",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frozen = load_frozen()
    if int(frozen["k"]) != APP_K or float(frozen["threshold"]) != APP_THRESHOLD:
        raise SystemExit(
            "The comparison is locked to the evaluated k=5 and threshold=0.30."
        )
    questions = load_jsonl(DEV_PATH)
    validate_dev_rows(questions)
    reported_paths = args.reported_questions or [str(DEFAULT_REPORTED_PATH)]
    try:
        check_reported_questions(questions, reported_paths)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    placeholder_paths = placeholder_reported_paths(reported_paths)
    if placeholder_paths:
        print(
            "Warning: reported overlap input contains placeholder questions; "
            "pass the real reported test JSONL before relying on split isolation.",
            file=sys.stderr,
        )
    index_path = Path(args.index)
    if not (index_path / "chunks.jsonl").is_file() or not (
        index_path / "chroma"
    ).is_dir():
        raise SystemExit(f"No RAG index at {index_path}")

    retriever = HybridRetriever(index_path)
    modes = MODES[:1] if args.retrieval_only else MODES
    rows = []
    try:
        for mode in modes:
            rows.extend(
                compare_mode(
                    mode,
                    questions,
                    retriever,
                    retrieval_only=args.retrieval_only,
                    model=args.model,
                    host=args.ollama_host,
                )
            )
    except UserFacingBackendError as error:
        raise SystemExit(error.user_message) from None
    except requests.RequestException as error:
        raise SystemExit(
            "Ollama could not complete the comparison. Start Ollama, install "
            f"the {args.model!r} model, and try again. ({type(error).__name__})"
        ) from None

    out_path = Path(args.out)
    write_csv(out_path, rows)
    summary_path = out_path.with_suffix(".md")
    write_summary(
        summary_path,
        rows,
        retrieval_only=args.retrieval_only,
        reported_paths=reported_paths,
        index_path=index_path,
    )
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
