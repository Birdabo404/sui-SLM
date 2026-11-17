#!/usr/bin/env python3
"""Capture raw app-shaped Ollama streams for student-output diagnosis."""

import argparse
import csv
import html
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.answer_format import app_messages
from app.engine import keep_alive_setting
from app.streaming import _guarded_chat_body


MARKDOWN = re.compile(
    r"(?:^|\n)(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|```|~~~)|`[^`\n]+`",
)
CITATION = re.compile(r"\[(\d{1,2})\]")
SENTENCE_END = re.compile(r"""[.!?](?:["')\]]+)?(?=\s|$)""")
ABSTENTION = re.compile(
    r"\b(?:i do not know|i don't know|cannot answer|can't answer|"
    r"documentation (?:does not|doesn't) cover)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--original-model", default="qwen3.5:2b")
    parser.add_argument("--base-model", default="thesis-base-q4km")
    parser.add_argument("--student-model", default="thesis-student")
    parser.add_argument("--student-intended-model", default="")
    parser.add_argument("--original-label", default="qwen3.5:2b")
    parser.add_argument("--base-label", default="thesis-base-q4km")
    parser.add_argument("--student-label", default="thesis-student")
    parser.add_argument("--student-intended-label", default="student with intended template")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "dev.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "diagnostics")
    parser.add_argument("--prefix", default="student-output-actual")
    parser.add_argument(
        "--scope-note",
        default="Actual thesis model tags on the machine that produced this capture.",
    )
    return parser.parse_args()


def dev_rows(path, limit):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == limit:
                break
    return rows


def excerpt_for(item):
    return {
        "title": item.get("source") or "DEV fixture",
        "path": f"eval/dev.jsonl#{item['id']}",
        "source": "dev",
        "text": item.get("gold_answer")
        or "The supplied development item is intentionally unanswerable.",
    }


def installed_models(host):
    response = requests.get(host.rstrip("/") + "/api/tags", timeout=10)
    response.raise_for_status()
    names = set()
    for item in response.json().get("models") or ():
        name = str(item.get("name") or item.get("model") or "")
        if name:
            names.add(name)
            if name.endswith(":latest"):
                names.add(name.removesuffix(":latest"))
    return names


def sentence_count(text):
    prose = re.sub(r"```.*?```|~~~.*?~~~", "", text or "", flags=re.DOTALL)
    prose = re.sub(r"(?m)^\s*Source:.*$", "", prose)
    return len(SENTENCE_END.findall(prose))


def markdown_present(text):
    return bool(MARKDOWN.search(text or ""))


def abstained(text):
    return bool(ABSTENTION.search(text or ""))


def capture_stream(host, body, identity, raw_file):
    response = requests.post(
        host.rstrip("/") + "/api/chat",
        json=body,
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    content = []
    thinking = []
    terminal = {}
    chunk_count = 0
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        payload = json.loads(raw_line)
        message = payload.get("message") or {}
        piece = message.get("content") or ""
        thought = message.get("thinking") or ""
        content.append(piece)
        thinking.append(thought)
        record = {
            "type": "chunk",
            **identity,
            "chunk_index": chunk_count,
            "message": {
                "content": piece,
                "thinking": thought,
            },
            "done": payload.get("done"),
            "done_reason": payload.get("done_reason"),
            "eval_count": payload.get("eval_count"),
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "raw": payload,
        }
        raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        chunk_count += 1
        if payload.get("done"):
            terminal = payload
    return "".join(content), "".join(thinking), terminal, chunk_count


def safe_cell(text, limit=260):
    compact = " ".join((text or "").split())
    if len(compact) > limit:
        compact = compact[: limit - 1].rstrip() + "…"
    return html.escape(compact).replace("|", "&#124;")


def write_modelfiles(path, models, host):
    sections = []
    env = {**os.environ, "OLLAMA_HOST": host}
    for role, label, model in models:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        sections.extend(
            [
                f"===== {role}: {label} ({model}) =====",
                result.stdout.rstrip(),
                "",
            ]
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def write_summary(path, rows, args, options):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_label"]].append(row)
    lines = [
        "# Raw app-path Ollama stream diagnosis",
        "",
        f"Dataset: first {args.limit} rows of `eval/dev.jsonl` (DEV, not held-out).",
        "",
        args.scope_note,
        "",
        "Each request was built by the app's `_guarded_chat_body(app_messages(...))` "
        "path with `stream:true`, `think:false`, temperature 0.2, context 4096, "
        "prediction cap 512, repeat penalty 1.1, the app keep-alive setting, "
        "and no seed. One synthetic excerpt "
        "containing the DEV gold answer was supplied to keep this a generation/stream "
        "sanity check rather than a retrieval-quality check.",
        "",
        f"App options: `{json.dumps(options, sort_keys=True)}`",
        "",
        "The complete system prompt is saved beside this report. The app sends the "
        "same +RAG+CoT thesis prompt and formatting overlay to every model role. "
        "The intended ChatML template renders `.System` and `.Messages`; a stale "
        "plain `{{ .Prompt }}` template does not explicitly render either field, so "
        "the deployed student's Modelfile still determines whether the model sees it.",
        "The stock Ollama model also displays `{{ .Prompt }}`, but pairs it with the "
        "Qwen3.5 renderer and parser; it is not equivalent to a stale export that has neither.",
        "",
        "## Does Ollama already send one sentence?",
        "",
    ]
    for label, model_rows in grouped.items():
        one = sum(row["one_sentence"] for row in model_rows)
        empty = sum(not row["content"] for row in model_rows)
        lines.append(
            f"- **{label}:** {one}/{len(model_rows)} raw responses were one sentence; "
            f"{empty}/{len(model_rows)} were empty before UI parsing."
        )
    lines.extend([
        "",
        "This statement applies only to the exact model IDs recorded below. A proxy "
        "capture does not establish what an unavailable fine-tuned student emits.",
        "",
        "## Aggregate comparison",
        "",
        "| Model | Natural stop | Token range (median) | Markdown | Any citation | Abstain | Non-empty thinking |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for label, model_rows in grouped.items():
        tokens = [row["token_length"] for row in model_rows]
        lines.append(
            f"| {label} | "
            f"{sum(row['done_reason'] == 'stop' for row in model_rows)}/{len(model_rows)} | "
            f"{min(tokens)}–{max(tokens)} ({statistics.median(tokens):g}) | "
            f"{sum(row['markdown'] for row in model_rows)}/{len(model_rows)} | "
            f"{sum(bool(row['citations']) for row in model_rows)}/{len(model_rows)} | "
            f"{sum(row['abstention'] for row in model_rows)}/{len(model_rows)} | "
            f"{sum(bool(row['thinking']) for row in model_rows)}/{len(model_rows)} |"
        )
    lines.extend([
        "",
        "## Capture the actual ThinkPad tags",
        "",
        "From the repository root on the ThinkPad:",
        "",
        "```bash",
        "python scripts/capture_raw_chat.py",
        "```",
        "",
        "The required defaults are `thesis-base-q4km` and `thesis-student`; "
        "the command fails instead of substituting another model when either is unavailable. "
        "It also captures `qwen3.5:2b` when that optional tag is installed. Raw chunks, "
        "the per-answer table, summary, system prompt, and every `ollama show --modelfile` "
        "result are saved under `docs/diagnostics/student-output-actual-*`.",
        "",
        "## Per-answer stream results",
        "",
        "| Model | DEV | Tokens | Done | Chunks | Markdown | Citations | Abstain | Sentences | Content / thinking chars |",
        "|---|---|---:|---|---:|---|---|---|---:|---:|",
    ])
    for row in rows:
        citations = ", ".join(f"[{value}]" for value in row["citations"]) or "none"
        lines.append(
            f"| {row['model_label']} | {row['dev_index']:02d} `{row['id']}` | "
            f"{row['token_length']} | {row['done_reason']} | {row['chunk_count']} | "
            f"{'yes' if row['markdown'] else 'no'} | {citations} | "
            f"{'yes' if row['abstention'] else 'no'} | {row['sentence_count']} | "
            f"{len(row['content'])} / {len(row['thinking'])} |"
        )
    role_labels = {}
    for row in rows:
        role_labels.setdefault(row["role"], row["model_label"])
    roles = tuple(role_labels)
    lines.extend([
        "",
        "## Side-by-side raw content samples",
        "",
        "| DEV question | " + " | ".join(role_labels[role] for role in roles) + " |",
        "|---|" + "---|" * len(roles),
    ])
    by_question = defaultdict(dict)
    for row in rows:
        by_question[row["id"]][row["role"]] = row
    for item in dev_rows(args.questions, args.limit):
        model_rows = by_question[item["id"]]
        cells = [safe_cell(model_rows[role]["content"]) for role in roles]
        lines.append(
            f"| {model_rows[roles[0]]['dev_index']:02d} "
            f"{html.escape(item['question'])} | "
            + " | ".join(cells)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / f"{args.prefix}-stream-captures.jsonl"
    table_path = args.out / f"{args.prefix}-sanity.csv"
    summary_path = args.out / f"{args.prefix}-summary.md"
    prompt_path = args.out / f"{args.prefix}-system-prompt.txt"
    modelfiles_path = args.out / f"{args.prefix}-modelfiles.txt"
    items = dev_rows(args.questions, args.limit)
    available = installed_models(args.host)
    required = (args.base_model, args.student_model)
    missing = [model for model in required if model not in available]
    if missing:
        raise SystemExit(
            "Required Ollama model tags are missing: "
            + ", ".join(missing)
        )
    models = [("base", args.base_label, args.base_model)]
    if args.student_intended_model:
        if args.student_intended_model not in available:
            raise SystemExit(
                "Requested intended-template model is missing: "
                + args.student_intended_model
            )
        models.append(
            (
                "student_intended",
                args.student_intended_label,
                args.student_intended_model,
            )
        )
    models.append(("student", args.student_label, args.student_model))
    if args.original_model in available:
        models.insert(
            0,
            ("original", args.original_label, args.original_model),
        )
    else:
        print(
            f"skip optional model {args.original_model}: not installed",
            flush=True,
        )
    write_modelfiles(modelfiles_path, models, args.host)
    rows = []
    system_prompt = None
    with open(raw_path, "w", encoding="utf-8") as raw_file:
        raw_file.write(
            json.dumps(
                {
                    "type": "capture_metadata",
                    "dataset": (
                        f"first {len(items)} rows of eval/dev.jsonl "
                        "(DEV, not held-out)"
                    ),
                    "scope_note": args.scope_note,
                }
            )
            + "\n"
        )
        for role, label, model in models:
            for index, item in enumerate(items, 1):
                messages = app_messages(
                    item["question"],
                    [excerpt_for(item)],
                    "full",
                )
                system_prompt = system_prompt or messages[0]["content"]
                body = _guarded_chat_body(
                    messages,
                    model,
                    temperature=0.2,
                    seed=None,
                    keep_alive=keep_alive_setting(),
                    stream=True,
                )
                identity = {
                    "role": role,
                    "model_label": label,
                    "actual_model": model,
                    "dev_index": index,
                    "id": item["id"],
                    "question": item["question"],
                }
                raw_file.write(
                    json.dumps(
                        {
                            "type": "request",
                            **identity,
                            "body": body,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                content, thinking, terminal, chunks = capture_stream(
                    args.host,
                    body,
                    identity,
                    raw_file,
                )
                citations = sorted(set(CITATION.findall(content)), key=int)
                count = sentence_count(content)
                row = {
                    **identity,
                    "token_length": terminal.get("eval_count"),
                    "done_reason": terminal.get("done_reason"),
                    "chunk_count": chunks,
                    "markdown": markdown_present(content),
                    "citations": citations,
                    "abstention": abstained(content),
                    "sentence_count": count,
                    "one_sentence": bool(content.strip()) and count <= 1,
                    "content": content,
                    "thinking": thinking,
                }
                rows.append(row)
                print(
                    f"{label} {index:02d}/{len(items)}: "
                    f"{row['token_length']} tokens, {row['done_reason']}, "
                    f"{chunks} chunks",
                    flush=True,
                )
    prompt_path.write_text((system_prompt or "") + "\n", encoding="utf-8")
    with open(table_path, "w", newline="", encoding="utf-8") as table_file:
        fields = list(rows[0])
        writer = csv.DictWriter(table_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    options = _guarded_chat_body(
        [],
        "model",
        temperature=0.2,
        seed=None,
        stream=True,
    )["options"]
    write_summary(summary_path, rows, args, options)
    for path in (
        raw_path,
        table_path,
        summary_path,
        prompt_path,
        modelfiles_path,
    ):
        print(path, flush=True)


if __name__ == "__main__":
    main()
