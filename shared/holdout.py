"""Held-out question and gold_source helpers for seed and SFT leakage checks."""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_holdout(path):
    """Return (questions, gold_sources) from JSONL or a one-question-per-line file."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Holdout file is missing: {path}")
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    questions = []
    gold_sources = []
    if lines and lines[0].lstrip().startswith("{"):
        for ln in lines:
            row = json.loads(ln)
            q = (row.get("question") or "").strip()
            if q:
                questions.append(q)
            src = (row.get("gold_source") or row.get("source") or "").strip()
            if src:
                gold_sources.append(src)
    else:
        questions = [ln.strip() for ln in lines]
    if not questions:
        raise SystemExit(f"No questions in {path}")
    return questions, gold_sources


def gold_source_stem(label):
    """Command stem from a gold_source label or a chunk path.

    ssh(1), tldr ls, man ls, tldr/common/ls.md, command-manpages/ls.1,
    and command-manpages/coreutils/ls.1.gz all become ls. Matching is
    stem equality, not a substring. .gz is stripped before the trailing
    man section so gzipped pages do not keep a leftover .1.
    """
    raw = str(label or "").strip().lower().replace("\\", "/")
    if not raw:
        return ""
    raw = re.sub(r"\([^)]*\)", "", raw)
    token = [t for t in re.split(r"[\s/]+", raw) if t]
    if not token:
        return ""
    name = token[-1]
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    name = re.sub(r"\.\d+[a-z]*$", "", name)
    stem = Path(name).stem
    stem = re.sub(r"^\d+\.", "", stem)
    return stem


def path_matches_gold_source(path, gold_sources):
    """True if the chunk is a page for a held-out command, in any source."""
    path_stem = gold_source_stem(path)
    if not path_stem:
        return False
    held = {gold_source_stem(raw) for raw in gold_sources}
    held.discard("")
    return path_stem in held
