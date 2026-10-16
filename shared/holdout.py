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


def path_matches_gold_source(path, gold_sources):
    """True if a chunk path matches a held-out gold_source.

    gold_source may be a path (tldr/common/ls.md) or a short label (tldr ls).
    A label's last token is matched to the file stem. Any earlier token must
    appear in the path so "tldr ls" does not drop command-manpages/ls.1.
    """
    path_l = str(path or "").replace("\\", "/").lower().strip()
    if not path_l:
        return False
    stem = Path(path_l).stem.lower()
    for raw in gold_sources:
        g = str(raw or "").strip().lower().replace("\\", "/")
        if not g:
            continue
        if g in path_l or path_l in g:
            return True
        tokens = [t for t in re.split(r"[\s/]+", g) if t]
        if not tokens:
            continue
        last = Path(tokens[-1]).stem
        if last != stem:
            continue
        families = [
            t for t in tokens[:-1]
            if t not in ("common", "pages", "man", "linux")
        ]
        if not families or any(t in path_l for t in families):
            return True
    return False
