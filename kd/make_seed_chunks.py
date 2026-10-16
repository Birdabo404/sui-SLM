#!/usr/bin/env python3
"""Pick the doc chunks the 9B teacher will write training Q&A from (step 1 of the KD data).

Takes the RAG index (default: rag/lab-index, no Arch Wiki, no Ubuntu Server docs),
drops Arch-only pages, drops tiny or alias-only chunks, then samples a fixed,
seeded mix per source. Optional --holdout drops chunks whose path matches a
held-out gold_source.

  python make_seed_chunks.py --chunks ../rag/lab-index/chunks.jsonl --out seed_chunks.jsonl
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.holdout import load_holdout, path_matches_gold_source
from shared.paths import DEFAULT_INDEX

DEFAULT_MIX = "tldr=460,command-manpages=295,man-pages=115,gnome-user-docs=230,ubuntu-docs=29"
ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default=str(Path(DEFAULT_INDEX) / "chunks.jsonl"))
ap.add_argument("--out", default="seed_chunks.jsonl")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--mix", default=DEFAULT_MIX)
ap.add_argument("--holdout", default=None)


def _keep_chunk(row, mix):
    if row["distro"] == "arch" or row["source"] not in mix:
        return False
    text = row["text"]
    if len(text) < 250:
        return False
    if re.search(r"This command is an alias of", text):
        return False
    if row["source"] == "ubuntu-server-docs" and text.count("\n- ") > 12:
        return False
    if row["source"] in ("command-manpages", "man-pages") and row["chunk"] > 1:
        return False
    return True


def main(argv=None):
    a = ap.parse_args(argv)
    mix = {k: int(v) for k, v in (p.split("=") for p in a.mix.split(","))}
    gold_sources = []
    if a.holdout:
        _, gold_sources = load_holdout(a.holdout)
    pool = collections.defaultdict(list)
    dropped_source_path_holdout = 0
    for line in open(a.chunks, encoding="utf-8"):
        row = json.loads(line)
        if not _keep_chunk(row, mix):
            continue
        if gold_sources and path_matches_gold_source(row.get("path"), gold_sources):
            dropped_source_path_holdout += 1
            continue
        pool[row["source"]].append(row)
    rng = random.Random(a.seed)
    out = []
    for src, n in mix.items():
        rows = pool[src]
        rng.shuffle(rows)
        seen = set()
        pick = []
        for row in rows:
            if row["path"] in seen:
                continue
            seen.add(row["path"])
            pick.append(row)
            if len(pick) == n:
                break
        print(f"{src:18s} pool {len(rows):6d}  picked {len(pick)}")
        out += pick
    rng.shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"dropped_source_path_holdout {dropped_source_path_holdout}")
    print(f"wrote {len(out)} seed chunks -> {a.out}")


if __name__ == "__main__":
    main()
