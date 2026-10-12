#!/usr/bin/env python3
"""Pick the doc chunks the 9B teacher will write training Q&A from (step 1 of the KD data).
Takes the RAG index (default rag-index-v2: no Arch Wiki, + Ubuntu Server docs), drops Arch-only pages (pacman/AUR answers are wrong on the Ubuntu lab
image), drops tiny or alias-only chunks, then samples a fixed, seeded mix per source so the run is
repeatable and can be described in Ch3.
  python make_seed_chunks.py --chunks ../rag/rag-index-v2/chunks.jsonl --out seed_chunks.jsonl
Old (v1) seeds: --chunks ../rag/rag-index-trim/chunks.jsonl --mix tldr=400,command-manpages=250,man-pages=100,gnome-user-docs=200,ubuntu-docs=29
"""
import argparse, collections, json, random, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default=str(ROOT / "rag" / "rag-index-v2" / "chunks.jsonl"))
ap.add_argument("--out", default="seed_chunks.jsonl")
ap.add_argument("--seed", type=int, default=42)
# How many chunks per source. tldr and Ubuntu/GNOME help are the "everyday lab question" sources, so
# they get most of the budget; man pages are long and dense, so we take their first chunks only.
# ubuntu-server-docs (added 2026-10-12): task-style Ubuntu how-tos (apt, netplan, ssh, users, ufw...),
# any chunk of a page, still one chunk per page; many pages are datacenter topics, so a modest share.
ap.add_argument("--mix", default="tldr=400,command-manpages=250,man-pages=100,gnome-user-docs=200,ubuntu-docs=29,ubuntu-server-docs=150")
a = ap.parse_args()
mix = {k: int(v) for k, v in (p.split("=") for p in a.mix.split(","))}
pool = collections.defaultdict(list)
for line in open(a.chunks, encoding="utf-8"):
    r = json.loads(line)
    if r["distro"] == "arch" or r["source"] not in mix: continue
    t = r["text"]
    if len(t) < 250: continue                                   # too little to ground a real answer
    if re.search(r"This command is an alias of", t): continue   # tldr alias stubs
    if r["source"] == "ubuntu-server-docs" and t.count("\n- ") > 12: continue  # link lists / landing pages
    if r["source"] in ("command-manpages", "man-pages") and r["chunk"] > 1: continue  # NAME/SYNOPSIS/DESCRIPTION area
    pool[r["source"]].append(r)
rng = random.Random(a.seed); out = []
for src, n in mix.items():
    rows = pool[src]; rng.shuffle(rows)
    # at most one chunk per page so one long page can't dominate
    seen, pick = set(), []
    for r in rows:
        if r["path"] in seen: continue
        seen.add(r["path"]); pick.append(r)
        if len(pick) == n: break
    print(f"{src:18s} pool {len(rows):6d}  picked {len(pick)}")
    out += pick
rng.shuffle(out)
with open(a.out, "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} seed chunks -> {a.out}")
