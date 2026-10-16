#!/usr/bin/env python3
"""Step 3 of the KD data: check the teacher's rows and turn them into chat-format SFT data for the 2B.
1. Grounding filter: every command and option the teacher put in backticks must appear in its source
   chunk. Rows that invent a flag are dropped and counted (that count is itself a thesis number: how
   often even the 9B teacher fabricates). Rows with bad JSON or "skip" are dropped too.
2. Each kept row becomes an ANSWERABLE example: retrieved context = the source chunk + 2 distractor
   chunks in random order, target = steps + answer + which source it used.
3. --abstain-rate of rows also become an ABSTAIN example: same question, context WITHOUT the source
   chunk, target = a fixed "the docs I have don't cover this" reply. This is what teaches the 2B to say
   it doesn't know instead of guessing.
4. Optional --holdout: a text file of your hand-written test questions (one per line). Training rows
   too similar to any of them are dropped so the evaluation stays held out.
Output is {"messages": [system, user, assistant]} per line, which TRL's SFTTrainer reads directly.
  python kd/build_sft.py --raw teacher_raw.jsonl --seeds seed_chunks.jsonl --holdout test_questions.txt
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
from shared.prompt import ABSTAIN, SYSTEM, user_msg


def invented(answer, src):
    """Return the backticked commands/options in the answer that are NOT in the source text."""
    norm = lambda s: re.sub(r"[{}\[\]|]", " ", s)   # tldr writes options as {{[-x|--extract]}}
    text = norm(src).lower(); bad = []
    for span in re.findall(r"`([^`]+)`", answer):
        toks = norm(span).split()
        if not toks: continue
        # first word is the command, unless it is a path or file name like /home/user or ~/notes.txt
        checks = [toks[0]] if not re.match(r"^[-/~.]|.*/", toks[0]) else []
        if toks[0] == "sudo" and len(toks) > 1: checks = [toks[1]]
        checks += [t.split("=")[0] for t in toks if re.match(r"^--?[A-Za-z]", t)]
        for c in checks:
            c = c.lower()
            # short flags may be bundled in the docs (-xzf); accept if every letter appears as a flag somewhere
            if re.match(r"^-[a-z0-9]{2,}$", c) and all(re.search(rf"(?<![\w-])-\w*{re.escape(ch)}", text) for ch in c[1:]): continue
            if c not in text: bad.append(c)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="teacher_raw.jsonl"); ap.add_argument("--seeds", default="seed_chunks.jsonl")
    ap.add_argument("--out-train", default="sft_train.jsonl"); ap.add_argument("--out-val", default="sft_val.jsonl")
    ap.add_argument("--abstain-rate", type=float, default=0.25); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--holdout", default=None); ap.add_argument("--holdout-sim", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = random.Random(a.seed)

    seeds = {json.loads(l)["id"]: json.loads(l) for l in open(a.seeds, encoding="utf-8")}

    stats = collections.Counter(); kept = []
    for l in open(a.raw, encoding="utf-8"):
        r = json.loads(l); src = seeds.get(r["seed_id"])
        if r.get("error"): stats["dropped_bad_json"] += 1; continue
        if r.get("skip") or not r.get("question") or not r.get("answer"): stats["dropped_skip_or_empty"] += 1; continue
        if r.get("judge") == "no": stats["dropped_by_teacher_judge"] += 1; continue
        steps = r.get("steps") or []
        bad = invented(r["answer"] + " " + " ".join(map(str, steps)), src["text"])
        if bad: stats["dropped_invented_command_or_flag"] += 1; r["_invented"] = bad; continue
        kept.append(r)
    stats["kept"] = len(kept)

    if a.holdout:
        hq, gold_sources = load_holdout(a.holdout)
        if gold_sources:
            before = len(kept)
            kept = [
                r for r in kept
                if not path_matches_gold_source(seeds[r["seed_id"]].get("path"), gold_sources)
            ]
            stats["dropped_source_path_holdout"] = before - len(kept)
        if kept and hq:
            from sentence_transformers import SentenceTransformer, util
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            sim = util.cos_sim(
                m.encode([r["question"] for r in kept], normalize_embeddings=True),
                m.encode(hq, normalize_embeddings=True),
            )
            before = len(kept)
            kept = [r for r, row in zip(kept, sim) if float(row.max()) < a.holdout_sim]
            stats["dropped_too_close_to_holdout"] = before - len(kept)
        else:
            stats["dropped_too_close_to_holdout"] = 0

    ids = list(seeds)
    cmd = lambda c: re.split(r"[\s(]", c["title"].strip().lower())[0]   # "readlink (tldr common)" -> "readlink"

    def distractors(src, n):
        """Random seed chunks about a different page AND a different command, so an abstain example never
        secretly contains the answer (e.g. tldr readlink vs man readlink)."""
        out = []
        while len(out) < n:
            d = seeds[rng.choice(ids)]
            if d["path"] != src["path"] and cmd(d) != cmd(src) and d not in out: out.append(d)
        return out

    examples = []
    for r in kept:
        src = seeds[r["seed_id"]]; ctx = distractors(src, 2) + [src]; rng.shuffle(ctx)
        n = ctx.index(src) + 1
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(r.get("steps") or []))
        target = f"Steps:\n{steps}\n\nAnswer: {r['answer']}\n\nSource: [{n}] {src['title']}"
        examples.append({"kind": "answerable", "seed_id": r["seed_id"], "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r["question"], ctx)},
            {"role": "assistant", "content": target}]})
        if rng.random() < a.abstain_rate:
            examples.append({"kind": "abstain", "seed_id": r["seed_id"], "messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r["question"], distractors(src, 3))},
                {"role": "assistant", "content": ABSTAIN}]})

    # split by seed chunk so an answerable row and its abstain twin never land on different sides
    seed_ids = sorted({e["seed_id"] for e in examples}); rng.shuffle(seed_ids)
    val_ids = set(seed_ids[: int(len(seed_ids) * a.val_frac)])
    with open(a.out_train, "w", encoding="utf-8") as ft, open(a.out_val, "w", encoding="utf-8") as fv:
        for e in examples: (fv if e["seed_id"] in val_ids else ft).write(json.dumps(e, ensure_ascii=False) + "\n")
    stats["answerable_examples"] = sum(e["kind"] == "answerable" for e in examples)
    stats["abstain_examples"] = sum(e["kind"] == "abstain" for e in examples)
    stats["val_seed_chunks"] = len(val_ids)
    for k, v in stats.items(): print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
