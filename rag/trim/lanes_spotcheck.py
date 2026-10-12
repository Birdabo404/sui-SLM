#!/usr/bin/env python3
"""Spot check of two cheap ranking tweaks on the same 24 questions (Arch excluded):
  dedup  = keep at most one chunk per page title in the top 5
  +tldr  = add a third RRF lane: dense top-20 restricted to tldr pages (task-phrased cheat sheets)
  python rag/trim/lanes_spotcheck.py rag/rag-index-trim
"""
import json, os, re, sys
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from chromadb.config import Settings
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
_spot = Path(__file__).resolve().parent / "spotcheck.py"
src = open(_spot).read(); QS = eval(src[src.index("QS = ")+5:src.index("tok = ")])
tok = lambda s: re.findall(r"[a-z0-9][a-z0-9_.+-]*", s.lower())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
qemb = model.encode([q for q, _ in QS], normalize_embeddings=True).tolist()
def rrf(*lanes):
    f = {}
    for rk in lanes:
        for r, i in enumerate(rk): f[i] = f.get(i, 0) + 1 / (60 + r)
    return sorted(f, key=lambda i: -f[i])
for idx in sys.argv[1:]:
    rows = [r for r in map(json.loads, open(f"{idx}/chunks.jsonl", encoding="utf-8")) if r["distro"] != "arch"]
    bm25 = BM25Okapi([tok(r["text"]) for r in rows]); byid = {r["id"]: r for r in rows}
    col = chromadb.PersistentClient(path=f"{idx}/chroma", settings=Settings(anonymized_telemetry=False)).get_collection("linux_lab")
    def top5(ranked, dedup):
        out, seen = [], set()
        for i in ranked:
            t = byid[i]["title"]
            if dedup and t in seen: continue
            seen.add(t); out.append(i)
            if len(out) == 5: break
        return out
    res = {k: 0 for k in ["dense", "dense+dedup", "hybrid", "hybrid+dedup", "hybrid+tldr", "hybrid+tldr+dedup", "dense+tldr+dedup"]}
    miss = {k: [] for k in res}
    for (q, exp), e in zip(QS, qemb):
        pat = re.compile(rf"^(gnu|rust-)?({exp})\b", re.I)
        d = col.query(query_embeddings=[e], n_results=50, where={"distro": {"$ne": "arch"}})["ids"][0]
        t = col.query(query_embeddings=[e], n_results=20, where={"source": "tldr"})["ids"][0]
        sc = bm25.get_scores(tok(q)); b = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -sc[i])[:50]]
        cand = {"dense": top5(d, False), "dense+dedup": top5(d, True),
                "hybrid": top5(rrf(d, b), False), "hybrid+dedup": top5(rrf(d, b), True),
                "hybrid+tldr": top5(rrf(d, b, t), False), "hybrid+tldr+dedup": top5(rrf(d, b, t), True),
                "dense+tldr+dedup": top5(rrf(d, t), True)}
        for k, ids in cand.items():
            ok = any(pat.match(byid[i]["title"]) for i in ids); res[k] += ok
            if not ok: miss[k].append(q)
    for k in res: print(f"{idx} {k:20s} hit@5 {res[k]}/{len(QS)}   misses: {miss[k]}")
