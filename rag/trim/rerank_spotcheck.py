#!/usr/bin/env python3
"""Same 24-question spot check as trim/spotcheck.py, plus a cross-encoder reranker
(cross-encoder/ms-marco-MiniLM-L-6-v2, ~90 MB, CPU) applied to the union of dense top-50 and BM25 top-50.
Also reports whether the expected page was in the candidate pool at all (recall@pool).
  python rag/trim/rerank_spotcheck.py rag/rag-index-trim
"""
import json, os, re, sys, time
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from chromadb.config import Settings
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
_trim = Path(__file__).resolve().parent
sys.path.insert(0, str(_trim))
src = open(_trim / "spotcheck.py").read(); QS = eval(src[src.index("QS = ")+5:src.index("tok = ")])
tok = lambda s: re.findall(r"[a-z0-9][a-z0-9_.+-]*", s.lower())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
qemb = model.encode([q for q, _ in QS], normalize_embeddings=True).tolist()
for idx in sys.argv[1:]:
    rows = [r for r in map(json.loads, open(f"{idx}/chunks.jsonl", encoding="utf-8")) if r["distro"] != "arch"]
    bm25 = BM25Okapi([tok(r["text"]) for r in rows]); byid = {r["id"]: r for r in rows}
    col = chromadb.PersistentClient(path=f"{idx}/chroma", settings=Settings(anonymized_telemetry=False)).get_collection("linux_lab")
    hd = hh = hr = hp = 0; misses = []; t_rr = 0
    for (q, exp), e in zip(QS, qemb):
        pat = re.compile(rf"^(gnu|rust-)?({exp})\b", re.I)
        d = col.query(query_embeddings=[e], n_results=50, where={"distro": {"$ne": "arch"}})["ids"][0]
        sc = bm25.get_scores(tok(q)); b = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -sc[i])[:50]]
        f = {}
        for rk in (d, b):
            for r, i in enumerate(rk): f[i] = f.get(i, 0) + 1 / (60 + r)
        hyb = sorted(f, key=lambda i: -f[i])[:5]
        pool = list(f)
        t0 = time.time()
        s = ce.predict([(q, byid[i]["title"] + "\n" + byid[i]["text"]) for i in pool], batch_size=32)
        t_rr += time.time() - t0
        rr = [pool[j] for j in sorted(range(len(pool)), key=lambda j: -s[j])[:5]]
        ok = lambda ids: any(pat.match(byid[i]["title"]) for i in ids)
        hd += ok(d[:5]); hh += ok(hyb); hr += ok(rr); hp += ok(pool)
        if not ok(rr): misses.append(f"{q} -> {[byid[i]['title'] for i in rr[:3]]} (expected in pool: {ok(pool)})")
    n = len(QS)
    print(f"{idx}: dense {hd}/{n}, hybrid {hh}/{n}, hybrid+rerank {hr}/{n}, expected page in pool {hp}/{n}")
    print(f"   rerank time: {t_rr/n:.2f} s per question on this box (~{len(pool)} candidates)")
    for m in misses: print("   miss:", m)
