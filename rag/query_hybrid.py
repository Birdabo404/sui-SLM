#!/usr/bin/env python3
"""Hybrid retrieval: MiniLM (meaning) + BM25 (exact words like 'ssh', 'apt', '-xzf'), merged with
reciprocal rank fusion. Dense-only misses obvious pages (e.g. 'ssh' query -> VirtualBox page).
  python rag/query_hybrid.py --index rag/rag-index-v2 "connect to another lab PC over ssh"
"""
import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

HERE = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*")


def tokenize(s):
    return TOKEN_RE.findall(s.lower())


def rrf_fuse(lanes, k=60):
    fused = {}
    for ranking in lanes:
        for rank, i in enumerate(ranking):
            fused[i] = fused.get(i, 0) + 1 / (k + rank)
    return fused


def load_rows(index, exclude_distro=None):
    rows = [json.loads(l) for l in open(f"{index}/chunks.jsonl", encoding="utf-8")]
    if exclude_distro:
        rows = [r for r in rows if r["distro"] != exclude_distro]
    return rows


def _chroma(path):
    try:
        from rag.chroma_client import persistent_client
    except ImportError:
        from chroma_client import persistent_client
    return persistent_client(path)


def _first_row(value):
    if value is None or len(value) == 0:
        return []
    row = value[0]
    if row is None:
        return []
    return row


def _vec_len(emb):
    if emb is None:
        return 0
    return len(emb)


def dense_scores(col, qemb, hit_ids, query_result):
    """MiniLM cosine for fused hits. Chroma cosine distance is 1 - cosine."""
    sim = {}
    qids = _first_row(query_result.get("ids"))
    dists = _first_row(query_result.get("distances"))
    for i, d in zip(qids, dists):
        sim[i] = 1.0 - float(d)
    missing = [i for i in hit_ids if i not in sim]
    if len(missing) == 0:
        return sim
    got = col.get(ids=missing, include=["embeddings"])
    qvec = qemb[0]
    embs = got.get("embeddings")
    got_ids = got.get("ids")
    if embs is None or got_ids is None:
        return sim
    for i, emb in zip(got_ids, embs):
        if _vec_len(emb) == 0:
            continue
        sim[i] = float(sum(a * b for a, b in zip(qvec, emb)))
    return sim


def retrieve(question, index="rag-index", k=5, exclude_distro=None, tldr_lane=False,
             rows=None, bm25=None, col=None, model=None):
    """Return top-k chunk dicts with rrf, dense (MiniLM cosine), and bm25 scores."""
    # Heavy deps stay behind the None branches so tokenize/rrf_fuse and injected
    # fakes can run without MiniLM.
    if rows is None:
        rows = load_rows(index, exclude_distro)
    if bm25 is None:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])
    sc = bm25.get_scores(tokenize(question))
    bm25_by_id = {rows[i]["id"]: float(sc[i]) for i in range(len(rows))}
    bm_ids = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -sc[i])[:50]]
    if col is None:
        col = _chroma(f"{index}/chroma").get_collection("linux_lab")
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            col.metadata.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device="cpu",
        )
    where = {"distro": {"$ne": exclude_distro}} if exclude_distro else None
    qemb = model.encode([question], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qemb, n_results=50, where=where)
    lanes = [r["ids"][0], bm_ids]
    if tldr_lane:
        lanes.append(
            col.query(query_embeddings=qemb, n_results=20, where={"source": "tldr"})["ids"][0]
        )
    fused = rrf_fuse(lanes)
    byid = {x["id"]: x for x in rows}
    ranked_ids = [i for i in sorted(fused, key=lambda i: -fused[i]) if i in byid][:k]
    sim = dense_scores(col, qemb, ranked_ids, r)
    hits = []
    for i in ranked_ids:
        m = dict(byid[i])
        m["rrf"] = fused[i]
        m["dense"] = sim.get(i)
        m["bm25"] = bm25_by_id.get(i)
        hits.append(m)
    return hits


class HybridRetriever:
    """Load chunks, BM25, Chroma, and MiniLM once; reuse for each chat turn."""

    def __init__(self, index, exclude_distro=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.index = index
        self.exclude_distro = exclude_distro
        self.rows = load_rows(index, exclude_distro)
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in self.rows])
        self.col = _chroma(f"{index}/chroma").get_collection("linux_lab")
        self.model = SentenceTransformer(
            self.col.metadata.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device="cpu",
        )

    def retrieve(self, question, k=5, tldr_lane=True):
        return retrieve(
            question,
            index=self.index,
            k=k,
            exclude_distro=self.exclude_distro,
            tldr_lane=tldr_lane,
            rows=self.rows,
            bm25=self.bm25,
            col=self.col,
            model=self.model,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=str(HERE / "rag-index-v2"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--exclude-distro", default=None)
    ap.add_argument("--tldr-lane", action="store_true", help="add dense top-20 tldr-only ranking as a 3rd RRF lane")
    a = ap.parse_args()
    hits = retrieve(
        a.question,
        index=a.index,
        k=a.k,
        exclude_distro=a.exclude_distro,
        tldr_lane=a.tldr_lane,
    )
    for m in hits:
        dense = m.get("dense")
        dense_s = f"{dense:.3f}" if dense is not None else "-"
        print(f"--- rrf {m['rrf']:.4f} | cos {dense_s} | {m['source']} | {m['title']} | {m['path']}")
        print(m["text"][:400].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
