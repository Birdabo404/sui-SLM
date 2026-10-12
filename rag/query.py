#!/usr/bin/env python3
"""Ask the index a question and see the top chunks with their sources.
  python rag/query.py --index rag/rag-index-v2 "how do I extract a tar.gz file"
  python rag/query.py --index rag/rag-index-v2 --exclude-distro arch "install a package"
"""
import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
try:
    from rag.chroma_client import persistent_client
except ImportError:
    from chroma_client import persistent_client

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--index", default=str(HERE / "rag-index-v2"))
ap.add_argument("-k", type=int, default=5)
ap.add_argument("--exclude-distro", default=None)
a = ap.parse_args()
col = persistent_client(f"{a.index}/chroma").get_collection("linux_lab")
model = SentenceTransformer(col.metadata.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"), device="cpu")
q = model.encode([a.question], normalize_embeddings=True).tolist()
where = {"distro": {"$ne": a.exclude_distro}} if a.exclude_distro else None
r = col.query(query_embeddings=q, n_results=a.k, where=where)
for doc, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
    print(f"--- score {1-d:.3f} | {m['source']} | {m['title']} | {m['path']}")
    print(doc[:500].replace("\n", " ") + ("..." if len(doc) > 500 else ""))
