#!/usr/bin/env python3
"""Make rag-index-v2 (2026-10-12): rag-index-trim WITHOUT the Arch Wiki, PLUS the Ubuntu Server docs.
Copies the existing embeddings for every kept chunk (no re-embedding), embeds only the new Ubuntu
Server chunks with the same MiniLM model, and leaves rag-index/ and rag-index-trim/ untouched.
  python rag/trim/make_v2_index.py --snap <snapshot> --src rag/rag-index-trim --out rag/rag-index-v2
"""
import argparse, collections, json, os, sys
from pathlib import Path
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from chromadb.config import Settings
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from build_index import docs, chunk, LICENSE, DISTRO
ap = argparse.ArgumentParser(); ap.add_argument("--snap", required=True)
ap.add_argument("--src", default=str(HERE / "rag-index-trim")); ap.add_argument("--out", default=str(HERE / "rag-index-v2"))
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
a = ap.parse_args(); snap = Path(a.snap).resolve(); out = Path(a.out)
assert out.resolve() not in (Path(a.src).resolve(), (HERE / "rag-index").resolve()), "refusing to overwrite an old index"
kept = [r for r in map(json.loads, open(f"{a.src}/chunks.jsonl", encoding="utf-8")) if r["source"] != "arch-wiki"]
new = []
for src, path, title, text in docs(snap, ["ubuntu-server-docs"]):
    rel = str(Path(path).resolve().relative_to(snap))
    for i, c in enumerate(chunk(text)):
        new.append({"id": f"{rel}#{i}", "text": f"{title}\n{c}", "source": src, "title": title, "path": rel,
                    "chunk": i, "license": LICENSE[src], "distro": DISTRO[src]})
out.mkdir(parents=True, exist_ok=True)
with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
    for r in kept + new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
_chroma = Settings(anonymized_telemetry=False)
src = chromadb.PersistentClient(path=f"{a.src}/chroma", settings=_chroma).get_collection("linux_lab")
dc = chromadb.PersistentClient(path=str(out / "chroma"), settings=_chroma)
if "linux_lab" in [c if isinstance(c, str) else c.name for c in dc.list_collections()]:
    dc.delete_collection("linux_lab")
meta = {k: v for k, v in src.metadata.items()}; meta.update(arch_wiki="dropped", ubuntu_server_docs="added")
dst = dc.create_collection("linux_lab", metadata=meta)
ids = [r["id"] for r in kept]
for s in range(0, len(ids), 2000):
    g = src.get(ids=ids[s:s+2000], include=["embeddings", "documents", "metadatas"])
    dst.add(ids=g["ids"], embeddings=g["embeddings"], documents=g["documents"], metadatas=g["metadatas"])
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(a.model, device="cpu")
for s in range(0, len(new), 512):
    b = new[s:s+512]
    emb = model.encode([r["text"] for r in b], batch_size=64, normalize_embeddings=True).tolist()
    dst.add(ids=[r["id"] for r in b], embeddings=emb, documents=[r["text"] for r in b],
            metadatas=[{k: r[k] for k in ("source", "title", "path", "chunk", "license", "distro")} for r in b])
c = collections.Counter(r["source"] for r in kept + new)
print(f"reused {len(kept)} embeddings, embedded {len(new)} new chunks from {len({r['path'] for r in new})} Ubuntu Server pages")
print(f"per source: {dict(c)}  total {dst.count()}")
