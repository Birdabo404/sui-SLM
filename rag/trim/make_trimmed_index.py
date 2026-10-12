#!/usr/bin/env python3
"""Make rag-index-trim: same index, but tldr pages only for commands that ship a man page in the
Ubuntu command-manpages set or the man-pages project (man1/man8), plus a short EXTRAS list of lab
tools. Re-uses the existing embeddings (no re-embedding), leaves rag-index untouched.
  python rag/trim/make_trimmed_index.py --src rag/rag-index --snap <snapshot> --out rag/rag-index-trim
"""
import argparse, json, re, chromadb
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
EXTRAS = {"git", "python", "python3", "pip", "pip3", "vim", "nano", "code", "gcc", "make", "java",
          "javac", "docker", "ufw", "nmcli", "ip", "ss", "ping", "traceroute", "ssh", "scp", "sftp",
          "systemctl", "journalctl", "snap", "flatpak", "apt", "apt-get", "dpkg", "ping6", "nmap"}
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(HERE / "rag-index"))
ap.add_argument("--snap", required=True); ap.add_argument("--out", default=str(HERE / "rag-index-trim"))
a = ap.parse_args(); snap = Path(a.snap)
names = set(EXTRAS)
for p in list((snap / "command-manpages").rglob("*")) + list((snap / "man-pages/man").glob("man[18]/*")):
    if p.is_file() or p.is_symlink():
        n = re.sub(r"\.[0-9][a-z0-9]*$", "", re.sub(r"\.gz$", "", p.name))
        # Ubuntu 25.10 ships coreutils pages as gnuchmod.1 / rust-chmod.1; map back to chmod
        if "gnu-coreutils" in p.parts: n = re.sub(r"^gnu", "", n)
        if "rust-coreutils" in p.parts: n = re.sub(r"^rust-", "", n)
        names.add(re.sub(r"\.ws$", "", n))  # sudo.ws.8 -> sudo
def keep(r):
    if r["source"] != "tldr": return True
    stem = Path(r["path"]).stem
    return stem in names or stem.split("-")[0] in names
rows = [json.loads(l) for l in open(f"{a.src}/chunks.jsonl", encoding="utf-8")]
kept = [r for r in rows if keep(r)]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
    for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
src = chromadb.PersistentClient(path=f"{a.src}/chroma").get_collection("linux_lab")
dst_c = chromadb.PersistentClient(path=str(out / "chroma"))
try: dst_c.delete_collection("linux_lab")
except Exception: pass
dst = dst_c.create_collection("linux_lab", metadata=dict(src.metadata, trimmed_tldr=True))
ids = [r["id"] for r in kept]
for s in range(0, len(ids), 2000):
    g = src.get(ids=ids[s:s+2000], include=["embeddings", "documents", "metadatas"])
    dst.add(ids=g["ids"], embeddings=g["embeddings"], documents=g["documents"], metadatas=g["metadatas"])
t_all = sum(r["source"] == "tldr" for r in rows); t_kept = sum(r["source"] == "tldr" for r in kept)
pages = lambda rs: len({r["path"] for r in rs if r["source"] == "tldr"})
print(f"allowlist names: {len(names)}")
print(f"tldr chunks {t_all} -> {t_kept}; tldr pages {pages(rows)} -> {pages(kept)}; total chunks {len(rows)} -> {dst.count()}")
