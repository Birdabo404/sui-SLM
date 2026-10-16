#!/usr/bin/env python3
"""Build the thesis RAG index: English Linux docs -> chunks -> MiniLM embeddings -> ChromaDB.

Usage (from the repo root):
  python rag/build_index.py --snap corpus/snapshot-2026-10-11 --out rag/rag-index
  python rag/build_index.py --snap ... --out ... --sources tldr,command-manpages   # subset
  python rag/build_index.py --snap ... --out ... --chunks-only                     # just write chunks.jsonl

What it keeps (English only, everything else is skipped on purpose):
  tldr             common/ + linux/ pages (skips windows, osx, android, ...)
  command-manpages Ubuntu man1/5/8 pages for lab commands (ls, grep, ssh, apt, sudo, ...)
  man-pages        Linux man-pages project, sections 1/5/7/8 only (2/3 are C syscalls/libc,
                   e.g. chmod.2, which would be served for "how do I chmod a file")
  ubuntu-docs      the 19 English Ubuntu pages (help/C)
  gnome-user-docs  the 301 English GNOME desktop help pages (help/C)
  arch-wiki        English pages only (html/en); not in the default --sources
  ubuntu-server-docs Ubuntu Server documentation (MyST). Unlicensed; not in the default --sources.
                     Opt in with --sources ...,ubuntu-server-docs
Needs: pip install sentence-transformers chromadb beautifulsoup4 lxml ; groff (for man pages).
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

DEFAULT_SOURCES = ["tldr", "command-manpages", "man-pages", "ubuntu-docs", "gnome-user-docs"]
# ubuntu-server-docs is unlicensed (no LICENSE file; conf.py licence is TODO). Opt-in only.
ALL = DEFAULT_SOURCES + ["ubuntu-server-docs"]
LICENSE = {"tldr": "CC-BY-4.0", "command-manpages": "per package (licenses/command-manpages)",
           "man-pages": "mixed (licenses/man-pages-LICENSES)", "ubuntu-docs": "CC-BY-SA-4.0",
           "gnome-user-docs": "CC-BY-SA-3.0", "arch-wiki": "GFDL-1.3-or-later",
           "ubuntu-server-docs": "unlicensed"}
# Distro tag lets the app/eval notice when an Arch-only answer (pacman, AUR) is served for Ubuntu.
DISTRO = {"tldr": "generic", "command-manpages": "ubuntu", "man-pages": "generic",
          "ubuntu-docs": "ubuntu", "gnome-user-docs": "ubuntu", "arch-wiki": "arch",
          "ubuntu-server-docs": "ubuntu"}


def clean(t):
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", t).strip()


def render_man(path):
    raw = gzip.open(path, "rb").read() if str(path).endswith(".gz") else open(path, "rb").read()
    head = raw[:200].decode("utf-8", "ignore").strip()
    if head.startswith(".so ") and head.count("\n") <= 1:
        return None  # alias stub pointing at another page
    try:
        out = subprocess.run(["groff", "-t", "-man", "-Tutf8", "-P", "-cbou", "-rLL=100n"], input=raw,
                             capture_output=True, timeout=30, env={**os.environ, "MANWIDTH": "100"}).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return clean(out.decode("utf-8", "ignore"))


def man_title(path):
    n = path.name[:-3] if path.name.endswith(".gz") else path.name
    base, _, sec = n.rpartition(".")
    return f"{base}({sec})"


def mallard(path):
    from bs4 import BeautifulSoup
    s = BeautifulSoup(open(path, encoding="utf-8").read(), "xml")
    title = s.find("title").get_text(" ", strip=True) if s.find("title") else path.stem
    for tag in s.find_all(["info", "credit", "revision", "include"]):
        tag.decompose()
    blocks = [b.get_text(" ", strip=True) for b in s.find_all(["p", "title", "item", "code", "screen"])]
    seen, out = set(), []
    for b in blocks:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return title, clean("\n\n".join(out))


def arch(path):
    from bs4 import BeautifulSoup
    s = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "lxml")
    title = (s.title.text if s.title else path.stem).replace(" - ArchWiki", "").strip()
    body = s.select_one(".mw-parser-output") or s.body
    if body is None:
        return title, ""
    for tag in body.select("table.archwiki-template-box, .toc, #toc, .mw-editsection, style, script, .navbox"):
        tag.decompose()
    return title, clean(body.get_text(""))


def myst(path):
    """Ubuntu Server docs page (MyST markdown) -> (title, plain-ish text). Keeps code blocks."""
    t = open(path, encoding="utf-8").read()
    t = re.sub(r"\A---\n.*?\n---\n", "", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"(?m)^\([^)\n]+\)=\s*$", "", t)
    t = re.sub(r"(?m)^\s*(`{3,}|:{3,})\{(toctree|contents|include|image|figure|raw)\}.*?^\s*\1\s*$", "", t, flags=re.S)
    t = re.sub(r"(?m)^\s*(`{3,}|:{3,})\{([\w-]+)\}[^\n]*$", lambda m: m.group(2).capitalize() + ":", t)
    t = re.sub(r"(?m)^\s*:[\w-]+:.*$", "", t)
    t = re.sub(r"(?m)^\s*(`{3,}|:{3,})\s*$", "", t)
    t = re.sub(r"(?m)^(`{3,})\w*\s*$", "", t)
    t = re.sub(r"\{\w+\}`([^`<]*?)\s*<[^`>]*>`", r"\1", t)
    t = re.sub(r"\{\w+\}`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    m = re.search(r"(?m)^#\s+(.+)$", t)
    title = m.group(1).strip() if m else path.stem.replace("-", " ")
    # keep leading indentation (netplan/YAML examples break without it); only trim trailing spaces
    t = re.sub(r"[ \t]+$", "", t, flags=re.M)
    return title, re.sub(r"\n\s*\n\s*\n+", "\n\n", t).strip()


def docs(snap, sources):
    snap = Path(snap)
    if "tldr" in sources:
        for plat in ("common", "linux"):
            for p in sorted((snap / "tldr" / plat).glob("*.md")):
                yield "tldr", p, f"{p.stem} (tldr {plat})", p.read_text(encoding="utf-8")
    if "command-manpages" in sources:
        for p in sorted((snap / "command-manpages").rglob("*")):
            if p.is_file() and not p.is_symlink() and re.search(r"\.\d\w*(\.gz)?$", p.name):
                t = render_man(p)
                if t:
                    yield "command-manpages", p, man_title(p), t
    if "man-pages" in sources:
        for sec in ("man1", "man5", "man7", "man8"):
            for p in sorted((snap / "man-pages" / "man" / sec).glob("*")):
                if p.is_file() and not p.is_symlink():
                    t = render_man(p)
                    if t:
                        yield "man-pages", p, man_title(p), t
    for src, sub in (("ubuntu-docs", "ubuntu-docs/usr/share/help/C/gnome-help"),
                     ("gnome-user-docs", "gnome-user-docs/gnome-help")):
        if src in sources:
            for p in sorted((snap / sub).glob("*.page")):
                title, t = mallard(p)
                if t:
                    yield src, p, title, t
    if "arch-wiki" in sources:
        for p in sorted((snap / "arch-wiki/usr/share/doc/arch-wiki/html/en").glob("*.html")):
            title, t = arch(p)
            if len(t) > 200:
                yield "arch-wiki", p, title, t
    if "ubuntu-server-docs" in sources:
        for sub in ("tutorial", "how-to", "explanation", "reference"):
            for p in sorted((snap / "ubuntu-server-docs/docs" / sub).rglob("*.md")):
                title, t = myst(p)
                if len(t) > 200:
                    yield "ubuntu-server-docs", p, title, t


def chunk(text, size=1000, overlap=150):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for para in paras:
        while len(para) > size:
            out.append((cur + "\n\n" + para[:size]).strip() if cur else para[:size])
            cur = ""
            para = para[size - overlap:]
            para = para[para.find(" ") + 1:] if " " in para[:50] else para
        if len(cur) + len(para) + 2 > size and cur:
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            out.append(cur)
            cur = tail + "\n\n" + para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur.strip():
        out.append(cur)
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", default=str(here / "rag-index"))
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--chunks-only", action="store_true")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(a.snap).resolve()
    t0 = time.time()
    rows = []
    per = {}
    for src, path, title, text in docs(snap, sources):
        rel = str(Path(path).resolve().relative_to(snap))
        for i, c in enumerate(chunk(text)):
            rows.append({"id": f"{rel}#{i}", "text": f"{title}\n{c}", "source": src, "title": title,
                         "path": rel, "chunk": i, "license": LICENSE[src], "distro": DISTRO[src]})
        per[src] = per.get(src, 0) + 1
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"docs per source: {per}\nchunks: {len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    if a.chunks_only:
        return
    from sentence_transformers import SentenceTransformer
    try:
        from rag.chroma_client import persistent_client
    except ImportError:
        from chroma_client import persistent_client
    model = SentenceTransformer(a.model, device="cpu")
    client = persistent_client(out / "chroma")
    existing = []
    for c in client.list_collections():
        existing.append(c if isinstance(c, str) else c.name)
    if "linux_lab" in existing:
        client.delete_collection("linux_lab")
    col = client.create_collection("linux_lab", metadata={"hnsw:space": "cosine", "embed_model": a.model,
                                                          "snapshot": snap.name})
    step = 512
    for s in range(0, len(rows), step):
        b = rows[s:s + step]
        emb = model.encode([r["text"] for r in b], batch_size=a.batch, normalize_embeddings=True).tolist()
        col.add(ids=[r["id"] for r in b], embeddings=emb, documents=[r["text"] for r in b],
                metadatas=[{k: r[k] for k in ("source", "title", "path", "chunk", "license", "distro")} for r in b])
        print(f"  embedded {min(s+step, len(rows))}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {col.count()} chunks in {out/'chroma'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
