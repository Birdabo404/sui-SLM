#!/usr/bin/env python3
"""Quick before/after retrieval spot check (NOT the thesis eval set). Hit = a top-5 chunk whose
title starts with the expected command (or an
obviously right GNOME help page for ssh/df). Ubuntu-safe mode (Arch Wiki excluded), hybrid BM25+MiniLM.
  python rag/trim/spotcheck.py rag/rag-index rag/rag-index-trim
"""
import json, re, sys, chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
QS = [("how do I extract a tar.gz file", "tar"), ("connect to another lab PC over ssh", "ssh|Enable remote access over Secure Shell"),
      ("install a package from the terminal", "apt"), ("make a script executable", "chmod"),
      ("change the owner of a folder", "chown"), ("check how much disk space is left", "df|Check how much disk space"),
      ("find which folder is using the most space", "du"), ("stop a frozen program", "kill"),
      ("see which processes use the most CPU", "top"), ("copy a file to another computer on the network", "scp|rsync|sftp"),
      ("show my IP address", "ip"), ("test if the internet connection works", "ping"),
      ("search for text inside files", "grep"), ("find a file by name", "find"),
      ("create a new user account", "useradd|adduser"), ("change my password", "passwd"),
      ("list USB drives and partitions", "lsblk"), ("restart a service", "systemctl"),
      ("unzip a zip file", "unzip"), ("run a command as administrator", "sudo"),
      ("enable the firewall", "ufw"), ("see how much RAM is free", "free"),
      ("clone a git repository", "git"), ("schedule a job to run every day", "cron|crontab")]
tok = lambda s: re.findall(r"[a-z0-9][a-z0-9_.+-]*", s.lower())
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
qemb = model.encode([q for q, _ in QS], normalize_embeddings=True).tolist()
for idx in sys.argv[1:]:
    rows = [r for r in map(json.loads, open(f"{idx}/chunks.jsonl", encoding="utf-8")) if r["distro"] != "arch"]
    bm25 = BM25Okapi([tok(r["text"]) for r in rows]); byid = {r["id"]: r for r in rows}
    col = chromadb.PersistentClient(path=f"{idx}/chroma").get_collection("linux_lab")
    hd = hh = 0; misses = []
    for (q, exp), e in zip(QS, qemb):
        pat = re.compile(rf"^(gnu|rust-)?({exp})\b", re.I)
        d = col.query(query_embeddings=[e], n_results=50, where={"distro": {"$ne": "arch"}})["ids"][0]
        sc = bm25.get_scores(tok(q)); b = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -sc[i])[:50]]
        f = {}
        for rk in (d, b):
            for r, i in enumerate(rk): f[i] = f.get(i, 0) + 1 / (60 + r)
        hyb = sorted(f, key=lambda i: -f[i])[:5]
        okd = any(pat.match(byid[i]["title"]) for i in d[:5]); okh = any(pat.match(byid[i]["title"]) for i in hyb)
        hd += okd; hh += okh
        if not okh: misses.append(f"{q} -> {[byid[i]['title'] for i in hyb[:3]]}")
    print(f"{idx}: dense hit@5 {hd}/{len(QS)}, hybrid hit@5 {hh}/{len(QS)}")
    for m in misses: print("   miss:", m)
