# Thesis RAG index

Current default: `rag/lab-index` (ChromaDB collection `linux_lab`, all-MiniLM-L6-v2, cosine).
Rebuild on the laptop. Default sources skip Ubuntu Server docs (no license) and Arch Wiki.
chromadb 1.0.8 cannot read a 0.6.3 index. Do not reuse `rag-index` or `rag-index-v2`:
  python3 -m venv .venv && . .venv/bin/activate
  pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/lab-index
Ask:  python rag/query.py "how do I extract a tar.gz file"            (dense only)
      python rag/query_hybrid.py --tldr-lane "..."                    (dense + keyword + tldr lane)

Older trees (`rag-index`, `rag-index-trim`, `rag-index-v2`) are historical only. Counts and
spot-check notes below describe those builds, not `lab-index`.

## Historical: first box build (2026-10-11), rag-index/

Corpus: corpus/snapshot-2026-10-11 (English only).
Index:  rag-index/chroma, rag-index/chunks.jsonl
Size:   48,391 chunks (~1000 chars each): arch-wiki 22.9k, command-manpages 14.7k, tldr 7.4k,
        man-pages 2.8k, gnome-user-docs 641, ubuntu-docs 29. Build took ~9 min on 8 CPU cores, 463 MB on disk.

Known weaknesses from spot checks:
- Dense-only retrieval misses obvious pages ("ssh to another PC" -> VirtualBox, Arch install pages).
  Hybrid (BM25 + MiniLM) is better but still not great.
- 6,788 tldr pages are mostly niche tools (dvc, tlmgr, pacstall) that crowd out ssh/apt/tar.
  Trimming tldr to commands that actually exist on the lab image should help a lot.
- Arch Wiki is 47% of chunks and gives pacman/AUR answers; every chunk has a distro tag to filter.
  Arch Wiki is dropped from the default build_index.py --sources.

## Trimmed index (added 2026-10-11 evening): rag-index-trim/
Same chunks and embeddings, but tldr keeps only pages for commands that have an Ubuntu or man-pages
man page (plus a short EXTRAS list in trim/make_trimmed_index.py). tldr 6,788 -> 842 pages,
total 48,391 -> 41,945 chunks. rag-index/ is untouched. Rebuild: python rag/trim/make_trimmed_index.py --snap <snapshot>
Spot check (24 everyday lab questions, Arch excluded, trim/spotcheck-2026-10-11.txt):
  original dense 19/24, hybrid 17/24  ->  trimmed dense 20/24, hybrid 18/24
Still missed by both: tar.gz extract, make script executable (chmod), biggest folder (du),
test internet (ping), list drives (lsblk), run as admin (sudo). Dense beat hybrid here, so the
next lever is probably a small reranker or boosting tldr/man pages, not more trimming.
This is a quick sanity check, not the thesis evaluation set.

## Ranking tweaks tested 2026-10-12 (trimmed index, same 24 spot-check questions, Arch excluded)
  dense 20/24, hybrid 18/24
  hybrid + cross-encoder rerank (ms-marco-MiniLM-L-6-v2): 16/24, worse, and ~1.2 s/question on the box -> dropped
  hybrid + tldr lane (3rd RRF ranking = dense top-20 over tldr pages only): 22/24  <- best
  Still missed: make a script executable (chmod not even in the candidate pool), test internet (ping).
  The right page was in the top-100 pool for 23/24 questions, so the problem is ordering, not coverage.
  Use: python rag/query_hybrid.py --index rag/rag-index-trim --tldr-lane "..."
  Caveat: tuned on these same 24 questions, so confirm on held-out questions before calling it a result.
  Scripts/outputs: trim/rerank_spotcheck.py, trim/lanes_spotcheck.py, trim/*-spotcheck-2026-10-12.txt

## rag-index-v2 (2026-10-12, superseded): Arch Wiki DROPPED, Ubuntu Server docs ADDED, tldr trim kept
Historical. Server docs were added here, then dropped from the default corpus (no license). Rebuild
with `rag/build_index.py` (defaults have no Server docs) instead of reusing this tree.
Built by trim/make_v2_index.py from rag-index-trim (rag-index/ and rag-index-trim/ untouched):
reuses the 19,182 non-Arch embeddings as-is, embeds only the new Ubuntu Server chunks (~50 s on the box).
Ubuntu Server docs: canonical/ubuntu-server-documentation @ 9c2b8ca3a6db (2026-10-12), 243 English pages
(tutorial/how-to/explanation/reference; contributing/ and legacy/ skipped) -> 2,632 chunks.
Size: 21,814 chunks, 220 MB: command-manpages 14,799, man-pages 2,782, ubuntu-server-docs 2,632,
      tldr 932, gnome-user-docs 640, ubuntu-docs 29.
The ubuntu-server-docs loader is still in build_index.py as an opt-in (`--sources ...,ubuntu-server-docs`).
Use current default:  python rag/query_hybrid.py --index rag/lab-index --tldr-lane "..."

Spot check, same 24 questions (TUNING questions, not a held-out eval), trim/*-v2-2026-10-12.txt:
                 trim (Arch excluded)   v2
  dense               20/24            20/24
  hybrid              18/24            17/24   (new miss: "install a package" -> Server page "Install and
                                                manage packages", which is about apt but the title check wants "apt")
  hybrid+tldr lane    22/24            22/24   (still missed: chmod, ping)
So the swap is neutral on these everyday questions; the gain is Ubuntu-correct server content
(netplan, ufw, users, ssh, apt) that the spot check does not measure.
