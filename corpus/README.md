# Thesis corpus (snapshot-2026-10-11, extended 2026-10-12)

corpus/snapshot-2026-10-11/  English docs used by ../rag. MANIFEST*.csv list version, URL, sha256,
licence and fetch time for every download; licences/ holds the licence files.
  tldr-pages v2.3 (CC BY 4.0) | man-pages 6.19 (mixed) | command-manpages (MANIFEST-command-manpages.csv)
  ubuntu-docs 26.04.1 + gnome-user-docs 50.0 (CC BY-SA 3.0)
  ubuntu-server-docs: canonical/ubuntu-server-documentation commit 9c2b8ca3a6db (license unverified, see note)
  arch-wiki 20261003: still on disk for reproducing the old indexes, but DROPPED from the corpus
                      2026-10-12 (not in rag-index-v2, not fetched by default any more).
Scripts: fetch_corpus.sh (tldr, man-pages, ubuntu-docs, ubuntu-server-docs; SKIP_ARCH=0 to also fetch
Arch, USD_COMMIT=<sha> to pin another Server-docs commit), fetch_command_manpages.sh, fetch_gnome_user_docs.sh.

Licence note for Ubuntu Server docs: the repo has no LICENSE file, docs/conf.py leaves the licence field
empty (TODO), and the site footer names no licence. Treat the licence as unverified until Canonical
confirms it. GitHub archive tarballs are not guaranteed byte-stable, so a refetch may give a different
sha256 for the same commit; the commit hash is the real pin.
