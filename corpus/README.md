# Thesis corpus (snapshot-2026-10-11, extended 2026-10-12)

corpus/snapshot-2026-10-11/  English docs used by ../rag. MANIFEST*.csv list version, URL, sha256,
licence and fetch time for every download; licences/ holds the licence files.
  tldr-pages v2.3 (CC BY 4.0) | man-pages 6.19 (mixed) | command-manpages (MANIFEST-command-manpages.csv)
  ubuntu-docs 26.04.1 (CC BY-SA 4.0, help.ubuntu.com/legal.html) + gnome-user-docs 50.0 (CC BY-SA 3.0)
  ubuntu-server-docs: not fetched by default. canonical/ubuntu-server-documentation has no LICENSE,
                      conf.py licence is TODO, and the site only shows a copyright footer.
  arch-wiki 20261003: still on disk for reproducing the old indexes, but DROPPED from the corpus
                      2026-10-12 (not in the default index, not fetched by default any more).
Scripts: fetch_corpus.sh (tldr, man-pages, ubuntu-docs; SKIP_ARCH=0 to also fetch Arch;
INCLUDE_UBUNTU_SERVER=1 to also fetch Server docs, unlicensed). fetch_command_manpages.sh,
fetch_gnome_user_docs.sh.

Ubuntu Server docs stay as an opt-in only. If you fetch them, MANIFEST.csv must say unlicensed,
not CC-BY-SA. GitHub archive tarballs are not byte-stable; pin by commit hash (`USD_COMMIT`).
