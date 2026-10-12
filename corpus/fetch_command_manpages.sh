#!/usr/bin/env bash
# Adds the man pages for everyday lab COMMANDS (ls, grep, ssh, apt, sudo, ...) to the corpus.
# Why: the Linux man-pages project fetched by fetch_corpus.sh is mostly syscalls/libc
# (sections 2/3); it has no ls/cp/grep/ssh/apt/sudo/tar/find/ps pages, and chmod/kill/mount
# only exist as C syscall pages (chmod.2), which retrieval would wrongly serve.
# Pulls the exact Ubuntu packages for the lab release, keeps only man1/man5/man8 + copyright.
# Additive: writes command-manpages/ and MANIFEST-command-manpages.csv; touches nothing else.
# Run from anywhere:  bash corpus/fetch_command_manpages.sh
# SNAP defaults to corpus/snapshot-2026-10-11 next to this script; set SNAP= after a fresh fetch.
set -euo pipefail
command -v dpkg-deb >/dev/null || { echo "Missing dpkg-deb (sudo apt install dpkg)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE="${SUITE:-resolute}"     # Ubuntu 26.04, same release as ubuntu-docs 26.04.1
SNAP="${SNAP:-$SCRIPT_DIR/snapshot-2026-10-11}"
PKGS="rust-coreutils gnu-coreutils sudo-rs sudo grep findutils tar gzip sed gawk mawk util-linux mount bsdutils bsdextrautils procps psmisc openssh-client openssh-server apt dpkg iproute2 iputils-ping net-tools bash passwd adduser systemd less nano git-man curl wget rsync zip unzip man-db file diffutils ufw network-manager cron hostname e2fsprogs"
OUT="$SNAP/command-manpages"; DL="$SNAP/downloads/command-debs"; LIC="$SNAP/licenses/command-manpages"
mkdir -p "$OUT" "$DL" "$LIC"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
NOW="$(date -Iseconds)"
for c in main universe; do
  curl -fsSL --retry 3 "http://archive.ubuntu.com/ubuntu/dists/$SUITE/$c/binary-amd64/Packages.xz" \
   | python3 -c "import sys,lzma;sys.stdout.buffer.write(lzma.decompress(sys.stdin.buffer.read()))" > "$TMP/$c"
done
M="$SNAP/MANIFEST-command-manpages.csv"
echo "package,version,url,sha256,license_file,fetched_at" > "$M"
for p in $PKGS; do
  read -r f v < <(awk -v p="$p" '$1=="Package:"{x=($2==p)} x&&$1=="Version:"{v=$2} x&&$1=="Filename:"{print $2, v; exit}' "$TMP/main" "$TMP/universe") || { echo "skip $p (not in $SUITE)"; continue; }
  deb="$DL/$(basename "$f")"
  [ -s "$deb" ] || curl -fsSL --retry 3 -o "$deb" "http://archive.ubuntu.com/ubuntu/$f"
  rm -rf "$TMP/x"; dpkg-deb -x "$deb" "$TMP/x"
  mkdir -p "$OUT/$p"
  for s in 1 5 8; do
    [ -d "$TMP/x/usr/share/man/man$s" ] && cp -a "$TMP/x/usr/share/man/man$s" "$OUT/$p/" || true
  done
  cp "$TMP/x/usr/share/doc/$p/copyright" "$LIC/$p-copyright" 2>/dev/null || true
  echo "$p,$v,http://archive.ubuntu.com/ubuntu/$f,$(sha256sum "$deb" | cut -d' ' -f1),licenses/command-manpages/$p-copyright,$NOW" >> "$M"
done
find "$OUT" -type d -empty -delete
echo "Done: $(find "$OUT" -type f -o -type l | wc -l) man pages from $(($(wc -l < "$M")-1)) packages in $OUT"
