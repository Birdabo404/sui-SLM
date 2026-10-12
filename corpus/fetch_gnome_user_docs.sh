#!/usr/bin/env bash
# Adds the English GNOME desktop help (the "Ubuntu Desktop Guide" body) to the corpus.
# Why: ubuntu-docs 26.04.1 only ships 19 Ubuntu-specific English pages (the other 703 .page
# files are the same 19 pages in 37 other languages). The bulk of the desktop help Ubuntu shows
# lives in gnome-user-docs (CC-BY-SA 3.0). Keeps only the English (C) gnome-help pages + copyright.
# Additive: writes gnome-user-docs/ and MANIFEST-gnome-user-docs.csv; touches nothing else.
# Run from anywhere:  bash corpus/fetch_gnome_user_docs.sh
set -euo pipefail
command -v dpkg-deb >/dev/null || { echo "Missing dpkg-deb (sudo apt install dpkg)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE="${SUITE:-resolute}"; SNAP="${SNAP:-$SCRIPT_DIR/snapshot-2026-10-11}"
OUT="$SNAP/gnome-user-docs"; DL="$SNAP/downloads"; LIC="$SNAP/licenses"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL --retry 3 "http://archive.ubuntu.com/ubuntu/dists/$SUITE/main/binary-amd64/Packages.xz" \
 | python3 -c "import sys,lzma;sys.stdout.buffer.write(lzma.decompress(sys.stdin.buffer.read()))" > "$TMP/main"
read -r f v < <(awk '$1=="Package:"{x=($2=="gnome-user-docs")} x&&$1=="Version:"{v=$2} x&&$1=="Filename:"{print $2, v; exit}' "$TMP/main")
deb="$DL/$(basename "$f")"
[ -s "$deb" ] || curl -fsSL --retry 3 -o "$deb" "http://archive.ubuntu.com/ubuntu/$f"
dpkg-deb -x "$deb" "$TMP/x"
mkdir -p "$OUT"; cp -a "$TMP/x/usr/share/help/C/gnome-help" "$OUT/"
cp "$TMP/x/usr/share/doc/gnome-user-docs/copyright" "$LIC/gnome-user-docs-copyright"
echo "source,version,url,sha256,license,fetched_at" > "$SNAP/MANIFEST-gnome-user-docs.csv"
echo "gnome-user-docs,$v,http://archive.ubuntu.com/ubuntu/$f,$(sha256sum "$deb" | cut -d' ' -f1),CC-BY-SA-3.0 (see licenses/gnome-user-docs-copyright),$(date -Iseconds)" >> "$SNAP/MANIFEST-gnome-user-docs.csv"
echo "Done: $(find "$OUT" -name '*.page' | wc -l) English pages in $OUT"
