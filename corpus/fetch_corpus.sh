#!/usr/bin/env bash
# Downloads dated snapshots of the thesis corpus sources + their license files,
# and writes MANIFEST.csv (source, version, url, sha256, license, fetched_at).
# Run from anywhere:  bash corpus/fetch_corpus.sh   (needs curl, unzip, xz, zstd, dpkg-deb)
# Override versions:  UBUNTU_DOCS=24.04.2 bash corpus/fetch_corpus.sh
set -euo pipefail
for t in curl unzip xz zstd dpkg-deb sha256sum; do command -v $t >/dev/null || { echo "Missing $t. Run: sudo apt install curl unzip xz-utils zstd"; exit 1; }; done
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UBUNTU_DOCS="${UBUNTU_DOCS:-26.04.1}"          # match the CCIS lab's Ubuntu release
# Ubuntu Server docs have no releases/tags; pin a commit of canonical/ubuntu-server-documentation (main)
USD_COMMIT="${USD_COMMIT:-9c2b8ca3a6db4f73fb93c257be198408b7ab2542}"   # 2026-10-12T09:15:13Z
SKIP_ARCH="${SKIP_ARCH:-1}"                   # Arch Wiki dropped from the corpus 2026-10-12; SKIP_ARCH=0 to fetch it anyway
OUT="${OUT:-$SCRIPT_DIR/snapshot-$(date +%Y-%m-%d)}"
mkdir -p "$OUT"/{downloads,tldr,man-pages,arch-wiki,ubuntu-docs,ubuntu-server-docs,licenses}
cd "$OUT"
NOW="$(date -Iseconds)"
echo "source,version,url,sha256,license,fetched_at" > MANIFEST.csv
get() { # name version url file license
  curl -fsSL --retry 3 -o "downloads/$4" "$3"
  echo "$1,$2,$3,$(sha256sum "downloads/$4" | cut -d' ' -f1),$5,$NOW" >> MANIFEST.csv
}

# 1) tldr-pages (English, latest release) - CC BY 4.0
TLDR_TAG="$(curl -fsIL -o /dev/null -w '%{url_effective}' https://github.com/tldr-pages/tldr/releases/latest | sed 's#.*/tag/##')"
get tldr-pages "$TLDR_TAG" "https://github.com/tldr-pages/tldr/releases/download/$TLDR_TAG/tldr-pages.en.zip" tldr.en.zip "CC-BY-4.0"
unzip -q downloads/tldr.en.zip -d tldr
curl -fsL -o licenses/tldr-LICENSE.md "https://raw.githubusercontent.com/tldr-pages/tldr/$TLDR_TAG/LICENSE.md"

# 2) Linux man-pages project (latest) - per-page licenses, see LICENSES/ in tarball
MP="$(curl -fsL https://mirrors.edge.kernel.org/pub/linux/docs/man-pages/ | grep -oE 'man-pages-[0-9.]+\.tar\.xz' | sort -uV | tail -1)"
get man-pages "$(echo "$MP" | sed -E "s/man-pages-(.*)\.tar\.xz/\1/")" "https://mirrors.edge.kernel.org/pub/linux/docs/man-pages/$MP" "$MP" "mixed (see LICENSES/)"
tar -xJf "downloads/$MP" -C man-pages --strip-components=1
cp -r man-pages/LICENSES licenses/man-pages-LICENSES 2>/dev/null || true

# 3) Arch Wiki offline HTML dump (arch-wiki-docs package) - GFDL 1.3 or later  (DROPPED 2026-10-12, kept for reproducing the old index)
if [ "$SKIP_ARCH" = 0 ]; then
AW="$(curl -fsL https://geo.mirror.pkgbuild.com/extra/os/x86_64/ | grep -oE 'arch-wiki-docs-[0-9]+-[0-9]+-any\.pkg\.tar\.zst' | sort -uV | tail -1)"
get arch-wiki "$(echo "$AW" | grep -oE '[0-9]{8}')" "https://geo.mirror.pkgbuild.com/extra/os/x86_64/$AW" "$AW" "GFDL-1.3-or-later"
tar --zstd -xf "downloads/$AW" -C arch-wiki usr/share/doc/arch-wiki/html
curl -fsL -o licenses/arch-wiki-GFDL-1.3.txt https://www.gnu.org/licenses/fdl-1.3.txt
fi

# 4) Ubuntu desktop help (ubuntu-docs package) - CC BY-SA 3.0
DEB="ubuntu-docs_${UBUNTU_DOCS}_all.deb"
get ubuntu-docs "$UBUNTU_DOCS" "http://archive.ubuntu.com/ubuntu/pool/main/u/ubuntu-docs/$DEB" "$DEB" "CC-BY-SA-3.0"
dpkg-deb -x "downloads/$DEB" ubuntu-docs
find ubuntu-docs -iname 'copyright' -exec cp {} licenses/ubuntu-docs-copyright \; -quit

# 5) Ubuntu Server documentation (Sphinx/MyST markdown source, English only), pinned commit.
#    The repo has no LICENSE file; the CC BY-SA 3.0 statement is the user's decision / Canonical's
#    docs licence -- see licenses/ubuntu-server-docs-LICENSE.txt for what was (not) found in the source.
get ubuntu-server-docs "${USD_COMMIT:0:12}" "https://github.com/canonical/ubuntu-server-documentation/archive/$USD_COMMIT.tar.gz" "ubuntu-server-documentation-$USD_COMMIT.tar.gz" "CC-BY-SA-3.0 (see licenses/ubuntu-server-docs-LICENSE.txt)"
tar -xzf "downloads/ubuntu-server-documentation-$USD_COMMIT.tar.gz" -C ubuntu-server-docs --strip-components=1 \
    --wildcards '*/docs/tutorial/*' '*/docs/how-to/*' '*/docs/explanation/*' '*/docs/reference/*' '*/docs/index.md' '*/docs/conf.py' '*/README.md'
find ubuntu-server-docs -type f ! -name '*.md' ! -name '*.rst' ! -name conf.py -delete
curl -fsL -o licenses/CC-BY-SA-3.0-legalcode.txt https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt || true

echo; cat MANIFEST.csv; echo; du -sh tldr man-pages arch-wiki ubuntu-docs ubuntu-server-docs
echo "Done: $(pwd). Keep downloads/ + MANIFEST.csv unchanged; cite these versions in Ch3."
