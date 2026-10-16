#!/usr/bin/env bash
# Downloads dated snapshots of the thesis corpus sources + their license files,
# and writes MANIFEST.csv (source, version, url, sha256, license, fetched_at).
# Run from anywhere:  bash corpus/fetch_corpus.sh   (needs curl, unzip, xz, zstd, dpkg-deb)
# Override versions:  UBUNTU_DOCS=24.04.2 bash corpus/fetch_corpus.sh
set -euo pipefail
for t in curl unzip xz zstd dpkg-deb sha256sum; do command -v $t >/dev/null || { echo "Missing $t. Run: sudo apt install curl unzip xz-utils zstd"; exit 1; }; done
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UBUNTU_DOCS="${UBUNTU_DOCS:-26.04.1}"          # match the CCIS lab's Ubuntu release
# Ubuntu Server docs: unlicensed (no LICENSE; conf.py TODO; site is a copyright footer only).
# Off by default. INCLUDE_UBUNTU_SERVER=1 to fetch anyway; pin a commit (no tags).
USD_COMMIT="${USD_COMMIT:-9c2b8ca3a6db4f73fb93c257be198408b7ab2542}"   # 2026-10-12T09:15:13Z
INCLUDE_UBUNTU_SERVER="${INCLUDE_UBUNTU_SERVER:-0}"
SKIP_ARCH="${SKIP_ARCH:-1}"                   # Arch Wiki dropped from the corpus 2026-10-12; SKIP_ARCH=0 to fetch it anyway
OUT="${OUT:-$SCRIPT_DIR/snapshot-$(date +%Y-%m-%d)}"
mkdir -p "$OUT"/{downloads,tldr,man-pages,arch-wiki,ubuntu-docs,licenses}
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

# 4) Ubuntu desktop help (ubuntu-docs). Indexed pages are ubuntu-help/*/*.page*.
# License is CC-BY-SA-3.0 in the ubuntu-docs .deb copyright:
# https://changelogs.ubuntu.com/changelogs/pool/main/u/ubuntu-docs/ubuntu-docs_26.04.1/copyright
# help.ubuntu.com/legal.html has said 4.0; we cite the package copyright.
DEB="ubuntu-docs_${UBUNTU_DOCS}_all.deb"
get ubuntu-docs "$UBUNTU_DOCS" "http://archive.ubuntu.com/ubuntu/pool/main/u/ubuntu-docs/$DEB" "$DEB" "CC-BY-SA-3.0"
dpkg-deb -x "downloads/$DEB" ubuntu-docs
find ubuntu-docs -iname 'copyright' -exec cp {} licenses/ubuntu-docs-copyright \; -quit
if [ ! -f licenses/ubuntu-docs-copyright ]; then
  echo "ERROR: ubuntu-docs .deb copyright file was not in the package." >&2
  exit 1
fi
if ! grep -q "CC-BY-SA-3.0" licenses/ubuntu-docs-copyright; then
  echo "ERROR: ubuntu-docs copyright does not say CC-BY-SA-3.0. Check licenses/ubuntu-docs-copyright before labeling the corpus." >&2
  exit 1
fi
if ! curl -fsL --retry 3 -o licenses/CC-BY-SA-3.0-legalcode.txt https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt; then
  echo "WARNING: could not download CC-BY-SA-3.0 legalcode.txt. The .deb copyright is in licenses/ubuntu-docs-copyright." >&2
fi

# 5) Ubuntu Server documentation. Unlicensed. Opt-in only.
if [ "$INCLUDE_UBUNTU_SERVER" = 1 ]; then
  mkdir -p ubuntu-server-docs
  get ubuntu-server-docs "${USD_COMMIT:0:12}" "https://github.com/canonical/ubuntu-server-documentation/archive/$USD_COMMIT.tar.gz" "ubuntu-server-documentation-$USD_COMMIT.tar.gz" "unlicensed (no LICENSE; conf.py TODO)"
  tar -xzf "downloads/ubuntu-server-documentation-$USD_COMMIT.tar.gz" -C ubuntu-server-docs --strip-components=1 \
      --wildcards '*/docs/tutorial/*' '*/docs/how-to/*' '*/docs/explanation/*' '*/docs/reference/*' '*/docs/index.md' '*/docs/conf.py' '*/README.md'
  find ubuntu-server-docs -type f ! -name '*.md' ! -name '*.rst' ! -name conf.py -delete
fi

echo; cat MANIFEST.csv; echo; du -sh tldr man-pages arch-wiki ubuntu-docs
if [ "$INCLUDE_UBUNTU_SERVER" = 1 ]; then du -sh ubuntu-server-docs; fi
echo "Done: $(pwd). Keep downloads/ + MANIFEST.csv unchanged; cite these versions in Ch3."
