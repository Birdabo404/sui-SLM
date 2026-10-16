#!/usr/bin/env bash
# Ubuntu setup for the offline lab assistant. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Print Python version first; fail before apt/pip if it is outside 3.12-3.14.
py_line="$(python3 --version 2>&1)" || { echo "python3 is missing or failed to run." >&2; exit 1; }
echo "$py_line"
py_ver="${py_line#Python }"
py_mm="${py_ver%.*}"
case "$py_mm" in
  3.12|3.13|3.14) ;;
  *)
    echo "This project supports Python 3.12, 3.13, and 3.14. Found $py_line." >&2
    echo "Install one of those versions and re-run this script." >&2
    exit 1
    ;;
esac
if [ "${1:-}" = --check-python ]; then
  exit 0
fi

apt_needed=()
for pkg in python3-venv python3-pip groff; do
  dpkg -s "$pkg" >/dev/null 2>&1 || apt_needed+=("$pkg")
done
if [ "${#apt_needed[@]}" -gt 0 ]; then
  sudo apt-get update
  sudo apt-get install -y "${apt_needed[@]}"
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
. "$ROOT/.venv/bin/activate"

python -m pip install --upgrade pip
export ANONYMIZED_TELEMETRY=False
CPU_INDEX=https://download.pytorch.org/whl/cpu
# CPU torch first so the later -r install does not replace it with a CUDA wheel (8G laptop).
python -m pip install torch==2.9.1 --index-url "$CPU_INDEX"
python -m pip install -r "$ROOT/requirements.txt" --extra-index-url "$CPU_INDEX"

# Cache MiniLM so the app does not need the network later.
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download/linux then re-run this script."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "Ollama is installed but not running. Start it with: ollama serve"
  echo "Then re-run this script to pull qwen3.5:2b."
  exit 1
fi
ollama pull qwen3.5:2b

cat <<'EOF'

Setup is done.

Fetch the corpus (needs the network, once). The man-page and GNOME scripts use the newest snapshot-* folder:
  bash corpus/fetch_corpus.sh
  bash corpus/fetch_command_manpages.sh
  bash corpus/fetch_gnome_user_docs.sh

Build the index (CPU, several minutes on the T14). Default sources skip Arch Wiki and Ubuntu Server
docs (Server docs have no license). Rebuild on the laptop into rag/lab-index. Do not reuse rag-index
or rag-index-v2 (v2 still has Server-docs chunks). chromadb 1.0.8 also cannot read an index built
with chromadb 0.6.3:
  . .venv/bin/activate
  python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/lab-index

Start Ollama if it is not running, then:
  streamlit run app/app.py
EOF
