#!/usr/bin/env bash
# Ubuntu setup for the offline lab assistant. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

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
# CPU torch first so the later -r install does not replace it with a CUDA wheel (8G laptop).
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$ROOT/requirements.txt"

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

Build the index (CPU, several minutes on the T14). Default sources already skip Arch Wiki and include Ubuntu Server docs:
  . .venv/bin/activate
  python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index
  python rag/trim/make_trimmed_index.py --src rag/rag-index --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index-v2

Start Ollama if it is not running, then:
  streamlit run app/app.py
EOF
