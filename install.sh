#!/usr/bin/env bash
# Ubuntu setup for the offline lab assistant. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MIN_FREE_KB="${SUI_SLM_MIN_FREE_KB:-4194304}"
OFFLINE=0
cd "$ROOT"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash install.sh [--offline|--check-python|--check-disk|--install-launcher]

  --offline           Verify and reuse an existing installation without network access.
  --check-python      Check only the supported Python version.
  --check-disk        Check only the available disk space.
  --install-launcher  Install only the command, desktop file, and icon.
EOF
}

check_python() {
  local py_line py_ver py_mm
  py_line="$(python3 --version 2>&1)" || fail "python3 is missing or failed to run."
  printf '%s\n' "$py_line"
  py_ver="${py_line#Python }"
  py_mm="${py_ver%.*}"
  case "$py_mm" in
    3.12|3.13|3.14) ;;
    *)
      fail "This project supports Python 3.12, 3.13, and 3.14. Found $py_line. Install one of those versions and re-run this script."
      ;;
  esac
}

check_disk() {
  local available_kb available_gib required_gib
  case "$MIN_FREE_KB" in
    ''|*[!0-9]*) fail "SUI_SLM_MIN_FREE_KB must be a whole number of KiB." ;;
  esac
  available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
  case "$available_kb" in
    ''|*[!0-9]*) fail "Could not determine free disk space for $ROOT." ;;
  esac
  available_gib=$((available_kb / 1024 / 1024))
  required_gib=$((MIN_FREE_KB / 1024 / 1024))
  if [ "$available_kb" -lt "$MIN_FREE_KB" ]; then
    fail "At least ${required_gib} GiB free is required; ${available_gib} GiB is available at $ROOT."
  fi
  printf 'Disk check: %s GiB free (minimum %s GiB).\n' "$available_gib" "$required_gib"
}

install_launcher() {
  local data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
  case "$data_home" in
    /*) ;;
    *) data_home="$HOME/.local/share" ;;
  esac
  local launcher="$HOME/.local/bin/sui-slm"
  local applications="$data_home/applications"
  local icons="$data_home/icons/hicolor/scalable/apps"
  local desktop
  mkdir -p "$(dirname "$launcher")" "$applications" "$icons"
  ln -sfn "$ROOT/bin/sui-slm" "$launcher"
  install -m 0644 "$ROOT/app/static/sui-slm.svg" "$icons/sui-slm.svg"
  desktop="$(<"$ROOT/packaging/sui-slm.desktop")"
  desktop="${desktop//@LAUNCHER@/$launcher}"
  printf '%s\n' "$desktop" > "$applications/.sui-slm.desktop.tmp"
  chmod 0644 "$applications/.sui-slm.desktop.tmp"
  mv -f "$applications/.sui-slm.desktop.tmp" "$applications/sui-slm.desktop"
  case ":${PATH:-}:" in
    *":$HOME/.local/bin:"*) ;;
    *)
      printf 'Warning: %s is not on PATH.\n' "$HOME/.local/bin" >&2
      printf 'Add this line to ~/.profile, then open a new terminal:\n' >&2
      printf '%s\n' "  export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
      ;;
  esac
}

model_installed() {
  local model="$1"
  ollama list | awk -v model="$model" '
    NR > 1 && ($1 == model || (index(model, ":") == 0 && $1 == model ":latest")) {
      found = 1
    }
    END { exit found ? 0 : 1 }
  '
}

ensure_ollama_model() {
  local model="$1"
  local action="$2"
  local model_file="${3:-}"
  if model_installed "$model"; then
    printf 'Ollama model check: %s is installed.\n' "$model"
    return
  fi
  case "$action" in
    pull)
      if [ "$OFFLINE" -eq 1 ]; then
        fail "Ollama model $model is missing. Re-run without --offline once to pull it."
      fi
      printf 'Pulling missing Ollama model %s.\n' "$model"
      ollama pull "$model"
      ;;
    create)
      if [ -z "$model_file" ] || [ ! -f "$model_file" ]; then
        fail "Ollama model $model is missing. Set OLLAMA_DISTILLED_MODELFILE to its Modelfile and re-run."
      fi
      printf 'Creating missing Ollama model %s from %s.\n' "$model" "$model_file"
      ollama create "$model" -f "$model_file"
      ;;
    *) fail "Unknown model action: $action." ;;
  esac
  model_installed "$model" || fail "Ollama finished but model $model is still missing."
}

check_python

case "${1:-}" in
  '') ;;
  --offline) OFFLINE=1 ;;
  --check-python) exit 0 ;;
  --check-disk)
    check_disk
    exit 0
    ;;
  --install-launcher)
    install_launcher
    exit 0
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    fail "Unknown option: $1"
    ;;
esac

check_disk

if ! command -v ollama >/dev/null 2>&1; then
  fail "Ollama is not installed. Install it from https://ollama.com/download/linux, then re-run this script."
fi
if ! ollama list >/dev/null 2>&1; then
  fail "Ollama is installed but not running. Start it with 'ollama serve', then re-run this script."
fi

apt_needed=()
for pkg in python3-venv python3-pip groff; do
  dpkg -s "$pkg" >/dev/null 2>&1 || apt_needed+=("$pkg")
done
if [ "${#apt_needed[@]}" -gt 0 ]; then
  if [ "$OFFLINE" -eq 1 ]; then
    fail "Missing system packages: ${apt_needed[*]}. Install them before using --offline."
  fi
  printf 'Installing system packages: %s\n' "${apt_needed[*]}"
  sudo apt-get update
  sudo apt-get install -y "${apt_needed[@]}"
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  if [ "$OFFLINE" -eq 1 ]; then
    fail "No existing .venv was found. Run this installer once without --offline."
  fi
  printf 'Creating %s/.venv.\n' "$ROOT"
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
. "$ROOT/.venv/bin/activate"

read -r requirements_hash _ < <(sha256sum "$ROOT/requirements.txt")
requirements_stamp="$ROOT/.venv/.sui-slm-requirements.sha256"
needs_packages=1
if [ -f "$requirements_stamp" ] &&
   [ "$(<"$requirements_stamp")" = "$requirements_hash" ] &&
   python -m pip check >/dev/null 2>&1; then
  needs_packages=0
fi

if [ "$OFFLINE" -eq 1 ]; then
  [ "$needs_packages" -eq 0 ] ||
    fail "The virtual environment is missing or out of date. Re-run without --offline once."
  printf 'Dependency check: reusing the pinned virtual environment.\n'
else
  if [ "$needs_packages" -eq 1 ]; then
    python -m pip install --upgrade pip
    export ANONYMIZED_TELEMETRY=False
    # CPU torch first so the later -r install cannot replace it with a CUDA wheel.
    python -m pip install torch==2.9.1 --index-url "$CPU_INDEX"
    python -m pip install -r "$ROOT/requirements.txt" --extra-index-url "$CPU_INDEX"
  else
    printf 'Dependency check: requirements are unchanged and healthy.\n'
  fi
fi

python -m pip check
python -c "import bs4, chromadb, lxml, psutil, rank_bm25, requests, sentence_transformers, streamlit, torch"
python -c "import torch; assert torch.version.cuda is None, 'CUDA-enabled torch is not supported on this CPU-only install'"
printf '%s\n' "$requirements_hash" > "$requirements_stamp"

if [ "$OFFLINE" -eq 1 ]; then
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBED_MODEL', device='cpu', local_files_only=True)" ||
    fail "Embedding model $EMBED_MODEL is not available in the local cache. Re-run without --offline once."
else
  # Cache MiniLM so later app runs do not need the network.
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBED_MODEL', device='cpu')" ||
    fail "Could not cache embedding model $EMBED_MODEL."
fi
printf 'Embedding model check: %s is cached.\n' "$EMBED_MODEL"

base_model="${OLLAMA_MODEL:-qwen3.5:2b}"
ensure_ollama_model "$base_model" pull
student_model="${OLLAMA_DISTILLED_MODEL:-thesis-student}"
ensure_ollama_model \
  "$student_model" \
  create \
  "${OLLAMA_DISTILLED_MODELFILE:-$ROOT/Modelfile}"

install_launcher

cat <<'EOF'

Setup is done.

Fetch the corpus (needs the network, once). The man-page and GNOME scripts use the newest snapshot-* folder:
  bash corpus/fetch_corpus.sh
  bash corpus/fetch_command_manpages.sh
  bash corpus/fetch_gnome_user_docs.sh

Build the index (CPU, several minutes on the T14). Default sources already skip Arch Wiki and include Ubuntu Server docs.
chromadb 1.0.8 cannot read an index built with chromadb 0.6.3. Rebuild; do not reuse an old rag-index:
  . .venv/bin/activate
  python rag/build_index.py --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index
  python rag/trim/make_trimmed_index.py --src rag/rag-index --snap corpus/snapshot-YYYY-MM-DD --out rag/rag-index-v2

If thesis-student is missing, set OLLAMA_DISTILLED_MODELFILE to its exported Modelfile before re-running.
Start Ollama if it is not running, then:
  sui-slm
EOF
