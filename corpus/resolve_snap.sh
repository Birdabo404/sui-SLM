# Sets SNAP to the newest $SCRIPT_DIR/snapshot-* folder unless SNAP is already set.
# SCRIPT_DIR must be set by the caller. Safe to source.
if [ -z "${SNAP:-}" ]; then
  newest=""
  for d in "$SCRIPT_DIR"/snapshot-*; do
    if [ -d "$d" ]; then
      newest="$d"
    fi
  done
  if [ -z "$newest" ]; then
    echo "No snapshot-* folder in $SCRIPT_DIR. Run bash corpus/fetch_corpus.sh first." >&2
    exit 1
  fi
  SNAP="$newest"
  echo "Using snapshot: $SNAP" >&2
fi
