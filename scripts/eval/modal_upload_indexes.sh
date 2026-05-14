#!/bin/bash
# One-time upload of Whoosh FTS + Kuzu indexes to the Modal Volume
# `klareco-indexes`. Re-run after rebuilding either index.
#
# Sizes (approx, as of 2026-04-28):
#   Whoosh FTS  17 GB  (data/indexes/whoosh_fts/)
#   Kuzu DB    9.9 GB  (data/indexes/v2.1_kuzu_index_full)
#
# Total upload time depends on bandwidth (~10-25 min on a typical link).
#
# Usage:
#   ./scripts/eval/modal_upload_indexes.sh           # upload both
#   ./scripts/eval/modal_upload_indexes.sh kuzu      # only Kuzu
#   ./scripts/eval/modal_upload_indexes.sh whoosh    # only Whoosh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VOLUME="klareco-indexes"
WHICH="${1:-both}"

WHOOSH_LOCAL="data/indexes/whoosh_fts"
KUZU_LOCAL="data/indexes/v2.1_kuzu_index_full"
WHOOSH_REMOTE="/whoosh_fts"
KUZU_REMOTE="/v2.1_kuzu_index_full"

if ! command -v modal >/dev/null; then
  echo "ERROR: 'modal' CLI not found. pip install modal && modal token new" >&2
  exit 1
fi

echo "Ensuring Modal volume '$VOLUME' exists..."
modal volume create "$VOLUME" 2>/dev/null || echo "  (already exists)"

upload_whoosh() {
  if [ ! -d "$WHOOSH_LOCAL" ]; then
    echo "ERROR: $WHOOSH_LOCAL not found" >&2; exit 1
  fi
  echo "Uploading Whoosh FTS index ($WHOOSH_LOCAL -> $VOLUME:$WHOOSH_REMOTE)..."
  modal volume put "$VOLUME" "$WHOOSH_LOCAL" "$WHOOSH_REMOTE" --force
}

upload_kuzu() {
  if [ ! -f "$KUZU_LOCAL" ]; then
    echo "ERROR: $KUZU_LOCAL not found (expected single Kuzu DB file)" >&2; exit 1
  fi
  echo "Uploading Kuzu DB ($KUZU_LOCAL -> $VOLUME:$KUZU_REMOTE)..."
  modal volume put "$VOLUME" "$KUZU_LOCAL" "$KUZU_REMOTE" --force
}

case "$WHICH" in
  both)   upload_whoosh; upload_kuzu ;;
  whoosh) upload_whoosh ;;
  kuzu)   upload_kuzu ;;
  *)      echo "Unknown target: $WHICH (use both|whoosh|kuzu)" >&2; exit 1 ;;
esac

echo
echo "Done. Inspect with:"
echo "  modal volume ls $VOLUME"
echo
echo "Run the eval with:"
echo "  modal run scripts/eval/modal_eval.py \\"
echo "      --test-set data/test_sets/general_knowledge_30_keyed.jsonl \\"
echo "      --output data/eval_results/general_knowledge_30_modal.json"
