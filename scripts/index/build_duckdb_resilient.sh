#!/bin/bash
# Resilient driver for build_duckdb_store.py on a shared/oversubscribed box.
#
# Why: the build runs on a 30GB machine alongside a ~9GB co-tenant
# process. The kernel global OOM-killer (constraint=CONSTRAINT_NONE) has
# SIGKILL-ed the build twice. The loader is resumable by design
# (start_after = min(duck_max, whoosh_count); DuckDB tail-delete; Whoosh
# unique-id idempotent re-add), and every merge=False flush is durable,
# so an OOM kill is NOT catastrophic — it just costs the work since the
# last 100k flush. This wrapper turns that into monotonic progress:
# run --resume, and if the process dies (nonzero / SIGKILL), wait and
# relaunch from the durable checkpoint. Stops on clean completion.
#
# Usage:  ./scripts/index/build_duckdb_resilient.sh [WORKERS]   (default 4)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if   [ -d .venv ]; then source .venv/bin/activate
elif [ -d venv  ]; then source venv/bin/activate
else echo "no venv"; exit 1; fi

WORKERS="${1:-4}"
LOG=/tmp/duckbuild_console.log
MAX_ATTEMPTS=40
attempt=0

while :; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "[resilient] gave up after $MAX_ATTEMPTS attempts" | tee -a "$LOG"
    exit 1
  fi
  echo "[resilient] attempt $attempt: python build_duckdb_store.py --resume --workers $WORKERS" | tee -a "$LOG"
  python scripts/index/build_duckdb_store.py --resume --workers "$WORKERS" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && grep -q "^.*DONE: .* sentences" "$LOG"; then
    echo "[resilient] build completed cleanly (attempt $attempt)" | tee -a "$LOG"
    exit 0
  fi
  echo "[resilient] process exited rc=$rc (likely OOM SIGKILL); checkpoint is durable, retrying in 45s" | tee -a "$LOG"
  sleep 45
done
