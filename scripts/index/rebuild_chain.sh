#!/bin/bash
#
# Chain: wait for Stage 1 reparse → build DuckDB + Whoosh →
# build all secondary indexes → bench AST-vs-BM25 retrieval.
#
# Designed to run unattended. Each stage gates on the previous
# producing the expected output and a sanity row-count check.
#
# Usage:
#   nohup ./scripts/index/rebuild_chain.sh <STAGE_1_PID> > logs/chain.log 2>&1 &
#
# After Stage 1 PID exits cleanly:
#   Stage 2: ./scripts/index/build_duckdb_resilient.sh 4
#   Stage 3: build entity_postings, pattern_kb, verb_klaso, verb_negated
#   Stage 4: verify row counts, sample queries
#   Stage 5: bench all rerankers + retrievers on capability_candidates_v1
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d .venv ]; then source .venv/bin/activate
elif [ -d venv  ]; then source venv/bin/activate
else echo "no venv"; exit 1; fi

STAGE1_PID="${1:-}"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/chain
CHAIN_LOG="logs/chain/chain_${TS}.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

# Test set used by Stage 5 benchmark
TEST_SET="${TEST_SET:-data/test_sets/capability_candidates_v1.jsonl}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
err() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }

# ---------- Stage 1: WAIT ----------
log "=== Stage 1 wait: PID=$STAGE1_PID ==="
if [ -z "$STAGE1_PID" ]; then
  log "  No PID given — proceeding immediately (assuming Stage 1 already done)."
else
  while kill -0 "$STAGE1_PID" 2>/dev/null; do
    sleep 60
  done
  log "  Stage 1 PID $STAGE1_PID exited."
fi

# Verify Stage 1 output exists and is non-trivial
NEW_CORPUS="data/enhanced_corpus/corpus_with_metadata.jsonl"
if [ ! -s "$NEW_CORPUS" ]; then
  err "Stage 1 output $NEW_CORPUS missing or empty"
  exit 1
fi
N_CORPUS=$(wc -l < "$NEW_CORPUS")
log "  Stage 1 output: $NEW_CORPUS — $N_CORPUS sentences"
if [ "$N_CORPUS" -lt 4000000 ]; then
  err "Corpus only $N_CORPUS rows; expected >= 4M"
  exit 1
fi

# build_duckdb_store.py expects data/corpus/unified_corpus.jsonl by default.
# Symlink the new corpus there.
log "  Symlinking enhanced corpus → data/corpus/unified_corpus.jsonl"
mkdir -p data/corpus
ln -sfv "$(readlink -f "$NEW_CORPUS")" data/corpus/unified_corpus.jsonl

# ---------- Stage 2: build DuckDB + Whoosh ----------
log ""
log "=== Stage 2: build_duckdb_store + Whoosh ==="
T0=$(date +%s)
./scripts/index/build_duckdb_resilient.sh 4
RC=$?
T1=$(date +%s)
log "  Stage 2 elapsed: $((T1-T0))s ($(( (T1-T0) / 60 )) min), rc=$RC"
if [ "$RC" -ne 0 ]; then
  err "Stage 2 failed"; exit 2
fi

DB="data/indexes/duckdb_store.db"
if [ ! -s "$DB" ]; then err "DB not created"; exit 2; fi
N_DB=$(python -c "import duckdb; c=duckdb.connect('$DB',read_only=True); print(c.execute('SELECT COUNT(*) FROM sentences').fetchone()[0])")
log "  DB sentence count: $N_DB"
if [ "$N_DB" -lt 4000000 ]; then err "DB has only $N_DB rows"; exit 2; fi

# ---------- Stage 3: secondary indexes ----------
log ""
log "=== Stage 3: secondary indexes ==="

run_or_skip() {
  local name="$1"; shift
  log "  > $name"
  T0=$(date +%s)
  "$@"
  RC=$?
  T1=$(date +%s)
  log "    $name elapsed $((T1-T0))s, rc=$RC"
  if [ "$RC" -ne 0 ]; then err "Stage 3 ($name) failed"; exit 3; fi
}

run_or_skip "entity_postings"    python scripts/index/build_entity_postings.py
run_or_skip "pattern_kb"         python scripts/index/build_pattern_kb.py
run_or_skip "verb_klaso column"  python scripts/index/add_verb_klaso_column.py
run_or_skip "verb_negated col"   python scripts/index/add_verb_negated_column.py

# Sanity table inventory
log "  Stage 3 table summary:"
python -c "
import duckdb
c = duckdb.connect('$DB', read_only=True)
for t in ['sentences','entity_postings','pattern_capital_of',
          'pattern_currency_of','pattern_founded_year_of',
          'pattern_official_language_of','ontology_nodes','ontology_edges']:
    try:
        n = c.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]
        print(f'  {t:<35s} {n:>10,}')
    except Exception as e:
        print(f'  {t:<35s} MISSING ({e!s:.40})')
"

# ---------- Stage 4: validate ----------
log ""
log "=== Stage 4: validate ==="
if [ -f scripts/index/validate_duckdb_store.py ]; then
  python scripts/index/validate_duckdb_store.py 2>&1 | tail -20
else
  log "  (no validate script; skipping)"
fi

# ---------- Stage 5: benchmark retrievers + rerankers ----------
log ""
log "=== Stage 5: bench AST-vs-BM25 on $TEST_SET ==="
if [ ! -s "$TEST_SET" ]; then
  err "Test set $TEST_SET missing — skipping bench"
  exit 0
fi

mkdir -p results
T0=$(date +%s)

log "  Multi-reranker bench (each AST reranker vs BM25 baseline)…"
python scripts/eval/multi_reranker_bench.py \
    --test-set "$TEST_SET" \
    --top-k 10 --candidate-pool 100 \
    --output-jsonl "results/bench_postrebuild_rerankers_${TS}.jsonl" \
    --output-summary "results/bench_postrebuild_rerankers_${TS}.json" \
    --append-history data/perf/bench_history.jsonl 2>&1 | tail -20

log ""
log "  Head-to-head: BM25 vs ASTRetriever…"
python scripts/eval/compare_retrievers.py \
    --test-set "$TEST_SET" \
    --top-k 10 \
    --append-history data/perf/bench_history.jsonl 2>&1 | tail -25

T1=$(date +%s)
log "  Stage 5 elapsed $((T1-T0))s ($(( (T1-T0) / 60 )) min)"

log ""
log "=== REBUILD CHAIN COMPLETE ==="
log "  Final disk:"
df -h / | head -2
