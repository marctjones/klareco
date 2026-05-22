#!/bin/bash
#
# Finish-rebuild chain: clean Whoosh + Stages 3 → 4 → 5.
#
# DuckDB store is already complete (5,415,600 rows). The killed Whoosh
# optimize left the index contaminated with stale May-18 docs, so we
# rebuild Whoosh alone from DuckDB (no parse, no optimize), gate on
# correctness, then run the remaining secondary indexes + bench.
#
# Stages:
#   1. NUKE + rebuild Whoosh from DuckDB (with hard correctness gates)
#   2. entity_postings, pattern_kb, verb_klaso col, verb_negated col
#   3. validate_duckdb_store
#   4. multi_reranker_bench + compare_retrievers
#
# Each stage exits the chain on failure with a distinct rc.
#
# Usage:
#   nohup setsid ./scripts/index/finish_rebuild_chain.sh > logs/finish_chain.log 2>&1 < /dev/null &
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d .venv ]; then source .venv/bin/activate
elif [ -d venv  ]; then source venv/bin/activate
else echo "no venv"; exit 1; fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/chain
CHAIN_LOG="logs/chain/finish_${TS}.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

TEST_SET="${TEST_SET:-data/test_sets/capability_candidates_v1.jsonl}"
DB="data/indexes/duckdb_store.db"

log() { echo "[$(date +%H:%M:%S)] $*"; }
err() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }

# ---------- Stage 1: clean Whoosh ----------
log "=== Stage 1: nuke + rebuild Whoosh from DuckDB ==="
T0=$(date +%s)
python scripts/index/rebuild_whoosh_from_duckdb.py
RC=$?
T1=$(date +%s)
log "  Stage 1 elapsed $((T1-T0))s ($(( (T1-T0)/60 ))min), rc=$RC"
if [ "$RC" -ne 0 ]; then err "Stage 1 (whoosh rebuild) FAILED"; exit 11; fi

# ---------- Stage 2: secondary indexes ----------
log ""
log "=== Stage 2: secondary indexes ==="
run_or_die() {
  local name="$1"; shift
  log "  > $name"
  T0=$(date +%s)
  "$@"
  RC=$?
  T1=$(date +%s)
  log "    $name elapsed $((T1-T0))s, rc=$RC"
  if [ "$RC" -ne 0 ]; then err "Stage 2 ($name) FAILED"; exit 12; fi
}

run_or_die "entity_postings scan"  python scripts/index/build_entity_postings.py --fresh --scan-only
run_or_die "entity_postings apply" python scripts/index/build_entity_postings.py --apply
run_or_die "pattern_kb scan"       python scripts/index/build_pattern_kb.py --fresh --scan-only
run_or_die "pattern_kb apply"      python scripts/index/build_pattern_kb.py --apply
run_or_die "verb_klaso column"     python scripts/index/add_verb_klaso_column.py --recompute
run_or_die "verb_negated col"      python scripts/index/add_verb_negated_column.py --recompute

log "  Stage 2 table summary:"
python -c "
import duckdb
c = duckdb.connect('$DB', read_only=True)
for t in ['sentences','entity_postings','pattern_capital_of',
          'pattern_currency_of','pattern_founded_year_of',
          'pattern_official_language_of','ontology_nodes','ontology_edges']:
    try:
        n = c.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]
        print(f'    {t:<35s} {n:>10,}')
    except Exception as e:
        print(f'    {t:<35s} MISSING ({e!s:.40})')
"

# ---------- Stage 3: validate ----------
log ""
log "=== Stage 3: validate ==="
if [ -f scripts/index/validate_duckdb_store.py ]; then
  python scripts/index/validate_duckdb_store.py
  RC=$?
  log "  validate rc=$RC"
  if [ "$RC" -ne 0 ]; then err "Stage 3 (validate) FAILED"; exit 13; fi
else
  log "  (no validate script; skipping)"
fi

# ---------- Stage 4: benchmark ----------
log ""
log "=== Stage 4: bench AST-vs-BM25 on $TEST_SET ==="
if [ ! -s "$TEST_SET" ]; then
  err "Test set $TEST_SET missing — skipping bench (still success)"
  log ""
  log "=== FINISH CHAIN COMPLETE (no bench) ==="
  exit 0
fi
mkdir -p results
T0=$(date +%s)

log "  Multi-reranker bench…"
python scripts/eval/multi_reranker_bench.py \
    --test-set "$TEST_SET" \
    --top-k 10 --candidate-pool 100 \
    --output-jsonl "results/bench_postrebuild_rerankers_${TS}.jsonl" \
    --output-summary "results/bench_postrebuild_rerankers_${TS}.json" \
    --append-history data/perf/bench_history.jsonl 2>&1 | tail -30
RC=$?
log "  multi_reranker_bench rc=$RC"

log ""
log "  Head-to-head: BM25 vs ASTRetriever…"
python scripts/eval/compare_retrievers.py \
    --test-set "$TEST_SET" \
    --top-k 10 \
    --append-history data/perf/bench_history.jsonl 2>&1 | tail -30
RC=$?
log "  compare_retrievers rc=$RC"

T1=$(date +%s)
log "  Stage 4 elapsed $((T1-T0))s ($(( (T1-T0)/60 ))min)"

log ""
log "=== FINISH CHAIN COMPLETE ==="
log "  bench outputs: results/bench_postrebuild_rerankers_${TS}.{json,jsonl}"
df -h / | head -2
