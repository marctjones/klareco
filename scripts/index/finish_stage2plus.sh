#!/bin/bash
#
# Stages 2-4 only — used when Stage 1 (Whoosh rebuild) is already known good
# and only the downstream secondary indexes + bench need to run. Mirrors
# finish_rebuild_chain.sh from Stage 2 onward, with correct CLI flags.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

"$PROJECT_ROOT/scripts/util/preflight_disk.sh" 50 "finish_stage2plus runs secondary indexes + validate + bench" || exit 1

if [ -d .venv ]; then source .venv/bin/activate
elif [ -d venv  ]; then source venv/bin/activate
else echo "no venv"; exit 1; fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/chain
CHAIN_LOG="logs/chain/finish_s2_${TS}.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

TEST_SET="${TEST_SET:-data/test_sets/capability_candidates_v1.jsonl}"
DB="data/indexes/duckdb_store.db"

log() { echo "[$(date +%H:%M:%S)] $*"; }
err() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }

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

# ---------- Stage 2: secondary indexes ----------
log "=== Stage 2: secondary indexes ==="
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
  err "Test set $TEST_SET missing — skipping bench"
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
log "  multi_reranker_bench rc=$?"

log ""
log "  Head-to-head: BM25 vs ASTRetriever…"
python scripts/eval/compare_retrievers.py \
    --test-set "$TEST_SET" \
    --top-k 10 \
    --append-history data/perf/bench_history.jsonl 2>&1 | tail -30
log "  compare_retrievers rc=$?"

T1=$(date +%s)
log "  Stage 4 elapsed $((T1-T0))s ($(( (T1-T0)/60 ))min)"

log ""
log "=== STAGE 2+ CHAIN COMPLETE ==="
log "  bench outputs: results/bench_postrebuild_rerankers_${TS}.{json,jsonl}"
df -h / | head -2
