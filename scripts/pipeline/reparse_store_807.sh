#!/bin/bash
#
# THE #807 ONE-PASS STORE REPARSE — build aside, verify, swap, measure.
#
# Why: the live store's ast_json blobs + shredded columns predate the current
# parser (#871 recovered ~182k subjects; #821/#818 fixed). No parser fix
# reaches retrieval until this regenerates the blobs. The #823 junk gate
# (klareco.corpus_quality) runs INSIDE build_duckdb_store.py, so junk is
# never parsed. The number that must move: deterministic-floor MRR /
# answer_accuracy on rebaseline_500, before vs after (paired bootstrap CI).
#
# The store is built to SIDE paths (duckdb_store_new.db / whoosh_v2_new),
# finalized and gate-checked there, and only then swapped live — the live
# store keeps serving until the swap, and the old store/index are kept as
# *_pre807 backups.
#
# Stages (each leaves a marker in data/staging/reparse_807/; rerunning the
# script skips completed stages — safe to interrupt and restart):
#   1 bench_before   rebaseline_500 bench vs the LIVE store        (~20 min)
#   2 build_store    reparse all sentences -> side store + Whoosh  (~2-5 h)
#   3 copy_ontology  carry hand-fixed ontology + ALIASO bridge     (~1 min)
#   4 clauses        rebuild clause table from new ASTs            (~1 h)
#   5 verb_klaso     populate clauses.verb_klaso (lexicon)         (~min)
#   6 dep_arcs       rebuild dependency_arcs from new ASTs         (~1-2 h)
#   7 entity_facts   re-extract entity facts from new store        (~1 h)
#   8 verify         hard gates: junk=0, subjects>=live, ontology  (~min)
#   9 swap           live -> *_pre807, side -> live, preflight     (~min)
#  10 bench_after    same bench vs the NEW live store              (~20 min)
#  11 report         paired before/after -> bench_history.jsonl    (~min)
#
# Usage:
#   ./scripts/pipeline/reparse_store_807.sh            # run / resume
#   ./scripts/pipeline/reparse_store_807.sh --fresh    # discard side store + markers, start over
#
# Run it in its own terminal (this is an overnight job):
#   ./scripts/pipeline/reparse_store_807.sh 2>&1 | tail -f is unnecessary — it tees itself.
# Monitor: tail -f logs/reparse807_*.log
#
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# --- venv ---------------------------------------------------------------
if [ -d ".venv" ]; then source .venv/bin/activate
elif [ -d "venv" ]; then source venv/bin/activate
else echo "❌ no venv found"; exit 1; fi

# --- paths --------------------------------------------------------------
LIVE_DB="data/indexes/duckdb_store.db"
LIVE_WHOOSH="data/indexes/whoosh_v2"
SIDE_DB="data/indexes/duckdb_store_new.db"
SIDE_WHOOSH="data/indexes/whoosh_v2_new"
BACKUP_DB="data/indexes/duckdb_store_pre807.db"
BACKUP_WHOOSH="data/indexes/whoosh_v2_pre807"
TEST_SET="data/test_sets/rebaseline_500.jsonl"
STATE="data/staging/reparse_807"
RESULTS="results"
BEFORE_ROWS="$RESULTS/reparse807_before_rows.jsonl"
BEFORE_SUM="$RESULTS/reparse807_before_summary.json"
AFTER_ROWS="$RESULTS/reparse807_after_rows.jsonl"
AFTER_SUM="$RESULTS/reparse807_after_summary.json"

mkdir -p logs "$STATE" "$RESULTS"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/reparse807_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- flags --------------------------------------------------------------
if [[ "$1" == "--fresh" ]]; then
    log "--fresh: removing side store, side whoosh, and stage markers"
    rm -rf "$STATE"; mkdir -p "$STATE"
    rm -f "$SIDE_DB" "$SIDE_DB.wal"
    rm -rf "$SIDE_WHOOSH"
fi

# --- sanity + disk preflight -------------------------------------------
[ -f "$TEST_SET" ] || { echo "❌ missing $TEST_SET"; exit 1; }
[ -f "$LIVE_DB" ]  || { echo "❌ missing $LIVE_DB"; exit 1; }
if [ -f "$BACKUP_DB" ] || [ -d "$BACKUP_WHOOSH" ]; then
    echo "❌ *_pre807 backups already exist — a previous run swapped."
    echo "   Inspect/remove $BACKUP_DB and $BACKUP_WHOOSH first."
    exit 1
fi
# side store ~44 GB + whoosh segments (2-5x of 3 GB) + slack
"$PROJECT_ROOT/scripts/util/preflight_disk.sh" 80 "reparse builds a full side store (#807)" || exit 1

done_stage() { [ -f "$STATE/$1.done" ]; }
mark_stage() { touch "$STATE/$1.done"; log "STAGE $1 done"; }

log "=== #807 REPARSE START (log: $LOG) ==="

# --- 1 bench_before (MUST run before the swap) --------------------------
if ! done_stage bench_before; then
    log "STAGE 1/11 bench_before: rebaseline_500 vs live store"
    python scripts/eval/multi_reranker_bench.py \
        --test-set "$TEST_SET" \
        --duckdb-path "$LIVE_DB" --whoosh-dir "$LIVE_WHOOSH" \
        --output-jsonl "$BEFORE_ROWS" --output-summary "$BEFORE_SUM"
    mark_stage bench_before
fi

# --- 2 build_store ------------------------------------------------------
if ! done_stage build_store; then
    RESUME_FLAG=""
    [ -f "$SIDE_DB" ] && RESUME_FLAG="--resume" && log "  side store exists — resuming"
    log "STAGE 2/11 build_store: reparse -> $SIDE_DB + $SIDE_WHOOSH $RESUME_FLAG"
    KLARECO_DUCK="$SIDE_DB" KLARECO_WHOOSH="$SIDE_WHOOSH" \
        python scripts/index/build_duckdb_store.py $RESUME_FLAG
    mark_stage build_store
fi

# --- 3 copy_ontology ----------------------------------------------------
if ! done_stage copy_ontology; then
    log "STAGE 3/11 copy_ontology: carry #713 hand-fix + #872 ALIASO from live"
    python scripts/index/finalize_reparsed_store.py \
        --new-db "$SIDE_DB" --live-db "$LIVE_DB" --copy-ontology
    mark_stage copy_ontology
fi

# --- 4 clauses ----------------------------------------------------------
if ! done_stage clauses; then
    log "STAGE 4/11 clauses: rebuild clause table from new ASTs"
    python scripts/index/build_clause_table.py --duckdb-path "$SIDE_DB"
    mark_stage clauses
fi

# --- 5 verb_klaso -------------------------------------------------------
if ! done_stage verb_klaso; then
    log "STAGE 5/11 verb_klaso: populate from typed root lexicon"
    python scripts/index/finalize_reparsed_store.py \
        --new-db "$SIDE_DB" --live-db "$LIVE_DB" --verb-klaso
    mark_stage verb_klaso
fi

# --- 6 dep_arcs ---------------------------------------------------------
if ! done_stage dep_arcs; then
    log "STAGE 6/11 dep_arcs: rebuild dependency_arcs from new ASTs"
    python scripts/index/build_dependency_arcs.py \
        --duckdb-path "$SIDE_DB" --apply --fresh
    mark_stage dep_arcs
fi

# --- 7 entity_facts -----------------------------------------------------
if ! done_stage entity_facts; then
    log "STAGE 7/11 entity_facts: re-extract from new store"
    python scripts/index/extract_entity_facts.py \
        --duckdb-path "$SIDE_DB" --apply --fresh
    mark_stage entity_facts
fi

# --- 8 verify -----------------------------------------------------------
if ! done_stage verify; then
    log "STAGE 8/11 verify: hard gates on the side store"
    python scripts/index/finalize_reparsed_store.py \
        --new-db "$SIDE_DB" --live-db "$LIVE_DB" --verify
    python scripts/index/validate_duckdb_store.py --duckdb-path "$SIDE_DB"
    mark_stage verify
fi

# --- 9 swap -------------------------------------------------------------
if ! done_stage swap; then
    log "STAGE 9/11 swap: live -> *_pre807 backups, side -> live"
    mv "$LIVE_DB" "$BACKUP_DB"
    [ -f "$LIVE_DB.wal" ] && mv "$LIVE_DB.wal" "$BACKUP_DB.wal"
    mv "$SIDE_DB" "$LIVE_DB"
    [ -f "$SIDE_DB.wal" ] && mv "$SIDE_DB.wal" "$LIVE_DB.wal"
    mv "$LIVE_WHOOSH" "$BACKUP_WHOOSH"
    mv "$SIDE_WHOOSH" "$LIVE_WHOOSH"
    log "  running klareco.preflight on the swapped-in store (fails loudly)"
    if ! python -m klareco.preflight; then
        log "  !!! preflight FAILED — rolling the swap back"
        mv "$LIVE_DB" "$SIDE_DB"
        [ -f "$LIVE_DB.wal" ] && mv "$LIVE_DB.wal" "$SIDE_DB.wal"
        mv "$BACKUP_DB" "$LIVE_DB"
        [ -f "$BACKUP_DB.wal" ] && mv "$BACKUP_DB.wal" "$LIVE_DB.wal"
        mv "$LIVE_WHOOSH" "$SIDE_WHOOSH"
        mv "$BACKUP_WHOOSH" "$LIVE_WHOOSH"
        log "  rollback complete — live store restored; investigate before re-running"
        exit 2
    fi
    mark_stage swap
fi

# --- 10 bench_after -----------------------------------------------------
if ! done_stage bench_after; then
    log "STAGE 10/11 bench_after: rebaseline_500 vs the reparsed store"
    python scripts/eval/multi_reranker_bench.py \
        --test-set "$TEST_SET" \
        --duckdb-path "$LIVE_DB" --whoosh-dir "$LIVE_WHOOSH" \
        --output-jsonl "$AFTER_ROWS" --output-summary "$AFTER_SUM"
    mark_stage bench_after
fi

# --- 11 report ----------------------------------------------------------
if ! done_stage report; then
    log "STAGE 11/11 report: paired before/after comparison"
    python scripts/eval/compare_reparse_bench.py \
        --before-rows "$BEFORE_ROWS" --after-rows "$AFTER_ROWS" \
        --label "#807 store reparse (one pass: current parser + #823 gate; #871 subjects reach retrieval)" \
        --append-history data/perf/bench_history.jsonl
    mark_stage report
else
    log "STAGE 11/11 report: already appended — re-printing without history append"
    python scripts/eval/compare_reparse_bench.py \
        --before-rows "$BEFORE_ROWS" --after-rows "$AFTER_ROWS"
fi

log "=== #807 REPARSE COMPLETE ==="
log "Old store kept as: $BACKUP_DB / $BACKUP_WHOOSH (delete after review)"
log "Next: post the before/after table to #807; close #818/#821 (store-stale symptoms)"
