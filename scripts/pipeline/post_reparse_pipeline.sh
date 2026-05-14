#!/bin/bash
# Chained pipeline that runs AFTER corpus_to_csv + load_csv_to_kuzu complete:
#   1. Load semantic ontology (~5-10 min)
#   2. Load ReVo dictionary relationships (~5-10 min)
#   3. Rebuild Whoosh index (~few min)
#   4. Rebuild synthetic test set (uses new propranoma_kategorio coverage)
#   5. Run eval at top_k=100 against new Kuzu
#   6. Compare with prior baseline
#
# Usage:
#   ./scripts/pipeline/post_reparse_pipeline.sh
#
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then source .venv/bin/activate; elif [ -d "venv" ]; then source venv/bin/activate; fi

LOG_DIR="logs/post_reparse"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/post_reparse_${TS}.log"
EVAL_OUT="data/eval_results/post_reparse_${TS}.json"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

log "=== POST-REPARSE PIPELINE START ==="

# Step 1: semantic ontology
log "STEP 1/5: Loading semantic ontology..."
python scripts/index/extend_kuzu_schema_semantic_ontology.py 2>&1 | tee -a "$MASTER_LOG"
log "  semantic ontology done"

# Step 2: ReVo (optional — skip if input not available)
log "STEP 2/5: Loading ReVo relationships..."
if [ -f "data/raw/eo/dictionaries/revo/revo_relations.json" ] || [ -f "data/revo/revo_semantic_relations.json" ]; then
    python scripts/index/load_revo_to_kuzu.py 2>&1 | tee -a "$MASTER_LOG" || log "  ReVo load failed — continuing"
else
    log "  ReVo input not found — skipping"
fi

# Step 3: Whoosh index rebuild (uses corpus directly, but better to refresh)
log "STEP 3/5: Rebuilding Whoosh index..."
python scripts/index/build_whoosh_index.py 2>&1 | tee -a "$MASTER_LOG" || log "  Whoosh build failed/already current"

# Step 4: Rebuild synthetic test set with new dict coverage
log "STEP 4/5: Rebuilding synthetic WHO test set..."
python scripts/eval/build_synthetic_who_test_set.py \
    --target-size 200 \
    --output data/test_sets/synthetic_who_active_v2.jsonl 2>&1 | tee -a "$MASTER_LOG"
log "  test set built"

# Step 5: Run eval (against the fresh test set + fresh Kuzu)
log "STEP 5/5: Running eval at top_k=100..."
python -u scripts/eval/evaluate_extractive_qa.py \
    --test-set data/test_sets/synthetic_who_active_v2.jsonl \
    --top-k 100 --workers 1 \
    --output "$EVAL_OUT" 2>&1 | tee -a "$MASTER_LOG"
log "  eval done → $EVAL_OUT"

# Compare with prior baseline
log "COMPARING with baseline..."
python scripts/eval/compare_eval_results.py \
    data/eval_results/local_synth_parser_fixes_20260506_233055.json \
    "$EVAL_OUT" \
    --label-baseline "PRE-REPARSE" \
    --label-new "POST-REPARSE" 2>&1 | tee -a "$MASTER_LOG"

log "=== POST-REPARSE PIPELINE DONE ==="
log "Master log: $MASTER_LOG"
log "Eval output: $EVAL_OUT"
