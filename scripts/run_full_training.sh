#!/bin/bash
#
# Full Fundamento-Centered Training Pipeline
#
# This script runs all training phases with:
# - Automatic resume from checkpoints (restartable)
# - Periodic checkpoint saves
# - Progress indicators
# - Comprehensive logging
#
# Usage:
#   ./scripts/run_full_training.sh           # Resume from checkpoints
#   ./scripts/run_full_training.sh --fresh   # Start fresh
#   ./scripts/run_full_training.sh --phase 2 # Run specific phase only
#
# Run in separate terminal, can be interrupted and resumed.
#

set -e

# Configuration
PROJECT_DIR="/home/marc/klareco"
LOG_DIR="${PROJECT_DIR}/logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/full_training_${TIMESTAMP}.log"

# Parse arguments
FRESH=false
PHASE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r  Progress: ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# Check if phase should run
should_run_phase() {
    local phase=$1
    if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "$phase" ]]; then
        return 0
    fi
    return 1
}

# Activate virtual environment
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

log "============================================================"
log "Fundamento-Centered Training Pipeline"
log "============================================================"
log "Project: $PROJECT_DIR"
log "Log: $MASTER_LOG"
log "Fresh: $FRESH"
log "Phase: $PHASE"
log ""

# ============================================================
# Phase 0: Data Extraction (if needed)
# ============================================================

if should_run_phase 0; then
    log "Phase 0: Data Extraction"
    log "------------------------------------------------------------"

    # Check if data exists
    UV_FILE="data/vocabularies/fundamento_roots.json"
    EKZ_FILE="data/training/ekzercaro_sentences.jsonl"
    REVO_FILE="data/revo/revo_definitions_with_roots.json"

    if [[ "$FRESH" == "true" ]] || [[ ! -f "$UV_FILE" ]]; then
        log "  Extracting Fundamento UV..."
        python scripts/training/extract_fundamento_uv.py 2>&1 | tee -a "$MASTER_LOG"
    else
        log "  Fundamento UV exists, skipping"
    fi

    if [[ "$FRESH" == "true" ]] || [[ ! -f "$EKZ_FILE" ]]; then
        log "  Extracting Ekzercaro..."
        python scripts/training/extract_ekzercaro.py 2>&1 | tee -a "$MASTER_LOG"
    else
        log "  Ekzercaro exists, skipping"
    fi

    if [[ ! -f "$REVO_FILE" ]]; then
        log "  WARNING: ReVo definitions not found at $REVO_FILE"
        log "  Download from: https://github.com/open-esperanto/revo-archive"
    else
        log "  ReVo definitions exist: $REVO_FILE"
    fi

    log "Phase 0 complete!"
    log ""
fi

# ============================================================
# Phase 2: Root Embedding Training
# ============================================================

if should_run_phase 2; then
    log "Phase 2: Root Embedding Training"
    log "------------------------------------------------------------"

    FRESH_FLAG=""
    if [[ "$FRESH" == "true" ]]; then
        FRESH_FLAG="--fresh"
    fi

    # Run with ReVo definitions (cleaner than old PV)
    python scripts/training/train_root_embeddings.py \
        --epochs 100 \
        --batch-size 128 \
        --learning-rate 0.01 \
        --patience 15 \
        $FRESH_FLAG \
        2>&1 | tee -a "$MASTER_LOG"

    log "Phase 2 complete!"
    log ""
fi

# ============================================================
# Phase 2b: Affix Embedding Training
# ============================================================

if should_run_phase 2b; then
    log "Phase 2b: Affix Embedding Training"
    log "------------------------------------------------------------"

    FRESH_FLAG=""
    if [[ "$FRESH" == "true" ]]; then
        FRESH_FLAG="--fresh"
    fi

    python scripts/training/train_affix_embeddings.py \
        --epochs 50 \
        $FRESH_FLAG \
        2>&1 | tee -a "$MASTER_LOG"

    log "Phase 2b complete!"
    log ""
fi

# ============================================================
# Phase 3: Corpus Integration (Optional)
# ============================================================

if should_run_phase 3; then
    log "Phase 3: Corpus Integration"
    log "------------------------------------------------------------"

    if [[ -f "data/corpus_with_sources_v2.jsonl" ]]; then
        python scripts/training/integrate_corpus.py \
            --epochs 20 \
            --max-sentences 50000 \
            2>&1 | tee -a "$MASTER_LOG"
        log "Phase 3 complete!"
    else
        log "  Corpus not found, skipping Phase 3"
    fi
    log ""
fi

# ============================================================
# Phase 4: Sentence Encoder (Optional)
# ============================================================

if should_run_phase 4; then
    log "Phase 4: Sentence Encoder Training"
    log "------------------------------------------------------------"

    if [[ -f "models/root_embeddings/best_model.pt" ]]; then
        python scripts/training/train_sentence_encoder.py \
            --epochs 50 \
            --max-sentences 10000 \
            2>&1 | tee -a "$MASTER_LOG"
        log "Phase 4 complete!"
    else
        log "  Root embeddings not found, skipping Phase 4"
    fi
    log ""
fi

# ============================================================
# Phase 5: Evaluation
# ============================================================

if should_run_phase 5; then
    log "Phase 5: Evaluation"
    log "------------------------------------------------------------"

    if [[ -f "models/root_embeddings/best_model.pt" ]]; then
        python scripts/training/evaluate_embeddings.py \
            2>&1 | tee -a "$MASTER_LOG"
    else
        log "  No models to evaluate"
    fi
    log ""
fi

# ============================================================
# Summary
# ============================================================

log "============================================================"
log "Training Pipeline Complete!"
log "============================================================"
log ""
log "Outputs:"

if [[ -f "models/root_embeddings/best_model.pt" ]]; then
    SIZE=$(du -h "models/root_embeddings/best_model.pt" | cut -f1)
    log "  Root embeddings: $SIZE"
fi

if [[ -f "models/affix_embeddings/best_model.pt" ]]; then
    SIZE=$(du -h "models/affix_embeddings/best_model.pt" | cut -f1)
    log "  Affix embeddings: $SIZE"
fi

if [[ -f "models/sentence_encoder/best_model.pt" ]]; then
    SIZE=$(du -h "models/sentence_encoder/best_model.pt" | cut -f1)
    log "  Sentence encoder: $SIZE"
fi

log ""
log "Logs: $MASTER_LOG"
log ""
log "Run demos:"
log "  python scripts/demo_root_embeddings.py"
log "  python scripts/demo_affix_embeddings.py"
log ""
log "Run evaluation:"
log "  python scripts/training/evaluate_embeddings.py"
