#!/bin/bash
#
# Fundamento-Centered Training Pipeline
#
# This script runs the complete training pipeline for root embeddings
# following the Fundamento-Centered approach (pure Esperanto, no cross-lingual).
#
# Usage:
#   ./scripts/run_fundamento_training.sh           # Run pipeline (resume if possible)
#   ./scripts/run_fundamento_training.sh --fresh   # Start completely fresh
#   ./scripts/run_fundamento_training.sh --phase 0 # Run only Phase 0
#   ./scripts/run_fundamento_training.sh --dry-run # Dry run (no output)
#
# Phases:
#   0.1 - Extract Fundamento Universala Vortaro (UV)
#   0.2 - Extract Ekzercaro sentences
#   1   - Train root embeddings (uses ReVo definitions from data/revo/)
#
# Outputs:
#   data/vocabularies/fundamento_roots.json
#   data/training/ekzercaro_sentences.jsonl
#   models/root_embeddings/best_model.pt
#
# Logs:
#   logs/training/fundamento_training_TIMESTAMP.log
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/fundamento_training_${TIMESTAMP}.log"

# Parse arguments
DRY_RUN=""
PHASE=""
FRESH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --fresh)
            FRESH="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/data/vocabularies"
mkdir -p "$PROJECT_DIR/data/training"
mkdir -p "$PROJECT_DIR/models/root_embeddings"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log_section() {
    log "============================================================"
    log "$1"
    log "============================================================"
}

# Check if phase should run
should_run_phase() {
    if [[ -z "$PHASE" ]]; then
        return 0  # No phase specified, run all
    fi
    local target_phase="$1"
    if [[ "$PHASE" == "$target_phase" ]] || [[ "$PHASE" == "${target_phase%.*}" ]]; then
        return 0  # Match exact or parent phase
    fi
    return 1
}

# Check if output already exists (for restartability)
# Returns 0 (true) if we should run the phase
# Returns 1 (false) if we should skip the phase
check_output() {
    local output_file="$1"
    if [[ -f "$output_file" ]] && [[ -z "$DRY_RUN" ]]; then
        log "Output exists: $output_file"
        if [[ -n "$FRESH" ]]; then
            log "Re-running due to --fresh flag"
            return 0
        fi
        log "Skipping (use --fresh to regenerate)"
        return 1
    fi
    return 0
}

cd "$PROJECT_DIR"

log_section "Fundamento-Centered Training Pipeline"
log "Project directory: $PROJECT_DIR"
log "Timestamp: $TIMESTAMP"
log "Dry run: ${DRY_RUN:-no}"
log "Phase: ${PHASE:-all}"
log "Fresh: ${FRESH:-no}"

# Activate virtual environment if exists
if [[ -f ".venv/bin/activate" ]]; then
    log "Activating virtual environment..."
    source .venv/bin/activate
fi

# Phase 0.1: Extract Fundamento UV
if should_run_phase "0.1" || should_run_phase "0"; then
    log_section "Phase 0.1: Fundamento UV Extraction"

    OUTPUT="$PROJECT_DIR/data/vocabularies/fundamento_roots.json"

    if check_output "$OUTPUT"; then
        log "Running: python scripts/training/extract_fundamento_uv.py $DRY_RUN"

        python scripts/training/extract_fundamento_uv.py \
            --input data/raw/fundamento/fundamento_de_esperanto.txt \
            --output "$OUTPUT" \
            --log-dir "$LOG_DIR" \
            $DRY_RUN \
            2>&1 | tee -a "$MASTER_LOG"

        if [[ $? -eq 0 ]]; then
            log "Phase 0.1 completed successfully"
        else
            log "ERROR: Phase 0.1 failed"
            exit 1
        fi
    fi
fi

# Phase 0.2: Extract Ekzercaro
if should_run_phase "0.2" || should_run_phase "0"; then
    log_section "Phase 0.2: Ekzercaro Extraction"

    OUTPUT="$PROJECT_DIR/data/training/ekzercaro_sentences.jsonl"

    if check_output "$OUTPUT"; then
        log "Running: python scripts/training/extract_ekzercaro.py $DRY_RUN"

        python scripts/training/extract_ekzercaro.py \
            --input data/raw/fundamento/fundamento_de_esperanto.txt \
            --output "$OUTPUT" \
            --log-dir "$LOG_DIR" \
            --use-parser \
            $DRY_RUN \
            2>&1 | tee -a "$MASTER_LOG"

        if [[ $? -eq 0 ]]; then
            log "Phase 0.2 completed successfully"
        else
            log "ERROR: Phase 0.2 failed"
            exit 1
        fi
    fi
fi

# Phase 1: Train Root Embeddings (uses ReVo from data/revo/)
if should_run_phase "1"; then
    log_section "Phase 1: Root Embedding Training"

    OUTPUT_DIR="$PROJECT_DIR/models/root_embeddings"

    # Default: resume from checkpoint. Use --fresh to start over
    FRESH_FLAG=""
    if [[ -n "$FRESH" ]]; then
        FRESH_FLAG="--fresh"
        log "Starting fresh training (--fresh flag set)"
    elif [[ -f "$OUTPUT_DIR/checkpoint.pt" ]]; then
        log "Resuming from existing checkpoint"
    else
        log "No checkpoint found, starting fresh"
    fi

    # First, ensure clean vocabulary exists
    if [[ ! -f "data/vocabularies/clean_roots.json" ]]; then
        log "Generating clean vocabulary..."
        python scripts/clean_revo_vocabulary.py 2>&1 | tee -a "$MASTER_LOG"
    fi

    # Extract ReVo semantic relations (synonyms, antonyms, hypernyms)
    if [[ ! -f "data/revo/revo_semantic_relations.json" ]]; then
        log "Extracting ReVo semantic relations..."
        python scripts/extract_revo_relations.py 2>&1 | tee -a "$MASTER_LOG"
    fi

    log "Running: python scripts/training/train_root_embeddings.py $DRY_RUN $FRESH_FLAG"

    python scripts/training/train_root_embeddings.py \
        --fundamento-roots data/vocabularies/fundamento_roots.json \
        --revo-definitions data/revo/revo_definitions_with_roots.json \
        --ekzercaro data/training/ekzercaro_sentences.jsonl \
        --clean-vocab data/vocabularies/clean_roots.json \
        --revo-relations data/revo/revo_semantic_relations.json \
        --output-dir "$OUTPUT_DIR" \
        --log-dir "$LOG_DIR" \
        --epochs 100 \
        --batch-size 128 \
        --learning-rate 0.005 \
        --patience 10 \
        $DRY_RUN \
        $FRESH_FLAG \
        2>&1 | tee -a "$MASTER_LOG"

    if [[ $? -eq 0 ]]; then
        log "Phase 1 completed successfully"
    else
        log "ERROR: Phase 1 failed"
        exit 1
    fi
fi

log_section "Pipeline Complete"
log "Master log: $MASTER_LOG"

# Summary of outputs
log ""
log "Outputs:"
if [[ -f "$PROJECT_DIR/data/vocabularies/fundamento_roots.json" ]]; then
    SIZE=$(wc -c < "$PROJECT_DIR/data/vocabularies/fundamento_roots.json")
    log "  - fundamento_roots.json: ${SIZE} bytes"
fi
if [[ -f "$PROJECT_DIR/data/training/ekzercaro_sentences.jsonl" ]]; then
    LINES=$(wc -l < "$PROJECT_DIR/data/training/ekzercaro_sentences.jsonl")
    log "  - ekzercaro_sentences.jsonl: ${LINES} sentences"
fi
if [[ -f "$PROJECT_DIR/data/revo/revo_definitions_with_roots.json" ]]; then
    SIZE=$(wc -c < "$PROJECT_DIR/data/revo/revo_definitions_with_roots.json")
    log "  - revo_definitions_with_roots.json: ${SIZE} bytes"
fi
if [[ -f "$PROJECT_DIR/models/root_embeddings/best_model.pt" ]]; then
    SIZE=$(wc -c < "$PROJECT_DIR/models/root_embeddings/best_model.pt")
    log "  - best_model.pt: ${SIZE} bytes"
fi

log ""
log "Done!"
