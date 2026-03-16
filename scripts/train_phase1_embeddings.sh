#!/bin/bash
#
# Train Phase 1 Root Embeddings (Skip-Gram with Cross-Sentence Context)
#
# This script runs the complete pipeline:
# 1. Extract training pairs from Kuzu v2.1 database
# 2. Train skip-gram model with negative sampling
# 3. Save embeddings for retriever integration
#
# Usage:
#   ./scripts/train_phase1_embeddings.sh                    # Full pipeline (auto-skips extraction if exists)
#   ./scripts/train_phase1_embeddings.sh --fresh            # Force re-extract even if pairs exist
#   ./scripts/train_phase1_embeddings.sh --extract-only     # Only extract pairs
#   ./scripts/train_phase1_embeddings.sh --train-only       # Only train (pairs must exist)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found (.venv or venv)"
    exit 1
fi

# Configuration
DB_PATH="data/indexes/v2.1_kuzu_index_full"
TRAINING_DIR="data/training/phase1_embeddings"
OUTPUT_DIR="models/root_embeddings_phase1"
LOG_DIR="logs/phase1_embeddings"

# Create directories
mkdir -p "$TRAINING_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Parse arguments
EXTRACT=true
TRAIN=true
FRESH=false

if [ "$1" == "--extract-only" ]; then
    TRAIN=false
elif [ "$1" == "--train-only" ]; then
    EXTRACT=false
elif [ "$1" == "--fresh" ]; then
    FRESH=true
fi

# Auto-detect existing training pairs (unless --fresh or --extract-only)
if [ "$FRESH" = false ] && [ "$EXTRACT" = true ]; then
    if [ -f "$TRAINING_DIR/root_embedding_pairs.jsonl" ] && \
       [ -f "$TRAINING_DIR/root_embedding_pairs_vocab.json" ] && \
       [ -f "$TRAINING_DIR/root_embedding_pairs_stats.json" ]; then
        echo ""
        echo "Found existing training pairs, skipping extraction (use --fresh to re-extract)"
        EXTRACT=false
    fi
fi

# Timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

echo "============================================================================"
echo "Phase 1 Root Embeddings Training Pipeline"
echo "============================================================================"
echo "Database: $DB_PATH"
echo "Training data: $TRAINING_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================================"

# Step 1: Extract training pairs (if needed)
if [ "$EXTRACT" = true ]; then
    echo ""
    echo "Step 1: Extracting training pairs with cross-sentence context..."
    echo "--------------------------------------------------------------------------------"

    python scripts/extract_embedding_training_pairs.py \
        --db-path "$DB_PATH" \
        --output "$TRAINING_DIR/root_embedding_pairs.jsonl" \
        --vocab-output "$TRAINING_DIR/root_embedding_pairs_vocab.json" \
        --window-size 5 \
        --cross-sentence-weight 0.5 \
        --min-frequency 5 \
        --subsample-threshold 1e-3 \
        2>&1 | tee -a "$LOG_FILE"

    echo "Training pairs extracted."
else
    echo "Skipping extraction (using existing pairs)"
fi

# Check if training pairs exist
if [ ! -f "$TRAINING_DIR/root_embedding_pairs.jsonl" ]; then
    echo "ERROR: Training pairs not found: $TRAINING_DIR/root_embedding_pairs.jsonl"
    echo "Run without --train-only flag to extract pairs first"
    exit 1
fi

# Step 2: Train skip-gram model (if needed)
if [ "$TRAIN" = true ]; then
    echo ""
    echo "Step 2: Training skip-gram model with negative sampling..."
    echo "--------------------------------------------------------------------------------"

    python scripts/train_root_embeddings_skipgram_v2_1.py \
        --training-pairs "$TRAINING_DIR/root_embedding_pairs.jsonl" \
        --vocabulary "$TRAINING_DIR/root_embedding_pairs_vocab.json" \
        --output "$OUTPUT_DIR" \
        --embedding-dim 64 \
        --epochs 10 \
        --batch-size 256 \
        --learning-rate 0.025 \
        --negative-samples 5 \
        --patience 3 \
        --min-delta 0.001 \
        --collapse-threshold 0.7 \
        --resume \
        2>&1 | tee -a "$LOG_FILE"

    echo "Training complete."
else
    echo "Skipping training"
fi

echo ""
echo "============================================================================"
echo "Pipeline Complete"
echo "============================================================================"

if [ "$TRAIN" = true ]; then
    echo "Best model: $OUTPUT_DIR/root_embeddings_best.pt"
    echo "Final model: $OUTPUT_DIR/root_embeddings_final.pt"
fi

echo "Training pairs: $TRAINING_DIR/root_embedding_pairs.jsonl"
echo "Vocabulary: $TRAINING_DIR/root_embedding_pairs_vocab.json"
echo "Statistics: $TRAINING_DIR/root_embedding_pairs_stats.json"
echo "Log file: $LOG_FILE"
echo "============================================================================"
