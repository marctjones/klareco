#!/bin/bash
#
# Complete M1 Training Pipeline with Quality Priority + Smart Role-Swap Negatives
#
# This script:
# 1. Generates training data with QUALITY PRIORITY (GOLD first, then sample BRONZE/COPPER)
# 2. Includes SMART ROLE-SWAP negatives to teach role-dependent selectional restrictions
# 3. Trains M1 with the improved data
#
# FIXES ISSUE #12: GOLD quality was excluded because max_triples limit was reached
#                  before GOLD appeared in corpus. This version processes
#                  GOLD FIRST to guarantee inclusion.
#
# Smart Role-Swap Enhancement:
#   - Checks corpus before swapping: only creates negatives for asymmetric relations
#   - Skips symmetric relations: if both (A,verb,B) and (B,verb,A) exist, both are valid
#   - Examples:
#     * "man fucks woman" ↔ "woman fucks man" → BOTH valid (symmetric) → no role-swap
#     * "dog eats food" exists, "food eats dog" doesn't → asymmetric → create role-swap
#   - Addresses synonym expansion issues where roots are valid but roles are wrong
#   - Data-driven: learns from corpus which verbs are symmetric vs asymmetric
#
# Usage:
#   ./scripts/train_m1_semantic_tier_priority.sh                # Generate data + train
#   ./scripts/train_m1_semantic_tier_priority.sh --skip-data    # Just train (use existing data)
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Parse arguments
SKIP_DATA=false
FRESH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --fresh)
            FRESH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-data] [--fresh]"
            echo "  --skip-data: Skip data generation (use existing)"
            echo "  --fresh: Start from scratch (ignore checkpoints)"
            exit 1
            ;;
    esac
done

# Setup paths
CORPUS_PATH="data/enhanced_corpus/corpus_with_metadata.jsonl"
DATA_DIR="data/training/m1_compositional"
MODEL_DIR="models/m1_compositional"
COMP_MODEL="models/root_embeddings_tier0/best_model.pt"
MAX_TRIPLES=500000

# Setup logging
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_LOG="$LOG_DIR/prepare_m1_tier_priority_${TIMESTAMP}.log"
TRAIN_LOG="$LOG_DIR/train_m1_tier_priority_${TIMESTAMP}.log"

echo "=============================================================================="
echo "M1 v2 Compositional Training Pipeline with Quality Priority"
echo "=============================================================================="
echo "Corpus: $CORPUS_PATH"
echo "Output: $MODEL_DIR"
echo ""
echo "QUALITY PRIORITY STRATEGY (OPTIMIZED FOR SPEED):"
echo "  Using 500K triples for faster training (<14 hour total time)"
echo ""
echo "  1. GOLD+SILVER (priority) - sample to fill 500K quota"
echo "  2. BRONZE+COPPER (fill) - if quota not met"
echo ""
echo "Expected quality distribution in training data:"
echo "  Mostly GOLD+SILVER (~100% high-quality authoritative sources)"
echo ""
echo "Total: 500K triples"
echo "Negative generation: ~2-4 hours"
echo "Model training: ~2-3 hours"
echo "Total pipeline time: ~5-7 hours (well under 14-hour limit!)"
echo "Expected: Accuracy 87-89% with excellent quality"
echo "=============================================================================="
echo ""

# Step 1: Generate quality-prioritized training data (unless skipped)
if [ "$SKIP_DATA" = true ]; then
    echo "Skipping data generation (using existing data)..."
    echo ""
else
    echo "Step 1: Generating quality-prioritized training data..."
    echo "  Logging to: $DATA_LOG"
    echo ""

    # Determine checkpoint flag
    DATA_CHECKPOINT="$DATA_DIR/data_generation_checkpoint.json"
    DATA_FLAG=""
    if [ "$FRESH" = true ]; then
        DATA_FLAG="--fresh"
        echo "  Mode: Fresh start (ignoring checkpoints)"
    elif [ -f "$DATA_CHECKPOINT" ]; then
        DATA_FLAG="--resume"
        echo "  Mode: Resuming from checkpoint"
    else
        echo "  Mode: Starting new generation"
    fi
    echo ""

    if python scripts/prepare_m1_training_data_tier_priority.py \
        --corpus "$CORPUS_PATH" \
        --stage1-model "$COMP_MODEL" \
        --output-dir "$DATA_DIR" \
        --max-triples $MAX_TRIPLES \
        --priority-qualities GOLD SILVER \
        --fill-qualities BRONZE COPPER \
        --similarity-threshold 0.15 \
        --min-parse-rate 0.0 \
        $DATA_FLAG \
        2>&1 | tee "$DATA_LOG"; then
        echo ""
        echo "✓ Data generation complete"
        echo ""
    else
        echo ""
        echo "✗ Data generation failed"
        echo "Check log: $DATA_LOG"
        exit 1
    fi
fi

# Verify data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ Training data not found: $DATA_DIR/train.jsonl"
    echo "Run without --skip-data to generate it"
    exit 1
fi

# Verify GOLD quality is included
echo "Verifying GOLD quality is included in training data..."
GOLD_COUNT=$(jq -r 'select(.source.quality == "GOLD")' "$DATA_DIR/train.jsonl" | wc -l)
echo "  GOLD examples in training data: $GOLD_COUNT"

if [ "$GOLD_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  WARNING: No GOLD quality examples in training data!"
    echo "    This should not happen with quality priority."
    echo "    Check the data generation log: $DATA_LOG"
    echo ""
else
    echo "  ✓ GOLD quality successfully included!"
    echo ""
fi

# Step 2: Train M1 with quality-prioritized data
echo "Step 2: Training M1 with quality-prioritized data..."
echo "  Logging to: $TRAIN_LOG"
echo ""

# Determine training checkpoint flag
MODEL_CHECKPOINT="$MODEL_DIR/best_model.pt"
TRAIN_FLAG=""
if [ "$FRESH" = true ]; then
    TRAIN_FLAG="--fresh"
    echo "  Mode: Fresh training (ignoring checkpoints)"
elif [ -f "$MODEL_CHECKPOINT" ]; then
    TRAIN_FLAG="--resume"
    echo "  Mode: Resuming from checkpoint"
else
    echo "  Mode: Starting new training"
fi
echo ""

if python scripts/train_m1_selectional.py \
    --comp-model "$COMP_MODEL" \
    --train-data "$DATA_DIR/train.jsonl" \
    --val-data "$DATA_DIR/val.jsonl" \
    --test-data "$DATA_DIR/test.jsonl" \
    --output-dir "$MODEL_DIR" \
    --hidden-dim 256 \
    --dropout 0.2 \
    --patience 20 \
    --epochs 50 \
    $TRAIN_FLAG \
    2>&1 | tee "$TRAIN_LOG"; then
    echo ""
    echo "=============================================================================="
    echo "✓ M1 quality-prioritized training complete!"
    echo "=============================================================================="
    echo ""
    echo "Results:"
    echo "  Model: $MODEL_DIR/best_model.pt"
    echo "  Data log: $DATA_LOG"
    echo "  Train log: $TRAIN_LOG"
    echo ""
    echo "GOLD quality in training: $GOLD_COUNT examples"
    echo ""
    echo "Next steps:"
    echo "  1. Check test accuracy in log above"
    echo "  2. Compare to baseline:"
    echo "     - Without GOLD:  86.37%"
    echo "     - With GOLD:     [see above]"
    echo "  3. Test with demo:"
    echo "     python scripts/demo_rag_with_m1.py --m1-model $MODEL_DIR/best_model.pt"
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo "✗ M1 training failed"
    echo "=============================================================================="
    echo "Check log: $TRAIN_LOG"
    exit 1
fi
