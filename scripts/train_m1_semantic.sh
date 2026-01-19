#!/bin/bash
#
# Complete M1 Training Pipeline with Semantic-Distance Corruption
#
# This script:
# 1. Generates training data with semantic-distance-based corruption
# 2. Trains M1 with the improved data
#
# FIXES BUG #2: Ensures corrupted negatives are semantically distant
#              from positives, creating a learnable signal.
#
# Usage:
#   ./scripts/train_m1_semantic.sh                    # Use tier0 only (30K examples)
#   ./scripts/train_m1_semantic.sh --full-corpus      # Use full corpus (400K examples)
#   ./scripts/train_m1_semantic.sh --skip-data        # Use existing data, just retrain model
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
USE_FULL_CORPUS=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --full-corpus)
            USE_FULL_CORPUS=true
            shift
            ;;
    esac
done

# Setup paths based on corpus choice
if [ "$USE_FULL_CORPUS" = true ]; then
    CORPUS_PATH="data/enhanced_corpus/corpus_full_with_tier0.jsonl"
    DATA_DIR="data/training/m1_semantic_full"
    MODEL_DIR="models/m1_semantic_full"
    MAX_TRIPLES=200000
    DATASET_DESC="Full corpus (400K examples, mixed quality)"
else
    CORPUS_PATH="data/enhanced_corpus/corpus_with_tier0.jsonl"
    DATA_DIR="data/training/m1_semantic_tier0"
    MODEL_DIR="models/m1_semantic_tier0"
    MAX_TRIPLES=""
    DATASET_DESC="Tier0 only (30K examples, highest quality)"
fi

# Setup logging
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_LOG="$LOG_DIR/prepare_m1_semantic_${TIMESTAMP}.log"
TRAIN_LOG="$LOG_DIR/train_m1_semantic_${TIMESTAMP}.log"

echo "=============================================="
echo "M1 Semantic-Distance Training Pipeline"
echo "=============================================="
echo "Dataset: $DATASET_DESC"
echo "Corpus: $CORPUS_PATH"
echo "Output: $MODEL_DIR"
echo ""
echo "KEY IMPROVEMENT:"
echo "  Corrupted negatives are semantically DISTANT"
echo "  from positives (similarity < 0.15), creating"
echo "  a learnable signal for M1."
echo ""
echo "Expected: Accuracy > 75% (vs 70% with random)"
echo "=============================================="
echo ""

# Step 1: Generate semantic-distance training data (unless skipped)
if [ "$SKIP_DATA" = true ]; then
    echo "Skipping data generation (using existing data)..."
    echo ""
else
    echo "Step 1: Generating semantic-distance training data..."
    echo "  Logging to: $DATA_LOG"
    echo ""

    MAX_TRIPLES_ARG=""
    if [ -n "$MAX_TRIPLES" ]; then
        MAX_TRIPLES_ARG="--max-triples $MAX_TRIPLES"
    fi

    if python scripts/prepare_m1_training_data_semantic.py \
        --corpus "$CORPUS_PATH" \
        --stage1-model models/root_embeddings_tier0/best_model.pt \
        --output-dir "$DATA_DIR" \
        $MAX_TRIPLES_ARG \
        --similarity-threshold 0.15 \
        --min-parse-rate 0.0 \
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

# Step 2: Train M1 with semantic data
echo "Step 2: Training M1 with semantic-distance data..."
echo "  Logging to: $TRAIN_LOG"
echo ""

if python scripts/train_m1_selectional.py \
    --stage1-model models/root_embeddings_tier0/best_model.pt \
    --data-dir "$DATA_DIR" \
    --output-dir "$MODEL_DIR" \
    --hidden-dim 256 \
    --dropout 0.2 \
    --patience 20 \
    --epochs 50 \
    --fresh \
    2>&1 | tee "$TRAIN_LOG"; then
    echo ""
    echo "=============================================="
    echo "✓ M1 semantic-distance training complete!"
    echo "=============================================="
    echo ""
    echo "Results:"
    echo "  Model: $MODEL_DIR/best_model.pt"
    echo "  Data log: $DATA_LOG"
    echo "  Train log: $TRAIN_LOG"
    echo ""
    echo "Next steps:"
    echo "  1. Check test accuracy in log above"
    echo "  2. Run validation: python scripts/validate_m1_extensive.py --test-data $DATA_DIR/test.jsonl"
    echo "  3. Compare to baseline:"
    echo "     - Random corruption (tier0): 69.2%"
    echo "     - Random corruption (full):  70.2%"
    echo "     - This (semantic):           [see above]"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "✗ M1 training failed"
    echo "=============================================="
    echo "Check log: $TRAIN_LOG"
    exit 1
fi
