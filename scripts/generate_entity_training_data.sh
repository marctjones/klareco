#!/bin/bash
#
# Generate Entity Type Classifier Training Data
#
# Generates training data using three strategies:
# 1. Auto-label corpus with deterministic features (~70% coverage)
# 2. Extract examples from test set
# 3. Generate synthetic examples from root vocabulary
#
# Usage:
#   ./scripts/generate_entity_training_data.sh              # Use default paths
#   ./scripts/generate_entity_training_data.sh --fresh      # Regenerate all data
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

CORPUS_PATH="data/corpus/unified_corpus.jsonl"
TEST_SET_PATH="data/test_sets/rag_test_set.jsonl"
ROOT_VOCAB_PATH="data/vocabularies/root_vocab.json"
OUTPUT_DIR="data/training/entity_classifier"
LOG_DIR="logs/data_generation"

MIN_CONFIDENCE=0.70
MAX_SYNTHETIC=100
VAL_SPLIT=0.15

# ============================================================================
# Parse arguments
# ============================================================================

FRESH_FLAG=""

for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_FLAG="--fresh"
            echo "🔄 Regenerating all data (ignoring existing)"
            ;;
        --help)
            echo "Usage: $0 [--fresh]"
            echo ""
            echo "Options:"
            echo "  --fresh   Regenerate all data, ignore existing files"
            echo ""
            echo "This script generates training data for the entity type classifier"
            echo "using three strategies:"
            echo "  1. Auto-label corpus (~70% deterministic coverage)"
            echo "  2. Extract from test set"
            echo "  3. Generate synthetic examples"
            echo ""
            exit 0
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

echo "="*60
echo "ENTITY TYPE CLASSIFIER DATA GENERATION"
echo "="*60
echo ""
echo "Configuration:"
echo "  Corpus:          $CORPUS_PATH"
echo "  Test set:        $TEST_SET_PATH"
echo "  Root vocab:      $ROOT_VOCAB_PATH"
echo "  Output:          $OUTPUT_DIR"
echo "  Min confidence:  $MIN_CONFIDENCE"
echo "  Max synthetic:   $MAX_SYNTHETIC"
echo "  Val split:       $VAL_SPLIT"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "📦 Activating .venv"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "📦 Activating venv"
    source venv/bin/activate
else
    echo "❌ ERROR: No virtual environment found (.venv or venv)"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if output already exists
if [ -f "$OUTPUT_DIR/train.jsonl" ] && [ -z "$FRESH_FLAG" ]; then
    echo "✓ Training data already exists: $OUTPUT_DIR/train.jsonl"
    echo ""
    read -p "Regenerate? This will overwrite existing data. [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing data. Use --fresh to skip this prompt."
        exit 0
    fi
    echo "🔄 Regenerating data..."
fi

# Warn if corpus not found
if [ ! -f "$CORPUS_PATH" ]; then
    echo "⚠️  WARNING: Corpus not found: $CORPUS_PATH"
    echo "  Will skip corpus auto-labeling"
    echo ""
fi

# ============================================================================
# Data Generation
# ============================================================================

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/entity_data_gen_$TIMESTAMP.log"

echo "📝 Logging to: $LOG_FILE"
echo ""
echo "="*60
echo "GENERATING TRAINING DATA"
echo "="*60
echo ""

# Run data generation with logging
python scripts/generate_entity_training_data.py \
    --corpus "$CORPUS_PATH" \
    --test-set "$TEST_SET_PATH" \
    --root-vocab "$ROOT_VOCAB_PATH" \
    --output "$OUTPUT_DIR" \
    --min-confidence $MIN_CONFIDENCE \
    --max-synthetic $MAX_SYNTHETIC \
    --val-split $VAL_SPLIT \
    2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "="*60
echo "DATA GENERATION COMPLETE"
echo "="*60
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training data generated successfully!"
    echo ""
    echo "Outputs:"
    echo "  Training data:   $OUTPUT_DIR/train.jsonl"
    echo "  Validation data: $OUTPUT_DIR/val.jsonl"
    echo "  Log file:        $LOG_FILE"
    echo ""

    # Show file sizes and counts
    if [ -f "$OUTPUT_DIR/train.jsonl" ]; then
        TRAIN_COUNT=$(wc -l < "$OUTPUT_DIR/train.jsonl")
        TRAIN_SIZE=$(du -h "$OUTPUT_DIR/train.jsonl" | cut -f1)
        echo "  Train examples:  $TRAIN_COUNT ($TRAIN_SIZE)"
    fi

    if [ -f "$OUTPUT_DIR/val.jsonl" ]; then
        VAL_COUNT=$(wc -l < "$OUTPUT_DIR/val.jsonl")
        VAL_SIZE=$(du -h "$OUTPUT_DIR/val.jsonl" | cut -f1)
        echo "  Val examples:    $VAL_COUNT ($VAL_SIZE)"
    fi

    echo ""
    echo "Next step: Train the model"
    echo "  ./scripts/train_entity_classifier.sh"
    echo ""
else
    echo "❌ Data generation failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the log for details: $LOG_FILE"
    echo ""
    exit $EXIT_CODE
fi

exit 0
