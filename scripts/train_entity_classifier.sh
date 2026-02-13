#!/bin/bash
#
# Train Entity Type Classifier
#
# Features:
# - Automatic checkpoint resume
# - Progress logging to file + console
# - Memory-efficient batch size
# - Early stopping
# - Timestamped logs
#
# Usage:
#   ./scripts/train_entity_classifier.sh              # Resume from checkpoint or start fresh
#   ./scripts/train_entity_classifier.sh --fresh      # Start fresh, ignore checkpoint
#   ./scripts/train_entity_classifier.sh --small      # Use smaller batch size (16)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR="data/training/entity_classifier"
OUTPUT_DIR="models/entity_classifier"
LOG_DIR="logs/training"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
PATIENCE=5
DEVICE="cpu"  # Change to "cuda" if GPU available

# ============================================================================
# Parse arguments
# ============================================================================

FRESH_FLAG=""
RESUME_FLAG="--resume"

for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_FLAG="--fresh"
            RESUME_FLAG=""
            echo "🔄 Starting fresh training (ignoring checkpoints)"
            ;;
        --small)
            BATCH_SIZE=16
            echo "📦 Using smaller batch size (16) for memory efficiency"
            ;;
        --gpu)
            DEVICE="cuda"
            echo "🚀 Using GPU acceleration"
            ;;
        --help)
            echo "Usage: $0 [--fresh] [--small] [--gpu]"
            echo ""
            echo "Options:"
            echo "  --fresh   Start fresh, ignore existing checkpoint"
            echo "  --small   Use smaller batch size (16 instead of 32)"
            echo "  --gpu     Use GPU if available (default: CPU)"
            echo ""
            echo "Training will automatically resume from checkpoint unless --fresh is used."
            exit 0
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

echo "="*60
echo "ENTITY TYPE CLASSIFIER TRAINING"
echo "="*60
echo ""
echo "Configuration:"
echo "  Data directory:    $DATA_DIR"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Epochs:            $EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Patience:          $PATIENCE"
echo "  Device:            $DEVICE"
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

# Check if training data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ ERROR: Training data not found: $DATA_DIR/train.jsonl"
    echo ""
    echo "Run this first to generate training data:"
    echo "  python scripts/generate_entity_training_data.py \\"
    echo "      --corpus data/corpus/unified_corpus.jsonl \\"
    echo "      --output $DATA_DIR"
    echo ""
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_PATH="$OUTPUT_DIR/best_model.pt"
if [ -f "$CHECKPOINT_PATH" ] && [ -z "$FRESH_FLAG" ]; then
    echo "✓ Found checkpoint: $CHECKPOINT_PATH"
    echo "  Will resume training from checkpoint"
    echo "  (Use --fresh to start over)"
else
    if [ -n "$FRESH_FLAG" ]; then
        echo "🔄 Starting fresh training (checkpoint ignored)"
    else
        echo "ℹ️  No checkpoint found, starting fresh"
    fi
    RESUME_FLAG=""
fi

# ============================================================================
# Training
# ============================================================================

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/entity_classifier_$TIMESTAMP.log"

echo ""
echo "📝 Logging to: $LOG_FILE"
echo ""
echo "="*60
echo "STARTING TRAINING"
echo "="*60
echo ""

# Run training with logging to both file and console
python scripts/train_entity_classifier.py \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --patience $PATIENCE \
    --device $DEVICE \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "="*60
echo "TRAINING COMPLETE"
echo "="*60
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training finished successfully!"
    echo ""
    echo "Outputs:"
    echo "  Model checkpoint: $CHECKPOINT_PATH"
    echo "  Training log:     $LOG_FILE"
    echo ""

    # Show latest checkpoint info if available
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Best model saved at: $CHECKPOINT_PATH"
        echo "Size: $(du -h "$CHECKPOINT_PATH" | cut -f1)"
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Evaluate model: python scripts/evaluate_entity_classifier.py"
    echo "  2. Integrate with answer extractor (Task 1.8)"
    echo ""
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the log for details: $LOG_FILE"
    echo ""
    echo "To resume training after fixing issues:"
    echo "  ./scripts/train_entity_classifier.sh"
    echo ""
    exit $EXIT_CODE
fi

exit 0
