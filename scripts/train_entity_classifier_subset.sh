#!/bin/bash
#
# Train Entity Type Classifier on Subset
#
# Uses a manageable subset of training data (100K examples) for practical training time.
# Full 8.9M dataset would take days - this completes in 30-60 minutes.
#
# Features:
# - Progress updates every 10 batches
# - Automatic checkpoint resume
# - Checkpoint saves after each epoch
# - Timestamped logs
# - Memory efficient
#
# Usage:
#   ./scripts/train_entity_classifier_subset.sh              # Train on 100K subset
#   ./scripts/train_entity_classifier_subset.sh --fresh      # Start fresh
#   ./scripts/train_entity_classifier_subset.sh --size 50000 # Use 50K examples
#   ./scripts/train_entity_classifier_subset.sh --gpu        # Use GPU
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR="data/training/entity_classifier"
SUBSET_DIR="data/training/entity_classifier_subset"
OUTPUT_DIR="models/entity_classifier"
LOG_DIR="logs/training"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
PATIENCE=5
DEVICE="cpu"

# Subset size (default: 100K for reasonable training time)
SUBSET_SIZE=100000
VAL_SUBSET_SIZE=15000

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
        --size)
            shift
            SUBSET_SIZE=$1
            VAL_SUBSET_SIZE=$((SUBSET_SIZE * 15 / 100))
            echo "📊 Using custom subset size: $SUBSET_SIZE train, $VAL_SUBSET_SIZE val"
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
            echo "Usage: $0 [--fresh] [--size N] [--small] [--gpu]"
            echo ""
            echo "Options:"
            echo "  --fresh      Start fresh, ignore existing checkpoint"
            echo "  --size N     Use N training examples (default: 100000)"
            echo "  --small      Use smaller batch size (16 instead of 32)"
            echo "  --gpu        Use GPU if available (default: CPU)"
            echo ""
            echo "This script trains on a subset of data for practical training time."
            echo "Full 8.9M dataset would take days. 100K subset takes 30-60 minutes."
            echo ""
            exit 0
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

echo "="*60
echo "ENTITY CLASSIFIER TRAINING (SUBSET)"
echo "="*60
echo ""
echo "Configuration:"
echo "  Training subset:   $SUBSET_SIZE examples"
echo "  Validation subset: $VAL_SUBSET_SIZE examples"
echo "  Epochs:            $EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Patience:          $PATIENCE"
echo "  Device:            $DEVICE"
echo ""
echo "Note: Using subset for practical training time (~30-60 min)"
echo "      Full 8.9M dataset would take several hours/days"
echo ""

# Create directories
mkdir -p "$SUBSET_DIR"
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
    exit 1
fi

# ============================================================================
# Create Subset (if needed)
# ============================================================================

TRAIN_SUBSET="$SUBSET_DIR/train.jsonl"
VAL_SUBSET="$SUBSET_DIR/val.jsonl"

if [ ! -f "$TRAIN_SUBSET" ] || [ -n "$FRESH_FLAG" ]; then
    echo "📊 Creating training subset..."
    echo ""

    # Check if full dataset exists
    if [ ! -f "$DATA_DIR/train.jsonl" ]; then
        echo "❌ ERROR: Full training data not found: $DATA_DIR/train.jsonl"
        echo "Run ./scripts/generate_entity_training_data.sh first"
        exit 1
    fi

    # Sample training data
    echo "  Sampling $SUBSET_SIZE examples from training set..."
    head -n $SUBSET_SIZE "$DATA_DIR/train.jsonl" > "$TRAIN_SUBSET"
    TRAIN_ACTUAL=$(wc -l < "$TRAIN_SUBSET")
    echo "  ✓ Created: $TRAIN_SUBSET ($TRAIN_ACTUAL examples)"

    # Sample validation data
    echo "  Sampling $VAL_SUBSET_SIZE examples from validation set..."
    head -n $VAL_SUBSET_SIZE "$DATA_DIR/val.jsonl" > "$VAL_SUBSET"
    VAL_ACTUAL=$(wc -l < "$VAL_SUBSET")
    echo "  ✓ Created: $VAL_SUBSET ($VAL_ACTUAL examples)"

    echo ""
else
    TRAIN_ACTUAL=$(wc -l < "$TRAIN_SUBSET")
    VAL_ACTUAL=$(wc -l < "$VAL_SUBSET")
    echo "✓ Using existing subset: $TRAIN_ACTUAL train, $VAL_ACTUAL val"
    echo "  (Use --fresh to recreate subset)"
    echo ""
fi

# ============================================================================
# Check for checkpoint
# ============================================================================

CHECKPOINT_PATH="$OUTPUT_DIR/best_model.pt"
if [ -f "$CHECKPOINT_PATH" ] && [ -z "$FRESH_FLAG" ]; then
    echo "✓ Found checkpoint: $CHECKPOINT_PATH"
    echo "  Will resume training from checkpoint"
    echo "  (Use --fresh to start over)"
else
    if [ -n "$FRESH_FLAG" ]; then
        echo "🔄 Starting fresh training (checkpoint ignored)"
        # Remove old checkpoint
        rm -f "$CHECKPOINT_PATH"
    else
        echo "ℹ️  No checkpoint found, starting fresh"
    fi
    RESUME_FLAG=""
fi

echo ""

# ============================================================================
# Training
# ============================================================================

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/entity_classifier_subset_$TIMESTAMP.log"

echo "📝 Logging to: $LOG_FILE"
echo ""
echo "="*60
echo "STARTING TRAINING"
echo "="*60
echo ""
echo "Progress updates:"
echo "  - Batch progress every 10 batches"
echo "  - Validation after each epoch"
echo "  - Checkpoint saved when validation improves"
echo "  - Early stopping if no improvement for $PATIENCE epochs"
echo ""
echo "Estimated time: 30-60 minutes (depends on CPU speed)"
echo ""

# Calculate expected batches per epoch
BATCHES_PER_EPOCH=$((TRAIN_ACTUAL / BATCH_SIZE))
echo "Training details:"
echo "  Batches per epoch: $BATCHES_PER_EPOCH"
echo "  Total batches (if 50 epochs): $((BATCHES_PER_EPOCH * EPOCHS))"
echo ""
echo "-"*60
echo ""

# Run training with logging to both file and console
python scripts/train_entity_classifier.py \
    --data "$SUBSET_DIR" \
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
        echo ""

        # Try to extract metrics from log
        if grep -q "Best validation accuracy" "$LOG_FILE"; then
            BEST_ACC=$(grep "Best validation accuracy" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+')
            echo "Best validation accuracy: $BEST_ACC"
        fi
    fi

    echo ""
    echo "Next steps:"
    echo "  1. View training curves: grep 'Validation' $LOG_FILE"
    echo "  2. Test the model: python scripts/test_entity_classifier.py"
    echo "  3. Integrate with answer extractor (Task 1.8)"
    echo ""
    echo "Note: This model was trained on $TRAIN_ACTUAL examples (subset)."
    echo "      For production, consider training on larger subset or full data."
    echo ""
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the log for details: $LOG_FILE"
    echo ""
    echo "To resume training after fixing issues:"
    echo "  ./scripts/train_entity_classifier_subset.sh"
    echo ""
    exit $EXIT_CODE
fi

exit 0
