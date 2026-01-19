#!/bin/bash
# Train M1 selectional preference model with new selectional-aware data
# Restartable with checkpoint support

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
    echo "❌ No venv found"
    exit 1
fi

# Parse arguments
FRESH_FLAG=""
RESUME_FLAG="--resume"
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    RESUME_FLAG=""
    echo "Running fresh (no checkpoint resume)"
else
    echo "Will resume from checkpoint if available"
fi

# Check if training data exists
TRAINING_DATA="data/training/m1_selectional_hard_only"
if [ ! -d "$TRAINING_DATA" ]; then
    echo "❌ Training data not found: $TRAINING_DATA"
    echo ""
    echo "Generate training data first:"
    echo "  python scripts/prepare_m1_training_data.py"
    exit 1
fi

# Create log directory
LOG_DIR="logs/m1"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Output directory
OUTPUT_DIR="models/m1_selectional_v2"

echo "========================================================================"
echo "M1 SELECTIONAL PREFERENCE TRAINING"
echo "========================================================================"
echo ""
echo "Training data: $TRAINING_DATA"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""
echo "Model architecture:"
echo "  - Input: 64D root embeddings from Stage 1"
echo "  - Hidden: 128D"
echo "  - Outputs: subj-verb score, verb-obj score, triple score"
echo "  - Parameters: ~222K"
echo ""
echo "Training settings:"
echo "  - Epochs: 50 (with early stopping)"
echo "  - Batch size: 64"
echo "  - Learning rate: 0.001"
echo "  - Patience: 3 epochs"
echo ""
echo "Estimated time: 5-10 minutes"
echo ""

# Run training
echo "Starting training..."
python scripts/train_m1_selectional.py \
    --data-dir "$TRAINING_DATA" \
    --output-dir "$OUTPUT_DIR" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Training complete!"
    echo ""
    echo "Next step: Validate model performance"
    echo "  Run: ./scripts/m1_validate_selectional.sh"
else
    echo ""
    echo "❌ Training failed (exit code: $EXIT_CODE)"
    echo "Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
