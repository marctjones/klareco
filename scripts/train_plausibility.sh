#!/bin/bash
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
    echo "No venv found!"
    exit 1
fi

# Configuration
TRAIN_DATA="data/plausibility_training_quality/train.jsonl"
VAL_DATA="data/plausibility_training_quality/val.jsonl"
OUTPUT_DIR="models/plausibility_scorer"
LOG_DIR="logs/plausibility_training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    shift
fi

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    echo "Please run dataset generation first:"
    echo "  ./scripts/build_quality_plausibility_pipeline.sh"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "ERROR: Validation data not found: $VAL_DATA"
    exit 1
fi

echo "============================================================"
echo "PLAUSIBILITY SCORER TRAINING"
echo "============================================================"
echo ""
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Architecture: Simple Concatenation MLP (98K params)"
echo "  - Frozen hybrid root embeddings (128D each)"
echo "  - Concatenate subject + verb + object (384D)"
echo "  - MLP: 384 → 256 → 128 → 1"
echo ""

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Device: CUDA (GPU)"
else
    echo "Device: CPU"
fi
echo ""

# Run training
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/train_plausibility_scorer.py \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 128 \
    --learning-rate 0.001 \
    --epochs 30 \
    --patience 5 \
    $RESUME_FLAG \
    --log-level INFO \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo "  - model_best.pt (best validation F1)"
echo "  - model_final.pt (final checkpoint)"
echo "  - training_log.json (training history)"
echo "  - config.json (model configuration)"
echo ""
echo "Log saved to: $LOG_FILE"
echo ""
