#!/bin/bash
# Train reranker model
# Usage: ./scripts/train_reranker.sh [--fresh]

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
    echo "No venv found"
    exit 1
fi

# Parse flags
FRESH_FLAG=""
RESUME_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
elif [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

# Create logs directory
mkdir -p logs/training

# Run training with logging
LOG_FILE="logs/training/reranker_$(date +%Y%m%d_%H%M%S).log"
echo "Starting reranker training..."
echo "Logging to: $LOG_FILE"
echo ""

python scripts/train_reranker.py \
    --train-data data/training/reranker/combined/train.jsonl \
    --val-data data/training/reranker/combined/val.jsonl \
    --compositional-model models/root_embeddings/best_model.pt \
    --output models/reranker/ \
    --batch-size 32 \
    --epochs 20 \
    --learning-rate 1e-3 \
    --patience 3 \
    --device cpu \
    $FRESH_FLAG $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete. Log saved to: $LOG_FILE"
