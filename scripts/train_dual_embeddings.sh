#!/bin/bash
# Train dual parallel embeddings (linguistic + topical)
# Run this after topical data generation is complete

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
    echo "Error: No virtual environment found"
    exit 1
fi

# Create output directory
mkdir -p models/dual_embeddings logs/training

# Configuration
TRAINING_DATA="data/training/topical_pairs_smart.jsonl"
LINGUISTIC_MODEL="models/root_embeddings/best_model.pt"
VOCAB="data/vocabularies/topical_vocab.json"
OUTPUT_DIR="models/dual_embeddings"

echo "=========================================="
echo "Dual Embeddings Training"
echo "=========================================="
echo ""
echo "Strategy: Sequential Training"
echo "  Phase 1: Train topical (freeze linguistic)"
echo "  Phase 2: Joint fine-tuning (optional)"
echo ""
echo "Input:"
echo "  Training data: $TRAINING_DATA"
echo "  Linguistic model: $LINGUISTIC_MODEL"
echo "  Vocabulary: $VOCAB"
echo ""
echo "Output:"
echo "  Model: $OUTPUT_DIR/dual_embeddings_final.pt"
echo ""
echo "Training configuration:"
echo "  Topical epochs: 10"
echo "  Joint epochs: 5"
echo "  Batch size: 1024"
echo "  Learning rate: 0.001"
echo ""
echo "Estimated time: ~30-60 minutes"
echo ""

# Check if inputs exist
if [ ! -f "$TRAINING_DATA" ]; then
    echo "Error: Training data not found: $TRAINING_DATA"
    echo "Run ./run_smart_generation.sh first"
    exit 1
fi

if [ ! -f "$LINGUISTIC_MODEL" ]; then
    echo "Error: Linguistic model not found: $LINGUISTIC_MODEL"
    echo "Run ./scripts/train_roots.sh first"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "Error: Vocabulary not found: $VOCAB"
    exit 1
fi

echo "Starting training..."
echo ""

# Run training with logging
LOG_FILE="logs/training/dual_training_$(date +%Y%m%d_%H%M%S).log"

python scripts/training/train_dual_embeddings.py \
    --training-data "$TRAINING_DATA" \
    --linguistic-model "$LINGUISTIC_MODEL" \
    --vocab "$VOCAB" \
    --output-dir "$OUTPUT_DIR" \
    --topical-epochs 10 \
    --joint-epochs 5 \
    --batch-size 1024 \
    --learning-rate 0.001 \
    --device cpu \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Output model: $OUTPUT_DIR/dual_embeddings_final.pt"
echo "Training log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Rebuild HNSW index with 128d embeddings (Task #73)"
echo "  2. Update AST-aware retriever with adaptive weighting (Task #75)"
echo "  3. Run benchmark evaluation (Task #76)"
echo ""
