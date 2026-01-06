#!/usr/bin/bash
# Train topical embeddings independently
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
mkdir -p models/topical_embeddings logs/training

# Configuration
TRAINING_DATA="data/training/topical_pairs_smart.jsonl"
VOCAB="data/vocabularies/topical_vocab.json"
OUTPUT_DIR="models/topical_embeddings"

echo "=========================================="
echo "Topical Embeddings Training"
echo "=========================================="
echo ""
echo "Strategy: Independent topical training"
echo "  - Skip-gram pairs from corpus"
echo "  - Captures co-occurrence patterns"
echo "  - Independent from linguistic embeddings"
echo ""
echo "Input:"
echo "  Training data: $TRAINING_DATA"
echo "  Vocabulary: $VOCAB"
echo ""
echo "Output:"
echo "  Model: $OUTPUT_DIR/best_model.pt"
echo ""
echo "Training configuration:"
echo "  Epochs: 10"
echo "  Batch size: 1024"
echo "  Learning rate: 0.001"
echo "  Embedding dim: 64"
echo ""
echo "Estimated time: ~30-60 minutes"
echo ""

# Check if inputs exist
if [ ! -f "$TRAINING_DATA" ]; then
    echo "Error: Training data not found: $TRAINING_DATA"
    echo "Run ./run_smart_generation.sh first"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "Error: Vocabulary not found: $VOCAB"
    exit 1
fi

echo "Starting training..."
echo ""

# Run training with logging
LOG_FILE="logs/training/topical_training_$(date +%Y%m%d_%H%M%S).log"

python scripts/training/train_topical_embeddings.py \
    --training-data "$TRAINING_DATA" \
    --vocab "$VOCAB" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10 \
    --batch-size 1024 \
    --learning-rate 0.001 \
    --device cpu \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Output model: $OUTPUT_DIR/best_model.pt"
echo "Training log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Load hybrid embeddings:"
echo "     from klareco.embeddings.hybrid_embeddings import HybridEmbeddings"
echo "     model = HybridEmbeddings.from_checkpoints("
echo "         'models/root_embeddings/best_model.pt',"
echo "         'models/topical_embeddings/best_model.pt'"
echo "     )"
echo "  2. Update retrieval pipeline to use HybridEmbeddings"
echo "  3. Rebuild HNSW index with 128d embeddings"
echo "  4. Run benchmark evaluation"
echo ""
