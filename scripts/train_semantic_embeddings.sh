#!/bin/bash
# Train Semantic Embeddings with Contrastive Learning
#
# This script:
# 1. Prepares triplet training data from SemanticRelationDB (if needed)
# 2. Trains semantic embeddings with triplet margin loss
# 3. Logs progress to screen and file
# 4. Saves checkpoints for restartability
#
# Usage:
#   ./scripts/train_semantic_embeddings.sh           # Normal run (resumes if checkpoint exists)
#   ./scripts/train_semantic_embeddings.sh --fresh   # Start fresh, ignore checkpoint
#   ./scripts/train_semantic_embeddings.sh --prepare-only  # Just prepare data, don't train
#
# Output:
#   models/semantic_embeddings/best_model.pt    # Best trained model
#   models/semantic_embeddings/vocabulary.json  # Root vocabulary
#   logs/training/semantic_training_*.log       # Training logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
FRESH_FLAG=""
PREPARE_ONLY=false
INCLUDE_HYPERNYMS=""

for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_FLAG="--fresh"
            shift
            ;;
        --prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        --include-hypernyms)
            INCLUDE_HYPERNYMS="--include-hypernyms"
            shift
            ;;
        *)
            ;;
    esac
done

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found (.venv or venv)"
    exit 1
fi

echo "============================================================"
echo "Training Semantic Embeddings"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo ""

# Check for required input files
RELATIONS_FILE="data/raw/eo/dictionaries/revo/revo_semantic_relations.json"
TRIPLETS_FILE="data/training/semantic_triplets.jsonl"

if [ ! -f "$RELATIONS_FILE" ]; then
    echo "Error: Semantic relations file not found: $RELATIONS_FILE"
    echo "Please ensure ReVo dictionary data is available."
    exit 1
fi

# Step 1: Prepare training data (if needed or if --fresh)
if [ ! -f "$TRIPLETS_FILE" ] || [ -n "$FRESH_FLAG" ]; then
    echo "Step 1: Preparing triplet training data..."
    echo ""

    python scripts/prepare_semantic_training_data.py \
        --relations "$RELATIONS_FILE" \
        --output "$TRIPLETS_FILE" \
        --negatives-per-pair 3 \
        $INCLUDE_HYPERNYMS

    echo ""
else
    echo "Step 1: Triplet data already exists: $TRIPLETS_FILE"
    echo "  (Use --fresh to regenerate)"
    echo ""
fi

# Exit if prepare-only
if [ "$PREPARE_ONLY" = true ]; then
    echo "Prepare-only mode: exiting without training."
    exit 0
fi

# Step 2: Train semantic embeddings
echo "Step 2: Training semantic embeddings..."
echo ""

# Create logs directory
mkdir -p logs/training

# Training configuration
EPOCHS=20
BATCH_SIZE=256
LEARNING_RATE=0.001
MARGIN=0.5
EMBEDDING_DIM=64
PATIENCE=5

# Detect device
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
elif python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Margin: $MARGIN"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  Device: $DEVICE"
echo "  Patience: $PATIENCE"
echo ""

python scripts/training/train_semantic_embeddings.py \
    --triplets "$TRIPLETS_FILE" \
    --output-dir models/semantic_embeddings \
    --embedding-dim $EMBEDDING_DIM \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --margin $MARGIN \
    --patience $PATIENCE \
    --device $DEVICE \
    $FRESH_FLAG

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Model: models/semantic_embeddings/best_model.pt"
echo "  Vocabulary: models/semantic_embeddings/vocabulary.json"
echo "  Logs: logs/training/semantic_training_*.log"
echo ""
echo "To test the embeddings:"
echo "  python -c \"import torch; m = torch.load('models/semantic_embeddings/best_model.pt'); print('Loaded!')\""
