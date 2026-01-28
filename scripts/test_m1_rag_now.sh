#!/bin/bash
#
# Test Current M1 Checkpoint While Training Continues
#
# This script safely copies the current best M1 checkpoint and tests it
# without interfering with ongoing training.
#
# Usage:
#   ./scripts/test_m1_rag_now.sh                          # Run example queries
#   ./scripts/test_m1_rag_now.sh -i                       # Interactive mode
#   ./scripts/test_m1_rag_now.sh "Kiu fondis Esperanton?" # Single query
#

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
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Model paths
M1_TRAINING="models/m1_compositional/best_model.pt"
M1_TEST="models/m1_compositional/best_model_test.pt"
COMP_MODEL="models/root_embeddings_tier0/best_model.pt"
INDEX="data/indexes/kuzu_index"

echo "=============================================================================="
echo "RAG Demo with M1 - Test Current Checkpoint"
echo "=============================================================================="

# Check if training model exists
if [ ! -f "$M1_TRAINING" ]; then
    echo "❌ M1 model not found: $M1_TRAINING"
    echo "Training may not have started yet."
    exit 1
fi

# Copy checkpoint for safe testing
echo "📋 Copying current checkpoint for testing..."
cp "$M1_TRAINING" "$M1_TEST"
echo "  ✓ Copied to: $M1_TEST"
echo ""

# Check checkpoint info
if command -v python &> /dev/null; then
    echo "📊 Checkpoint info:"
    python -c "
import torch
ckpt = torch.load('$M1_TEST', map_location='cpu')
print(f'  Epoch: {ckpt.get(\"epoch\", \"?\")}/50')
print(f'  Best val accuracy: {ckpt.get(\"best_val_acc\", 0):.4f}')
print(f'  Parameters: {sum(p.numel() for p in ckpt[\"model_state_dict\"].values()):,}')
" 2>/dev/null || echo "  (Unable to read checkpoint details)"
    echo ""
fi

# Check if CompositionalEmbedding model exists
if [ ! -f "$COMP_MODEL" ]; then
    echo "❌ CompositionalEmbedding not found: $COMP_MODEL"
    exit 1
fi

# Check if index exists
if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/kuzu.db" ]; then
    echo "❌ Kuzu index not found: $INDEX/kuzu.db"
    exit 1
fi

echo "🚀 Running RAG demo with M1 filtering..."
echo "   (Training continues in the background)"
echo ""

# Run demo with test checkpoint
python scripts/demo_rag_with_m1.py \
    --m1-model "$M1_TEST" \
    --comp-model "$COMP_MODEL" \
    --index "$INDEX" \
    "$@"

echo ""
echo "=============================================================================="
echo "✓ Test complete"
echo "=============================================================================="
echo ""
echo "Notes:"
echo "  • Test used snapshot: $M1_TEST"
echo "  • Training continues with: $M1_TRAINING"
echo "  • Test snapshot will not be updated (safe to reuse)"
echo ""
echo "To test latest checkpoint again, re-run this script."
echo "=============================================================================="
