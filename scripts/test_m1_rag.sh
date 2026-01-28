#!/bin/bash
#
# Test RAG with M1 Plausibility Filtering
#
# This script runs the RAG demo with the newly trained M1 model.
#
# M1 filters semantically implausible SVO triples after synonym expansion,
# improving retrieval quality by removing nonsense combinations.
#
# Usage:
#   ./scripts/test_m1_rag.sh                          # Run example queries
#   ./scripts/test_m1_rag.sh -i                       # Interactive mode
#   ./scripts/test_m1_rag.sh "Kiu fondis Esperanton?" # Single query
#   ./scripts/test_m1_rag.sh --no-translate           # Pure Esperanto (no EN translations)
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
M1_MODEL="models/m1_semantic_tier_priority/best_model.pt"
STAGE1_MODEL="models/root_embeddings_tier0/best_model.pt"
INDEX="data/indexes/kuzu_index"

# Check if M1 model exists
if [ ! -f "$M1_MODEL" ]; then
    echo "❌ M1 model not found: $M1_MODEL"
    echo ""
    echo "The model is currently training. Options:"
    echo "  1. Wait for training to complete (~5-7 hours)"
    echo "  2. Copy current checkpoint to test while training continues:"
    echo "     cp models/m1_semantic_tier_priority/best_model.pt \\"
    echo "        models/m1_semantic_tier_priority/best_model_test.pt"
    echo "     Then edit this script to use best_model_test.pt"
    echo ""
    exit 1
fi

# Check if Stage 1 model exists
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "❌ Stage 1 embeddings not found: $STAGE1_MODEL"
    echo "Train with: ./scripts/train_roots.sh"
    exit 1
fi

# Check if index exists
if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/kuzu.db" ]; then
    echo "❌ Kuzu index not found: $INDEX/kuzu.db"
    echo "Build index with: python scripts/index_kuzu.py"
    exit 1
fi

echo "=============================================================================="
echo "RAG Demo with M1 Plausibility Filtering"
echo "=============================================================================="
echo "M1 Model: $M1_MODEL"
echo "Stage 1: $STAGE1_MODEL"
echo "Index: $INDEX"
echo ""
echo "M1 filters implausible SVO triples from synonym expansion results."
echo "Expected: Higher precision, fewer nonsense answers."
echo "=============================================================================="
echo ""

# Run demo with arguments passed through
python scripts/demo_rag_with_m1.py \
    --m1-model "$M1_MODEL" \
    --stage1 "$STAGE1_MODEL" \
    --index "$INDEX" \
    "$@"
