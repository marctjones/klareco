#!/bin/bash
#
# Demo: M1 Query Expansion with Plausibility Filtering
#
# This demonstrates M1's INTENDED PURPOSE: filtering synonym expansions
# BEFORE search to avoid retrieving nonsense documents.
#
# Usage:
#   ./scripts/demo_m1_expansion.sh                          # Run example queries
#   ./scripts/demo_m1_expansion.sh -i                       # Interactive mode
#   ./scripts/demo_m1_expansion.sh "Kiu fondis Esperanton?" # Single query
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
M1_MODEL="models/m1_compositional/best_model.pt"
COMP_MODEL="models/root_embeddings_tier0/best_model.pt"
INDEX="data/indexes/kuzu_index"

# Check if models exist
if [ ! -f "$M1_MODEL" ]; then
    echo "❌ M1 model not found: $M1_MODEL"
    echo "Train M1: ./scripts/train_m1_semantic_tier_priority.sh"
    exit 1
fi

if [ ! -f "$COMP_MODEL" ]; then
    echo "❌ CompositionalEmbedding not found: $COMP_MODEL"
    echo "Train embeddings: ./scripts/train_roots.sh"
    exit 1
fi

if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/kuzu.db" ]; then
    echo "❌ Kuzu index not found: $INDEX/kuzu.db"
    echo "Build index: python scripts/index_kuzu.py"
    exit 1
fi

# Run demo
python scripts/demo_m1_expansion.py \
    --m1-model "$M1_MODEL" \
    --comp-model "$COMP_MODEL" \
    --index "$INDEX" \
    "$@"
