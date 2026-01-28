#!/bin/bash
#
# Demo: Full RAG Pipeline with M1 Filtering + Reranking
#
# This script demonstrates the complete Klareco RAG pipeline:
#   1. AST-aware retrieval (structural matching)
#   2. M1 plausibility filtering (removes nonsense from synonym expansion)
#   3. Neural reranking (learned relevance scoring)
#
# Usage:
#   ./scripts/demo_full_rag.sh                      # Run example queries
#   ./scripts/demo_full_rag.sh "Kiu fondis Esperanton?"  # Single query
#   ./scripts/demo_full_rag.sh --no-m1              # Skip M1 filtering
#   ./scripts/demo_full_rag.sh --no-rerank          # Skip reranking
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

# Check if index exists
if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/kuzu.db" ]; then
    echo "❌ Kuzu index not found: $INDEX/kuzu.db"
    echo "Build index: python scripts/index_kuzu.py"
    exit 1
fi

# Check if reranker exists
if [ ! -f "models/reranker/best_model.pt" ]; then
    echo "❌ Reranker model not found: models/reranker/best_model.pt"
    echo "Train reranker first: ./scripts/train_reranker.sh"
    exit 1
fi

# Check if Stage 1 model exists
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "❌ Stage 1 embeddings not found: $STAGE1_MODEL"
    echo "Train Stage 1: ./scripts/train_roots.sh"
    exit 1
fi

# M1 is optional - warn if missing but continue
if [ ! -f "$M1_MODEL" ]; then
    echo "⚠️  M1 model not found: $M1_MODEL"
    echo "   Demo will run without M1 filtering"
    echo "   Train M1: ./scripts/train_m1_semantic_tier_priority.sh"
    echo ""
fi

echo "=============================================================================="
echo "Full RAG Pipeline Demo"
echo "=============================================================================="
echo "Pipeline: Retrieval → M1 Filtering → Reranking"
echo ""
echo "Models:"
echo "  Index:    $INDEX"
echo "  M1:       $M1_MODEL"
echo "  Stage 1:  $STAGE1_MODEL"
echo "  Reranker: models/reranker/best_model.pt"
echo ""
echo "=============================================================================="
echo ""

# Run demo with all arguments passed through
PYTHONPATH=. python scripts/demo_reranked_rag.py \
    --m1-model "$M1_MODEL" \
    --stage1-model "$STAGE1_MODEL" \
    --index-dir "$INDEX" \
    "$@"
