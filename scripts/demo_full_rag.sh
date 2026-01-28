#!/bin/bash
#
# Demo: Full RAG Pipeline with Reranking
#
# This script demonstrates the complete Klareco RAG pipeline:
#   1. AST-aware retrieval (structural matching)
#   2. Neural reranking (learned relevance scoring)
#   3. M1 plausibility filtering (optional - removes nonsense from synonym expansion)
#
# Usage:
#   ./scripts/demo_full_rag.sh                      # Run example queries (M1 disabled by default)
#   ./scripts/demo_full_rag.sh "Kiu fondis Esperanton?"  # Single query
#   ./scripts/demo_full_rag.sh --use-m1             # Enable M1 filtering
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

# Parse arguments to check for --use-m1
USE_M1=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--use-m1" ]; then
        USE_M1=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

echo "=============================================================================="
echo "Full RAG Pipeline Demo"
echo "=============================================================================="
if [ "$USE_M1" == "true" ]; then
    echo "Pipeline: Retrieval → M1 Filtering → Reranking"
else
    echo "Pipeline: Retrieval → Reranking (M1 disabled by default)"
fi
echo ""
echo "Models:"
echo "  Index:    $INDEX"
if [ "$USE_M1" == "true" ]; then
    echo "  M1:       $M1_MODEL (enabled)"
else
    echo "  M1:       disabled (use --use-m1 to enable)"
fi
echo "  Stage 1:  $STAGE1_MODEL"
echo "  Reranker: models/reranker/best_model.pt"
echo ""
echo "=============================================================================="
echo ""

# Run demo with arguments
if [ "$USE_M1" == "true" ]; then
    PYTHONPATH=. python scripts/demo_reranked_rag.py \
        --m1-model "$M1_MODEL" \
        --stage1-model "$STAGE1_MODEL" \
        --index-dir "$INDEX" \
        "${EXTRA_ARGS[@]}"
else
    PYTHONPATH=. python scripts/demo_reranked_rag.py \
        --no-m1 \
        --stage1-model "$STAGE1_MODEL" \
        --index-dir "$INDEX" \
        "${EXTRA_ARGS[@]}"
fi
