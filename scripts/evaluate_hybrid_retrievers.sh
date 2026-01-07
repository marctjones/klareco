#!/bin/bash
# Evaluate all active retrievers on hybrid embeddings (128d)
#
# Tests:
#   1. ASTAwareRetriever - Full AST analysis + HNSW prefilter
#   2. HNSWSlotRetriever - HNSW prefilter + mmap slots
#   3. FAISSSlotRetriever - FAISS prefilter + slot rerank
#   4. HybridFAISSMmapRetriever - FAISS + mmap hybrid
#
# PREREQUISITE: Run ./scripts/build_hybrid_mmap_faiss.sh first!
#
# Usage:
#   ./scripts/evaluate_hybrid_retrievers.sh           # Evaluate all retrievers
#   ./scripts/evaluate_hybrid_retrievers.sh --fresh   # Re-run all (ignore checkpoint)
#   ./scripts/evaluate_hybrid_retrievers.sh --retriever ASTAware  # Single retriever
#
# Output:
#   data/benchmarks/results/hybrid_retriever_comparison_YYYYMMDD_HHMMSS.json

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
INDEX_DIR="data/indexes/slot_hybrid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/evaluate_hybrid_$(date +%Y%m%d_%H%M%S).log"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Hybrid Retriever Evaluation"
echo "=========================================="
echo ""
echo "Index directory: $INDEX_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$INDEX_DIR/slot_index.jsonl" ]; then
    echo "ERROR: slot_index.jsonl not found!"
    echo "Run ./scripts/build_hybrid_indexes.sh first"
    exit 1
fi
echo "  ✓ slot_index.jsonl"

if [ ! -d "$INDEX_DIR/hnsw" ]; then
    echo "ERROR: hnsw/ not found!"
    echo "Run ./scripts/build_hnsw_index.sh first"
    exit 1
fi
echo "  ✓ hnsw/ index"

if [ ! -d "$INDEX_DIR/mmap" ]; then
    echo "WARNING: mmap/ not found - HNSW and HybridFAISS retrievers will be skipped"
    echo "Run ./scripts/build_hybrid_mmap_faiss.sh to build"
else
    echo "  ✓ mmap/ arrays"
fi

if [ ! -d "$INDEX_DIR/faiss" ]; then
    echo "WARNING: faiss/ not found - FAISS and HybridFAISS retrievers will be skipped"
    echo "Run ./scripts/build_hybrid_mmap_faiss.sh to build"
else
    echo "  ✓ faiss/ index"
fi

echo ""
echo "Retrievers to evaluate:"
echo "  1. ASTAwareRetriever (requires: hnsw/)"
echo "  2. HNSWSlotRetriever (requires: hnsw/, mmap/)"
echo "  3. FAISSSlotRetriever (requires: faiss/)"
echo "  4. HybridFAISSMmapRetriever (requires: faiss/, mmap/)"
echo ""
echo "Benchmark: 17 retrieval-requiring questions"
echo "Metrics: Recall@1, Recall@5, Recall@10, MRR, Latency"
echo ""
echo "Progress will be logged to: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run evaluation
echo ""
echo "Evaluation started at $(date)"
echo "=========================================="

python scripts/evaluate_hybrid_retrievers.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Results saved to: data/benchmarks/results/hybrid_retriever_comparison_*.json"
echo ""
echo "To view latest results:"
echo "  cat data/benchmarks/results/hybrid_retriever_comparison_*.json | python -m json.tool"
