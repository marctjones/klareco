#!/bin/bash
# Enhanced Retriever Evaluation with Diagnostic Logging
#
# Evaluates all 4 active retrievers with detailed diagnostic information:
#   1. ASTAwareRetriever - Question classification + entity recognition + pattern matching
#   2. HNSWSlotRetriever - HNSW pre-filter + mmap slot reranking
#   3. FAISSSlotRetriever - FAISS pre-filter + slot reranking
#   4. HybridFAISSMmapRetriever - FAISS + mmap hybrid
#
# Provides detailed logs for understanding WHY retrievers succeed or fail:
#   - Query parsing details (AST, slots extracted, features)
#   - Pre-filter stage results (candidates returned, scores)
#   - Reranking stage results (slot similarities, feature bonuses)
#   - Final ranking with explanations
#
# PREREQUISITE: Run ./scripts/build_hybrid_mmap_faiss.sh first!
#
# Usage:
#   ./scripts/evaluate_retrievers_diagnostic.sh                    # Evaluate all retrievers
#   ./scripts/evaluate_retrievers_diagnostic.sh --fresh            # Re-run all (ignore checkpoint)
#   ./scripts/evaluate_retrievers_diagnostic.sh --retriever HNSW   # Single retriever
#   ./scripts/evaluate_retrievers_diagnostic.sh --diagnostic       # Diagnostic questions only
#   ./scripts/evaluate_retrievers_diagnostic.sh --benchmark        # Benchmark questions only
#
# Output:
#   - Console: Real-time progress with diagnostic details
#   - Log file: logs/evaluate_diagnostic_YYYYMMDD_HHMMSS.log
#   - Results: data/benchmarks/results/diagnostic_retriever_comparison_*.json

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
INDEX_DIR="data/indexes/slot_hybrid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/evaluate_diagnostic_$(date +%Y%m%d_%H%M%S).log"

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
echo "Enhanced Retriever Evaluation (Diagnostic)"
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
echo "  1. ASTAwareRetriever (Question classification + pattern matching)"
echo "  2. HNSWSlotRetriever (HNSW prefilter + mmap slot reranking)"
echo "  3. FAISSSlotRetriever (FAISS prefilter + slot reranking)"
echo "  4. HybridFAISSMmapRetriever (FAISS + mmap combined)"
echo ""
echo "Questions:"
echo "  - Benchmark: 17 retrieval-requiring questions"
echo "  - Diagnostic: 3 questions per retriever (12 total)"
echo ""
echo "Metrics tracked:"
echo "  - Recall@1, Recall@5, Recall@10, MRR"
echo "  - Latency, Memory usage"
echo "  - Per-query diagnostics (parsing, slots, top results)"
echo ""
echo "Diagnostic logging includes:"
echo "  - Query AST structure (roots, slots, question type)"
echo "  - Why each query succeeded or failed"
echo "  - Top result preview for failed queries"
echo ""
echo "Progress will be logged to: $LOG_FILE"
echo ""
echo "To monitor progress in another terminal:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run evaluation
echo ""
echo "Evaluation started at $(date)"
echo "=========================================="

python scripts/evaluate_retrievers_diagnostic.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Results saved to: data/benchmarks/results/diagnostic_retriever_comparison_*.json"
echo ""
echo "To view latest results:"
echo "  cat data/benchmarks/results/diagnostic_retriever_comparison_*.json | python -m json.tool | less"
echo ""
echo "To see just the comparison table:"
echo "  grep -A 50 'RETRIEVER COMPARISON' $LOG_FILE"
