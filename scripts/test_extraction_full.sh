#!/bin/bash
# Test RAG pipeline on full 50-question test set with comprehensive diagnostics
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse arguments
VERBOSE_FLAG=""
if [[ "$*" == *"--verbose"* ]] || [[ "$*" == *"-v"* ]]; then
    VERBOSE_FLAG="--verbose"
fi

echo "============================================================"
echo "RAG Pipeline Evaluation with Comprehensive Diagnostics"
echo "============================================================"
echo ""
echo "Pipeline: retrieval + entity boost + quality filter + reranker + extraction"
echo "Output: Compact status lines + bottleneck analysis"
if [ -n "$VERBOSE_FLAG" ]; then
    echo "Mode: VERBOSE (showing detailed entity boost/quality filter logs)"
else
    echo "Mode: COMPACT (use --verbose or -v for detailed logs)"
fi
echo ""

# Run evaluation with extraction and comprehensive reporting
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/evaluate_rag_test_set.py" \
    --test-set data/test_sets/qa_test_set_50.jsonl \
    --output /tmp/results_with_extraction.jsonl \
    --no-m1 \
    $VERBOSE_FLAG

echo ""
echo "============================================================"
echo "Results saved to: /tmp/results_with_extraction.jsonl"
echo "============================================================"
