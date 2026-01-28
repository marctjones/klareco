#!/bin/bash
# Evaluate answer extraction on 50-question test set (FULL pipeline with reranking)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Configuration
TEST_SET="data/test_sets/qa_test_set_50.jsonl"
OUTPUT_DIR="data/evaluation"
LOG_FILE="logs/evaluation/extraction_50q_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "============================================================"
echo "RAG Extraction Evaluation (50 questions)"
echo "============================================================"
echo ""
echo "Test set: $TEST_SET"
echo "Output: $OUTPUT_DIR/extraction_50q_results.jsonl"
echo "Log: $LOG_FILE"
echo ""
echo "Pipeline: retrieval + entity boost + quality filter + RERANKING + extraction"
echo "Expected time: ~8-10 minutes"
echo ""
echo "Running evaluation..."
echo ""

# Run evaluation (FULL pipeline - no flags, includes reranking)
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/evaluate_rag_test_set.py" \
    --test-set "$TEST_SET" \
    --output "$OUTPUT_DIR/extraction_50q_results.jsonl" \
    --no-m1 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
echo "Results: $OUTPUT_DIR/extraction_50q_results.jsonl"
echo "Log: $LOG_FILE"
