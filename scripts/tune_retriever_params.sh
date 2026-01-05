#!/bin/bash
# Example script demonstrating parameter tuning for retrievers
# This compares different parameter configurations to find optimal settings

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

INDEX_PATH="${1:-data/indexes/slot_full}"
RETRIEVER="${2:-scann}"
OUTPUT_DIR="benchmark_results/parameter_tuning"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Retriever Parameter Tuning"
echo "========================================="
echo "Index: $INDEX_PATH"
echo "Retriever: $RETRIEVER"
echo "Output: $OUTPUT_DIR"
echo ""

# Test 1: Baseline (default parameters)
echo "Test 1: Baseline (prefilter=500, rerank=100)"
python scripts/compare_retrievers.py \
    --index "$INDEX_PATH" \
    --retrievers "$RETRIEVER" \
    -k 10 \
    --prefilter-n 500 \
    --rerank-n 100 \
    --output "$OUTPUT_DIR/baseline.json" \
    2>&1 | grep -E "(Retriever|Fastest|Memory|---)"

echo ""

# Test 2: Fast mode (lower counts)
echo "Test 2: Fast Mode (prefilter=100, rerank=20)"
python scripts/compare_retrievers.py \
    --index "$INDEX_PATH" \
    --retrievers "$RETRIEVER" \
    -k 10 \
    --prefilter-n 100 \
    --rerank-n 20 \
    --output "$OUTPUT_DIR/fast.json" \
    2>&1 | grep -E "(Retriever|Fastest|Memory|---)"

echo ""

# Test 3: Accurate mode (higher counts)
echo "Test 3: Accurate Mode (prefilter=2000, rerank=500)"
python scripts/compare_retrievers.py \
    --index "$INDEX_PATH" \
    --retrievers "$RETRIEVER" \
    -k 10 \
    --prefilter-n 2000 \
    --rerank-n 500 \
    --output "$OUTPUT_DIR/accurate.json" \
    2>&1 | grep -E "(Retriever|Fastest|Memory|---)"

echo ""
echo "========================================="
echo "Tuning Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/baseline.json"
echo "  - $OUTPUT_DIR/fast.json"
echo "  - $OUTPUT_DIR/accurate.json"
echo ""
echo "Summary:"
echo "  Baseline:  Good balance (default)"
echo "  Fast:      2-3x faster, may lose some accuracy"
echo "  Accurate:  2-3x slower, best recall"
echo ""
echo "Next steps:"
echo "  1. Compare results to choose best config"
echo "  2. Run full benchmark suite with chosen params"
echo "  3. Update default values in retriever code"
echo ""
