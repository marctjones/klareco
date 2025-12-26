#!/bin/bash
# Build semantic signature index from corpus
# Usage: ./scripts/run_semantic_index_builder.sh [--clean]

set -e

cd "$(dirname "$0")/.."

LOG_FILE="data/semantic_index_build.log"
CORPUS="data/corpus_with_sources_v2.jsonl"
OUTPUT="data/semantic_index"

echo "============================================================"
echo "Building Semantic Signature Index"
echo "============================================================"
echo "Corpus: $CORPUS"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

# Check for --clean flag
CLEAN_FLAG=""
if [ "$1" == "--clean" ]; then
    CLEAN_FLAG="--clean"
    echo "Clean start requested - will delete existing output"
fi

echo "Starting at $(date)"
echo ""

# Run with unbuffered output for real-time logging
python3 -u scripts/build_semantic_index.py \
    --corpus "$CORPUS" \
    --output "$OUTPUT" \
    --batch-size 100000 \
    $CLEAN_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Finished at $(date)"
