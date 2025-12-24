#!/bin/bash
# Extract root vocabulary from corpus for compositional embeddings
# Usage: ./scripts/run_root_extraction.sh [--clean]

set -e

cd "$(dirname "$0")/.."

LOG_FILE="data/root_extraction.log"
CORPUS="data/corpus_with_sources_v2.jsonl"
OUTPUT="data/vocabularies"

echo "============================================================"
echo "Extracting Root Vocabulary for Compositional Embeddings"
echo "============================================================"
echo "Corpus: $CORPUS"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

# Check for --clean flag
CLEAN_FLAG=""
if [ "$1" == "--clean" ]; then
    CLEAN_FLAG="--clean"
    echo "Clean start requested"
fi

echo "Starting at $(date)"
echo ""

# Run with unbuffered output
python3 -u scripts/extract_root_vocabulary.py \
    --corpus "$CORPUS" \
    --output "$OUTPUT" \
    --min-frequency 2 \
    --batch-size 100000 \
    $CLEAN_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Finished at $(date)"
