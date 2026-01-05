#!/bin/bash
# Build proper noun dictionary from corpus
# Usage: ./scripts/run_proper_noun_builder.sh [--clean]

set -e

cd "$(dirname "$0")/.."

LOG_FILE="data/proper_noun_build.log"
CORPUS="data/corpus_with_sources_v2.jsonl"
OUTPUT="data/proper_nouns_dynamic.json"

echo "============================================================"
echo "Building Proper Noun Dictionary"
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
python3 -u scripts/build_proper_noun_dict.py \
    --corpus "$CORPUS" \
    --output "$OUTPUT" \
    --min-frequency 3 \
    --batch-size 50000 \
    $CLEAN_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Finished at $(date)"
