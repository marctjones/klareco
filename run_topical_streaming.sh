#!/bin/bash
# Run topical data preparation with streaming (memory efficient)
# This version writes pairs to disk immediately instead of accumulating in RAM

set -e
cd "$(dirname "$0")"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found"
    exit 1
fi

# Create directories
mkdir -p logs data/training data/vocabularies

# Run with fresh start (remove old checkpoint and output)
LOG_FILE="logs/topical_streaming_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Topical Data Preparation (STREAMING MODE)"
echo "=========================================="
echo "Memory efficient: writes pairs to disk immediately"
echo "Output: data/training/topical_pairs.jsonl"
echo "Log: $LOG_FILE"
echo ""
echo "Starting..."
echo ""

python scripts/data/prepare_topical_pairs.py \
    --fresh \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/training/topical_pairs.jsonl \
    --vocab-output data/vocabularies/topical_vocab.json \
    --window-size 5 \
    --negative-ratio 5 \
    --checkpoint-interval 500000 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Output: data/training/topical_pairs.jsonl"
echo "Vocabulary: data/vocabularies/topical_vocab.json"
echo "Log: $LOG_FILE"
