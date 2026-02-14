#!/bin/bash
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
    echo "No venv found"; exit 1
fi

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default paths
CORPUS="${CORPUS:-data/corpus/unified_corpus.jsonl}"
VOCAB="${VOCAB:-data/vocabularies/root_vocab.json}"
OUTPUT="${OUTPUT:-data/indexes/v2_kuzu_index}"
MAX_ENTRIES="${MAX_ENTRIES:-}"

# Parse flags
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    shift
fi

if [[ "$1" == "--test" ]]; then
    MAX_ENTRIES="100"
    OUTPUT="data/indexes/v2_kuzu_index_test"
    FRESH_FLAG="--fresh"
    shift
fi

# Build command
CMD="python scripts/index_corpus_v2.py --corpus $CORPUS --vocab $VOCAB --output $OUTPUT $FRESH_FLAG"
if [ -n "$MAX_ENTRIES" ]; then
    CMD="$CMD --max-entries $MAX_ENTRIES"
fi

# Run with logging
LOG_FILE="logs/index_corpus_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Indexing corpus into v2.0 Kuzu database..."
echo "Corpus: $CORPUS"
echo "Vocabulary: $VOCAB"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Indexing complete. Log saved to $LOG_FILE"
