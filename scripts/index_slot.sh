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
    echo "No venv found"
    exit 1
fi

# Parse --fresh flag
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    echo "Starting fresh (ignoring checkpoint)"
else
    echo "Resuming from checkpoint (use --fresh to start over)"
fi

# Create logs directory
mkdir -p logs

# Run with logging
LOG_FILE="logs/slot_indexing_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/index_slot_based.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/indexes/slot_full \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Indexing complete! Log saved to: $LOG_FILE"
