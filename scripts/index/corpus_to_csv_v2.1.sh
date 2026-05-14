#!/bin/bash
# Shell wrapper for corpus_to_csv_v2.1.py with checkpoint support
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
    echo "ERROR: No Python virtual environment found (.venv or venv)"
    exit 1
fi

# Parse flags
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    shift
fi

# Set paths
CORPUS_FILE="${1:-data/corpus/unified_corpus.jsonl}"
OUTPUT_DIR="${2:-data/csv_export_v2.1_full}"
LOG_DIR="logs/corpus_export"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/corpus_to_csv_v2.1_$(date +%Y%m%d_%H%M%S).log"

echo "=== Corpus to CSV v2.1 Export ==="
echo "Corpus: $CORPUS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

if [[ -n "$FRESH_FLAG" ]]; then
    echo "Starting FRESH export (ignoring checkpoint)"
else
    echo "Resuming from checkpoint (if exists)"
fi
echo ""

# Run with logging
python scripts/index/corpus_to_csv_v2.1.py \
    --corpus "$CORPUS_FILE" \
    --output "$OUTPUT_DIR" \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ CSV export complete"
    echo "  Output: $OUTPUT_DIR"
    echo "  Log: $LOG_FILE"
else
    echo ""
    echo "✗ Export failed (exit code: $EXIT_CODE)"
    echo "  Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
