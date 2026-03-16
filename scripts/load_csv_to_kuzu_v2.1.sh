#!/bin/bash
# Shell wrapper for load_csv_to_kuzu_v2.1_batched.py with checkpoint support
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
CSV_DIR="${1:-data/csv_export_v2.1_full}"
DB_PATH="${2:-data/indexes/v2.1_kuzu_index_full}"
LOG_DIR="logs/kuzu_loading"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/load_v2.1_$(date +%Y%m%d_%H%M%S).log"

echo "=== Load CSV to Kuzu v2.1 Database ==="
echo "CSVs: $CSV_DIR"
echo "Database: $DB_PATH"
echo "Log: $LOG_FILE"
echo ""

if [[ -n "$FRESH_FLAG" ]]; then
    echo "Starting FRESH load (deleting existing database)"
else
    echo "Resuming from checkpoint (if exists)"
fi
echo ""

# Estimate time
if [[ -n "$FRESH_FLAG" ]]; then
    echo "Estimated time: ~2 hours for full load"
    echo "  - Nodes: ~30 minutes"
    echo "  - Relationships: ~30 minutes"
    echo "  - HAVAS_RADIKON: ~60 minutes"
else
    echo "Estimated time: Variable (depends on completed steps)"
fi
echo ""

# Run with logging
python scripts/load_csv_to_kuzu_v2.1_batched.py \
    --csvs "$CSV_DIR" \
    --output "$DB_PATH" \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Database loading complete"
    echo "  Database: $DB_PATH"
    echo "  Log: $LOG_FILE"
else
    echo ""
    echo "✗ Loading failed (exit code: $EXIT_CODE)"
    echo "  Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
