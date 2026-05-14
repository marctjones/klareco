#!/bin/bash
# Shell wrapper for validate_kuzu_v2.1.py
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

# Parse mode
MODE="--quick"
if [[ "$1" == "--thorough" ]]; then
    MODE="--thorough"
    shift
fi

# Set paths
DB_PATH="${1:-data/indexes/v2.1_kuzu_index_full}"
LOG_DIR="logs/validation"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

echo "=== Kuzu v2.1 Database Validation ==="
echo "Database: $DB_PATH"
echo "Mode: $MODE"
echo "Log: $LOG_FILE"
echo ""

if [[ ! -d "$DB_PATH" ]]; then
    echo "ERROR: Database not found: $DB_PATH"
    exit 1
fi

if [[ "$MODE" == "--quick" ]]; then
    echo "Running QUICK validation (~30 seconds)"
else
    echo "Running THOROUGH validation (~5-10 minutes)"
fi
echo ""

# Run validation with logging
python scripts/validate/validate_kuzu_v2.1.py \
    --db "$DB_PATH" \
    $MODE \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Validation passed"
    echo "  Log: $LOG_FILE"
else
    echo ""
    echo "✗ Validation failed (exit code: $EXIT_CODE)"
    echo "  Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
