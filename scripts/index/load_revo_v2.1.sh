#!/bin/bash
# Shell wrapper for load_revo_to_kuzu.py with v2.1 Pure Esperanto schema
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
KUZU_DB="${1:-data/indexes/v2.1_kuzu_index_full}"
RELATIONS_JSON="${2:-data/raw/eo/dictionaries/revo/revo_semantic_relations.json}"
TEMP_DIR="${3:-data/indexes/temp_revo}"
LOG_DIR="logs/revo_loading"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/load_revo_$(date +%Y%m%d_%H%M%S).log"

echo "=== Load ReVo Semantic Relations to Kuzu v2.1 ==="
echo "Database: $KUZU_DB"
echo "Relations: $RELATIONS_JSON"
echo "Temp dir: $TEMP_DIR"
echo "Log: $LOG_FILE"
echo ""

if [[ ! -f "$RELATIONS_JSON" ]]; then
    echo "ERROR: ReVo relations file not found: $RELATIONS_JSON"
    echo "Supply --relations pointing at the ReVo semantic-relations JSON,"
    echo "or place it at data/revo/revo_semantic_relations.json"
    exit 1
fi

if [[ ! -e "$KUZU_DB" ]]; then
    echo "ERROR: Kuzu database not found: $KUZU_DB"
    echo "Please run: ./scripts/index/load_csv_to_kuzu_v2.1.sh"
    exit 1
fi

if [[ -n "$FRESH_FLAG" ]]; then
    echo "Starting FRESH load (rebuilding ReVo data)"
else
    echo "Resuming from checkpoint (if exists)"
fi
echo ""

echo "Estimated time: ~5-10 minutes"
echo ""

# Run with logging
python scripts/index/load_revo_to_kuzu.py \
    --kuzu-db "$KUZU_DB" \
    --relations "$RELATIONS_JSON" \
    --temp-dir "$TEMP_DIR" \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ ReVo relations loaded successfully"
    echo "  Database: $KUZU_DB"
    echo "  Log: $LOG_FILE"
else
    echo ""
    echo "✗ Loading failed (exit code: $EXIT_CODE)"
    echo "  Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
