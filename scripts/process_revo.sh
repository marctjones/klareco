#!/bin/bash
# Complete ReVo processing pipeline: Extract → Validate → Load

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
    echo "Error: No virtual environment found (.venv or venv)"
    exit 1
fi

# Create logs directory
mkdir -p logs/revo

# Master log
MASTER_LOG="logs/revo/process_revo_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "REVO SEMANTIC RELATIONS PIPELINE"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Extract semantic relations from ReVo SQLite database"
echo "  2. Validate extracted relations for quality"
echo "  3. Load relations into Kuzu graph database"
echo ""
echo "Master log: $MASTER_LOG"
echo ""

# Parse --fresh flag
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    echo "Fresh start requested - will re-extract all data"
    echo ""
fi

# Step 1: Extract
echo "========================================================================"
echo "STEP 1: Extracting semantic relations from ReVo database"
echo "========================================================================"
echo ""

LOG_FILE="logs/revo/extract_$(date +%Y%m%d_%H%M%S).log"
if python scripts/extract_revo_semantic_relations.py $FRESH_FLAG 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "✓ Extraction complete"
else
    echo ""
    echo "✗ Extraction failed"
    exit 1
fi

# Step 2: Validate
echo ""
echo "========================================================================"
echo "STEP 2: Validating extracted relations"
echo "========================================================================"
echo ""

LOG_FILE="logs/revo/validate_$(date +%Y%m%d_%H%M%S).log"
if python scripts/validate_revo_relations.py 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "✓ Validation complete"
else
    echo ""
    echo "✗ Validation failed"
    exit 1
fi

# Step 3: Load into Kuzu
echo ""
echo "========================================================================"
echo "STEP 3: Loading relations into Kuzu database"
echo "========================================================================"
echo ""

LOG_FILE="logs/revo/load_$(date +%Y%m%d_%H%M%S).log"
if python scripts/load_revo_to_kuzu.py $FRESH_FLAG 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "✓ Loading complete"
else
    echo ""
    echo "✗ Loading failed"
    exit 1
fi

# Summary
echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "ReVo semantic relations are now loaded into Kuzu!"
echo ""
echo "Query examples:"
echo "  python -c \"import kuzu; db=kuzu.Database('data/indexes/kuzu_index/kuzu.db'); conn=kuzu.Connection(db); print(conn.execute('MATCH (r:Root {root: \\\"dormi\\\"})-[:REVO_SYNONYM]->(s) RETURN s.root').get_next())\""
echo ""
echo "Logs saved to: logs/revo/"
echo ""
