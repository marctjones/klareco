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
    echo "ERROR: No venv found (.venv or venv)"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/classification_copy_from_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "Klareco Root Classification (FASTEST!)"
echo "============================================"
echo ""
echo "Database: data/indexes/v2.1_kuzu_index_full"
echo "Log file: $LOG_FILE"
echo ""
echo "ULTIMATE OPTIMIZATION:"
echo "  ✓ Classify all nodes in Python (fast dict lookups)"
echo "  ✓ Write to CSV"
echo "  ✓ COPY FROM into temp table (Kuzu's fastest operation)"
echo "  ✓ Single UPDATE with JOIN (ONE checkpoint!)"
echo "  ✓ No more 1.2M individual transactions!"
echo ""
echo "Classification:"
echo "  - Tier 0: 190 grammatical words (10 subcategories)"
echo "  - Tier 1a: 787 Unua Libro lexical roots"
echo "  - Tier 1b: ~1,500 Fundamento extended"
echo "  - Tier 2: ~21K ReVo technical terms"
echo "  - Tier 3: ~66K corpus-validated"
echo "  - Tier 4: ~69K proper names"
echo "  - Tier 5: ~1M parse failures"
echo "  - Tier 6: Unknown/unclassified"
echo ""
echo "Properties set:"
echo "  - nivelo: Tier classification (grammatical role)"
echo "  - fonto: Historical source (provenance)"
echo "  - ofteco: Usage frequency"
echo ""
echo "Estimated time: ~10-30 seconds total!"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "Starting classification..."
echo "Logs will be written to: $LOG_FILE"
echo ""

# Run COPY FROM classification with logging
# -u: unbuffered Python output
# stdbuf -oL -eL: line-buffered output for tee
stdbuf -oL -eL python -u scripts/classify_roots_copy_from.py \
    --kuzu data/indexes/v2.1_kuzu_index_full \
    2>&1 | tee "$LOG_FILE"

# Check exit code
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ FASTEST classification completed!"
    echo "============================================"
    echo ""
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "============================================"
    echo "✗ Classification failed!"
    echo "============================================"
    echo ""
    echo "Check log file: $LOG_FILE"
    exit 1
fi
