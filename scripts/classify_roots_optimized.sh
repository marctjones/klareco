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

# Parse arguments
RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "Resuming from checkpoint..."
fi

# Create logs directory
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/classification_optimized_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Klareco Root Classification (OPTIMIZED)"
echo "=========================================="
echo ""
echo "Database: data/indexes/v2.1_kuzu_index_full"
echo "Log file: $LOG_FILE"
echo ""
echo "OPTIMIZATIONS:"
echo "  - Transaction batching (1000 UPDATEs per transaction)"
echo "  - Reduces checkpoint overhead"
echo "  - Fetch batches: 10K nodes at a time"
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
echo "Estimated time: ~30-50 minutes (2-3x faster than standard)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "Starting classification..."
echo "Logs will be written to: $LOG_FILE"
echo ""

# Run optimized classification with logging
# -u: unbuffered Python output
# stdbuf -oL -eL: line-buffered output for tee
stdbuf -oL -eL python -u scripts/classify_roots_optimized.py \
    --kuzu data/indexes/v2.1_kuzu_index_full \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

# Check exit code
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Optimized classification completed!"
    echo "=========================================="
    echo ""
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "=========================================="
    echo "✗ Classification failed!"
    echo "=========================================="
    echo ""
    echo "Check log file: $LOG_FILE"
    echo ""
    echo "To resume from checkpoint:"
    echo "  ./scripts/classify_roots_optimized.sh --resume"
    exit 1
fi
