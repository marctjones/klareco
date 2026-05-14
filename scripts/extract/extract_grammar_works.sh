#!/bin/bash
# Extract sentences from Tier 0 grammar works (PMEG, PAG, Lingvaj Respondoj)

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
mkdir -p logs/extraction

# Parse flags
FLAGS=""
if [[ "$1" == "--resume" ]]; then
    FLAGS="--resume"
    echo "Resuming from checkpoint..."
elif [[ "$1" == "--fresh" ]]; then
    FLAGS="--fresh"
    echo "Starting fresh (ignoring checkpoint)..."
fi

# Run extraction with logging
LOG_FILE="logs/extraction/grammar_works_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "EXTRACT TIER 0 GRAMMAR WORKS"
echo "========================================================================"
echo ""
echo "Sources:"
echo "  - PMEG v15.5 (Bertilo Wennergren, 2024)"
echo "  - PAG (Kalocsay & Waringhien, 1985)"
echo "  - Lingvaj Respondoj (Zamenhof, 1908)"
echo ""
echo "Extraction includes:"
echo "  - Grammar explanations (descriptive text)"
echo "  - Example sentences (illustrating rules)"
echo ""
echo "Output: data/extracted/eo/tier0/grammar/*.jsonl"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/extract/extract_grammar_works.py $FLAGS 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Extraction complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review sentence classification (example vs explanation)"
    echo "  2. Verify quality of extracted sentences"
    echo "  3. Integrate into unified corpus"
else
    echo ""
    echo "✗ Extraction failed with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
