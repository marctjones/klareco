#!/bin/bash
# Extract sentences from Tier 0 literary works (Gutenberg, Ekzercaro, Krestomatio)

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
LOG_FILE="logs/extraction/tier0_literary_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "EXTRACT TIER 0 LITERARY WORKS"
echo "========================================================================"
echo ""
echo "Sources:"
echo "  - Alice in Wonderland (Lewis Carroll)"
echo "  - Fabeloj de Andersen (H.C. Andersen)"
echo "  - Ekzercaro (Zamenhof)"
echo "  - Fundamenta Krestomatio (Zamenhof)"
echo ""
echo "Output: data/extracted/eo/tier0/literary/*.jsonl"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/extract_tier0_literary.py $FLAGS 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Extraction complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review extracted sentences for quality"
    echo "  2. Run extract_grammar_works.sh for grammar sources"
    echo "  3. Run extract_proverbaro.sh for proverbs"
else
    echo ""
    echo "✗ Extraction failed with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
