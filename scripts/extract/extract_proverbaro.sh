#!/bin/bash
# Extract proverbs from Proverbaro Esperanta (Zamenhof, 1910)

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

# Run extraction with logging
LOG_FILE="logs/extraction/proverbaro_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "EXTRACT PROVERBARO ESPERANTA"
echo "========================================================================"
echo ""
echo "Source: Proverbaro Esperanta (L.L. Zamenhof, 1910)"
echo "Expected: 2,630 proverbs from 98-page scanned book"
echo ""
echo "Format: Numbered proverbs like '1. — La hundo bojas...'"
echo ""
echo "Output:"
echo "  - data/extracted/eo/tier0/proverbaro_sentences.jsonl"
echo "  - data/extracted/eo/tier0/proverbaro_sentences.validation.json"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/extract/extract_proverbaro.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Extraction complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review validation report for gaps/duplicates"
    echo "  2. Check random sample of proverbs for quality"
    echo "  3. Integrate into unified corpus"
else
    echo ""
    echo "✗ Extraction failed with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
