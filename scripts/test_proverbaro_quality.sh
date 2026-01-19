#!/bin/bash
# Test Proverbaro Esperanta quality before full acquisition
# Downloads sample and checks for OCR errors

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

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed"
    echo "Please install: sudo apt install pandoc"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Run quality test with logging
LOG_FILE="logs/test_proverbaro_quality_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "TEST PROVERBARO QUALITY"
echo "========================================================================"
echo ""
echo "Source: Wikimedia Commons (1910 scanned PDF)"
echo "URL: https://upload.wikimedia.org/wikipedia/commons/6/67/EO_L._L._Zamenhof_-_Proverbaro_Esperanta_1910.pdf"
echo ""
echo "This script will:"
echo "  1. Download the PDF"
echo "  2. Extract sample pages using pandoc"
echo "  3. Check for OCR errors (diacritics, substitutions)"
echo "  4. Display sample proverbs for manual inspection"
echo "  5. Calculate quality score and verdict"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/test_proverbaro_quality.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Quality test complete!"
echo ""
echo "Review the results above and decide:"
echo "  - Score ≥80: Proceed with acquisition"
echo "  - Score 60-79: Proceed with manual correction plan"
echo "  - Score <60: Find alternative source"
