#!/bin/bash
# Acquire Lingvaj Respondoj from Tekstaro.com - Tier 0 authoritative grammar Q&A
# Scrapes HTML and extracts clean text

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
mkdir -p logs

# Run acquisition with logging
LOG_FILE="logs/acquire_lingvaj_respondoj_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE LINGVAJ RESPONDOJ FROM TEKSTARO.COM"
echo "========================================================================"
echo ""
echo "Author: L.L. Zamenhof"
echo "Editor: Gaston Waringhien"
echo "Edition: 7th (1990)"
echo "Source: https://tekstaro.com/t?nomo=lingvaj-respondoj"
echo "Quality: Born-digital, zero OCR errors"
echo ""
echo "Output:"
echo "  - data/raw/eo/lingvaj_respondoj/lingvaj_respondoj.html (original HTML)"
echo "  - data/raw/eo/lingvaj_respondoj/lingvaj_respondoj_raw.txt (raw extracted text)"
echo "  - data/raw/eo/lingvaj_respondoj/lingvaj_respondoj.txt (cleaned text)"
echo "  - data/raw/eo/lingvaj_respondoj/lingvaj_respondoj.metadata.json (metadata)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire_lingvaj_respondoj.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/lingvaj_respondoj/lingvaj_respondoj.txt for quality"
echo "  2. Check that sections and numbering are preserved"
echo "  3. Proceed to extraction if quality is good"
