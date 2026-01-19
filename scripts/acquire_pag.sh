#!/bin/bash
# Acquire PAG (Plena Analiza Gramatiko) - Tier 0 authoritative grammar
# Downloads PDF and converts to markdown using pandoc

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

# Parse flags
SKIP_DOWNLOAD=""
[[ "$1" == "--skip-download" ]] && SKIP_DOWNLOAD="--skip-download"

# Create logs directory
mkdir -p logs

# Run acquisition with logging
LOG_FILE="logs/acquire_pag_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE PAG (Plena Analiza Gramatiko)"
echo "========================================================================"
echo ""
echo "Author: Kálmán Kalocsay, Gaston Waringhien"
echo "Edition: 5th (1985)"
echo "Source: http://luisguillermo.com/PAG/"
echo ""
echo "Output:"
echo "  - data/raw/eo/pag/pag.pdf (original PDF)"
echo "  - data/raw/eo/pag/pag.md (structured markdown)"
echo "  - data/raw/eo/pag/pag.txt (plain text)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire_pag.py $SKIP_DOWNLOAD 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/pag/pag.md for quality"
echo "  2. Check that Esperanto diacritics are preserved"
echo "  3. Proceed to extraction if quality is good"
