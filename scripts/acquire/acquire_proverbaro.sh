#!/bin/bash
# Acquire Proverbaro Esperanta from Wikimedia Commons - Tier 0 proverbs collection

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

# Parse flags
SKIP_DOWNLOAD=""
[[ "$1" == "--skip-download" ]] && SKIP_DOWNLOAD="--skip-download"

# Create logs directory
mkdir -p logs

# Run acquisition with logging
LOG_FILE="logs/acquire_proverbaro_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE PROVERBARO ESPERANTA"
echo "========================================================================"
echo ""
echo "Author: L.L. Zamenhof (arranged from Mark Zamenhof)"
echo "Edition: 1910"
echo "Source: Wikimedia Commons"
echo "Expected: 2,630 proverbs, 98 pages"
echo ""
echo "Output:"
echo "  - data/raw/eo/proverbaro/proverbaro.pdf (original PDF)"
echo "  - data/raw/eo/proverbaro/proverbaro_raw.txt (raw extraction)"
echo "  - data/raw/eo/proverbaro/proverbaro.txt (cleaned text)"
echo "  - data/raw/eo/proverbaro/proverbaro.metadata.json (metadata)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire/acquire_proverbaro.py $SKIP_DOWNLOAD 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/proverbaro/proverbaro.txt for quality"
echo "  2. Verify proverb count is close to 2,630"
echo "  3. Proceed to extraction if quality is good"
