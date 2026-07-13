#!/bin/bash
# Acquire PMEG (Plena Manlibro de Esperanta Gramatiko) - Tier 0 authoritative grammar
# Downloads PDF and converts to markdown using pandoc

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
LOG_FILE="logs/acquire_pmeg_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE PMEG (Plena Manlibro de Esperanta Gramatiko)"
echo "========================================================================"
echo ""
echo "Author: Bertilo Wennergren"
echo "Edition: v15.5 (2024)"
echo "License: CC BY-SA 4.0"
echo "Source: https://bertilow.com/pmeg/elshutebla/pmeg15.5.pdf"
echo ""
echo "Output:"
echo "  - data/raw/eo/pmeg/pmeg.pdf (original PDF)"
echo "  - data/raw/eo/pmeg/pmeg.md (structured markdown)"
echo "  - data/raw/eo/pmeg/pmeg.txt (plain text)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire/acquire_pmeg.py $SKIP_DOWNLOAD 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/pmeg/pmeg.md for quality"
echo "  2. Check that grammar tables and section headers are preserved"
echo "  3. Proceed to extraction if quality is good"
