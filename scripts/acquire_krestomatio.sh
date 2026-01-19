#!/bin/bash
# Acquire Fundamenta Krestomatio from Project Gutenberg

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
LOG_FILE="logs/acquire_krestomatio_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE FUNDAMENTA KRESTOMATIO"
echo "========================================================================"
echo ""
echo "Source: Project Gutenberg #8224"
echo "Author: L.L. Zamenhof (1903)"
echo "Format: Plain text UTF-8"
echo "Size: ~855 KB"
echo "Quality: Born-digital (PGDP proofread)"
echo ""
echo "Output: data/raw/eo/gutenberg/krestomatio.txt"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire_krestomatio.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Clean: python scripts/clean_gutenberg.py --input data/raw/eo/gutenberg --output data/cleaned/eo/tier0"
echo "  2. Extract: python scripts/extract_tier0_literary.py"
echo "  3. Integrate into corpus"
