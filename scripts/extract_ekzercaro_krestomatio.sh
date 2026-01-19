#!/bin/bash
# Extract Ekzercaro and Krestomatio from existing Fundamento file
# These sections are already in our corpus, just need to be separated

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

# Run extraction with logging
LOG_FILE="logs/extract_ekzercaro_krestomatio_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "EXTRACT EKZERCARO & KRESTOMATIO FROM FUNDAMENTO"
echo "========================================================================"
echo ""
echo "Source: data/raw/eo/fundamento/fundamento_de_esperanto.txt"
echo "Author: L.L. Zamenhof"
echo ""
echo "Sections to extract:"
echo "  - Ekzercaro (exercises)"
echo "  - Fundamenta Krestomatio (readings)"
echo ""
echo "Output:"
echo "  - data/raw/eo/fundamento/ekzercaro.txt"
echo "  - data/raw/eo/fundamento/krestomatio.txt"
echo "  - data/raw/eo/fundamento/extraction_metadata.json"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/extract_ekzercaro_krestomatio.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Extraction complete!"
echo ""
echo "Next steps:"
echo "  1. Review extracted files for quality"
echo "  2. Verify section boundaries are correct"
echo "  3. Proceed to sentence extraction"
