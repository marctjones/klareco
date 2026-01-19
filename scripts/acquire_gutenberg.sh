#!/bin/bash
# Acquire Esperanto literature from Project Gutenberg - Tier 1 literary texts
# Downloads selected literary works (born-digital, manually proofread)

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
LOG_FILE="logs/acquire_gutenberg_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE ESPERANTO LITERATURE FROM PROJECT GUTENBERG"
echo "========================================================================"
echo ""
echo "Quality: Born-digital, manually proofread by PGDP team"
echo "License: Public domain"
echo "Source: https://www.gutenberg.org/"
echo ""
echo "Literary works to acquire:"
echo "  - La Aventuroj de Alicio en Mirlando (Alice in Wonderland)"
echo "  - Metamorfozo (The Metamorphosis)"
echo "  - Elektitaj fabeloj (Selected Fairy Tales by Andersen)"
echo ""
echo "Output:"
echo "  - data/raw/eo/gutenberg/*.txt (cleaned texts)"
echo "  - data/raw/eo/gutenberg/*.metadata.json (metadata)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/acquire_gutenberg.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ Acquisition complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/gutenberg/ for quality"
echo "  2. Check that Gutenberg headers/footers were removed"
echo "  3. Proceed to extraction if quality is good"
