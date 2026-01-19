#!/bin/bash
# Apply Tesseract OCR to PAG PDF to recover Esperanto diacritics

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

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "========================================================================" echo "TESSERACT OCR NOT INSTALLED"
    echo "========================================================================"
    echo ""
    echo "Tesseract is required to recover Esperanto diacritics from scanned PDFs."
    echo ""
    echo "Install Tesseract with Esperanto language pack:"
    echo "  sudo apt install tesseract-ocr tesseract-ocr-epo"
    echo ""
    echo "Or on macOS:"
    echo "  brew install tesseract tesseract-lang"
    echo ""
    exit 1
fi

# Check if Esperanto language pack is installed
if ! tesseract --list-langs 2>&1 | grep -q "epo"; then
    echo "========================================================================"
    echo "ESPERANTO LANGUAGE PACK NOT INSTALLED"
    echo "========================================================================"
    echo ""
    echo "Tesseract Esperanto language pack is required for best results."
    echo ""
    echo "Install:"
    echo "  sudo apt install tesseract-ocr-epo"
    echo ""
    echo "Continuing with English (may result in poor diacritic recognition)..."
    echo ""
    sleep 3
fi

# Create logs directory
mkdir -p logs

# Run OCR with logging
LOG_FILE="logs/ocr_pag_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "OCR PAG (Plena Analiza Gramatiko)"
echo "========================================================================"
echo ""
echo "Source: data/raw/eo/pag/pag.pdf (32 MB, 590 pages)"
echo "Method: Tesseract OCR with Esperanto language pack"
echo ""
echo "This will take 30-60 minutes to complete."
echo ""
echo "Output:"
echo "  - data/raw/eo/pag/pag_ocr.txt (raw OCR output)"
echo "  - data/raw/eo/pag/pag_ocr_cleaned.txt (cleaned text)"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

python scripts/ocr_pag.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✓ OCR complete!"
echo ""
echo "Next steps:"
echo "  1. Review data/raw/eo/pag/pag_ocr_cleaned.txt for quality"
echo "  2. Compare with pag.txt (PyPDF2 version) to verify OCR accuracy"
echo "  3. If quality is good, use pag_ocr_cleaned.txt for sentence extraction"
