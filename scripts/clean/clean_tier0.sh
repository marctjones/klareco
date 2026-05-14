#!/bin/bash
# Clean newly acquired Tier 0 texts

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

# Create output directory
mkdir -p data/cleaned/eo/tier0

echo "========================================================================"
echo "CLEAN TIER 0 TEXTS"
echo "========================================================================"
echo ""
echo "This will clean:"
echo "  1. Alice in Wonderland - Remove Gutenberg boilerplate"
echo "  2. Fabeloj de Andersen - Remove Gutenberg boilerplate"
echo ""
echo "Skipped (already clean):"
echo "  - PMEG (born-digital)"
echo "  - Lingvaj Respondoj (born-digital)"
echo "  - Ekzercaro (historical metadata acceptable)"
echo "  - Krestomatio (minimal content)"
echo "  - PAG (needs OCR first - run ./scripts/ocr/ocr_pag.sh)"
echo ""

# Clean Gutenberg texts using existing cleaning script
echo "Cleaning Gutenberg texts..."
echo ""

# Create temporary directory with only the files we want to clean
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Copy only Alice and Andersen to temp directory
cp data/raw/eo/gutenberg/gutenberg_17482_la_aventuroj_de_alicio_en_mirlando.txt "$TMP_DIR/alice.txt"
cp data/raw/eo/gutenberg/gutenberg_27915_fabeloj_de_andersen.txt "$TMP_DIR/andersen.txt"

# Clean them
python scripts/clean/clean_gutenberg.py \
    --input "$TMP_DIR" \
    --output data/cleaned/eo/tier0 \
    --verbose

# Copy already-clean texts to tier0 directory (for consistency)
echo ""
echo "Copying already-clean texts..."

# PMEG
cp data/raw/eo/pmeg/pmeg.txt data/cleaned/eo/tier0/pmeg.txt
echo "  Copied: pmeg.txt (1.7M chars)"

# Lingvaj Respondoj
cp data/raw/eo/lingvaj_respondoj/lingvaj_respondoj.txt data/cleaned/eo/tier0/lingvaj_respondoj.txt
echo "  Copied: lingvaj_respondoj.txt (1.0M chars)"

# Ekzercaro
cp data/raw/eo/fundamento/ekzercaro.txt data/cleaned/eo/tier0/ekzercaro.txt
echo "  Copied: ekzercaro.txt (150K chars)"

# Krestomatio
cp data/raw/eo/fundamento/krestomatio.txt data/cleaned/eo/tier0/krestomatio.txt
echo "  Copied: krestomatio.txt (7K chars)"

# Check if PAG OCR output exists
if [ -f "data/raw/eo/pag/pag_ocr.txt" ]; then
    cp data/raw/eo/pag/pag_ocr.txt data/cleaned/eo/tier0/pag.txt
    echo "  Copied: pag.txt (OCR version, 1.4M chars)"
else
    echo "  ⚠ PAG skipped - needs OCR (run ./scripts/ocr/ocr_pag.sh)"
fi

# Check if Proverbaro exists
if [ -f "data/raw/eo/proverbaro/proverbaro.txt" ]; then
    cp data/raw/eo/proverbaro/proverbaro.txt data/cleaned/eo/tier0/proverbaro.txt
    echo "  Copied: proverbaro.txt (2,630 proverbs)"
else
    echo "  ⚠ Proverbaro skipped - not acquired yet (run ./scripts/acquire/acquire_proverbaro.sh)"
fi

echo ""
echo "========================================================================"
echo "CLEANING COMPLETE"
echo "========================================================================"
echo ""
echo "Cleaned files in: data/cleaned/eo/tier0/"
echo ""

# Count files
FILE_COUNT=$(find data/cleaned/eo/tier0 -name "*.txt" | wc -l)
echo "Total files: $FILE_COUNT"
echo ""

# List files
echo "Files ready for extraction:"
ls -lh data/cleaned/eo/tier0/*.txt | awk '{printf "  %-40s %6s\n", $9, $5}'
echo ""

echo "Next step: ./scripts/extract/extract_all_tier0.sh"
