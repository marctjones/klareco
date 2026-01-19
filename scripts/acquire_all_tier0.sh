#!/bin/bash
# Acquire all Tier 0 authoritative Esperanto sources
# Runs all acquisition scripts in recommended priority order

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

# Check required dependencies
echo "Checking dependencies..."
if ! command -v pandoc &> /dev/null; then
    echo "WARNING: pandoc is not installed (required for PDF conversion)"
    echo "Install: sudo apt install pandoc"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python packages
python -c "import beautifulsoup4" 2>/dev/null || pip install beautifulsoup4
python -c "import PyPDF2" 2>/dev/null || pip install PyPDF2

# Create logs directory
mkdir -p logs

# Master log file
MASTER_LOG="logs/acquire_all_tier0_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "ACQUIRE ALL TIER 0 AUTHORITATIVE ESPERANTO SOURCES"
echo "========================================================================"
echo ""
echo "This script will acquire:"
echo "  1. Lingvaj Respondoj (Tekstaro.com - web scraping)"
echo "  2. Project Gutenberg (3 literary works)"
echo "  3. Ekzercaro & Krestomatio (extract from existing Fundamento)"
echo "  4. PMEG (PDF → markdown conversion)"
echo "  5. PAG (PDF → markdown conversion)"
echo "  6. Proverbaro (quality test first, then acquire if acceptable)"
echo ""
echo "Estimated total time: 10-20 minutes"
echo "Master log: $MASTER_LOG"
echo ""
read -p "Press Enter to begin, or Ctrl+C to cancel..."
echo ""

# Function to run script and log results
run_acquisition() {
    local script_name=$1
    local description=$2

    echo "========================================================================"
    echo "[$script_name] $description"
    echo "========================================================================"
    echo ""

    if ./"$script_name"; then
        echo "✓ SUCCESS: $description" | tee -a "$MASTER_LOG"
    else
        echo "✗ FAILED: $description" | tee -a "$MASTER_LOG"
        read -p "Continue with remaining acquisitions? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    echo ""
    echo ""
}

# Track start time
START_TIME=$(date +%s)

# 1. Lingvaj Respondoj (HIGHEST PRIORITY - born-digital, zero OCR errors)
run_acquisition "scripts/acquire_lingvaj_respondoj.sh" "Lingvaj Respondoj from Tekstaro.com"

# 2. Project Gutenberg (EXCELLENT QUALITY - born-digital, PGDP proofread)
run_acquisition "scripts/acquire_gutenberg.sh" "Project Gutenberg literary works"

# 3. Ekzercaro & Krestomatio (QUICK - just extraction from existing file)
run_acquisition "scripts/extract_ekzercaro_krestomatio.sh" "Extract Ekzercaro & Krestomatio"

# 4. PMEG (MODERN GRAMMAR - large PDF, may take time)
run_acquisition "scripts/acquire_pmeg.sh" "PMEG (modern authoritative grammar)"

# 5. PAG (OLDER GRAMMAR - PDF conversion)
run_acquisition "scripts/acquire_pag.sh" "PAG (classic analytic grammar)"

# 6. Proverbaro (NEEDS TESTING FIRST - scanned PDF)
echo "========================================================================"
echo "[Proverbaro] Testing quality before acquisition"
echo "========================================================================"
echo ""
echo "Proverbaro is a 1910 scanned PDF - quality must be tested first."
echo ""

if ./scripts/test_proverbaro_quality.sh; then
    echo ""
    echo "Quality test complete. Review the results above."
    echo ""
    read -p "Proceed with Proverbaro acquisition? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Note: Would need to create acquire_proverbaro.sh if quality is acceptable
        echo "NOTE: If quality is acceptable, create acquire_proverbaro.sh (similar to PAG)"
        echo "For now, skipping full acquisition - manual review required."
    else
        echo "Skipping Proverbaro acquisition." | tee -a "$MASTER_LOG"
    fi
else
    echo "✗ Proverbaro quality test failed - skipping acquisition" | tee -a "$MASTER_LOG"
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Summary
echo ""
echo "========================================================================"
echo "ACQUISITION COMPLETE"
echo "========================================================================"
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Acquired sources:"
echo "  ✓ Lingvaj Respondoj → data/raw/eo/lingvaj_respondoj/"
echo "  ✓ Gutenberg works → data/raw/eo/gutenberg/"
echo "  ✓ Ekzercaro & Krestomatio → data/raw/eo/fundamento/"
echo "  ✓ PMEG → data/raw/eo/pmeg/"
echo "  ✓ PAG → data/raw/eo/pag/"
echo "  ⚠ Proverbaro → data/raw/eo/proverbaro/ (quality test only)"
echo ""
echo "Next steps:"
echo "  1. Review each source for quality (check .txt files)"
echo "  2. Verify Esperanto diacritics are preserved"
echo "  3. Run sentence extraction scripts for each source"
echo "  4. Integrate into unified corpus"
echo ""
echo "Master log: $MASTER_LOG"
echo ""
echo "See individual logs in logs/ directory for details."
