#!/bin/bash
# Master script to extract sentences from all Tier 0 sources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create master log
mkdir -p logs/extraction
MASTER_LOG="logs/extraction/tier0_master_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================" | tee -a "$MASTER_LOG"
echo "EXTRACT ALL TIER 0 SOURCES" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "This will extract sentences from:" | tee -a "$MASTER_LOG"
echo "  1. Literary works (Alice, Andersen, Ekzercaro, Krestomatio)" | tee -a "$MASTER_LOG"
echo "  2. Grammar works (PMEG, PAG, Lingvaj Respondoj)" | tee -a "$MASTER_LOG"
echo "  3. Proverbaro Esperanta (2,630 proverbs)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Output: data/extracted/eo/tier0/" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Helper function to run extraction and track success
run_extraction() {
    local script_name=$1
    local description=$2

    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "RUNNING: $description" | tee -a "$MASTER_LOG"
    echo "========================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    if ./"$script_name"; then
        echo "" | tee -a "$MASTER_LOG"
        echo "✓ SUCCESS: $description" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
        return 0
    else
        echo "" | tee -a "$MASTER_LOG"
        echo "✗ FAILED: $description" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"

        read -p "Continue with remaining extractions? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 1
    fi
}

# Run extractions in priority order

# 1. Literary works (highest quality - born digital)
run_extraction "scripts/extract_tier0_literary.sh" "Literary works (Gutenberg + Fundamento)"

# 2. Grammar works (authoritative references)
run_extraction "scripts/extract_grammar_works.sh" "Grammar works (PMEG, PAG, Lingvaj Respondoj)"

# 3. Proverbaro (scanned, may have OCR issues)
run_extraction "scripts/extract_proverbaro.sh" "Proverbaro Esperanta (proverbs)"

# Final summary
echo "" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "EXTRACTION COMPLETE" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Count total sentences extracted
TOTAL_FILES=$(find data/extracted/eo/tier0 -name "*.jsonl" 2>/dev/null | wc -l)
TOTAL_SENTENCES=0

if [ $TOTAL_FILES -gt 0 ]; then
    for file in data/extracted/eo/tier0/**/*.jsonl; do
        if [ -f "$file" ]; then
            COUNT=$(wc -l < "$file" 2>/dev/null || echo 0)
            TOTAL_SENTENCES=$((TOTAL_SENTENCES + COUNT))
            BASENAME=$(basename "$file" .jsonl)
            echo "  ✓ $BASENAME: $COUNT sentences" | tee -a "$MASTER_LOG"
        fi
    done
fi

echo "" | tee -a "$MASTER_LOG"
echo "Total extracted: $TOTAL_SENTENCES sentences from $TOTAL_FILES files" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Output directory: data/extracted/eo/tier0/" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Next steps:" | tee -a "$MASTER_LOG"
echo "  1. Review extraction quality (random sampling)" | tee -a "$MASTER_LOG"
echo "  2. Validate sentence parsing (run parser on samples)" | tee -a "$MASTER_LOG"
echo "  3. Integrate into unified corpus (merge with existing data)" | tee -a "$MASTER_LOG"
echo "  4. Update corpus statistics and documentation" | tee -a "$MASTER_LOG"
