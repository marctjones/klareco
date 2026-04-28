#!/bin/bash
#
# Merge Tier0 data into unified corpus with proper parse_rate values
#
# This regenerates corpus_full_with_tier0.jsonl by:
# 1. Starting with base corpus (corpus_with_metadata.jsonl)
# 2. Re-parsing and merging tier0 sentences with correct parse_rate field
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Setup paths
BASE_CORPUS="data/enhanced_corpus/corpus_with_metadata.jsonl"
OUTPUT_CORPUS="data/enhanced_corpus/corpus_full_with_tier0.jsonl"
TIER0_DIR="data/extracted/eo/tier0_filtered"
BACKUP_SUFFIX="$(date +%Y%m%d_%H%M%S)"

echo "=============================================================================="
echo "Merge Tier0 into Unified Corpus"
echo "=============================================================================="
echo ""
echo "This will regenerate corpus_full_with_tier0.jsonl with correct parse_rate"
echo "values for all tier0 entries."
echo ""
echo "Base corpus: $BASE_CORPUS"
echo "Tier0 data:  $TIER0_DIR"
echo "Output:      $OUTPUT_CORPUS"
echo ""

# Check if base corpus exists
if [ ! -f "$BASE_CORPUS" ]; then
    echo "❌ Base corpus not found: $BASE_CORPUS"
    exit 1
fi

# Backup existing corpus if it exists
if [ -f "$OUTPUT_CORPUS" ]; then
    BACKUP_FILE="${OUTPUT_CORPUS}.backup_${BACKUP_SUFFIX}"
    echo "📦 Backing up existing corpus to:"
    echo "   $BACKUP_FILE"
    cp "$OUTPUT_CORPUS" "$BACKUP_FILE"
    echo ""
fi

# Copy base corpus as starting point
echo "Step 1: Copying base corpus..."
cp "$BASE_CORPUS" "$OUTPUT_CORPUS"
BASE_COUNT=$(wc -l < "$BASE_CORPUS")
echo "  Base corpus sentences: $BASE_COUNT"
echo ""

# Merge tier0 with corrected parse_rate
echo "Step 2: Merging tier0 data with correct parse_rate..."
echo ""

if python scripts/merge_tier0_into_corpus.py \
    --tier0-dir "$TIER0_DIR" \
    --output "$OUTPUT_CORPUS" \
    --append \
    --fresh; then
    
    echo ""
    echo "=============================================================================="
    echo "✓ Merge Complete!"
    echo "=============================================================================="
    echo ""
    
    TOTAL_COUNT=$(wc -l < "$OUTPUT_CORPUS")
    TIER0_COUNT=$((TOTAL_COUNT - BASE_COUNT))
    
    echo "Results:"
    echo "  Base corpus:  ${BASE_COUNT} sentences"
    echo "  Tier0 added:  ${TIER0_COUNT} sentences"
    echo "  Total:        ${TOTAL_COUNT} sentences"
    echo ""
    echo "Output: $OUTPUT_CORPUS"
    echo "Size:   $(du -h "$OUTPUT_CORPUS" | cut -f1)"
    echo ""
    
    # Verify tier0 has parse_rate values
    echo "Verifying tier0 parse_rate values..."
    TIER0_NULL_COUNT=$(jq -r 'select(.source.tier == 0 and .parse_rate == null)' "$OUTPUT_CORPUS" | wc -l)
    
    if [ "$TIER0_NULL_COUNT" -eq 0 ]; then
        echo "  ✓ All tier0 entries have parse_rate values!"
    else
        echo "  ⚠️  Warning: $TIER0_NULL_COUNT tier0 entries still have null parse_rate"
    fi
    echo ""
    
    echo "Next steps:"
    echo "  1. Test with tier priority training:"
    echo ""
    
else
    echo ""
    echo "❌ Merge failed"
    exit 1
fi
