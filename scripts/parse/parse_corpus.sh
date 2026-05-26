#!/bin/bash
#
# Parse Corpus - Build Unified Corpus from Extracted Sources
#
# This script reads all extracted sentence files (tier0-6) and builds
# a unified corpus with ASTs, parse rates, and preserved metadata.
#
# Features:
# - Checkpoint/resume support for restartability
# - Progress tracking with statistics
# - Proper parse_rate calculation
# - Tier and quality metadata preservation
#
# Usage:
#   ./scripts/parse/parse_corpus.sh                # Fresh build
#   ./scripts/parse/parse_corpus.sh --resume       # Resume from checkpoint
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Preflight: refuse to start without enough disk for the corpus output
"$PROJECT_ROOT/scripts/util/preflight_disk.sh" 30 "parse_corpus writes ~20 GB AST corpus" || exit 1

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Parse arguments
RESUME_FLAG=""
FRESH_FLAG="--fresh"

for arg in "$@"; do
    case $arg in
        --resume)
            RESUME_FLAG="--resume"
            FRESH_FLAG=""
            shift
            ;;
        --fresh)
            FRESH_FLAG="--fresh"
            RESUME_FLAG=""
            shift
            ;;
    esac
done

# Setup paths
OUTPUT_FILE="data/enhanced_corpus/corpus_with_metadata.jsonl"
LOG_DIR="logs/corpus"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/build_corpus_${TIMESTAMP}.log"

echo "=============================================================================="
echo "Build Unified Corpus"
echo "=============================================================================="
echo ""
echo "This will parse all extracted sentences and build a unified corpus with:"
echo "  - Current parser (with proper noun/correlative fixes)"
echo "  - Proper parse_rate calculation"
echo "  - Tier and quality metadata"
echo ""
echo "Sources:"
echo "  Tier 0: Authoritative grammar texts (PMEG, Lingvaj Respondoj, etc.)"
echo "  Tier 5: Wikipedia"
echo "  Tier 6: Gutenberg books"
echo ""
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG_FILE"
echo ""
echo "Estimated time: 1-3 hours (depends on corpus size)"
echo ""

# Check if extracted data exists
if [ ! -d "data/extracted" ]; then
    echo "❌ No extracted data found at data/extracted/"
    echo "   Run extraction first:"
    echo "     ./scripts/extract/extract_all.sh"
    exit 1
fi

# Run corpus builder
echo "Starting corpus build..."
echo ""

if python scripts/parse/build_unified_corpus.py \
    --output "$OUTPUT_FILE" \
    $FRESH_FLAG $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"; then

    echo ""
    echo "=============================================================================="
    echo "✓ Corpus Build Complete!"
    echo "=============================================================================="
    echo ""
    echo "Output: $OUTPUT_FILE"
    echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Sentences: $(wc -l < "$OUTPUT_FILE" | tr -d ' ')"
    echo "Log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Export corpus to CSV:"
    echo "     ./scripts/index/corpus_to_csv_v2.1.sh"
    echo ""
    echo "  2. Load into Kuzu v2.1:"
    echo "     ./scripts/index/load_csv_to_kuzu_v2.1.sh"
    echo ""
    echo "  (Or run ./scripts/index/reindex_kuzu_v2.1.sh which chains both.)"
    echo ""

else
    echo ""
    echo "=============================================================================="
    echo "✗ Corpus build failed"
    echo "=============================================================================="
    echo "Check log: $LOG_FILE"
    exit 1
fi
