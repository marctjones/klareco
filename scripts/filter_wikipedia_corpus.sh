#!/bin/bash
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
    echo "Error: No venv found"
    exit 1
fi

# Default paths
CORPUS="data/enhanced_corpus/corpus_with_metadata.jsonl"
INDEX="data/indexes/kuzu_index"
OUTPUT="data/enhanced_corpus/corpus_with_metadata.jsonl.filtered"

# Parse arguments
DRY_RUN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p logs

# Generate log filename
LOG_FILE="logs/filter_wikipedia_$(date +%Y%m%d_%H%M%S).log"

echo "Filtering Wikipedia corpus..."
echo "Corpus: $CORPUS"
echo "Index: $INDEX"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo "DRY RUN MODE - no changes will be made"
    echo ""
fi

# Run filtering
python scripts/filter_wikipedia_corpus.py \
    --corpus "$CORPUS" \
    --index "$INDEX" \
    --output "$OUTPUT" \
    $DRY_RUN \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Filtering Complete!"
echo "============================================================"
echo ""
echo "Log: $LOG_FILE"

if [ -z "$DRY_RUN" ]; then
    echo ""
    echo "Next steps:"
    echo "  1. Review filtered corpus: $OUTPUT"
    echo "  2. If satisfied, replace original:"
    echo "     mv $OUTPUT $CORPUS"
    echo "  3. Test RAG quality with filtered corpus"
fi
