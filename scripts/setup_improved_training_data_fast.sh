#!/bin/bash
#
# Setup Improved Training Data (Fast Version)
#
# Uses curated common roots instead of corpus extraction.
# Completes in 2-3 minutes instead of 15+ minutes.
#
# Usage:
#   ./scripts/setup_improved_training_data_fast.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "="*60
echo "SETUP IMPROVED TRAINING DATA (FAST)"
echo "="*60
echo ""
echo "Using curated Fundamento + common roots (250 roots)"
echo "Much faster than corpus extraction (2-3 min vs 15+ min)"
echo ""

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

# Check if curated roots exist
CURATED_ROOTS="data/vocabularies/common_roots.json"
if [ ! -f "$CURATED_ROOTS" ]; then
    echo "ERROR: Curated roots not found: $CURATED_ROOTS"
    exit 1
fi

echo "✓ Using curated root vocabulary: $CURATED_ROOTS"
echo ""

# Copy to standard location
cp "$CURATED_ROOTS" data/vocabularies/root_vocab.json
echo "✓ Copied to: data/vocabularies/root_vocab.json"
echo ""

echo "Regenerate training data (improved)"
echo "-"*60
echo ""

python scripts/regenerate_training_data_improved.py \
    --corpus-labeled data/training/entity_classifier/enriched_corpus.jsonl \
    --root-vocab data/vocabularies/root_vocab.json \
    --output data/training/entity_classifier_improved \
    --max-per-type 1000 \
    --synthetic-per-type 500

echo ""
echo "="*60
echo "SETUP COMPLETE"
echo "="*60
echo ""
echo "✅ Improved training data ready!"
echo ""
echo "What changed:"
echo "  ✅ Used curated Fundamento + common roots (fast!)"
echo "  ✅ Filtered to only examples with tier3_type labels"
echo "  ✅ Generated balanced synthetic examples"
echo "  ✅ Ensured coverage of all entity types"
echo "  ✅ Removed low-confidence examples"
echo ""
echo "Dataset location: data/training/entity_classifier_improved/"
echo ""
echo "Next step: Train on improved data"
echo "  python scripts/train_entity_classifier.py \\"
echo "      --data data/training/entity_classifier_improved \\"
echo "      --output models/entity_classifier \\"
echo "      --epochs 50"
echo ""
