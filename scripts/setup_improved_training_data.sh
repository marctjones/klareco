#!/bin/bash
#
# Setup Improved Training Data
#
# Complete pipeline to create high-quality training data:
# 1. Extract root vocabulary from corpus
# 2. Filter corpus to only labeled examples
# 3. Generate balanced synthetic examples
# 4. Create balanced dataset
#
# Usage:
#   ./scripts/setup_improved_training_data.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "="*60
echo "SETUP IMPROVED TRAINING DATA PIPELINE"
echo "="*60
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

echo "STEP 1/2: Create root vocabulary"
echo "-"*60
echo ""

python scripts/create_root_vocabulary.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/vocabularies/root_vocab.json \
    --min-frequency 10 \
    --max-roots 1000

echo ""
echo "✓ Root vocabulary created"
echo ""

echo "STEP 2/2: Regenerate training data (improved)"
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
echo "  ✅ Filtered to only examples with tier3_type labels"
echo "  ✅ Generated balanced synthetic examples"
echo "  ✅ Ensured coverage of all entity types"
echo "  ✅ Removed low-confidence examples"
echo ""
echo "Next step: Train on improved data"
echo "  ./scripts/train_entity_classifier_subset.sh --data data/training/entity_classifier_improved"
echo ""
