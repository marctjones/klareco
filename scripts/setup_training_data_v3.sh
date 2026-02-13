#!/bin/bash
#
# Setup Training Data V3 - With Expanded Vocabulary and Variations
#
# Improvements over V2:
#   - 512 semantic roots (vs 131 in V2)
#   - 27 categories (vs 17 in V2)
#   - 4 grammatical variations per root (nom/akuz × sing/plur)
#   - Generates 2,048+ semantic examples (hits 50% target!)
#
# Usage:
#   ./scripts/setup_training_data_v3.sh [--size 10000] [--variations 4]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Defaults
TARGET_SIZE=10000
VARIATIONS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            TARGET_SIZE="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--size 10000] [--variations 4]"
            exit 1
            ;;
    esac
done

echo "="
echo "SETUP TRAINING DATA V3"
echo "="
echo ""
echo "Target dataset size: $TARGET_SIZE examples"
echo "Variations per root: $VARIATIONS"
echo ""
echo "Improvements:"
echo "  ✅ 512 semantic roots (vs 131 in V2)"
echo "  ✅ 27 entity categories (vs 17 in V2)"
echo "  ✅ $VARIATIONS grammatical variations per root"
echo "  ✅ Generates ~2,048+ semantic examples (50% target!)"
echo ""
echo "Strategy:"
echo "  - 30% high-confidence (affix-based, trust deterministic)"
echo "  - 50% semantic roots (fill gaps when no affix)"
echo "  - 20% ambiguous (use context to disambiguate)"
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

# Check dependencies
echo "Checking dependencies..."
python -c "import torch_geometric" 2>/dev/null || {
    echo "ERROR: torch-geometric not installed"
    echo "Install with: pip install torch-geometric"
    exit 1
}

# Check semantic roots
SEMANTIC_ROOTS="data/vocabularies/semantic_roots_expanded.json"
if [ ! -f "$SEMANTIC_ROOTS" ]; then
    echo "ERROR: Expanded semantic roots not found: $SEMANTIC_ROOTS"
    exit 1
fi

echo "✓ All dependencies ready"
echo ""

# Ensure root vocab exists
if [ ! -f "data/vocabularies/root_vocab.json" ]; then
    echo "Copying curated common roots to root_vocab.json..."
    cp data/vocabularies/common_roots.json data/vocabularies/root_vocab.json
fi

echo "Running training data generation (V3)..."
echo "-"
echo ""

python scripts/regenerate_training_data_v3.py \
    --corpus-labeled data/training/entity_classifier/enriched_corpus.jsonl \
    --root-vocab data/vocabularies/root_vocab.json \
    --semantic-roots data/vocabularies/semantic_roots_expanded.json \
    --output data/training/entity_classifier_v3 \
    --target-size $TARGET_SIZE \
    --variations-per-root $VARIATIONS

echo ""
echo "="
echo "SETUP COMPLETE"
echo "="
echo ""
echo "✅ V3 training data ready!"
echo ""
echo "What's new in V3:"
echo "  ✅ 512 semantic roots (4x more than V2)"
echo "  ✅ 27 entity categories (expanded coverage)"
echo "  ✅ $VARIATIONS variations per root (case/number combinations)"
echo "  ✅ ~2,048+ semantic examples (hits 50% target!)"
echo ""
echo "Dataset location: data/training/entity_classifier_v3/"
echo ""
echo "Next step: Train on V3 data"
echo "  python scripts/train_entity_classifier.py \\"
echo "      --data data/training/entity_classifier_v3 \\"
echo "      --output models/entity_classifier_v3 \\"
echo "      --epochs 50"
echo ""
