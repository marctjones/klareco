#!/bin/bash
#
# Setup Training Data V2 - Redesigned Approach
#
# Generates three types of examples:
#   30% High-confidence (affix-based, trust deterministic)
#   50% Semantic roots (fill gaps when no affix)
#   20% Ambiguous (use context to disambiguate)
#
# Usage:
#   ./scripts/setup_training_data_v2.sh [--size 10000]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default target size
TARGET_SIZE=10000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            TARGET_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "="
echo "SETUP TRAINING DATA V2"
echo "="
echo ""
echo "Target dataset size: $TARGET_SIZE examples"
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

# Ensure semantic roots exist
SEMANTIC_ROOTS="data/vocabularies/semantic_roots.json"
if [ ! -f "$SEMANTIC_ROOTS" ]; then
    echo "ERROR: Semantic roots not found: $SEMANTIC_ROOTS"
    echo "This file should have been created with the redesign."
    exit 1
fi

echo "✓ All dependencies ready"
echo ""

# Ensure root vocab exists (use curated common roots)
if [ ! -f "data/vocabularies/root_vocab.json" ]; then
    echo "Copying curated common roots to root_vocab.json..."
    cp data/vocabularies/common_roots.json data/vocabularies/root_vocab.json
fi

echo "Running training data generation (V2)..."
echo "-"
echo ""

python scripts/regenerate_training_data_v2.py \
    --corpus-labeled data/training/entity_classifier/enriched_corpus.jsonl \
    --root-vocab data/vocabularies/root_vocab.json \
    --semantic-roots data/vocabularies/semantic_roots.json \
    --output data/training/entity_classifier_v2 \
    --target-size $TARGET_SIZE

echo ""
echo "="
echo "SETUP COMPLETE"
echo "="
echo ""
echo "✅ V2 training data ready!"
echo ""
echo "What's different:"
echo "  ✅ 30% affix-based (teach: trust deterministic when confident)"
echo "  ✅ 50% semantic roots (teach: fill semantic gap)"
echo "  ✅ 20% ambiguous (teach: use context to disambiguate)"
echo ""
echo "Dataset location: data/training/entity_classifier_v2/"
echo ""
echo "Next step: Train on V2 data"
echo "  python scripts/train_entity_classifier.py \\"
echo "      --data data/training/entity_classifier_v2 \\"
echo "      --output models/entity_classifier \\"
echo "      --epochs 50"
echo ""
