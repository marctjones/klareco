#!/bin/bash
# Test hybrid embeddings by loading both models and running tests

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
    echo "Error: No virtual environment found"
    exit 1
fi

echo "=========================================="
echo "Hybrid Embeddings Test"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Load linguistic embeddings (11K roots)"
echo "  2. Load topical embeddings (77K roots)"
echo "  3. Test vocabulary overlap and classification"
echo "  4. Compare similarity in different modes"
echo "  5. Verify proper noun handling"
echo ""

# Check if models exist
LINGUISTIC_MODEL="models/root_embeddings/best_model.pt"
TOPICAL_MODEL="models/topical_embeddings/best_model.pt"

if [ ! -f "$LINGUISTIC_MODEL" ]; then
    echo "Error: Linguistic model not found: $LINGUISTIC_MODEL"
    echo "Run ./scripts/train_roots.sh first"
    exit 1
fi

if [ ! -f "$TOPICAL_MODEL" ]; then
    echo "Error: Topical model not found: $TOPICAL_MODEL"
    echo "Run ./scripts/train_topical_embeddings.sh first"
    exit 1
fi

echo "Models found:"
echo "  ✓ Linguistic: $LINGUISTIC_MODEL"
echo "  ✓ Topical: $TOPICAL_MODEL"
echo ""
echo "Running tests..."
echo ""

# Run test script
python scripts/test_hybrid_embeddings.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
