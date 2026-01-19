#!/bin/bash
# Generate M1 training data with semantic violations

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
    echo "❌ No venv found"
    exit 1
fi

# Create log directory
LOG_DIR="logs/m1"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_generation_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "M1 SEMANTIC VIOLATIONS DATA GENERATION"
echo "========================================================================"
echo ""
echo "Output: data/training/m1_semantic_violations/"
echo "Log: $LOG_FILE"
echo ""
echo "This will:"
echo "  1. Extract SVO triples from corpus"
echo "  2. Generate semantic violations (70%)"
echo "  3. Generate random corruptions (30%)"
echo "  4. Save train/val/test splits"
echo ""
echo "Estimated time: 2-5 minutes"
echo ""

# Run generation
echo "Starting data generation..."
python scripts/generate_m1_semantic_data.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output-dir data/training/m1_semantic_violations \
    --semantic-categories data/vocabularies/semantic_categories_expanded.json \
    --max-triples 20000 \
    --semantic-ratio 0.7 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ DATA GENERATION COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Next step: Train M1 model with new data"
    echo "  Run: ./scripts/m1_train_selectional.sh --data-dir data/training/m1_semantic_violations"
else
    echo ""
    echo "========================================================================"
    echo "✗ DATA GENERATION FAILED"
    echo "========================================================================"
    echo ""
    echo "Check log: $LOG_FILE"
fi

exit $EXIT_CODE
