#!/bin/bash
# Generate reranker training data
# Usage: ./scripts/generate_reranker_data.sh [synthetic|pattern_mining|both]

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
    echo "No venv found"
    exit 1
fi

# Parse strategy
STRATEGY="${1:-both}"  # Default to 'both'

# Create logs directory
mkdir -p logs/training

# Run data generation with logging
LOG_FILE="logs/training/reranker_data_gen_$(date +%Y%m%d_%H%M%S).log"
echo "Generating reranker training data..."
echo "Strategy: $STRATEGY"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/generate_reranker_training_data.py \
    --strategy "$STRATEGY" \
    --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
    --output data/training/reranker/ \
    --num-samples 30000 \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Data generation complete. Log saved to: $LOG_FILE"
