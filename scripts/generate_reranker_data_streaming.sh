#!/bin/bash
# Generate reranker training data (STREAMING VERSION with checkpoints)
# Usage:
#   ./scripts/generate_reranker_data_streaming.sh [synthetic|pattern_mining|both]
#   ./scripts/generate_reranker_data_streaming.sh --resume
#   ./scripts/generate_reranker_data_streaming.sh --fresh

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

# Parse arguments
STRATEGY="both"
RESUME_FLAG=""
FRESH_FLAG=""

for arg in "$@"; do
    case $arg in
        synthetic|pattern_mining|both)
            STRATEGY="$arg"
            ;;
        --resume)
            RESUME_FLAG="--resume"
            ;;
        --fresh)
            FRESH_FLAG="--fresh"
            ;;
    esac
done

# Create logs directory
mkdir -p logs/training

# Run data generation with logging
LOG_FILE="logs/training/reranker_data_gen_$(date +%Y%m%d_%H%M%S).log"
echo "Generating reranker training data (STREAMING VERSION)..."
echo "Strategy: $STRATEGY"
echo "Logging to: $LOG_FILE"
echo ""

python scripts/generate_reranker_training_data_streaming.py \
    --strategy "$STRATEGY" \
    --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
    --output data/training/reranker/ \
    --num-synthetic 30000 \
    --num-mined 20000 \
    --seed 42 \
    $RESUME_FLAG $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Data generation complete. Log saved to: $LOG_FILE"
