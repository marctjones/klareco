#!/bin/bash
# Run morpheme-aware embedding training
#
# Usage:
#   ./scripts/run_morpheme_training.sh [--test]  # Use --test for quick testing with limited samples

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if test mode
if [ "$1" == "--test" ]; then
    echo "Running in TEST MODE (limited samples for quick iteration)"
    python3 scripts/train_morpheme_aware.py \
        --corpus data/corpus_enhanced_m1.jsonl \
        --vocab-dir data/vocabularies \
        --output-dir models/morpheme_aware \
        --vocab-size 5000 \
        --embed-dim 128 \
        --composition-method sum \
        --max-samples 10000 \
        --batch-size 16 \
        --epochs 3 \
        --lr 1e-3
else
    echo "Running FULL TRAINING on M1 corpus"
    python3 scripts/train_morpheme_aware.py \
        --corpus data/corpus_enhanced_m1.jsonl \
        --vocab-dir data/vocabularies \
        --output-dir models/morpheme_aware \
        --vocab-size 5000 \
        --embed-dim 128 \
        --composition-method sum \
        --batch-size 32 \
        --epochs 10 \
        --lr 1e-3
fi

echo ""
echo "Training complete! Model saved to models/morpheme_aware/"
echo "View training log: cat models/morpheme_aware/training.log"
