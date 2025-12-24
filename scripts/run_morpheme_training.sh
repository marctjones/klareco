#!/bin/bash
# Run morpheme-aware embedding training
#
# Usage:
#   ./scripts/run_morpheme_training.sh              # Full training (auto-resumes from checkpoint)
#   ./scripts/run_morpheme_training.sh --test       # Test mode (limited samples, 3 epochs)
#   ./scripts/run_morpheme_training.sh --fresh      # Start fresh (ignore checkpoints)
#
# Checkpoint Management:
#   - Automatically resumes from latest_checkpoint.pt if it exists
#   - Saves best model to best_model.pt (use this for production)
#   - Rotates previous best to best_model.prev.pt as backup
#   - Early stopping after 3 epochs without improvement
#
# Model Files:
#   models/morpheme_aware/best_model.pt           - Best model so far (for production use)
#   models/morpheme_aware/best_model.prev.pt      - Previous best (backup)
#   models/morpheme_aware/latest_checkpoint.pt    - Latest epoch (for resume)
#   models/morpheme_aware/training.log            - Training log

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
