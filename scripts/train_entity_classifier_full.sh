#!/bin/bash
#
# Full Entity Classifier Training Pipeline
#
# Runs both data generation and training in sequence.
# This is a convenience script that combines:
#   1. generate_entity_training_data.sh
#   2. train_entity_classifier.sh
#
# Usage:
#   ./scripts/train_entity_classifier_full.sh              # Run full pipeline
#   ./scripts/train_entity_classifier_full.sh --fresh      # Regenerate data + train fresh
#   ./scripts/train_entity_classifier_full.sh --skip-data  # Skip data gen, just train
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================================
# Parse arguments
# ============================================================================

DATA_FLAGS=""
TRAIN_FLAGS=""
SKIP_DATA=false

for arg in "$@"; do
    case $arg in
        --fresh)
            DATA_FLAGS="--fresh"
            TRAIN_FLAGS="--fresh"
            ;;
        --skip-data)
            SKIP_DATA=true
            ;;
        --small)
            TRAIN_FLAGS="$TRAIN_FLAGS --small"
            ;;
        --gpu)
            TRAIN_FLAGS="$TRAIN_FLAGS --gpu"
            ;;
        --help)
            echo "Usage: $0 [--fresh] [--skip-data] [--small] [--gpu]"
            echo ""
            echo "Options:"
            echo "  --fresh      Regenerate data and train fresh (ignore checkpoints)"
            echo "  --skip-data  Skip data generation, go straight to training"
            echo "  --small      Use smaller batch size (16 for memory efficiency)"
            echo "  --gpu        Use GPU acceleration"
            echo ""
            echo "This script runs the full pipeline:"
            echo "  1. Generate training data (unless --skip-data)"
            echo "  2. Train entity classifier"
            echo ""
            exit 0
            ;;
    esac
done

# ============================================================================
# Pipeline
# ============================================================================

echo "="*60
echo "FULL ENTITY CLASSIFIER TRAINING PIPELINE"
echo "="*60
echo ""

# Step 1: Generate training data
if [ "$SKIP_DATA" = false ]; then
    echo "STEP 1/2: Generate training data"
    echo "-"*60
    ./scripts/generate_entity_training_data.sh $DATA_FLAGS

    echo ""
    echo "✓ Data generation complete"
    echo ""
else
    echo "Skipping data generation (--skip-data)"
    echo ""
fi

# Step 2: Train model
echo "STEP 2/2: Train entity classifier"
echo "-"*60
./scripts/train_entity_classifier.sh $TRAIN_FLAGS

echo ""
echo "="*60
echo "FULL PIPELINE COMPLETE"
echo "="*60
echo ""
echo "✅ Entity type classifier trained and ready!"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python scripts/evaluate_entity_classifier.py"
echo "  2. Integrate with answer extractor (Task 1.8)"
echo ""

exit 0
