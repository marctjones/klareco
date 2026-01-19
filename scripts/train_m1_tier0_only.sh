#!/bin/bash
#
# Train M1 on ONLY tier0 data (high quality, small dataset)
#
# This tests whether quality > quantity for selectional preferences.
# Training on 29,874 examples (14,937 tier0/tier1 positive + 14,937 negative)
#
# Usage:
#   ./scripts/train_m1_tier0_only.sh           # Resume from checkpoint
#   ./scripts/train_m1_tier0_only.sh --fresh   # Start from scratch
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Parse --fresh flag
FRESH_FLAG=""
if [[ "$1" == "--fresh" ]]; then
    FRESH_FLAG="--fresh"
    echo "Starting fresh training (ignoring checkpoint)..."
else
    echo "Resuming from checkpoint (if exists)..."
fi

# Setup logging
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_m1_tier0_only_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "M1 Training: Tier0 Only (Quality Test)"
echo "=============================================="
echo "Dataset:"
echo "  Training examples: 23,899"
echo "  Validation: 2,987"
echo "  Test: 2,988"
echo "  Total: 29,874 (14,937 positive + 14,937 negative)"
echo "  Vocabulary: 2,136 nouns, 848 verbs"
echo ""
echo "Hypothesis: High-quality tier0 data will achieve"
echo "            better accuracy despite 13x smaller dataset"
echo "            (29K vs 400K examples)"
echo ""
echo "Configuration:"
echo "  Hidden dimension: 256d (same as previous)"
echo "  Dropout: 0.2"
echo "  Patience: 20"
echo "  Learning rate: 0.001"
echo ""
echo "Logging to: $LOG_FILE"
echo "=============================================="
echo ""

# Run training
python scripts/train_m1_selectional.py \
    --stage1-model models/root_embeddings_tier0/best_model.pt \
    --data-dir data/training/m1_tier0_only \
    --output-dir models/m1_tier0_only \
    --hidden-dim 256 \
    --dropout 0.2 \
    --patience 20 \
    --epochs 50 \
    $FRESH_FLAG \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ M1 tier0-only training complete!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run validation: python scripts/validate_m1_extensive.py --test-data data/training/m1_tier0_only/test.jsonl"
    echo "  2. Compare to baseline:"
    echo "     - Previous (400K mixed quality): 70.2% accuracy"
    echo "     - This (30K tier0 only): [check test results above]"
    echo "  3. Check logs: $LOG_FILE"
    echo ""
    echo "Model saved to: models/m1_tier0_only/best_model.pt"
else
    echo ""
    echo "=============================================="
    echo "❌ M1 tier0-only training failed (exit code: $EXIT_CODE)"
    echo "=============================================="
    echo "Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
