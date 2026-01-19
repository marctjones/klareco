#!/bin/bash
#
# Retrain M1 with improved hyperparameters
#
# This script retrains M1 selectional preferences model with:
# - Larger hidden dimension (256d instead of 128d)
# - More regularization (dropout 0.2 instead of 0.1)
# - Longer patience (20 instead of 10)
#
# Addresses low accuracy issue (70% vs 82% target) identified on 2026-01-18.
#
# Usage:
#   ./scripts/retrain_m1_improved.sh           # Resume from checkpoint
#   ./scripts/retrain_m1_improved.sh --fresh   # Start from scratch
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
LOG_FILE="$LOG_DIR/retrain_m1_improved_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "M1 Improved Retraining"
echo "=============================================="
echo "Configuration:"
echo "  Hidden dimension: 256d (was 128d)"
echo "  Dropout: 0.2 (was 0.1)"
echo "  Patience: 20 (was 10)"
echo "  Learning rate: 0.001 (default)"
echo ""
echo "Expected improvements:"
echo "  - Better capacity for 3-way interactions"
echo "  - Better generalization (more dropout)"
echo "  - More training time before early stop"
echo ""
echo "Logging to: $LOG_FILE"
echo "=============================================="
echo ""

# Run training
python scripts/train_m1_selectional.py \
    --stage1-model models/root_embeddings_tier0/best_model.pt \
    --data-dir data/training/m1_with_tier0 \
    --output-dir models/m1_selectional_tier0 \
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
    echo "✓ M1 improved training complete!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run validation: python scripts/validate_m1_extensive.py"
    echo "  2. Compare to previous: check accuracy vs 70.2% baseline"
    echo "  3. Check logs: $LOG_FILE"
    echo ""
    echo "Model saved to: models/m1_selectional_tier0/best_model.pt"
else
    echo ""
    echo "=============================================="
    echo "❌ M1 improved training failed (exit code: $EXIT_CODE)"
    echo "=============================================="
    echo "Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
