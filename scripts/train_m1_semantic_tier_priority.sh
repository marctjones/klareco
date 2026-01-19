#!/bin/bash
#
# Complete M1 Training Pipeline with Tier Priority
#
# This script:
# 1. Generates training data with TIER PRIORITY (tier0 first, then tier2, then sample 5/6)
# 2. Trains M1 with the improved data
#
# FIXES ISSUE #12: Tier0 was excluded because max_triples limit was reached
#                  before tier0 appeared in corpus. This version processes
#                  tier0 FIRST to guarantee inclusion.
#
# Usage:
#   ./scripts/train_m1_semantic_tier_priority.sh                # Generate data + train
#   ./scripts/train_m1_semantic_tier_priority.sh --skip-data    # Just train (use existing data)
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

# Parse arguments
SKIP_DATA=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
    esac
done

# Setup paths
CORPUS_PATH="data/enhanced_corpus/corpus_full_with_tier0.jsonl"
DATA_DIR="data/training/m1_semantic_tier_priority"
MODEL_DIR="models/m1_semantic_tier_priority"
MAX_TRIPLES=200000

# Setup logging
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_LOG="$LOG_DIR/prepare_m1_tier_priority_${TIMESTAMP}.log"
TRAIN_LOG="$LOG_DIR/train_m1_tier_priority_${TIMESTAMP}.log"

echo "=============================================================================="
echo "M1 Semantic-Distance Training Pipeline with Tier Priority"
echo "=============================================================================="
echo "Corpus: $CORPUS_PATH"
echo "Output: $MODEL_DIR"
echo ""
echo "TIER PRIORITY STRATEGY:"
echo "  1. Tier 0 (ALL) - PMEG, Krestomatio, Lingvaj Respondoj (~22K triples)"
echo "  2. Tier 2 (ALL) - Fundamento, born-digital high quality"
echo "  3. Tier 5 + 6 (SAMPLE) - Wikipedia + Gutenberg to fill remaining quota"
echo ""
echo "This GUARANTEES tier0 inclusion even if it appears late in corpus!"
echo ""
echo "Expected tier distribution in training data:"
echo "  Tier 0: ~20K (10%) ← FIXED: Previously 0!"
echo "  Tier 2: ~20K (10%)"
echo "  Tier 5: ~80K (40%)"
echo "  Tier 6: ~80K (40%)"
echo ""
echo "Expected: Accuracy 87-88% (vs 86.37% without tier0)"
echo "=============================================================================="
echo ""

# Step 1: Generate tier-prioritized training data (unless skipped)
if [ "$SKIP_DATA" = true ]; then
    echo "Skipping data generation (using existing data)..."
    echo ""
else
    echo "Step 1: Generating tier-prioritized training data..."
    echo "  Logging to: $DATA_LOG"
    echo ""

    if python scripts/prepare_m1_training_data_tier_priority.py \
        --corpus "$CORPUS_PATH" \
        --stage1-model models/root_embeddings_tier0/best_model.pt \
        --output-dir "$DATA_DIR" \
        --max-triples $MAX_TRIPLES \
        --priority-tiers 0 2 \
        --fill-tiers 5 6 \
        --similarity-threshold 0.15 \
        --min-parse-rate 0.0 \
        2>&1 | tee "$DATA_LOG"; then
        echo ""
        echo "✓ Data generation complete"
        echo ""
    else
        echo ""
        echo "✗ Data generation failed"
        echo "Check log: $DATA_LOG"
        exit 1
    fi
fi

# Verify data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ Training data not found: $DATA_DIR/train.jsonl"
    echo "Run without --skip-data to generate it"
    exit 1
fi

# Verify tier0 is included
echo "Verifying tier0 is included in training data..."
TIER0_COUNT=$(jq -r 'select(.source.tier == 0)' "$DATA_DIR/train.jsonl" | wc -l)
echo "  Tier0 examples in training data: $TIER0_COUNT"

if [ "$TIER0_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  WARNING: Still no tier0 in training data!"
    echo "    This should not happen with tier priority."
    echo "    Check the data generation log: $DATA_LOG"
    echo ""
else
    echo "  ✓ Tier0 successfully included!"
    echo ""
fi

# Step 2: Train M1 with tier-prioritized data
echo "Step 2: Training M1 with tier-prioritized data..."
echo "  Logging to: $TRAIN_LOG"
echo ""

if python scripts/train_m1_selectional.py \
    --stage1-model models/root_embeddings_tier0/best_model.pt \
    --data-dir "$DATA_DIR" \
    --output-dir "$MODEL_DIR" \
    --hidden-dim 256 \
    --dropout 0.2 \
    --patience 20 \
    --epochs 50 \
    --fresh \
    2>&1 | tee "$TRAIN_LOG"; then
    echo ""
    echo "=============================================================================="
    echo "✓ M1 tier-prioritized training complete!"
    echo "=============================================================================="
    echo ""
    echo "Results:"
    echo "  Model: $MODEL_DIR/best_model.pt"
    echo "  Data log: $DATA_LOG"
    echo "  Train log: $TRAIN_LOG"
    echo ""
    echo "Tier0 in training: $TIER0_COUNT examples"
    echo ""
    echo "Next steps:"
    echo "  1. Check test accuracy in log above"
    echo "  2. Compare to baseline:"
    echo "     - Without tier0:  86.37%"
    echo "     - With tier0:     [see above]"
    echo "  3. Test with demo:"
    echo "     python scripts/demo_rag_with_m1.py --m1-model $MODEL_DIR/best_model.pt"
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo "✗ M1 training failed"
    echo "=============================================================================="
    echo "Check log: $TRAIN_LOG"
    exit 1
fi
