#!/bin/bash
#
# Full TreeMatchReranker Pipeline
#
# Runs: Data Generation → Training → Evaluation
# Expected time: ~30 minutes total on CPU
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No venv found"; exit 1
fi

echo "========================================="
echo "TreeMatchReranker Full Pipeline"
echo "========================================="
echo

# Configuration
TEST_SET="data/test_sets/qa_test_diverse_30.jsonl"
TRAINING_DATA="data/training/tree_reranker_train.jsonl"
MODEL_DIR="models/tree_reranker"
RESULTS_FILE="results/tree_reranker_eval.json"

NUM_QUESTIONS=100
EXAMPLES_PER_Q=20
EPOCHS=20
BATCH_SIZE=32

# =========================================
# Phase 1: Generate Training Data (~5 min)
# =========================================
echo "Phase 1: Generating Training Data"
echo "=================================="
echo "  Questions: $NUM_QUESTIONS"
echo "  Examples per question: $EXAMPLES_PER_Q"
echo "  Expected total: $((NUM_QUESTIONS * EXAMPLES_PER_Q)) examples"
echo

if [ -f "$TRAINING_DATA" ]; then
    echo "Training data already exists at $TRAINING_DATA"
    read -p "Regenerate? (y/N): " regenerate
    if [ "$regenerate" != "y" ]; then
        echo "Skipping data generation"
    else
        python scripts/generate_tree_reranker_data.py \
            --questions "$TEST_SET" \
            --output "$TRAINING_DATA" \
            --num-questions $NUM_QUESTIONS \
            --examples-per-question $EXAMPLES_PER_Q
    fi
else
    python scripts/generate_tree_reranker_data.py \
        --questions "$TEST_SET" \
        --output "$TRAINING_DATA" \
        --num-questions $NUM_QUESTIONS \
        --examples-per-question $EXAMPLES_PER_Q
fi

echo "✓ Training data generated"
echo

# =========================================
# Phase 2: Train Model (~10-20 min)
# =========================================
echo "Phase 2: Training TreeMatchReranker"
echo "===================================="
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo

python scripts/train_tree_reranker.py \
    --data "$TRAINING_DATA" \
    --output "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-3 \
    --patience 3

echo "✓ Model trained"
echo

# =========================================
# Phase 3: Evaluate Model (~5 min)
# =========================================
echo "Phase 3: Evaluating on Test Set"
echo "================================"
echo

python scripts/evaluate_tree_reranker.py \
    --test-set "$TEST_SET" \
    --model "$MODEL_DIR/best_model.pt" \
    --output "$RESULTS_FILE" \
    --top-k 5

echo "✓ Evaluation complete"
echo

# =========================================
# Summary
# =========================================
echo
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo
echo "Model saved to: $MODEL_DIR/best_model.pt"
echo "Results saved to: $RESULTS_FILE"
echo
echo "Check results:"
echo "  cat $RESULTS_FILE | jq '.overall_accuracy'"
echo
echo "Compare with baselines:"
echo "  No reranker: 73.3%"
echo "  Old MLP reranker: 56.7%"
echo "  Target: 75%+"
echo
