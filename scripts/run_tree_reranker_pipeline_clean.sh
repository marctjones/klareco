#!/bin/bash
#
# TreeMatchReranker Pipeline with Clean Train/Test Split
#
# Uses auto-generated questions from corpus:
# - 150 questions for training data generation
# - 50 questions held out for final evaluation
# - NO DATA LEAKAGE

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
    echo "No venv found"; exit 1
fi

echo "========================================="
echo "TreeMatchReranker Pipeline (Clean Split)"
echo "========================================="
echo

# Configuration
TRAIN_QUESTIONS="data/test_sets/train_questions_150.jsonl"
TEST_QUESTIONS="data/test_sets/test_questions_50.jsonl"
TRAINING_DATA="data/training/tree_reranker_train.jsonl"
MODEL_DIR="models/tree_reranker"
RESULTS_FILE="results/tree_reranker_eval.json"

NUM_EXAMPLES_PER_Q=20
EPOCHS=20
BATCH_SIZE=32

# Phase 1: Generate Training Data
echo "Phase 1: Generating Training Data"
echo "=================================="
echo "  Training questions: 150"
echo "  Examples per question: $NUM_EXAMPLES_PER_Q"
echo "  Expected total: $((150 * NUM_EXAMPLES_PER_Q)) examples"
echo

python scripts/generate_tree_reranker_data.py \
    --questions "$TRAIN_QUESTIONS" \
    --output "$TRAINING_DATA" \
    --num-questions 150 \
    --examples-per-question $NUM_EXAMPLES_PER_Q

echo "✓ Training data generated"
echo

# Phase 2: Train Model
echo "Phase 2: Training TreeMatchReranker"
echo "====================================="
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

# Phase 3: Evaluate on Held-Out Test Set
echo "Phase 3: Evaluating on Test Set"
echo "================================"
echo "  Test questions: 50 (HELD OUT)"
echo

python scripts/evaluate_tree_reranker.py \
    --test-set "$TEST_QUESTIONS" \
    --model "$MODEL_DIR/best_model.pt" \
    --output "$RESULTS_FILE" \
    --top-k 5

echo "✓ Evaluation complete"
echo

# Summary
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
