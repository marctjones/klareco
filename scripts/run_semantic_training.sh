#!/bin/bash
#
# Train Semantic Similarity Embeddings
# Run this in a separate terminal
#
# Usage: ./scripts/run_semantic_training.sh
#

set -e

cd /home/marc/klareco

echo "=============================================="
echo "Training Semantic Similarity Embeddings"
echo "=============================================="
echo ""
echo "Training data:"
wc -l data/similarity_pairs_train.jsonl data/similarity_pairs_val.jsonl
echo ""
echo "This trains the compositional Tree-LSTM to predict"
echo "semantic similarity between Esperanto sentence pairs."
echo ""
echo "Key metric: Pearson correlation between predicted and target similarity"
echo "  - Random baseline: ~0.0"
echo "  - Good model: >0.6"
echo "  - Excellent: >0.8"
echo ""

# Run training (use --resume to continue from checkpoint)
python3 -u scripts/train_semantic_similarity.py \
    --train-file data/similarity_pairs_train.jsonl \
    --val-file data/similarity_pairs_val.jsonl \
    --vocab-dir data/vocabularies \
    --output models/semantic_similarity \
    --embed-dim 128 \
    --hidden-dim 256 \
    --output-dim 384 \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --max-train-samples 50000 \
    --max-val-samples 5000 \
    --device auto \
    --resume

echo ""
echo "=============================================="
echo "DONE!"
echo "=============================================="
echo ""
echo "Model saved to: models/semantic_similarity/best_model.pt"
echo ""
echo "Next step: Return to Claude Code to evaluate and integrate."
