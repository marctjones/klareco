#!/bin/bash
# Train Tree-LSTM with compositional embeddings
# Usage: ./scripts/run_compositional_training.sh

set -e

cd "$(dirname "$0")/.."

LOG_FILE="data/compositional_training.log"
CORPUS="data/corpus_with_sources_v2.jsonl"
VOCAB_DIR="data/vocabularies"
OUTPUT="models/tree_lstm_compositional"

echo "============================================================"
echo "Training Tree-LSTM with Compositional Embeddings"
echo "============================================================"
echo "Corpus: $CORPUS"
echo "Vocabularies: $VOCAB_DIR"
echo "Output: $OUTPUT"
echo "Log: $LOG_FILE"
echo ""
echo "Starting at $(date)"
echo ""

# Run training with unbuffered output
python3 -u scripts/train_compositional_tree_lstm.py \
    --corpus "$CORPUS" \
    --vocab-dir "$VOCAB_DIR" \
    --output "$OUTPUT" \
    --embed-dim 128 \
    --hidden-dim 256 \
    --output-dim 512 \
    --epochs 10 \
    --batch-size 32 \
    --max-samples 50000 \
    --device auto \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Finished at $(date)"
