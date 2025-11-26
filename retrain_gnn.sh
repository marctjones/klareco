#!/bin/bash
# Retrain Tree-LSTM GNN with new 60K training pairs
# Minimal output - shows only epoch completion (1 line per epoch)

echo "========================================================================"
echo "RETRAINING TREE-LSTM GNN"
echo "========================================================================"
echo ""
echo "Training: 60,000 pairs → models/tree_lstm (20 epochs)"
echo "Old model: Archived to models/tree_lstm_old"
echo "Expected: ~20-30 minutes | Auto-resume: Enabled"
echo ""
echo "========================================================================"
echo "Progress:"
echo ""

# Run training and show only epoch summaries (filters out progress bars)
python scripts/train_tree_lstm.py \
    --training-data data/training_pairs_v2 \
    --output models/tree_lstm \
    --resume auto \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.001 2>&1 | \
    stdbuf -oL tr '\r' '\n' | \
    stdbuf -oL grep -E "^(Epoch |  Train loss:|  Saved checkpoint:|Resumed from epoch|TRAINING COMPLETE)" | \
    stdbuf -oL sed 's/  Train loss:/   /'

echo ""
echo "========================================================================"
echo "✓ Training complete!"
echo "  New model: models/tree_lstm/checkpoint_epoch_20.pt"
echo "  Old model: models/tree_lstm_old/ (archived)"
echo "========================================================================"
echo ""
