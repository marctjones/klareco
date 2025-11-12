# Tree-LSTM Training Guide

**Status**: ✅ Production Ready
**Date**: 2025-11-11
**Training Complete**: 12 epochs completed successfully

---

## Quick Start

### Pre-Flight Check (Always run first!)

```bash
python scripts/preflight_check_training.py --training-data data/training_pairs
```

This validates:
- ✅ All dependencies installed (PyTorch, PyTorch Geometric)
- ✅ Training data files exist and are valid
- ✅ Data format is correct (AST pairs with labels)
- ✅ DataLoader can load batches
- ✅ Model can be created
- ✅ Existing checkpoints (if any)

### Start Fresh Training

```bash
python scripts/train_tree_lstm.py \
    --training-data data/training_pairs \
    --output models/tree_lstm \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.001
```

### Resume Training After Interruption

```bash
python scripts/train_tree_lstm.py \
    --training-data data/training_pairs \
    --output models/tree_lstm \
    --resume auto \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.001
```

**Key point**: `--resume auto` automatically finds and loads the latest checkpoint!

---

## Current Training Status

### Model Performance (12 epochs completed)

| Epoch | Train Loss | Train Accuracy | Notes |
|-------|------------|----------------|-------|
| 1-10  | 0.080      | 91.2%          | Original training |
| 11    | 0.019      | **98.7%**      | After resume |
| 12    | 0.018      | **98.9%**      | Latest |

**Training Data**:
- Positive pairs: 495 (similar sentences)
- Negative pairs: 5,000 (dissimilar sentences)
- Total: 5,495 pairs
- Class ratio: 10.1:1 (negative:positive)

### Model Architecture

```
TreeLSTMEncoder(
  vocab_size=10000
  embed_dim=128
  hidden_dim=256
  output_dim=512
)
Parameters: 1,695,232 trainable
```

---

## Features Implemented

### ✅ Checkpoint Resumption

**Automatic resumption if interrupted/crashed**:
- Saves checkpoint after every epoch
- Stores model state, optimizer state, loss, accuracy
- Training history preserved across sessions
- Use `--resume auto` to continue from last checkpoint

**How it works**:
1. Training saves `checkpoint_epoch_N.pt` after each epoch
2. If interrupted, run with `--resume auto`
3. Script finds latest checkpoint and resumes from there
4. Training continues from next epoch

### ✅ Pre-Flight Validation

**Comprehensive checks before training**:
- Dependency verification (torch, torch-geometric)
- Data file existence and size validation
- Data format validation (first 5 lines of each file)
- DataLoader test (load 10 pairs, create batch)
- Model creation test
- Class balance warnings

**Usage**:
```bash
python scripts/preflight_check_training.py
```

### ✅ Training History

All training runs logged to `training_history.json`:
```json
[
  {
    "epoch": 12,
    "train_loss": 0.018,
    "train_acc": 0.989
  }
]
```

---

## Training Options

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--training-data` | `data/training_pairs` | Directory with training pairs |
| `--output` | `models/tree_lstm` | Output directory for checkpoints |
| `--epochs` | 10 | Number of epochs to train |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab-size` | 10000 | Vocabulary size |
| `--embed-dim` | 128 | Embedding dimension |
| `--hidden-dim` | 256 | LSTM hidden dimension |
| `--output-dim` | 512 | Output embedding dimension |

### Advanced Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--resume` | None | Resume from checkpoint (`auto` or path) |
| `--margin` | 1.0 | Contrastive loss margin |
| `--max-pairs` | None | Limit training pairs (for testing) |
| `--device` | `cpu` | Device (`cpu` or `cuda`) |
| `--seed` | 42 | Random seed |
| `--debug` | False | Enable debug logging |

---

## Overnight Training Checklist

✅ **Before starting training overnight**:

1. **Run pre-flight check**:
   ```bash
   python scripts/preflight_check_training.py
   ```
   Confirm: "✅ ALL CHECKS PASSED - READY FOR TRAINING"

2. **Start training in background** (using nohup or tmux):
   ```bash
   # Option 1: Using nohup
   nohup python scripts/train_tree_lstm.py \
       --training-data data/training_pairs \
       --output models/tree_lstm \
       --resume auto \
       --epochs 50 \
       --batch-size 16 \
       > training.log 2>&1 &

   # Option 2: Using tmux (recommended)
   tmux new -s training
   python scripts/train_tree_lstm.py \
       --training-data data/training_pairs \
       --output models/tree_lstm \
       --resume auto \
       --epochs 50 \
       --batch-size 16
   # Detach with: Ctrl-B, then D
   ```

3. **Monitor progress**:
   ```bash
   # If using nohup:
   tail -f training.log

   # If using tmux:
   tmux attach -t training

   # Check log file:
   tail -f klareco.log
   ```

4. **If training crashes**:
   - Check `klareco.log` for errors
   - Fix the issue
   - Run with `--resume auto` to continue

---

## Training Time Estimates

Based on current hardware (CPU training):

- **Small dataset** (50 pairs, testing): ~1 minute per epoch
- **PoC dataset** (5,495 pairs): ~2.5 minutes per epoch
- **Large dataset** (50,000 pairs): ~25 minutes per epoch

**Overnight training** (8 hours):
- PoC dataset: ~192 epochs
- Large dataset: ~19 epochs

---

## Output Files

### During Training

```
models/tree_lstm/
├── checkpoint_epoch_1.pt    # Checkpoint after epoch 1
├── checkpoint_epoch_2.pt    # Checkpoint after epoch 2
├── ...
├── checkpoint_epoch_N.pt    # Latest checkpoint
├── best_model.pt            # Best model (lowest val loss)
├── training_history.json    # Training history (all epochs)
└── final_model.pt           # Final model after training
```

### Checkpoint Contents

Each checkpoint file contains:
```python
{
    'epoch': 12,
    'model_state_dict': ...,     # Model weights
    'optimizer_state_dict': ..., # Optimizer state
    'train_loss': 0.018,
    'val_loss': 0.0              # (future: validation loss)
}
```

---

## Troubleshooting

### Problem: Training crashes with "Out of Memory"

**Solution**: Reduce batch size
```bash
python scripts/train_tree_lstm.py --batch-size 8  # or 4
```

### Problem: "No checkpoint found" when using --resume auto

**Solution**: Check if checkpoints exist
```bash
ls -l models/tree_lstm/checkpoint_*.pt
```

If no checkpoints, training will start from scratch automatically.

### Problem: Training is too slow

**Solutions**:
1. Use GPU if available: `--device cuda`
2. Reduce batch size for faster iterations: `--batch-size 8`
3. Use smaller model: `--hidden-dim 128 --output-dim 256`
4. Limit training pairs for testing: `--max-pairs 1000`

### Problem: Loss not decreasing

**Solutions**:
1. Check training data quality with pre-flight check
2. Adjust learning rate: `--lr 0.0001` (lower) or `--lr 0.01` (higher)
3. Increase epochs: `--epochs 50`
4. Check class balance (10:1 is acceptable, >20:1 may need adjustment)

---

## Next Steps After Training

### 1. Evaluate GNN vs Baseline

Compare Tree-LSTM encoder against baseline sentence-transformers:

```bash
python scripts/evaluate_rag_comparison.py \
    --baseline data/faiss_baseline \
    --gnn models/tree_lstm \
    --corpus data/ast_corpus
```

### 2. Generate More Training Data (if needed)

If performance is suboptimal, generate larger dataset:

```bash
python scripts/prepare_training_data.py \
    --corpus data/ast_corpus \
    --output data/training_pairs_large \
    --num-pairs 25000 \
    --max-asts 200000
```

### 3. Scale Up Training

Train on larger dataset:

```bash
python scripts/train_tree_lstm.py \
    --training-data data/training_pairs_large \
    --output models/tree_lstm_large \
    --epochs 20 \
    --batch-size 32
```

---

## Success Criteria

### ✅ Training Success

- [x] Loss decreases over epochs
- [x] Accuracy > 90% on training set
- [x] Training completes without errors
- [x] Checkpoints saved successfully

**Current Status**: ✅ **ALL CRITERIA MET** (98.9% accuracy)

### 🎯 Evaluation Success (Next Phase)

- [ ] GNN Precision@5 ≥ Baseline Precision@5
- [ ] GNN recall improvements on structural queries
- [ ] Encoding latency < 100ms per sentence

---

## Technical Details

### Contrastive Loss Function

The model uses contrastive loss to learn semantic similarity:

```python
# For positive pairs (similar): minimize distance
positive_loss = labels * distance²

# For negative pairs (dissimilar): maximize distance up to margin
negative_loss = (1 - labels) * max(0, margin - distance)²

# Total loss
loss = mean(positive_loss + negative_loss)
```

**Margin**: Distance threshold for negative pairs (default: 1.0)

### Tree-LSTM Architecture

The encoder processes AST graphs using Tree-LSTM:

1. **Input**: AST graph (nodes = morphemes, edges = syntax)
2. **Embedding**: Each node → embedding vector
3. **Tree-LSTM**: Bottom-up aggregation respecting tree structure
4. **Output**: Sentence embedding (512-d vector)

**Advantage over baseline**: Captures syntactic structure, not just bag-of-words

---

## FAQ

**Q: Can I run multiple training jobs in parallel?**

A: Yes, but use different output directories:
```bash
python scripts/train_tree_lstm.py --output models/tree_lstm_run1 &
python scripts/train_tree_lstm.py --output models/tree_lstm_run2 &
```

**Q: How do I know training is making progress?**

A: Check:
1. Loss decreasing over epochs
2. Accuracy increasing over epochs
3. Training history file updated after each epoch

**Q: What if I want to start fresh (ignore checkpoints)?**

A: Delete checkpoints or use a new output directory:
```bash
rm models/tree_lstm/checkpoint_*.pt
# or
python scripts/train_tree_lstm.py --output models/tree_lstm_v2
```

**Q: Can I resume from a specific epoch?**

A: Yes:
```bash
python scripts/train_tree_lstm.py \
    --resume models/tree_lstm/checkpoint_epoch_5.pt \
    --epochs 20
```

**Q: How long does training take?**

A: ~2.5 minutes per epoch on CPU for 5K pairs. Overnight training (8 hours) can complete ~192 epochs.

---

## Summary

✅ **Training system is production-ready** with:
- Automatic checkpoint resumption (handles crashes/interruptions)
- Pre-flight validation (catches issues before training)
- Comprehensive logging (training history, checkpoints)
- Flexible configuration (all hyperparameters tunable)

✅ **Current model performance**:
- 12 epochs completed
- 98.9% training accuracy
- 0.018 training loss
- Ready for evaluation against baseline

✅ **Safe for overnight training**:
- Will automatically resume if interrupted
- All progress saved after each epoch
- Complete error handling and logging

---

**Last Updated**: 2025-11-11
**Next Action**: Evaluate GNN encoder vs baseline RAG
**Contact**: See PHASE3_FINAL_SUMMARY.md for full project status
