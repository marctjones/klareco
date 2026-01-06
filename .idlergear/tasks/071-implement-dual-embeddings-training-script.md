---
id: 71
title: Implement dual embeddings training script
state: closed
created: '2026-01-05T23:03:12.246067Z'
labels:
- training
- 'priority: high'
priority: high
---
**Phase 3: Training - Multi-task training for linguistic + topical embeddings**

## Goal
Create training script that trains both linguistic and topical embeddings jointly or sequentially.

## Implementation

**File:** `scripts/training/train_dual_embeddings.py` (NEW - can copy structure from `train_root_embeddings.py`)

**Training strategies:**

**Option A: Sequential (RECOMMENDED for first attempt)**
1. Load/migrate existing linguistic embedding (64d)
2. Initialize topical embedding with same weights (copy)
3. Freeze linguistic, train only topical for 50 epochs
4. Fine-tune both together for 20 epochs
5. Save dual checkpoint

**Option B: Joint from scratch**
1. Initialize both embeddings randomly
2. Train with multi-task loss
3. Monitor separate correlations

**Loss functions:**
```python
# Linguistic: MSE + margin (existing)
ling_loss = compute_linguistic_loss(model, ling_pairs, mode='linguistic')

# Topical: Binary cross-entropy with negative sampling
topic_loss = compute_topical_loss(model, topic_pairs, mode='topical')

# Combined
total_loss = 0.5 * ling_loss + 0.5 * topic_loss
```

**Key features:**
- Checkpoint resume support
- Separate evaluation for each embedding
- Logging: linguistic_corr, topical_corr, both losses
- Early stopping on combined metric
- Atomic checkpoint saves
- File logging to `logs/training/`

**Training data:**
- Linguistic: From existing `train_root_embeddings.py` data
- Topical: From `data/training/topical_skipgram_pairs.json` (#70)

**Hyperparameters:**
- Learning rate: 0.001 (start)
- Batch size: 256
- Epochs: 100 (or until convergence)
- Loss weights: 0.5/0.5 (tune if needed)

## Acceptance Criteria
- [ ] Script trains dual embeddings
- [ ] Sequential training mode works
- [ ] Joint training mode works (optional)
- [ ] Separate losses logged per epoch
- [ ] Separate correlations computed
- [ ] Checkpoints save correctly
- [ ] Can resume from checkpoint
- [ ] Achieves linguistic_corr > 0.85
- [ ] Achieves topical_corr > 0.65

## Dependencies
- **Blocks:** Evaluation (#72), integration (#73-75)
- **Depends on:** DualRootEmbeddings (#68), topical data (#70)

## Estimated Effort
8-12 hours (including testing both strategies)

## References
Design doc Section 2.2, 2.3

## Success Metrics
After 100 epochs:
- Linguistic correlation: >0.85 (maintain current quality)
- Topical correlation: >0.65 (new objective)
- Combined performance better than single embedding
