---
id: 87
title: Consider training topical embeddings for 15-20 epochs
state: open
created: '2026-01-06T06:24:55.230504Z'
labels:
- enhancement
- embeddings
- training
priority: low
---
## Observation

Current topical embeddings were trained for 10 epochs with loss still decreasing:

```
Epoch 1:  0.1448
Epoch 5:  0.0507
Epoch 8:  0.0389
Epoch 9:  0.0372
Epoch 10: 0.0358  ← stopped here
```

Loss curve shows:
- **Strong convergence** in first 5 epochs (0.1448 → 0.0507)
- **Continued improvement** in epochs 6-10 (0.0507 → 0.0358)
- **No plateau** - loss still decreasing at epoch 10

## Hypothesis

Training for 15-20 epochs may improve semantic quality:
- Current: 73% semantic test accuracy
- Target: 80-85% with more training

## Evidence

**Loss reduction per epoch (epochs 6-10):**
- Epoch 6→7: 0.0447 → 0.0413 (-7.6%)
- Epoch 7→8: 0.0413 → 0.0389 (-5.8%)
- Epoch 8→9: 0.0389 → 0.0372 (-4.4%)
- Epoch 9→10: 0.0372 → 0.0358 (-3.8%)

Still improving ~4% per epoch, suggesting more headroom.

## Proposed Experiment

### Phase 1: Validate with 15 epochs

```bash
# Continue training from epoch 10
python scripts/train_topical_embeddings.py \
    --pairs data/training/topical_pairs_smart.jsonl \
    --vocab data/vocabularies/topical_vocab.json \
    --output models/topical_embeddings_v2 \
    --resume models/topical_embeddings/checkpoint_epoch10.pt \
    --epochs 15
```

**Expected:**
- Loss: 0.0358 → ~0.032 (target)
- Semantic quality: 73% → ~78%

### Phase 2: If still improving, try 20 epochs

Watch for:
- ✅ Loss plateau (stop training)
- ✅ Validation metrics stop improving
- ⚠️ Overfitting (unlikely with 8.66M pairs)

## Cost-Benefit Analysis

**Cost:**
- Training time: +30 minutes for 5 more epochs
- Storage: Minimal (checkpoints ~61 MB each)
- Risk: Low (can always revert to epoch 10)

**Benefit:**
- Potential 5-10% semantic quality improvement
- Better geographic/city embeddings
- More stable embeddings overall

## When to Do This

**Not urgent** - deprioritized because:
1. Current model is usable (73% quality)
2. Hybrid index build is running now
3. Should benchmark current model first
4. Can revisit after seeing real-world performance

**Trigger for re-prioritization:**
- If benchmarks show poor proper noun handling
- If geographic queries underperform
- After fixing vocabulary issues (Tasks #84, #85)

## Alternative: Early Stopping

Instead of fixed epochs, implement early stopping:
- Monitor validation loss
- Stop when no improvement for 3 epochs
- Prevents over/under-training automatically

## Related

- Note #80: Topical model validation (73% quality)
- Current model: 10 epochs, loss 0.0358
- Training data: 8.66M pairs
- Task #84: Fix vocabulary
- Task #85: Clean proper nouns
- Task #86: Add geography pairs
