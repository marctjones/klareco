# Training Results: 2026-01-18

## Summary

Completed retraining of Stage 1 (root embeddings) and M1 (selectional preferences) with tier0 corpus and ReVo semantic relations.

**Stage 1**: ✅ **Success** - Meets all targets
**M1**: ⚠️ **Below Target** - Requires investigation and retraining

## Stage 1: Root Embeddings

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Correlation** | 0.8491 | > 0.80 | ✅ PASS |
| **Positive similarity** | 0.529 | 0.4-0.6 | ✅ PASS |
| **Negative similarity** | 0.030 | < 0.1 | ✅ PASS |
| **Separation gap** | 0.499 | > 0.4 | ✅ PASS |
| **Mean pairwise similarity** | 0.0284 | < 0.5 | ✅ PASS |
| **Training time** | 68 minutes | - | - |
| **Epochs** | 32 (early stopped at 17) | - | - |

### Assessment

**Excellent performance!** Stage 1 embeddings achieve correlation comparable to BERT (0.80-0.85 range) with only 692K parameters (vs BERT's 110M+). The embeddings show:

- ✅ No collapse (mean pairwise similarity 0.0284 is very healthy)
- ✅ Good separation (0.499 gap between positive and negative pairs)
- ✅ Well-calibrated similarity predictions
- ✅ Function words included but not causing collapse

**Ready for production use.**

## M1: Selectional Preferences

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Test accuracy (full set)** | 70.20% | > 82% | ❌ FAIL |
| **Test accuracy (hard set)** | 58.90% | > 82% | ❌ FAIL |
| **Validation accuracy** | 70.25% | > 82% | ❌ FAIL |
| **Subject-verb loss** | 0.6255 | < 0.3 | ❌ FAIL |
| **Verb-object loss** | 0.6244 | < 0.3 | ❌ FAIL |
| **Triple loss** | 0.5527 | < 0.3 | ❌ FAIL |
| **Score mean** | 0.494-0.511 | 0.4-0.6 | ✅ PASS |
| **Score std** | 0.254-0.274 | > 0.05 | ✅ PASS |
| **Training time** | 18 minutes | - | - |
| **Epochs** | 21 (early stopped at 11) | - | - |

### Problem Analysis

**M1 is severely biased toward predicting "implausible" (score ~0.0) for most inputs.**

Detailed findings from extensive validation (`scripts/validate_m1_extensive.py`):

1. **Class imbalance in predictions** (threshold 0.6):
   - Plausible recall: **25.4%** (model fails to recognize 75% of plausible triples)
   - Implausible recall: **86.8%** (model correctly rejects most implausible)
   - The model has learned "when in doubt, say implausible"

2. **Score distributions heavily skewed**:
   - **66.3%** of plausible examples scored 0.0-0.1 (should be high!)
   - **78.0%** of implausible examples scored 0.0-0.1 (correct)
   - Bimodal distribution for plausible: peaks at 0.0 and 0.9 (suggests model CAN learn some patterns but fails on most)

3. **Very low mean separation**:
   - Plausible mean: 0.266
   - Implausible mean: 0.149
   - Gap: **0.117** (far too small - they should be well separated)

4. **All component losses high**:
   - SV loss: 0.6255 (target: < 0.3)
   - VO loss: 0.6244 (target: < 0.3)
   - Triple loss: 0.5527 (target: < 0.3)
   - Model is struggling to learn meaningful patterns

### What's NOT the Problem

✅ **Training data balance**: Perfectly balanced 50/50 positive/negative
✅ **Stage 1 embeddings**: No collapse (mean similarity 0.0284), good quality
✅ **Score collapse**: Std 0.26 shows model IS making distinctions
✅ **Training errors**: No crashes, checkpoints saved correctly

### Root Cause Hypotheses

1. **Insufficient model capacity** (MOST LIKELY)
   - Hidden dimension 128d may be too small
   - Only ~50K parameters total
   - Complex three-way interaction (S-V-O) requires more expressiveness

2. **Early plateau / local minimum**
   - Training stopped at epoch 11 (patience=10)
   - Losses plateaued early, didn't have chance to improve
   - May need longer patience or learning rate schedule

3. **Loss function weighting**
   - Three-component loss (SV, VO, SVO) weighted equally
   - Maybe triple loss should dominate or use different weighting

4. **Overfitting despite regularization**
   - Dropout 0.1 may be insufficient
   - Model memorizing training patterns but not generalizing

## Next Steps

### Immediate (M1 Retraining)

**Quick start** - Run the improved training script:
```bash
./scripts/retrain_m1_improved.sh --fresh
```

This script trains with:
- Hidden dimension: 256d (was 128d) - double capacity
- Dropout: 0.2 (was 0.1) - better regularization
- Patience: 20 (was 10) - more time before early stopping

**Alternative approaches to try if still below target:**

1. **Further increase capacity**:
   ```bash
   python scripts/train_m1_selectional.py \
     --hidden-dim 512 \
     --dropout 0.3 \
     --patience 20 \
     --fresh
   ```

2. **Reduce learning rate for stability**:
   ```bash
   python scripts/train_m1_selectional.py \
     --hidden-dim 256 \
     --learning-rate 0.0005 \
     --patience 20 \
     --fresh
   ```

3. **Try larger dataset**:
   ```bash
   python scripts/prepare_m1_training_data.py --max-triples 600000
   ./scripts/retrain_m1_improved.sh --fresh
   ```

### Analysis Tasks

1. **Run comprehensive validation** after each retrain:
   ```bash
   python scripts/validate_m1_extensive.py --full
   ```

2. **Analyze error patterns**: Which specific triples are being misclassified?
   - Create script to dump worst-performing examples
   - Check if errors are systematic (e.g., all abstract concepts)

3. **Compare to baseline**: Train simpler model (logistic regression on concatenated embeddings)
   - Establishes whether problem is architecture vs data

### Long-term Improvements

1. **Curriculum learning**: Train on easy negatives first, gradually add hard ones

2. **Data augmentation**: Generate more positive examples from tier0

3. **Alternative architectures**:
   - Separate binary classifiers for SV, VO, then ensemble
   - Attention mechanism to focus on relevant interactions
   - Larger hidden layers (512d, 1024d)

4. **Better loss function**:
   - Focal loss to address class imbalance in predictions
   - Contrastive loss to increase separation
   - Triplet loss (anchor-positive-negative)

## Wiki Documentation Status

All wiki templates have been populated with actual results:

- ✅ `Stage-1-Root-Embeddings.md` - Complete with metrics (correlation 0.8491)
- ⚠️ `M1-Selectional-Preferences.md` - Complete with warning about low accuracy
- ✅ `Model-Overview.md` - Updated status for both models
- ✅ `Understanding-Model-Metrics.md` - Educational guide ready
- ✅ `README.md` - Usage instructions ready

**Ready to copy to wiki once M1 is retrained successfully.**

## References

- Training logs: `logs/training/retrain_with_tier0_*.log`
- Stage 1 model: `models/root_embeddings_tier0/best_model.pt`
- M1 model: `models/m1_selectional_tier0/best_model.pt`
- Validation script: `scripts/validate_m1_extensive.py`
- Training data: `data/training/m1_with_tier0/` (400K examples)
- Hard test set: `data/training/m1_selectional_hard_only/` (1,723 examples)

## Tasks Created

- Task #9: "Investigate M1 low accuracy (70% vs 82% target) - model predicting 0.0 for most inputs"
- Note #9: Detailed analysis of validation results

---

**Recommendation**: Retrain M1 with `--hidden-dim 256` as first attempt. The model capacity appears insufficient for the complexity of three-way (subject-verb-object) interaction learning.
