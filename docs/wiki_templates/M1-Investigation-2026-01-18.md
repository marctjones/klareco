# M1 Investigation: Why Accuracy Remains at 70%

**Date**: 2026-01-18
**Issue**: M1 test accuracy stuck at 70% despite doubling model capacity (128d → 256d)

## Summary

Two critical bugs were discovered that explain why M1 cannot learn selectional preferences:

1. **Tier0 corpus NOT included** in M1 training data (bug in data generation script)
2. **Corrupted negatives indistinguishable** from positives in embedding space

## Training Results Comparison

| Config | Hidden Dim | Test Accuracy | Val Accuracy | Component Losses |
|--------|------------|---------------|--------------|------------------|
| Initial | 128d | 70.20% | 70.25% | SV: 0.6255, VO: 0.6244, Triple: 0.5527 |
| Doubled | 256d | 70.25% | 70.18% | SV: 0.6285, VO: 0.6254, Triple: 0.5535 |

**Result**: Virtually identical performance. Doubling capacity had **zero effect**.

## Bug #1: Tier0 Corpus Not Included

### Discovery

Analysis of M1 training data (`data/training/m1_with_tier0/train.jsonl`) revealed:

```
Tier distribution (320,000 examples):
  Tier 2 (Krestomatio):   6,316 (2.0%)
  Tier 5 (General):     160,671 (50.2%)
  Tier 6 (General):     153,013 (47.8%)
  Tier 0 (High-quality):      0 (0.0%)  ❌
```

**Expected**: Tier 0 examples with high weight (10.0-15.0) from authoritative sources
**Actual**: Zero tier 0 examples

### Root Cause

File: `scripts/prepare_m1_training_data.py:339`

```python
default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
```

The script defaults to the **old corpus without tier0**, not `corpus_with_tier0.jsonl`.

### Impact

Both training runs (initial and "improved") used **the same old data**:
- No tier0 high-quality examples
- No improvement in data quality
- Explains why doubling capacity had no effect

### Fix

Change default corpus path:
```bash
python scripts/prepare_m1_training_data.py \
  --corpus data/enhanced_corpus/corpus_with_tier0.jsonl \
  --output-dir data/training/m1_with_tier0 \
  --fresh
```

## Bug #2: Corruption Doesn't Create Distinguishable Negatives

### Discovery

Analysis of embedding space distances for positive vs corrupted triples:

**Positive (plausible) triples:**
- Mean pairwise embedding similarity: **0.2356**
- Standard deviation: 0.2177

**Negative (corrupted) triples:**
- Subject corruption: **0.1943** (slightly lower)
- **Verb corruption: 0.3079** (HIGHER!)
- Object corruption: **0.2369** (same as positives!)

**Expected**: Corrupted triples should have LOWER similarity
**Actual**: Corrupted triples have same or HIGHER similarity

### Why This Happens

**Random word substitution doesn't guarantee semantic distinction.**

Example:
```
Original (plausible): (hund, manĝ, viand) - "dog eats meat"
  - S-V similarity: low (dog ≠ eat)
  - V-O similarity: high (eat ≈ meat)
  - S-O similarity: medium (dog ≈ meat, both food-related)

Corrupted object: (hund, manĝ, libro) - "dog eats book"
  - S-V similarity: low (same as original)
  - V-O similarity: medium (eat ≈ read?)
  - S-O similarity: low (dog ≠ book)

  Result: Average similarity might be SAME or HIGHER!
```

### Fundamental Problem

**Plausibility ≠ Embedding Similarity**

A plausible triple can have completely dissimilar words:
- (animate_being, action_verb, inanimate_object)
- "dog eats meat" - all three words are from different semantic categories
- Low embedding similarity, but HIGH plausibility

Random corruption doesn't change the embedding similarity pattern enough to be detectable.

### Impact

The model has **no learnable signal**:
- Input: Root embeddings (S, V, O)
- Label: Plausible (1.0) vs Implausible (0.0)
- Problem: Both classes have same embedding similarity distribution

This is why losses plateau at 0.55-0.62 (random guessing territory).

## Why Model Predicts 0.0 for Everything

Validation analysis showed:

- **66.3% of plausible** examples scored 0.0-0.1
- **78.0% of implausible** examples scored 0.0-0.1
- Plausible recall: **25.4%** (fails to recognize 75% of plausible triples)
- Implausible recall: **86.8%** (correctly rejects most)

**The model learned a simple heuristic**: "When in doubt, predict implausible (0.0)."

This achieves 70% accuracy on balanced data (50% positive, 50% negative) because:
- Always predicting 0.0 gets 50% correct (all negatives)
- The model learned to recognize ~20% of positives (adds another 10%)
- Total: 50% + 10% = 60-70% accuracy

But this is **not learning selectional preferences** - it's learning a bias.

## Solutions

### Immediate: Fix Tier0 Integration

1. **Regenerate M1 training data with tier0 corpus**:
   ```bash
   python scripts/prepare_m1_training_data.py \
     --corpus data/enhanced_corpus/corpus_with_tier0.jsonl \
     --output-dir data/training/m1_with_tier0 \
     --max-triples 400000 \
     --fresh
   ```

2. **Retrain M1 with new data**:
   ```bash
   ./scripts/retrain_m1_improved.sh --fresh
   ```

**Expected improvement**: Minimal, because Bug #2 still exists.

### Medium-term: Improve Corruption Strategy

Current corruption: **Random substitution**

Better approaches:

1. **Semantic distance-based corruption**:
   - Replace with word from **different semantic category** (use ConceptNet/ReVo)
   - Ensure corrupted component is semantically distant (similarity < 0.1)
   - Example: Replace "meat" (food) with "idea" (abstract concept)

2. **Category-aware corruption**:
   - Nouns: Replace animate with inanimate, concrete with abstract
   - Verbs: Replace action with state, physical with mental
   - Use ReVo semantic categories for guidance

3. **Hard negative mining**:
   - Find triples that scored HIGH but should be implausible
   - Use these as training examples to correct model bias

### Long-term: Rethink Training Approach

The fundamental issue: **We're training a binary classifier on inputs that don't differ.**

Alternative architectures:

1. **Contrastive learning**:
   - Don't predict plausible/implausible directly
   - Learn to rank: "Is (S1, V, O) more plausible than (S2, V, O)?"
   - Use triplet loss: anchor, positive, negative

2. **Separate selectional restrictions**:
   - Train classifier: "Can X be subject of Y?" (binary, simpler)
   - Train classifier: "Can Y take Z as object?" (binary, simpler)
   - Combine at inference time

3. **Use ConceptNet categories directly**:
   - Don't learn from examples, use explicit semantic constraints
   - "Eat" requires [animate] subject and [physical] object
   - Check if roots satisfy constraints (deterministic or learned)

4. **Abandon M1 entirely for now**:
   - Focus on Stage 1 embeddings quality
   - Use retrieval without plausibility filtering
   - Come back to M1 after semantic knowledge graph is better integrated

## Recommendations

**Short-term** (this week):

1. ✅ Document bugs (this page)
2. ☐ Fix `prepare_m1_training_data.py` default corpus path
3. ☐ Regenerate M1 data with tier0 + semantic distance-based corruption
4. ☐ Retrain and validate

**Medium-term** (next month):

1. ☐ Implement semantic distance-based corruption using ReVo/ConceptNet
2. ☐ Experiment with contrastive learning approach
3. ☐ Consider whether M1 is even necessary (test retrieval without it)

**Long-term** (Q2 2026):

1. ☐ Integrate semantic knowledge graph more deeply
2. ☐ Consider deterministic selectional restrictions instead of learned model
3. ☐ Design M2 (grammatical model) with lessons learned from M1

## Files to Update

- `scripts/prepare_m1_training_data.py` - Change default corpus path
- `docs/wiki_templates/M1-Selectional-Preferences.md` - Document bugs and status
- `docs/wiki_templates/Training-Results-2026-01-18.md` - Update with investigation findings
- Create: `scripts/prepare_m1_training_data_improved.py` - Semantic distance corruption

## References

- Task #9: Investigate M1 low accuracy
- Task #10: Fix M1 training data generation to use tier0 corpus
- Note #9: M1 validation results analysis
- Note #10: M1 critical bugs discovery
- Training log: `logs/training/retrain_m1_improved_20260118_170545.log`
- Validation output: See user message above

---

**Conclusion**: M1's 70% accuracy is not due to insufficient model capacity, but due to:
1. Wrong training data (missing tier0)
2. Flawed negative sampling (corruption doesn't create distinguishable examples)

Both must be fixed before M1 can learn selectional preferences effectively.
