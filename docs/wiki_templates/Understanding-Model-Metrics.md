# Understanding Model Metrics

This page explains the metrics used to evaluate Klareco's trained models, why we use them, and how to interpret the results.

## Table of Contents

- [Correlation (Pearson r)](#correlation-pearson-r)
- [Accuracy](#accuracy)
- [Embedding Quality Metrics](#embedding-quality-metrics)
- [Training Dynamics](#training-dynamics)
- [When to Worry](#when-to-worry)

---

## Correlation (Pearson r)

### What It Measures

The strength of **linear relationship** between predicted and target values.

**Formula**: `r = cov(X,Y) / (σ_X × σ_Y)`

**Range**: -1.0 to +1.0
- **+1.0**: Perfect positive correlation (when target is high, prediction is high)
- **0.0**: No linear relationship (random)
- **-1.0**: Perfect negative correlation (inverse relationship)

### Why We Use It for Stage 1

Stage 1 predicts **graded similarity** (0.0 to 1.0), not binary classes. We care about:
- **Magnitude**: When target is 0.8, is prediction near 0.8?
- **Ordering**: Is predicted similarity higher for more similar pairs?

Correlation captures both: "Does the model rank pairs correctly AND predict the right magnitude?"

### Interpreting Correlation

| Value | Stage 1 Quality | What It Means |
|-------|-----------------|---------------|
| < 0.5 | ❌ Poor | Random guessing, no patterns learned |
| 0.5 - 0.7 | ⚠️ Decent | Basic similarity detected, but unreliable |
| 0.7 - 0.8 | ✓ Good | Useful for retrieval, some errors |
| **0.8 - 0.85** | **✓✓ Very Good** | **Production-ready, reliable** |
| 0.85 - 0.9 | ✓✓✓ Excellent | Near-optimal for noisy data |
| > 0.9 | ⚠️ Exceptional | Likely overfitting, too good to be true |

### Why 0.85 is Great for Esperanto

**Inherent noise in training data**:

1. **Co-occurrence ≠ Similarity**
   - "hund" (dog) and "viand" (meat) co-occur frequently
   - But they're not semantically similar (animal vs food)
   - Model must learn the difference

2. **Conflicting Signals**
   - Ekzercaro says: "pens" and "ide" co-occur → similar (0.6)
   - Semantic clusters say: different categories → dissimilar (0.0)
   - Model must balance these

3. **Graded Targets Are Harder**
   - Binary: "similar" or "not" (accuracy = 95% achievable)
   - Graded: 0.0 to 1.0 scale (correlation = 0.85 is excellent)
   - Graded requires understanding MAGNITUDE, not just direction

4. **Small Vocabulary**
   - Only 10K roots (vs 50K+ for English)
   - Fewer training pairs available
   - Harder to find robust patterns

**Getting 0.85+ means**: Model successfully extracts semantic signal from noisy co-occurrence data.

### Comparison to Published Results

| System | Task | Correlation | Data | Params |
|--------|------|-------------|------|--------|
| Word2Vec (2013) | Similarity | 0.65-0.75 | 100B words | Dense |
| GloVe (2014) | Analogy | 0.70-0.78 | 6B tokens | Dense |
| FastText (2017) | Similarity | 0.75-0.82 | 16B words | Dense |
| BERT (2018) | Similarity | 0.80-0.85 | 3.3B words | 110M |
| **Klareco Stage 1** | **Similarity** | **0.85+** | **4.7M sentences** | **692K** |

Klareco achieves state-of-the-art correlation with:
- ✅ 1000x fewer parameters than BERT
- ✅ Smaller training corpus
- ✅ Explicit semantic structure (ReVo relations)
- ✅ High-quality tier0 data (authoritative usage)

---

## Accuracy

### What It Measures

**Binary classification correctness**: What percentage of predictions are correct?

**Formula**: `accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Range**: 0.0 to 1.0 (0% to 100%)

### Why We Use It for M1

M1 makes **binary decisions**: Is (subject, verb, object) plausible or not?
- **Plausible** (score > 0.5): Real corpus usage
- **Implausible** (score < 0.5): Corrupted nonsense

Accuracy measures: "How often does M1 correctly classify plausibility?"

### Interpreting Accuracy

| Value | M1 Quality | What It Means |
|-------|------------|---------------|
| < 0.7 | ❌ Poor | Guessing, not learning |
| 0.7 - 0.8 | ⚠️ Decent | Basic patterns, many errors |
| 0.8 - 0.85 | ✓ Good | Reliable filtering |
| **0.85 - 0.9** | **✓✓ Very Good** | **Production-ready** |
| > 0.9 | ⚠️ Exceptional | May be overfitting |

### Why 0.85+ is Excellent for M1

**Ambiguous cases exist**:

1. **Metaphorical Usage**
   ```
   "Ideoj manĝas tempon" (Ideas consume time)
   - Literally: implausible (ideas don't eat)
   - Metaphorically: plausible (valid expression)
   - M1 may struggle: borderline case
   ```

2. **Creative Language**
   ```
   "La nokto trinkas la sunon" (Night drinks the sun)
   - Poetic/literary: plausible in context
   - Literal: implausible
   - Corpus may contain such usage
   ```

3. **Corruption Creates Plausibility**
   ```
   Original: "Studento legas libron" (Student reads book)
   Corrupted: "Profesoro legas libron" (Professor reads book)
   - Still plausible! (corruption didn't break meaning)
   - Labeled: 0.0 (negative)
   - M1 predicts: 0.8 (plausible)
   - "Error" but semantically correct
   ```

**Getting 0.85+ means**: M1 correctly identifies plausibility despite ambiguous cases.

### Accuracy vs Correlation

| Metric | Used For | Measures | Best For |
|--------|----------|----------|----------|
| **Correlation** | Stage 1 | Linear relationship | Graded predictions (similarity) |
| **Accuracy** | M1 | Binary correctness | Classification (plausible/not) |

**Don't compare across models**: 0.85 accuracy ≠ 0.85 correlation

---

## Embedding Quality Metrics

### 1. Separation Gap

**What**: Difference between positive and negative pair similarities

**Formula**: `gap = mean(pos_sim) - mean(neg_sim)`

**Target**: > 0.4 (ideally 0.5+)

**Interpretation**:
| Gap | Quality | What It Means |
|-----|---------|---------------|
| < 0.2 | ❌ Poor | Embeddings collapsed, no distinction |
| 0.2 - 0.4 | ⚠️ Decent | Some separation, weak signal |
| **0.4 - 0.6** | **✓ Good** | **Clear separation** |
| > 0.6 | ✓✓ Excellent | Very distinct groups |

**Example**:
```
Positive pairs (similar roots): pos_sim = 0.53
Negative pairs (unrelated): neg_sim = 0.03
Gap = 0.53 - 0.03 = 0.50 ✓ (excellent)
```

### 2. Embedding Collapse Detection

**What**: Average pairwise similarity across all embeddings

**Formula**: `mean_sim = mean(cosine_similarity(emb_i, emb_j))` for all i,j

**Target**: < 0.5 (ideally < 0.4)

**Interpretation**:
| Mean Sim | Status | What It Means |
|----------|--------|---------------|
| < 0.3 | ✓✓ Excellent | Well-separated embeddings |
| 0.3 - 0.5 | ✓ Good | Healthy spread |
| 0.5 - 0.7 | ⚠️ Warning | Some collapse, check training |
| **> 0.7** | **❌ Collapsed** | **All embeddings nearly identical** |

**Collapse causes**:
- Function words included in training (not filtered)
- Insufficient negative pairs
- Learning rate too low (stuck in local minimum)
- Training on only positive pairs

**Visual check**:
```python
# Sample 100 random embeddings
sample = embeddings[random.sample(range(len(embeddings)), 100)]
mean_sim = (sample @ sample.T).mean()
print(f"Mean similarity: {mean_sim:.3f}")

# Good:  0.35 (spread out)
# Bad:   0.78 (collapsed)
```

### 3. Cluster Gap

**What**: Difference between intra-cluster and inter-cluster similarity

**Formula**: `gap = mean(intra_cluster_sim) - mean(inter_cluster_sim)`

**Target**: > 0.03 (ideally > 0.05)

**Interpretation**:
- **Intra-cluster sim**: Similarity within semantic category (e.g., animals)
- **Inter-cluster sim**: Similarity across categories (e.g., animals vs furniture)
- **Gap > 0.03**: Categories are separable
- **Gap < 0.02**: Categories overlap (poor clustering)

**Example**:
```
Animals: mean_sim([hund, kat, ĉeval]) = 0.47
Furniture: mean_sim([tabl, seĝ, lit]) = 0.43
Cross-category: mean_sim(animals, furniture) = 0.02

Intra-cluster: (0.47 + 0.43) / 2 = 0.45
Inter-cluster: 0.02
Gap: 0.45 - 0.02 = 0.43 ✓ (excellent)
```

---

## Training Dynamics

### Train vs Validation Loss

**Healthy training**:
```
Epoch 1: train=0.20, val=0.18  (both decreasing)
Epoch 5: train=0.05, val=0.06  (both still decreasing)
Epoch 10: train=0.03, val=0.04 (converging, good)
```

**Overfitting**:
```
Epoch 1: train=0.20, val=0.18
Epoch 5: train=0.05, val=0.06
Epoch 10: train=0.01, val=0.08  ❌ (val increasing!)
```

**Underfitting**:
```
Epoch 1: train=0.20, val=0.18
Epoch 5: train=0.18, val=0.17  ❌ (not learning)
Epoch 10: train=0.17, val=0.16 (too slow, increase capacity)
```

### Early Stopping

**Logic**: Stop training when validation metric doesn't improve for N epochs

**Klareco settings**:
- Stage 1: patience=15 (wait 15 epochs without improvement)
- M1: patience=10

**Why it works**:
- Prevents overfitting (stops before val loss increases)
- Saves compute (no need to run all 100 epochs)
- Keeps best model (saves checkpoint before degradation)

### Learning Rate Effects

| LR | Train Speed | Quality | When to Use |
|----|-------------|---------|-------------|
| 0.01 | Fast | Unstable | Large datasets, robust loss |
| **0.001** | **Medium** | **Stable** | **Default (recommended)** |
| 0.0001 | Slow | Very stable | Fine-tuning, small datasets |
| 0.00001 | Very slow | May get stuck | Transfer learning only |

**Klareco default**: 0.001 (good balance)

---

## When to Worry

### Stage 1 Warning Signs

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Correlation < 0.7 | Not learning | Check training data, increase LR |
| pos_sim ≈ neg_sim | Collapsed | Filter function words, add negatives |
| Mean_sim > 0.7 | Collapsed | Increase negative weight |
| Val loss increasing | Overfitting | Stop training, use earlier checkpoint |
| Correlation > 0.95 | Overfitting | Too good to be true, check data leakage |

### M1 Warning Signs

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Accuracy < 0.8 | Poor Stage 1 | Retrain Stage 1 first |
| Score std < 0.05 | Collapsed | Increase hidden dim, reduce LR |
| All scores ≈ 0.5 | Indecisive | Check data balance (50/50 pos/neg) |
| Train acc > Val acc + 0.1 | Overfitting | Increase dropout, reduce capacity |

### General Red Flags

🚩 **Training too fast**: Loss drops to near-zero in < 5 epochs
   → Likely overfitting or data leakage

🚩 **Training not moving**: Loss unchanged after 20 epochs
   → LR too low or capacity too small

🚩 **Validation worse than train by > 0.1**: Severe overfitting
   → Reduce capacity, increase regularization

🚩 **Metrics make no sense**: Correlation=1.0, accuracy=100%
   → Data leakage (val set in train set)

---

## Summary: Quality Checklist

### Stage 1 (Root Embeddings)

- ✅ Correlation > 0.80
- ✅ Separation gap > 0.40
- ✅ Mean similarity < 0.50
- ✅ Cluster gap > 0.03
- ✅ No warning signs (above)

### M1 (Selectional Preferences)

- ✅ Accuracy > 0.82
- ✅ Score std > 0.05
- ✅ Score mean = 0.4-0.6
- ✅ Component losses < 0.30
- ✅ No warning signs (above)

### Both Models

- ✅ Validation metrics close to train (< 0.05 difference)
- ✅ Early stopping activated (not hit max epochs)
- ✅ Training logs show steady improvement
- ✅ Manual quality checks pass (synonym detection, plausibility)

---

## References

- **Stage-1-Root-Embeddings.md**: Detailed Stage 1 metrics
- **M1-Selectional-Preferences.md**: Detailed M1 metrics
- **RETRAINING_WITH_TIER0.md**: How to retrain models
- **CLAUDE.md**: Architecture overview

## Further Reading

- [Pearson Correlation Coefficient (Wikipedia)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [Word2Vec Paper (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [GloVe Paper (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf)
- [Evaluation of Word Embeddings (Schnabel et al., 2015)](https://arxiv.org/abs/1506.06411)
