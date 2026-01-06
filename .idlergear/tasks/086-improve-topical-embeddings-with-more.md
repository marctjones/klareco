---
id: 86
title: Improve topical embeddings with more city/geography training pairs
state: open
created: '2026-01-06T06:24:33.565837Z'
labels:
- enhancement
- embeddings
- training
priority: medium
---
## Issue

City and geography pairs show lower similarity than expected in topical embeddings:
- **pariz↔london:** 0.307 (expected >0.4)
- **histori↔geografi:** 0.336 (borderline)

While other domains perform well:
- Animals: 0.48-0.63 (strong)
- Science: 0.48 (strong)
- Nature: 0.63 (strong)

This suggests insufficient training data for geographic concepts.

## Root Cause

Training data (`topical_pairs_smart.jsonl`) may have:
1. **Fewer city pairs** - Wikipedia may have separate articles for cities without strong co-occurrence
2. **Sparse geography relationships** - Countries/cities mentioned in isolation
3. **Unbalanced domain coverage** - More animal/nature articles than geography articles

## Evidence

From validation testing:
```
Domain          Example              Similarity  Status
Animals         hund↔kat             0.600       ✓ Strong
Nature          arb↔flor             0.631       ✓ Strong  
Science         fizik↔kemi           0.476       ✓ Strong
Geography       pariz↔london         0.307       ⚠ Weak
Geography       histori↔geografi     0.336       ⚠ Borderline
```

## Proposed Solution

### 1. Audit Current Training Data

```bash
# Count geography-related pairs
grep -E "pariz|london|berlin|rom|paris" data/training/topical_pairs_smart.jsonl | wc -l

# Count animal-related pairs (for comparison)
grep -E "hund|kat|leon|tigr" data/training/topical_pairs_smart.jsonl | wc -l

# Check domain balance
```

### 2. Add Targeted Geography Pairs

Create supplementary training data with city/country clusters:

```python
# Geographic clusters to add
cities = ['pariz', 'london', 'berlin', 'rom', 'moskv', 'tokyo', 'peki']
countries = ['franc', 'german', 'ital', 'rus', 'japan', 'ĉin']
continents = ['eŭrop', 'azi', 'afrik', 'amerik']

# Generate within-cluster positive pairs
# All cities should be similar to each other
# All countries should be similar to each other
```

### 3. Mine Wikipedia for Geographic Co-occurrence

Check if Wikipedia corpus has geography articles:
```bash
# Check for geography content in corpus
grep -i "geography\|city\|country\|capital" data/extracted/eo_wikipedia/*.jsonl
```

### 4. Re-train with Augmented Data

```bash
# Combine existing + geography-focused pairs
cat data/training/topical_pairs_smart.jsonl \
    data/training/geography_pairs.jsonl \
    > data/training/topical_pairs_v2.jsonl

# Re-train topical embeddings
./scripts/train_topical_embeddings.sh --pairs data/training/topical_pairs_v2.jsonl
```

## Expected Improvement

**Before:**
- pariz↔london: 0.307
- histori↔geografi: 0.336

**After (target):**
- pariz↔london: >0.45
- histori↔geografi: >0.40
- Overall semantic quality: 73% → 85%+

## Alternative: Domain-Specific Embeddings

If geography consistently underperforms, consider:
- Separate geography embeddings (32d)
- Combined: linguistic (64d) + topical (64d) + geography (32d) = 160d
- Use domain detection to select appropriate embeddings

## Priority

**Medium** - Not blocking current work:
- Current 73% quality is usable
- Hybrid mode compensates
- Can improve in v2

## Related

- Note #80: Topical model validation results
- Task #84: Fix vocabulary extraction
- Task #85: Clean proper noun classification
- Current training data: 8.66M pairs
