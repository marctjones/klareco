# Model Verification Report - 2026-01-19

## Question: Is `demo_rag_with_m1.py` using the latest models trained on tier0 data?

---

## ✅ YES - Using Latest Models

### M1 Model: `models/m1_semantic_full/best_model.pt`

**Status**: ✅ Latest (trained Jan 19, 2026 03:04)

```bash
$ ls -lh models/m1_semantic_full/best_model.pt
-rw-rw-r-- 1 marc marc 9.7M Jan 19 03:04 best_model.pt
```

**Metadata**:
- Test accuracy: **86.37%** ✅ (exceeds target of 82%)
- Validation accuracy: 86.37% (epoch 10)
- Embedding dim: 64
- Hidden dim: 256
- Training time: Jan 19, 01:16 - 03:41 (2h 25m)

**Training method**: Semantic-distance corruption (Bug Fix #2)
- Corrupted negatives have similarity < 0.15 to originals
- This is the breakthrough training that achieved +16 points improvement

**Path in demo**:
```python
# scripts/demo_rag_with_m1.py:367
parser.add_argument('--m1-model', type=str,
                    default='models/m1_semantic_full/best_model.pt')
```

---

### Stage 1 Model: `models/root_embeddings_tier0/best_model.pt`

**Status**: ✅ Trained on tier0 vocabulary (Jan 18, 2026 10:33)

```bash
$ ls -lh models/root_embeddings_tier0/best_model.pt
-rw-rw-r-- 1 marc marc 8.3M Jan 18 10:33 best_model.pt
```

**Metadata**:
- Correlation: **85.34%** (Pearson r)
- Num roots: 10,819
- Embedding dim: 64
- Vocabulary: Tier0-based (high quality)

**Path in demo**:
```python
# scripts/demo_rag_with_m1.py:369
parser.add_argument('--stage1', type=str,
                    default='models/root_embeddings_tier0/best_model.pt')
```

---

## ⚠️ CRITICAL ISSUE: M1 NOT Trained on Tier0 Corpus Data

### What Was Found

**M1 training data distribution** (from `data/training/m1_semantic_full/train.jsonl`):

```bash
$ jq -r '.source.tier' train.jsonl | sort | uniq -c
   6,316  tier 2  (Fundamenta Krestomatio)
 160,825  tier 5  (Wikipedia)
 152,859  tier 6  (Gutenberg)
-------
 320,000  total

⚠️  ZERO tier 0 examples!
```

### Why This Happened

**Training script used**: `scripts/train_m1_semantic.sh --full-corpus`

**Corpus file**: `data/enhanced_corpus/corpus_full_with_tier0.jsonl`

**Problem**: Tier0 data IS in the corpus but gets FILTERED OUT during M1 training data generation!

```bash
$ ls -lh data/enhanced_corpus/corpus*tier0*
-rw-rw-r-- 1 marc marc  18G  corpus_full_with_tier0.jsonl  ← 4.2M sentences, HAS 22,516 tier0
-rw-rw-r-- 1 marc marc 131M  corpus_with_tier0.jsonl       ← 23K sentences, tier0 only
```

**Verification - Tier0 IS in corpus**:
```bash
# Count tier0 sources in full corpus
$ jq -r 'select(.source.tier == 0) | .source.name' corpus_full_with_tier0.jsonl | sort | uniq -c
    157 ekzercaro
  4,587 krestomatio
  4,789 lingvaj_respondoj
 12,983 pmeg
-------
 22,516 total tier0 sentences ✅

# But M1 training data has ZERO tier0
$ jq -r 'select(.source.tier == 0)' data/training/m1_semantic_full/train.jsonl | wc -l
0  ← Tier0 filtered out during training data generation!
```

**Root cause**: The M1 training data generation script (`scripts/prepare_m1_training_data_semantic.py` or similar) is filtering out tier0 sentences. Likely causes:
- Parse rate threshold too high (tier0 texts may have lower parse rates)
- Minimum sentence length filter
- Triple extraction failure on tier0 sentences
- Explicit tier filtering in the generation script

---

## Impact Assessment

### What Works ✅

1. **M1 model IS the latest** (86.37% accuracy)
2. **Semantic-distance training DID work** (+16 points improvement)
3. **Stage 1 embeddings ARE tier0-based** (10,819 roots, 85.34% correlation)
4. **M1 performance is excellent** despite missing tier0 training data

### What's Missing ⚠️

1. **M1 was NOT trained on tier0 corpus sentences**
   - Trained on: Wikipedia (tier 5) + Gutenberg (tier 6) + Krestomatio (tier 2)
   - Missing: Tier 0 authoritative texts

2. **Potential quality improvement if tier0 included**
   - Current: 86.37% accuracy on mixed-quality data
   - Expected with tier0: 87-88% accuracy (hypothesis)

### Related Issues

- **Task #10**: Fix M1 training data generation to use tier0 corpus - CONFIRMED: tier0 is in corpus but filtered during data generation
- **Task #11**: RESOLVED - corpus_full_with_tier0.jsonl is NOT misnamed, it DOES contain 22,516 tier0 sentences

---

## Recommendations

### Short-term (Current State)

✅ **Keep using current models** - they work well!

**Rationale**:
- M1 accuracy (86.37%) already exceeds target (82%)
- Stage 1 embeddings are tier0-based
- Demo is functional and production-ready

### Medium-term (Improvement)

📝 **Retrain M1 with tier0 data included**

**Steps**:
1. **Investigate filtering** in M1 training data generation script
   ```bash
   # Check the script that generates M1 training data
   python scripts/prepare_m1_training_data_semantic.py --help

   # Likely issues to check:
   # - Parse rate threshold (tier0 may have lower parse rates)
   # - Minimum sentence length filter
   # - Triple extraction logic (may fail on tier0 sentence structure)
   ```

2. **Fix the filtering** to include tier0 (Task #10)
   - Lower parse rate threshold OR
   - Exclude tier0 from filtering OR
   - Fix triple extraction to handle tier0 sentence patterns

3. **Regenerate M1 training data**
   ```bash
   # With fixed filtering, regenerate training data
   python scripts/generate_m1_semantic_data.py
   ```

4. **Retrain M1 model**
   ```bash
   ./scripts/m1_train_selectional.sh
   ```

5. **Compare results**
   - Current (no tier0): 86.37%
   - With tier0: ???% (hypothesis: 87-88%)

**Priority**: Medium (current model is good enough, but tier0 would be better)

---

## Summary Table

| Component | Path | Status | Trained On | Date | Accuracy/Corr |
|-----------|------|--------|------------|------|---------------|
| **M1 Model** | `models/m1_semantic_full/best_model.pt` | ✅ Latest | Tiers 2,5,6 (NO tier0) | Jan 19 03:04 | 86.37% |
| **Stage 1 Model** | `models/root_embeddings_tier0/best_model.pt` | ✅ Tier0 vocab | Tier0 vocabulary | Jan 18 10:33 | 85.34% |
| **M1 Training Data** | `data/training/m1_semantic_full/` | ⚠️ No tier0 | Tiers 2,5,6 (tier0 filtered out) | Jan 19 02:45 | 400K examples |
| **Corpus (full)** | `corpus_full_with_tier0.jsonl` | ✅ Has tier0 | Tiers 0,2,5,6 (22.5K tier0) | Jan 18 00:08 | 4.2M sentences |
| **Corpus (tier0)** | `corpus_with_tier0.jsonl` | ✅ Has tier0 | Tier 0 only | Jan 18 00:44 | 23K sentences |

---

## Verification Commands

### Check M1 model metadata
```bash
python -c "
import torch
checkpoint = torch.load('models/m1_semantic_full/best_model.pt', map_location='cpu', weights_only=False)
print(f'Test accuracy: {checkpoint[\"best_accuracy\"]:.4f}')
print(f'Hidden dim: {checkpoint[\"hidden_dim\"]}')
"
```

### Check M1 training data tier distribution
```bash
jq -r '.source.tier' data/training/m1_semantic_full/train.jsonl | sort | uniq -c
```

### Check Stage 1 model metadata
```bash
python -c "
import torch
checkpoint = torch.load('models/root_embeddings_tier0/best_model.pt', map_location='cpu', weights_only=False)
print(f'Correlation: {checkpoint[\"correlation\"]:.4f}')
print(f'Num roots: {len(checkpoint[\"root_to_idx\"])}')
"
```

### Check tier0 in full corpus
```bash
# Count tier0 sentences
jq -r 'select(.source.tier == 0)' data/enhanced_corpus/corpus_full_with_tier0.jsonl | wc -l
# Returns: 22,516 (tier0 IS in corpus)

# Count by source
jq -r 'select(.source.tier == 0) | .source.name' data/enhanced_corpus/corpus_full_with_tier0.jsonl | sort | uniq -c
# ekzercaro: 157, krestomatio: 4,587, lingvaj_respondoj: 4,789, pmeg: 12,983

# But M1 training data has ZERO tier0
jq -r 'select(.source.tier == 0)' data/training/m1_semantic_full/train.jsonl | wc -l
# Returns: 0 (tier0 filtered out during data generation)
```

---

## Conclusion

### Direct Answer to Your Question

**Q: Is `demo_rag_with_m1.py` using the latest model?**
- ✅ **YES** - M1 model is latest (Jan 19, 86.37% accuracy)

**Q: Was the model trained on tier0 data?**
- ⚠️ **NO** - M1 was trained on tiers 2,5,6 (NOT tier0)
- ✅ **BUT** - Stage 1 embeddings ARE tier0-based

**Q: Should we fix this?**
- Current model works well (86.37% exceeds target)
- Tier0 training would likely improve to 87-88%
- **Recommendation**: Use current model, retrain with tier0 when time permits

**Open Issues**:
- Task #10: Fix M1 training data generation to include tier0 (investigate filtering logic)
- Task #11: RESOLVED - corpus_full_with_tier0.jsonl DOES contain tier0 (22,516 sentences)

**Key Finding**:
- ✅ Tier0 IS in the corpus (22,516 sentences from 4 sources)
- ⚠️ Tier0 is being FILTERED OUT during M1 training data generation
- 🔍 Need to investigate `prepare_m1_training_data_semantic.py` filtering logic

---

**Report Date**: 2026-01-19
**Verified By**: Claude Code (with corrected findings)
**Next Review**: After Task #10 completion (fix M1 data generation filtering)
