# Option 2: Proper Fix - Progress Report

**Date**: 2026-03-22 23:50
**Goal**: Fix dimension mismatch + improve training data quality

## ✅ Phase 1: Fix Dimension Mismatch (COMPLETE)

### Problem
```
Production Root Embeddings:  128D  (6,719 roots)
AST Root Embeddings:          64D  (2,369 roots)  ← MISMATCH!
Plausibility Scorer:         128D  (expects uniform)
```

**Impact of padding**:
- AST embeddings padded: `[64 real values | 64 zeros]`
- Half the input dimensions wasted when using AST roots
- Model learns weights only work for Production roots
- Suboptimal semantic quality

### Solution Implemented ✅

**Modified**: `scripts/train_fundamento_ast_embeddings.sh`
- Changed `--embedding-dim 64` → `--embedding-dim 128`
- Backed up old 64D model → `models/root_embeddings_fundamento_ast/backup_64d/`

**Trained**: New AST model at 128D
- Training time: 27 seconds (10 epochs, early stopping)
- Model size: 606K params (vs 150K for old 64D)
- Best loss: 1.1347
- Dimensions: `torch.Size([2369, 128])` ✅

**Verification**:
```
✅ Production Model: torch.Size([6719, 128])
✅ AST Model (NEW):   torch.Size([2369, 128])
🎉 SUCCESS: Dimensions match! Both at 128D
```

**Result**: No more padding needed! Clean 128D integration.

---

## ⏳ Phase 2: Improve Training Data Quality (IN PROGRESS)

### Problem
```
Old training data had 20-50% false positives:
- (k, okup, l) - single-letter parsing errors
- (jar, verk, libr) - "year writes book" (temporal misparse)
- (respublik, agnosk, februar) - "republic recognizes February"
```

Model learned "surface co-occurrence" not "semantic constraints":
```
❌ (pom, manĝ, hom) → 0.778   # apple eats person (should be <0.2)
❌ (tabl, lern, lingv) → 0.742  # table learns language (should be <0.2)
❌ (hom, manĝ, sonĝ) → 0.740    # person eats dream (should be <0.3)
```

### Solution Implemented ✅

**Added quality filters** to `scripts/extract_svo_triples.py`:

```python
def is_quality_triple(subject_root, verb_root, object_root, sentence):
    """
    Quality filter for SVO triples to reduce noise.

    Filters out:
    - Very short roots (≤2 chars)
    - Temporal roots (jar, dat, tag, etc.)
    - Month names as objects
    - Very long sentences (>100 words)
    - Single uppercase letters
    """
    # Filter 1: Reject very short roots (≤2 chars)
    if len(subject_root) <= 2 or len(verb_root) <= 2 or len(object_root) <= 2:
        return False

    # Filter 2: Reject temporal roots as subjects/objects
    temporal_roots = {'jar', 'dat', 'tag', 'hor', 'minut', 'sekund', 'monat'}
    if subject_root in temporal_roots or object_root in temporal_roots:
        return False

    # Filter 3: Reject month names
    months = {'januar', 'februar', 'mart', ...}
    if object_root in months:
        return False

    # Filter 4: Reject long sentences (>100 words)
    if len(sentence.split()) > 100:
        return False

    # Filter 5: Reject single uppercase letters
    if (len(subject_root) == 1 and subject_root.isupper()) or \
       (len(object_root) == 1 and object_root.isupper()):
        return False

    return True
```

**Applied to all extraction paths**:
- ✅ Simple SVO extraction
- ✅ Coordinated verbs (... kaj ...)
- ✅ Passive voice (estis ... de ...)

### Current Status

**Running**: SVO extraction with quality filters
- Input: 500K Wikipedia + Books sentences
- Filters: Active (rejecting noisy triples)
- Output: `data/semantic_types/svo_triples_quality.jsonl`
- ETA: ~15-20 minutes
- Expected: ~100-120K clean triples (vs 137K noisy triples before)

**Next**:
1. Generate plausibility training dataset from clean SVO triples
2. Retrain plausibility scorer with clean data + uniform 128D embeddings
3. Test on hand-crafted constraint examples
4. Expected improvement: F1 65% → 80-90%

---

## 📊 Expected Improvements

### Dimension Mismatch Fix

**Before**:
- AST roots: 64D real + 64D zeros
- Production roots: 128D real
- Inconsistent signal to model

**After**:
- AST roots: 128D real ✅
- Production roots: 128D real ✅
- Uniform signal, better integration

**Impact**: Better embedding quality when using AST roots (Fundamento vocabulary)

### Training Data Quality Fix

**Before**:
- 182,728 examples (82,728 positive + 100,000 negative)
- ~20-50% false positives in "positive" examples
- Model learned co-occurrence, not constraints

**After** (expected):
- ~160-180K examples (fewer but cleaner positives)
- <10% false positives (quality filters applied)
- Model should learn actual semantic constraints

**Expected Test Results**:
```
✅ (hom, manĝ, pom) → >0.8      # person eats apple
✅ (pom, manĝ, hom) → <0.2      # apple eats person (FIXED!)
✅ (tabl, lern, lingv) → <0.2    # table learns language (FIXED!)
✅ (hom, manĝ, sonĝ) → <0.3      # person eats dream (FIXED!)
```

---

## 🎯 Next Steps After Pipeline Completes

1. **Verify SVO extraction quality** (~15 min from now)
   - Check triple count: expect ~100-120K (down from 137K)
   - Sample random triples: manual quality check
   - Confirm no single-letter roots, no temporal subjects

2. **Regenerate plausibility training dataset** (~5 min)
   - Delete old dataset
   - Run generation with clean SVO triples
   - Expected: ~160-180K examples

3. **Retrain plausibility scorer** (~3 min)
   - Same architecture (98K params)
   - Clean data + uniform 128D embeddings
   - Expected: F1 65% → 80-90%

4. **Test on constraint examples** (~1 min)
   - Run `scripts/test_plausibility_scorer.py`
   - Verify reversed roles rejected
   - Verify type violations rejected

5. **Build hybrid scorer (Option B)** (2-3 hours)
   - Create animacy lexicon
   - Define selectional restrictions
   - Combine deterministic + learned
   - Target: 95%+ accuracy

---

## Files Modified

```
✅ scripts/train_fundamento_ast_embeddings.sh
   - Line 77: Updated echo message (64D → 128D)
   - Line 104: Changed --embedding-dim 64 → 128
   - Line 125: Updated final message

✅ scripts/extract_svo_triples.py
   - Added is_quality_triple() function (lines 347-393)
   - Applied filter to extract_from_frazo() (line 461)
   - Applied filter to extract_coordinated_verbs() (lines 559, 582)
   - Applied filter to extract_passive_voice() (line 692)

✅ models/root_embeddings_fundamento_ast/
   - Backed up: backup_64d/root_embeddings_best_64d.pt (old 64D)
   - New model: root_embeddings_best.pt (128D, 606K params)
   - Training log: logs/fundamento_ast_training/training_20260322_234546.log
```

---

## Validation Checklist

Before proceeding to plausibility retraining:

- [x] AST model trained at 128D
- [x] Dimensions verified (both 128D)
- [x] Quality filters added to SVO extraction
- [x] Old SVO data deleted
- [ ] New SVO extraction complete (~15 min remaining)
- [ ] Sample quality check of extracted triples
- [ ] Plausibility dataset regenerated
- [ ] Plausibility scorer retrained
- [ ] Constraint tests passed

**Status**: 2/8 complete, on track for Option 2 success! 🚀
