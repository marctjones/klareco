# Plausibility Scorer - Next Steps (2026-03-22)

## What We Built

✅ **Plausibility Scorer v1.0**
- Architecture: 98K params, simple concatenation (subject + verb + object → MLP)
- Training: 182K examples, 2.5 min training time, F1 65.59%
- Status: **Trained successfully, but NOT production-ready**

## Critical Finding: Data Quality Issue

**The model architecture works, but the training data is noisy.**

### Test Results

**✅ Works for common patterns:**
```
(hom, manĝ, pom) → 0.836       # person eats apple ✓
(hund, vid, kat) → 0.824        # dog sees cat ✓
(student, lern, lingv) → 0.759  # student learns language ✓
```

**❌ Fails basic semantic constraints:**
```
(pom, manĝ, hom) → 0.778        # apple eats person ✗ (should be <0.2)
(hom, manĝ, sonĝ) → 0.740       # person eats dream ✗ (should be <0.3)
(tabl, lern, lingv) → 0.742     # table learns language ✗ (should be <0.2)
(ĉiel, verk, libr) → 0.702      # sky writes book ✗ (should be <0.2)
```

### Root Cause

**SVO extraction from Wikipedia produced 20-50% false positives:**
- (k, okup, l) - parsing errors (single-letter roots)
- (jar, verk, libr) - "year writes book" (temporal phrase misparse)
- (respublik, agnosk, februar) - "republic recognizes February" (date misparse)

**The model learned**: "If roots co-occur in corpus → plausible" (not true semantic constraints)

## Decision Point: Three Options

### Option A: Fix Data Quality & Retrain (1-2 days)

**Strategy**: Clean SVO extraction → higher quality positives → retrain

**Steps**:
1. Add extraction filters:
   - Only simple sentences (≤10 words, single clause, active voice)
   - Reject roots ≤2 chars
   - Reject temporal/spatial phrases as subjects/objects
2. Manual validation:
   - Sample 1000 random "positives"
   - Estimate true positive rate
   - Build filter classifier
3. Retrain with clean data

**Pros**: Same architecture, just better data
**Cons**: 1-2 days work, may still have noise

**Expected result**: 80-90% accuracy on constraint tests

---

### Option B: Hybrid Deterministic + Learned (2-3 days)

**Strategy**: Hand-coded rules for common verbs + learned model for rare verbs

**Steps**:
1. Build animacy lexicon (100 roots: humans, animals, objects)
2. Define selectional restrictions for top 50 verbs:
   - `manĝ`: animate subject + physical object
   - `verk`: sentient subject + artifact object
   - `lern`: sentient subject + abstract object
3. Learned model for rare verbs (use current model as baseline)

**Pros**:
- 100% accurate on common verbs (covered by rules)
- Generalizes to rare verbs (via embeddings)
- Explainable (rules) + flexible (learned)

**Cons**: More engineering, manual lexicon work

**Expected result**: 95%+ accuracy on constraint tests

---

### Option C: Use Current Model for Low-Stakes Filtering

**Strategy**: Deploy v1.0 for soft filtering only (not hard constraints)

**Use cases where v1.0 is acceptable:**
- **Synthetic data ranking**: Sort generated examples by plausibility (top 50% likely good)
- **Parse candidate reranking**: Choose best parse among alternatives
- **Corpus quality scoring**: Flag suspicious examples for review

**Use cases where v1.0 is NOT acceptable:**
- ❌ Fact verification: False negatives (rejects valid facts)
- ❌ Hard filtering: False positives (accepts nonsense)

**Pros**: Use it now, no additional work
**Cons**: Limited applicability, can't trust for critical decisions

---

## Recommendation

**Short-term (this week)**: Option C - Use for soft filtering
- Parse candidate reranking in AST pipeline
- Synthetic data ranking
- Don't use for fact verification

**Medium-term (next 2 weeks)**: Option B - Hybrid approach
- Better accuracy (95%+ vs 65%)
- Explainable for common cases
- Still generalizes via embeddings

**Long-term**: Integrate into RAG pipeline for selectional preference filtering

## Technical Artifacts

**Created**:
- `models/plausibility_scorer/model_best.pt` (1.6M, F1 65.59%)
- `scripts/test_plausibility_scorer.py` (evaluation tool)
- `scripts/train_plausibility_scorer.py` (training script)
- `scripts/generate_plausibility_training_data_quality.py` (dataset generator)
- `docs/PLAUSIBILITY_SCORER_V1_ANALYSIS.md` (detailed analysis)

**Fixed bugs**:
- Hybrid embedder dimension mismatch (AST 64D → padded to 128D)
- SVO extraction null pointer errors (3 locations)
- Unknown root handling (zero embedding fallback)

## Questions for Discussion

1. **Acceptable accuracy?** Is 65% F1 good enough for soft filtering, or must we fix data?
2. **Time investment?** 2-3 days for Option B vs deploy current model now?
3. **Use cases?** What exactly do you want to use plausibility scoring for?

See `docs/PLAUSIBILITY_SCORER_V1_ANALYSIS.md` for full technical analysis.
