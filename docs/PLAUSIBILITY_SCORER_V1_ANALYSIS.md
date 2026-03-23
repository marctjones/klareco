# Plausibility Scorer v1.0 - Post-Training Analysis

**Date**: 2026-03-22
**Model**: `models/plausibility_scorer/model_best.pt`
**Training Data**: `data/plausibility_training_quality/` (182K examples)

## Training Results

**Validation Metrics** (Epoch 3, best):
- F1: 65.59%
- Accuracy: 67.43%
- Precision: 62.94%
- Recall: 68.48%

**Training Time**: 2.5 minutes (8 epochs, early stopping)

## Evaluation on Hand-Crafted Test Cases

### ✅ Cases That Work

```
(hom, manĝ, pom) → 0.836       # person eats apple
(hund, vid, kat) → 0.824        # dog sees cat
(student, lern, lingv) → 0.759  # student learns language
```

**Pattern**: Simple, high-frequency corpus patterns with animate subjects.

### ❌ Critical Failures

```
# Should be IMPLAUSIBLE but scored PLAUSIBLE:
(pom, manĝ, hom) → 0.778        # apple eats person (reversed!)
(hom, manĝ, sonĝ) → 0.740       # person eats dream (type violation!)
(tabl, lern, lingv) → 0.742     # table learns language (inanimate agent!)
(ĉiel, verk, libr) → 0.702      # sky writes book (impossible!)
```

**Pattern**: Model does NOT enforce semantic constraints (animacy, selectional restrictions, role directionality).

## Root Cause Analysis

### 1. Training Data Quality Issues

**Automatic SVO extraction from Wikipedia produced noisy "positive" examples:**

| Issue | Prevalence | Example |
|-------|------------|---------|
| Single-letter roots (parsing errors) | 20% | (k, okup, l) |
| Temporal phrases as SVO | Common | (mart, dat, ekvinoks) = "March dates equinox" |
| Complex sentence misparses | Common | (jar, verk, libr) = "year writes book" |
| Spatial phrases as SVO | Common | (ocean, apud, bord) = "ocean next-to shore" |

**Actual sample "positives" from training data:**
```
(k, okup, l)              # "K occupies l" - parsing error
(a, viol, ati)            # "a violates ati" - article as subject
(jar, verk, libr)         # "year writes book" - temporal phrase
(respublik, agnosk, februar)  # "republic recognizes February" - date extraction
```

**Estimated true positive rate**: ~50-60% (not the assumed 100%)

### 2. Why the Model Learned Wrong Patterns

1. **Garbage in, garbage out**: 20-50% of "positive" examples are semantically nonsensical
2. **Model learned surface co-occurrence**: If roots appear together in corpus, score high
3. **No directional constraints learned**: Because "positives" include reversed/scrambled patterns
4. **No animacy constraints learned**: Because "positives" include inanimate subjects doing actions

### 3. Negative Generation Strategy Issues

The negative examples were actually CLEANER than the positives:
- Type-compatible swaps: systematic, semantically wrong
- Role confusion: systematic, semantically wrong
- Random filtered: obviously wrong

**But**: Model learned "anything that looks vaguely corpus-like is plausible" because the positives were so noisy.

## Lessons Learned

### ❌ What Didn't Work

1. **Trusting parser confidence**: `confidence=1.0` means "parse completed", not "parse is correct"
2. **Automatic positive extraction**: Complex sentences → parsing errors → noisy positives
3. **No manual validation**: Assumed corpus examples are ground truth (they're not)
4. **Quality over quantity myth**: 200K examples is "quality" only if examples are actually correct

### ✅ What DID Work

1. **Simple concatenation architecture**: Model learned SOMETHING from the data (F1 65%)
2. **Frozen embeddings**: Training was fast (2.5 min), embeddings provided semantic signal
3. **Challenging negative generation**: Negatives were actually higher quality than positives
4. **Early stopping**: Prevented overfitting to noise

## Recommendations for v2.0

### Option A: Fix Data Pipeline (Recommended)

1. **Extract only from simple sentences**:
   - Single clause (no subordination)
   - Active voice only
   - No temporal/spatial phrases
   - Max 10 words

2. **Add automatic filters**:
   - Reject roots ≤2 characters
   - Reject proper name subjects (unless in lexicon)
   - Reject abstract/temporal objects for physical verbs (manĝ, vid, etc.)

3. **Manual validation**:
   - Human-validate 1000 random positives
   - Estimate true positive rate
   - Build classifier to auto-reject low-quality patterns

4. **Synthetic positives**:
   - Hand-craft 1000 core positive patterns
   - Generate variations using vocabulary
   - Guarantees quality > relying on corpus

### Option B: Supervised Lexical Patterns (Alternative)

Instead of learned model, use deterministic rules:

1. **Build animacy lexicon**: humans, animals = animate; objects = inanimate
2. **Build selectional restriction rules**:
   - `manĝ` requires animate subject + physical object
   - `verk` requires sentient subject + artifact object
3. **Score by rule matching**: plausible if passes all constraints

**Pros**: Explainable, 100% accurate on covered cases
**Cons**: Doesn't generalize to rare verbs, requires manual lexicon

### Option C: Hybrid Approach (Best)

1. **Deterministic constraints** for common verbs (top 100): manĝ, vid, verk, etc.
2. **Learned model** for rare verbs: uses embeddings to generalize
3. **Confidence scoring**: deterministic rules → confidence 1.0, learned → 0.0-1.0

## Metrics for v2.0 Success

A successful v2.0 must pass these tests:

```python
# Basic directional constraints
assert score("hom", "manĝ", "pom") > 0.8
assert score("pom", "manĝ", "hom") < 0.2

# Animacy constraints
assert score("hom", "lern", "lingv") > 0.8
assert score("tabl", "lern", "lingv") < 0.2

# Type constraints
assert score("hom", "manĝ", "pom") > 0.8
assert score("hom", "manĝ", "sonĝ") < 0.3

# Physical action constraints
assert score("hund", "kur", "park") > 0.8
assert score("ĉiel", "kur", "park") < 0.2
```

**Acceptance criteria**: ≥90% accuracy on hand-crafted constraint tests

## Conclusion

The v1.0 plausibility scorer demonstrates that:
1. ✅ The architecture works (simple concatenation + frozen embeddings)
2. ✅ Training is fast and efficient
3. ❌ Automatic SVO extraction produces too much noise (50% false positive rate)
4. ❌ Model learns surface patterns, not semantic constraints

**Next steps**:
1. Don't integrate v1.0 into production (fails basic constraint tests)
2. Rebuild training data with quality filters OR
3. Pivot to hybrid deterministic+learned approach

The concept is sound, but **data quality is the bottleneck**, not model architecture.
