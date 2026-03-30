# Session Summary: Importance Scoring Improvements

**Date:** 2026-03-29
**Focus:** Phase 1 (Proper Noun Detection) + Phase 2 (Embedding Similarity) + Retrieval Analysis

---

## What We Accomplished

### 1. Implemented Phase 1: Proper Noun Detection ✅

**Goal:** Fix "Fundamento" vs "fundamentoj" discrimination problem

**Changes:**
- Added `entity_is_proper_noun` and `entity_capitalized_form` fields to `Fact` dataclass
- Fixed bug: parser outputs `vortspeco="propra_nomo"` but fact extractor checked `"propranomo"`
- Implemented `_get_entity_info()` to extract proper noun status from AST
- Implemented `_entity_matches()` with proper noun-aware matching logic:
  - Proper nouns: Match if query is prefix + Esperanto ending (e.g., "fundament" matches "Fundamento")
  - Common nouns: Match exact root after stripping endings (e.g., "hund" matches "hundo", not "hundego")
- Updated importance scorer to use AST annotations instead of string capitalization

**Result:** +2% end-to-end accuracy, +3.9% generation quality

**Files Modified:**
- `klareco/rag/fact_extractor.py` - Added proper noun tracking
- `klareco/rag/importance_scorer.py` - AST-aware entity matching

---

### 2. Implemented Phase 2: Embedding Similarity ❌ (Then Disabled)

**Goal:** Add learned semantic signal to importance scoring

**Implementation:**
- Added `use_embeddings` parameter to `FactImportanceScorer`
- Implemented `_score_embedding_similarity()` using root embeddings (64D)
- Computed cosine similarity between query roots and fact roots
- Added 10% weight to scoring (reduced other weights proportionally)
- Integrated into extraction pipeline with query root extraction

**Result:** **-4% end-to-end accuracy** (embeddings hurt performance!)

**Root Cause Analysis:**
- Embeddings lack discrimination (all facts about "Esperanto" get 0.88-0.97 similarity)
- Range too narrow (0.49) to help selection
- Dominated by common terms (query "fond + esperant" vs fact "esperant + ..." → high similarity)
- 10% weight adds uniform noise without improving discrimination

**Decision:** **Disabled embeddings**, kept Phase 1 only

**Files Modified:**
- `klareco/rag/importance_scorer.py` - Embedding similarity implementation
- `klareco/rag/extractive_answering.py` - Integration (disabled)

---

### 3. Ablation Study: Measured Individual Contributions ✅

**Created:** `scripts/ablation_study_importance_scoring.py`

**Results on 50 Questions:**

| Configuration | Accuracy | Generation (given retrieval) |
|--------------|----------|------------------------------|
| Baseline | 10.0% | 29.4% |
| **Phase 1 Only** | **12.0%** ✅ | **33.3%** ✅ |
| Phase 2 Only | 6.0% ❌ | 16.7% ❌ |
| Phase 1+2 Combined | 6.0% ❌ | 17.6% ❌ |

**Key Finding:** Phase 1 improves performance, Phase 2 hurts it.

---

### 4. Deep Retrieval Analysis ✅

**Created:**
- `scripts/analyze_retrieval_failures.py` - Comprehensive retrieval diagnostic
- `scripts/debug_embedding_scores.py` - Embedding score analysis
- `docs/RETRIEVAL_IMPROVEMENT_PLAN.md` - Strategic improvement plan

**Key Findings:**

**Retrieval Performance @ k=30:**
- 38% recall (19/50 questions)
- 16% success (8/50) - answer in top 10
- 22% ranking failures (11/50) - answer at rank 11-22
- 62% pattern mismatches (31/50) - AST pattern doesn't match corpus

**Critical Insight:** 22% of questions are **quick wins** - the answer is retrieved but ranked too low (rank 11-22).

**Examples:**
1. "Kiu fondis Esperanton?" → Answer at rank **16** (should be top 3)
2. "Kio estas hundo?" → Answer at rank **22** (should be top 1 for IS-A definition)
3. "Kie okazas konversacio?" → Answer at rank **16** (should be top 5)

---

## Current Performance Baseline

**With Phase 1 Only (Proper Noun Detection):**

| Metric | Performance |
|--------|-------------|
| Retrieval @ k=30 | 38% (19/50) |
| Extraction (given retrieval) | 100% (19/19) ✅ |
| Selection (given extraction) | 100% (19/19) ✅ |
| Generation (given selection) | 33% (6/19) |
| **End-to-End Accuracy** | **12% (6/50)** |

**Bottleneck:** Retrieval (38% recall) + Generation (33% quality)

---

## Why Embeddings Failed

### Technical Analysis

**Expected:** Embeddings would add semantic discrimination
**Reality:** Embeddings added uniform noise

**Evidence:**
```
Query: ["fond", "esperant"]
Fact 1 (correct): ["esperant", "zamenhof"] → similarity 0.94
Fact 2 (wrong): ["esperant", "li"] → similarity 0.97
Fact 3 (wrong): ["esperant", "grup"] → similarity 0.96
```

**Problem:** All facts containing "esperant" get ~0.9+ similarity because:
- Query vector dominated by "esperant" (common term)
- Fact vectors also dominated by "esperant"
- Cosine similarity ≈ 1.0 for everything
- Range: 0.49 (not enough discrimination)

**Lesson Learned:**
- Simple average embeddings don't work when common terms dominate
- Need more sophisticated computation: verb-to-verb similarity only, or TF-IDF weighting
- OR: Use embeddings only as tiebreaker when deterministic scores are equal

---

## Next Steps (Priority Order)

### Immediate (This Week) - Phase 1 Ranking Improvements

**Target:** 38% → 45% retrieval recall @ k=30

**Actions:**
1. **Boost IS-A facts for WHAT questions** (fix "Kio estas hundo?" ranking #22 → #1)
2. **Boost agent facts for WHO questions** (fix "Kiu fondis?" ranking #16 → top 3)
3. **Penalize generic facts** (reduce score for facts with no entity match)

**Expected Impact:** +12% recall (11 ranking failures → 5 failures)

**Implementation:** Add targeted boosts in `importance_scorer.py`

---

### Short-term (Next 2 Weeks) - BM25 Fallback

**Target:** 45% → 52% retrieval recall @ k=30

**Actions:**
1. Implement hybrid retrieval: AST-first, BM25 fallback when 0 results
2. Merge strategies for combining AST + BM25 results
3. Test on 50-question set

**Expected Impact:** +32% on pattern mismatch failures (31 → 15 failures)

---

### Medium-term (3-6 Weeks) - Pattern Variant Expansion

**Target:** 52% → 60% retrieval recall @ k=30

**Actions:**
1. Design grammatical pattern variants (passive voice, participial, appositive)
2. Implement variant generation in AST pattern matching
3. Test and optimize deduplication strategy

**Expected Impact:** +20% on remaining pattern mismatches (15 → 5 failures)

---

## Performance Targets

| Metric | Current | After Ranking | After BM25 | After Variants | Final Goal |
|--------|---------|---------------|------------|----------------|------------|
| Retrieval @ k=30 | 38% | 45% | 52% | 60% | 60%+ |
| End-to-End | 12% | 18% | 23% | 30% | 30-35% |
| Timeline | Today | 1 week | 3 weeks | 6 weeks | Q2 2026 |

---

## Files Created/Modified

### New Files
1. `scripts/ablation_study_importance_scoring.py` - Component contribution analysis
2. `scripts/debug_embedding_scores.py` - Embedding discrimination analysis
3. `scripts/analyze_retrieval_failures.py` - Comprehensive retrieval diagnostic
4. `docs/RETRIEVAL_IMPROVEMENT_PLAN.md` - Strategic improvement roadmap
5. `docs/SESSION_2026_03_29_IMPORTANCE_SCORING.md` - This summary

### Modified Files
1. `klareco/rag/fact_extractor.py` - Added proper noun detection (Phase 1)
2. `klareco/rag/importance_scorer.py` - Proper noun matching + embedding similarity (Phase 1+2)
3. `klareco/rag/extractive_answering.py` - Integrated improvements (embeddings disabled)

### Results Files
1. `results/ablation_study_importance_scoring.json` - Ablation study data
2. `results/retrieval_failure_analysis.json` - Detailed failure analysis
3. `results/extraction_diagnosis.json` - Pipeline stage diagnosis

---

## Key Decisions Made

1. **✅ Keep Phase 1 (Proper Noun Detection)** - Provides measurable improvement
2. **❌ Disable Phase 2 (Embedding Similarity)** - Hurts performance due to lack of discrimination
3. **🎯 Focus on Retrieval** - It's the primary bottleneck (38% vs extraction 100%)
4. **🚀 Prioritize Ranking Improvements** - 22% of questions are quick wins

---

## Questions for User

1. **Ranking improvements** - Should I implement Phase 1 ranking boosts now? (1 hour work, +7% expected)
2. **BM25 fallback** - Is 2-week timeline acceptable for hybrid retrieval?
3. **Embedding fix** - Should we try to fix embeddings (verb-only similarity) or abandon them?
4. **Full evaluation** - Run on all 761 questions to get better statistics? (~4 hours)

---

## Conclusion

**Today's Success:**
- ✅ Implemented proper noun detection (+2% accuracy)
- ✅ Discovered why embeddings fail (lack of discrimination)
- ✅ Identified 22% quick wins (ranking improvements)
- ✅ Created comprehensive retrieval improvement plan

**The Path Forward is Clear:**
1. Fix ranking (1 week) → +7% improvement
2. Add BM25 fallback (2 weeks) → +14% improvement
3. Expand patterns (3 weeks) → +8% improvement
4. **Total: 30% end-to-end accuracy in 6 weeks** (2.5x improvement from baseline)

The foundation is strong. The next wins are within reach.
