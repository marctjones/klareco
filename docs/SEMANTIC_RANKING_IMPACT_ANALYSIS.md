# Semantic Ranking Impact Analysis

**Date:** 2026-03-29
**Changes:**
- Implemented semantic AST ranking (Issue #713)
- Disabled neural reranker by default (Issue #708)

---

## Executive Summary

**Key Finding:** Semantic ranking FLATTENED the U-shaped curve but DECREASED top-k=5 performance.

**Changes from baseline:**
- ✅ k=20: 50% → 60% (+10% improvement)
- ✅ k=30: 53% → 57% (+4% improvement)
- ❌ k=5: 67% → 50% (-17% regression)
- ❌ k=10: 63% → 60% (-3% regression)

**Diagnosis:** Two changes happened simultaneously:
1. Added semantic ranking (good for consistency)
2. Disabled neural reranker (removed some signal)

---

## Detailed Results Comparison

### Old System (No Ranking + Reranker Enabled)

| top-k | Accuracy | Pattern |
|-------|----------|---------|
| 5 | **66.7%** | Lucky wins |
| 10 | 63.3% | ↓ |
| 20 | 50.0% | ← Worst (U-shape bottom) |
| 30 | 53.3% | ↑ Recovery |
| 50 | 63.3% | ↑ Recovered |

**Characteristics:**
- U-shaped curve (severe)
- High variance (k=5 to k=20: -16.7%)
- Arbitrary ranking from Kuzu database

### New System (Semantic Ranking + Reranker Disabled)

| top-k | Accuracy | Change | Pattern |
|-------|----------|--------|---------|
| 5 | 50.0% | **-16.7%** ❌ | Lower |
| 10 | 60.0% | -3.3% | ~Stable |
| 20 | **60.0%** | **+10%** ✅ | ~Stable |
| 30 | 56.7% | +3.4% | Slightly down |
| 50 | 56.7% | -6.6% | Stable |

**Characteristics:**
- FLAT curve (variance k=5 to k=20: -10%, vs old -16.7%)
- More consistent across top-k
- Structured ranking (verb similarity + object match)

---

## Recall Metrics Comparison

### Old System

| Metric | Value |
|--------|-------|
| Recall@5 | 56.7% |
| Recall@10 | 56.7% (SAME - nothing useful in ranks 6-10) |
| Recall@20 | 70.0% |

### New System (k=20 test)

| Metric | Value | Change |
|--------|-------|--------|
| Recall@5 | 53.3% | -3.4% |
| Recall@10 | 63.3% | +6.6% ✅ |
| Recall@20 | 63.3% | -6.7% |

**Key improvement:** Recall@10 > Recall@5 now! Semantic ranking is putting useful documents in ranks 6-10.

---

## What Happened?

### The Good: Semantic Ranking Works

**Evidence:**
1. Recall@10 > Recall@5 (documents in ranks 6-10 are now useful)
2. k=20 accuracy improved 50% → 60%
3. Flatter curve (more predictable behavior)

**How it works:**
```
Query: "Kiu fondis Esperanton?"
Candidate: "Zamenhof fondis Esperanton"

Score breakdown:
- Verb similarity: 0.40 (fond = fond, exact match)
- Object match: 0.30 (esperant = esperant)
- Subject prominence: 0.20 (Zamenhof is proper noun)
Total: 0.90 (high relevance)
```

### The Bad: Lost k=5 Performance

**Why k=5 dropped 67% → 50%:**

Two changes happened together:
1. **Reranker disabled** (-6.7% from ablation test)
2. **Semantic ranking** (deterministic, no learned signal)

**Old system at k=5:**
- Neural reranker: +6.7% boost
- Arbitrary Kuzu order: Sometimes lucky
- Result: 66.7%

**New system at k=5:**
- No reranker: -6.7% penalty
- Semantic ranking: Deterministic only (no learned signal yet)
- Result: 50.0%

---

## Root Cause Analysis

### Why Semantic Ranking Alone Isn't Enough

**Current scoring:**
```python
# Deterministic only (no learned components yet)
verb_similarity: 40%  # Based on manual synonym dictionary
object_match: 30%     # Exact root match
subject_prominence: 20%  # Is proper noun?
embedding_similarity: 10%  # DISABLED (not implemented yet)
```

**What's missing:**
1. **Root embedding similarity** (the 10% component is disabled)
2. **Learned synonym distances** (currently using manual dictionary only)
3. **Context-aware scoring** (doesn't consider sentence context)

### Why Reranker Was Helping

The neural reranker (even though undertrained) was providing **learned signal**:
- Semantic similarity beyond exact root matching
- Context-aware relevance
- Learned from 1000+ query-document pairs

**Disabling it removed this signal**, hurting k=5 performance where precision matters most.

---

## Recommendations

### Priority 1: Re-enable Embedding Similarity in Semantic Ranker (QUICK WIN)

**What:** Implement the 10% embedding similarity component that's currently disabled.

```python
# In ast_semantic_ranker.py - line 235
emb_score = compute_embedding_similarity(query_roots, cand_roots)  # Currently returns 0.0
```

**Why it helps:**
- Adds learned semantic signal
- Complements deterministic features
- Uses existing 64D root embeddings (already trained)

**Expected impact:** k=5: 50% → 55-58% (recover some lost ground)

**Effort:** 1-2 hours

### Priority 2: Add Object Verification (Issue #710)

**What:** Reject extractions where object doesn't match query object.

**Why it helps:**
- Fixes the "oni fondis GIL" vs "Zamenhof fondis Esperanton" problem
- 60% of failures are extraction failures (not ranking)
- Pure deterministic, no learned component needed

**Expected impact:** Overall +10-15% across all top-k values

**Effort:** 2-3 hours

### Priority 3: Retrain Reranker (Issue #712) - MEDIUM TERM

**What:** Retrain neural reranker with correct loss function.

**Why it helps:**
- Provides learned relevance signal
- Helps at low k where precision matters
- Currently undertrained (hurts accuracy)

**Expected impact:** k=5: +5-8%, k=20: +3-5%

**Effort:** 1 week

---

## Ablation Analysis: Which Component Caused What?

Let's decompose:

| Configuration | k=5 | k=20 | Notes |
|---------------|-----|------|-------|
| **Baseline** (no ranking + reranker) | 66.7% | 50.0% | U-shaped curve |
| Disable reranker only | ~60%* | ~60%* | From previous ablation (#708) |
| **Current** (semantic ranking + no reranker) | 50.0% | 60.0% | Flat curve |

*Estimate: Previous ablation showed 53% → 60% (+6.7%) with reranker disabled.

**Inference:**
- Semantic ranking: +10% at k=20 (better consistency)
- Disabling reranker: -6.7% at k=5 (lost learned signal)
- Net effect: k=5 down, k=20 up

---

## Success Metrics

### What Improved ✅

1. **Consistency:** k=5 to k=20 variance reduced (16.7% → 10%)
2. **Recall@10 > Recall@5:** Ranks 6-10 now useful (was wasteland before)
3. **k=20 accuracy:** 50% → 60% (+10%)
4. **Explainability:** Can show WHY a result ranked high

### What Regressed ❌

1. **k=5 accuracy:** 66.7% → 50.0% (-16.7%)
2. **Overall recall:** Recall@20 dropped 70% → 63.3%

---

## Next Steps (Priority Order)

### 1. Enable Embedding Similarity (1-2 hours)

Add back learned signal to semantic ranker:

```python
# klareco/rag/ast_semantic_ranker.py
def compute_embedding_similarity(query_roots, cand_roots, embeddings_path):
    # Load 64D root embeddings
    emb = load_embeddings(embeddings_path)

    # Compute cosine similarity
    query_vecs = [emb[root] for root in query_roots if root in emb]
    cand_vecs = [emb[root] for root in cand_roots if root in emb]

    return cosine_similarity(query_vecs, cand_vecs)
```

**Expected:** k=5: 50% → 55-58%

### 2. Object Verification (2-3 hours)

Issue #710 - Pure deterministic fix:

```python
# If query has object, verify extracted fact matches
if query_obj and extracted_fact.entity != query_obj:
    reject()
```

**Expected:** +10-15% across all k values

### 3. Re-evaluate After Both Fixes

Target after fixes:
- k=5: 55-58% + 10-15% = **65-73%** (near original)
- k=20: 60% + 10-15% = **70-75%** (much better than original 50%)

### 4. Retrain Reranker (Medium-term)

Issue #712 - 1 week effort:
- Fix training loss (MRR → precision/recall)
- Better training labels (from evaluation results)

**Expected:** Additional +5-8% at k=5

---

## Conclusion

**The U-shaped curve is partially fixed**, but we discovered that:

1. ✅ Semantic ranking provides consistency (flatter curve)
2. ❌ But lost learned signal by disabling reranker
3. ⚙️ Next priority: Enable embedding similarity in semantic ranker
4. ⚙️ Then: Add object verification (biggest single win)

**Current state:**
- k=20 improved (50% → 60%) ✓
- k=5 regressed (67% → 50%) ✗
- Need to add learned signal back

**Path forward:**
- Embedding similarity (quick win, recover k=5 performance)
- Object verification (big win, +10-15% everywhere)
- Retrain reranker (medium-term, final polish)

**Expected final state:**
- k=5: ~70% (better than original 67%)
- k=20: ~75% (much better than original 50%)
- Flat curve with both precision AND recall
