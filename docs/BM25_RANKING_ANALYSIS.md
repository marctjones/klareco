# BM25 Ranking Problem Analysis

**Date:** 2026-03-29
**Issue:** #709
**Related:** #713 (QA Improvement Epic)

---

## Executive Summary

The top-K sweep revealed a **U-shaped accuracy curve** indicating a serious ranking problem:

| top-k | Accuracy | Pattern |
|-------|----------|---------|
| 5 | 66.7% | ✓ Best |
| 10 | 63.3% | ↓ |
| 20 | 50.0% | ← Worst |
| 30 | 53.3% | ↑ |
| 50 | 63.3% | ↑ Recovered |

**Key Finding:** 5 questions SUCCEED at k=5 but FAIL at k=20.

This contradicts normal behavior: More documents should give more chances to find the answer.

---

## Root Cause: Two Distinct Problems

### Problem 1: Retrieval Failures (40% of failures)

**6 questions:** Answer document not retrieved at all (even in top-200 BM25 results).

**Likely causes:**
- Query expansion too conservative (current threshold=0.70 for embeddings)
- Missing synonyms in manual dictionary
- Proper name handling issues

**Example:**
- Query asks about "Lincoln" but doesn't expand to "Abraham", "prezidento"
- Result: Document containing answer never retrieved

---

### Problem 2: Extraction Failures (60% of failures)

**9 questions:** Answer retrieved but extraction fails.

**Answer ranks:** [3, 5, 6, 8, 14, 16, 22, 23, 24]
- 2 in top-5 (ranks 3, 5)
- 2 in ranks 6-10
- 2 in ranks 11-20
- 3 beyond rank 20

**Critical observation:** Even when answer is ranked #3 or #5, extraction still fails.

**Likely causes:**
1. **Object verification missing** - Extracts "fondis GIL" when query asks "fondis Esperanton?"
2. **Definition patterns missing** - WHAT questions need "X estas Y" pattern matching
3. **Temporal extraction missing** - WHEN questions need date/time recognition

---

## The U-Shaped Curve Explained

**Why does k=20 perform WORSE than k=5?**

### Hypothesis: Noise Overwhelms Signal

At k=5:
- Only get top 5 BM25-ranked documents
- High signal-to-noise ratio
- Extraction succeeds on clean signal
- **Result: 66.7% accuracy**

At k=20:
- Get top 20 documents
- Includes noisy documents at ranks 6-20
- Extraction attempts on 20 documents, many irrelevant
- M1 filter rate: 96.1% (filters 96.1% of extracted facts as implausible)
- **Noise obscures signal**
- **Result: 50.0% accuracy**

At k=50:
- Eventually retrieves answer document (even if ranked #30-40)
- Still noisy, but correct document appears
- Extraction has more chances
- **Partial recovery: 63.3% accuracy**

---

## Evidence: Recall Metrics

**From baseline evaluation:**

| Metric | Value |
|--------|-------|
| Recall@5 | 56.7% |
| Recall@10 | 56.7% (SAME!) |
| Recall@20 | 70.0% |

**Recall@5 = Recall@10** means **nothing useful in ranks 6-10.**

BM25 is ranking wrong documents at positions 6-20.

---

## What's in Ranks 6-20? (Analysis Needed)

**To investigate:**

1. **For failed questions at k=20, what documents are at ranks 1-5 vs 6-20?**
   - Are 6-20 synonym expansion noise?
   - Are they partial keyword matches?
   - Are they from wrong Wikipedia articles?

2. **BM25 scores:**
   - What's the score distribution?
   - Is there a natural cutoff around rank 5-6?
   - Can we threshold by score instead of k?

3. **Query expansion contribution:**
   - Which documents come from original query terms?
   - Which come from expanded synonyms?
   - Are expanded synonyms adding noise?

---

## Extraction Failures: Not Just Ranking

**Critical insight:** Answer at rank #3 still fails extraction.

This means extraction has problems INDEPENDENT of ranking:

### Example Extraction Failure

**Query:** "Kiu fondis Esperanton?" (Who founded Esperanto?)

**Document (rank 3):** "La junularo fondis GIL, organizaĵon por esperantistaj gejunuloj."

**Problem:**
- Verb matches: "fondis" ✓
- Object doesn't match: GIL ≠ Esperanto ✗
- Extraction incorrectly returns: "junularo" (wrong answer)

**Solution:** Object verification (Issue #710)

---

## Recommendations

### Immediate (Do First)

1. **#710 - Add object verification** (1-2 hours)
   - Fix extraction failures even when answer is ranked #3-5
   - Pure deterministic AST-based fix

2. **Investigate ranks 6-20 content** (1-2 hours)
   - Manual inspection: What documents appear there?
   - Are they synonym expansion noise?
   - Inform query expansion model design

### Short-Term

3. **#711 - Build learned query expansion distance model** (2-3 days)
   - Learn: "How far should we expand synonyms?"
   - Stop cascade before noise (e.g., fond → kre ✓, but not establ → firme ✗)

4. **Experiment with top-k=10 + object verification**
   - Maybe k=10 is sweet spot after fixing extraction
   - Less noise than k=20, more coverage than k=5

### Medium-Term

5. **#712 - Retrain reranker** (1 week)
   - Fix ranking so useful documents appear in top 20
   - Use precision/recall metrics, not MRR

---

## Success Metrics

After fixes, we should see:

1. **Extraction failures drop from 9 to ~2** (object verification)
2. **Retrieval failures drop from 6 to ~3** (learned expansion)
3. **k=20 accuracy matches or beats k=5** (ranking fixed)
4. **Recall@10 > Recall@5** (something useful in ranks 6-10)

---

## Next Steps

1. ✅ Issue #708 closed - Reranker disabled
2. 🔍 Manual inspection - Check what's at ranks 6-20 for failed questions
3. ⚙️ Issue #710 - Implement object verification
4. 📊 Re-evaluate after each fix
5. 🧠 Issue #711 - Train learned expansion model

---

## Related

- Issue #708: Disable reranker (DONE)
- Issue #710: Object verification
- Issue #711: Learned query expansion
- Issue #713: QA Improvement Epic
