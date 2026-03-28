# Reranker Diagnostic Session - 2026-03-23

## Problem Statement

User reported: "The reranker we built was pretty good, what happened to it?"

After implementing deterministic QA improvements (query expansion threshold, entity-aware retrieval, question-type filtering), the system was still giving wrong answers for "Kiu fondis Esperanton?" even WITH the reranker enabled.

## Hypothesis

Initial hypothesis: The reranker is not working or not scoring sentences correctly.

## Diagnostic Process

### Step 1: Test Reranker Directly

Created `/tmp/test_reranker_scores.py` to score retrieved sentences with the reranker.

**Result**: Reranker correctly scored "Zamenhof kreis Esperanton" highest (0.8368)!

**Conclusion**: The reranker IS working correctly.

### Step 2: Check What's Being Retrieved

Added debug output to see what sentences were being retrieved by the demo script.

**Finding**: Retrieved 268 documents matching query roots, but **0 documents mentioning Zamenhof**!

**Conclusion**: The problem is NOT the reranker - it's RETRIEVAL.

### Step 3: Identify Retrieval Issue

**Original Kuzu query** (line 283-288):
```cypher
MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)
WHERE ft.teksto IS NOT NULL
RETURN ft.teksto AS text, ft.id AS id, d.titolo AS doc_title, d.metadatenoj AS metadata
LIMIT {top_k * 50}  # 1000 for top_k=20
```

**Test script query**:
```cypher
MATCH (ft:Frazoteksto)
WHERE ft.teksto IS NOT NULL
RETURN ft.teksto AS text, ft.id AS id
LIMIT 1000
```

**Difference**:
- `MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)` returns a different subset than `MATCH (ft:Frazoteksto)`
- Kuzu returns results in arbitrary order (likely insertion order or node ID order)
- The perfect answer "Zamenhof kreis Esperanton" was beyond the first 1000 results

### Step 4: Fix Retrieval

**Changes made**:

1. **Simplified query** (line 285):
   - Changed from: `MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)`
   - Changed to: `MATCH (ft:Frazoteksto)`
   - Reason: Direct sentence matching, no path traversal

2. **Increased LIMIT** (line 289):
   - Changed from: `LIMIT {top_k * 50}` (1000 for top_k=20)
   - Changed to: `LIMIT 5000`
   - Reason: Ensure good candidates are available regardless of Kuzu ordering

### Step 5: Verify Fix

**Test**: `python scripts/demo_extractive_qa.py "Kiu fondis Esperanton?"`

**Result**:
```
✅ CORRECT ANSWER:

[1] "Laŭ li mem, Zamenhof kreis Esperanton por la tuta homaro."
    (According to him, Zamenhof created Esperanto for all humanity.)

[2] "La genia doktoro Zamenhof kreis Esperanton."
    (The brilliant doctor Zamenhof created Esperanto.)
```

**Additional test**: `"Kiu estis Zamenhof?"`

**Result**: Correct biographical information from Wikipedia

## Root Cause

The issue was **NOT** the reranker or M1 filtering. The issue was:

1. **Retrieval bottleneck**: Kuzu query `LIMIT 1000` didn't return the best sentences in arbitrary order
2. **Path query subset**: Using `MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)` returned a different subset than all sentences
3. **Missing candidates**: The perfect answer was not in the retrieved pool for reranking

## Key Insight

**The reranker can only rerank what it's given.** If the retrieval doesn't include good candidates, reranking can't help.

This validates the **retrieval-reranking architecture**:
1. Retrieve broadly (large LIMIT)
2. Filter by matching roots
3. Rerank with neural model
4. Select top-k

The bottleneck was step 1 (retrieval) being too narrow.

## Performance Impact

**Before fix**:
- Retrieved: 268 documents, 0 mentioning Zamenhof
- Answer: Wrong (about word usage)
- Reranker contribution: 0% (no good candidates to rerank)

**After fix**:
- Retrieved: 993 documents, 14 mentioning Zamenhof
- Answer: Correct ("Zamenhof kreis Esperanton")
- Reranker contribution: HIGH (correctly scored perfect answer 0.8368)

## Files Modified

1. `/home/marc/Projects/klareco/scripts/demo_extractive_qa.py`:
   - Line 285: Simplified Kuzu query to `MATCH (ft:Frazoteksto)`
   - Line 289: Increased LIMIT to 5000
   - Line 118: Query expansion threshold 0.4 → 0.65 (earlier fix)
   - Line 221: Disabled entity-aware retrieval for WHO questions (earlier fix)
   - Lines 317-351: Added proper noun boosting for WHO questions (earlier fix)

2. `/home/marc/Projects/klareco/docs/DETERMINISTIC_VS_NEURAL_QA_TEST.md`:
   - Added "✅ COMPLETED FIXES" section documenting all changes
   - Added test results showing correct answers

## Lessons Learned

1. **Debug reranker by checking retrieved candidates first**
   - If good candidates aren't retrieved, reranking can't help

2. **Kuzu query ordering matters**
   - Without ORDER BY, results are in arbitrary order
   - Path queries `[*1..3]` may return different subsets
   - Use generous LIMIT to ensure coverage

3. **Test retrieval independently**
   - Create diagnostic scripts to inspect retrieval before reranking
   - Check if expected sentences are in retrieved pool

4. **User feedback was correct**
   - "The reranker we built was pretty good" - YES, it was working!
   - The problem was elsewhere in the pipeline

## Recommendations

1. **Consider adding ORDER BY to Kuzu queries**
   - Order by document quality signal (e.g., Wikipedia first)
   - Or order by sentence position in document
   - This reduces dependence on large LIMIT

2. **Monitor retrieval quality**
   - Log: % of queries where perfect answer is in retrieved pool
   - Alert if retrieval quality drops

3. **Tune LIMIT dynamically**
   - For WHO questions, may need larger LIMIT
   - For WHAT questions, smaller LIMIT may suffice

4. **Document query semantics**
   - `MATCH (ft:Frazoteksto)` vs `MATCH (d:Dokumento)-[*1..3]-(ft:Frazoteksto)`
   - When to use each, expected behavior

---

**Last Updated**: 2026-03-23
**Author**: Claude Sonnet 4.5 (with Marc)
**Status**: Resolved - Fix verified working
