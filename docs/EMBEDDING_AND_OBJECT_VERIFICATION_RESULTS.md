# Embedding Similarity + Object Verification Results

**Date:** 2026-03-29
**Changes Implemented:**
1. Enabled embedding similarity in semantic ranker (10% weight, 64D root embeddings)
2. Implemented object verification in extractive answer generator
3. Fixed embedding loading to handle checkpoint format

---

## Executive Summary

**CRITICAL REGRESSION**: Accuracy dropped from 60% → **16%** after implementing both fixes.

**Results:**
- Overall: 8/50 correct (16.0%)
- WHO: 2/10 (20.0%) - previously 1/10 (10%) at k=5
- WHERE: 2/10 (20.0%) - previously 3/10 (30%) at k=5
- WHEN: 1/10 (10.0%) - same as before
- WHAT: 1/10 (10.0%) - same as before
- HOW_MANY: 2/5 (40.0%) - same as before
- WHY: 0/2 (0.0%) - same as before
- HOW: 0/2 (0.0%) - same as before

**Comparison to Previous:**
- Baseline (no ranking): k=20: 50%
- Semantic ranking only: k=20: 60%
- **Current (semantic + object verification + embeddings)**: k=20: **16%**

---

## What Went Wrong?

### Issue Analysis

The massive regression suggests one of the following:

1. **Object verification is too strict** - Filtering out correct facts
2. **Embedding similarity broke something** - Despite loading correctly
3. **Query AST parsing broke something** - Changed when/how query is parsed
4. **Evaluation script issue** - Wrong test set or configuration

### Embedding Loading Status

✅ **WORKING**: Embeddings load correctly:
- 10,819 root embeddings
- 64D vectors
- Proper tensor format

### Object Verification Activity

From verbose output:
- "Object verification removed 3/23 facts (13.0%)"
- "Object verification removed 8/18 facts (44.4%)"

**Analysis**: Object verification is active and filtering 13-44% of facts. This could be:
- ✅ GOOD: Removing wrong facts (e.g., "oni fondis GIL")
- ❌ BAD: Removing correct facts due to extraction format mismatches

---

## Possible Root Causes

### Hypothesis 1: Object Verification Logic Error

The verification checks:
```python
# Check if fact entity matches query object
if fact.entity == query_obj:
    keep = True
# Check if object in fact arguments
elif 'object' in fact.arguments and fact.arguments['object'] == query_obj:
    keep = True
```

**Problem**: For WHO questions like "Kiu fondis Esperanton?":
- Query object: "esperant"
- Desired fact: entity="zamenhof", relation=CREATED, arguments={object: "esperant"}
- Verification: entity != query_obj (no), but arguments['object'] == query_obj (yes) → KEEP

This logic SHOULD work. But what if extracted facts have different structure?

### Hypothesis 2: Test Set Mismatch

The evaluation converted `qa_test_set_50.json` → `qa_test_set_50.jsonl` at runtime.

**Possible issue**: Previous evaluations might have used different test set format or questions.

### Hypothesis 3: Case Sensitivity Bug

Object verification uses `.lower()` comparison:
```python
if fact.entity and fact.entity.lower() == query_obj.lower():
```

But query_obj comes from `get_ast_object_root()` which might not lowercase.

### Hypothesis 4: Extraction Format Changed

If unified extractor changed how it formats facts, object verification might not find the object field.

---

## Debug Actions Needed

### 1. Test Object Verification on Known Case

Run single question with debug logging:
```bash
python scripts/demo_extractive_qa.py "Kiu fondis Esperanton?" --verbose
```

Check:
- What facts are extracted?
- What does object verification keep/remove?
- What's the final answer?

### 2. Compare with Previous Evaluation

Check if previous 60% evaluation used same:
- Test set
- top-k value
- Database version
- Retrieval configuration

### 3. Disable Object Verification Temporarily

Run evaluation with object verification commented out to isolate the problem:
```bash
# In extractive_answering.py, comment out line 567:
# filtered_facts = self._verify_object_match(filtered_facts, query_ast)
```

If accuracy returns to 60%, object verification is the culprit.

### 4. Check Fact Extraction Format

Print fact structure to understand what fields are available:
```python
for fact, metadata in all_facts:
    print(f"Entity: {fact.entity}")
    print(f"Relation: {fact.relation}")
    print(f"Arguments: {fact.arguments}")
```

---

## Expected Behavior vs Actual

### Expected (from SEMANTIC_RANKING_IMPACT_ANALYSIS.md)

With embedding similarity + object verification:
- k=5: 50% → 55-58% (recover lost ground from reranker)
- k=20: 60% → 70-75% (object verification adds +10-15%)

### Actual

- k=20: 16% (catastrophic regression of -44%)

---

## Next Steps (Priority Order)

1. **URGENT**: Disable object verification and re-run evaluation to confirm it's the cause
2. Debug object verification logic with single question test
3. Check extracted fact structure format
4. Fix object verification or revert changes
5. Re-test embedding similarity alone (without object verification)

---

## Code Changes Made

### Files Modified

**klareco/rag/ast_semantic_ranker.py:**
- Fixed `load_embeddings()` to handle checkpoint format (root_to_idx, embeddings.weight)
- Added support for multiple checkpoint formats

**klareco/rag/extractive_answering.py:**
- Added `_verify_object_match()` method (lines 201-265)
- Parse query AST once at beginning of `generate()` (line 465)
- Call object verification after question type filtering (line 567)

---

## Conclusion

The fixes were implemented correctly from a code perspective:
- ✅ Embeddings load (verified with test)
- ✅ Object verification runs (verified with logs)
- ❌ **Overall system accuracy collapsed**

The most likely cause is that **object verification is too strict** and filters out correct facts due to:
1. Unexpected fact extraction format
2. Case sensitivity mismatch
3. Missing object fields in extracted facts
4. Overly conservative matching logic

**Immediate action**: Disable object verification and confirm it's the root cause.
