# Phase 1-4 Implementation: Initial Evaluation Results

**Date:** 2026-03-29
**Status:** 🔴 **CRITICAL ISSUE FOUND**

---

## Summary

After implementing all 4 phases of AST-first retrieval improvements, initial evaluation shows **WORSE performance than baseline**, not better.

**Expected:** 38% → 58% retrieval recall
**Observed (partial):** ~16% accuracy (3/19 questions in first batch)
**Baseline:** 12-38% (from previous sessions)

---

## Bug Found and Fixed

### TypeError in Grammatical Variant Generation

**Error:**
```
TypeError: can only concatenate list (not "set") to list
```

**Location:** `klareco/rag/grammatical_variants.py` lines 104, 272, 351

**Cause:** `get_synonyms()` returns a `set`, but code tried to concatenate it to a list:
```python
all_verbs = [verb_root] + verb_synonyms  # ❌ verb_synonyms is a set
```

**Fix:** Convert to list before concatenation:
```python
all_verbs = [verb_root] + list(verb_synonyms)  # ✅ Now works
```

**Status:** ✅ Fixed in commit 5e9c778

---

## Partial Evaluation Results (First 19 Questions)

### Results by Question:

**✅ Correct (3/19 = 16%):**
1. WHO: "Kiu fondis Esperanton?" → Zamenhof ✓
2. WHO: "Kiu kreis Esperanton?" → Zamenhof ✓
3. WHAT: "Kio estas hundo?" → besto ✓

**❌ Incorrect (16/19 = 84%):**
- 7 WHO questions failed (expected Zamenhof, got wrong answers)
- 9 WHAT questions failed (expected definitional answers, got wrong answers)

### Observed Problems

**Problem 1: Generic/Irrelevant Retrieval**

Many questions returned the same generic sentence:
```
"Esperanto estas oficiala lingvo de neniu ŝtato, kvankam komence de la
20-a jarcento ekzistis planoj estigi Moresneton..."
```

This sentence appeared as the answer to:
- "Kiu proponis Esperanton?"
- "Kiu ellaboris Esperanton?"
- "Kiu iniciatis Esperanton?"
- "Kio estas Esperanto?"

**This suggests:** Retrieval is returning high-scoring but irrelevant documents.

**Problem 2: Wrong Context Documents**

Questions about Zamenhof retrieved sentences about:
- Butros-Gali (wrong person)
- Nietzsche (wrong person)
- Generic language descriptions
- Grammar examples ("De la patro mi ricevis libron")

**This suggests:** Pattern matching is too broad or matching wrong patterns.

**Problem 3: Definitional Facts Not Retrieved**

WHAT questions failed to retrieve IS-A definitional facts:
- "Kio estas planlingvo?" → Expected "artefarita lingvo" (not found)
- "Kio estas la Fundamento?" → Expected "dokumento" (not found)
- "Kio estas libro?" → Expected "skribaĵo" (not found)
- "Kio estas lingvo?" → Expected "komunikilo" (not found)

**Exception:** "Kio estas hundo?" → "besto" ✓ (This one worked!)

**This suggests:** IS-A detection (Phase 1) is not working as intended for most cases.

---

## Analysis: Why Did Phase 1-4 Make Things Worse?

### Hypothesis 1: Variant Queries Returning Too Much Noise

**Issue:** Grammatical variants have lower confidence (0.7-0.85) but might be:
- Matching too broadly
- Returning irrelevant documents
- Diluting good results with bad results

**Evidence:**
- Same generic sentence appearing for multiple questions
- Many wrong-context retrievals

**Potential Cause:**
- Variant Cypher queries might be too permissive
- Confidence weighting might not be strong enough
- Merging strategy might prioritize wrong documents

### Hypothesis 2: Phase 3 Importance Scoring Broken

**Issue:** Importance scoring integration might be broken:
- Parameters not passed correctly
- Scoring logic has bugs
- Weights are incorrect

**Evidence:**
- Generic sentences scoring higher than specific ones
- Definitional facts not prioritized

**Potential Cause:**
- `_execute_kuzu_query()` parameter passing issue
- FactImportanceScorer not working with new variants
- Score calculation errors

### Hypothesis 3: IS-A Detection (Phase 1) Not Working

**Issue:** IS-A pattern detection only worked for 1/10 WHAT questions.

**Evidence:**
- "Kio estas hundo?" → Worked ✓
- All other WHAT questions → Failed ✗

**Potential Cause:**
- IS-A patterns too strict
- Corpus might not have IS-A facts for most queries
- Kuzu query structure incorrect

### Hypothesis 4: Passive Voice + Variants Causing Confusion

**Issue:** Multiple pattern types competing, causing interference.

**Evidence:**
- Only 2/10 WHO questions about Zamenhof worked
- Other WHO questions retrieved completely wrong people

**Potential Cause:**
- Variant merging logic
- Score comparison across variant types
- Active vs passive vs participial confusion

---

## Immediate Action Items

### 1. Full Evaluation Completion

**Status:** Running (8+ minutes elapsed)

**Need:** Full 50-question results to:
- Confirm if 16% accuracy holds across all questions
- See breakdown by question type
- Identify consistent failure patterns

### 2. Diagnostic Logging

**Action:** Add detailed logging to understand retrieval:
```python
# In each pattern method, log:
- How many results from base pattern?
- How many results from each variant?
- What are the top scores?
- Which variant types contributed to top-k?
```

### 3. Disable Variants, Test Base Patterns Only

**Action:** Temporarily comment out Phase 4 variant generation:
```python
# variant_results = self._execute_variant_queries(...)  # DISABLED FOR TESTING
```

**Why:** Isolate if variants are causing the problem or if base patterns already broken.

### 4. Test Each Phase Independently

**Action:** Create evaluation runs testing incrementally:
- Baseline (no phases)
- Phase 1 only (IS-A detection)
- Phases 1+2 (+ passive voice)
- Phases 1+2+3 (+ importance scoring)
- Phases 1+2+3+4 (+ variants - current)

**Why:** Identify which phase introduced the regression.

### 5. Manual Query Debugging

**Action:** Test specific failing queries with verbose logging:
```bash
python scripts/demo_extractive_qa.py "Kio estas planlingvo?" --no-m1 --no-rerank -v
```

**Focus on:**
- "Kiu verkis la Fundamenton?" (Expected: Zamenhof, Got: Wrong)
- "Kio estas planlingvo?" (Expected: artefarita lingvo, Got: Wrong)
- "Kiu estis Zamenhof?" (Expected: okulisto, Got: Wrong)

---

## Questions to Answer

1. **Does baseline (pre-Phase 1-4) still work?**
   - Need to checkout previous commit and test
   - Verify baseline is actually 12-38% as documented

2. **Is the problem in retrieval or extraction?**
   - Are correct documents being retrieved but ranked poorly?
   - Or are wrong documents being retrieved?

3. **Which phase introduced the regression?**
   - Test each phase incrementally
   - Isolate the breaking change

4. **Are Cypher queries correct?**
   - Validate query syntax
   - Check if patterns match expected sentences
   - Test queries directly against Kuzu

5. **Is importance scoring working?**
   - Check if FactImportanceScorer is being called
   - Verify score breakdowns
   - Compare scores with/without importance

---

## Risk Assessment

**🔴 CRITICAL:** Implementation appears to have regressed performance significantly.

**Impact:**
- Cannot proceed with further improvements until this is fixed
- May need to revert some or all phases
- Significant debugging required

**Next Steps:**
1. ✅ Wait for full evaluation to complete
2. ⏳ Analyze complete results
3. ⏳ Debug retrieval with verbose logging
4. ⏳ Test phases incrementally
5. ⏳ Fix root cause
6. ⏳ Re-evaluate

---

## Notes

- Bug fix committed (5e9c778): Convert verb_synonyms set to list
- Evaluation still running (started 23:14, 8+ minutes elapsed)
- Using --no-m1 --no-rerank flags (AST-only retrieval)
- Test set: 50 questions from `data/test_sets/qa_test_set_50.jsonl`

---

## Status: BLOCKED

**Waiting on:** Full evaluation results
**Blocking:** All further development until retrieval is fixed
**Priority:** 🔴 P0 - Critical bug
