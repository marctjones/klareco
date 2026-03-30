# Debugging Session: Phase 1-4 Implementation Issues

**Date:** 2026-03-29
**Status:** 🟡 **PARTIAL PROGRESS** - 2 critical bugs fixed, 1 remaining

---

## Summary

Phase 1-4 implementation had **3 critical bugs** that caused complete failure:

1. ✅ **FIXED**: TypeError in variant generation (verb_synonyms set → list)
2. ✅ **FIXED**: Compound word handling (planlingvo → lingv extraction error)
3. 🔴 **REMAINING**: Importance scoring not discriminating (all scores = 0.400)

---

## Bug #1: TypeError in Grammatical Variant Generation

### Symptom
```
TypeError: can only concatenate list (not "set") to list
```

### Root Cause
`get_synonyms()` returns a `set`, but code tried to concatenate to list:
```python
all_verbs = [verb_root] + verb_synonyms  # ❌ verb_synonyms is a set
```

### Fix
Convert to list before concatenation in all 3 methods:
```python
all_verbs = [verb_root] + list(verb_synonyms)  # ✅
```

**Status**: ✅ Fixed in commit 5e9c778

---

## Bug #2: Compound Word Handling

### Symptom
WHAT questions for compound words failed to retrieve definitional sentences:
- "Kio estas planlingvo?" retrieved generic language sentences, not definitions
- Expected: "planlingvo estas... konstruita" (definitional)
- Got: "Kelkaj planlingvoj..." (narrative)

###Root Cause

**Part 1: Entity Extraction**
Parser breaks compound words into parts:
```python
# AST for "planlingvo":
{
  "radiko": "lingv",  # Main root only!
  "kunmetitaj_radikoj": ["plan", "lingv"]  # Full compound
}
```

Entity extraction only used `radiko`, so:
- Query: "Kio estas planlingvo?"
- Extracted: "lingv" (wrong - too general)
- Should extract: "planlingv" (full compound)

**Part 2: Database Schema**
Kuzu database stores words with separate compound roots:
- No words have `radiko='planlingv'`
- Words have `radiko='lingv'` + `kunmetitaj_radikoj=['plan','lingv']`
- IS-A queries matched `WHERE subj.radiko = '{entity_root}'` - fails for compounds!

### Fix

**Part 1: Extract full compound in `_extract_verb_and_object()`**
```python
if alia.get('vortspeco') == 'substantivo':
    if alia.get('kunmetitaj_radikoj'):
        obj_root = ''.join(alia['kunmetitaj_radikoj'])  # "planlingv"
    else:
        obj_root = alia.get('radiko')  # "hund"
```

**Part 2: Match using `plena_vorto` in IS-A queries**
```python
# OLD: WHERE subj.radiko = 'lingv'  ❌ Too general
# NEW: WHERE subj.plena_vorto STARTS WITH 'planlingv'  ✅ Matches compounds
```

### Results
```bash
# Before fix:
IS-A query for "lingv" → 20 generic language sentences
Top result: "La lingvo estas..."  # Wrong

# After fix:
IS-A query for "planlingv" → 10 definitional sentences including:
Result #6: "Internacia planlingvo... estas planlingvo konstruita..."  ✓ CORRECT

# But ranked #7 after semantic ranking (all scores = 0.400)
```

**Status**: ✅ Fixed in commit 942d9be, but **importance scoring needed to rank #1**

---

## Bug #3: Importance Scoring Not Discriminating

### Symptom
All retrieved sentences have identical scores (0.400), regardless of definitional vs narrative content:
```
1. Score: 0.400  # Narrative: "Estas kelkaj planlingvoj..."
2. Score: 0.400  # Narrative: "Estis kreitaj..."
...
7. Score: 0.400  # DEFINITIONAL: "...estas planlingvo konstruita..."  ← SHOULD BE #1!
```

### Expected Behavior
Phase 3 importance scoring should heavily weight definitional facts:
- Definitional: Score > 0.7 (high importance)
- Narrative: Score < 0.4 (low importance)

### Diagnostic Questions

1. **Is FactImportanceScorer being called?**
   - Need to add logging to verify

2. **Are importance scores being calculated?**
   - Check if score breakdown exists
   - Verify not all returning same score

3. **Are scores being applied to final ranking?**
   - Weight distribution: grammatical 30%, **importance 40%**, subject 20%, emb 10%
   - Check if 40% weight is actually being applied

4. **Are parameters being passed correctly?**
   - `question_type='KIO'` → Should trigger IS-A boost
   - `query_entity='planlingv'` → Should match entity in facts
   - `query_roots=['planlingv']` → For context

### Investigation Needed

Add debug logging to `rank_ast_matches()`:
```python
if use_importance_scoring and importance_scorer:
    logger.info(f"IMPORTANCE SCORING ENABLED")
    logger.info(f"  question_type: {question_type}")
    logger.info(f"  query_entity: {query_entity}")

    fact_breakdown = importance_scorer.score(...)
    logger.info(f"  Importance score: {fact_breakdown.final_score}")
    logger.info(f"    - Question relevance: {fact_breakdown.question_relevance}")
    logger.info(f"    - Definitional: {fact_breakdown.definitional}")
```

**Status**: 🔴 **NOT FIXED** - Requires investigation

---

## Current State

### What Works ✅
1. Kuzu database connection (13GB, 5.4M Frazoteksto nodes)
2. IS-A pattern retrieval (finds definitional sentences)
3. Compound word extraction ("planlingvo" → "planlingv")
4. Grammatical variants execute without errors

### What's Broken 🔴
1. **Importance scoring** - Not discriminating between definitional vs narrative
2. **Result ranking** - All scores identical (0.400)
3. **WHAT questions** - Definitional sentences ranked #7 instead of #1

### Impact on Evaluation

Partial run (19/50 questions):
- WHO: 2/10 correct (20%) - Similar to baseline
- WHAT: 1/10 correct (10%) - Worse than expected (target: 60%)

**Root cause**: Definitional sentences ARE being retrieved but NOT ranked first.

---

## Next Steps

### Immediate (Priority 1)
1. **Debug importance scoring**
   - Add logging to rank_ast_matches()
   - Verify FactImportanceScorer is called
   - Check score breakdowns

2. **Test with one query**
   ```bash
   python -c "
   # Add debug logging
   # Test 'Kio estas planlingvo?'
   # Print importance scores
   "
   ```

3. **Fix ranking weights**
   - Verify 40% importance weight is applied
   - Check if scores are being normalized/capped
   - Ensure definitional boost (+0.2) is working

### After Fix (Priority 2)
1. Run full 50-question evaluation
2. Compare to baseline and expected targets
3. Test all question types (WHO, WHAT, WHERE, WHEN)

### Future Improvements (Priority 3)
1. Tune importance weights if needed
2. Add more synonym patterns
3. Improve passive voice detection

---

## Lessons Learned

### 1. Database Schema Matters
Kuzu stores compound words differently than expected:
- Don't assume `radiko` contains full word
- Check for `kunmetitaj_radikoj` array
- Use `plena_vorto` for matching

### 2. Test Each Component Independently
Should have tested:
- Entity extraction alone
- IS-A queries alone
- Importance scoring alone
- Full pipeline together

### 3. Add Debug Logging Early
Would have caught issues faster with logging:
- What entity was extracted?
- What Kuzu queries were executed?
- What importance scores were calculated?

### 4. Verify Assumptions
Assumptions that were wrong:
- ❌ "Retrieval will automatically use correct database" (used wrong path initially)
- ❌ "Entity extraction handles compounds correctly" (it didn't)
- ❌ "Importance scoring works out of the box" (it doesn't discriminate)

---

## Files Modified

| File | Changes | Commits |
|------|---------|---------|
| `klareco/rag/grammatical_variants.py` | Fix verb_synonyms set→list | 5e9c778 |
| `klareco/rag/whoosh_retriever.py` | Compound word extraction, IS-A matching | 942d9be |

---

## Commits

1. **5e9c778**: Fix bug: convert verb_synonyms set to list
2. **942d9be**: Fix compound word handling in entity extraction and IS-A queries

---

## Status Summary

🟡 **PARTIAL PROGRESS**

**Fixed (2/3 bugs)**:
- ✅ Variant generation TypeError
- ✅ Compound word handling

**Remaining (1/3 bugs)**:
- 🔴 Importance scoring not discriminating

**Next Action**: Debug importance scoring with logging, fix ranking weights.
