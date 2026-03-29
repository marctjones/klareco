# Test Set Audit and Recommendations

## Overview

Analysis of 4 test sets to determine which are useful for general-purpose Esperanto QA evaluation.

## Test Set Analysis

### 1. ✅ **qa_test_set_50.jsonl** - KEEP (Primary Test Set)

**Size:** 50 questions
**Quality:** High (hand-curated)
**Language:** Esperanto questions with Esperanto keyword evaluation

**Strengths:**
- ✅ Balanced question type distribution (WHO: 10, WHAT: 10, WHERE: 10, WHEN: 10, etc.)
- ✅ High-quality Esperanto questions
- ✅ Esperanto keywords for evaluation (not English)
- ✅ Clean format with metadata (difficulty, category, expected patterns)
- ✅ Focused on Esperanto-specific topics (Zamenhof, Fundamento, etc.)

**Sample:**
```json
{
  "question": "Kiu fondis Esperanton?",
  "expected_keywords": ["zamenhof", "ludovic", "lazaro"],
  "question_type": "WHO"
}
```

**Verdict:** **PRIMARY TEST SET** - Keep and use as default for all evaluations

---

### 2. ⚠️  **qa_test_diverse_30.jsonl** - MERGE INTO #1 OR DELETE

**Size:** 30 questions
**Quality:** Medium (hand-curated but narrow domain)
**Language:** Esperanto questions with Esperanto keywords

**Strengths:**
- ✅ Clean format
- ✅ Esperanto keywords

**Weaknesses:**
- ❌ Heavily skewed to WHO questions (17/30 = 57%)
- ❌ Very focused on American history ("Kiu estis Lincoln?", "Kiu estis Thomas Jefferson?")
- ❌ Too specialized for general-purpose QA (US presidents, not diverse topics)
- ❌ Redundant with qa_test_set_50.jsonl

**Sample:**
```json
{
  "question": "Kiu estis Lincoln?",
  "expected_keywords": ["prezidento", "usona"],
  "category": "american_history"
}
```

**Verdict:** **SPECIALIZED** - Either:
1. **Merge** unique non-duplicate questions into qa_test_set_50.jsonl
2. **Delete** if too specialized for general-purpose needs
3. **Keep separate** only if specifically testing US history domain

**Recommendation:** DELETE (too narrow domain for general-purpose system)

---

### 3. ❌ **generated_questions_200.jsonl** - DELETE (Garbage)

**Size:** 200 questions
**Quality:** Very low (auto-generated, broken)
**Language:** Broken Esperanto

**Critical Problems:**
- ❌ Grammatically broken questions:
  - "Kiu eniris ni mond?" (should be "la mondon")
  - "Kiu havis alir?" (incomplete sentence)
  - "Kiu estas hind lm lm?" (gibberish)
  - "Kion montris ali li teolog?" (word salad)
- ❌ Expected keywords don't match actual answers
- ❌ Makes no semantic sense
- ❌ Heavily skewed (WHO: 102, WHAT: 94, WHEN: 4)

**Sample (Broken):**
```json
{
  "question": "Kiu eniris ni mond?",
  "expected_keywords": ["origin", "unu", "pek"],
  "source_sentence": "Peko eniris la mondon..."
}
```

**Actual correct question from source should be:**
"Kio eniris la mondon?" (What entered the world?) → "Peko" (Sin)

**Verdict:** **GARBAGE** - Delete immediately

**Why it's garbage:**
1. Auto-generated without quality control
2. Grammatical errors that native speakers wouldn't make
3. Keywords don't align with actual answers
4. Would pollute evaluation metrics (false negatives from broken grammar)

---

### 4. ❌ **translated_qa_diverse.jsonl** - DELETE OR FIX (Currently Unusable)

**Size:** 791 questions
**Quality:** Medium questions, broken evaluation
**Language:** Esperanto questions BUT English keywords (broken!)

**Critical Problem:**
- ✅ Good Esperanto questions (translated from TriviaQA)
- ❌ **Expected keywords are in ENGLISH**
- ❌ **Answer variants are in ENGLISH**
- ❌ Makes evaluation impossible (system generates Esperanto, but checks against English)

**Sample (Broken Evaluation):**
```json
{
  "question": "Kiu estis Prezidanto kiam la unua karikaturo de Peanuts estis publikigita?",
  "expected_keywords": ["Presidency of Harry S. Truman"],  ← ENGLISH!
  "answer_variants": ["Harry S. Truman", "President Truman"]  ← ENGLISH!
}
```

**System generates:** "Trumano estis prezidanto..." (Esperanto)
**Evaluation checks for:** "Harry S. Truman" (English)
**Result:** False negative (correct answer marked wrong)

**Verdict:** **UNUSABLE** - Delete OR fix by translating keywords to Esperanto

**Options:**
1. **Delete** - If not worth the effort to fix
2. **Fix** - Translate expected keywords to Esperanto:
   - "Presidency of Harry S. Truman" → ["truman", "prezidento", "usona"]
   - This is a lot of manual work (791 questions!)
3. **Hybrid approach** - Use only as a **retrieval test** (ignore answer evaluation)

**Recommendation:** DELETE (fixing 791 questions is too much effort, and qa_test_set_50 is sufficient)

---

## Summary Table

| Test Set | Size | Quality | Usable? | Recommendation |
|----------|------|---------|---------|----------------|
| **qa_test_set_50.jsonl** | 50 | High | ✅ YES | **KEEP** (primary) |
| **qa_test_diverse_30.jsonl** | 30 | Medium | ⚠️ Specialized | **DELETE** (too narrow) |
| **generated_questions_200.jsonl** | 200 | Very Low | ❌ NO | **DELETE** (garbage) |
| **translated_qa_diverse.jsonl** | 791 | Broken | ❌ NO | **DELETE** (English keywords) |

---

## Recommendations

### Immediate Actions

1. **Keep only:** `qa_test_set_50.jsonl`
   - This is the **primary test set**
   - High quality, balanced, Esperanto-focused
   - Use as default for all evaluations

2. **Delete:**
   - `generated_questions_200.jsonl` (garbage, broken grammar)
   - `translated_qa_diverse.jsonl` (English keywords, unusable)
   - `qa_test_diverse_30.jsonl` (too specialized, redundant)

3. **Update evaluation scripts:**
   - Change default to `qa_test_set_50.jsonl` only
   - Remove references to other test sets

### For General-Purpose QA System

**What you have:** 50 high-quality questions covering 8 question types

**What you need for general-purpose:**
- ✅ Balanced question types (you have this)
- ✅ Diverse topics (you have this - Esperanto, history, geography, etc.)
- ✅ Esperanto evaluation (you have this)
- ❌ Large-scale testing (50 is small, but quality > quantity)

**50 questions is sufficient for:**
- ✅ Development iteration (quick feedback)
- ✅ Detecting regressions
- ✅ Comparing approaches
- ✅ Identifying bottlenecks

**50 questions is NOT sufficient for:**
- ❌ Statistical significance (need 200+ for confidence intervals)
- ❌ Rare question types (only 2 WHY questions)
- ❌ Edge case detection

### Future: Creating Better Test Set

If you need larger test set in the future:

**Option 1: Expand qa_test_set_50.jsonl manually**
- Add 50-100 more hand-curated questions
- Maintain quality standards
- Balance question types (add more WHY, HOW, WHICH)
- Effort: High (manual curation)
- Quality: High

**Option 2: Fix translated_qa_diverse.jsonl**
- Translate expected keywords to Esperanto (791 questions!)
- Validate translations
- Effort: Very High (months of work)
- Quality: Medium (machine translation issues)

**Option 3: Hybrid approach**
- Keep qa_test_set_50.jsonl for answer quality evaluation
- Use translated_qa_diverse.jsonl ONLY for retrieval testing (ignore answer evaluation)
- Effort: Low (no keyword translation needed)
- Quality: Medium (can test retrieval at scale)

**Recommendation:** Stick with qa_test_set_50.jsonl for now. 50 high-quality questions is better than 791 broken ones.

---

## Commands to Clean Up

```bash
# Backup first (just in case)
mkdir -p data/test_sets/archive/
mv data/test_sets/generated_questions_200.jsonl data/test_sets/archive/
mv data/test_sets/translated_qa_diverse.jsonl data/test_sets/archive/
mv data/test_sets/qa_test_diverse_30.jsonl data/test_sets/archive/

# Verify only qa_test_set_50.jsonl remains active
ls -lh data/test_sets/*.jsonl
```

---

## Impact on Evaluation Suite

After cleanup:
- Default test set: `qa_test_set_50.jsonl` (50 questions)
- Full suite timing: 11.1 minutes
- Adaptive suite (10 min): 45 questions (90% of test set)
- Adaptive suite (5 min): 23 questions (46% of test set)

**With only 50 questions total:**
- 10-minute target = use all 50 questions
- 5-minute target = use 23 questions (stratified sample)

This is perfect for rapid iteration!
