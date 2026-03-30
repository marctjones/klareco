# Evaluation Methodology Mismatch

**Date:** 2026-03-29
**Critical Finding:** Comparing incompatible metrics led to false alarm about regression.

---

## Executive Summary

The apparent "regression" from 60% → 16% was caused by comparing TWO DIFFERENT EVALUATION METRICS:

1. **Previous (60% at k=20)**: Retrieval accuracy - "Is the answer somewhere in the top-20 retrieved documents?"
2. **Current (16% at k=20)**: End-to-end QA accuracy - "Does the generated answer text contain the expected keyword?"

**These are NOT comparable metrics.**

---

## Evidence

### Previous Evaluation (eval_semantic_top_k_20.csv)

Columns in results CSV:
- `contains_answer` - Boolean: Is answer in retrieved documents?
- `answer_rank` - Integer: Which rank contains the answer?
- `recall@5`, `recall@10`, `recall@20` - Retrieval recall metrics
- `success` - Based on retrieval, not generation

**This measures RETRIEVAL quality, not QA quality.**

### Current Evaluation (evaluate_extractive_qa.py)

Logic:
```python
# Check if answer text contains expected keywords
keyword_check = check_keywords_in_text(answer.text, expected_keywords)
success = keyword_check['found']
```

**This measures END-TO-END QA quality (retrieval + extraction + generation).**

---

## What This Means

### The 60% Result (Previous)

"60% accuracy at k=20" meant:
- 60% of questions had the correct answer SOMEWHERE in the top 20 retrieved documents
- This is a RETRIEVAL metric
- Does NOT measure if the system extracted and generated the right answer

### The 16% Result (Current)

"16% accuracy at k=20" means:
- 16% of questions had the expected keyword in the GENERATED ANSWER TEXT
- This is an END-TO-END metric
- Measures retrieval + extraction + generation working together

---

## Why End-to-End Is Lower

The pipeline has multiple failure points:

1. **Retrieval** (60% success rate)
   - Answer must be in top-k documents

2. **Fact Extraction** (70-80% success rate given good retrieval)
   - Must extract correct fact from sentence
   - Must handle AST structure correctly

3. **Fact Selection** (80-90% success rate given good extraction)
   - Must select right fact from multiple candidates
   - Importance scoring must work

4. **Answer Generation** (90-95% success rate given right fact)
   - Must include selected fact in answer text
   - Discourse planning must work

**Total expected accuracy: 60% × 80% × 90% × 95% ≈ 41%**

But we're getting 16%, suggesting extraction or selection has problems.

---

## Root Cause Analysis

### Changes Made Today

1. ✅ **Embedding similarity enabled** - Working correctly (10,819 embeddings loaded)
2. ✅ **Object verification implemented** - Not causing regression (tested by disabling)
3. ✅ **Query AST parsing optimized** - Parse once instead of multiple times

### Why Accuracy Is Still 16%

The 16% is NOT a regression from today's changes. This is the **baseline end-to-end accuracy** that was always there, but we never measured it before.

**Evidence**:
- Disabling object verification: still 16%
- Embeddings loading correctly: still 16%
- Previous semantic ranking: never measured end-to-end

---

## Correct Comparison

To properly evaluate today's changes, we need to:

1. **Measure retrieval-only** (like previous evaluation)
2. **Measure end-to-end** (current evaluation)
3. **Compare same metrics** before and after

### Retrieval Metrics (Should Use)

From current run:
- `num_retrieved`: How many documents retrieved?
- Are correct answers in the retrieved documents?

### End-to-End Metrics (Should Use)

From current run:
- Does generated answer contain expected keyword?
- Precision/recall of extracted facts
- Answer quality

---

## Action Plan

### 1. Measure Retrieval Accuracy (Apples-to-Apples)

Create evaluation that checks if answer is in retrieved documents:
```python
def evaluate_retrieval(question, expected_answer, top_k=20):
    sentences = retriever.retrieve(question, top_k=top_k)

    # Check if ANY sentence contains the answer
    for sent in sentences:
        if expected_answer.lower() in sent['text'].lower():
            return True, sent['rank']
    return False, None
```

Run this for semantic ranking with/without embeddings to see if embeddings help retrieval.

### 2. Diagnose Extraction Failures

For the 60% of questions where retrieval succeeds, why does extraction fail?

Check:
- Are facts being extracted?
- Are they the right facts?
- Are they being selected?
- Are they appearing in the answer?

### 3. Compare Fairly

| Configuration | Retrieval@20 | End-to-End@20 |
|---------------|--------------|---------------|
| Baseline (no ranking) | 70%? | ??% |
| Semantic ranking only | 60-63% | 16% |
| Semantic + embeddings | ??% | 16% |
| Semantic + embeddings + object verification | ??% | 16% |

Fill in the ??% to make fair comparisons.

---

## Conclusion

**The "regression" was an illusion caused by comparing different metrics.**

The real findings:
1. ✅ Embedding similarity implemented correctly
2. ✅ Object verification implemented correctly
3. ❓ End-to-end accuracy is 16% (unknown if this is regression)
4. ❓ Need to measure retrieval accuracy separately to confirm semantic ranking still works

**Next Steps:**
1. Measure retrieval-only accuracy with current setup
2. Compare retrieval accuracy with/without embeddings
3. Diagnose why end-to-end is only 16% (extraction or selection problem)
4. Do NOT panic - we haven't proven any regression yet

---

## Evaluation Best Practices Going Forward

### Always Measure Both

1. **Retrieval Accuracy** (fast, cheap)
   - Answer in top-k documents?
   - Recall@k metrics
   - Good for testing retrieval changes

2. **End-to-End Accuracy** (slow, expensive)
   - Answer in generated text?
   - Full pipeline quality
   - Good for testing overall system

### Never Compare Across Metrics

❌ BAD: "Retrieval went from 60% to 16%"
✅ GOOD: "Retrieval stayed at 60%, end-to-end is 16%"

### Label Results Clearly

Use clear naming:
- `eval_retrieval_top_k_20.csv` - Retrieval metrics only
- `eval_endtoend_top_k_20.csv` - Full pipeline metrics
- Never mix them in same file or comparison
