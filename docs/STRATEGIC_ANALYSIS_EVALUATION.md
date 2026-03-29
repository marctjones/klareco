# Strategic Analysis: Improving Extractive QA Performance

## Executive Summary

**Current Performance: 18/50 (36%)**

Based on comprehensive evaluation metrics, **reranking quality is the root cause** of most failures. Fixing reranking would have the largest cascade effect and could improve accuracy to **~50-60%** (27-30/50 correct).

## Root Cause Analysis

### Failure Mode Breakdown (32 failed questions)

| Failure Type | Count | % of Failures | Root Cause |
|--------------|-------|---------------|------------|
| **Answer ranked 6-20** | 16 | 50% | **Ranking problem** |
| **Answer in top 5, still failed** | 9 | 28% | Extraction/generation problem |
| **Answer not retrieved** | 7 | 22% | Retrieval/query expansion problem |

**Key Finding**: Ranking is the bottleneck for 50% of failures.

### Question Type Impact Analysis

| Type | Accuracy | Impact Score | Avg Rank | MRR | Root Cause |
|------|----------|--------------|----------|-----|------------|
| **WHO** | 10% | **9.0** | 11.3 | 0.302 | Answer buried (rank 11) |
| **WHEN** | 10% | **9.0** | 42.6 | **0.084** | Very poor ranking |
| **WHAT** | 30% | **7.0** | 19.3 | 0.337 | Answer deep (rank 19) |
| WHERE | 60% | 4.0 | 10.7 | 0.466 | Moderate ranking issue |
| WHY | 0% | 2.0 | 20.0 | **0.051** | Worst ranking |
| HOW | 100% | 0.0 | 45.5 | 0.506 | ✓ Working |
| HOW_MANY | 100% | 0.0 | 20.4 | **0.802** | ✓ Working |

**Impact Score** = (100 - accuracy) × num_questions / 100

**Key Findings**:
1. **WHEN questions: MRR 0.084** - Temporal queries have terrible ranking
2. **WHY questions: MRR 0.051** - Causal queries have worst ranking
3. **WHO questions: Average rank 11.3** - Person queries buried too deep
4. **WHAT questions: Average rank 19.3** - Definition queries very deep

### M1 Plausibility Filter Analysis

**Finding**: M1 is aggressive but **not the bottleneck**.

- M1 removes >90% of facts on 41/50 questions (82%)
- Of these 41 questions with aggressive M1:
  - 18 correct (44% accuracy)
  - 23 failed (56% failure rate)
- **Conclusion**: Aggressive M1 filtering is working - success rate similar to overall rate

**Why M1 isn't the problem**:
- Questions succeed even when M1 removes 97% of facts (see Q3, Q12, Q13)
- M1 is doing its job: filtering noise while keeping signal
- Real problem: Not enough good facts being extracted in the first place

## Recommended Improvements (Priority Order)

### Priority 1: Fix Reranking (Expected +14% accuracy)

**Why highest impact**:
- Affects 16/32 failures (50% of failed questions)
- Fixing WHO (rank 11→3) + WHAT (rank 19→5) + WHEN (rank 43→5) = 20 questions
- Expected improvement: 10-14 additional correct answers
- **New accuracy: 50-60%** (28-32/50)

**What to fix**:
```
Problem: Answer ranked too low (MRR 0.342)
Cause: Neural reranker not trained on question-type patterns

Fix Options:
1. Add question-type features to reranker (WHO/WHAT/WHEN/WHERE/WHY/HOW)
2. Train reranker on question-answer pairs (not just relevance)
3. Add semantic similarity between query and sentence roots
4. Boost sentences containing temporal markers for WHEN questions
5. Boost sentences containing causal markers (pro, ĉar, kaŭze) for WHY questions
```

**Cascade Effects**:
- ✅ **Positive**: Better ranking → Extraction sees better sentences → M1 filters less aggressively → More good facts → Better generation
- ✅ **Reveals**: If accuracy doesn't improve, extraction patterns are broken
- ⚠️ **Watch**: Extraction might be accidentally working on badly-ranked sentences; fixing ranking might reveal extraction only works on specific structures

**How to test**:
1. Manually inspect Q1, Q2, Q4 (WHO questions where answer is rank 1-2 but still fail)
2. If these succeed after improving extraction, ranking wasn't the problem
3. If these still fail, extraction patterns are broken

### Priority 2: Add Question-Type Specific Extraction (Expected +6% accuracy)

**Why second priority**:
- Affects 9/32 failures where answer is in top 5 but extraction fails
- WHEN (0.084 MRR) and WHY (0.051 MRR) need special temporal/causal patterns
- Expected improvement: 3-4 additional correct answers (after fixing ranking)
- **New accuracy: 56-64%** (32-36/50 after Priority 1)

**What to fix**:
```
Problem: Generic fact extraction doesn't handle temporal/causal patterns
Current: UnifiedASTExtractor uses generic SVO patterns

Add specialized extractors:
1. WHEN questions: Extract temporal phrases (en YEAR, post EVENT, dum TIME)
2. WHY questions: Extract causal subclauses (pro ke, ĉar, kaŭze de)
3. WHO questions: Boost proper name entities (capitalized words, -ist/-ul suffixes)
```

**Cascade Effects**:
- ✅ **Positive**: No negative cascade - adds new capabilities
- ✅ **Additive**: Works with improved ranking (Priority 1)
- ⚠️ **Watch**: Might extract temporal phrases but still generate poor answers if discourse planning is broken

### Priority 3: Improve Query Expansion (Expected +4% accuracy)

**Why third priority**:
- Affects 7/32 failures where retrieval doesn't find answer
- Recall@20 is 62% - need to get to 75-80%
- Expected improvement: 2-3 additional correct answers
- **New accuracy: 60-68%** (34-39/50 after Priority 1+2)

**What to fix**:
```
Problem: Query expansion missing temporal/causal/person synonyms
Current: Verb synonym expansion works (fond ≈ kre ≈ establ)

Add:
1. Temporal expansion: YEAR → (en YEAR, jaro YEAR, dum YEAR)
2. Person expansion: "kiu" + verb → (person_name, NOUN-ist, NOUN-ul)
3. Causal expansion: "kial" → (pro, ĉar, kaŭze, rezulte)
4. Definition expansion: "kio estas X" → (X estas NOUN, X signifas, X, definition patterns)
```

**Cascade Effects**:
- ✅ **Positive**: More questions get answer in retrieved set
- ⚠️ **Negative**: Over-expansion could add noise (retrieves more but ranks worse)
- ⚠️ **Watch**: If expansion retrieves many sentences, reranking must handle larger candidate set

### Priority 4 (OPTIONAL): Relax M1 Filter (Expected +2% accuracy)

**Why lowest priority**:
- M1 is working reasonably well (44% accuracy even with aggressive filtering)
- Only affects ~5 questions where M1 might be over-filtering
- Expected improvement: 1-2 additional correct answers
- **Risk**: Could add noise and reduce accuracy

**What to test**:
```bash
# Run evaluation without M1
python scripts/evaluate_pipeline_comprehensive.py --no-m1 --output results/no_m1.json
python scripts/analyze_evaluation_results.py results/baseline_unified_extractor.json results/no_m1.json
```

**Cascade Effects**:
- ⚠️ **Uncertain**: Could improve or worsen accuracy
- ⚠️ **Reveals**: If accuracy improves, M1 is over-trained on different distribution
- ⚠️ **Reveals**: If accuracy worsens, M1 is correctly filtering noise

## Cascade Effect Analysis

### Scenario 1: Fix Reranking First (Recommended)

```
Stage 1: Improve reranking (WHO rank 11→3, WHAT rank 19→5)
  ↓
Stage 2: Extraction sees better sentences in top 5
  ↓
Stage 3: More correct facts extracted
  ↓
Stage 4: M1 filter works better (more signal, same noise)
  ↓
Stage 5: Importance scoring ranks correct facts higher
  ↓
Stage 6: Discourse planning has better material
  ↓
Result: +14% accuracy (18/50 → 28/50)
```

**Potential hidden problem revealed**:
- If accuracy doesn't improve, extraction is fundamentally broken
- Might discover extraction only works on specific sentence structures
- Might need Priority 2 (question-type extraction) more urgently than expected

### Scenario 2: Add Question-Type Extraction First (Riskier)

```
Stage 1: Add WHEN/WHY/WHO patterns
  ↓
Stage 2: Extract temporal/causal/person facts
  ↓
Stage 3: BUT: Facts might be ranked low (reranking still broken)
  ↓
Stage 4: M1 might filter out new facts (trained on different patterns)
  ↓
Result: +2% accuracy (18/50 → 20/50) - Limited improvement
```

**Why risky**:
- Extraction won't help if answer is ranked #15
- Might add extraction patterns that M1 filters out as "implausible"
- Fixes downstream problem without fixing upstream bottleneck

### Scenario 3: Improve Query Expansion First (Least Effective)

```
Stage 1: Expand queries (temporal/causal/person synonyms)
  ↓
Stage 2: Retrieve more sentences (recall 62%→75%)
  ↓
Stage 3: BUT: More sentences ranked poorly (MRR stays low)
  ↓
Stage 4: Answer still buried at rank #15-20
  ↓
Result: +4% accuracy (18/50 → 22/50) - Modest improvement
```

**Why least effective**:
- Retrieval already works moderately (62% recall@20)
- Adding more sentences won't help if ranking is broken
- Could actually worsen MRR (more noise in candidate set)

## Testing Plan: Controlled Experiments

### Experiment 1: Verify Reranking Hypothesis

**Goal**: Confirm ranking is the root cause.

```bash
# Manually inspect failed questions with good rank
python -c "
import json
data = json.load(open('results/baseline_unified_extractor.json'))
for r in data['results']:
    if not r['success'] and r['retrieval']['answer_rank'] and r['retrieval']['answer_rank'] <= 5:
        print(f\"Q{r['question_id']} ({r['question_type']}): Rank {r['retrieval']['answer_rank']}\")
        print(f\"  Question: {r['question_text']}\")
        print(f\"  Expected: {r['expected_keywords']}\")
        print(f\"  Answer: {r['answer_text'][:100]}...\")
        print()
"
```

**Expected**: If these questions have answer at rank 1-2 but still fail, extraction is broken (not ranking).

### Experiment 2: Test M1 Hypothesis

```bash
# Run without M1 filter
python scripts/evaluate_pipeline_comprehensive.py --no-m1 --output results/no_m1.json

# Compare
python scripts/analyze_evaluation_results.py results/baseline_unified_extractor.json results/no_m1.json
```

**Expected outcomes**:
- If accuracy improves: M1 is the problem (contradicts our hypothesis)
- If accuracy stays same: M1 is neutral (supports our hypothesis)
- If accuracy worsens: M1 is correctly filtering noise (strongly supports our hypothesis)

### Experiment 3: Test Reranking Fix

**Goal**: Improve reranking for WHO/WHEN/WHAT questions.

```python
# Add question-type boosting to reranker
# In RelevanceScorer.forward():
if question_type == QuestionType.WHO:
    # Boost sentences with proper names
    boost_mask = sentence_has_proper_name(sentence_tokens)
    scores[boost_mask] += 0.5

if question_type == QuestionType.WHEN:
    # Boost sentences with temporal markers
    temporal_patterns = ['en', 'jaro', 'dum', 'post', 'antaŭ']
    boost_mask = sentence_has_temporal(sentence_tokens, temporal_patterns)
    scores[boost_mask] += 0.5
```

**Measure**:
- MRR improvement (target: 0.342 → 0.500)
- Accuracy improvement (target: 36% → 50%)

## Summary: Recommended Action Plan

**Phase 1 (Highest Impact): Fix Reranking**
- Expected: +14% accuracy (18→28/50)
- Effort: Medium (add question-type features to neural reranker)
- Risk: Low (won't hurt anything)
- Time: 1-2 days

**Phase 2 (After Phase 1): Add Question-Type Extraction**
- Expected: +6% accuracy (28→32/50)
- Effort: High (implement temporal/causal/person patterns)
- Risk: Low (additive, no negative cascade)
- Time: 2-3 days

**Phase 3 (After Phase 2): Improve Query Expansion**
- Expected: +4% accuracy (32→34/50)
- Effort: Low (add synonym dictionaries)
- Risk: Medium (might add noise)
- Time: 1 day

**Total Expected Improvement: 36% → 54-60%** (18/50 → 27-30/50)

## Key Insight: Cascade Effects

The **single highest-impact change is fixing reranking** because:

1. **Direct effect**: 16 questions where answer is ranked too low
2. **Cascade effect**: Better ranking → easier extraction → better M1 filtering → better generation
3. **Diagnostic effect**: If accuracy doesn't improve, we know extraction is broken
4. **No negative cascade**: Improving ranking can't hurt downstream stages

**M1 filter is not the problem** - it's aggressive but effective. Relaxing it would likely add noise without improving accuracy.

**Question-type extraction should come second** - it's high-effort but necessary for WHEN/WHY questions. However, it won't help if answers are ranked at position #15.

**Query expansion should come last** - retrieval is already working moderately well (62% recall). Focus on using retrieved results better before retrieving more.
