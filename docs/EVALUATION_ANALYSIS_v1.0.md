# Partial Evaluation Analysis (22/30 questions)

**Date:** 2026-02-13
**Test Set:** General Knowledge (rag_test_set_general_knowledge.jsonl)
**Pipeline:** retrieval → reranker → extraction (no M1)

## Overall Performance

**Accuracy:** 3/22 correct (13.6%)
**Average Granular Score:** 0.326 / 1.0

### Component Breakdown

| Component | Avg Score | Interpretation |
|-----------|-----------|----------------|
| **Retrieval (R)** | 0.414 | Answer in top-10 for only ~40% of questions |
| **Extraction (E)** | 0.136 | Correct extraction in only 13.6% |
| **Alignment (A)** | 0.673 | Extraction picks reasonable doc ranks |
| **Robustness (B)** | 0.109 | Very few redundant good docs in top-5 |

## Root Cause Analysis

### Primary Bottleneck: RETRIEVAL FAILURE (50%)

**11/22 questions (50%)** had R=0.0 (answer not in top-10 documents)

Examples:
- Q: "Kio estas la plej alta monto?" → Retrieved docs about "montaro" (mountain range), not Everest
- Q: "Kio estas la ĉefurbo de Japanio?" → Retrieved docs mention "ĉefurbo" (capital) but not Tokyo
- Q: "George Washington" → Retrieved Karen Uhlenbeck (mathematician), got "Stanley"

**Pattern:** Queries retrieve semantically related but WRONG documents.

### Secondary Bottleneck: EXTRACTION FAILURE (36%)

**8/22 questions (36%)** had R>0 but E=0.0 (answer retrieved but extracted wrong text)

Examples:
- Q: "Kiu estas la plej granda oceano?" → Answer "Pacifiko" at rank #2, extracted "Hindoceana" instead
- Q: "Kiom da kontinentoj?" → Answer "sep" at rank #8, extracted "623"
- Q: "Kion malkovris Marie Curie?" → Answer "radioaktiveco" at rank #5, extracted "monatojn"

**Pattern:** Even when answer is in documents, extractor picks WRONG span.

## Performance by Category

| Category | Accuracy | Avg R | Notes |
|----------|----------|-------|-------|
| **History** | 50% (2/4) | 0.45 | Best performing |
| **Science** | 25% (1/4) | 0.65 | Good retrieval, poor extraction |
| **Geography** | 0% (0/5) | 0.26 | Terrible retrieval |
| **Animals** | 0% (0/4) | 0.00 | Complete retrieval failure |
| **Culture** | 0% (0/4) | 0.60 | Retrieval OK, extraction fails |
| **Human Body** | 0% (1/1) | 1.00 | Perfect retrieval, wrong extraction |

## Specific Failure Patterns

### 1. Superlative Questions Fail (WHAT/WHO is the BIGGEST/TALLEST/etc.)

❌ "Kio estas la plej alta monto?" → Retrieved "montaro" (wrong)
❌ "Kiu estas la plej granda oceano?" → Rank #2 (not extracted)
❌ "Kio estas la plej granda planedo?" → Retrieved "Sunsistemo" (wrong)
❌ "Kio estas la plej rapida besto?" → TBD

**Why:** Query parser doesn't boost superlative constraints ("plej + adjektivo").

### 2. Proper Names Not Retrieved

❌ "George Washington" → Got "Stanley" (wrong person entirely)
❌ "Tokio" (Tokyo) → Not in top-10
❌ "Everest" → Not in top-10

**Why:** Proper nouns may not be indexed correctly or lack boosting.

### 3. Extraction Picks Generic Terms

❌ Q: "Kio estas la ĉefurbo de Japanio?" → Extracted "ĉefurbo" (the word "capital" from query)
❌ Q: "Kio estas la plej granda planedo?" → Extracted "Sunsistemo" (solar system, too generic)
❌ Q: "Kio estas la simbolo de oksigeno?" → Extracted "O₂" instead of "O"

**Why:** Extractor doesn't filter out query terms or prefer specific answers.

### 4. Number Extraction Completely Wrong

❌ "Kiom da kontinentoj?" → Expected "sep" (7), got "623"
❌ "Kio estas la simbolo de oksigeno?" → Expected "O", got "O₂"

**Why:** No numeric entity recognition or validation.

## Successful Cases (What Worked)

✅ **history_003:** "Kie okazis la Franca Revolucio?" → "Parizo" (R=0.80, E=1.00)
✅ **history_004:** "Kiu inventis la presmaŝinon?" → "Johannes" (R=1.00, E=1.00)
✅ **science_002:** "Kiu inventis la telefonon?" → "Alexander" (R=1.00, E=1.00)

**Common pattern:** WHO questions with famous inventors/people, clear verb "inventis" (invented).

## Recommendations

### Immediate Fixes (v1.0)

1. **Boost superlative queries:** "plej + adjektivo" should heavily weight ranking
2. **Proper noun boosting:** Named entities should get higher retrieval weight
3. **Filter query terms from extraction:** Don't extract words from the question
4. **Numeric validation:** Cross-check numeric answers for plausibility

### Architectural (v2.0)

1. **AST-native storage:** Pre-index superlatives, named entities in graph structure
2. **Entity-centric retrieval:** Query by entity type (PERSON, LOCATION, etc.)
3. **Type-aware extraction:** Know that "Kiom" expects number, "Kiu" expects person
4. **Redundancy checking:** Require multiple docs to confirm answer

## Comparison to Predictions

**Predicted accuracy:** 60-80% (18-24 correct)
**Actual accuracy:** 13.6% (3 correct)
**Gap:** -46 to -66 percentage points

**Why were predictions wrong?**

1. Assumed corpus has good coverage → **FALSE** (missing basic facts like "Tokyo is capital of Japan")
2. Assumed semantic similarity works → **PARTIAL** (works for some, fails for superlatives)
3. Assumed extraction is reliable → **FALSE** (86% extraction failure rate)
4. Underestimated proper noun challenges → **TRUE** (proper nouns not boosted)

## Next Steps

1. Analyze corpus coverage: Does Wikipedia Esperanto have these facts?
2. Inspect failed queries: What docs were actually retrieved?
3. Test superlative boosting: Add "plej" constraint
4. Evaluate entity recognition: Are proper nouns being classified?
