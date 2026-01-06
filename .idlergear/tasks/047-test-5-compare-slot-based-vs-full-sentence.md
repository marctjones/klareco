---
id: 47
title: 'Test 5: Compare slot-based vs full-sentence similarity'
state: open
created: '2026-01-05T15:37:01.527662Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Determine if the retrieval failure is due to slot-based matching being too strict, or if full sentence embeddings are also failing.

## Approach
For each question and its gold standard relevant sentences:

1. **Method A: Slot-based similarity** (current approach)
   - Extract SUBJ/VERB/OBJ slots from question and document
   - Compute weighted slot similarity
   - Apply feature bonuses

2. **Method B: Full sentence similarity** (simpler baseline)
   - Compute mean of all content word embeddings
   - Direct cosine similarity

Compare which method ranks relevant documents higher.

## Test Cases
```python
q = "Kiu fondis Esperanton?"
relevant = "ZAMENHOF, Aŭtoro de la lingvo Esperanto"
irrelevant = "Tropikaj pluvarbaroj estas biodiversaj"

# Method A: Slot-based
slot_sim_relevant = slot_similarity(q, relevant)     # Current system
slot_sim_irrelevant = slot_similarity(q, irrelevant)

# Method B: Full sentence
full_sim_relevant = cosine(mean_words(q), mean_words(relevant))
full_sim_irrelevant = cosine(mean_words(q), mean_words(irrelevant))
```

## Deliverable
Script: `scripts/compare_similarity_methods.py`

Output for each question:
```json
{
  "question_id": "q001",
  "slot_ranking": "FAIL - irrelevant scored higher",
  "full_ranking": "FAIL - irrelevant scored higher",
  "slot_sim_relevant": 0.42,
  "slot_sim_irrelevant": 0.58,
  "full_sim_relevant": 0.38,
  "full_sim_irrelevant": 0.61,
  "verdict": "Both methods fail - embedding quality issue"
}
```

## Interpretation
| Outcome | Diagnosis | Action |
|---------|-----------|--------|
| Both fail | Embedding quality issue | Fix embeddings (Tasks #42-46) |
| Slot fails, full works | Slot matching too strict | Relax slot weights, add fallback |
| Slot works, full fails | Slot matching helps | Keep slots, fix slot extraction |
| Both work | Retrieval algorithm bug | Debug FAISS/HNSW |

## Decision Point
This test determines whether the fix is:
- **Embedding layer** (both fail) → Priority: Tasks #42-46
- **Retrieval algorithm** (slots too strict) → Priority: Implement fallbacks

## Dependencies
- **REQUIRES**: Task #40 (gold standard pairs)
- **PARALLEL WITH**: Task #41 (Test 4)

## Effort
~2 hours
