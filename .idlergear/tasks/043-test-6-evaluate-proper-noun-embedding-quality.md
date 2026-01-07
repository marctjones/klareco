---
id: 43
title: 'Test 6: Evaluate proper noun embedding quality'
state: closed
created: '2026-01-05T15:35:49.735582Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Determine if unknown proper nouns (Zamenhof, Bjalistoko, etc.) are getting reasonable embeddings or random/collapsed vectors.

## Approach
1. Identify proper nouns from failed questions (Zamenhof, Fundamento, UEA, etc.)
2. Check parser output: are they marked as unknown/nekonata?
3. Compare embeddings:
   - `sim(zamenhof, esperanto)` - should be high (related concepts)
   - `sim(zamenhof, hundo)` - should be low (unrelated)
   - `sim(zamenhof, bjalistoko)` - different proper nouns

## Test Cases
```python
# Known related words
sim(esperanto, lingvo)  # Should be ~0.6

# Unknown proper noun vs semantically related
sim(zamenhof, esperanto)  # What is this?
sim(zamenhof, aŭtoro)     # What is this?

# Unknown proper noun vs unrelated
sim(zamenhof, hundo)      # Should be LOWER than above
sim(zamenhof, arbo)       # Should be LOWER than above

# Two different unknown proper nouns
sim(zamenhof, bjalistoko)  # Should be ~0.3-0.5 (different entities)
```

## Deliverable
Script: `scripts/analyze_proper_noun_embeddings.py`

Output showing whether unknown words:
- Get random embeddings (all similarities ~0.5)
- Get zero/default embeddings (all similarities ~1.0)
- Get reasonable compositional embeddings based on characters

## Interpretation
- **Random**: All similarities ~0.4-0.6 → proper nouns need special handling
- **Collapsed**: All similarities >0.9 → default embedding issue
- **Reasonable**: Related concepts >0.6, unrelated <0.3 → compositional working

## Decision Point
- If random/collapsed → need proper noun embedding strategy (character CNN, cross-lingual, etc.)
- If reasonable → proper nouns aren't the main issue

## Dependencies
- **SHOULD RUN AFTER**: Task #41 (Test 4) to confirm embeddings are the issue

## Effort
~2 hours
