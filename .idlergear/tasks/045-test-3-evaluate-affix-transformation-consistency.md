---
id: 45
title: 'Test 3: Evaluate affix transformation consistency'
state: closed
created: '2026-01-05T15:36:26.960472Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Test if affixes (mal-, re-, -ej-, -ist-, etc.) create consistent semantic transformations in the embedding space.

## Approach
Test if affixes produce consistent vector transformations:
```python
# mal- (opposite) should create consistent delta
delta_mal_1 = emb('malbona') - emb('bona')
delta_mal_2 = emb('malgranda') - emb('granda')
delta_mal_3 = emb('malrapida') - emb('rapida')

# Check consistency
sim(delta_mal_1, delta_mal_2)  # Should be >0.7
sim(delta_mal_2, delta_mal_3)  # Should be >0.7
```

## Test Cases
**Prefix mal- (opposite)**:
- bona → malbona
- granda → malgranda  
- rapida → malrapida
- alta → malalta

**Prefix re- (again)**:
- fari → refari
- diri → rediri
- legi → relegi

**Suffix -ej- (place)**:
- lerni → lernejo
- vendi → vendejo
- dormi → dormejo

**Suffix -ist- (profession)**:
- scienci → sciencisto
- art → artisto
- dent → dentisto

## Deliverable
Script: `scripts/analyze_affix_consistency.py`

Metrics per affix:
- Mean pairwise similarity of delta vectors
- Consistency score (>0.7 = good, <0.5 = bad)

## Interpretation
- **Healthy**: Affix deltas have >0.7 similarity (learned consistent transformation)
- **Warning**: 0.5-0.7 similarity (partially learned)
- **Critical**: <0.5 similarity (affixes not learned properly)

## Decision Point
- If affixes inconsistent → affix training failed, need Stage 2 work
- If affixes consistent → composition is working, issue is in roots or proper nouns

## Dependencies
- **SHOULD RUN AFTER**: Task #41 (Test 4) confirms embedding issue
- **PARALLEL WITH**: Tasks #42, #43, #44

## Effort
~3 hours
