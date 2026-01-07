---
id: 44
title: 'Test 9: Evaluate root embedding quality in isolation'
state: closed
created: '2026-01-05T15:36:04.846860Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Test if the root embeddings (Stage 1) are capturing semantic meaning, or if they're the bottleneck.

## Approach
1. Extract root embeddings directly from the model (before composition)
2. Test semantic clustering for known roots
3. Check if semantically similar roots have similar embeddings

## Test Cases
```python
# Animal roots (should cluster)
roots_animals = ['hund', 'kat', 'bird', 'fiŝ']
# Should have high within-group similarity >0.5

# Color roots (should cluster)
roots_colors = ['ruĝ', 'blu', 'verd', 'flav']

# Action roots (should cluster)
roots_actions = ['kur', 'salt', 'manĝ', 'dorm']

# Cross-cluster (should be low <0.3)
sim(root['hund'], root['ruĝ'])
sim(root['kur'], root['blu'])

# Esperanto-related roots
sim(root['esperant'], root['fond'])  # Related to Esperanto creation
sim(root['zamenho'], root['esperant'])  # If 'zamenho' is in vocab
```

## Deliverable
Script: `scripts/analyze_root_embeddings.py`

Metrics:
- Within-cluster mean similarity
- Between-cluster mean similarity  
- Cluster separation score = within / between (should be >2.0)

## Interpretation
| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Within-cluster | >0.5 | 0.3-0.5 | <0.3 |
| Between-cluster | <0.3 | 0.3-0.5 | >0.5 |
| Separation ratio | >2.0 | 1.5-2.0 | <1.5 |

## Decision Point
- If roots are bad (separation <1.5) → **root embeddings are the bottleneck**, need Stage 1 retraining
- If roots are good → issue is in composition or affixes

## Dependencies
- **SHOULD RUN AFTER**: Task #41 (Test 4) confirms embedding issue
- **PARALLEL WITH**: Tasks #42, #43

## Effort
~2 hours
