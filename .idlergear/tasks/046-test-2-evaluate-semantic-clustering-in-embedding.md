---
id: 46
title: 'Test 2: Evaluate semantic clustering in embedding space'
state: closed
created: '2026-01-05T15:36:43.325053Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Verify that semantically related words cluster together in embedding space (e.g., animals with animals, colors with colors).

## Approach
1. Define semantic categories with known Esperanto words
2. Compute embeddings for all words in each category
3. Measure within-category vs between-category similarity

## Test Categories
```python
animals = ['hundo', 'kato', 'birdo', 'fiŝo', 'ĉevalo', 'bovo']
colors = ['ruĝa', 'blua', 'verda', 'flava', 'nigra', 'blanka']
actions = ['kuri', 'salti', 'manĝi', 'dormi', 'legi', 'skribi']
places = ['domo', 'urbo', 'lando', 'maro', 'montaro', 'arbaro']
time = ['hodiaŭ', 'morgaŭ', 'hieraŭ', 'nun', 'poste', 'antaŭe']
```

## Metrics
```python
# Within-category similarity (should be high)
within_animals = mean([sim(w1, w2) for w1, w2 in pairs(animals)])

# Between-category similarity (should be low)
between_animals_colors = mean([sim(a, c) for a in animals for c in colors])

# Cluster quality
silhouette_score  # Standard clustering metric
```

## Deliverable
Script: `scripts/analyze_semantic_clustering.py`

Output:
```json
{
  "animals": {"within": 0.52, "between": 0.28},
  "colors": {"within": 0.48, "between": 0.31},
  "actions": {"within": 0.44, "between": 0.35},
  "cluster_quality": 0.42,
  "verdict": "WARNING - weak clustering"
}
```

## Interpretation
- **Healthy**: within >0.5, between <0.3, silhouette >0.4
- **Warning**: within 0.3-0.5, between 0.3-0.5, silhouette 0.2-0.4
- **Critical**: within ≈ between, silhouette <0.2 (no semantic structure)

## Decision Point
- If no clustering → global semantic failure, need retraining
- If some clustering → embeddings work for known words, issue may be in unknown/proper nouns

## Dependencies
- **SHOULD RUN AFTER**: Task #41 (Test 4) confirms embedding issue
- **PARALLEL WITH**: Tasks #42, #43, #44, #45

## Effort
~2 hours
