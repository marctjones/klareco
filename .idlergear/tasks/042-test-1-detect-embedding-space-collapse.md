---
id: 42
title: 'Test 1: Detect embedding space collapse'
state: open
created: '2026-01-05T15:35:34.571413Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
Determine if embeddings have collapsed (all vectors too similar) by analyzing pairwise similarity distribution.

## Approach
1. Sample 10,000 random sentence embeddings from corpus
2. Compute all pairwise cosine similarities (10K × 10K)
3. Analyze distribution: mean, std, histogram

## Deliverable
Script: `scripts/analyze_embedding_collapse.py`

Output metrics:
```python
mean_similarity = 0.62  # Should be 0.0-0.2
std_similarity = 0.08   # Should be >0.15
percentiles = [0.25, 0.50, 0.75]
```

## Interpretation
| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Mean | 0.0-0.2 | 0.2-0.4 | >0.4 |
| Std | >0.15 | 0.1-0.15 | <0.1 |

**Critical indicators**:
- Mean > 0.4 → global collapse
- Std < 0.1 → no discrimination

## Decision Point
- If collapsed → need to retrain with contrastive loss, hard negatives
- If healthy → issue is in specific domains (proper nouns, etc.)

## Dependencies
- **SHOULD RUN AFTER**: Task #41 (Test 4) to confirm embeddings are the issue

## Effort
~2 hours (sampling + compute)
