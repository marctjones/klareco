---
id: 41
title: 'Test 4: Compute similarity scores for gold standard pairs'
state: closed
created: '2026-01-05T15:35:22.487008Z'
labels:
- research
- evaluation
- blocked
priority: low
---
## Objective
**MOST DIAGNOSTIC TEST** - Determine if embeddings capture ANY semantic meaning by comparing relevant vs irrelevant similarity scores.

## Status: BLOCKED
**Blocked by**: Prioritizing AST-aware retrieval implementation (Tasks #49-53)
**Reason**: If AST-aware retrieval achieves >60% accuracy, this investigation becomes unnecessary for solving the retrieval problem.

## Original Plan
For each question:
1. Compute question embedding
2. Compute embeddings for relevant sentences (from Task #40)
3. Compute embeddings for irrelevant retrieved sentences
4. Compare: `sim(q, relevant)` vs `sim(q, irrelevant)`

## Deliverable
Python script: `scripts/evaluate_embedding_similarity.py`

Output: `benchmark_results/qa/similarity_analysis.json`
```json
{
  "question_id": "q001",
  "sim_relevant_mean": 0.45,
  "sim_relevant_max": 0.52,
  "sim_irrelevant_mean": 0.63,
  "sim_irrelevant_max": 0.68,
  "verdict": "CRITICAL - irrelevant scores higher than relevant"
}
```

## When to Revisit
After AST-aware retrieval (Task #53):
- If accuracy <60% → Unblock and run this test to diagnose if embeddings are the issue
- If accuracy >60% → Keep blocked, embeddings are good enough as fallback

## Dependencies
- **REQUIRES**: Task #40 (gold standard pairs)
- **BLOCKED BY**: Task #53 results needed first

## Effort
~3 hours (script + analysis)
