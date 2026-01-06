---
id: 40
title: 'Phase 1: Find gold standard question-answer pairs in corpus'
state: open
created: '2026-01-05T15:35:04.485753Z'
labels:
- research
- evaluation
priority: low
---
## Objective
For each of the 50 benchmark questions, manually identify sentences in the corpus that contain the correct answer. Create ground truth pairs for evaluation.

## Status: DEPRIORITIZED
**Reason**: We're implementing AST-aware retrieval (Tasks #49-53) first, which should solve the retrieval problem by exploiting structure rather than embeddings. This investigation becomes relevant only if AST-aware retrieval doesn't achieve target accuracy (>60%).

## Original Plan
```bash
# For each question, grep corpus for relevant sentences
# q001: "Kiu fondis Esperanton?" → grep for Zamenhof + Esperanto
# q002: "Kiam aperis la Fundamento?" → grep for Fundamento + 1905
# etc.
```

## Deliverable
Create `benchmark_results/qa/gold_standard_pairs.jsonl` with:
```json
{
  "question_id": "q001",
  "question": "Kiu fondis Esperanton?",
  "relevant_sentences": [
    "ZAMENHOF, Aŭtoro de la lingvo Esperanto",
    "Ludoviko Lazaro Zamenhof fondis Esperanton en 1887"
  ],
  "irrelevant_retrieved": [
    "Ĝi estis fondita la 28-an de majo 2012 per simpla kunveno"
  ]
}
```

## Success Criteria
- All 50 questions have at least 1 relevant sentence identified
- Relevant sentences verified to contain acceptable answers

## When to Revisit
After implementing AST-aware retrieval (Task #53), run benchmark:
- If accuracy >60% → embedding investigation not needed for retrieval
- If accuracy <60% → resume this task to diagnose embedding issues

## Dependencies
None - this is the foundation for all other tests

## Effort
~2 hours (manual corpus search for 50 questions)
