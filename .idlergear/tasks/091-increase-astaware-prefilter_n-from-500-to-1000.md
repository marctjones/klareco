---
id: 91
title: Increase ASTAware prefilter_n from 500 to 1000+ for factual questions
state: open
created: '2026-01-06T22:08:49.089507Z'
labels:
- enhancement
- retrieval
priority: high
---
## Problem

The ASTAware retriever's HNSW prefilter returns only 500 candidates, which may not include the correct answer for factual questions. The correct answer could be ranked #501-1000 in embedding space.

## Evidence

From evaluation results on "Kiu fondis Esperanton?":
- Query roots: `['kiu', 'fond', 'esp']`
- Top 10 results all contain `fond*` + `esperant*` but none mention Zamenhof
- The correct answer may be beyond the 500-candidate cutoff

## Proposed Solution

Increase `prefilter_n` dynamically based on question type:

```python
def search(self, query, top_k=10, strategy='auto', prefilter_n=500):
    # Detect factual questions (Kiu, Kio, Kiam, Kie, etc.)
    classification = self.question_classifier.classify(query, query_ast)
    
    if classification['question_type'] in [QuestionType.WHO, QuestionType.WHAT, 
                                            QuestionType.WHEN, QuestionType.WHERE]:
        prefilter_n = max(prefilter_n, 1000)  # Increase for factual
```

Or make it configurable per-query type.

## Files to Modify

- `klareco/rag/ast_aware_retriever.py`: `search()` method around line 450

## Expected Impact

- Better recall on factual questions (currently 20% on factual category)
- Trade-off: Slightly slower (more candidates to rerank)

## Acceptance Criteria

- [ ] prefilter_n increased to 1000 for factual question types
- [ ] Configurable via parameter or question-type heuristic
- [ ] Re-run evaluation to measure impact
