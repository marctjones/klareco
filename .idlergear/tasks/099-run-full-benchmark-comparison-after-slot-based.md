---
id: 99
title: Run full benchmark comparison after slot-based fixes
state: open
created: '2026-01-06T22:25:42.007112Z'
labels:
- benchmark
- evaluation
priority: high
---
## Task

Run the diagnostic retriever evaluation to measure impact of fixes from #95 and #96:

```bash
./scripts/evaluate_retrievers_diagnostic.sh --fresh
```

## Expected Improvements

1. **ASTAware R@1/R@5/R@10** should improve due to:
   - Slot-based reranking (SUBJ/VERB/OBJ comparison)
   - Keyword prefilter requiring ALL terms
   - Better candidate ranking

2. **Compare to FAISS** which was previously outperforming ASTAware at R@1

## Metrics to Track

| Retriever | R@1 (before) | R@1 (after) | MRR (before) | MRR (after) |
|-----------|--------------|-------------|--------------|-------------|
| ASTAware  | 2            | ?           | 0.165        | ?           |
| FAISS     | 4            | baseline    | 0.200        | baseline    |

## Success Criteria
- ASTAware R@1 >= FAISS R@1 (currently losing 2 vs 4)
- ASTAware MRR >= 0.20
