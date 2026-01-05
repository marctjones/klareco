---
id: 9
title: "Fix FAISS nlist calculation for 4.2M dataset (use 4\xD7\u221AN instead of\
  \ 1\xD7\u221AN)"
state: open
created: '2026-01-04T00:37:08.347850Z'
labels:
- enhancement
- M2
- faiss
priority: high
---
## Problem

Current `FAISSSlotRetriever` uses suboptimal clustering for large datasets:

```python
nlist = int(np.sqrt(embeddings.shape[0]))  # √4.2M = 2,049 clusters
```

**FAISS Guidelines for 1M-10M datasets**:
> "Use IVF65536_HNSW32" or at minimum 4×√N to 16×√N clusters

## Impact

- Current: 2,049 clusters for 4.2M docs
- Recommended: 8,196-32,784 clusters
- **Too few clusters = coarser partitioning = more false negatives**

## Proposed Solution

### Option A: Follow FAISS guidelines exactly (for 1M-10M scale)
```python
if embeddings.shape[0] >= 1_000_000:
    nlist = 65536  # FAISS recommended for this scale
else:
    nlist = int(4 * np.sqrt(embeddings.shape[0]))  # 4×√N minimum
```

### Option B: Conservative improvement (4×√N)
```python
nlist = int(4 * np.sqrt(embeddings.shape[0]))  # 4×√N instead of 1×√N
# For 4.2M: nlist = 8,196 instead of 2,049
```

## Expected Improvement

- **Accuracy**: 85% → 88-90% recall
- **Speed**: Minimal impact (better partitioning may even improve speed)
- **Memory**: Negligible increase

## Files to Change

- `klareco/rag/slot_retriever_faiss.py:115`
- `klareco/rag/slot_retriever_multifaiss.py:139`

## References

- FAISS Guidelines: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- "For 1M-10M vectors: IVF65536_HNSW32"

## Acceptance Criteria

- [ ] nlist calculation updated to 4×√N or 65536 for large datasets
- [ ] Index rebuild script updated
- [ ] Benchmark on full 4.2M index shows improved recall
- [ ] No regression in latency
