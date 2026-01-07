---
id: 10
title: Add HNSW quantizer to IVF index for faster coarse quantization
state: closed
created: '2026-01-04T00:37:08.958222Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Problem

Current `FAISSSlotRetriever` uses `IndexFlatIP` as the coarse quantizer:

```python
quantizer = faiss.IndexFlatIP(dim)  # Brute-force search
index = faiss.IndexIVFFlat(quantizer, dim, nlist, ...)
```

For 4.2M docs with nlist=65K clusters, coarse quantization does **brute-force search over 65K centroids** on every query.

## FAISS Best Practice

**FAISS Guidelines for 1M-10M datasets**:
> "Use IVF65536_HNSW32" - HNSW quantizer instead of Flat

**Benefits of HNSW quantizer**:
- ✅ O(log N) search instead of O(N) for centroid assignment
- ✅ Faster query time (especially with many clusters)
- ✅ No accuracy loss (still finds correct Voronoi cell)

## Proposed Solution

```python
# Replace IndexFlatIP with IndexHNSWFlat
quantizer = faiss.IndexHNSWFlat(dim, M=32)  # M=32 is FAISS recommended
quantizer.hnsw.efConstruction = 40  # Build-time parameter

index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# At search time, can tune efSearch
quantizer.hnsw.efSearch = 16  # Lower = faster, higher = more accurate
```

## Expected Improvement

- **Coarse quantization speed**: 10-50× faster (especially with 65K clusters)
- **Overall query latency**: 5.1ms → 3-4ms estimated
- **Accuracy**: No change (same IVF+Flat vectors, just faster quantizer)

## Implementation Notes

**Build-time parameters** (set once during index creation):
- `M`: Connectivity (32 is good default)
- `efConstruction`: 40 recommended

**Runtime parameters** (tunable per query):
- `efSearch`: 16 default, increase for better accuracy

## Files to Change

- `klareco/rag/slot_retriever_faiss.py:115-117`
- `klareco/rag/slot_retriever_multifaiss.py:139-141`

## References

- FAISS Guidelines: "IVF65536_HNSW32 for 1M-10M vectors"
- HNSW paper: https://arxiv.org/abs/1603.09320

## Acceptance Criteria

- [ ] HNSW quantizer implemented in FAISSSlotRetriever
- [ ] efSearch parameter exposed for runtime tuning
- [ ] Benchmark shows improved latency (target: <4ms)
- [ ] No regression in recall
