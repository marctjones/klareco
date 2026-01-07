---
id: 89
title: Fix SlotBasedRetriever O(n) linear scan - add prefilter for 4.4M corpus
state: closed
created: '2026-01-06T19:50:46.766476Z'
labels:
- bug
- retrieval
- performance
priority: high
---
## Problem

`SlotBasedRetriever` in `klareco/rag/slot_retriever.py` does a **linear scan of ALL documents** even when using lazy loading mode. This makes it **unusable for the 4.4M corpus**.

## Code Location

`slot_retriever.py:270-277`:
```python
# Stage 1: Slot-based filtering
num_docs = len(self.doc_offsets) if self.use_lazy_loading else len(self.documents)
logger.info(f"Stage 1: Slot-based filtering ({num_docs:,} docs)")

candidates = []
if self.use_lazy_loading:
    # P0: Lazy loading mode - load each doc on demand
    for doc_id in range(len(self.doc_offsets)):  # 4.4M iterations!
        doc = self._get_document(doc_id)  # Disk I/O for EACH doc
        slot_sim = self.slot_similarity(query_slots, doc['slots_np'], ...)
```

## Impact

- **4.4M disk reads per query** = hours per query
- Memory efficient but **time complexity is O(n)** where n = corpus size
- Effectively unusable for production

## Proposed Fix

Add an embedding-based prefilter (FAISS or HNSW) to narrow candidates before slot matching:

```python
def search(self, query: str, top_k: int = 10, prefilter_n: int = 500):
    # NEW: Stage 0 - Prefilter with FAISS/HNSW
    if self.prefilter_index is not None:
        # Get top-N candidates from embedding search (~5ms)
        candidate_ids = self._prefilter_candidates(query_full_emb, prefilter_n)
    else:
        # Fallback: scan all (slow!)
        candidate_ids = range(len(self.doc_offsets))
    
    # Stage 1: Slot-based filtering (now only N candidates)
    for doc_id in candidate_ids:  # 500 iterations instead of 4.4M
        ...
```

## Options

1. **Add HNSW prefilter** (recommended - already have HNSW index built)
2. **Add FAISS prefilter** (alternative)
3. **Deprecate in favor of FAISSSlotRetriever/HNSWSlotRetriever** (simplest)

## Acceptance Criteria

- [ ] SlotBasedRetriever can search 4.4M corpus in <5 seconds
- [ ] Memory usage stays reasonable (<4GB)
- [ ] Recall remains >85% compared to brute-force

## Workaround

Until fixed, use one of these memory-efficient alternatives:
- `FAISSSlotRetriever` 
- `HNSWSlotRetriever`
- `HybridFAISSMmapRetriever`

## Priority

**P0** - Currently broken for production use
