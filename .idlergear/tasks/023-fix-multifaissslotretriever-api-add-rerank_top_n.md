---
id: 23
title: 'Fix MultiFAISSSlotRetriever API: add rerank_top_n parameter'
state: open
created: '2026-01-05T00:13:48.941453Z'
labels:
- bug
- api
priority: high
---
**Problem**: `MultiFAISSSlotRetriever.search()` doesn't accept `rerank_top_n` parameter, causing TypeError in demo script.

**Error**:
```
TypeError: MultiFAISSSlotRetriever.search() got an unexpected keyword argument 'rerank_top_n'
```

**Other retrievers that DO accept it**:
- FAISSSlotRetriever
- HNSWSlotRetriever
- MemoryMappedSlotRetriever
- HybridFAISSMmapRetriever
- ScaNNSlotRetriever
- SQLiteSlotRetriever

**Fix needed**: Add `rerank_top_n` parameter to `MultiFAISSSlotRetriever.search()` method signature for API consistency.

**File**: `klareco/rag/slot_retriever_multifaiss.py`

**Priority**: High (P1) - blocks demo script
**Labels**: bug, api
