---
id: 35
title: Fix MemoryMapped retriever shape mismatch bug causing 100% failure
state: closed
created: '2026-01-05T15:22:16.649473Z'
labels:
- bug
- 'priority: high'
---
## Problem
MemoryMapped retriever fails on ALL queries with:
```
boolean index did not match indexed array along axis 0; size of axis is 1000 but size of corresponding boolean axis is 277
```

All 50 benchmark queries returned ZERO results (0% accuracy).

## Root Cause
Shape mismatch in reranking logic in `klareco/rag/slot_retriever_mmap.py`. The boolean mask has different size than the array being indexed.

## Expected Behavior
Should return top-10 results per query like other retrievers.

## Fix Required
- Debug the reranking code
- Fix array shape mismatch
- Ensure boolean indexing works correctly
- Test on benchmark queries

## Impact
High - Blocks use of MemoryMapped retriever entirely
