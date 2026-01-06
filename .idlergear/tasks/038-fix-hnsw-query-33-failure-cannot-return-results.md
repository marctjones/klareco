---
id: 38
title: 'Fix HNSW query #33 failure: "Cannot return results in contiguous 2D array"'
state: open
created: '2026-01-05T15:22:17.216220Z'
labels:
- bug
- 'priority: low'
---
## Problem
HNSW retriever fails on query #33 with:
```
Cannot return the results in a contiguous 2D array. Probably ef or M is too small
```

## Root Cause
HNSW parameters (M=16, ef_construction=200, ef_search=500) may be too small for the 4.2M document index, causing issues with certain queries.

## Fix Required
- Increase HNSW parameters (ef_search, possibly M)
- Or handle this exception gracefully and return partial results
- Test on query #33 specifically

## Impact
Low - Only affects 1/50 queries, but indicates parameter tuning needed
