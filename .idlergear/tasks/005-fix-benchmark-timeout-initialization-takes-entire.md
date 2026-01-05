---
id: 5
title: 'Fix benchmark timeout: initialization takes entire 10min budget'
state: closed
created: '2026-01-03T06:24:16.770398Z'
labels:
- bug
- 'priority: high'
---
## Problem

The benchmark script has a 10-minute timeout (`timeout 600`), but retriever initialization takes the entire timeout:

- **mmap**: Loading 4.2M documents into RAM took >10 minutes
- **faiss**: Building FAISS index took 6-7 minutes  
- **multifaiss**: Building 3 FAISS indexes took 8-9 minutes
- **sqlite**: Building SQLite database took 7+ minutes

Result: Only sqlite completed actual benchmark queries. The other 3 timed out during initialization.

## Root Cause

`scripts/benchmark_all_retrievers.sh:204`:
```bash
timeout 600 python scripts/benchmark_slot_retrievers.py ...
```

The 600s timeout applies to the ENTIRE script (initialization + queries), not just the queries.

## Solution Options

1. **Increase timeout** to 20-30 minutes to accommodate initialization
2. **Separate initialization** from benchmark queries (init once, reuse)
3. **Remove timeout** and rely on checkpoint/resume
4. **Add initialization timeout separately** from query timeout

## Evidence

```
mmap log: Stopped at "Processed 2,670,000 documents" during init
faiss log: Stopped after "✓ Indexer loaded", never reached "Starting benchmark run"
multifaiss log: Same - initialization only
sqlite log: Completed because init was faster (7min) + queries (7s) < 10min
```

## Impact

- Benchmark results are incomplete (only 1/4 retrievers)
- Combined results missing mmap, faiss, multifaiss data
- Can't compare retriever performance properly
