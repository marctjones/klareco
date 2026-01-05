---
id: 8
title: Benchmark needs much longer timeout or skip mmap for large indexes
state: closed
created: '2026-01-03T12:59:42.216830Z'
labels:
- enhancement
- 'priority: high'
---
## Problem

Even with 30-minute timeout, the benchmark cannot complete when mmap retriever is included on 4.2M document index.

**Current situation:**
- mmap: 160s/query × 50 queries = **133 minutes (2.2 hours)** needed
- 30-minute timeout only allows 6 queries to complete
- faiss was terminated after mmap timed out

## Root Cause

The timeout fix (#179) increased from 10min to 30min, but that's still insufficient when mmap is included because mmap does brute-force search.

**Time breakdown for full benchmark with mmap:**
```
mmap init: 11 minutes
mmap queries (50 × 160s): 133 minutes
faiss init: 7 minutes  
faiss queries: 5-10 minutes
multifaiss init: 9 minutes
multifaiss queries: 5-10 minutes
sqlite init: 7 minutes
sqlite queries: 8 seconds

TOTAL: ~180 minutes (3 hours)
```

## Solution Options

### Option 1: Skip mmap for large indexes (RECOMMENDED)
```bash
# In benchmark_all_retrievers.sh
if [ "$INDEX_SIZE" -gt 1000000 ]; then
    SOLUTIONS=("faiss" "multifaiss" "sqlite")
else
    SOLUTIONS=("mmap" "faiss" "multifaiss" "sqlite")
fi
```

### Option 2: Increase timeout to 4 hours
- Set timeout to 14400s (4 hours)
- Only viable for overnight/CI runs
- Still wasteful to wait 2+ hours for mmap

### Option 3: Make mmap optional with flag
```bash
./scripts/benchmark_safe.sh --index slot_full --skip-mmap
```

### Option 4: Separate benchmark suites
- `benchmark_fast.sh`: faiss, multifaiss, sqlite only
- `benchmark_full.sh`: includes mmap baseline (long-running)

## Recommendation

**Use Option 1** - automatically skip mmap for indexes >1M documents. This gives us:
- Fast benchmarks on production indexes (faiss, multifaiss, sqlite)
- Baseline comparison still available on small indexes
- No manual flags needed

## Impact

Without this fix:
- Cannot benchmark 4.2M index with all retrievers
- Wastes hours waiting for mmap brute-force
- Makes iteration on improvements very slow
