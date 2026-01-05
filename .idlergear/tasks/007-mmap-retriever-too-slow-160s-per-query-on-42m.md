---
id: 7
title: 'mmap retriever too slow: 160s per query on 4.2M docs (2.7min/query)'
state: open
created: '2026-01-03T12:59:21.097297Z'
labels:
- performance
- 'priority: high'
---
## Problem

The mmap retriever is performing brute-force similarity computation over all 4.2M documents for each query, taking ~160 seconds (2.7 minutes) per query.

**Performance data from benchmark:**
```
[1/50] | Latency: 160090.0ms | Recall: 0.000 | Memory: 14714MB
[2/50] | Latency: 167410.1ms | Recall: 0.000 | Memory: 14888MB
[3/50] | Latency: 164435.2ms | Recall: 0.333 | Memory: 14922MB
[4/50] | Latency: 159049.8ms | Recall: 0.250 | Memory: 14888MB
[5/50] | Latency: 162201.8ms | Recall: 0.200 | Memory: 14935MB
[6/50] | Latency: 165428.1ms | Recall: 0.200 | Memory: 14945MB
```

**Impact:**
- Only completed 6/50 queries in 30 minutes (hit timeout)
- Estimated time for 50 queries: **2.5 hours**
- Memory usage: 14-15GB RAM
- This is not viable for production use

## Root Cause

Memory-mapped retriever does:
1. Load all 4.2M × 64d embeddings into mmap'd arrays
2. For each query, compute cosine similarity with ALL 4.2M documents
3. No indexing, no pruning - pure brute force O(N) per query

## Solution Options

1. **Add approximate search** to mmap retriever (e.g., random sampling, early stopping)
2. **Reduce test set** to smaller index for mmap baseline (e.g., 100K docs)
3. **Mark mmap as baseline only** - not for production benchmarks
4. **Skip mmap** in default benchmarks, only run on small indexes

## Recommendation

The mmap retriever is a **baseline reference implementation**, not a production solution. It should only be benchmarked on small indexes (<100K docs) or with reduced query sets (<10 queries).

For 4.2M document benchmarks, use only:
- faiss (FAISS-accelerated)
- multifaiss (multi-index FAISS)
- sqlite (database-backed)

## Location

- Implementation: `klareco/rag/slot_retriever_mmap.py`
- Benchmark script: `scripts/benchmark_slot_retrievers.py`
