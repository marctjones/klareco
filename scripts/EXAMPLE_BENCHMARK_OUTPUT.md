# Example Benchmark Output

This shows what you'll see when running the safe benchmark.

## Initial Banner

```
╔════════════════════════════════════════════════════════════════════╗
║            🛡️  SAFE SLOT RETRIEVER BENCHMARK 🛡️                    ║
╚════════════════════════════════════════════════════════════════════╝

This benchmark tests 4 MEMORY-EFFICIENT retrievers:

  1️⃣  mmap       - Memory-mapped retriever
                  Memory: ~500MB  | Speed: Medium

  2️⃣  faiss      - FAISS-accelerated retrieval
                  Memory: ~3-5GB  | Speed: Fast

  3️⃣  multifaiss - Multi-FAISS (separate index per slot)
                  Memory: ~4-6GB  | Speed: Fastest

  4️⃣  sqlite     - SQLite database backend
                  Memory: ~1-2GB  | Speed: Medium

────────────────────────────────────────────────────────────────────
⚠️  BASELINE RETRIEVER IS SKIPPED
────────────────────────────────────────────────────────────────────
The baseline loads the entire index into RAM (~30GB for 4.2M docs)
and will freeze most systems. It is NOT included in this safe run.

════════════════════════════════════════════════════════════════════


╔════════════════════════════════════════════════════════════════════╗
║          SLOT RETRIEVER BENCHMARK SUITE                           ║
╚════════════════════════════════════════════════════════════════════╝

Configuration:
  Index:        data/indexes/slot_full
  Output:       benchmark_results
  Queries:      50
  Timestamp:    20260103_014523

Checking index...
  ✓ Found 4229277 documents (19G)
```

## Step 1: Query Creation

```
════════════════════════════════════════════════════════════════════
STEP 1: Creating Benchmark Queries
════════════════════════════════════════════════════════════════════

Found existing queries at data/indexes/slot_full/benchmark_queries.jsonl
  ✓ Query count matches (50 queries)
  → Reusing existing queries

✓ Queries ready: benchmark_results/queries_20260103_014523.jsonl
```

## Step 2: Retriever Benchmarks

```
════════════════════════════════════════════════════════════════════
STEP 2: Benchmarking Retrievers
════════════════════════════════════════════════════════════════════

Testing 4 retrievers: mmap faiss multifaiss sqlite
Expected total time: ~60-100 minutes


────────────────────────────────────────────────────────────────────
[1/4] Testing: mmap
────────────────────────────────────────────────────────────────────

Starting mmap benchmark at 01:45:30...

============================================================
Loading benchmark queries
============================================================
✓ Loaded 50 queries from benchmark_results/queries_20260103_014523.jsonl

============================================================
Initializing indexer (loading embedding models)
============================================================
  Root model: models/root_embeddings/best_model.pt
  Affix model: models/affix_transforms_v2/best_model.pt
✓ Indexer loaded in 2.3s

============================================================
Benchmarking: mmap
============================================================
  Initializing mmap retriever...
  ✓ Retriever initialized in 8.4s

  Starting benchmark run (50 queries)...

  [5/50] Latency: 245.3ms | Recall: 0.920 | Memory: 512MB | ETA: 3m 45s
  [10/50] Latency: 238.7ms | Recall: 0.920 | Memory: 513MB | ETA: 3m 12s | 💾 checkpoint
  [15/50] Latency: 235.2ms | Recall: 0.920 | Memory: 515MB | ETA: 2m 40s
  [20/50] Latency: 233.8ms | Recall: 0.920 | Memory: 517MB | ETA: 2m 05s | 💾 checkpoint
  [25/50] Latency: 232.9ms | Recall: 0.920 | Memory: 518MB | ETA: 1m 32s
  [30/50] Latency: 231.2ms | Recall: 0.920 | Memory: 520MB | ETA: 1m 00s | 💾 checkpoint
  [35/50] Latency: 230.8ms | Recall: 0.920 | Memory: 521MB | ETA: 0m 35s
  [40/50] Latency: 229.9ms | Recall: 0.920 | Memory: 522MB | ETA: 0m 23s | 💾 checkpoint
  [45/50] Latency: 229.2ms | Recall: 0.920 | Memory: 523MB | ETA: 0m 11s
  [50/50] Latency: 228.8ms | Recall: 0.920 | Memory: 524MB | ETA: 0s | 💾 checkpoint

  ✓ Benchmark completed in 189.2s

============================================================
RESULTS SUMMARY: mmap
============================================================

Latency Metrics:
  Mean:       234.56 ms
  Median:     218.34 ms
  P95:        456.78 ms
  P99:        512.89 ms
  Range:       178.23 - 589.12 ms

Memory Usage:
  Peak:       524.8 MB
  Delta:       12.3 MB (increase from baseline)

Accuracy Metrics:
  Recall@10:   0.920 (92.0%)
  MRR:         0.854
  NDCG@10:     0.887

============================================================

✓ mmap completed successfully in 3m 9s
  Results: benchmark_results/mmap_20260103_014523.json
  Log:     benchmark_results/mmap_20260103_014523.log


────────────────────────────────────────────────────────────────────
[2/4] Testing: faiss
────────────────────────────────────────────────────────────────────

Starting faiss benchmark at 01:48:39...

============================================================
Initializing indexer (loading embedding models)
============================================================
  Root model: models/root_embeddings/best_model.pt
  Affix model: models/affix_transforms_v2/best_model.pt
✓ Indexer loaded in 2.1s

============================================================
Benchmarking: faiss
============================================================
  Initializing faiss retriever...
  Building FAISS indexes for slots...
    Building SUBJ index (1,234,567 vectors)...
    Building VERB index (892,345 vectors)...
    Building OBJ index (1,102,365 vectors)...
  ✓ Retriever initialized in 45.2s

  Starting benchmark run (50 queries)...

  [5/50] Latency: 125.3ms | Recall: 0.920 | Memory: 3245MB | ETA: 1m 55s
  [7/50] Latency: 132.1ms | Recall: 0.918 | Memory: 3248MB | ETA: 1m 48s | ⚠ slow query: 287ms
  [10/50] Latency: 118.7ms | Recall: 0.920 | Memory: 3251MB | ETA: 1m 35s | 💾 checkpoint
  [15/50] Latency: 115.2ms | Recall: 0.922 | Memory: 3255MB | ETA: 1m 18s
  [20/50] Latency: 116.8ms | Recall: 0.921 | Memory: 3258MB | ETA: 0m 58s | 💾 checkpoint
  [25/50] Latency: 114.9ms | Recall: 0.923 | Memory: 3260MB | ETA: 0m 45s
  [30/50] Latency: 113.2ms | Recall: 0.924 | Memory: 3262MB | ETA: 0m 37s | 💾 checkpoint ⏰ 1min
  [35/50] Latency: 112.8ms | Recall: 0.924 | Memory: 3264MB | ETA: 0m 28s
  [40/50] Latency: 111.9ms | Recall: 0.925 | Memory: 3265MB | ETA: 0m 18s | 💾 checkpoint
  [45/50] Latency: 111.2ms | Recall: 0.925 | Memory: 3267MB | ETA: 0m 09s
  [50/50] Latency: 110.8ms | Recall: 0.926 | Memory: 3268MB | ETA: 0s | 💾 checkpoint

  ✓ Benchmark completed in 156.8s

============================================================
RESULTS SUMMARY: faiss
============================================================

Latency Metrics:
  Mean:       125.34 ms
  Median:     118.50 ms
  P95:        245.67 ms
  P99:        312.89 ms
  Range:        78.23 - 456.12 ms

Memory Usage:
  Peak:      3268.8 MB
  Delta:     2756.3 MB (increase from baseline)

Accuracy Metrics:
  Recall@10:   0.926 (92.6%)
  MRR:         0.862
  NDCG@10:     0.891

============================================================

✓ faiss completed successfully in 2m 37s
  Results: benchmark_results/faiss_20260103_014523.json
  Log:     benchmark_results/faiss_20260103_014523.log


────────────────────────────────────────────────────────────────────
[3/4] Testing: multifaiss
────────────────────────────────────────────────────────────────────

Starting multifaiss benchmark at 01:51:16...

  [Similar output structure as above...]

✓ multifaiss completed successfully in 1m 45s


────────────────────────────────────────────────────────────────────
[4/4] Testing: sqlite
────────────────────────────────────────────────────────────────────

Starting sqlite benchmark at 01:53:01...

  [Similar output structure as above...]

✓ sqlite completed successfully in 2m 18s
```

## Step 3: Combining Results

```
════════════════════════════════════════════════════════════════════
STEP 3: Combining Results
════════════════════════════════════════════════════════════════════

Saved combined results to: benchmark_results/combined_20260103_014523.json
```

## Step 4: Visualizations

```
════════════════════════════════════════════════════════════════════
STEP 4: Generating Visualizations
════════════════════════════════════════════════════════════════════

✓ HTML report generated: benchmark_results/report_20260103_014523.html
✓ Charts saved to: benchmark_results/charts_20260103_014523/
```

## Step 5: Final Summary

```
════════════════════════════════════════════════════════════════════
FINAL SUMMARY
════════════════════════════════════════════════════════════════════

Solution         Mean (ms)    P95 (ms)     Memory (MB)  Recall@10
────────────────────────────────────────────────────────────────────
mmap             234.56       456.78       524.8        0.920
faiss            125.34       245.67       3268.8       0.926
multifaiss       89.23        178.45       4523.5       0.925
sqlite           178.90       345.12       1534.6       0.922


╔════════════════════════════════════════════════════════════════════╗
║                   ✓ BENCHMARK COMPLETE!                           ║
╚════════════════════════════════════════════════════════════════════╝

Output Files:
  📊 Combined JSON:  benchmark_results/combined_20260103_014523.json
  📈 HTML Report:    benchmark_results/report_20260103_014523.html
  📉 Charts:         benchmark_results/charts_20260103_014523/

View Results:
  firefox benchmark_results/report_20260103_014523.html
  # or
  google-chrome benchmark_results/report_20260103_014523.html

Benchmark completed at 01:55:19
```

## Progress Indicators Reference

| Symbol | Meaning |
|--------|---------|
| ✓ | Success / Completed |
| ⚠ | Warning / Slow query detected |
| 💾 | Checkpoint saved (resume point) |
| ⏰ | Time-based update (1 minute elapsed) |
| 📈 | Memory spike detected |
| ℹ️ | Information |
| ✗ | Error / Failed |

## Timing Breakdown

For 4.2M documents, 50 queries:

| Retriever | Init Time | Query Time | Total |
|-----------|-----------|------------|-------|
| mmap      | ~8s       | ~3m        | ~3m 10s |
| faiss     | ~45s      | ~2m        | ~2m 45s |
| multifaiss| ~60s      | ~1m 15s    | ~2m 15s |
| sqlite    | ~120s     | ~1m 30s    | ~3m 30s |

**Total**: ~11-12 minutes for all 4 retrievers
