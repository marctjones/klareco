# Benchmark Script Improvements

## Overview

The slot retriever benchmark script has been enhanced with better progress reporting, checkpoint/resume support, and clearer output.

## Key Improvements

### 1. **Progress Tracking** 🎯

**Before:**
```
Processed 10/50 queries
Processed 20/50 queries
```

**After:**
```
[5/50] Latency: 125.3ms | Recall: 0.850 | Memory: 2048MB | ETA: 5m 30s
[7/50] Latency: 132.1ms | Recall: 0.845 | Memory: 2049MB | ETA: 5m 15s | ⚠ slow query: 287ms
[10/50] Latency: 118.7ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s | 💾 checkpoint
[15/50] Latency: 115.2ms | Recall: 0.868 | Memory: 2053MB | ETA: 4m 10s
[22/50] Latency: 119.5ms | Recall: 0.870 | Memory: 2245MB | ETA: 3m 30s | 📈 mem spike: +192MB
[25/50] Latency: 117.8ms | Recall: 0.872 | Memory: 2250MB | ETA: 3m 05s
[30/50] Latency: 116.2ms | Recall: 0.875 | Memory: 2255MB | ETA: 2m 30s | 💾 checkpoint
```

**Shows:**
- Running average of last 5 queries (latency & recall)
- Current memory usage
- **Estimated time remaining (ETA)**

**Updates triggered by:**
- ⏰ **Every 1 minute** (time-based)
- 🔢 **Every 5 queries** (milestone)
- 💾 **Every 10 queries** (checkpoint)
- ⚠️ **Slow queries** (2x slower than average)
- 📈 **Memory spikes** (>100MB increase)
- ✅ **Completion** (final query)

### 2. **Smart Event-Based Updates** 🎯

Progress updates are now **adaptive** rather than fixed intervals:

#### Regular Updates
- ⏰ **Time-based**: At least once per minute (even if few queries completed)
- 🔢 **Milestones**: Every 5 queries
- 💾 **Checkpoints**: Every 10 queries (saves progress)

#### Meaningful Event Alerts
- ⚠️ **Slow Query Alert**: Query took 2x longer than average
  - Example: Average 100ms, query took 250ms → immediate alert
  - Helps identify problematic queries or index issues

- 📈 **Memory Spike Alert**: Memory increased >100MB in single query
  - Example: Memory jumped from 2GB to 2.2GB
  - Helps catch memory leaks or inefficient operations

**Why this matters:**
- Don't miss important events (slow queries, memory issues)
- Still get regular updates even if queries are slow
- See exactly which queries cause problems

### 3. **Checkpoint/Resume Support** 💾

**Automatic checkpointing:**
- Saves progress every 10 queries
- Atomic saves (prevents corruption)
- Auto-resume on restart

**If interrupted:**
```bash
# Just rerun the same command
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full

# Output:
#   Loading checkpoint from benchmark_results/mmap_checkpoint.json
#   Resuming from query 30/50
```

### 3. **Skip Completed Solutions** ⏭️

**Behavior:**
- Checks for existing result files
- Skips solutions that already completed
- Loads results instead of re-running

**Example:**
```
Found existing results for mmap, loading from mmap_results.json
Skipping mmap (already completed)
```

### 4. **Detailed Initialization Logging** 📋

See exactly what's happening during startup:

```
============================================================
Loading benchmark queries
============================================================
✓ Loaded 50 queries from queries_20260103.jsonl

============================================================
Initializing indexer (loading embedding models)
============================================================
  Root model: models/root_embeddings/best_model.pt
  Affix model: models/affix_transforms_v2/best_model.pt
✓ Indexer loaded in 2.3s

============================================================
Benchmarking: faiss
============================================================
  Initializing faiss retriever...
  Building FAISS indexes for slots...
  ✓ Retriever initialized in 45.2s

  Starting benchmark run (50 queries)...
```

**Benefits:**
- Know what stage is running (not just hanging)
- See initialization times
- Identify bottlenecks early

### 5. **Memory-Safe by Default** 🛡️

**Baseline retriever now SKIPPED by default** (prevents system freeze)

**Use `--include-baseline` only for small indexes:**
```bash
# Safe for small indexes (<100K docs)
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_test \
  --include-baseline
```

### 5. **Detailed Output During Execution** 📊

#### Query Creation
```
============================================================
Step 1: Creating benchmark queries
============================================================
  Loading index documents...
    Loaded 500,000 documents...
    Loaded 1,000,000 documents...
    ...
  Loaded 4,229,277 documents from index
  Sampling 50 queries...
  Writing queries to benchmark_results/queries_20260103.jsonl...
  ✓ Saved 50 queries
```

#### Indexer Loading
```
============================================================
Initializing indexer (loading embedding models)
============================================================
  Root model: models/root_embeddings/best_model.pt
  Affix model: models/affix_transforms_v2/best_model.pt
✓ Indexer loaded in 2.3s
```

#### Retriever Initialization
```
============================================================
Benchmarking: faiss
============================================================
  Initializing faiss retriever...
  ✓ Retriever initialized in 45.2s

  Starting benchmark run (50 queries)...

  [5/50] Latency: 125.3ms | Recall: 0.850 | Memory: 2048MB | ETA: 5m 30s
  [10/50] Latency: 118.7ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s
  ...
```

#### Per-Solution Summary
```
============================================================
RESULTS SUMMARY: faiss
============================================================

Latency Metrics:
  Mean:       125.34 ms
  Median:     118.50 ms
  P95:        245.67 ms
  P99:        312.89 ms
  Range:       78.23 - 456.12 ms

Memory Usage:
  Peak:      3245.8 MB
  Delta:     2987.3 MB (increase from baseline)

Accuracy Metrics:
  Recall@10:   0.920 (92.0%)
  MRR:         0.854
  NDCG@10:     0.887

============================================================

  ✓ Benchmark completed in 156.8s
```

### 6. **Final Comparison with Recommendations** 🏆

```
====================================================================================================
FINAL BENCHMARK COMPARISON
====================================================================================================

Index: data/indexes/slot_full
Queries: 50

Solution         Mean (ms)    Median (ms)  P95 (ms)     Memory (MB)  Recall@10
----------------------------------------------------------------------------------------------------
mmap             234.56       218.34       456.78       512.3        0.920
faiss            125.34       118.50       245.67       2987.3       0.920
multifaiss       89.23        85.12        178.45       4123.5       0.920
sqlite           178.90       165.23       345.12       1234.6       0.920

====================================================================================================

RECOMMENDATIONS:

  🚀 FASTEST:       multifaiss (89.2ms mean latency)
  🎯 MOST ACCURATE:  faiss (92.0% recall)
  💾 LOWEST MEMORY:  mmap (512MB)

====================================================================================================
```

## Usage Examples

### Standard run (safe, skips baseline)
```bash
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full
```

### Include baseline (WARNING: needs ~30GB RAM)
```bash
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_test \
  --include-baseline
```

### Custom number of queries
```bash
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_full \
  --num-queries 100
```

### Resume after interruption
```bash
# Same command - auto-detects checkpoints and completed solutions
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full
```

## Files Created

### During Benchmark
- `benchmark_results/queries_TIMESTAMP.jsonl` - Test queries
- `benchmark_results/*_checkpoint.json` - Progress checkpoints (every 10 queries)
- `benchmark_results/*_results.json` - Completed solution results

### After Completion
- `benchmark_results/combined_TIMESTAMP.json` - All results combined
- `benchmark_results/report_TIMESTAMP.html` - Interactive HTML report

## Performance Expectations

For 4.2M documents, 50 queries:

| Retriever   | Init Time | Query Time | Total Time | Memory Peak |
|-------------|-----------|------------|------------|-------------|
| Mmap        | ~30s      | ~15-20 min | ~20-25 min | ~500MB      |
| FAISS       | ~45s      | ~8-12 min  | ~10-15 min | ~3-5GB      |
| MultiFAISS  | ~60s      | ~5-8 min   | ~6-10 min  | ~4-6GB      |
| SQLite      | ~120s     | ~12-15 min | ~14-18 min | ~1-2GB      |

**Total for all 4**: ~1-1.5 hours

## Error Handling

The script now gracefully handles:
- ✅ System interruptions (Ctrl+C)
- ✅ Out of memory errors
- ✅ Corrupted checkpoints
- ✅ Missing dependencies
- ✅ Individual solution failures (continues with others)

## What Changed in Code

### `benchmark_slot_retrievers.py`

1. **Added checkpoint support** in `benchmark_retriever()`:
   - Loads existing progress
   - Saves every 10 queries
   - Atomic saves (write to .tmp, then rename)

2. **Enhanced progress reporting**:
   - ETA calculation
   - Rolling averages (last 5 queries)
   - Memory tracking
   - Updates every 5 queries

3. **Better logging**:
   - Structured output with separators
   - Timing information for all stages
   - Detailed per-solution summaries

4. **Recommendations**:
   - Auto-identifies fastest, most accurate, lowest memory
   - Prints at end of benchmark

### `benchmark_all_retrievers.sh`

1. **Baseline skip by default**:
   - Added `--include-baseline` flag
   - Warning message if used

2. **Auto-detect completed solutions**:
   - Checks for `*_results.json` files
   - Loads instead of re-running

3. **Combined results handling**:
   - Only includes solutions that ran
   - Handles missing baseline gracefully

## Related Documentation

- `scripts/README_RUNNING_BENCHMARKS.md` - Full usage guide
- `scripts/README_SLOT_BENCHMARKS.md` - Slot retriever architecture
