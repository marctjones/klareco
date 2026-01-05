# Running Slot Retriever Benchmarks

## Quick Start

```bash
# Run all memory-efficient retrievers (recommended for large indexes)
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full

# Test on smaller index with baseline included
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_test \
  --include-baseline
```

## What Gets Tested

By default, tests **4 memory-efficient implementations**:

1. **Mmap** - Memory-mapped file access (lowest RAM, slower)
2. **FAISS** - FAISS-accelerated retrieval (fast, medium RAM)
3. **MultiFAISS** - Separate FAISS index per slot (fastest)
4. **SQLite** - Database backend (low RAM, medium speed)

**Baseline is SKIPPED** by default because it loads the entire index into RAM:
- 4.2M docs = 19GB file → **~30GB+ RAM needed**
- Will freeze most systems

## Command-Line Options

```bash
./scripts/benchmark_all_retrievers.sh [OPTIONS]

Options:
  --index DIR           Index directory (default: data/indexes/slot_full)
  --output-dir DIR      Output directory (default: benchmark_results)
  --num-queries N       Number of test queries (default: 50)
  --include-baseline    Include baseline (WARNING: needs ~30GB RAM)
  --help                Show help message
```

## Output Files

All results saved to `benchmark_results/`:

```
benchmark_results/
├── queries_TIMESTAMP.jsonl           # Test queries
├── mmap_TIMESTAMP.json               # Mmap results
├── faiss_TIMESTAMP.json              # FAISS results
├── multifaiss_TIMESTAMP.json         # MultiFAISS results
├── sqlite_TIMESTAMP.json             # SQLite results
├── combined_TIMESTAMP.json           # All results combined
└── report_TIMESTAMP.html             # Interactive HTML report
```

## Resuming from Interruptions

The script automatically:
- **Loads completed results** if a solution finished
- **Resumes from checkpoint** if a solution was interrupted mid-run
- **Skips completed solutions** when rerunning

Just rerun the same command:
```bash
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full
```

## Progress Monitoring

The benchmark shows real-time progress:

```
[5/50] Avg latency: 125.3ms | Avg recall: 0.850 | Memory: 2048MB
[10/50] Avg latency: 118.7ms | Avg recall: 0.862 | Memory: 2051MB
...
```

- Updates every 5 queries
- Saves checkpoint every 10 queries
- Shows running average of last 5 queries

## Estimated Runtime

For 4.2M documents, 50 queries:

| Retriever   | Time Estimate | Memory Peak |
|-------------|---------------|-------------|
| Mmap        | ~20-30 min    | ~500MB      |
| FAISS       | ~10-15 min    | ~3-5GB      |
| MultiFAISS  | ~5-10 min     | ~4-6GB      |
| SQLite      | ~15-20 min    | ~1-2GB      |

**Total**: ~1-1.5 hours for all 4 retrievers

## Troubleshooting

### System freezes during benchmark

**Cause**: Ran with `--include-baseline` on large index

**Solution**:
1. Kill the process: `Ctrl+C`
2. Rerun WITHOUT `--include-baseline`

### Checkpoint not resuming

**Check**: Look for `*_checkpoint.json` files in output directory

**Fix**: Checkpoint saves every 10 queries. If interrupted before first checkpoint, no resume data exists.

### Out of memory error

**Solutions**:
- Close other applications
- Use smaller test index: `--index data/indexes/slot_test`
- Reduce queries: `--num-queries 10`

## When to Use Each Retriever

| Use Case | Recommended Retriever |
|----------|----------------------|
| **Production (low memory)** | Mmap or SQLite |
| **Production (speed)** | MultiFAISS |
| **Development/testing** | FAISS (balanced) |
| **Small indexes (<100K)** | Baseline OK |
| **Large indexes (>1M)** | Never baseline |

## Example Workflows

### Quick test on small index
```bash
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_test \
  --num-queries 20 \
  --include-baseline
```

### Full production benchmark
```bash
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_full \
  --num-queries 100 \
  --output-dir benchmark_results/production_test
```

### Compare with previous run
```bash
# First run
./scripts/benchmark_all_retrievers.sh --output-dir benchmark_results/run1

# After code changes
./scripts/benchmark_all_retrievers.sh --output-dir benchmark_results/run2

# Compare
python scripts/visualize_benchmark_results.py \
  --results benchmark_results/run1/combined_*.json \
             benchmark_results/run2/combined_*.json
```
