# Slot Retriever Benchmark - Quick Start

## ⚡ Run the Safe Benchmark (Recommended)

```bash
./scripts/benchmark_safe.sh --index data/indexes/slot_full
```

**OR** use the main script (same thing):

```bash
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full
```

Both commands are **memory-safe** and will NOT freeze your system.

## 🛡️ What Gets Tested

### ✅ Memory-Safe Retrievers (Default)

| Retriever   | Memory Peak | Speed     | Use Case |
|-------------|-------------|-----------|----------|
| **mmap**    | ~500MB      | Slower    | Production (low memory) |
| **faiss**   | ~3-5GB      | Fast      | Production (balanced) |
| **multifaiss** | ~4-6GB   | Fastest   | Production (speed priority) |
| **sqlite**  | ~1-2GB      | Medium    | Production (database backend) |

### ❌ SKIPPED by Default

| Retriever   | Why Skipped |
|-------------|-------------|
| **baseline** | Loads entire 19GB index into RAM (~30GB needed), **freezes most systems** |

## ⏱️ Expected Runtime

For **4.2M documents, 50 queries**:

- **Total time**: ~1-1.5 hours for all 4 retrievers
- **Updates**: Every 1 min minimum (+ events)
- **Checkpoints**: Every 10 queries (auto-resume)

## 📊 What You'll See

### Progress Updates
```bash
[5/50] Latency: 125ms | Recall: 0.850 | Memory: 2048MB | ETA: 5m 30s
[7/50] Latency: 132ms | Recall: 0.845 | Memory: 2049MB | ETA: 5m 15s | ⚠ slow query: 287ms
[10/50] Latency: 118ms | Recall: 0.862 | Memory: 2051MB | ETA: 4m 45s | 💾 checkpoint
[18/50] Latency: 117ms | Recall: 0.865 | Memory: 2342MB | ETA: 3m 45s | 📈 mem spike: +289MB
```

### Final Results
```
====================================================================================================
FINAL BENCHMARK COMPARISON
====================================================================================================

Solution         Mean (ms)    Median (ms)  P95 (ms)     Memory (MB)  Recall@10
----------------------------------------------------------------------------------------------------
mmap             234.56       218.34       456.78       512.3        0.920
faiss            125.34       118.50       245.67       2987.3       0.920
multifaiss       89.23        85.12        178.45       4123.5       0.920
sqlite           178.90       165.23       345.12       1234.6       0.920

RECOMMENDATIONS:

  🚀 FASTEST:       multifaiss (89.2ms mean latency)
  🎯 MOST ACCURATE:  faiss (92.0% recall)
  💾 LOWEST MEMORY:  mmap (512MB)
```

## 🔄 Resume After Interruption

If interrupted (Ctrl+C, crash, etc.), just rerun:

```bash
./scripts/benchmark_safe.sh --index data/indexes/slot_full
```

The script will:
- ✅ Skip completed retrievers
- ✅ Resume from checkpoint (every 10 queries)
- ✅ Continue where it left off

## 🚫 DO NOT RUN (Will Freeze!)

**NEVER run with baseline on large index:**
```bash
# ❌ DON'T DO THIS on slot_full:
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full --include-baseline
```

Baseline is **only safe for small indexes** (<100K docs):
```bash
# ✅ OK for small test index:
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_test --include-baseline
```

## 📁 Output Files

All results saved to `benchmark_results/`:

```
benchmark_results/
├── queries_20260103_003218.jsonl     # Test queries
├── mmap_20260103_003218.json         # Mmap results
├── faiss_20260103_003218.json        # FAISS results
├── multifaiss_20260103_003218.json   # MultiFAISS results
├── sqlite_20260103_003218.json       # SQLite results
├── combined_20260103_003218.json     # Combined results
└── report_20260103_003218.html       # Interactive HTML report
```

## 🛠️ Command Options

```bash
./scripts/benchmark_safe.sh [OPTIONS]

Options:
  --index DIR           Index directory (default: data/indexes/slot_full)
  --output-dir DIR      Output directory (default: benchmark_results)
  --num-queries N       Number of test queries (default: 50)
  --help                Show help message

Examples:
  # Use different index
  ./scripts/benchmark_safe.sh --index data/indexes/slot_test

  # Test with more queries
  ./scripts/benchmark_safe.sh --num-queries 100

  # Custom output location
  ./scripts/benchmark_safe.sh --output-dir my_benchmark_results
```

## 🐛 Troubleshooting

### Benchmark appears frozen

**Check**: Updates show at least every 1 minute. If >2 minutes with no output, it may be stuck.

**Solution**:
1. Press Ctrl+C in the terminal
2. Check if baseline was accidentally included
3. Rerun with safe script

### Out of memory error

**Cause**: System ran out of RAM during FAISS or MultiFAISS retriever.

**Solutions**:
- Close other applications
- Use smaller test index: `--index data/indexes/slot_test`
- Reduce queries: `--num-queries 10`
- Skip memory-intensive retrievers (edit script to remove faiss/multifaiss)

### Process killed

**Cause**: System OOM killer terminated the process.

**Solution**: Use smaller index or test fewer retrievers at a time:
```bash
# Test only mmap (lowest memory)
python scripts/benchmark_slot_retrievers.py \
  --index data/indexes/slot_full \
  --queries benchmark_results/queries_*.jsonl \
  --solution mmap \
  --output benchmark_results/mmap_results.json
```

## 📚 Documentation

- `scripts/README_RUNNING_BENCHMARKS.md` - Full usage guide
- `scripts/BENCHMARK_IMPROVEMENTS.md` - What's new
- `scripts/BENCHMARK_UPDATE_STRATEGY.md` - Update trigger logic
- `scripts/README_SLOT_BENCHMARKS.md` - Slot retriever architecture

## ✅ Verification Checklist

Before running on production index:

- [ ] Using `benchmark_safe.sh` or `benchmark_all_retrievers.sh` (NOT --include-baseline)
- [ ] Index is `slot_full` (4.2M docs)
- [ ] Have 1-2 hours available for full benchmark
- [ ] System has at least 8GB free RAM
- [ ] Other heavy applications closed

---

## 🎯 Quick Reference

**Safe command** (copy-paste ready):
```bash
cd ~/Projects/klareco
./scripts/benchmark_safe.sh --index data/indexes/slot_full
```

**Expected output**: 4 retrievers tested in ~1-1.5 hours, automatic checkpoints, real-time progress.

**Memory usage**: Peak ~6GB (MultiFAISS), minimum ~500MB (Mmap).

**Safe to interrupt**: Yes, resume with same command.
