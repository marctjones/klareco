# Slot-Based Retriever Benchmark Suite

This directory contains scripts to benchmark different implementations of slot-based retrieval against the original memory-intensive version.

## Problem Overview

The original `SlotBasedRetriever` loads the entire index into RAM and performs exhaustive linear search. This caused OOM kills on systems with limited RAM when indexing millions of documents (~27GB RAM usage).

## Solutions Implemented

| Solution | Description | Tradeoff |
|----------|-------------|----------|
| **Baseline** | Original in-memory retriever | Fast, but requires all data in RAM |
| **Solution 1: Memory-Mapped** | Uses numpy memory-mapped arrays with batching | Low RAM, slower (10-100x) |
| **Solution 2: FAISS** | FAISS pre-filtering + slot reranking | Fast, medium RAM, slight accuracy loss |
| **Solution 3: Multi-FAISS** | Separate FAISS index per slot (SUBJ, VERB, OBJ) | Fastest, medium RAM, preserves slot logic |
| **Solution 4: SQLite** | SQLite database with BLOB storage | Low RAM, medium speed, SQL filtering |

## Quick Start

### One Command Benchmark (Recommended)

Run all benchmarks with a single script:

```bash
# Use your existing index
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full

# Or specify custom options
./scripts/benchmark_all_retrievers.sh \
  --index data/indexes/slot_full \
  --output-dir benchmark_results \
  --num-queries 100
```

This script will:
1. Create benchmark queries from your index
2. Run all 5 solutions (baseline, mmap, faiss, multifaiss, sqlite)
3. Combine results into a single JSON
4. Generate HTML report with charts
5. Print summary table

**Output**: Opens `benchmark_results/report_<timestamp>.html` when done.

### Manual Step-by-Step (Advanced)

If you want more control:

#### 1. Run Benchmarks

Benchmark all solutions:

```bash
python scripts/benchmark_slot_retrievers.py \
  --index data/indexes/slot_full \
  --create-queries \
  --num-queries 50 \
  --output benchmark_results.json
```

Or benchmark a specific solution:

```bash
python scripts/benchmark_slot_retrievers.py \
  --index data/indexes/slot_full \
  --queries my_queries.jsonl \
  --solution faiss \
  --output faiss_results.json
```

#### 2. Visualize Results

Generate an HTML report with comparison charts:

```bash
python scripts/visualize_benchmark_results.py \
  --results benchmark_results.json \
  --output benchmark_report.html
```

Then open `benchmark_report.html` in your browser.

## Metrics Measured

### Performance Metrics
- **Latency**: Mean, median, P95, P99 query times (milliseconds)
- **Memory**: Peak RSS and delta from baseline (megabytes)

### Accuracy Metrics (if ground truth provided)
- **Recall@10**: Fraction of relevant docs in top 10
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain

## Implementation Files

| File | Description |
|------|-------------|
| `klareco/rag/slot_retriever.py` | Original baseline (in-memory) |
| `klareco/rag/slot_retriever_mmap.py` | Solution 1: Memory-mapped arrays |
| `klareco/rag/slot_retriever_faiss.py` | Solution 2: FAISS pre-filtering |
| `klareco/rag/slot_retriever_multifaiss.py` | Solution 3: Multi-slot FAISS |
| `klareco/rag/slot_retriever_sqlite.py` | Solution 4: SQLite backend |

## Usage Examples

### Using a Specific Retriever

```python
from klareco.rag.slot_indexer import SlotBasedIndexer
from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever

# Load indexer
indexer = SlotBasedIndexer(
    root_model_path=Path('models/root_embeddings/best_model.pt'),
    affix_model_path=Path('models/affix_transforms_v2/best_model.pt'),
    output_dir=Path('data/indexes/slot_full'),
)

# Create retriever
retriever = FAISSSlotRetriever(
    index_path=Path('data/indexes/slot_full'),
    indexer=indexer,
)

# Search
results = retriever.search("Kiu kreis Esperanton?", top_k=10)

for score, doc in results:
    print(f"[{score:.3f}] {doc['text']}")
```

### Custom Benchmark Queries

Instead of auto-generating queries, you can provide your own JSONL file:

```jsonl
{"query": "Kiu kreis Esperanton?", "relevant_docs": ["Ludoviko Lazaro Zamenhof kreis Esperanton en 1887."]}
{"query": "Kio estas la Unu Ringo?", "relevant_docs": ["La Unu Ringo estas la plej potenca el la Ringoj de Potenco."]}
```

Then run:

```bash
python scripts/benchmark_slot_retrievers.py \
  --index data/indexes/slot_test \
  --queries my_queries.jsonl \
  --output results.json
```

## Expected Results

Based on the design analysis, expected performance characteristics:

| Solution | Query Speed | Memory Usage | Accuracy | Best For |
|----------|-------------|--------------|----------|----------|
| Baseline | Fast (10-50ms) | Very High (10-30GB) | 100% | Small datasets (<100K docs) |
| Memory-Mapped | Slow (100-500ms) | Low (100-500MB) | 100% | Large datasets, limited RAM |
| FAISS | Fast (20-100ms) | Medium (1-5GB) | 95-99% | Production, balanced needs |
| Multi-FAISS | Very Fast (10-50ms) | Medium (2-6GB) | 98-100% | Production, slot-aware queries |
| SQLite | Medium (50-200ms) | Low (100-500MB) | 100% | SQL filtering, disk-based |

## Why Benchmark on Your Existing Index?

**You should use your actual production index** (`slot_full`) for benchmarking, not a test index. Here's why:

1. **Real Performance**: Memory usage and query times scale with index size. A 5K doc test index won't reveal OOM issues that appear with 5M docs.

2. **Accurate Comparison**: The whole point is to see which solution handles YOUR data size within YOUR RAM constraints.

3. **Production Relevance**: Results on toy data don't predict real-world behavior.

**The benchmark script samples random queries from your index**, so it works on any size index. The shell script handles everything:

```bash
# Benchmark on your actual index (millions of docs)
./scripts/benchmark_all_retrievers.sh --index data/indexes/slot_full

# This will show you which solution works for YOUR scale
```

If a solution OOMs or times out (>10 min), the script marks it as failed and continues testing other solutions. This is exactly what you want to know!

## Scaling Recommendations

- **< 100K docs**: Use baseline or Multi-FAISS
- **100K - 1M docs**: Use Multi-FAISS or FAISS
- **> 1M docs**: Use Multi-FAISS with GPU or Memory-Mapped
- **Limited RAM (<8GB)**: Use Memory-Mapped or SQLite

**Run the benchmark to find out what works for your setup!**

## Troubleshooting

### "FAISS index not found"
The FAISS indexes are built automatically on first use. This adds a one-time overhead.

### "Memory-mapped arrays not found"
Similarly, mmap arrays are created on first load. This is a one-time conversion.

### SQLite "database is locked"
Only one process can write to SQLite at a time. Use read-only mode for concurrent queries.

## Future Improvements

Potential enhancements (see issues):
- GPU-accelerated FAISS for Multi-FAISS (10x speedup)
- Quantization for memory-mapped arrays (4x memory reduction)
- PostgreSQL backend with pgvector extension (distributed scaling)
- Hybrid retriever that auto-selects strategy based on query type

## References

- FAISS documentation: https://github.com/facebookresearch/faiss
- Memory-mapped arrays: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
- Benchmark design inspired by BEIR: https://github.com/beir-cellar/beir
