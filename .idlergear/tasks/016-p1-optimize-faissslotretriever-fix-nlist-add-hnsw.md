---
id: 16
title: '[P1] Optimize FAISSSlotRetriever: Fix nlist + add HNSW quantizer (Tasks #9,
  #10)'
state: open
created: '2026-01-04T00:46:06.894375Z'
labels:
- enhancement
- M2
- faiss
- P1
priority: high
---
## Goal

Optimize the current FAISSSlotRetriever with FAISS best practices for 4.2M dataset.

**Priority**: P1 (Highest) - Quick wins, immediate impact

**Expected Results:**
- Memory: 2.6GB (same as current)
- Latency: 5.1ms → 3-4ms (25% faster)
- Recall: 85% → 88-90% (meets M2 goal!)

## Implementation Tasks

### 1. Fix nlist calculation (Task #9)

**Current:**
```python
nlist = int(np.sqrt(embeddings.shape[0]))  # √4.2M = 2,049 clusters
```

**Fix to:**
```python
# For 4.2M docs (1M-10M range), use FAISS recommended value
if embeddings.shape[0] >= 1_000_000:
    nlist = 65536  # FAISS guideline for 1M-10M
else:
    nlist = int(4 * np.sqrt(embeddings.shape[0]))  # 4×√N minimum
```

### 2. Add HNSW quantizer (Task #10)

**Current:**
```python
quantizer = faiss.IndexFlatIP(dim)  # Brute-force
```

**Fix to:**
```python
# HNSW quantizer for faster coarse quantization
quantizer = faiss.IndexHNSWFlat(dim, M=32)
quantizer.hnsw.efConstruction = 40
```

### 3. Make search parameters tunable (Task #11)

**Add to search() method:**
```python
def search(self, query: str, top_k: int = 10, faiss_top_n: int = 500,
           nprobe: int = None, efSearch: int = None, **kwargs):
    """
    Args:
        nprobe: IVF cells to search (default: 64 for nlist=65K)
        efSearch: HNSW quantizer search depth (default: 16)
    """
    # Set runtime parameters
    if nprobe is not None and hasattr(self.faiss_index, 'nprobe'):
        self.faiss_index.nprobe = nprobe
    else:
        # Default: 64 for 65K clusters (~0.1% of clusters)
        self.faiss_index.nprobe = 64
    
    if efSearch is not None and hasattr(self.faiss_index.quantizer, 'hnsw'):
        self.faiss_index.quantizer.hnsw.efSearch = efSearch
    # Continue with search...
```

## Memory Tuning for <16GB RAM

**Current memory usage:**
- Index embeddings: 128d × 4 bytes × 4.2M = 2.1GB
- FAISS overhead: ~500MB
- Slot metadata: ~100MB
- Python overhead: ~200MB
- **Total: ~2.9GB** ✅ Well under 16GB

**Parameters:**
```python
# Optimized for laptop (16 cores, 30GB RAM)
CONFIG = {
    'nlist': 65536,           # FAISS recommended
    'nprobe': 64,             # Search 64/65536 = 0.1% of clusters
    'efConstruction': 40,     # HNSW build quality
    'efSearch': 16,           # HNSW search depth (tunable at runtime)
    'faiss_top_n': 500,       # Stage 1 candidates
}
```

## Files to Modify

- `klareco/rag/slot_retriever_faiss.py`
  - Update `_create_faiss_index()` method
  - Update `search()` method to accept runtime params

## Validation Test

Create `tests/test_faiss_optimized.py`:

```python
import pytest
from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever

def test_optimized_index_creation(tmp_path):
    """Test that optimized index uses correct parameters."""
    # Create small test index
    retriever = FAISSSlotRetriever.create(
        corpus_file=test_corpus,
        output_dir=tmp_path,
    )
    
    # Verify nlist
    assert retriever.faiss_index.nlist == 65536
    
    # Verify HNSW quantizer
    assert hasattr(retriever.faiss_index.quantizer, 'hnsw')
    assert retriever.faiss_index.quantizer.hnsw.efConstruction == 40

def test_runtime_parameter_tuning(retriever):
    """Test that nprobe/efSearch can be adjusted at runtime."""
    # Default search
    results1 = retriever.search("Kiu kreis Esperanton?", top_k=10)
    
    # Higher accuracy search
    results2 = retriever.search("Kiu kreis Esperanton?", top_k=10, 
                                nprobe=128, efSearch=32)
    
    # Should return results (may differ due to parameters)
    assert len(results1) == 10
    assert len(results2) == 10

def test_memory_usage(retriever):
    """Test that memory stays under 4GB."""
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    assert mem_mb < 4000, f"Memory usage {mem_mb:.0f}MB exceeds 4GB limit"

def test_accuracy_improvement():
    """Test that recall improves over baseline."""
    # Compare old vs new on benchmark queries
    old_recall = run_benchmark(old_retriever, queries)
    new_recall = run_benchmark(optimized_retriever, queries)
    
    assert new_recall >= old_recall + 0.03  # At least 3% improvement
```

## Demo Script

Create `scripts/demo_faiss_optimized.py`:

```python
#!/usr/bin/env python3
"""
Demo: Optimized FAISSSlotRetriever with runtime parameter tuning.

Usage:
    python scripts/demo_faiss_optimized.py --interactive
    python scripts/demo_faiss_optimized.py --benchmark
"""

import time
from pathlib import Path
from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
from klareco.rag.slot_indexer import SlotBasedIndexer

def main():
    print("Loading optimized FAISS retriever...")
    
    index_path = Path("data/indexes/slot_full")
    indexer = SlotBasedIndexer.load()
    retriever = FAISSSlotRetriever(index_path, indexer)
    
    print(f"✓ Loaded {retriever.faiss_index.ntotal:,} documents")
    print(f"✓ Index type: {type(retriever.faiss_index).__name__}")
    print(f"✓ Clusters (nlist): {retriever.faiss_index.nlist:,}")
    print(f"✓ Quantizer: {type(retriever.faiss_index.quantizer).__name__}")
    print()
    
    # Demo 1: Default search
    print("=" * 60)
    print("Demo 1: Default search (nprobe=64, efSearch=16)")
    print("=" * 60)
    
    query = "Kiu kreis Esperanton?"
    print(f"Query: {query}")
    
    start = time.time()
    results = retriever.search(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    
    print(f"⏱️  Latency: {elapsed:.1f}ms")
    print(f"📊 Results:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {doc['text'][:80]}...")
    print()
    
    # Demo 2: Fast search
    print("=" * 60)
    print("Demo 2: Fast search (nprobe=32, efSearch=8)")
    print("=" * 60)
    
    start = time.time()
    results = retriever.search(query, top_k=5, nprobe=32, efSearch=8)
    elapsed = (time.time() - start) * 1000
    
    print(f"⏱️  Latency: {elapsed:.1f}ms (faster!)")
    print(f"📊 Results:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {doc['text'][:80]}...")
    print()
    
    # Demo 3: High accuracy search
    print("=" * 60)
    print("Demo 3: High accuracy search (nprobe=128, efSearch=32)")
    print("=" * 60)
    
    start = time.time()
    results = retriever.search(query, top_k=5, nprobe=128, efSearch=32)
    elapsed = (time.time() - start) * 1000
    
    print(f"⏱️  Latency: {elapsed:.1f}ms (more thorough)")
    print(f"📊 Results:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {doc['text'][:80]}...")
    print()
    
    # Demo 4: Parameter sweep
    print("=" * 60)
    print("Demo 4: Parameter sweep (speed vs accuracy)")
    print("=" * 60)
    
    configs = [
        ("Fast", {'nprobe': 16, 'efSearch': 8}),
        ("Default", {'nprobe': 64, 'efSearch': 16}),
        ("Accurate", {'nprobe': 128, 'efSearch': 32}),
        ("Thorough", {'nprobe': 256, 'efSearch': 64}),
    ]
    
    print(f"{'Config':<12} {'Latency':<12} {'Top-1 Score':<12}")
    print("-" * 40)
    
    for name, params in configs:
        start = time.time()
        results = retriever.search(query, top_k=5, **params)
        elapsed = (time.time() - start) * 1000
        
        top_score = results[0][0] if results else 0.0
        print(f"{name:<12} {elapsed:>8.1f}ms    {top_score:.4f}")
    
    print()
    print("✅ Demo complete!")
    print()
    print("Key takeaway: You can tune speed/accuracy at runtime!")

if __name__ == '__main__':
    main()
```

## Rebuild Index

```bash
# Rebuild with optimized parameters
python scripts/index_slot_based.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/indexes/slot_faiss_optimized \
    --fresh
```

## Benchmark

```bash
# Run benchmark with optimized index
python scripts/benchmark_slot_retrievers.py \
    --index data/indexes/slot_faiss_optimized \
    --retrievers faiss \
    --output benchmark_results/faiss_optimized.json
```

## Acceptance Criteria

- [ ] FAISSSlotRetriever updated with nlist=65536
- [ ] HNSW quantizer implemented (M=32, efConstruction=40)
- [ ] Runtime parameter tuning (nprobe, efSearch) working
- [ ] Memory usage < 4GB confirmed
- [ ] Validation tests pass
- [ ] Demo script works end-to-end
- [ ] Benchmark shows 3-4ms latency
- [ ] Benchmark shows 88-90% recall

## Related Tasks

- Implements: Task #9 (nlist fix)
- Implements: Task #10 (HNSW quantizer)
- Implements: Task #11 (runtime tuning)
- Replaces: Current FAISSSlotRetriever
