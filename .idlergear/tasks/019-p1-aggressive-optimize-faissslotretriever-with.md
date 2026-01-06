---
id: 19
title: '[P1-AGGRESSIVE] Optimize FAISSSlotRetriever with aggressive tuning for 30GB
  RAM'
state: open
created: '2026-01-04T01:14:57.195223Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Goal

Aggressive tuning of FAISSSlotRetriever to maximize accuracy and speed using available 30GB RAM and 16 CPU cores.

**Priority**: P1-AGGRESSIVE (After basic P1 is working)

**Conservative tuning (Task #16):**
- Memory: 2.9GB
- nprobe: 64
- efSearch: 16
- Latency: 3-4ms
- Recall: 88-90%

**Aggressive tuning (this task):**
- Memory: 4-5GB (still only 16% of 30GB!)
- nprobe: 128-256
- efSearch: 32-64
- Latency: 4-6ms (slightly slower but much more accurate)
- Recall: **92-95%** (near-perfect!)

## Aggressive Parameter Configuration

```python
# configs/faiss_aggressive.py

AGGRESSIVE_CONFIG = {
    # Index building (one-time cost)
    'nlist': 65536,              # Same as conservative
    'M': 48,                     # HNSW connectivity (32 → 48, more neighbors)
    'efConstruction': 80,        # Build quality (40 → 80, better graph)
    
    # Search parameters (runtime tunable)
    'nprobe_default': 128,       # Search 128/65536 = 0.2% of clusters (2× conservative)
    'nprobe_max': 256,           # For critical queries (4× conservative)
    'efSearch_default': 32,      # HNSW search depth (2× conservative)
    'efSearch_max': 64,          # For highest accuracy (4× conservative)
    
    # Reranking
    'faiss_top_n': 1000,         # Stage 1 candidates (500 → 1000, more to rerank)
    'slot_rerank': True,         # Always rerank with slot similarity
    
    # Memory settings
    'preload_index': True,       # Load entire index into RAM (no mmap)
    'cache_queries': 50000,      # Cache 50K most common queries (vs 10K)
}
```

## Memory Impact

**Conservative (Task #16):**
```
Index:           2.1GB
FAISS overhead:  0.5GB
Metadata:        0.1GB
Python:          0.2GB
Total:           2.9GB
```

**Aggressive (this task):**
```
Index:           2.1GB   (same - full vectors, no compression)
FAISS overhead:  1.2GB   (larger HNSW graph: M=48 vs M=32)
                         (more clusters searched: nprobe=128 vs 64)
Metadata:        0.1GB   (same)
Query cache:     0.4GB   (50K queries × ~8KB each)
Preloaded data:  0.5GB   (hot data pinned in RAM)
Python:          0.2GB   (same)
────────────────────────
Total:           4.5GB   ✅ Only 15% of 30GB RAM!
```

**Remaining RAM: 25.5GB** for OS, browser, other apps.

## CPU Utilization

**Conservative:**
- Single query: 1 core
- Concurrent: Limited by Python GIL

**Aggressive:**
```python
# Enable FAISS multi-threading
import faiss

# Use all 16 cores for FAISS operations
faiss.omp_set_num_threads(16)

# This parallelizes:
# - IVF cluster search (nprobe=128 clusters in parallel)
# - HNSW graph traversal (parallel beam search)
# - Distance computations (SIMD vectorization)
```

**Expected speedup:**
- Single query: 3-4ms → **2-3ms** (faster despite more work!)
- Batch queries: Near-linear scaling with cores

## Implementation

### 1. Update FAISSSlotRetriever with Aggressive Config

```python
# klareco/rag/slot_retriever_faiss.py

class FAISSSlotRetriever:
    """Existing class..."""
    
    @classmethod
    def create_aggressive(
        cls,
        index_path: Path,
        indexer: SlotBasedIndexer,
        use_all_cores: bool = True,
    ):
        """
        Create retriever with aggressive tuning for high-end hardware.
        
        Optimized for:
        - 30GB+ RAM
        - 16+ CPU cores
        - Maximum accuracy priority
        """
        import faiss
        
        # Use all CPU cores
        if use_all_cores:
            import os
            num_cores = os.cpu_count() or 16
            faiss.omp_set_num_threads(num_cores)
            logger.info(f"  FAISS using {num_cores} threads")
        
        # Load with aggressive settings
        retriever = cls(index_path, indexer)
        
        # Override defaults
        retriever.default_search_params = {
            'nprobe': 128,      # 2× conservative
            'efSearch': 32,     # 2× conservative
            'faiss_top_n': 1000,  # 2× conservative
        }
        
        # Preload index into RAM (no page faults)
        if hasattr(retriever.faiss_index, 'make_direct_map'):
            retriever.faiss_index.make_direct_map()
            logger.info("  Direct map enabled (faster lookup)")
        
        return retriever
    
    def search_aggressive(
        self,
        query: str,
        top_k: int = 10,
        accuracy: str = 'high',  # 'high', 'maximum', 'ultra'
    ):
        """
        Search with aggressive accuracy settings.
        
        Args:
            accuracy: Accuracy level
                - 'high': nprobe=128, efSearch=32 (2× conservative)
                - 'maximum': nprobe=256, efSearch=64 (4× conservative)
                - 'ultra': nprobe=512, efSearch=128 (8× conservative, very slow)
        """
        if accuracy == 'high':
            nprobe, efSearch = 128, 32
        elif accuracy == 'maximum':
            nprobe, efSearch = 256, 64
        elif accuracy == 'ultra':
            nprobe, efSearch = 512, 128
        else:
            raise ValueError(f"Unknown accuracy: {accuracy}")
        
        return self.search(
            query,
            top_k=top_k,
            nprobe=nprobe,
            efSearch=efSearch,
            faiss_top_n=1000,
        )
```

### 2. Build Aggressive Index

```python
# scripts/build_aggressive_index.py

def build_aggressive_faiss_index(
    corpus_file: Path,
    output_dir: Path,
):
    """Build FAISS index with aggressive parameters."""
    
    # Load embeddings
    embeddings = load_embeddings(corpus_file)
    n, dim = embeddings.shape
    
    print(f"Building aggressive FAISS index for {n:,} documents...")
    
    # Create aggressive index
    nlist = 65536  # Same as conservative
    M = 48         # More HNSW connections (32 → 48)
    
    # HNSW quantizer with higher quality
    quantizer = faiss.IndexHNSWFlat(dim, M)
    quantizer.hnsw.efConstruction = 80  # Higher build quality (40 → 80)
    
    # IVF index
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train with more samples
    train_size = min(n, 512 * nlist)  # 2× conservative (256 → 512)
    print(f"  Training on {train_size:,} samples...")
    
    train_indices = np.random.choice(n, train_size, replace=False)
    index.train(embeddings[train_indices])
    
    # Add all vectors
    print(f"  Adding {n:,} vectors...")
    index.add(embeddings)
    
    # Save
    faiss.write_index(index, str(output_dir / "aggressive_faiss.index"))
    
    # Save config
    config = {
        'mode': 'aggressive',
        'nlist': nlist,
        'M': M,
        'efConstruction': 80,
        'train_samples': train_size,
        'recommended_nprobe': 128,
        'recommended_efSearch': 32,
    }
    
    with open(output_dir / "aggressive_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Aggressive index saved to {output_dir}")
    print(f"  Memory: ~4.5GB")
    print(f"  Expected recall: 92-95%")
    print(f"  Expected latency: 2-3ms (with 16 cores)")
```

### 3. Query Caching Layer

```python
# klareco/rag/cached_retriever.py

from functools import lru_cache
import hashlib

class CachedAggressiveRetriever:
    """Wrapper that caches query results."""
    
    def __init__(
        self,
        base_retriever: FAISSSlotRetriever,
        cache_size: int = 50000,  # 50K queries
    ):
        self.base = base_retriever
        self.cache_size = cache_size
        self._create_cache()
    
    def _create_cache(self):
        """Create LRU cache for query results."""
        @lru_cache(maxsize=self.cache_size)
        def _cached_search(query_hash: str, top_k: int, nprobe: int, efSearch: int):
            # Actual search (expensive)
            return self.base.search(
                self._unhash_query(query_hash),
                top_k=top_k,
                nprobe=nprobe,
                efSearch=efSearch,
            )
        
        self._cached_search = _cached_search
        self._query_map = {}  # hash → query text
    
    def _hash_query(self, query: str) -> str:
        """Hash query for cache key."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def search(self, query: str, **kwargs):
        """Search with caching."""
        query_hash = self._hash_query(query)
        self._query_map[query_hash] = query
        
        # Extract params for cache key
        top_k = kwargs.get('top_k', 10)
        nprobe = kwargs.get('nprobe', 128)
        efSearch = kwargs.get('efSearch', 32)
        
        # Lookup or compute
        return self._cached_search(query_hash, top_k, nprobe, efSearch)
    
    def get_cache_stats(self):
        """Get cache statistics."""
        info = self._cached_search.cache_info()
        return {
            'size': info.currsize,
            'maxsize': info.maxsize,
            'hits': info.hits,
            'misses': info.misses,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0,
        }
```

## Expected Performance

### Accuracy Comparison

| Config | nprobe | efSearch | Recall | MRR |
|--------|--------|----------|--------|-----|
| Conservative | 64 | 16 | 88-90% | 0.85 |
| Aggressive (high) | 128 | 32 | **92-94%** | **0.90** |
| Aggressive (max) | 256 | 64 | **94-96%** | **0.93** |
| Aggressive (ultra) | 512 | 128 | **96-98%** | **0.95** |

### Latency Comparison

| Config | Single Query | With Cache | Batch (10 queries) |
|--------|--------------|------------|-------------------|
| Conservative (1 core) | 3-4ms | - | 30-40ms |
| Aggressive (16 cores) | **2-3ms** | **0.1ms** | **8-12ms** |

**Speedup from multi-threading:** 2.5× on batch queries!

### Memory Usage

| Component | Conservative | Aggressive | Your RAM |
|-----------|--------------|------------|----------|
| FAISS index | 2.6GB | 3.3GB | 30GB |
| Query cache | - | 0.4GB | |
| Hot data | - | 0.5GB | |
| **Total** | **2.9GB** | **4.5GB** | **✅ 15% used** |

## Validation Test

```python
# tests/test_faiss_aggressive.py

def test_aggressive_accuracy():
    """Test that aggressive tuning improves accuracy."""
    conservative = FAISSSlotRetriever(index_path, indexer)
    aggressive = FAISSSlotRetriever.create_aggressive(index_path, indexer)
    
    # Run on benchmark
    queries = load_benchmark_queries()
    
    recall_conservative = measure_recall(conservative, queries, nprobe=64, efSearch=16)
    recall_aggressive = measure_recall(aggressive, queries, nprobe=128, efSearch=32)
    
    # Should improve by at least 3%
    assert recall_aggressive >= recall_conservative + 0.03
    assert recall_aggressive >= 0.92  # Target: 92%+

def test_multicore_speedup():
    """Test that 16 cores improve batch performance."""
    import faiss
    
    # Single-threaded
    faiss.omp_set_num_threads(1)
    retriever_1core = FAISSSlotRetriever(index_path, indexer)
    time_1core = benchmark_batch(retriever_1core, queries=10)
    
    # Multi-threaded
    faiss.omp_set_num_threads(16)
    retriever_16core = FAISSSlotRetriever(index_path, indexer)
    time_16core = benchmark_batch(retriever_16core, queries=10)
    
    # Should be at least 2× faster
    speedup = time_1core / time_16core
    assert speedup >= 2.0, f"Speedup {speedup:.1f}× is less than 2×"

def test_memory_under_limit():
    """Test that aggressive mode stays under 5GB."""
    import psutil
    
    retriever = FAISSSlotRetriever.create_aggressive(index_path, indexer)
    
    # Run queries to populate cache
    for _ in range(100):
        retriever.search_aggressive("Kiu kreis Esperanton?", accuracy='high')
    
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    
    assert mem_gb < 5.0, f"Memory {mem_gb:.1f}GB exceeds 5GB limit"

def test_cache_effectiveness():
    """Test that query cache improves latency."""
    cached = CachedAggressiveRetriever(base_retriever, cache_size=50000)
    
    query = "Kiu kreis Esperanton?"
    
    # First query (miss)
    import time
    start = time.time()
    cached.search(query)
    miss_time = (time.time() - start) * 1000
    
    # Second query (hit)
    start = time.time()
    cached.search(query)
    hit_time = (time.time() - start) * 1000
    
    # Should be at least 20× faster
    speedup = miss_time / hit_time
    assert speedup >= 20, f"Cache speedup {speedup:.1f}× is less than 20×"
    
    # Check stats
    stats = cached.get_cache_stats()
    assert stats['hit_rate'] > 0  # At least one hit
```

## Demo Script

```python
# scripts/demo_faiss_aggressive.py

def main():
    print("=" * 70)
    print("Aggressive FAISS Tuning Demo (30GB RAM, 16 Cores)")
    print("=" * 70)
    print()
    
    # Load both configs
    conservative = FAISSSlotRetriever(index_path, indexer)
    aggressive = FAISSSlotRetriever.create_aggressive(index_path, indexer)
    
    print("System info:")
    import os
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  FAISS threads: {faiss.omp_get_max_threads()}")
    print()
    
    # Demo 1: Accuracy levels
    print("=" * 70)
    print("Demo 1: Accuracy Levels")
    print("=" * 70)
    print()
    
    query = "Kiu kreis Esperanton?"
    
    configs = [
        ("Conservative", conservative, {'nprobe': 64, 'efSearch': 16}),
        ("Aggressive (high)", aggressive, {'accuracy': 'high'}),
        ("Aggressive (max)", aggressive, {'accuracy': 'maximum'}),
        ("Aggressive (ultra)", aggressive, {'accuracy': 'ultra'}),
    ]
    
    print(f"{'Config':<20} {'Latency':<12} {'Top-1 Score':<12} {'Memory':<10}")
    print("-" * 60)
    
    for name, ret, params in configs:
        import time, psutil
        
        start = time.time()
        if 'accuracy' in params:
            results = ret.search_aggressive(query, top_k=10, **params)
        else:
            results = ret.search(query, top_k=10, **params)
        latency = (time.time() - start) * 1000
        
        score = results[0][0] if results else 0.0
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"{name:<20} {latency:>8.1f}ms    {score:.4f}        {mem_mb:>6.0f}MB")
    
    print()
    
    # Demo 2: Batch processing speedup
    print("=" * 70)
    print("Demo 2: Multi-core Batch Speedup")
    print("=" * 70)
    print()
    
    queries = [
        "Kiu kreis Esperanton?",
        "Kiam Zamenhof kreis Esperanton?",
        "Kie naski\u011dis Zamenhof?",
        "Kio estas Esperanto?",
        "Kial Esperanto estas facila?",
    ]
    
    # Single-threaded
    import faiss
    faiss.omp_set_num_threads(1)
    start = time.time()
    for q in queries:
        conservative.search(q, top_k=10, nprobe=64, efSearch=16)
    time_1core = (time.time() - start) * 1000
    
    # Multi-threaded
    faiss.omp_set_num_threads(16)
    start = time.time()
    for q in queries:
        aggressive.search(q, top_k=10, nprobe=128, efSearch=32)
    time_16core = (time.time() - start) * 1000
    
    print(f"1 core (conservative):  {time_1core:>8.1f}ms ({time_1core/5:.1f}ms/query)")
    print(f"16 cores (aggressive):  {time_16core:>8.1f}ms ({time_16core/5:.1f}ms/query)")
    print(f"Speedup:                {time_1core/time_16core:>8.2f}×")
    print()
    
    # Demo 3: Query caching
    print("=" * 70)
    print("Demo 3: Query Result Caching (50K cache)")
    print("=" * 70)
    print()
    
    cached = CachedAggressiveRetriever(aggressive, cache_size=50000)
    
    # Cold query
    start = time.time()
    cached.search("Kiu kreis Esperanton?", nprobe=128, efSearch=32)
    cold_time = (time.time() - start) * 1000
    
    # Hot query (cached)
    start = time.time()
    cached.search("Kiu kreis Esperanton?", nprobe=128, efSearch=32)
    hot_time = (time.time() - start) * 1000
    
    stats = cached.get_cache_stats()
    
    print(f"Cold query:     {cold_time:>8.1f}ms (computed)")
    print(f"Hot query:      {hot_time:>8.1f}ms (cached)")
    print(f"Speedup:        {cold_time/hot_time:>8.1f}×")
    print()
    print(f"Cache stats:")
    print(f"  Size:         {stats['size']:,} / {stats['maxsize']:,}")
    print(f"  Hit rate:     {stats['hit_rate']:.1%}")
    print()
    
    print("✅ Demo complete!")
    print()
    print("Aggressive tuning summary:")
    print("  • Memory: 4.5GB (15% of your 30GB)")
    print("  • Accuracy: 92-96% (vs 88-90% conservative)")
    print("  • Latency: 2-3ms single, 0.1ms cached")
    print("  • Batch: 2.5× faster with 16 cores")
    print("  • Cache: 50K queries = near-instant for common requests")

if __name__ == '__main__':
    main()
```

## Build Command

```bash
# Build aggressive index (takes ~3 hours, uses all 16 cores)
python scripts/build_aggressive_index.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/indexes/slot_faiss_aggressive \
    --cores 16 \
    --fresh

# Benchmark
python scripts/benchmark_slot_retrievers.py \
    --index data/indexes/slot_faiss_aggressive \
    --retrievers faiss-aggressive \
    --cores 16 \
    --output benchmark_results/faiss_aggressive.json
```

## Acceptance Criteria

- [ ] Aggressive index built with M=48, efConstruction=80
- [ ] Multi-threading enabled (16 cores)
- [ ] Query cache working (50K capacity)
- [ ] Memory usage < 5GB confirmed
- [ ] Recall ≥ 92% on benchmark
- [ ] Batch speedup ≥ 2× vs single-core
- [ ] Cached queries < 0.5ms latency
- [ ] Validation tests pass
- [ ] Demo script works end-to-end

## Related Tasks

- Extends: Task #16 (Conservative FAISS)
- Alternative to: Task #17 (FusedMultiSlot)
- For: High-end hardware (30GB RAM, 16+ cores)
