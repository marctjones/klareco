---
id: 20
title: '[P2-AGGRESSIVE] FusedMultiSlot with aggressive parallelism for 16 cores'
state: open
created: '2026-01-04T01:14:58.123662Z'
labels:
- enhancement
- M2
- P2-aggressive
- performance
- parallelism
priority: medium
---
## Goal

Aggressive tuning of FusedMultiSlotRetriever to maximize parallelism using 16 CPU cores and 30GB RAM.

**Priority**: P2-AGGRESSIVE (After basic P2 is working)

**Conservative tuning (Task #17):**
- Memory: 2.4GB
- Parallelism: 3 slot queries (ThreadPoolExecutor, max_workers=3)
- slot_k: 100 candidates per slot
- Latency: 1-2ms
- Recall: 85-90%

**Aggressive tuning (this task):**
- Memory: 6-8GB (still only 20-26% of 30GB!)
- Parallelism: 3 parallel slots × 5 concurrent requests = 15 threads
- slot_k: 500 candidates per slot
- Query batching with multiprocessing
- Latency: **0.5-1ms** per query (2× faster!)
- Throughput: **1000+ queries/sec**
- Recall: 90-92% (better candidates)

## Aggressive Parameter Configuration

```python
# configs/fused_aggressive.py

AGGRESSIVE_CONFIG = {
    # Per-slot retrieval
    'slot_k': 500,              # Candidates per slot (100 → 500)
    'fusion_top_n': 200,        # Candidates after fusion (100 → 200)
    
    # Parallelism
    'max_workers': 15,          # ThreadPool size (3 → 15)
                                # Allows 5 concurrent requests × 3 slots each
    
    'use_process_pool': True,   # Use ProcessPoolExecutor for batch queries
    'batch_size': 50,           # Process 50 queries at once
    
    # Slot index configuration
    'slot_index_nlist': 16384,  # More clusters per slot (better partitioning)
    'slot_nprobe': 32,          # Search more clusters per slot
    
    # Fusion weights (tuned for accuracy)
    'slot_weights': {
        'SUBJ': 0.25,           # Slightly lower (subjects vary more)
        'VERB': 0.50,           # Higher (verbs most discriminative)
        'OBJ': 0.25,            # Slightly lower
    },
    
    # Caching
    'cache_size': 100000,       # Cache 100K queries (vs 10K)
    'cache_fusion_results': True,  # Cache fusion scores (not just final results)
}
```

## Memory Impact

**Conservative (Task #17):**
```
SUBJ index:      0.7GB  (1.4M vectors)
VERB index:      0.7GB  (1.4M vectors)
OBJ index:       0.7GB  (1.4M vectors)
Metadata:        0.1GB
Python:          0.2GB
Total:           2.4GB
```

**Aggressive (this task):**
```
SUBJ index:      1.2GB  (better clustering: nlist=16K vs 2K)
VERB index:      1.2GB
OBJ index:       1.2GB
Metadata:        0.1GB
Query cache:     0.8GB  (100K queries × 8KB)
Fusion cache:    0.5GB  (intermediate results)
ThreadPool:      0.1GB  (15 threads)
ProcessPool:     2GB    (worker processes for batching)
Python:          0.3GB
────────────────────────
Total:           7.4GB  ✅ Only 24% of 30GB RAM!
```

**With process pool running:** Peak 8-9GB (still only 30% of RAM)

## CPU Utilization Strategy

**Conservative:**
- 3 threads per query (SUBJ, VERB, OBJ)
- Sequential query processing
- Wastes 13 of your 16 cores

**Aggressive:**

### Strategy 1: Increased ThreadPool (Simple)
```python
# Allow 5 concurrent queries (5 × 3 = 15 threads)
ThreadPoolExecutor(max_workers=15)

# Throughput: 5 queries in parallel
# Each query: 3 slot searches (SUBJ, VERB, OBJ)
# Total: 15 concurrent FAISS searches
```

### Strategy 2: ProcessPoolExecutor (Batch Queries)
```python
# For batch processing (>10 queries)
ProcessPoolExecutor(max_workers=8)

# Each worker process:
# - Loads own copy of indexes (~2GB per process)
# - Processes batch of queries independently
# - No GIL contention

# Throughput: 8 × 3 = 24 concurrent slot searches
# (More than 16 cores, but I/O overlaps with CPU)
```

### Strategy 3: Hybrid (Best)
```python
class AggressiveFusedRetriever:
    def __init__(self):
        # ThreadPool for single/small batches (<10 queries)
        self.thread_pool = ThreadPoolExecutor(max_workers=15)
        
        # ProcessPool for large batches (≥10 queries)
        self.process_pool = ProcessPoolExecutor(max_workers=8)
    
    def search(self, query):
        """Single query → ThreadPool"""
        return self._search_threaded(query)
    
    def batch_search(self, queries):
        """Batch queries → ProcessPool"""
        if len(queries) < 10:
            return self._search_threaded_batch(queries)
        else:
            return self._search_process_batch(queries)
```

## Implementation

### 1. Aggressive Index Building

```python
# scripts/build_aggressive_multislot.py

def build_aggressive_slot_indexes(corpus_file: Path, output_dir: Path):
    """Build slot indexes with aggressive clustering."""
    
    # Load slot embeddings
    slots_by_role = extract_all_slots(corpus_file)
    
    for slot in ['SUBJ', 'VERB', 'OBJ']:
        print(f"Building aggressive {slot} index...")
        
        embeddings = slots_by_role[slot]  # e.g., 1.4M vectors
        doc_ids = slots_by_role[f'{slot}_ids']
        
        dim = embeddings.shape[1]
        n = embeddings.shape[0]
        
        # Aggressive clustering
        nlist = 16384  # 4× conservative (4096 → 16384)
        
        # Better quantizer
        quantizer = faiss.IndexHNSWFlat(dim, M=32)
        quantizer.hnsw.efConstruction = 40
        
        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train with more samples
        train_size = min(n, 256 * nlist)  # ~4M samples
        train_indices = np.random.choice(n, train_size, replace=False)
        index.train(embeddings[train_indices])
        
        # Add vectors
        index.add(embeddings)
        
        # Save
        slot_dir = output_dir / f"aggressive_{slot.lower()}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(slot_dir / "index.bin"))
        np.save(slot_dir / "doc_ids.npy", doc_ids)
        
        print(f"  ✓ {slot}: {n:,} vectors, {nlist:,} clusters")
```

### 2. Aggressive Retriever Implementation

```python
# klareco/rag/slot_retriever_fused_aggressive.py

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class AggressiveFusedMultiSlotRetriever:
    """
    Aggressive multi-slot retriever optimized for 16+ cores.
    
    Features:
    - 15-thread pool for concurrent queries
    - ProcessPool for batch processing
    - Query result caching (100K cache)
    - Fusion result caching
    - Optimized for throughput
    """
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        max_workers: int = 15,
        use_process_pool: bool = True,
        cache_size: int = 100000,
    ):
        self.index_path = Path(index_path)
        self.indexer = indexer
        self.cache_size = cache_size
        
        # Slot weights (aggressive tuning - verb more important)
        self.slot_weights = {
            'SUBJ': 0.25,
            'VERB': 0.50,  # Verbs most discriminative
            'OBJ': 0.25,
        }
        
        # Load indexes
        self._load_indexes()
        
        # Create thread pool (for single/small queries)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="slot_query"
        )
        
        # Create process pool (for batch queries)
        if use_process_pool:
            import multiprocessing
            num_processes = min(8, multiprocessing.cpu_count())
            self.process_pool = ProcessPoolExecutor(
                max_workers=num_processes,
            )
            logger.info(f"  ProcessPool: {num_processes} workers")
        else:
            self.process_pool = None
        
        # Create caches
        self._init_caches()
    
    def _init_caches(self):
        """Initialize query and fusion caches."""
        @lru_cache(maxsize=self.cache_size)
        def _cache_query_results(query_hash: str, slot_k: int):
            # Actual expensive query
            query = self._unhash_query(query_hash)
            return self._search_uncached(query, slot_k)
        
        @lru_cache(maxsize=self.cache_size)
        def _cache_fusion_scores(query_hash: str, slot_k: int, fusion_method: str):
            # Cache fusion results (intermediate step)
            results_by_slot = self._cache_query_results(query_hash, slot_k)
            return self._fuse_results(results_by_slot, fusion_method)
        
        self._cache_query = _cache_query_results
        self._cache_fusion = _cache_fusion_scores
        self._query_map = {}  # hash → query text
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        slot_k: int = 500,  # 5× conservative!
        fusion_method: str = 'weighted_sum',
    ):
        """
        Single query search (uses ThreadPool).
        
        Args:
            slot_k: Candidates per slot (500 vs 100 conservative)
        """
        # Use cache
        query_hash = self._hash_query(query)
        self._query_map[query_hash] = query
        
        # Get fusion scores (cached)
        scores = self._cache_fusion(query_hash, slot_k, fusion_method)
        
        # Return top-k
        final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(score, self.documents[doc_id]) for doc_id, score in final_results]
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        slot_k: int = 500,
    ) -> List[List[Tuple]]:
        """
        Batch query search (uses ProcessPool for large batches).
        
        Automatically chooses:
        - ThreadPool: <10 queries
        - ProcessPool: ≥10 queries (better throughput)
        """
        if len(queries) < 10 or self.process_pool is None:
            # Small batch: use ThreadPool
            return self._batch_search_threaded(queries, top_k, slot_k)
        else:
            # Large batch: use ProcessPool
            return self._batch_search_process(queries, top_k, slot_k)
    
    def _batch_search_threaded(self, queries, top_k, slot_k):
        """ThreadPool batch search."""
        futures = [
            self.thread_pool.submit(self.search, q, top_k, slot_k)
            for q in queries
        ]
        return [f.result() for f in futures]
    
    def _batch_search_process(self, queries, top_k, slot_k):
        """ProcessPool batch search (for large batches)."""
        # Chunk queries for process pool
        chunk_size = 10
        chunks = [queries[i:i+chunk_size] for i in range(0, len(queries), chunk_size)]
        
        # Process chunks in parallel
        def process_chunk(chunk):
            # Each process has own retriever instance
            retriever = AggressiveFusedMultiSlotRetriever(
                self.index_path,
                self.indexer,
                use_process_pool=False,  # Don't nest process pools
            )
            return [retriever.search(q, top_k, slot_k) for q in chunk]
        
        futures = [
            self.process_pool.submit(process_chunk, chunk)
            for chunk in chunks
        ]
        
        # Flatten results
        all_results = []
        for f in futures:
            all_results.extend(f.result())
        
        return all_results
    
    def _query_slots_parallel_aggressive(self, query_slots, slot_k):
        """
        Query slots with aggressive parallelism.
        
        Uses larger thread pool (15 threads) to allow multiple
        concurrent requests.
        """
        def query_slot(slot, emb):
            if emb is None or slot not in self.slot_indexes:
                return slot, []
            
            # Normalize
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1).astype(np.float32)
            
            # Search with aggressive nprobe
            if hasattr(self.slot_indexes[slot], 'nprobe'):
                self.slot_indexes[slot].nprobe = 32  # 4× conservative
            
            scores, indices = self.slot_indexes[slot].search(emb, slot_k)
            
            # Map back to doc IDs
            results = [
                (float(scores[0][i]), int(self.slot_doc_ids[slot][indices[0][i]]))
                for i in range(len(indices[0]))
                if indices[0][i] >= 0
            ]
            
            return slot, results
        
        # Submit to thread pool
        futures = [
            self.thread_pool.submit(query_slot, slot, emb)
            for slot, emb in query_slots.items()
        ]
        
        # Gather results
        results_by_slot = {
            f.result()[0]: f.result()[1]
            for f in futures
            if f.result()[1]
        }
        
        return results_by_slot
    
    def get_stats(self):
        """Get performance statistics."""
        query_cache = self._cache_query.cache_info()
        fusion_cache = self._cache_fusion.cache_info()
        
        return {
            'query_cache': {
                'size': query_cache.currsize,
                'maxsize': query_cache.maxsize,
                'hit_rate': query_cache.hits / (query_cache.hits + query_cache.misses)
                    if (query_cache.hits + query_cache.misses) > 0 else 0.0,
            },
            'fusion_cache': {
                'size': fusion_cache.currsize,
                'maxsize': fusion_cache.maxsize,
                'hit_rate': fusion_cache.hits / (fusion_cache.hits + fusion_cache.misses)
                    if (fusion_cache.hits + fusion_cache.misses) > 0 else 0.0,
            },
            'thread_pool': {
                '_threads': len(self.thread_pool._threads),
                'max_workers': self.thread_pool._max_workers,
            },
        }
```

### 3. Throughput Benchmark

```python
# scripts/benchmark_throughput.py

import time
from concurrent.futures import ThreadPoolExecutor

def benchmark_throughput(retriever, num_queries=1000):
    """Measure queries per second."""
    
    # Generate test queries
    queries = generate_test_queries(num_queries)
    
    print(f"Benchmarking throughput with {num_queries} queries...")
    
    # Sequential baseline
    start = time.time()
    for q in queries[:100]:  # Sample
        retriever.search(q, top_k=10)
    sequential_time = (time.time() - start)
    sequential_qps = 100 / sequential_time
    
    print(f"Sequential: {sequential_qps:.0f} queries/sec")
    
    # Batch processing
    start = time.time()
    results = retriever.batch_search(queries, top_k=10)
    batch_time = (time.time() - start)
    batch_qps = num_queries / batch_time
    
    print(f"Batch (16 cores): {batch_qps:.0f} queries/sec")
    print(f"Speedup: {batch_qps / sequential_qps:.1f}×")
    
    return {
        'sequential_qps': sequential_qps,
        'batch_qps': batch_qps,
        'speedup': batch_qps / sequential_qps,
    }
```

## Expected Performance

### Throughput Comparison

| Mode | Conservative | Aggressive | Speedup |
|------|--------------|------------|---------|
| Sequential | 500 q/s | 1000 q/s | 2× |
| Batch (16 cores) | 1500 q/s | **5000+ q/s** | **3.3×** |

### Latency Comparison

| Query Type | Conservative | Aggressive | Improvement |
|------------|--------------|------------|-------------|
| Single (cold) | 1-2ms | 0.8-1.2ms | 1.5× faster |
| Single (cached) | - | **0.05ms** | 20-40× faster |
| Batch (10 queries) | 10-15ms | **5-8ms** | 2× faster |

### Memory Usage

| Component | Conservative | Aggressive | Your RAM |
|-----------|--------------|------------|----------|
| Slot indexes | 2.1GB | 3.6GB | 30GB |
| Caches | - | 1.3GB | |
| Thread pool | 0.1GB | 0.1GB | |
| Process pool (peak) | - | 3GB | |
| **Total (peak)** | **2.4GB** | **8GB** | **✅ 26% used** |

## Validation Test

```python
# tests/test_fused_aggressive.py

def test_throughput_improvement():
    """Test that aggressive mode improves throughput."""
    conservative = FusedMultiSlotRetriever(index_path, indexer, max_workers=3)
    aggressive = AggressiveFusedMultiSlotRetriever(index_path, indexer, max_workers=15)
    
    queries = [f"Query {i}" for i in range(100)]
    
    # Conservative
    start = time.time()
    conservative.batch_search(queries, top_k=10)
    time_conservative = time.time() - start
    
    # Aggressive
    start = time.time()
    aggressive.batch_search(queries, top_k=10)
    time_aggressive = time.time() - start
    
    speedup = time_conservative / time_aggressive
    assert speedup >= 2.0, f"Speedup {speedup:.1f}× less than 2×"

def test_cache_effectiveness():
    """Test that dual caching improves performance."""
    retriever = AggressiveFusedMultiSlotRetriever(index_path, indexer)
    
    query = "Kiu kreis Esperanton?"
    
    # First query (cold)
    start = time.time()
    retriever.search(query)
    cold_time = (time.time() - start) * 1000
    
    # Second query (cached)
    start = time.time()
    retriever.search(query)
    cached_time = (time.time() - start) * 1000
    
    # Should be at least 10× faster
    assert cached_time < cold_time / 10

def test_memory_under_10gb():
    """Test that peak memory stays under 10GB."""
    import psutil
    
    retriever = AggressiveFusedMultiSlotRetriever(index_path, indexer)
    
    # Run batch to trigger process pool
    queries = [f"Query {i}" for i in range(100)]
    retriever.batch_search(queries)
    
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    
    assert mem_gb < 10.0, f"Memory {mem_gb:.1f}GB exceeds 10GB"

def test_process_pool_batch():
    """Test that process pool handles large batches."""
    retriever = AggressiveFusedMultiSlotRetriever(index_path, indexer, use_process_pool=True)
    
    # Large batch (triggers process pool)
    queries = [f"Query {i}" for i in range(100)]
    results = retriever.batch_search(queries, top_k=10)
    
    # All queries should return results
    assert len(results) == 100
    for r in results:
        assert len(r) == 10  # top_k=10
```

## Demo Script

```python
# scripts/demo_fused_aggressive.py

def main():
    print("=" * 70)
    print("Aggressive FusedMultiSlot Demo (16 Cores, 30GB RAM)")
    print("=" * 70)
    print()
    
    # Load both configs
    conservative = FusedMultiSlotRetriever(index_path, indexer, max_workers=3)
    aggressive = AggressiveFusedMultiSlotRetriever(index_path, indexer, max_workers=15)
    
    print("Configuration:")
    print(f"  Conservative: 3 threads, slot_k=100")
    print(f"  Aggressive:   15 threads, slot_k=500, 100K cache")
    print()
    
    # Demo 1: Single query latency
    print("=" * 70)
    print("Demo 1: Single Query Latency")
    print("=" * 70)
    print()
    
    query = "Kiu kreis Esperanton?"
    
    # Conservative
    start = time.time()
    results_cons = conservative.search(query, top_k=10)
    time_cons = (time.time() - start) * 1000
    
    # Aggressive (cold)
    start = time.time()
    results_agg = aggressive.search(query, top_k=10)
    time_agg_cold = (time.time() - start) * 1000
    
    # Aggressive (hot)
    start = time.time()
    results_agg = aggressive.search(query, top_k=10)
    time_agg_hot = (time.time() - start) * 1000
    
    print(f"Conservative:       {time_cons:>6.1f}ms")
    print(f"Aggressive (cold):  {time_agg_cold:>6.1f}ms")
    print(f"Aggressive (hot):   {time_agg_hot:>6.1f}ms  ({time_cons/time_agg_hot:.0f}× faster!)")
    print()
    
    # Demo 2: Batch throughput
    print("=" * 70)
    print("Demo 2: Batch Throughput (100 queries)")
    print("=" * 70)
    print()
    
    queries = [f"Kiu kreis Esperanton numero {i}?" for i in range(100)]
    
    # Conservative
    start = time.time()
    conservative.batch_search(queries[:20], top_k=10)  # Sample
    time_cons_batch = (time.time() - start)
    qps_cons = 20 / time_cons_batch
    
    # Aggressive
    start = time.time()
    aggressive.batch_search(queries, top_k=10)
    time_agg_batch = (time.time() - start)
    qps_agg = 100 / time_agg_batch
    
    print(f"Conservative: {qps_cons:>6.0f} queries/sec")
    print(f"Aggressive:   {qps_agg:>6.0f} queries/sec ({qps_agg/qps_cons:.1f}× faster)")
    print()
    
    # Demo 3: Cache statistics
    print("=" * 70)
    print("Demo 3: Cache Performance")
    print("=" * 70)
    print()
    
    stats = aggressive.get_stats()
    
    print(f"Query cache:")
    print(f"  Size:     {stats['query_cache']['size']:,} / {stats['query_cache']['maxsize']:,}")
    print(f"  Hit rate: {stats['query_cache']['hit_rate']:.1%}")
    print()
    print(f"Fusion cache:")
    print(f"  Size:     {stats['fusion_cache']['size']:,} / {stats['fusion_cache']['maxsize']:,}")
    print(f"  Hit rate: {stats['fusion_cache']['hit_rate']:.1%}")
    print()
    
    # Demo 4: Memory usage
    print("=" * 70)
    print("Demo 4: Memory Usage")
    print("=" * 70)
    print()
    
    import psutil
    mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    
    print(f"Current memory: {mem_gb:.1f}GB / 30GB ({mem_gb/30*100:.0f}%)")
    print(f"Peak (with ProcessPool): ~8-9GB (30%)")
    print()
    
    print("✅ Demo complete!")
    print()
    print("Aggressive tuning summary:")
    print("  • Latency: 0.05ms cached, 0.8ms cold")
    print("  • Throughput: 5000+ queries/sec")
    print("  • Memory: 8GB peak (26% of your 30GB)")
    print("  • Parallelism: 15 threads + 8 processes")
    print("  • Cache: 100K queries (dual-layer)")

if __name__ == '__main__':
    main()
```

## Build Command

```bash
# Build aggressive multi-slot indexes
python scripts/build_aggressive_multislot.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/indexes/multislot_aggressive \
    --cores 16 \
    --fresh

# Benchmark throughput
python scripts/benchmark_throughput.py \
    --index data/indexes/multislot_aggressive \
    --num-queries 1000 \
    --cores 16
```

## Acceptance Criteria

- [ ] Aggressive slot indexes built (nlist=16K per slot)
- [ ] ThreadPool with 15 workers operational
- [ ] ProcessPool with 8 workers operational
- [ ] Dual caching working (query + fusion)
- [ ] Memory usage < 10GB peak confirmed
- [ ] Throughput ≥ 5000 q/s on batch benchmark
- [ ] Cached queries < 0.1ms latency
- [ ] Validation tests pass
- [ ] Demo script works end-to-end

## Related Tasks

- Extends: Task #17 (Conservative FusedMultiSlot)
- Alternative to: Task #16 (FAISS approaches)
- For: High-throughput production (chat applications, real-time search)
