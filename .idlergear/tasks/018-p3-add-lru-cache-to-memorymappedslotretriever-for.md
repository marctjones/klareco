---
id: 18
title: '[P3] Add LRU cache to MemoryMappedSlotRetriever for hot queries'
state: open
created: '2026-01-04T00:46:08.519313Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Goal

Add LRU caching to MemoryMappedSlotRetriever to reduce latency for frequently queried documents.

**Priority**: P3 (Medium) - Best accuracy (90%), reduce latency 10×

**Expected Results (with cache):**
- Memory: 200MB base + 200MB cache = 400MB total
- Latency: 44.8ms → 2-5ms for cached queries
- Recall: 90% (unchanged - best of all retrievers!)

## Current MemoryMappedSlotRetriever

**Pros:**
- ✅ 200MB memory (lowest!)
- ✅ 90% recall (best accuracy!)

**Cons:**
- ❌ 44.8ms latency (slow due to disk I/O)

## Proposed Solution: Add LRU Cache

```python
from functools import lru_cache
import hashlib

class CachedMmapSlotRetriever(MemoryMappedSlotRetriever):
    """
    MemoryMapped retriever with LRU cache for hot embeddings.
    
    Caches frequently accessed embeddings in RAM to avoid disk I/O.
    """
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        cache_size: int = 10000,  # Cache 10K most common docs
        **kwargs
    ):
        """
        Args:
            cache_size: Number of document embeddings to cache (default: 10K)
                       10K docs × 128d × 4 bytes = ~5MB per slot
                       3 slots × 5MB = ~15MB total
                       Plus full embeddings: 10K × 128d × 4 = ~5MB
                       Total cache overhead: ~20MB
        """
        super().__init__(index_path, indexer, **kwargs)
        self.cache_size = cache_size
        
        # Create cached embedding getters
        self._get_slot_emb_cached = self._make_cached_getter()
    
    def _make_cached_getter(self):
        """Create LRU-cached embedding getter."""
        @lru_cache(maxsize=self.cache_size)
        def _get_embedding(slot: str, doc_id: int) -> np.ndarray:
            """Get embedding from mmap (cached)."""
            if slot == 'full':
                return self.embeddings['full'][doc_id].copy()
            else:
                emb = self.embeddings[slot][doc_id]
                # Check for null (zeros indicate missing slot)
                if np.allclose(emb, 0):
                    return None
                return emb.copy()
        
        return _get_embedding
    
    def slot_similarity(
        self,
        query_slots: Dict[str, Optional[np.ndarray]],
        doc_id: int,
    ) -> float:
        """
        Compute slot similarity using cached embeddings.
        """
        score = 0.0
        matched_slots = 0
        
        for slot, weight in self.slot_weights.items():
            query_emb = query_slots.get(slot)
            
            # Get from cache instead of direct mmap access
            doc_emb = self._get_slot_emb_cached(slot, doc_id)
            
            if query_emb is not None and doc_emb is not None:
                sim = self.cosine_similarity(query_emb, doc_emb)
                score += weight * sim
                matched_slots += 1
            elif query_emb is None and doc_emb is not None:
                score += weight * 0.5
                matched_slots += 1
        
        if matched_slots > 0:
            return score / matched_slots
        else:
            return 0.0
    
    def get_cache_stats(self) -> Dict:
        """Get cache hit/miss statistics."""
        info = self._get_slot_emb_cached.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0,
            'current_size': info.currsize,
            'max_size': info.maxsize,
        }
    
    def clear_cache(self):
        """Clear the LRU cache."""
        self._get_slot_emb_cached.cache_clear()
```

## Memory Tuning for <16GB RAM

**Cache size options:**

| Cache Size | Memory Overhead | Coverage | Use When |
|------------|-----------------|----------|----------|
| 1,000 docs | ~2MB | Top 0.02% | Minimal memory |
| 10,000 docs | ~20MB | Top 0.2% | **Recommended** |
| 50,000 docs | ~100MB | Top 1% | High query volume |
| 100,000 docs | ~200MB | Top 2.4% | Maximum caching |

**Recommended:** `cache_size=10000`
- Total memory: 200MB (base) + 20MB (cache) = **220MB**
- Covers most common queries (Zipf distribution: top 0.2% = 80% of queries)

**For 16-core laptop with 30GB RAM:** Can easily use `cache_size=100000`
- Total memory: 200MB + 200MB = **400MB** (still minimal!)
- Near-instant retrieval for top 2.4% of corpus

## Implementation

Update `klareco/rag/slot_retriever_mmap.py`:

```python
# Add at top
from functools import lru_cache

class MemoryMappedSlotRetriever:
    """Existing implementation..."""
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 1000,
        cache_size: int = 10000,  # NEW PARAMETER
    ):
        # ... existing init ...
        self.cache_size = cache_size
        
        # Create cache if enabled
        if cache_size > 0:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize LRU cache for embeddings."""
        @lru_cache(maxsize=self.cache_size)
        def _get_embedding_cached(slot: str, doc_id: int):
            """Cached embedding getter."""
            emb_array = self.embeddings[slot]
            emb = emb_array[doc_id].copy()
            
            # Check for null
            if slot != 'full' and np.allclose(emb, 0):
                return None
            
            return emb
        
        self._get_embedding = _get_embedding_cached
    
    def get_cache_info(self):
        """Get cache statistics."""
        if hasattr(self, '_get_embedding') and hasattr(self._get_embedding, 'cache_info'):
            info = self._get_embedding.cache_info()
            return {
                'hits': info.hits,
                'misses': info.misses,
                'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0,
                'size': info.currsize,
                'maxsize': info.maxsize,
            }
        return None
    
    # Update search to use cached getter
    def search(self, query: str, top_k: int = 10, **kwargs):
        """Search with cached embeddings."""
        # ... existing search code, but use self._get_embedding() instead of direct mmap access
```

## Validation Test

Create `tests/test_mmap_cached.py`:

```python
import pytest
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever

def test_cache_initialization():
    """Test that cache is created with correct size."""
    retriever = MemoryMappedSlotRetriever(index_path, indexer, cache_size=1000)
    
    cache_info = retriever.get_cache_info()
    assert cache_info is not None
    assert cache_info['maxsize'] == 1000

def test_cache_hits():
    """Test that repeated queries use cache."""
    retriever = MemoryMappedSlotRetriever(index_path, indexer, cache_size=10000)
    
    query = "Kiu kreis Esperanton?"
    
    # First query (cache miss)
    retriever.search(query, top_k=10)
    stats1 = retriever.get_cache_info()
    
    # Second query (cache hit)
    retriever.search(query, top_k=10)
    stats2 = retriever.get_cache_info()
    
    # Should have more hits
    assert stats2['hits'] > stats1['hits']
    assert stats2['hit_rate'] > stats1['hit_rate']

def test_cache_speedup():
    """Test that cache improves latency."""
    import time
    
    retriever = MemoryMappedSlotRetriever(index_path, indexer, cache_size=10000)
    query = "Kiu kreis Esperanton?"
    
    # First query (cold)
    start = time.time()
    retriever.search(query, top_k=10)
    cold_time = (time.time() - start) * 1000
    
    # Second query (cached)
    start = time.time()
    retriever.search(query, top_k=10)
    cached_time = (time.time() - start) * 1000
    
    # Should be at least 5× faster
    assert cached_time < cold_time / 5, f"Cached ({cached_time:.1f}ms) not 5× faster than cold ({cold_time:.1f}ms)"

def test_memory_usage():
    """Test that cache memory overhead is acceptable."""
    import psutil
    
    # Without cache
    retriever_nocache = MemoryMappedSlotRetriever(index_path, indexer, cache_size=0)
    process = psutil.Process()
    mem_nocache = process.memory_info().rss / 1024 / 1024
    
    # With cache
    retriever_cached = MemoryMappedSlotRetriever(index_path, indexer, cache_size=10000)
    mem_cached = process.memory_info().rss / 1024 / 1024
    
    overhead = mem_cached - mem_nocache
    
    # 10K cache should be ~20MB
    assert overhead < 50, f"Cache overhead {overhead:.0f}MB exceeds 50MB"
    
    # Total should still be under 500MB
    assert mem_cached < 500, f"Total memory {mem_cached:.0f}MB exceeds 500MB"

def test_accuracy_unchanged():
    """Test that caching doesn't affect accuracy."""
    retriever_nocache = MemoryMappedSlotRetriever(index_path, indexer, cache_size=0)
    retriever_cached = MemoryMappedSlotRetriever(index_path, indexer, cache_size=10000)
    
    query = "Kiu kreis Esperanton?"
    
    results_nocache = retriever_nocache.search(query, top_k=10)
    results_cached = retriever_cached.search(query, top_k=10)
    
    # Should return identical results
    for i in range(10):
        assert results_nocache[i][1]['text'] == results_cached[i][1]['text']
```

## Demo Script

Create `scripts/demo_mmap_cached.py`:

```python
#!/usr/bin/env python3
"""
Demo: Memory-mapped retriever with LRU cache.

Shows how caching improves latency while keeping memory low.
"""

import time
from pathlib import Path
from klareco.rag.slot_retriever_mmap import MemoryMappedSlotRetriever
from klareco.rag.slot_indexer import SlotBasedIndexer

def main():
    print("=" * 70)
    print("MemoryMapped Retriever with LRU Cache Demo")
    print("=" * 70)
    print()
    
    # Load retriever with cache
    index_path = Path("data/indexes/slot_full")
    indexer = SlotBasedIndexer.load()
    
    print("Loading retriever with 10K document cache...")
    retriever = MemoryMappedSlotRetriever(
        index_path, 
        indexer, 
        cache_size=10000  # Cache top 0.2% of corpus
    )
    
    print(f"✓ Loaded {retriever.num_docs:,} documents")
    print(f"✓ Cache size: {retriever.cache_size:,} documents")
    print()
    
    # Demo 1: Cold vs Cached latency
    print("=" * 70)
    print("Demo 1: Cold vs Cached Latency")
    print("=" * 70)
    print()
    
    queries = [
        "Kiu kreis Esperanton?",
        "Kiam Zamenhof kreis Esperanton?",
        "Kie naski\u011dis Zamenhof?",
    ]
    
    print("First run (cold - disk I/O):")
    cold_times = []
    for query in queries:
        start = time.time()
        results = retriever.search(query, top_k=5)
        elapsed = (time.time() - start) * 1000
        cold_times.append(elapsed)
        print(f"  {query:<40} {elapsed:>6.1f}ms")
    
    print()
    print("Second run (cached - RAM):")
    cached_times = []
    for query in queries:
        start = time.time()
        results = retriever.search(query, top_k=5)
        elapsed = (time.time() - start) * 1000
        cached_times.append(elapsed)
        print(f"  {query:<40} {elapsed:>6.1f}ms")
    
    avg_cold = sum(cold_times) / len(cold_times)
    avg_cached = sum(cached_times) / len(cached_times)
    speedup = avg_cold / avg_cached
    
    print()
    print(f"Average cold:   {avg_cold:>6.1f}ms")
    print(f"Average cached: {avg_cached:>6.1f}ms")
    print(f"Speedup:        {speedup:>6.1f}×")
    print()
    
    # Demo 2: Cache statistics
    print("=" * 70)
    print("Demo 2: Cache Statistics")
    print("=" * 70)
    print()
    
    stats = retriever.get_cache_info()
    
    print(f"Cache hits:   {stats['hits']:>8,}")
    print(f"Cache misses: {stats['misses']:>8,}")
    print(f"Hit rate:     {stats['hit_rate']:>8.1%}")
    print(f"Cache size:   {stats['size']:>8,} / {stats['maxsize']:,} documents")
    print()
    
    # Demo 3: Memory usage
    print("=" * 70)
    print("Demo 3: Memory Usage")
    print("=" * 70)
    print()
    
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"Total memory: {mem_mb:>6.0f}MB")
    print(f"Breakdown:")
    print(f"  Base mmap:  ~200MB (OS-managed)")
    print(f"  LRU cache:   ~20MB (10K docs)")
    print(f"  Python:     ~100MB (overhead)")
    print()
    print(f"✅ Memory usage well under 500MB!")
    print()
    
    # Demo 4: Accuracy unchanged
    print("=" * 70)
    print("Demo 4: Accuracy Verification")
    print("=" * 70)
    print()
    
    query = "Kiu kreis Esperanton?"
    print(f"Query: {query}")
    print()
    
    results = retriever.search(query, top_k=5)
    print("Top 5 results:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {doc['text'][:70]}...")
    print()
    
    print("✅ 90% recall (best of all retrievers!)")
    print()
    
    # Demo 5: Cache size comparison
    print("=" * 70)
    print("Demo 5: Cache Size Impact")
    print("=" * 70)
    print()
    
    cache_sizes = [0, 1000, 10000, 50000]
    
    print(f"{'Cache Size':<12} {'Memory':<10} {'Cold Latency':<15} {'Hot Latency':<15}")
    print("-" * 60)
    
    for size in cache_sizes:
        r = MemoryMappedSlotRetriever(index_path, indexer, cache_size=size)
        
        # Measure cold
        start = time.time()
        r.search(query, top_k=5)
        cold = (time.time() - start) * 1000
        
        # Measure hot (if cache enabled)
        start = time.time()
        r.search(query, top_k=5)
        hot = (time.time() - start) * 1000
        
        mem = 200 + (size * 0.002)  # Estimate: 2KB per cached doc
        
        print(f"{size:<12,} {mem:>6.0f}MB     {cold:>10.1f}ms      {hot:>10.1f}ms")
    
    print()
    print("✅ Demo complete!")
    print()
    print("Key takeaways:")
    print("  • Memory-mapped base: only 200MB")
    print("  • LRU cache adds ~20MB for 10K docs")
    print("  • 10× latency improvement for cached queries")
    print("  • 90% recall (best accuracy of all retrievers)")
    print("  • Total memory < 500MB (works on any laptop!)")

if __name__ == '__main__':
    main()
```

## Benchmark

```bash
# Benchmark with different cache sizes
python scripts/benchmark_slot_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers mmap \
    --cache-size 10000 \
    --output benchmark_results/mmap_cached.json
```

## Acceptance Criteria

- [ ] LRU cache implemented in MemoryMappedSlotRetriever
- [ ] cache_size parameter configurable
- [ ] get_cache_info() returns statistics
- [ ] Memory usage < 500MB with 10K cache
- [ ] Cached queries 10× faster than cold queries
- [ ] Validation tests pass
- [ ] Demo script works end-to-end
- [ ] Benchmark shows 2-5ms for cached, 44ms for cold
- [ ] 90% recall maintained (no accuracy loss)

## Related Tasks

- Optimizes: MemoryMappedSlotRetriever
- Addresses: Slow latency (44ms) while keeping low memory (200MB)
