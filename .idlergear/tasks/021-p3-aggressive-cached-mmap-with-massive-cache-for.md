---
id: 21
title: '[P3-AGGRESSIVE] Cached Mmap with massive cache for near-instant retrieval'
state: open
created: '2026-01-04T01:14:59.151939Z'
labels:
- enhancement
- M2
- P3-aggressive
- performance
- caching
priority: low
---
## Goal

Aggressive caching strategy for MemoryMappedSlotRetriever to achieve near-instant retrieval for most queries while keeping base memory minimal.

**Priority**: P3-AGGRESSIVE (Optional, for specific use cases)

**Conservative tuning (Task #18):**
- Memory: 200MB base + 20MB cache = 220MB total
- Cache size: 10K documents (0.2% of corpus)
- Latency: 2-5ms cached, 44ms cold
- Cache hit rate: ~60% (Zipf distribution)
- Recall: 90%

**Aggressive tuning (this task):**
- Memory: 200MB base + **5GB cache** = 5.2GB total (still only 17% of 30GB!)
- Cache size: **2.5M documents** (60% of entire corpus!)
- Latency: **0.1ms** cached, 44ms cold
- Cache hit rate: **99%+** (caches majority of corpus)
- Recall: 90% (unchanged)

## Strategy: Predictive Warm-Up

**Insight:** With 30GB RAM, you can cache the **entire "hot" subset** of the corpus (top 60%).

```python
AGGRESSIVE_CACHE_CONFIG = {
    # Cache sizing (aggressive for 30GB RAM)
    'cache_size': 2_500_000,      # 2.5M docs = 60% of 4.2M corpus
    
    # Predictive loading
    'preload_hot_docs': True,     # Load frequently accessed docs at startup
    'hot_doc_threshold': 0.6,     # Top 60% by access frequency
    
    # Multi-level caching
    'l1_cache_size': 100_000,     # Hot cache (100K most frequent)
    'l2_cache_size': 2_400_000,   # Warm cache (rest of top 60%)
    
    # Cache warming strategies
    'warmup_on_startup': True,    # Pre-cache common queries
    'warmup_queries': 10_000,     # Number of queries to pre-run
    'background_prefetch': True,  # Prefetch related docs
    
    # Memory management
    'max_memory_gb': 6,           # Hard limit (20% of 30GB)
    'eviction_policy': 'lfu',     # Least frequently used (vs LRU)
}
```

## Memory Breakdown

**Conservative (Task #18):**
```
Mmap base:       200MB  (OS-managed)
LRU cache:       20MB   (10K docs)
Total:           220MB
```

**Aggressive (this task):**
```
Mmap base:       200MB  (OS-managed, unchanged)
L1 cache (hot):  800MB  (100K docs × 4 slots × 128d × 4 bytes)
L2 cache (warm): 4.2GB  (2.4M docs × 4 slots × 128d × 4 bytes)
Warmup buffer:   200MB  (temporary during startup)
Python overhead: 100MB
────────────────────────
Total:           5.5GB  ✅ Only 18% of 30GB RAM!
```

**Cache coverage:**
- L1 (100K): Top 2.4% → 80% of queries (Zipf distribution)
- L2 (2.4M): Top 60% → 99% of queries
- Mmap fallback: Remaining 40% → <1% of queries

## Implementation

### 1. Multi-Level Cache Architecture

```python
# klareco/rag/cached_mmap_aggressive.py

from functools import lru_cache
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AggressiveCachedMmapRetriever(MemoryMappedSlotRetriever):
    """
    Memory-mapped retriever with aggressive multi-level caching.
    
    Cache hierarchy:
    - L1 (100K docs): Hot cache, instant access
    - L2 (2.4M docs): Warm cache, near-instant access
    - Mmap (rest): Cold storage, disk I/O fallback
    """
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        l1_size: int = 100_000,
        l2_size: int = 2_400_000,
        preload_hot: bool = True,
        **kwargs
    ):
        super().__init__(index_path, indexer, **kwargs)
        
        self.l1_size = l1_size
        self.l2_size = l2_size
        
        # Access frequency tracking
        self.access_counts = defaultdict(int)
        
        # Create multi-level caches
        self._init_multilevel_cache()
        
        # Preload hot documents if enabled
        if preload_hot:
            self._preload_hot_documents()
    
    def _init_multilevel_cache(self):
        """Initialize L1 and L2 caches."""
        # L1: Hot cache (LFU - least frequently used)
        @lru_cache(maxsize=self.l1_size)
        def _l1_cache(slot: str, doc_id: int):
            # Try L2 first
            return self._l2_cache(slot, doc_id)
        
        # L2: Warm cache (LRU - least recently used)
        @lru_cache(maxsize=self.l2_size)
        def _l2_cache(slot: str, doc_id: int):
            # Fallback to mmap
            emb_array = self.embeddings[slot]
            emb = emb_array[doc_id].copy()
            
            if slot != 'full' and np.allclose(emb, 0):
                return None
            
            return emb
        
        self._l1_cache = _l1_cache
        self._l2_cache = _l2_cache
    
    def _preload_hot_documents(self):
        """
        Pre-cache hot documents based on access patterns.
        
        Strategy:
        1. Analyze query logs (if available)
        2. Identify top 2.5M most-accessed docs
        3. Pre-load into L2 cache
        4. Warmup L1 with top 100K
        """
        logger.info("Preloading hot documents into cache...")
        
        # Option 1: Load from access log
        hot_doc_ids = self._load_hot_docs_from_log()
        
        # Option 2: Heuristic (no log available)
        if not hot_doc_ids:
            hot_doc_ids = self._estimate_hot_docs()
        
        # Preload L2 cache (2.4M docs)
        logger.info(f"  Loading {len(hot_doc_ids):,} docs into L2 cache...")
        for doc_id in hot_doc_ids[:self.l2_size]:
            for slot in ['SUBJ', 'VERB', 'OBJ', 'full']:
                self._l2_cache(slot, doc_id)
        
        # Preload L1 cache (100K docs)
        logger.info(f"  Loading {min(self.l1_size, len(hot_doc_ids)):,} docs into L1 cache...")
        for doc_id in hot_doc_ids[:self.l1_size]:
            for slot in ['SUBJ', 'VERB', 'OBJ', 'full']:
                self._l1_cache(slot, doc_id)
        
        logger.info("✓ Cache preloading complete")
        logger.info(f"  L1: {self._l1_cache.cache_info().currsize:,} / {self.l1_size:,}")
        logger.info(f"  L2: {self._l2_cache.cache_info().currsize:,} / {self.l2_size:,}")
    
    def _estimate_hot_docs(self):
        """
        Estimate hot documents using heuristics.
        
        Heuristics:
        - Docs with Fundamento/ReVo content (tier 0-3)
        - Docs with common words (high TF-IDF)
        - Docs with short sentences (faster to read)
        - Random sample for diversity
        """
        hot_ids = []
        
        # Priority 1: Authoritative sources (tier 0-3)
        for i, doc in enumerate(self.metadata):
            if doc.get('tier', 6) <= 3:
                hot_ids.append(i)
        
        logger.info(f"  Authoritative docs: {len(hot_ids):,}")
        
        # Priority 2: Common topic documents (simple heuristic: sample evenly)
        remaining = self.l2_size - len(hot_ids)
        if remaining > 0:
            # Sample from entire corpus
            import random
            sample_ids = random.sample(range(len(self.metadata)), min(remaining, len(self.metadata)))
            hot_ids.extend(sample_ids)
        
        return hot_ids[:self.l2_size]
    
    def _load_hot_docs_from_log(self):
        """Load hot documents from access log (if available)."""
        log_file = self.index_path / "access_log.jsonl"
        
        if not log_file.exists():
            return []
        
        # Count accesses per document
        doc_counts = defaultdict(int)
        
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                for doc_id in entry.get('retrieved_docs', []):
                    doc_counts[doc_id] += 1
        
        # Sort by frequency
        hot_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N doc IDs
        return [doc_id for doc_id, count in hot_docs[:self.l2_size]]
    
    def search(self, query: str, top_k: int = 10, **kwargs):
        """
        Search with multi-level caching.
        
        Cache lookup order:
        1. L1 cache (instant)
        2. L2 cache (near-instant)
        3. Mmap (disk I/O)
        """
        # Track access
        results = super().search(query, top_k, **kwargs)
        
        # Update access counts for cache promotion
        for _, doc in results:
            doc_id = doc.get('id')
            if doc_id is not None:
                self.access_counts[doc_id] += 1
        
        return results
    
    def get_cache_stats(self):
        """Get multi-level cache statistics."""
        l1_info = self._l1_cache.cache_info()
        l2_info = self._l2_cache.cache_info()
        
        total_hits = l1_info.hits + l2_info.hits
        total_misses = l1_info.misses + l2_info.misses
        total_accesses = total_hits + total_misses
        
        return {
            'l1': {
                'size': l1_info.currsize,
                'maxsize': l1_info.maxsize,
                'hits': l1_info.hits,
                'misses': l1_info.misses,
                'hit_rate': l1_info.hits / total_accesses if total_accesses > 0 else 0.0,
            },
            'l2': {
                'size': l2_info.currsize,
                'maxsize': l2_info.maxsize,
                'hits': l2_info.hits,
                'misses': l2_info.misses,
                'hit_rate': l2_info.hits / total_accesses if total_accesses > 0 else 0.0,
            },
            'combined': {
                'total_hit_rate': total_hits / total_accesses if total_accesses > 0 else 0.0,
                'cache_coverage': (l1_info.currsize + l2_info.currsize) / self.num_docs,
            },
            'top_docs': sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        }
    
    def warmup_from_queries(self, queries: List[str]):
        """
        Warm up cache by running common queries.
        
        Args:
            queries: List of common/expected queries
        """
        logger.info(f"Warming up cache with {len(queries)} queries...")
        
        for i, q in enumerate(queries):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(queries)}")
            
            self.search(q, top_k=10)
        
        stats = self.get_cache_stats()
        logger.info(f"✓ Warmup complete")
        logger.info(f"  Combined hit rate: {stats['combined']['total_hit_rate']:.1%}")
```

### 2. Query Log Analysis Tool

```python
# scripts/analyze_query_logs.py

def analyze_query_patterns(log_file: Path, output_file: Path):
    """
    Analyze query logs to identify hot documents.
    
    Generates:
    - Top N most-accessed documents
    - Access frequency distribution
    - Query patterns
    """
    doc_access_counts = defaultdict(int)
    query_counts = defaultdict(int)
    
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            
            # Track query
            query = entry.get('query')
            query_counts[query] += 1
            
            # Track retrieved docs
            for doc_id in entry.get('retrieved_docs', []):
                doc_access_counts[doc_id] += 1
    
    # Analyze distribution
    total_accesses = sum(doc_access_counts.values())
    sorted_docs = sorted(doc_access_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate coverage
    cumulative = 0
    for i, (doc_id, count) in enumerate(sorted_docs):
        cumulative += count
        coverage = cumulative / total_accesses
        
        if coverage >= 0.99:
            print(f"99% coverage: top {i+1:,} documents ({i+1/len(sorted_docs)*100:.1f}%)")
            break
    
    # Save hot docs
    hot_docs = [doc_id for doc_id, _ in sorted_docs[:2_500_000]]
    
    with open(output_file, 'w') as f:
        json.dump({
            'hot_documents': hot_docs,
            'total_queries': len(query_counts),
            'total_documents_accessed': len(doc_access_counts),
            'total_accesses': total_accesses,
        }, f)
    
    print(f"✓ Saved {len(hot_docs):,} hot documents to {output_file}")
```

### 3. Startup Warmup Script

```python
# scripts/warmup_cache.py

def warmup_cache_from_corpus(
    retriever: AggressiveCachedMmapRetriever,
    num_queries: int = 10_000,
):
    """
    Generate and run warmup queries from corpus.
    
    Strategy:
    - Extract common queries from Fundamento
    - Generate variations
    - Run all to populate cache
    """
    logger.info(f"Generating {num_queries} warmup queries...")
    
    # Load common question patterns
    patterns = [
        "Kiu {}?",
        "Kio estas {}?",
        "Kiam {}?",
        "Kie {}?",
        "Kial {}?",
        "Kiel {}?",
    ]
    
    # Sample topics from corpus
    topics = extract_common_topics(retriever.metadata, n=2000)
    
    # Generate queries
    queries = []
    for pattern in patterns:
        for topic in topics:
            queries.append(pattern.format(topic))
            if len(queries) >= num_queries:
                break
        if len(queries) >= num_queries:
            break
    
    # Run warmup
    retriever.warmup_from_queries(queries)
    
    return retriever
```

## Expected Performance

### Latency Distribution

| Cache Level | Docs Cached | Coverage | Latency | Memory |
|-------------|-------------|----------|---------|--------|
| L1 (hot) | 100K | 80% of queries | **0.05ms** | 800MB |
| L2 (warm) | 2.4M | 99% of queries | **0.5ms** | 4.2GB |
| Mmap (cold) | 1.8M | 1% of queries | 44ms | 200MB |
| **Total** | **4.2M** | **100%** | **~0.5ms avg** | **5.2GB** |

### Memory vs Latency Tradeoff

| Cache Size | Memory | Avg Latency | 99% Coverage |
|------------|--------|-------------|--------------|
| 10K (conservative) | 220MB | 25ms | ❌ No |
| 100K | 1GB | 10ms | ❌ No |
| 500K | 3GB | 3ms | ❌ No |
| **2.5M (aggressive)** | **5.2GB** | **0.5ms** | **✅ Yes** |

### Comparison with Other Approaches

| Approach | Memory | Avg Latency | Recall |
|----------|--------|-------------|--------|
| Optimized FAISS | 2.9GB | 3-4ms | 88-90% |
| Fused MultiSlot | 2.4GB | 1-2ms | 85-90% |
| **Aggressive Mmap** | **5.2GB** | **0.5ms** | **90%** |

**Best for:** Maximum accuracy + near-instant response

## Validation Test

```python
# tests/test_mmap_aggressive.py

def test_cache_coverage():
    """Test that 2.5M cache covers 99% of queries."""
    retriever = AggressiveCachedMmapRetriever(
        index_path,
        indexer,
        l1_size=100_000,
        l2_size=2_400_000,
    )
    
    # Run 1000 random queries
    queries = generate_test_queries(1000)
    
    for q in queries:
        retriever.search(q, top_k=10)
    
    stats = retriever.get_cache_stats()
    
    # Should have >99% hit rate
    assert stats['combined']['total_hit_rate'] >= 0.99

def test_memory_under_6gb():
    """Test that memory stays under 6GB."""
    import psutil
    
    retriever = AggressiveCachedMmapRetriever(
        index_path,
        indexer,
        l1_size=100_000,
        l2_size=2_400_000,
        preload_hot=True,
    )
    
    mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    
    assert mem_gb < 6.0, f"Memory {mem_gb:.1f}GB exceeds 6GB limit"

def test_sub_millisecond_cached():
    """Test that cached queries are sub-millisecond."""
    retriever = AggressiveCachedMmapRetriever(index_path, indexer)
    
    query = "Kiu kreis Esperanton?"
    
    # Prime cache
    retriever.search(query)
    
    # Measure cached latency
    import time
    latencies = []
    for _ in range(100):
        start = time.time()
        retriever.search(query)
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    
    # Should be sub-millisecond
    assert avg_latency < 1.0, f"Avg latency {avg_latency:.2f}ms exceeds 1ms"
```

## Demo Script

```python
# scripts/demo_mmap_aggressive.py

def main():
    print("=" * 70)
    print("Aggressive Cached Mmap Demo (5GB Cache, 99% Coverage)")
    print("=" * 70)
    print()
    
    # Load with aggressive caching
    print("Loading retriever with 2.5M document cache...")
    print("(This will take 2-3 minutes to preload...)")
    print()
    
    retriever = AggressiveCachedMmapRetriever(
        index_path,
        indexer,
        l1_size=100_000,
        l2_size=2_400_000,
        preload_hot=True,
    )
    
    print("✓ Loaded!")
    print()
    
    # Show stats
    stats = retriever.get_cache_stats()
    
    print("Cache statistics:")
    print(f"  L1 (hot):  {stats['l1']['size']:>8,} / {stats['l1']['maxsize']:,} docs")
    print(f"  L2 (warm): {stats['l2']['size']:>8,} / {stats['l2']['maxsize']:,} docs")
    print(f"  Coverage:  {stats['combined']['cache_coverage']:>8.1%} of corpus")
    print()
    
    # Demo 1: Latency distribution
    print("=" * 70)
    print("Demo 1: Latency Distribution Across Cache Levels")
    print("=" * 70)
    print()
    
    test_queries = generate_test_queries(100)
    
    latencies = {'l1': [], 'l2': [], 'cold': []}
    
    for q in test_queries:
        start = time.time()
        retriever.search(q)
        latency = (time.time() - start) * 1000
        
        # Classify by latency
        if latency < 0.2:
            latencies['l1'].append(latency)
        elif latency < 2:
            latencies['l2'].append(latency)
        else:
            latencies['cold'].append(latency)
    
    print(f"L1 cache (instant):     {len(latencies['l1'])} queries, avg {np.mean(latencies['l1']):.2f}ms")
    print(f"L2 cache (fast):        {len(latencies['l2'])} queries, avg {np.mean(latencies['l2']):.2f}ms")
    print(f"Cold (disk I/O):        {len(latencies['cold'])} queries, avg {np.mean(latencies['cold']):.2f}ms")
    print()
    print(f"Cache hit rate: {(len(latencies['l1']) + len(latencies['l2'])) / 100:.0%}")
    print()
    
    # Demo 2: Memory usage
    print("=" * 70)
    print("Demo 2: Memory Breakdown")
    print("=" * 70)
    print()
    
    import psutil
    mem_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    
    print(f"Total memory:     {mem_gb:.1f}GB / 30GB ({mem_gb/30*100:.0f}%)")
    print()
    print("Breakdown:")
    print(f"  Mmap base:      0.2GB  (OS-managed)")
    print(f"  L1 cache:       0.8GB  (100K docs)")
    print(f"  L2 cache:       4.2GB  (2.4M docs)")
    print(f"  Python:         0.1GB  (overhead)")
    print()
    
    # Demo 3: Comparison
    print("=" * 70)
    print("Demo 3: Conservative vs Aggressive")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<20} {'Conservative':<15} {'Aggressive':<15}")
    print("-" * 50)
    print(f"{'Cache size':<20} {'10K docs':<15} {'2.5M docs':<15}")
    print(f"{'Memory':<20} {'220MB':<15} {'5.2GB':<15}")
    print(f"{'Coverage':<20} {'~60%':<15} {'~99%':<15}")
    print(f"{'Avg latency':<20} {'25ms':<15} {'0.5ms':<15}")
    print(f"{'Speedup':<20} {'1×':<15} {'50×':<15}")
    print()
    
    print("✅ Demo complete!")
    print()
    print("Aggressive caching summary:")
    print("  • Cache: 2.5M docs (60% of corpus)")
    print("  • Memory: 5.2GB (17% of your 30GB)")
    print("  • Hit rate: 99% (nearly all queries cached)")
    print("  • Latency: 0.5ms average (50× faster than conservative)")
    print("  • Recall: 90% (best of all retrievers)")

if __name__ == '__main__':
    main()
```

## Build Command

```bash
# No index rebuild needed (uses existing mmap)

# Generate warmup queries
python scripts/generate_warmup_queries.py \
    --corpus data/corpus/unified_corpus.jsonl \
    --output data/warmup_queries.txt \
    --num 10000

# Warm up cache
python scripts/warmup_cache.py \
    --index data/indexes/slot_full \
    --queries data/warmup_queries.txt
```

## Acceptance Criteria

- [ ] Multi-level cache (L1 + L2) implemented
- [ ] Preloading hot documents on startup working
- [ ] Cache coverage ≥ 99% on benchmark
- [ ] Memory usage < 6GB confirmed
- [ ] Cached queries < 1ms latency
- [ ] Combined hit rate ≥ 99%
- [ ] Validation tests pass
- [ ] Demo script works end-to-end
- [ ] 90% recall maintained (no accuracy loss)

## Related Tasks

- Extends: Task #18 (Conservative Cached Mmap)
- Alternative to: Task #16, #17 (FAISS approaches)
- For: Ultra-low latency requirements (<1ms), maximum accuracy (90%)
