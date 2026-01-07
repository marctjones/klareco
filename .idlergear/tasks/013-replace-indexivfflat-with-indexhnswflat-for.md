---
id: 13
title: Replace IndexIVFFlat with IndexHNSWFlat for better accuracy
state: closed
created: '2026-01-04T00:37:10.856296Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Problem

Current `FAISSSlotRetriever` uses `IndexIVFFlat`:
- Partitions space into Voronoi cells (coarse quantization)
- Searches subset of cells (controlled by nprobe)
- **Can miss exact matches** if they're in unsearched cells

**FAISS Guidelines**:
> "No memory concerns? Use HNSW_M (M=4-64): Fast, accurate graph-based index"

## Why HNSW is Better

**Advantages**:
- ✅ **Better accuracy** - No quantization loss, graph-based search finds true nearest neighbors
- ✅ **Sub-linear search** - O(log N) graph traversal vs O(N/nlist) for IVF
- ✅ **No training required** - IVF needs training phase, HNSW doesn't
- ✅ **Runtime tunable** - efSearch parameter adjusts speed/accuracy tradeoff

**Tradeoffs**:
- ⚠️ **Higher memory** - (d×4 + M×2×4) bytes/vector vs d×4 for Flat
  - 128d: 768 bytes/vector (HNSW) vs 512 bytes/vector (Flat)
  - 4.2M docs: ~3GB vs ~2GB (+50% memory)
- ⚠️ **Slower build time** - But Klareco builds index offline, so not critical

## Proposed Solution

```python
def _create_faiss_index(self, embeddings):
    """Create HNSW index instead of IVF."""
    dim = embeddings.shape[1]
    
    # Create HNSW index
    M = 32  # Number of connections per layer (16-64 recommended, 32 is sweet spot)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    
    # Build-time parameter (higher = better accuracy, slower build)
    index.hnsw.efConstruction = 40  # Default is 40, can go up to 200
    
    # Add vectors (no training needed!)
    logger.info(f"Building HNSW index with M={M}, efConstruction={index.hnsw.efConstruction}")
    index.add(embeddings)
    
    return index
```

### Search-time Tuning

```python
def search(self, query: str, top_k: int = 10, efSearch: int = 16):
    """
    Args:
        efSearch: HNSW search depth (higher = more accurate, slower)
            - 16: Fast, good accuracy (default)
            - 32: Better accuracy
            - 64: Excellent accuracy
            - 128: Near-exact search
    """
    # Set search parameter
    self.faiss_index.hnsw.efSearch = efSearch
    
    # Search
    scores, indices = self.faiss_index.search(query_emb, top_k)
```

## Expected Improvements

**Accuracy**:
- 85% → 90-95% recall (no quantization loss)
- Better ranking quality (MRR improvement)

**Speed**:
- 5.1ms → 3-8ms (depends on efSearch)
- Sub-linear scaling with dataset size

**Memory**:
- 683MB → ~1GB for 4.2M docs (+50%)
- Acceptable for current scale

## Parameter Guidelines

| efSearch | Accuracy | Latency | Use Case |
|----------|----------|---------|----------|
| 8 | Good | ~2ms | Fast bulk queries |
| 16 | Better | ~4ms | **Default (recommended)** |
| 32 | Excellent | ~6ms | High-accuracy queries |
| 64 | Near-perfect | ~10ms | Critical queries |

| M | Memory | Accuracy | Build Time |
|---|--------|----------|------------|
| 16 | Low | Good | Fast |
| 32 | Medium | **Better (recommended)** | Medium |
| 64 | High | Best | Slow |

## Implementation Plan

### Phase 1: New HNSWSlotRetriever class

```python
class HNSWSlotRetriever:
    """Slot-based retriever using HNSW index."""
    
    def _create_faiss_index(self, faiss_dir: Path):
        # Load embeddings
        embeddings = self._load_full_embeddings()
        
        # Create HNSW index
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, M=32, metric=faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        
        # Build index
        index.add(embeddings)
        
        # Save
        faiss.write_index(index, str(faiss_dir / "hnsw_index.bin"))
        
        # Save config
        config = {
            'index_type': 'HNSW',
            'M': 32,
            'efConstruction': 40,
            'default_efSearch': 16,
        }
        with open(faiss_dir / "hnsw_config.json", 'w') as f:
            json.dump(config, f, indent=2)
```

### Phase 2: Benchmark vs IVF

```python
# scripts/benchmark_hnsw_vs_ivf.py
retrievers = {
    'IVF': FAISSSlotRetriever(index_path, indexer),
    'HNSW_ef16': HNSWSlotRetriever(index_path, indexer, efSearch=16),
    'HNSW_ef32': HNSWSlotRetriever(index_path, indexer, efSearch=32),
}

for name, retriever in retrievers.items():
    results = benchmark(retriever, queries)
    print(f"{name}: Recall={results.recall:.2%}, Latency={results.latency:.1f}ms")
```

## Files to Create

- `klareco/rag/slot_retriever_hnsw.py` - HNSWSlotRetriever class
- `scripts/benchmark_hnsw_vs_ivf.py` - Comparison benchmark

## References

- FAISS Guidelines: "HNSW_M for no memory concerns"
- HNSW Paper: https://arxiv.org/abs/1603.09320
- FAISS HNSW docs: https://github.com/facebookresearch/faiss/wiki/Faster-search

## Acceptance Criteria

- [ ] HNSWSlotRetriever implemented
- [ ] efSearch parameter tunable at runtime
- [ ] Benchmark shows 90%+ recall
- [ ] Memory usage acceptable (<1.5GB for 4.2M docs)
- [ ] Latency competitive with IVF (target: <8ms at efSearch=16)
