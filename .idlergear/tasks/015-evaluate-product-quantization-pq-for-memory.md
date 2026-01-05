---
id: 15
title: Evaluate Product Quantization (PQ) for memory reduction if scaling >10M docs
state: open
created: '2026-01-04T00:37:12.948719Z'
labels:
- enhancement
- faiss
- future
priority: low
---
## Context

Current memory usage for 4.2M documents:
- Full embeddings: 128d × 4 bytes × 4.2M = **2.1GB**
- FAISS index overhead: ~500MB
- Total: **~2.6GB**

This is manageable for current scale, but **Product Quantization (PQ)** could reduce memory 16× if we scale to 10M+ docs.

## What is Product Quantization?

**FAISS technique**: Lossy compression of vectors using codebooks

**How it works**:
1. Split 128d vector into M sub-vectors (e.g., M=64 → 64 sub-vectors of 2d each)
2. Quantize each sub-vector to N-bit code (e.g., 4 bits → 16 codebook entries)
3. Store only codes (M × N bits) instead of full vector

**Memory savings**:
- Full: 128d × 4 bytes = 512 bytes/vector
- PQ64x4: 64 × 0.5 bytes = **32 bytes/vector** (16× compression!)
- 4.2M docs: 2.1GB → **134MB**

## FAISS PQ Indexes

### Option 1: IVF + PQ (Recommended)
```python
# OPQ: Optimized Product Quantization with rotation
m = 64  # Sub-quantizers (typically d/2)
nbits = 4  # Bits per code (4 bits = 16 codebook entries)

# Index factory string
index = faiss.index_factory(
    dim,
    f"IVF65536_HNSW32,PQ{m}x{nbits}fsr",
    faiss.METRIC_INNER_PRODUCT
)

# Train on sample
index.train(embeddings[:1000000])  # Need 30×nlist to 256×nlist samples

# Add vectors
index.add(embeddings)
```

**Features**:
- `PQ64x4`: 64 sub-quantizers, 4-bit codes
- `fsr`: FastScan + Re-ranking for better accuracy
- IVF partitioning + PQ compression

### Option 2: Pure PQ (No IVF)
```python
# Simpler but slower
index = faiss.index_factory(dim, f"PQ{m}x{nbits}")
index.train(embeddings[:100000])
index.add(embeddings)
```

## Expected Tradeoffs

| Index Type | Memory | Latency | Recall | Use When |
|------------|--------|---------|--------|----------|
| **IndexIVFFlat** (current) | 2.1GB | 5ms | 85% | <10M docs, memory OK |
| **IVF_HNSW,PQ64x4fsr** | 134MB | 6-8ms | 80-83% | 10M-100M docs, memory critical |
| **IVF_HNSW,PQ64x8** | 268MB | 7-10ms | 82-85% | 10M-100M, better accuracy |

**Accuracy loss**: ~2-5% recall drop from quantization error

## When to Use PQ

**Use PQ if**:
- ✅ Scaling beyond 10M documents
- ✅ Memory is a bottleneck (embedding in RAM)
- ✅ Can tolerate 2-5% recall loss

**Skip PQ if**:
- ❌ <10M documents (current: 4.2M)
- ❌ Memory is not constrained
- ❌ Need maximum accuracy

## Current Recommendation

**For Klareco M2 milestone**: **Skip PQ for now**

**Reasons**:
1. Current scale (4.2M docs) is manageable at 2.6GB
2. M2 goal is 80% recall - can't afford 2-5% loss
3. Other optimizations (HNSW, weighted fusion) give better accuracy/speed

**Revisit PQ when**:
- Scaling to 10M+ documents
- Memory becomes bottleneck
- Accuracy target is met (>80% recall) and can afford small loss

## Implementation Plan (If Needed)

### Phase 1: Benchmark PQ vs Flat

```python
# scripts/benchmark_pq_compression.py

indexes = {
    'Flat': faiss.IndexIVFFlat(...),
    'PQ64x4': faiss.index_factory(dim, "IVF65K_HNSW32,PQ64x4fsr"),
    'PQ64x8': faiss.index_factory(dim, "IVF65K_HNSW32,PQ64x8fsr"),
}

for name, index in indexes.items():
    # Train if needed
    if hasattr(index, 'train'):
        index.train(train_data)
    
    # Add vectors
    index.add(embeddings)
    
    # Measure
    memory = index.sa_code_size() if hasattr(index, 'sa_code_size') else None
    results = benchmark(index, queries)
    
    print(f"{name}: Memory={memory}MB, Recall={results.recall:.2%}")
```

### Phase 2: Auto-select based on scale

```python
def create_index(embeddings: np.ndarray, use_pq: bool = None):
    """
    Auto-select index type based on dataset size.
    
    Args:
        use_pq: Force PQ on/off, or None for auto-select
    """
    n, dim = embeddings.shape
    
    # Auto-select
    if use_pq is None:
        use_pq = (n > 10_000_000)  # Use PQ for >10M docs
    
    if use_pq:
        # Product Quantization for large scale
        index = faiss.index_factory(dim, "IVF65536_HNSW32,PQ64x4fsr")
        logger.info(f"Using PQ index for {n:,} docs (memory: ~{n*32/1e6:.0f}MB)")
    else:
        # Flat for smaller scale
        index = faiss.IndexIVFFlat(...)
        logger.info(f"Using Flat index for {n:,} docs (memory: ~{n*dim*4/1e6:.0f}MB)")
    
    return index
```

## Files to Create (Future)

- `scripts/benchmark_pq_compression.py` - PQ vs Flat comparison
- `klareco/rag/slot_retriever_pq.py` - PQ-compressed retriever (if needed)

## References

- FAISS Guidelines: "Critical memory limits? Use OPQ_M__D_,...,PQ_M_x4fsr"
- FAISS PQ Tutorial: https://github.com/facebookresearch/faiss/wiki/FAQ#how-to-use-product-quantization
- PQ Paper: https://hal.inria.fr/inria-00514462v2/document

## Acceptance Criteria (If Implemented)

- [ ] PQ benchmark shows acceptable accuracy loss (<5%)
- [ ] Memory reduction confirmed (target: 16× compression)
- [ ] Latency acceptable (target: <10ms)
- [ ] Auto-select logic based on dataset size
- [ ] Documentation updated with PQ usage guidelines

## Priority

**Low** for M2 milestone - Revisit if:
1. Scaling beyond 10M docs
2. Memory becomes bottleneck
3. Other optimizations exhausted
