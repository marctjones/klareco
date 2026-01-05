# Retriever Parameter Tuning Guide

All slot-based retrievers have tunable parameters that affect the speed/accuracy tradeoff. This guide explains what parameters are available, when to tune them, and their impact on performance.

## Quick Reference

| Parameter Type | When to Set | Impact |
|----------------|-------------|--------|
| **Search-time** | Every query | Speed ↔ Accuracy tradeoff |
| **Load-time** | When creating retriever | Search behavior |
| **Build-time** | When building index | Index size, build time, accuracy ceiling |

---

## 1. Search-Time Parameters (Query-Level)

These parameters can be adjusted per query for fine-grained control.

### Common to All Retrievers

#### `top_k` (default: 10)
- **What**: Number of final results to return
- **Range**: 1-1000+
- **Impact**:
  - Higher = More results, slightly slower
  - Lower = Faster, fewer results
- **When to tune**:
  - `top_k=5`: Quick preview
  - `top_k=20`: Standard search
  - `top_k=100`: Comprehensive results

```python
results = retriever.search(query, top_k=20)
```

---

### HNSW-Specific

#### `hnsw_top_n` (default: 500)
- **What**: Number of candidates from HNSW pre-filtering (Stage 0)
- **Range**: 50-2000
- **Impact**:
  - **Higher**: Better recall, slower (more slot reranking)
  - **Lower**: Faster, may miss relevant docs
- **Guideline**: Should be 5-50× larger than `top_k`

#### `slot_top_n` (default: 100)
- **What**: Candidates after slot reranking (Stage 1)
- **Range**: 10-500
- **Impact**:
  - **Higher**: Better quality top-k, slightly slower
  - **Lower**: Faster, may degrade quality
- **Guideline**: Should be 5-20× larger than `top_k`

#### `slot_weight` (default: 0.6)
- **What**: Weight for slot-based similarity
- **Range**: 0.0-1.0
- **Impact**:
  - **Higher**: Favor grammatical role matching
  - **Lower**: Favor overall semantic similarity

#### `hnsw_weight` (default: 0.4)
- **What**: Weight for HNSW similarity
- **Range**: 0.0-1.0
- **Constraint**: `slot_weight + hnsw_weight` should equal 1.0

**Example configurations**:

```python
# Fast mode (sacrifice some accuracy)
results = retriever.search(
    query,
    top_k=10,
    hnsw_top_n=100,   # 10x top_k
    slot_top_n=50,    # 5x top_k
)

# Balanced (default)
results = retriever.search(
    query,
    top_k=10,
    hnsw_top_n=500,   # 50x top_k
    slot_top_n=100,   # 10x top_k
)

# High accuracy mode
results = retriever.search(
    query,
    top_k=10,
    hnsw_top_n=2000,  # 200x top_k
    slot_top_n=500,   # 50x top_k
)
```

---

### ScaNN-Specific

Same as HNSW but with `scann_*` prefix:

- `scann_top_n` (default: 500)
- `slot_top_n` (default: 100)
- `slot_weight` (default: 0.6)
- `scann_weight` (default: 0.4)

---

### Hybrid FAISS-Specific

- `faiss_top_n` (default: 500)
- `slot_weight` (default: 0.6)
- `full_weight` (default: 0.4)

---

### MultiFAISS-Specific

#### `slot_top_n` (default: 200)
- **What**: Candidates per slot from FAISS
- **Range**: 50-1000
- **Impact**: Higher = better recall, slower

#### `merge_strategy` (default: 'union')
- **What**: How to combine results from 3 slot indexes
- **Options**:
  - `'union'`: Take all unique results from all slots
  - `'intersection'`: Only docs appearing in multiple slots
  - `'vote'`: Rank by number of slots matching

---

## 2. Load-Time Parameters (Retriever Creation)

These parameters are set when creating the retriever instance and affect all subsequent searches.

### HNSW Load-Time Parameters

#### `hnsw_ef_search` (default: 500)
- **What**: HNSW search depth (controls accuracy ceiling)
- **Range**: 10-2000
- **Impact**:
  - **Higher**: Better recall ceiling, slower
  - **Lower**: Faster, lower max recall
- **Build-time equivalent**: Set during index build
- **Note**: Can also override at load time

```python
retriever = HNSWSlotRetriever(
    index_path,
    indexer,
    hnsw_ef_search=1000,  # Higher accuracy ceiling
)
```

#### `slot_weights` (default: equal)
- **What**: Importance of each grammatical role
- **Default**: `{'SUBJ': 0.3, 'VERB': 0.4, 'OBJ': 0.3}`
- **Impact**:
  - Higher VERB weight: Favor action matching
  - Higher SUBJ/OBJ weight: Favor entity matching

```python
retriever = HNSWSlotRetriever(
    index_path,
    indexer,
    slot_weights={
        'SUBJ': 0.2,  # Lower subject importance
        'VERB': 0.6,  # Higher verb importance (action-focused)
        'OBJ': 0.2,
    }
)
```

### ScaNN Load-Time Parameters

Same slot_weights as HNSW, plus:

#### `scann_num_leaves_to_search` (default: 100)
- **What**: Number of clusters to search
- **Range**: 10-500
- **Impact**: Like `ef_search` for HNSW

---

## 3. Build-Time Parameters (Index Creation)

These parameters are set when building the index and cannot be changed without rebuilding.

### HNSW Build Parameters

Set in `scripts/build_hnsw_index.sh`:

#### `M` (default: 16)
- **What**: Number of connections per node in HNSW graph
- **Range**: 4-64
- **Impact**:
  - **Higher**: Better recall, larger index, slower build
  - **Lower**: Faster build/search, smaller index, lower recall
- **Typical values**:
  - `M=4`: Fast, low memory (~50MB for 4M docs)
  - `M=16`: Balanced (current default) (~180MB)
  - `M=32`: High accuracy (~400MB)
  - `M=64`: Maximum accuracy (~800MB)

#### `ef_construction` (default: 200)
- **What**: Build-time search depth
- **Range**: 50-500
- **Impact**:
  - **Higher**: Better index quality, much slower build
  - **Lower**: Faster build, lower max recall
- **Guideline**: `ef_construction ≥ M`
- **Build time impact**: Linear (2x ef_construction ≈ 2x build time)

**Example configurations**:

```python
# Fast build, good for experimentation (2-3 min)
M=8, ef_construction=100

# Balanced (current default, ~10 min)
M=16, ef_construction=200

# High quality (20-30 min)
M=32, ef_construction=400
```

### ScaNN Build Parameters

Set in `scripts/build_scann_index.sh`:

#### `num_leaves` (default: 2000)
- **What**: Number of clusters for tree partitioning
- **Range**: 100-10000
- **Impact**:
  - **Higher**: Better recall, larger index, slower build
  - **Lower**: Faster, smaller, lower recall
- **Guideline**: `sqrt(num_docs)` to `2*sqrt(num_docs)`
  - For 4.2M docs: sqrt = 2048, so 1000-4000 is reasonable

#### `dimensions_per_block` (default: 2)
- **What**: Quantization granularity
- **Range**: 1-8
- **Impact**:
  - **Lower**: Better compression, faster, slightly lower accuracy
  - **Higher**: Less compression, more accurate
- **Typical**: 2 is optimal for most use cases

#### `reorder_k` (default: 100)
- **What**: Number of candidates to rerank with exact scores
- **Range**: 50-500
- **Impact**: Higher = better accuracy, slightly slower

**Example configurations**:

```python
# Fast build, moderate accuracy (3-5 min)
num_leaves=1000, reorder_k=50

# Balanced (current default, ~7 min)
num_leaves=2000, reorder_k=100

# High accuracy (15-20 min)
num_leaves=4000, reorder_k=200
```

---

## 4. Parameter Tuning Workflows

### Scenario 1: Optimizing for Speed

**Goal**: Minimize latency, acceptable recall drop

**Adjustments**:
1. Search-time: Lower stage counts
   ```python
   retriever.search(query, top_k=5, hnsw_top_n=100, slot_top_n=20)
   ```
2. Load-time: Lower ef_search
   ```python
   HNSWSlotRetriever(index_path, indexer, hnsw_ef_search=200)
   ```
3. Build-time: Smaller index
   ```bash
   # Rebuild with M=8, ef_construction=100
   ```

**Expected**: 2-5x faster, 5-10% recall drop

---

### Scenario 2: Optimizing for Accuracy

**Goal**: Maximize recall, tolerate slower queries

**Adjustments**:
1. Search-time: Higher stage counts
   ```python
   retriever.search(query, top_k=10, hnsw_top_n=2000, slot_top_n=500)
   ```
2. Load-time: Higher ef_search
   ```python
   HNSWSlotRetriever(index_path, indexer, hnsw_ef_search=1000)
   ```
3. Build-time: Larger index
   ```bash
   # Rebuild with M=32, ef_construction=400
   ```

**Expected**: 2-5x slower, 5-10% recall gain

---

### Scenario 3: Testing Parameter Sensitivity

Use `compare_retrievers.py` to benchmark different configurations:

```bash
# Test different top_n values
for top_n in 100 500 1000; do
    python scripts/compare_retrievers.py \
        --index data/indexes/slot_full \
        --retrievers hnsw \
        --output results_top${top_n}.json
    # Modify retriever code to use top_n
done
```

---

## 5. Recommendations by Use Case

### Interactive Search UI
- **Priority**: Speed
- **Config**: `top_k=10, hnsw_top_n=200, slot_top_n=50`
- **Expected**: <30ms queries

### Research/Analysis
- **Priority**: Accuracy
- **Config**: `top_k=100, hnsw_top_n=2000, slot_top_n=500`
- **Expected**: 100-200ms queries, maximum recall

### Production API
- **Priority**: Balance
- **Config**: Default values (current)
- **Expected**: 50-100ms queries, 85-90% recall

### Experimentation
- **Priority**: Fast iteration
- **Config**: Build with `M=8, ef_construction=100`
- **Expected**: Quick builds, test different approaches

---

## 6. How to Modify Parameters

### Search-time (easiest)
Just pass parameters to `search()`:
```python
results = retriever.search(query, hnsw_top_n=1000, slot_top_n=200)
```

### Load-time (medium)
Pass parameters to retriever constructor:
```python
retriever = HNSWSlotRetriever(
    index_path,
    indexer,
    hnsw_ef_search=1000,
    slot_weights={'SUBJ': 0.2, 'VERB': 0.6, 'OBJ': 0.2}
)
```

### Build-time (requires rebuild)
Edit build script and rebuild index:
```bash
# Edit scripts/build_hnsw_index.sh
# Change: M=32, ef_construction=400
./scripts/build_hnsw_index.sh
```

---

## 7. Monitoring Impact

Use `compare_retrievers.py` to measure parameter changes:

```bash
# Baseline
python scripts/compare_retrievers.py --index data/indexes/slot_full --output baseline.json

# After tuning
python scripts/compare_retrievers.py --index data/indexes/slot_full --output tuned.json

# Compare
python scripts/visualize_benchmark_results.py baseline.json tuned.json
```

---

## 8. Parameter Impact Summary

| Parameter | Speed Impact | Accuracy Impact | Memory Impact |
|-----------|--------------|-----------------|---------------|
| `top_k` | Low | N/A | Low |
| `hnsw_top_n` | **High** | High | Low |
| `slot_top_n` | Medium | Medium | Low |
| `ef_search` | **High** | **High** | Low |
| `M` (build) | Medium | **High** | **High** |
| `ef_construction` | Build-time | **High** | Low |
| `slot_weights` | Low | Task-specific | Low |

**Bold** = Most impactful parameters to tune first

---

## Next Steps

1. Run baseline comparison:
   ```bash
   python scripts/compare_retrievers.py --index data/indexes/slot_full
   ```

2. Identify bottlenecks (speed vs accuracy)

3. Tune search-time parameters first (no rebuild needed)

4. If needed, tune build-time parameters and rebuild index

5. Benchmark and iterate

See also: `scripts/compare_retrievers_USAGE.md` for benchmarking guide.
