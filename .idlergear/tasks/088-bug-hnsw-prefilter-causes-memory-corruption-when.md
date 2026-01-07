---
id: 88
title: 'Bug: HNSW prefilter causes memory corruption when loading SlotBasedIndexer'
state: closed
created: '2026-01-06T19:41:15.855092Z'
labels:
- bug
- retrieval
priority: high
---
## Problem

The AST-aware retriever crashes with memory corruption errors when HNSW prefilter is enabled. The crash occurs in `_load_prefilter()` when both `hnswlib.Index` and `SlotBasedIndexer` are initialized in the same process.

## Error Messages

```
free(): invalid size
corrupted double-linked list
Exit code 134 (Aborted/core dump)
```

## Root Cause Analysis

The crash occurs at `klareco/rag/ast_aware_retriever.py:174-191` in `_load_prefilter()`:

```python
# This loads fine
self.hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)
self.hnsw_index.load_index(str(hnsw_file))

# This crashes the process
self.query_embedder = SlotBasedIndexer(
    root_model_path=...,
    affix_model_path=...,
    topical_model_path=...,
    output_dir=...
)
```

**Potential causes:**
1. **Memory allocator conflict**: hnswlib uses C++ allocators, PyTorch uses its own memory allocator, causing heap corruption when both are active
2. **Threading issues**: hnswlib might set OMP_NUM_THREADS or use OpenMP parallelism conflicting with PyTorch's threading
3. **NumPy/hnswlib interaction**: hnswlib uses NumPy arrays internally, potential ABI mismatch

## Reproduction

```bash
# Crashes with memory corruption
python scripts/demo_ast_retriever.py "Kie naskiĝis Zamenhof?"

# Works fine (keyword prefilter only)
python scripts/demo_ast_retriever.py --no-prefilter "Kie naskiĝis Zamenhof?"
```

## Working Workaround

Use `--no-prefilter` flag which uses grep-based keyword prefilter instead of HNSW:
- Performance: ~0.5-2s per query (acceptable)
- Quality: Good results for keyword-matching queries

## Proposed Fix Options

### Option 1: Lazy embedding initialization (RECOMMENDED)
Only load SlotBasedIndexer when actually needed for query embedding:

```python
def _load_prefilter(self):
    # Load HNSW index only
    self.hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)
    self.hnsw_index.load_index(str(hnsw_file))
    
    # DON'T load SlotBasedIndexer here
    # Store paths for lazy loading
    self._embedder_config = {
        'root_model_path': Path("models/root_embeddings/best_model.pt"),
        ...
    }
    self.query_embedder = None

def _get_query_embedder(self):
    """Lazy-load query embedder on first use."""
    if self.query_embedder is None:
        self.query_embedder = SlotBasedIndexer(**self._embedder_config)
    return self.query_embedder
```

### Option 2: Pre-compute query embeddings separately
Run embedding in subprocess to avoid memory allocator conflicts.

### Option 3: Use simpler query embedding
Instead of full SlotBasedIndexer, use lightweight HybridEmbeddings directly without the full indexer infrastructure.

## Acceptance Criteria

- [ ] HNSW prefilter works without crashes
- [ ] Query latency < 3s for HNSW mode
- [ ] Memory usage stable across multiple queries
- [ ] Test with 10+ consecutive queries without crash

## Priority

P1 - Currently blocking HNSW prefilter usage, but keyword prefilter works as alternative.

## Related

- Keyword prefilter implemented in same file (works fine)
- scripts/demo_ast_retriever.py: --no-prefilter flag as workaround
