---
id: 131
title: Debug memory corruption in ASTAwareRetriever search
state: closed
created: '2026-01-07T05:17:09.720712Z'
labels:
- bug
- P1
priority: high
---
## Problem
When running `retriever.search()` in `ASTAwareRetriever`, Python crashes with:
```
corrupted size vs. prev_size
Aborted (core dumped)
```

## Context
- The retriever initializes successfully with SemanticPipeline and HNSW index
- Crash happens during `search()` when using "entity" strategy
- Likely caused by memory management issue in HNSW index or concurrent access

## Reproduction
```python
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
retriever = ASTAwareRetriever(index_path="data/indexes/slot_hybrid")
results = retriever.search("Kiu kreis Esperanton?")  # Crashes here
```

## Root Cause Analysis

### Environment
- hnswlib 0.8.0
- torch 2.9.1 
- numpy 2.2.6

### Code Path to Crash
```
search() 
  → _select_strategy() → "entity"
  → _search_entity_focused()
    → _hnsw_prefilter(query_ast, max_results=2000)
      → _embed_query_ast(query_ast)  ← New P0 code uses SemanticPipeline
      → hnsw_index.knn_query(query_emb, k=4000)  ← CRASH HERE
```

### **PRIMARY SUSPECT: Embedding Dimension Mismatch**

The crash is likely caused by a dimension mismatch between:
1. **SemanticPipeline embedding** (64 dimensions) - from root_embeddings model
2. **HNSW index** (128 dimensions) - built with hybrid embeddings

**Evidence:**
- Line 227-228: HNSW index gets dimension from first doc's `full_embedding` (128d hybrid)
- Line 262: `self.hnsw_index = hnswlib.Index(space='cosine', dim=embedding_dim)`
- SemanticPipeline (line 261): `sentence_embedding = torch.zeros(self.embedding_dim)` where `embedding_dim` is 64

**What happens:**
1. `_embed_query_ast()` calls SemanticPipeline which returns a **64-dim** embedding
2. Line 310: `query_emb = np.array([query_emb], dtype=np.float32)` - creates (1, 64) array
3. Line 316: `hnsw_index.knn_query(query_emb, k=hnsw_k)` - expects (1, **128**) array
4. hnswlib attempts to read beyond the 64-element buffer → memory corruption

### Secondary Suspects (Less Likely)

1. **Memory allocator conflict** (documented in code comment as "issue #88"):
   - Code attempts to mitigate by loading PyTorch models BEFORE hnswlib
   - But SemanticPipeline loads its own PyTorch models separately in `__init__`
   - Order: SemanticPipeline → HybridEmbeddings → hnswlib
   
2. **Tensor lifecycle issue**:
   - Line 542-543: `emb.numpy()` creates view of PyTorch tensor
   - If tensor is garbage collected, numpy array becomes invalid
   - But this would typically cause segfault, not heap corruption

3. **Thread safety in hnswlib**:
   - hnswlib 0.8.0 may have threading issues
   - But we're not using threads explicitly

## Recommended Fix (DO NOT IMPLEMENT YET)

Add dimension validation in `_embed_query_ast()`:
```python
# After line 548
if len(emb) != expected_dim:  # expected_dim should match HNSW index
    logger.warning(f"Dimension mismatch: query={len(emb)}, index={expected_dim}")
    return None  # Fall back to hybrid_embedder
```

Or better: ensure SemanticPipeline output matches HNSW index dimension by:
1. Padding 64d to 128d (with zeros for topical component)
2. Or rebuilding HNSW index with 64d embeddings
3. Or using HybridEmbeddings exclusively for HNSW queries

## Related
This crash blocks testing the prefilter improvements made in this session.
