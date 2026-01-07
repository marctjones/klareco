---
id: 1
title: Retriever Evaluation Plan v2 - Hybrid Embeddings
created: '2026-01-06T20:56:12.876238Z'
updated: '2026-01-06T20:56:12.876251Z'
---
# Retriever Evaluation Plan v2 - Hybrid Embeddings

## Decision: Use slot_hybrid/ with 128d Dual Embeddings

We're evaluating using the newer **hybrid embeddings** approach:
- 64d linguistic embeddings (root morphology)
- 64d topical embeddings (semantic similarity)
- Combined 128d for richer representation

## Current State of slot_hybrid/

| Component | Status | Size |
|-----------|--------|------|
| slot_index.jsonl | ✅ Ready | 32GB (4.37M docs) |
| slot_index.offsets.npy | ✅ Ready | 35MB |
| hnsw/full_embeddings.hnsw | ✅ Ready | 2.9GB |
| faiss/ | ❌ Missing | - |
| multifaiss/ | ❌ Missing | - |
| mmap/ | ❌ Missing | - |
| slot_index.db (SQLite) | ❌ Missing | - |
| scann/ | ❌ Missing | - |

## Retrievers We Can Test Now

| Retriever | Can Test? | Notes |
|-----------|-----------|-------|
| **ASTAwareRetriever** | ✅ Yes | Uses HNSW prefilter + slot_index |
| **HNSWSlotRetriever** | ⚠️ Partial | Has HNSW but needs mmap/ for slot arrays |
| FAISSSlotRetriever | ❌ No | Needs faiss/ index |
| HybridFAISSMmapRetriever | ❌ No | Needs faiss/ + mmap/ |
| MultiFAISSSlotRetriever | ❌ No | Needs multifaiss/ |
| ScaNNSlotRetriever | ❌ No | Needs scann/ |
| SQLiteSlotRetriever | ❌ No | Needs slot_index.db |
| MemoryMappedSlotRetriever | ❌ No | Needs mmap/ |

## Options

### Option A: Build Missing Indexes for slot_hybrid/
Build FAISS, MultiFAISS, mmap, SQLite indexes for the hybrid embeddings.
- **Pro:** Can compare all 8 retrievers with hybrid embeddings
- **Con:** Hours of index building time

### Option B: Test What We Have Now
Run evaluation with just ASTAwareRetriever on slot_hybrid/.
- **Pro:** Immediate results
- **Con:** Can't compare retriever strategies

### Option C: Build Only Essential Indexes
Build mmap/ arrays (needed for HNSWSlotRetriever) + FAISS (popular baseline).
- **Pro:** Enables 4 retrievers: ASTAware, HNSW, FAISS, Hybrid
- **Con:** Still missing MultiFAISS, ScaNN, SQLite

## Recommended: Option C

Build these indexes for slot_hybrid/:
1. **mmap/ arrays** - Enables HNSWSlotRetriever full functionality
2. **faiss/ index** - Enables FAISSSlotRetriever + HybridFAISSMmapRetriever

This allows testing the 4 most important retrievers with hybrid embeddings.

## Index Building Commands

```bash
# Build mmap arrays from slot_index.jsonl
python scripts/build_mmap_arrays.py \
  --index-dir data/indexes/slot_hybrid

# Build FAISS index from slot_index.jsonl  
python scripts/build_faiss_index.py \
  --index-dir data/indexes/slot_hybrid
```

## Revised Evaluation Plan

### Phase 1: Verify ASTAwareRetriever Works with Hybrid

Quick sanity test:
```python
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
retriever = ASTAwareRetriever(
    index_path="data/indexes/slot_hybrid",
    use_prefilter=True,
)
results = retriever.search("Kiu fondis Esperanton?", top_k=10)
```

### Phase 2: Build Missing Indexes (if needed)

Only if we need to compare multiple retrievers.

### Phase 3: Retrieval Benchmark

Create proper retrieval-focused benchmark:
1. Select 50 questions requiring retrieval
2. Find gold documents in corpus
3. Run ASTAwareRetriever (and others if indexes built)
4. Measure Recall@1/5/10, MRR, latency

### Phase 4: Analysis

- Is hybrid embedding improving retrieval?
- What's the AST analysis overhead?
- Which question types benefit most?
