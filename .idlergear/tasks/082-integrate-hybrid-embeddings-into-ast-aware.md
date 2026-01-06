---
id: 82
title: Integrate hybrid embeddings into AST-aware retriever
state: open
created: '2026-01-06T05:44:20.085016Z'
labels:
- enhancement
- embeddings
- ast-aware
priority: high
---
Update AST-aware retriever to use hybrid embeddings for improved semantic matching.

**Prerequisites:**
- Task #78 complete (retrievers updated to use hybrid embeddings)

**Files to modify:**
- `klareco/rag/ast_aware_retriever.py` - Main AST-aware retriever

**Current implementation:**
The AST-aware retriever currently uses linguistic-only embeddings for the semantic scoring phase. This limits its ability to match proper nouns and topical content.

**Changes needed:**
1. Update embedding model initialization to use HybridEmbeddings:
```python
self.embedding_model = HybridEmbeddings.from_checkpoints(
    linguistic_checkpoint='models/root_embeddings/best_model.pt',
    topical_checkpoint='models/topical_embeddings/best_model.pt',
    default_mode='hybrid'
)
```

2. Update embedding dimension handling (64d → 128d)

3. Verify semantic scoring works with hybrid embeddings

4. Test with queries containing proper nouns (e.g., "Kio estas Parizo?")

**Expected improvements:**
- Better matching on proper noun queries
- Improved topical clustering
- More robust semantic scoring across diverse query types

**Success criteria:**
- AST-aware retriever loads hybrid embeddings
- Tests pass with 128d embeddings
- Benchmark shows improved recall on proper noun queries
- No regression on linguistic queries
