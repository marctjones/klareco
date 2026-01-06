---
id: 73
title: Update SlotBasedIndexer for dual embeddings
state: open
created: '2026-01-05T23:03:45.679508Z'
labels:
- enhancement
- 'priority: medium'
priority: medium
---
**Phase 5: Integration - Support 128d dual embeddings in indexer**

## Goal
Update SlotBasedIndexer to handle dual embeddings with mode selection.

## Implementation

**File:** `klareco/rag/slot_indexer.py` (MODIFY)

**Changes:**

1. **Add parameters:**
   - `use_dual: bool = True`
   - `embedding_mode: str = 'combined'`  # linguistic | topical | combined

2. **Update embedding dimension handling:**
   - Detect if model is dual or single
   - Set `self.embedding_dim` based on mode (64d or 128d)

3. **Update `embed_slots()` method:**
   - Pass `mode` parameter to root embedding forward()
   - Handle 64d (single mode) or 128d (combined mode)

4. **Backward compatibility:**
   - Default to single embeddings if model is old format
   - Auto-detect model type from checkpoint

**Key logic:**
```python
if self.use_dual:
    root_embs = self.root_model.forward(
        torch.tensor(root_indices),
        mode=self.embedding_mode
    )
else:
    root_embs = self.root_model.forward(torch.tensor(root_indices))
```

**Testing:**
- Test with single embedding model (backward compat)
- Test with dual embedding model (all modes)
- Test slot embedding shape (64d vs 128d)
- Integration test: build index with dual embeddings

## Acceptance Criteria
- [ ] Indexer supports dual embeddings
- [ ] Mode selection works (linguistic, topical, combined)
- [ ] Backward compatible with single embeddings
- [ ] Can build index with 128d embeddings
- [ ] All existing tests pass
- [ ] New tests for dual mode pass

## Dependencies
- **Blocks:** HNSW retriever (#74), AST retriever (#75)
- **Depends on:** DualRootEmbeddings (#68), trained model (#71)

## Estimated Effort
3-4 hours

## References
Design doc Section 4.1
