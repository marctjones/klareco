---
id: 74
title: Update HNSW retriever for dual embeddings
state: open
created: '2026-01-05T23:04:00.927954Z'
labels:
- enhancement
- 'priority: medium'
priority: medium
---
**Phase 5: Integration - Support dual embeddings in HNSW pre-filter**

## Goal
Update HNSWSlotRetriever to handle 128d embeddings and mode selection.

## Implementation

**File:** `klareco/rag/slot_retriever_hnsw.py` (MODIFY)

**Changes:**

1. **Add parameters:**
   - `embedding_mode: str = 'combined'`
   - `embedding_weights: Dict[str, float] = None`

2. **Rebuild HNSW index with 128d vectors:**
   - Detect embedding dimension from indexer
   - Build HNSW with correct dimension
   - Update ef_search parameter if needed

3. **Query embedding generation:**
   - Use same mode as indexing
   - Support weighted combination if mode='combined'

4. **Pass mode to indexer:**
   - Forward embedding_mode to SlotBasedIndexer
   - Ensure query and index use same mode

**Rebuild process:**
```bash
# Rebuild HNSW index with dual embeddings
python scripts/build_hnsw_index.sh \
  --index-dir data/indexes/slot_dual \
  --dual-embeddings \
  --mode combined
```

**Testing:**
- Build index with dual embeddings (128d)
- Query with linguistic mode
- Query with topical mode
- Query with combined mode
- Verify results make sense

## Acceptance Criteria
- [ ] HNSW index builds with 128d embeddings
- [ ] Query works with all modes
- [ ] Pre-filter returns correct results
- [ ] Performance acceptable (<10ms per query)
- [ ] Backward compatible with 64d indexes

## Dependencies
- **Blocks:** AST retriever (#75), benchmark (#76)
- **Depends on:** SlotIndexer update (#73)

## Estimated Effort
4-5 hours (including index rebuild time)

## References
Design doc Section 4.1

## Notes
Index rebuild will take ~2-3 hours for 4.3M docs
