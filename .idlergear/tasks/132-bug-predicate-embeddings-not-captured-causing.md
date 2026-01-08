---
id: 132
title: 'BUG: Predicate embeddings not captured causing massive embedding collisions'
state: open
created: '2026-01-07T15:27:04.939386Z'
labels:
- bug
- retrieval
- high-priority
priority: high
---
## Summary
Diagnostic investigation revealed that sentences like "La patro estas sana" and "La patro estas tajloro" get **identical embeddings** because the predicate (sana, tajloro) is not captured in any slot.

## Root Cause
1. The parser correctly parses predicates but puts them in `aliaj` (other elements), not OBJ
2. The slot-based indexer only combines SUBJ + VERB + OBJ to create `full_embedding`
3. `aliaj` elements (including predicates) are **ignored** in the full_embedding
4. Result: 17.5% of first 1000 docs have duplicate embeddings

## Evidence from Diagnostic
```
Doc 25: --La patro estas sana
  SUBJ: norm=11.3547  (patro)
  VERB: norm=3.6742   (estas)
  OBJ: None
  full_embedding: [0.0546, 0.0631, -0.1434, ...]

Doc 26: --La patro estas tajloro
  SUBJ: norm=11.3547  (patro) - IDENTICAL
  VERB: norm=3.6742   (estas) - IDENTICAL  
  OBJ: None
  full_embedding: [0.0546, 0.0631, -0.1434, ...] - IDENTICAL!
```

## Impact
- HNSW search returns wrong documents (self-retrieval success: 1/10)
- Recall@10 only 5.9% because queries match unrelated docs with same structure
- Any sentence with pattern "X estas Y" will collide with all other "X estas Z"

## Proposed Fix
1. Add PRED slot for predicate embeddings in copular sentences
2. Include `aliaj` embeddings in full_embedding computation
3. Rebuild slot index with fixed embedding calculation
4. Rebuild HNSW index from fixed slot index

## Files to Modify
- `scripts/index_slot_based.py` - Add PRED slot extraction
- `klareco/rag/slot_retriever_hybrid.py` - Add PRED slot handling
- `klareco/rag/ast_aware_retriever.py` - Include PRED in query matching

## Testing
After fix, verify:
- Self-retrieval success should be 10/10
- "La patro estas sana" ≠ "La patro estas tajloro" embeddings
- Recall@10 should improve significantly
