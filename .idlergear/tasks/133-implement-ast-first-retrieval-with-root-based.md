---
id: 133
title: Implement AST-first retrieval with root-based inverted index
state: closed
created: '2026-01-07T15:34:06.368379Z'
labels:
- enhancement
- retrieval
- architecture
priority: high
---
## Problem
Current retrieval averages word embeddings into sentence vectors, losing all AST structure. This causes:
- Embedding collisions (17.5% of docs have duplicates)
- No grammar-aware retrieval
- Slow keyword prefilter (grep on 30GB file)

## Solution: AST-First Retrieval

### 1. Build Root-Based Inverted Index
```python
# Index structure
{
  "fond": [
    {"doc_id": 123, "role": "verbo", "tempo": "pasinteco"},
    {"doc_id": 456, "role": "verbo", "tempo": "prezenco"},
  ],
  "esperant": [
    {"doc_id": 123, "role": "objekto", "kazo": "akuzativo"},
    {"doc_id": 789, "role": "subjekto"},
  ],
  ...
}
```

### 2. Query Processing
```
"Kiu kreis Esperanton?"
  → Parse to AST
  → Extract roots: ["kre", "esperant"]
  → Expand synonyms: ["kre", "fond", "establ", "iniciat", "esperant"]
  → Lookup in inverted index
  → Filter/boost by grammar (tempo=pasinteco, etc.)
```

### 3. Store Per-Root Embeddings (not averaged)
For semantic fallback when root not in index:
```json
{
  "doc_id": 123,
  "roots": {
    "zamenhof": {"embedding": [...], "role": "subjekto"},
    "kre": {"embedding": [...], "role": "verbo", "tempo": "pasinteco"},
    "esperant": {"embedding": [...], "role": "objekto"}
  }
}
```

## Implementation Steps
1. [ ] Create `RootInvertedIndex` class
2. [ ] Build indexer from unified_corpus.jsonl (has full ASTs)
3. [ ] Implement query-time synonym expansion via SemanticRelationDB
4. [ ] Add grammar-aware scoring
5. [ ] Fallback to embedding similarity for OOV roots
6. [ ] Benchmark vs current slot-based approach

## Benefits
- O(1) root lookup instead of O(n) grep
- No embedding collisions
- Grammar-aware retrieval
- Explainable: "matched doc contains 'fond' (synonym of 'kre') as verb"

## Files to Create/Modify
- NEW: `klareco/rag/root_inverted_index.py`
- NEW: `scripts/build_root_index.py`
- MODIFY: `klareco/rag/ast_aware_retriever.py` - use new index
