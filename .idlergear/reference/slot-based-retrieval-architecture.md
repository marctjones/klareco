---
id: 1
title: Slot-Based Retrieval Architecture
created: '2026-01-06T22:26:00.179039Z'
updated: '2026-01-06T22:26:00.185348Z'
---
## Overview

ASTAware retriever uses slot-based retrieval to leverage Esperanto's deterministic AST structure for better question answering.

## Key Principle

**Don't average embeddings** - it destroys structural information.

Instead, compare slots separately:
- Query SUBJ ↔ Document SUBJ
- Query VERB ↔ Document VERB  
- Query OBJ ↔ Document OBJ

## Slot Extraction

From query AST:
```python
slots = {
    'SUBJ': embed(query_ast['subjekto']),  # Who/what is doing
    'VERB': embed(query_ast['verbo']),      # The action
    'OBJ': embed(query_ast['objekto']),     # What is acted upon
}
```

## Slot Similarity Scoring

```python
slot_weights = {'SUBJ': 0.4, 'VERB': 0.3, 'OBJ': 0.3}

for slot, weight in slot_weights.items():
    if query_slot and doc_slot:
        score += weight * cosine_sim(query_slot, doc_slot)
    elif query_slot is None and doc_slot:
        # Question missing slot → partial bonus (answer might have it)
        score += weight * 0.8
```

## Why This Matters

**Question**: "Kiu fondis Esperanton?" (Who founded Esperanto?)
- SUBJ: None (we're asking WHO)
- VERB: "fond" (founded)
- OBJ: "esperant" (Esperanto)

**Good answer**: "Zamenhof fondis Esperanton"
- SUBJ: "Zamenhof" → gets partial bonus (answers the WHO)
- VERB: "fond" → high similarity
- OBJ: "esperant" → high similarity

**Bad answer (with averaging)**: Any document mentioning founding + Esperanto separately would score equally, losing the structural match.

## Implementation Files

- `klareco/rag/ast_aware_retriever.py`:
  - `_extract_query_slots()` - Extract SUBJ/VERB/OBJ from AST
  - `_compute_slot_similarity()` - Compare slots with weighting
  - `_hnsw_prefilter()` - HNSW + slot reranking
  - `_keyword_prefilter()` - Keyword + slot reranking

## Related Issues

- #96: Fixed averaged embeddings → slot-based reranking
- #95: Fixed keyword prefilter to require ALL terms
- #97: Semantic keyword expansion (future)
- #98: Hybrid prefilter (future)
