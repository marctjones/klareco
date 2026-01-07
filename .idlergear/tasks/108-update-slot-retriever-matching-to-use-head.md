---
id: 108
title: Update slot retriever matching to use HEAD-weighted scoring
state: open
created: '2026-01-07T00:10:13.379244Z'
labels:
- enhancement
- retrieval
priority: high
---
## Goal

Modify slot retriever to use HEAD-weighted matching with cross-match penalty.

## Current Implementation

All slot retrievers use `_compute_slot_similarity()` which treats all roots equally.

## Proposed Implementation

```python
def _compute_slot_similarity(self, query_slots, doc_slots, is_question=True):
    """HEAD-weighted slot matching with cross-match penalty."""
    
    # Weight distribution (total = 1.0)
    weights = {
        'SUBJ_HEAD': 0.20,
        'SUBJ_MOD': 0.05,
        'VERB': 0.30,
        'OBJ_HEAD': 0.20,
        'OBJ_MOD': 0.05,
    }
    # Remaining 0.20 reserved for feature bonuses
    
    score = 0.0
    for slot, weight in weights.items():
        q_emb = query_slots.get(slot)
        d_emb = doc_slots.get(slot)
        if q_emb is not None and d_emb is not None:
            score += weight * cosine_sim(q_emb, d_emb)
        elif q_emb is None and d_emb is not None and is_question:
            score += weight * 0.8  # Partial bonus for questions
    
    # Cross-match penalty: query HEAD in doc MOD only
    if self._is_cross_match(query_slots, doc_slots):
        score *= 0.5
    
    return score

def _is_cross_match(self, query_slots, doc_slots):
    """Detect if query HEAD appears only as doc MODIFIER."""
    for role in ['SUBJ', 'OBJ']:
        q_head = query_slots.get(f'{role}_HEAD')
        d_head = doc_slots.get(f'{role}_HEAD')
        d_mod = doc_slots.get(f'{role}_MOD')
        
        if q_head is not None and d_mod is not None:
            head_to_mod_sim = cosine_sim(q_head, d_mod)
            head_to_head_sim = cosine_sim(q_head, d_head) if d_head is not None else 0
            
            # Query HEAD matches doc MOD but not doc HEAD
            if head_to_mod_sim > 0.8 and head_to_head_sim < 0.5:
                return True
    return False
```

## Files to Update

- `klareco/rag/slot_retriever.py` (base class)
- `klareco/rag/slot_retriever_faiss.py`
- `klareco/rag/slot_retriever_hnsw.py`
- `klareco/rag/slot_retriever_hybrid.py`
- `klareco/rag/slot_retriever_mmap.py`
- `klareco/rag/slot_retriever_multifaiss.py`
- `klareco/rag/slot_retriever_scann.py`
- `klareco/rag/slot_retriever_sqlite.py`

## Acceptance Criteria

- [ ] HEAD-weighted scoring implemented
- [ ] Cross-match penalty implemented
- [ ] All 8 retriever variants updated
- [ ] Unit tests for new matching logic
- [ ] Benchmark shows improvement

## Depends On

- #107 (slot indexer update) - needs new index format

## Related

Parent: #105 (HEAD/MODIFIER distinction in retrieval)
