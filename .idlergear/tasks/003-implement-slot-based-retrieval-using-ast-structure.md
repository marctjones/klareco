---
id: 3
title: Implement slot-based indexing and retrieval (M2 priority)
state: open
created: '2026-01-02T07:48:46.443364Z'
labels:
- enhancement
- retrieval
- M2
priority: high
---
## Context

**VALIDATED 2026-01-02**: Testing with merged index proves sentence averaging is the problem, not corpus quality!

Query: "Kiu kreis Esperanton?" (Who created Esperanto?)
- ✅ Corpus HAS answer: 4,998 Zamenhof mentions including "Zamenhof kreis Esperanton"
- ❌ Results: "Apple kreis ĝin", "Li kreis altlernejon"
- **Root cause**: Averaging loses SUBJ/VERB/OBJ roles

Current retrieval uses simple sentence averaging, which loses important AST structure:
- Subject/verb/object roles not distinguished
- Negation not handled explicitly  
- Partial queries not supported well
- **Current recall: 35%** (M2 target: 80%)

## Proposal: Slot-Based Indexing

Store AST slots separately instead of averaging:

```python
# Current (wasteful):
{
    'embedding': mean([subj_vec, verb_vec, obj_vec, ...]),  # Lost structure!
    'text': "La hundo mordas la katon"
}

# Proposed (preserves structure):
{
    'slots': {
        'SUBJ': embed_node(ast['subjekto']),      # hund_vec
        'VERB': embed_node(ast['verbo']),         # mord_vec
        'OBJ': embed_node(ast['objekto']),        # kat_vec
        'MODIFIERS': [embed_node(m) for m in ast['aliaj']]
    },
    'features': {
        'negita': False,
        'tempo': 'prezenco',
        'fraztipo': 'deklaro',
        'modo': 'indikativo'
    },
    'full_embedding': mean(...),  # Fallback for complex queries
    'text': "La hundo mordas la katon",
    'source': {...}
}
```

## Two-Stage Retrieval Strategy

### Stage 1: Slot-Based Filtering (Fast, Structural)
```python
def slot_similarity(query_slots, doc_slots):
    """Compute weighted slot similarity."""
    score = 0.0
    weights = {'SUBJ': 0.3, 'VERB': 0.4, 'OBJ': 0.3}
    
    for slot, weight in weights.items():
        if query_slots.get(slot) and doc_slots.get(slot):
            score += weight * cosine(query_slots[slot], doc_slots[slot])
        elif not query_slots.get(slot):
            score += weight * 0.5  # Partial match bonus
    
    # Grammar feature matching
    if query_features['negita'] == doc_features['negita']:
        score *= 1.1  # Boost for negation match
    
    return score
```

### Stage 2: Full Embedding Reranking (Accurate)
```python
# Rerank top-K candidates from Stage 1 using full embeddings
final_score = 0.6 * slot_score + 0.4 * full_embedding_score
```

## Implementation Plan

### Phase 1: Indexer Extension
- [ ] Create `SlotBasedIndexer` class extending current indexer
- [ ] Add `embed_ast_node(node)` to extract slot embeddings
- [ ] Store both slot-based and full embeddings in metadata
- [ ] Test on small corpus subset (1K sentences)

### Phase 2: Retriever Implementation
- [ ] Create `SlotBasedRetriever` in `klareco/rag/retriever.py`
- [ ] Implement slot similarity scoring
- [ ] Implement two-stage retrieval pipeline
- [ ] Handle partial queries (missing slots)

### Phase 3: Integration & Evaluation
- [ ] Build slot-based index for authoritative corpus (18K sentences)
- [ ] Run M1 benchmark with slot-based retrieval
- [ ] Measure recall improvement vs baseline
- [ ] If successful, rebuild full index (4.4M sentences)

## Benefits

1. **Partial query support**: "Kiu mordas?" matches any sentence with `mord` as verb
2. **Role-aware matching**: Distinguishes subject vs object
3. **Grammar-aware**: Negation, tense, mood as explicit features
4. **No new training**: Uses existing 733K embeddings + AST structure (0 params)
5. **Backward compatible**: Full embeddings as fallback

## Expected Impact

- Recall: 35% → **60-70%** (slot matching handles partial queries)
- Precision: Should improve (better role matching)
- Latency: Stage 1 filtering reduces Stage 2 candidates

## Success Criteria

- [ ] Slot-based indexer implemented and tested
- [ ] Two-stage retriever working
- [ ] Recall improves to 60%+ on M1 benchmark
- [ ] Partial queries handled correctly (e.g., "Kiu X?" matches subject slot)
- [ ] No regression on full-query performance
- [ ] **CRITICAL**: "Kiu kreis Esperanton?" returns Zamenhof sentences

## Files to Create/Modify

```
klareco/rag/
├── slot_indexer.py          # New: SlotBasedIndexer
├── slot_retriever.py        # New: SlotBasedRetriever
└── retriever.py             # Extend: add slot-based mode

scripts/
└── index_slot_based.py      # New: Build slot-based index
```

## Related

- GitHub issue: #177
- Part of M2 milestone (#171)
- Complements tiered corpus architecture
- Prerequisite for graph embeddings (#4)
- Leverages Esperanto's explicit case marking
