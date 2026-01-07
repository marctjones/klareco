---
id: 105
title: Implement HEAD/MODIFIER distinction in slot-based retrieval
state: open
created: '2026-01-07T00:09:08.635262Z'
labels:
- enhancement
- retrieval
- P0
priority: high
---
## Problem

Current slot-based retrieval averages all roots in a slot, treating HEAD nouns and MODIFIERS equally. This causes documents about "Esperanto-klubo" (Esperanto club) to rank as high as documents about "Esperanto" itself for queries like "Kiu fondis Esperanton?"

## Root Cause Analysis

When matching query OBJ="esperant" against documents:
- "Zamenhof fondis Esperanton" → OBJ contains "esperant" as HEAD ✓
- "Li fondis Esperanto-klubon" → OBJ contains "klub" as HEAD, "esperant" as modifier

Currently both get similar scores because we average all roots. The HEAD/MODIFIER distinction is lost.

## Solution: Separate HEAD and MODIFIER Embeddings

### Slot Structure Change

**Current:**
```python
slots = {
    'SUBJ': embedding,  # Average of all roots
    'VERB': embedding,
    'OBJ': embedding,
}
```

**Proposed:**
```python
slots = {
    'SUBJ_HEAD': embedding,  # Head noun only
    'SUBJ_MOD': embedding,   # Modifiers (adjectives, compound prefixes)
    'VERB': embedding,       # Verb root (unchanged)
    'OBJ_HEAD': embedding,   # Head noun only
    'OBJ_MOD': embedding,    # Modifiers
}
```

### Matching Strategy

1. **HEAD-to-HEAD matching**: highest weight (0.7)
2. **MODIFIER-to-MODIFIER matching**: lower weight (0.3)  
3. **Cross-match penalty**: If query HEAD matches only doc MODIFIER, apply 0.5x penalty

### Scoring Formula

```python
def slot_similarity(query_slots, doc_slots):
    score = 0.0
    
    # VERB matching (unchanged)
    score += 0.30 * cosine_sim(query.VERB, doc.VERB)
    
    # SUBJ matching with HEAD/MOD distinction
    score += 0.20 * cosine_sim(query.SUBJ_HEAD, doc.SUBJ_HEAD)
    score += 0.05 * cosine_sim(query.SUBJ_MOD, doc.SUBJ_MOD)
    
    # OBJ matching with HEAD/MOD distinction  
    score += 0.20 * cosine_sim(query.OBJ_HEAD, doc.OBJ_HEAD)
    score += 0.05 * cosine_sim(query.OBJ_MOD, doc.OBJ_MOD)
    
    # Structural mismatch penalty
    if query.OBJ_HEAD matches doc.OBJ_MOD but NOT doc.OBJ_HEAD:
        score *= 0.5
    
    return score
```

## Implementation Plan

This is a parent issue. Sub-issues will track individual components.

## Expected Impact

- "Kiu fondis Esperanton?" returns Zamenhof documents first (not club-founding docs)
- General improvement for all queries where HEAD/MODIFIER distinction matters
- No hardcoding for specific words - works for any noun

## Acceptance Criteria

- [ ] Slot extraction separates HEAD from MODIFIER
- [ ] Index stores HEAD and MODIFIER embeddings separately
- [ ] Retriever uses HEAD-weighted matching
- [ ] Cross-match penalty implemented
- [ ] Benchmark shows improvement on Q&A accuracy
- [ ] No regression on grammar questions
