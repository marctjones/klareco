---
id: 107
title: Update slot indexer to extract HEAD and MODIFIER embeddings separately
state: open
created: '2026-01-07T00:10:12.841994Z'
labels:
- enhancement
- retrieval
priority: high
---
## Goal

Modify `SlotBasedIndexer` to extract and store separate embeddings for HEAD and MODIFIER components of each slot.

## Current Implementation

`klareco/rag/slot_indexer.py` extracts:
```python
slots = {
    'SUBJ': average_embedding(all_roots_in_subject),
    'VERB': embedding(verb_root),
    'OBJ': average_embedding(all_roots_in_object),
}
```

## Proposed Implementation

```python
slots = {
    'SUBJ_HEAD': embedding(subject.kerno.radiko),
    'SUBJ_MOD': average_embedding(subject.priskriboj + subject.kunmetaĵoj),
    'VERB': embedding(verb_root),
    'OBJ_HEAD': embedding(object.kerno.radiko),
    'OBJ_MOD': average_embedding(object.priskriboj + object.kunmetaĵoj),
}
```

## Changes Required

1. **Extract HEAD from kerno:** `slot_group.kerno.radiko`
2. **Extract MODIFIERS from priskriboj:** All adjectives and compound modifiers
3. **Handle missing modifiers:** Use NaN or zero vector when no modifiers
4. **Update index format:** Store 5 slot embeddings instead of 3

## Index Format Change

**Current:**
```json
{
  "text": "...",
  "slots": {"SUBJ": [...], "VERB": [...], "OBJ": [...]},
  "features": {...}
}
```

**Proposed:**
```json
{
  "text": "...",
  "slots": {
    "SUBJ_HEAD": [...], "SUBJ_MOD": [...],
    "VERB": [...],
    "OBJ_HEAD": [...], "OBJ_MOD": [...]
  },
  "features": {...}
}
```

## Backward Compatibility

- New index format is NOT backward compatible
- Will require full index rebuild
- Consider version field in index for future migrations

## Acceptance Criteria

- [ ] HEAD extracted from kerno for SUBJ and OBJ
- [ ] MODIFIERS extracted from priskriboj + kunmetaĵoj
- [ ] Index stores 5 slot embeddings
- [ ] Tests verify correct extraction
- [ ] Documentation updated

## Depends On

- #106 (parser compound word fix) - needs correct modifier extraction from parser

## Related

Parent: #105 (HEAD/MODIFIER distinction in retrieval)
