---
id: 122
title: 'Enhancement: Add PRED slot for predicate nominatives in slot-based retrieval'
state: open
created: '2026-01-07T02:28:21.961608Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Background

Esperanto (and other languages) has predicate nominatives - nouns that complement copular verbs (esti) but aren't accusative objects.

Examples:
- "Hundo estas besto" (A dog is an animal) - "besto" is predicate nominative
- "Zamenhof estis kuracisto" (Zamenhof was a doctor) - "kuracisto" is predicate nominative

These are currently put in `aliaj` by the parser but ignored by slot retrieval.

## Proposal

Add a 4th slot: **PRED** (predicate nominative)

**Slot definitions:**
- SUBJ: Subject (nominative noun before verb)
- VERB: Verb
- OBJ: Direct object (accusative noun)
- **PRED: Predicate nominative (nominative noun after "esti" verb)**

**Extraction rule:**
```python
if verb.radiko == 'est':
    # For copular "esti", nominative nouns in aliaj are predicates, not objects
    for item in aliaj:
        if item.kazo == 'nominativo' and item.vortspeco == 'substantivo':
            slots['PRED'] = item
```

## Benefits

1. Better matching for definition questions ("Kio estas X?" → find docs with SUBJ=X)
2. Better matching for identity statements ("X estas Y")
3. Grammatically correct slot assignment

## Slot Weights

For definition questions:
- SUBJ: 0.1 (just the question word "Kio")
- VERB: 0.2 (just "estas")  
- OBJ: N/A
- PRED: 0.7 (the actual thing being defined)
