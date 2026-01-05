---
id: 26
title: Add named entity recognition and special handling for proper nouns
state: open
created: '2026-01-05T00:40:30.913873Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Problem

Proper nouns (names, places) are critical for factual Q&A but currently get same treatment as common nouns:
- "Kiu fondis Esperanton?" needs to match "Zamenhof"
- "Kie okazis la unua kongreso?" needs to match "Boulogne-sur-Mer"

## Proposed Solution

**Option 1: Entity-aware slots** (simpler)
- Add separate slots for named entities (SUBJ_ENTITY, OBJ_ENTITY)
- Higher weight for entity matches in factual queries
- Detect using capitalization + proper name grammar

**Option 2: Entity embeddings** (more powerful)
- Learn special embeddings for named entities
- Could use entity linking (Zamenhof → Wikidata Q157143)
- Share embeddings across mentions of same entity

## Implementation Priority

- Complexity: Low (Option 1) to Medium (Option 2)
- Impact: Medium-High for factual queries
- Could start with Option 1, upgrade to Option 2 later

## Related

- Complements pronoun resolution (#26)
- Part of broader retrieval improvements discussion
