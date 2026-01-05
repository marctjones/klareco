---
id: 27
title: Add coreference resolution for pronoun handling
state: open
created: '2026-01-05T00:40:31.089478Z'
labels:
- enhancement
- stage-3
- retrieval
priority: medium
---
## Problem

Pronouns refer back to previously mentioned entities, but we index sentences independently:

**Example:**
```
"Zamenhof kreis Esperanton. Li estis okulisto."
```
The pronoun "Li" refers to "Zamenhof", but "Li estis okulisto" loses this connection when retrieved alone.

## Proposed Solutions

**Option 1: Pre-retrieval expansion**
- Resolve pronouns before indexing
- "Li estis okulisto" → "Zamenhof estis okulisto"
- Better matching when query mentions "Zamenhof"
- Simpler but less flexible

**Option 2: Discourse-aware embeddings**
- Encode coreference chains in embeddings
- Part of Stage 3 (Discourse Model) in roadmap
- Preserves ambiguity, more powerful
- More complex, planned for later

## Implementation Priority

- Complexity: Medium-High (requires coreference resolution algorithm)
- Impact: High (improves cross-sentence understanding)
- Timeline: Stage 3 work (after Stage 2 grammatical model complete)

## Related

- Part of Stage 3 Discourse Model
- Complements named entity handling (#27)
- See `IMPLEMENTATION_ROADMAP_V2.md` Stage 3
