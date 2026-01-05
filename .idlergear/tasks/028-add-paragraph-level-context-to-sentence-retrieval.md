---
id: 28
title: Add paragraph-level context to sentence retrieval (hybrid approach)
state: open
created: '2026-01-05T00:40:31.301423Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Problem

Current sentence-level retrieval is precise but loses cross-sentence context:
- ✅ Precise matching (exact answer location)
- ❌ Loses pronoun references
- ❌ Multi-sentence answers split apart

Pure paragraph-level indexing has opposite trade-offs:
- ✅ Preserves context
- ❌ Less precise (which sentence has answer?)
- ❌ Noisier embeddings (off-topic sentences dilute signal)

## Proposed Solution: Hybrid Approach

**Best of both worlds:**
1. Index sentences (current - precise retrieval)
2. Store paragraph metadata with each sentence
3. Three-stage retrieval:
   - Stage 1: Retrieve top-N sentences (current)
   - Stage 2: Expand to full paragraphs for reranking
   - Stage 3: Return sentences with paragraph context

**Example:**
```
Query: "Kiu fondis Esperanton?"

Stage 1 retrieves: "Zamenhof publigis la unuan libron en 1887."
Stage 2 expands: [Full paragraph with pronouns resolved]
Stage 3 returns: Answer sentence + context for verification
```

## Implementation Plan

1. Augment index with `paragraph_id` field (low effort)
2. Keep sentence-level retrieval as primary (current works)
3. Add paragraph expansion as optional reranking stage
4. Evaluate impact on benchmark queries

## Benefits

- Preserves current precision
- Adds context when needed
- Helps with pronoun resolution
- Enables multi-sentence answers

## Related

- Complements coreference resolution (#26)
- Part of broader retrieval improvements
