---
id: 97
title: Add semantic keyword expansion using ReVo relations
state: open
created: '2026-01-06T22:25:41.604701Z'
labels:
- enhancement
- retrieval
- semantic
priority: medium
---
## Problem

The keyword prefilter is too literal. It can't bridge semantic gaps:
- "fondis" (founded) ↔ "aŭtoro" (author)
- "fondis" (founded) ↔ "kreinto" (creator)

**Example**: "Kiu fondis Esperanton?" misses "ZAMENHOF, Aŭtoro de la lingvo Esperanto" because "fondis" ≠ "aŭtoro" lexically.

## Proposed Solution

Use the existing SemanticRelationDB (loaded from ReVo) to expand query keywords:

```python
# In _keyword_prefilter:
expanded_keywords = []
for kw in keywords:
    expanded_keywords.append(kw)
    # Add synonyms from ReVo
    if kw in self.semantic_db.synonyms:
        expanded_keywords.extend(self.semantic_db.synonyms[kw])
```

This would expand:
- "fond" → ["fond", "kre", "iniciato", "aŭtor"]  (if in ReVo)

## Implementation Notes

1. Check what semantic relations exist in ReVo for common verbs
2. May need to build a "creator/author" relation manually
3. Balance expansion vs query drift (too many synonyms = noise)

## Related
- Note #86: Keyword Prefilter vs Semantic Search Trade-off
- Closed #95, #96: Fixed slot-based retrieval
