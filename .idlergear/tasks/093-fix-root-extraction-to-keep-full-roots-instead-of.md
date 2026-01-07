---
id: 93
title: Fix root extraction to keep full roots instead of truncating
state: open
created: '2026-01-06T22:08:49.840803Z'
labels:
- bug
- retrieval
priority: medium
---
## Problem

The root extraction in ASTAware truncates roots, losing specificity:
- "esperanton" → "esp" (should be "esperant")
- This causes false matches with other "esp*" words

## Evidence

From evaluation diagnostics:
```json
"extracted_roots": ["kiu", "fond", "esp"],
"slots": {"SUBJ": "kiu", "VERB": "fond", "OBJ": "esp"}
```

The root "esp" is too short - it should be "esperant" (the actual root).

## Root Cause

Likely in `_embed_query_ast()` or slot extraction code:

```python
# ast_aware_retriever.py line 278
root = node.get('radiko', '')
if root and len(root) >= 2:
    roots.append(root.lower())
```

The issue may be in the parser returning truncated roots, or in the slot indexer.

## Investigation Needed

1. Check what parser returns for "Esperanton":
   ```python
   from klareco.parser import parse
   ast = parse("Kiu fondis Esperanton?")
   # Check ast['objekto']['kerno']['radiko']
   ```

2. If parser is correct, issue is in slot extraction

## Files to Investigate

- `klareco/parser.py`: Root extraction for proper nouns
- `klareco/rag/ast_aware_retriever.py`: `_embed_query_ast()` 
- `klareco/rag/slot_indexer.py`: `extract_slots()`

## Expected Impact

- More precise matching (won't match unrelated "esp*" words)
- Better embedding similarity (full root has more semantic info)

## Acceptance Criteria

- [ ] "esperanton" extracts to "esperant" not "esp"
- [ ] Verify parser output is correct
- [ ] Fix extraction if needed
- [ ] Re-run evaluation to measure impact
