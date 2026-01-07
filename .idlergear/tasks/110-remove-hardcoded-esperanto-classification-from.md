---
id: 110
title: Remove hardcoded Esperanto classification from ast_aware_retriever.py
state: open
created: '2026-01-07T00:10:14.743009Z'
labels:
- cleanup
- retrieval
priority: low
---
## Problem

Task #104 added hardcoded Esperanto-specific classification functions that should be removed in favor of the general HEAD/MODIFIER solution.

## Code to Remove

Lines 31-189 in `klareco/rag/ast_aware_retriever.py`:
- `LANGUAGE_CREATION_VERBS` constant
- `classify_esperanto_usage()` function
- `get_esperanto_usage_from_text()` function
- `score_esperanto_relevance()` function

Also remove integration code in:
- `_search_hybrid()` method (lines ~1344-1364)
- `_search_entity_focused()` method (lines ~1215-1225)

## Why Remove

1. **Hardcoded for one word** - Only helps "Esperanto", not general
2. **Wrong approach** - Should fix slot matching, not add word-specific heuristics
3. **Superseded** - HEAD/MODIFIER distinction (#105) solves the same problem generally

## Acceptance Criteria

- [ ] All Esperanto-specific classification code removed
- [ ] No references to `LANGUAGE_CREATION_VERBS`
- [ ] No references to `classify_esperanto_usage`
- [ ] Tests still pass
- [ ] Retrieval quality maintained (by HEAD/MODIFIER fix)

## Depends On

- #105, #106, #107, #108, #109 - general fix must be working first

## Related

Closes: #104 (superseded by general solution)
