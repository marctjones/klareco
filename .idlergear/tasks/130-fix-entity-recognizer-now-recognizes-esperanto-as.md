---
id: 130
title: 'Fix: Entity recognizer now recognizes ''Esperanto'' as LANGUAGE entity'
state: closed
created: '2026-01-07T04:43:41.346988Z'
labels:
- bug
- decision
priority: high
---
## Problem

The EntityRecognizer in `klareco/rag/entity_recognizer.py` was explicitly skipping "Esperanto" and its variants from entity recognition by including them in a `common_words` set (lines 277-284):

```python
common_words = {
    'esperanto', 'esperanton', 'esperanta', 'esperante',  # <-- BUG!
    'la', 'kaj', 'de', 'en', 'al', 'por', 'kun', 'pri',
    ...
}
```

This caused queries like "Kiu fondis Esperanton?" to have 0 entities recognized, severely hurting retrieval quality for questions about Esperanto itself.

## Impact

- Query "Kiu fondis Esperanton?" had 0 entities before fix
- Query "Kio estas Esperanto?" had 0 entities before fix
- This prevented entity-aware boosting in ASTAwareRetriever from working

## Solution

1. Added new `EntityType.LANGUAGE` to represent language entities
2. Added `LANGUAGE_ROOTS` class variable with language-related roots
3. Added `KNOWN_LANGUAGES` class variable with common language names in Esperanto
4. Modified `_check_proper_name()` to check for languages BEFORE skipping function words
5. Renamed `common_words` to `function_words` to clarify intent

## Test Results

Before:
```
"Kiu fondis Esperanton?" → 0 entities
```

After:
```
"Kiu fondis Esperanton?" → 1 entity: Esperanton (language), confidence=0.95
"Zamenhof kreis Esperanton" → 2 entities: Zamenhof (person), Esperanton (language)
```

## Files Changed

- `klareco/rag/entity_recognizer.py`

## Related

- Diagnostic from previous session identified this as a key retrieval blocker
- Part of ASTAwareRetriever improvements
