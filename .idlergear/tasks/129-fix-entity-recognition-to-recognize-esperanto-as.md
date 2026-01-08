---
id: 129
title: Fix entity recognition to recognize 'Esperanto' as a named entity
state: open
created: '2026-01-07T04:40:41.805191Z'
labels:
- bug
- retrieval
priority: high
---
## Problem

The EntityRecognizer explicitly skips "esperanto" from being recognized as an entity:

```python
# In entity_recognizer.py lines 277-284
common_words = {
    'esperanto', 'esperanton', 'esperanta', 'esperante',
    ...
}
if text_lower in common_words:
    return None
```

This causes the query "Kiu fondis Esperanton?" to have 0 recognized entities, which breaks entity-focused retrieval.

## Root Cause Analysis

1. The intent was to prevent "Esperanto" from being treated as an unknown proper noun
   (since it parses successfully as a regular substantivo)
2. But this also prevents it from being used for entity matching in WHO/WHAT questions
3. Without entity recognition, the retriever can't use entity overlap scoring

## Diagnostic Evidence

From `diagnose_ast_retriever.py`:
```
--- Stage 3: Entity Recognition ---
  Found 0 entities
```

The query "Kiu fondis Esperanton?" has:
- Subject: kiu (correlative, skipped)
- Verb: fondis
- Object: Esperanton → should be recognized as LANGUAGE entity

## Solution Options

### Option 1: Add "Esperanto" to a languages gazetteer
Create `data/gazetteers/languages.json` with Esperanto and other language names.
Modify EntityRecognizer to check gazetteers BEFORE common_words filtering.

### Option 2: Change common_words logic
Instead of skipping entity recognition entirely, still recognize these words
but with a different entity type (e.g., CONCEPT, LANGUAGE).

### Option 3: Use a two-pass approach
First pass: entity recognition for retrieval (include Esperanto)
Second pass: entity filtering for extraction (exclude common words)

## Recommended Fix

**Option 2** - Keep recognizing "Esperanto" as an entity but with type LANGUAGE.
This allows entity matching to work while not confusing it with person names.

```python
# Change this:
if text_lower in common_words:
    return None

# To this:
LANGUAGE_WORDS = {'esperanto', 'esperanton', 'esperanta', 'esperante'}
if text_lower in LANGUAGE_WORDS:
    return Entity(
        text=text,
        entity_type=EntityType.WORK,  # Or add EntityType.LANGUAGE
        root=root,
        slot=slot,
        confidence=0.9,
    )
```

## Impact

Without this fix:
- "Kiu fondis Esperanton?" finds 0 entities → uses pattern-only matching
- Pattern matching finds documents with "fondis" but not about Esperanto
- Correct answer "ZAMENHOF, Aŭtoro de Esperanto" is missed

With this fix:
- "Esperanto" recognized as LANGUAGE entity
- Entity overlap scoring boosts documents mentioning Esperanto
- Expected: +20-30pp improvement on Esperanto-related factual questions

## Related Issues

- Part of AST-aware retriever diagnostic findings
- See note #97 for full diagnostic report
