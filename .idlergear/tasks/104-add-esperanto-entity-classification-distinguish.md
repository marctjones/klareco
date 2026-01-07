---
id: 104
title: 'Add Esperanto entity classification: distinguish language vs organization/modifier'
state: closed
created: '2026-01-06T23:52:19.860646Z'
labels:
- enhancement
- retrieval
- parser
priority: high
---
## Problem

The parser currently treats all uses of "esperant" identically, but they have different semantic meanings:

| Usage | Example | Current Parsing | Should Be |
|-------|---------|-----------------|-----------|
| The language | "Zamenhof fondis **Esperanton**" | `radiko=esperant, vortspeco=substantivo` | `entity_type=LANGUAGE` |
| Organization | "**Esperanto-Asocio** fondiĝis" | `radiko=esperant, vortspeco=substantivo` | `entity_type=ORG` |
| Modifier | "**Esperanto-**klubo" | `radiko=esperant` (compound) | `usage=MODIFIER` |
| Adjective | "**Esperantista** movado" | `radiko=esperant, sufiksoj=[ist]` | `usage=ADJECTIVE` |

## Impact on Retrieval

Query: "Kiu fondis Esperanton?" (Who founded Esperanto - THE LANGUAGE)

**Current behavior:** Returns documents about people founding Esperanto clubs, associations, studios - NOT about Zamenhof founding the language.

**With this fix:** Can filter/boost documents where "Esperanto" is the direct target of a creation verb, not a modifier.

## Proposed Solution: 100% Deterministic Classification

Add `esperanto_usage` field to AST nodes containing "esperant":

```python
def classify_esperanto_usage(word_node, sentence_context):
    """Classify whether 'esperant' refers to the language or a related entity."""
    
    plena_vorto = word_node.get('plena_vorto', '')
    
    # Compound word: "Esperanto-klubo", "Esperanto-Asocio"
    if '-' in plena_vorto:
        return 'MODIFIER'
    
    # Adjective form: "Esperanta", "Esperantista"
    if word_node.get('vortspeco') == 'adjektivo':
        return 'ADJECTIVE'
    
    # Direct object of creation verbs: "fondis Esperanton"
    if word_node.get('kazo') == 'akuzativo':
        verb = sentence_context.get('verbo', {})
        if verb.get('radiko') in {'fond', 'kre', 'invent', 'establ'}:
            return 'THE_LANGUAGE'
    
    # Subject with "estas lingvo": "Esperanto estas lingvo"
    if word_node.get('kazo') == 'nominativo':
        if 'lingvo' in str(sentence_context):
            return 'THE_LANGUAGE'
    
    return 'AMBIGUOUS'
```

## Implementation Options

### Option A: Parser-level (requires re-parsing)
- Add to `klareco/parser.py`
- New AST field: `esperanto_usage: LANGUAGE|MODIFIER|ADJECTIVE|AMBIGUOUS`
- Requires corpus re-parse

### Option B: Retriever-level (no re-parsing)
- Add to `klareco/rag/ast_aware_retriever.py`
- Classify at query time AND when processing candidate documents
- Works with existing corpus/index

### Option C: Hybrid (recommended)
- Add to parser for new parses
- Add runtime classification in retriever for existing index
- Migrate to parser-only when corpus is rebuilt

## Acceptance Criteria

- [ ] Classification function implemented
- [ ] "Kiu fondis Esperanton?" returns Zamenhof documents in top-5
- [ ] "Kiu fondis Esperanto-klubon?" returns club-founding documents
- [ ] No false positives (Esperanto-Asocio not confused with the language)
- [ ] 0 learned parameters (100% deterministic)
