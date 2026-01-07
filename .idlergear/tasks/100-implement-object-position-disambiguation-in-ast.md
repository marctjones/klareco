---
id: 100
title: Implement object position disambiguation in AST pattern matcher
state: closed
created: '2026-01-06T23:08:13.919937Z'
labels:
- enhancement
- retrieval
- ast-aware
priority: high
---
## Problem

Current pattern matching doesn't distinguish between object appearing as CORE vs MODIFIER:

**Example:**
- Query: "Kiu fondis Esperanton?" → object = "Esperanto" (CORE position)
- Doc A: "fondis Esperanton" → object = "Esperanto" (CORE) ✅ Good match
- Doc B: "fondis Esperanto-rondon" → object = "rondo" (CORE), "Esperanto" is MODIFIER ❌ Should score lower

Both currently get same object match score, but Doc B is talking about founding an Esperanto CLUB, not founding Esperanto itself.

## Evidence from Testing

When testing "Kiu fondis Esperanton?":
- Documents with "fondis Esperanto-rondon" (founded Esperanto club) score ~1.6
- Target "ZAMENHOF, Aŭtoro de la lingvo Esperanto" scores ~0.89
- The compound word match shouldn't score as high as direct object match

## Proposed Solution

In `_match_core_positions()` method of `ASTPatternMatcher`:

1. Check if query object appears as CORE in document vortgrupo
2. If it appears only as MODIFIER (priskriboj), apply penalty
3. Penalty could be 0.5x the normal match score

```python
def _match_core_positions(self, query_ast, doc_ast):
    # Extract object from query
    q_obj_root = self._get_object_root(query_ast)
    
    # Check document object vortgrupo
    doc_obj = doc_ast.get('objekto', {})
    doc_obj_core = doc_obj.get('kerno', {}).get('radiko', '')
    doc_obj_modifiers = [p.get('radiko', '') for p in doc_obj.get('priskriboj', [])]
    
    if q_obj_root == doc_obj_core:
        return 1.0  # Exact core match
    elif q_obj_root in doc_obj_modifiers:
        return 0.5  # Modifier match (penalty)
    else:
        return 0.0  # No match
```

## Files to Modify

- `klareco/rag/ast_pattern_matcher.py` - Add position-aware matching
- `klareco/rag/ast_aware_retriever.py` - Integrate into scoring

## Acceptance Criteria

- [ ] Query object as document CORE scores 1.0
- [ ] Query object as document MODIFIER scores 0.5
- [ ] "Kiu fondis Esperanton?" ranks ZAMENHOF doc higher than Esperanto-rondon doc
- [ ] Unit tests for position disambiguation
