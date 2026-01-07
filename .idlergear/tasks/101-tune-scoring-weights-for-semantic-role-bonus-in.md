---
id: 101
title: Tune scoring weights for semantic role bonus in hybrid retrieval
state: open
created: '2026-01-06T23:08:14.389819Z'
labels:
- enhancement
- retrieval
- tuning
priority: medium
---
## Problem

Current scoring formula in `_search_hybrid`:
```python
combined_score = entity_score * 0.35 + pattern_result.score * 0.35 + semantic_role_bonus * 0.3
```

With semantic role bonus = 2.5 for title/heading documents, this gives maximum contribution of 0.75.

But documents with literal verb match + object match can score ~1.6 total, dominating.

## Evidence

For query "Kiu fondis Esperanton?":
- "li fondis firmaeton Interland" scores ~1.6 (literal "fondis" match)
- "ZAMENHOF, Aŭtoro de la lingvo Esperanto" scores ~0.89 (semantic role bonus)

The semantic equivalent should rank closer to or above the literal match (since it's the actual answer).

## Options to Consider

### Option A: Increase semantic role bonus weight
```python
# Increase from 0.3 to 0.4 or 0.5
combined_score = entity_score * 0.3 + pattern_result.score * 0.3 + semantic_role_bonus * 0.4
```

### Option B: Higher semantic role bonus values
```python
# Increase bonus values
semantic_role_bonus = 4.0  # was 2.5 for title/heading
semantic_role_bonus = 3.0  # was 1.5 for sentence with verb
```

### Option C: Cap pattern score when semantic match present
```python
# If semantic role matches, don't let pattern dominate
if semantic_role_bonus > 0:
    pattern_contribution = min(pattern_result.score * 0.35, 0.5)
```

### Option D: Normalize scores before combining
```python
# Normalize each component to 0-1 before combining
entity_norm = min(entity_score / max_entity, 1.0)
pattern_norm = min(pattern_result.score / max_pattern, 1.0)
semantic_norm = min(semantic_role_bonus / max_bonus, 1.0)
combined_score = entity_norm * 0.35 + pattern_norm * 0.35 + semantic_norm * 0.3
```

## Proposed Approach

Start with Option B (increase bonus values) as it's least disruptive:
- Title/heading bonus: 2.5 → 4.0
- Sentence with verb bonus: 1.5 → 2.5

## Files to Modify

- `klareco/rag/ast_aware_retriever.py` - Adjust bonus values in `_search_hybrid`

## Acceptance Criteria

- [ ] Semantic equivalent documents rank in top 10 for benchmark queries
- [ ] "ZAMENHOF, Aŭtoro" ranks above "fondis firmaeton" for Esperanto founder query
- [ ] No regression on other benchmark queries
- [ ] Document the rationale for chosen weights
