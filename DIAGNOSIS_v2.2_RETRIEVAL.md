# Diagnosis: Why v2.2 Semantic Ontology Integration Failed (24% accuracy)

**Date**: 2026-03-28
**Status**: Root cause identified
**Baseline**: 38% accuracy (19/50) → **v2.2**: 24% accuracy (12/50) ❌ **REGRESSION**

## Three Critical Problems Found

### Problem 1: Synonym Expansion Too Broad (8x dilution)

**Issue**: Query "Kiu fondis Esperanton?" expanded from 2 roots → 16 roots (8x)
- Original: `fond`, `esperant`
- Expanded: `fond`, `kre`, `establ`, `konstru`, `far`, `produk`, `fabrik`, `desegn`, `invent`, `develop`, `kompon`, `aŭtor`, `form`, `komenc`, `startig`, `esperant`

**Impact**: OR query over 16 terms dilutes BM25 scores. Documents matching rare terms ("fabrik" in etymology contexts) score higher than documents with actual answer.

**Evidence**:
```
Search: "zamenhof AND kre* AND esperanto" → 144 results including:
- "Zamenhof kreis Esperanton" (exact match!)
- "Zamenhof, la kreinto de Esperanto"

But OR query over 16 roots → 0/20 top results contain Zamenhof!
```

**Solution**: Limit to top 3 semantically closest synonyms + boost original query terms 3x.

---

### Problem 2: Entity Term Expansion Too Broad

**Issue**: "esperant" (entity = language name) expanded to ALL word forms:
- esperantas, esperantis, esperanti (verb forms) ← WRONG
- esperanta, esperantaj (adjective forms) ← WRONG
- esperantisto, esperantujo (derived nouns) ← WRONG

**Impact**: Retrieves documents about "Esperanto congresses", "Esperanto dialects", "Esperantists" instead of the LANGUAGE itself.

**Evidence**:
```
Search: "kre* AND esperant*" → 3,354 results, but top 10 are:
- "kreinto de esperantaj dialektoj" (creator of Esperanto dialects)
- "Esperanto-kongreso" (Esperanto congress)
- "esperantigan version" (Esperantized version)

NONE mention Zamenhof!
```

**Solution**: For entity/proper nouns, expand ONLY to noun forms (-o, -on, -oj, -ojn).

---

### Problem 3: No AST Grammatical Role Filtering

**Issue**: Retriever has methods to check AST roles (`subjekto`, `objekto`) but they're not being used effectively.

**Expected**: For "Kiu fondis Esperanton?":
- Find sentences with:
  - `objekto`: contains root "esperant" (the language)
  - `verbo`: contains root "fond" or synonym "kre"
  - `subjekto`: extract as answer

**Actual**: Retrieval uses text-level BM25 matching, not AST role constraints.

**Solution**: Add AST role filtering to retrieval, not just post-retrieval filtering.

---

## Why The Corpus HAS The Answer But We're Not Finding It

The corpus contains:
- "Zamenhof kreis Esperanton" (4 exact matches)
- "Zamenhof, la kreinto de Esperanto" (144 matches)

But our retrieval pipeline:
1. Expands "fond" → 14 synonyms (dilutes signal)
2. Expands "esperant" → all word forms (retrieves wrong contexts)
3. Uses OR query → documents matching ANY term score highly
4. No AST role constraints → accepts "esperantaj" (adjective) as match

Result: Documents about "creating Esperanto groups/congresses/versions" rank higher than "creating THE LANGUAGE Esperanto".

---

## Implementation Plan

### Phase 3.1: Smart Synonym Limiting (HIGHEST PRIORITY)

**File**: `klareco/knowledge/synonyms.py`

```python
def get_synonyms(root: str, max_count: int = 3) -> Set[str]:
    """Get top N semantically closest synonyms."""
    from .synonym_ranking import get_top_synonyms

    # Get all verb class members from ontology
    ontology_synonyms = get_verb_synonyms_from_ontology(root)

    if ontology_synonyms and len(ontology_synonyms) > 1:
        # Rank by semantic closeness, take top N
        ranked = get_top_synonyms(root, ontology_synonyms, max_count=max_count)
        return set(ranked)

    # Fallback to hardcoded
    if root in verb_synonyms:
        return verb_synonyms[root]

    return set()
```

**Expected improvement**: Reduces 8x expansion to 3x, preserves signal.

---

### Phase 3.2: Entity-Specific Expansion

**File**: `klareco/rag/whoosh_retriever.py`

Add new function:
```python
def expand_entity_noun(root: str) -> List[str]:
    """Expand entity noun to ONLY noun forms (-o, -on, -oj, -ojn)."""
    return [
        root + 'o',   # nominative singular
        root + 'on',  # accusative singular
        root + 'oj',  # nominative plural
        root + 'ojn', # accusative plural
    ]
```

Modify `retrieve()`:
```python
# Detect if root is an entity (proper noun)
if root in place_names or root == query_entity:
    # Expand to noun forms only
    forms = expand_entity_noun(root)
else:
    # Expand to all forms (existing logic)
    forms = expand_esperanto_root(root, question_type)
```

**Expected improvement**: "esperant" matches ONLY language noun forms, not adjectives/verbs.

---

### Phase 3.3: Term Boosting

**File**: `scripts/demo_extractive_qa.py`

```python
def expand_with_manual_synonyms(roots):
    """Expand with boosting: original terms 3x, synonyms 1x."""
    expanded = []

    for root in roots:
        if is_entity_root(root):
            # Entities: no expansion, but triple weight
            expanded.extend([root] * 3)
            continue

        # Add original root (triple weight)
        expanded.extend([root] * 3)

        # Add top 3 synonyms (single weight)
        synonyms = get_synonyms(root, max_count=3)
        expanded.extend(synonyms)

    return expanded
```

**Note**: Whoosh BM25 will naturally score documents with original terms higher due to repetition.

---

## Expected Results After Fix

| Question Type | Current | Expected | Improvement Source |
|---------------|---------|----------|-------------------|
| WHO | 10% | 60%+ | Entity expansion + boosting |
| WHERE | 30% | 50%+ | Entity expansion |
| WHEN | 0% | 30%+ | Better synonym matching |
| WHAT | 30% | 45%+ | Better retrieval |
| **Overall** | **24%** | **45-55%** | |

---

## Files to Modify

1. ✅ **CREATED**: `klareco/knowledge/synonym_ranking.py` - Rank synonyms by closeness
2. **MODIFY**: `klareco/knowledge/synonyms.py` - Use ranked synonyms (max 3)
3. **MODIFY**: `klareco/rag/whoosh_retriever.py` - Add `expand_entity_noun()`
4. **MODIFY**: `scripts/demo_extractive_qa.py` - Add term boosting via repetition

---

## Next Steps

1. Implement Phase 3.1 (synonym limiting) - **15 min**
2. Test on single question: "Kiu fondis Esperanton?" - **5 min**
3. If improved, implement Phase 3.2 (entity expansion) - **20 min**
4. Test again - **5 min**
5. If still improved, run full 50-question evaluation - **10 min**
6. If regression, implement Phase 3.3 (term boosting) - **15 min**

**Total estimated time**: 1-1.5 hours
