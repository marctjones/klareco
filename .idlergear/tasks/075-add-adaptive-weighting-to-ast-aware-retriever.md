---
id: 75
title: Add adaptive weighting to AST-aware retriever
state: open
created: '2026-01-05T23:04:14.688855Z'
labels:
- enhancement
- 'priority: medium'
priority: medium
---
**Phase 5: Integration - Query-type-based embedding mode selection**

## Goal
Update AST-aware retriever to use adaptive weighting based on query type, enabling explainable multi-faceted retrieval.

## Implementation

**File:** `klareco/rag/ast_aware_retriever.py` (MODIFY)

**Changes:**

1. **Add embedding mode selection:**
   - Parameter: `embedding_mode: str = 'adaptive'`
   - Choices: 'linguistic', 'topical', 'combined', 'adaptive'

2. **Implement adaptive weighting:**
```python
if embedding_mode == 'adaptive':
    if question_type == 'definition':
        # Definitions need linguistic similarity
        weights = {'linguistic': 0.7, 'topical': 0.3}
    elif question_type in ['who', 'where', 'when']:
        # Factual questions need topical context
        weights = {'linguistic': 0.3, 'topical': 0.7}
    else:
        # Default: balanced
        weights = {'linguistic': 0.5, 'topical': 0.5}
```

3. **Add explainability output:**
```python
def explain_retrieval(self, query, doc, score):
    return {
        'total_score': score,
        'components': {
            'ast_pattern': {...},
            'linguistic': {...},
            'topical': {...}
        }
    }
```

4. **Update search pipeline:**
   - Pass embedding_mode to pre-filter
   - Use mode consistently through pipeline
   - Expose scores from each component

**Testing:**
- Test adaptive mode with different question types
- Test linguistic-only mode
- Test topical-only mode
- Test combined mode (50/50)
- Verify explainability output

## Acceptance Criteria
- [ ] Adaptive weighting implemented
- [ ] Different question types use different weights
- [ ] Explainability shows 3 components (AST, ling, topic)
- [ ] All modes work correctly
- [ ] Tests pass for each mode

## Dependencies
- **Blocks:** Benchmark (#76)
- **Depends on:** HNSW retriever update (#74)

## Estimated Effort
4-6 hours

## References
Design doc Section 4.2, 4.3
