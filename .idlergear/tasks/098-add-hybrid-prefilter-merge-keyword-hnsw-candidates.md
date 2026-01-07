---
id: 98
title: 'Add hybrid prefilter: merge keyword + HNSW candidates'
state: open
created: '2026-01-06T22:25:41.789874Z'
labels:
- enhancement
- retrieval
- architecture
priority: medium
---
## Problem

Currently ASTAware uses EITHER keyword prefilter OR HNSW prefilter, but not both together. This means:
- Keyword prefilter: Good precision, misses semantic matches
- HNSW prefilter: Good semantic recall, may miss exact matches

## Proposed Solution

Merge candidates from both prefilters:

```python
def _hybrid_prefilter(self, query_ast, max_results=500):
    # Get keyword candidates (high precision)
    keyword_candidates = self._keyword_prefilter(query_ast, max_results=max_results//2)
    
    # Get HNSW candidates (semantic recall)
    hnsw_candidates = self._hnsw_prefilter(query_ast, max_results=max_results//2)
    
    # Merge and deduplicate by doc ID
    seen = set()
    merged = []
    for score, doc in keyword_candidates + hnsw_candidates:
        doc_id = doc.get('id') or hash(doc.get('text', '')[:100])
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append((score, doc))
    
    # Re-rank merged candidates by slot similarity
    return self._rerank_by_slots(merged, query_ast)[:max_results]
```

## Benefits
- Keyword prefilter catches exact matches (high precision)
- HNSW prefilter catches semantic matches (good recall)
- Slot-based reranking ensures best candidates rise to top

## Related
- Note #86: Keyword Prefilter vs Semantic Search Trade-off
