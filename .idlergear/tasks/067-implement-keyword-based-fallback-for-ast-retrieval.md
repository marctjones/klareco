---
id: 67
title: Implement keyword-based fallback for AST retrieval
state: open
created: '2026-01-05T22:19:34.163584Z'
labels:
- enhancement
- 'priority: medium'
priority: medium
---
**Goal**: Add BM25-style keyword matching as fallback when AST pattern matching fails or returns low scores.

**Motivation**:
- AST pattern matching is powerful but requires grammatical similarity
- Some questions may be answerable via keyword overlap alone
- Example: "Kio estas Esperanto?" → documents with "Esperanto estas..." might not match AST pattern but contain answer

**Proposed Architecture**:

```python
def search_with_fallback(query, top_k=10, prefilter_n=500):
    # Stage 1: HNSW pre-filter
    candidates = hnsw_search(query, prefilter_n)
    
    # Stage 2: AST pattern matching
    ast_results = ast_match(candidates, top_k)
    
    # Stage 3: Keyword fallback (if AST results weak)
    if max(ast_results.scores) < 0.5:  # Low confidence
        keyword_results = bm25_match(candidates, query, top_k)
        # Combine or fallback to keyword results
        return merge_results(ast_results, keyword_results)
    
    return ast_results
```

**Implementation Options**:

1. **Pure keyword re-ranking**:
   - After AST matching, re-rank by keyword overlap (TF-IDF/BM25)
   - Combine scores: `0.6 * ast_score + 0.4 * bm25_score`

2. **Conditional fallback**:
   - Use AST results if confidence > threshold
   - Otherwise use keyword-based ranking

3. **Hybrid scoring**:
   - Always combine both signals
   - Learn weights from validation set

**BM25 Implementation**:
```python
from rank_bpython import BM25Okapi

class ASTAwareRetriever:
    def __init__(self, ...):
        # ... existing init ...
        self.use_keyword_fallback = True
    
    def _keyword_score(self, query_words, doc_text):
        # Simple BM25-style scoring
        # Or use existing library
        pass
    
    def _search_with_fallback(self, ...):
        # Combine AST + keyword scores
        pass
```

**Success Criteria**:
- Improve accuracy on questions where AST matching fails
- Especially help with definition questions ("Kio estas X?")
- Don't hurt accuracy on questions where AST works well

**Testing**:
1. Run benchmark with keyword fallback enabled
2. Compare accuracy: baseline vs fallback
3. Analyze which question types benefit most

**Related**: Task #63 (parent - improve AST retrieval accuracy)
