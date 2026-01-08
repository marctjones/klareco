---
id: 53
title: Implement multi-strategy AST-aware retriever
state: closed
created: '2026-01-05T15:48:57.102148Z'
labels:
- enhancement
- retrieval
- 'priority: high'
priority: high
---
## Objective
Build the integrated retriever that combines all AST-aware strategies (pattern matching, entity search, semantic roles, embeddings) for robust question answering.

## Implementation Complete ✓

### Components Integrated

1. **QuestionClassifier** - Classifies questions by type (WHO/WHAT/WHEN/WHERE) and determines target entity type
2. **EntityRecognizer** - Extracts named entities using gazetteers and heuristics
3. **SemanticRelationDB** - Provides synonyms and agent noun mappings for query expansion
4. **ASTPatternMatcher** - Matches structural patterns between query and document ASTs
5. **HybridEmbeddings** - Query embedding with semantic expansion

### Multi-Strategy Search

- `_search_pattern_matching`: Pure AST pattern matching
- `_search_entity_focused`: Entity overlap + pattern matching
- `_search_hybrid`: Combined entity + pattern + semantic role bonus
- Strategy selection based on question type

### Key Features

- **Semantic expansion in query embedding**: Query "fondis" now includes embeddings for synonyms (kre, establ) and agent nouns (aŭtoro, kreinto)
- **HNSW prefilter with slot reranking**: Fast vector search + grammatical role matching
- **Creator role bonus**: Documents with "aŭtoro" get boost for "fondis"-type queries
- **Explainability**: `explain_retrieval()` method shows why document was matched

### Files Modified

- `klareco/rag/ast_aware_retriever.py` - Main multi-strategy retriever
- `data/semantic_relations/curated_synonyms.json` - Added "aŭtoro" to fond agent nouns

### Known Limitation

Retrieval accuracy is limited by embedding recall (~35% per Task #222 evaluation). The embeddings don't fully capture semantic relationships like "fond" ↔ "aŭtor". This is a separate issue to be addressed by improving embeddings or adding keyword-based fallback.

### Testing

```python
retriever = ASTAwareRetriever(index_path=Path('data/indexes/slot_hybrid'))
results = retriever.search("Kiu fondis Esperanton?", top_k=10)
# Returns Esperanto-related founding documents
# Zamenhof not in top-10 due to embedding recall limitation
```

## Status: COMPLETE

All components integrated as designed. Follow-up work needed for improving retrieval recall (separate task).
