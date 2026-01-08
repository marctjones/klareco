---
id: 134
title: Add embedding-based synonym expansion to root inverted index
state: open
created: '2026-01-07T16:06:17.729075Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Summary
Extend synonym expansion beyond SemanticRelationDB (9K curated pairs) by using linguistic/topical embeddings to find similar roots.

## Current Flow
```
Query roots → SemanticRelationDB.get_synonyms() → Inverted index lookup
```
Limited to curated synonyms in ReVo thesaurus.

## Proposed Flow (Option 2 - Hybrid)
```
Query roots 
  → SemanticRelationDB.get_synonyms() [deterministic first]
  → Embedding similarity search [learned expansion]
  → Inverted index lookup
```

## Implementation

1. Add method to find similar roots via embeddings:
```python
def _expand_roots_with_embeddings(
    self, 
    roots: Dict[str, float],
    similarity_threshold: float = 0.8,
    max_expansions_per_root: int = 5,
) -> Dict[str, float]:
    """Expand roots using embedding similarity."""
    expanded = dict(roots)
    
    for root, weight in roots.items():
        # Get embedding for this root
        emb = self.hybrid_embedder.get_root_embedding(root)
        if emb is None:
            continue
            
        # Find similar roots in vocabulary
        similar = self.hybrid_embedder.find_similar_roots(
            emb, 
            top_k=max_expansions_per_root,
            threshold=similarity_threshold,
        )
        
        for sim_root, sim_score in similar:
            if sim_root not in expanded:
                # Weight by similarity (e.g., 0.85 sim → 0.85 * original weight)
                expanded[sim_root] = weight * sim_score * 0.7
                
    return expanded
```

2. Add `find_similar_roots()` to HybridEmbeddings class

3. Add switch to control this:
```python
class ExpansionMode(Enum):
    DETERMINISTIC = auto()  # SemanticRelationDB only
    HYBRID = auto()         # SemanticDB + embeddings
    EMBEDDING_ONLY = auto() # Embeddings only (for comparison)
```

4. Track in stats which synonyms came from where:
```python
@dataclass
class RetrievalStats:
    synonyms_from_revo: List[str]
    synonyms_from_embeddings: List[str]
```

## Benefits
- Finds synonyms not in ReVo (broader coverage)
- Still controlled via threshold (explainable)
- Can A/B test deterministic vs hybrid
- Stats show exactly what came from where

## Files to Modify
- `klareco/rag/root_inverted_index.py` - Add hybrid expansion
- `klareco/embeddings/hybrid_embeddings.py` - Add find_similar_roots()
- `tests/test_deterministic_retrieval.py` - Add tests for hybrid mode
