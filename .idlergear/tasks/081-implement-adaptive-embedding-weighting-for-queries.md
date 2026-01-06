---
id: 81
title: Implement adaptive embedding weighting for queries
state: open
created: '2026-01-06T05:44:10.127948Z'
labels:
- enhancement
- embeddings
- retrieval
priority: medium
---
Add smart weighting between linguistic and topical embeddings based on query type.

**Prerequisites:**
- Task #78 complete (retrievers updated)
- Task #80 complete (benchmark shows which queries benefit from which embedding type)

**Concept:**
Different queries benefit from different embedding types:
- **Linguistic queries**: "Kio estas hundo?" (what is a dog?) - benefit from linguistic embeddings
- **Topical queries**: "Kio estas Parizo?" (what is Paris?) - benefit from topical embeddings
- **Mixed queries**: "Kie loĝas Napoleon?" (where does Napoleon live?) - benefit from both

**Implementation approach:**
1. Classify query type based on AST patterns:
   - Contains proper nouns → increase topical weight
   - Contains only content words → increase linguistic weight
   - Mixed → balanced weighting

2. Add weighting parameter to HybridEmbeddings:
```python
def get_root_embedding(self, root: str, 
                       linguistic_weight: float = 0.5,
                       topical_weight: float = 0.5):
    # Weighted combination instead of simple concatenation
    ling_emb = linguistic_weight * self.linguistic_model.get_root_embedding(root)
    top_emb = topical_weight * self.topical_model.get_root_embedding(root)
    return torch.cat([ling_emb, top_emb])
```

3. Update retrievers to accept and use query-specific weights

**Success criteria:**
- Query classifier implemented with AST pattern matching
- Weighted embedding combination working
- Benchmark shows improvement over fixed weighting
- Documented decision rules for weight selection
