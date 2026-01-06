---
id: 64
title: Train semantic embeddings with similarity objectives
state: open
created: '2026-01-05T22:18:42.845409Z'
labels:
- training
- 'priority: high'
- enhancement
priority: high
---
**Goal**: Improve root embeddings to capture semantic similarity, not just morphological features.

**Current Problem**: 
- Root embeddings were trained for morphological composition (Stage 1)
- They lack semantic signal needed for retrieval
- Example: "Fundamento" and "Esperanto" embeddings may not be close to related concepts
- Pre-filter returns irrelevant documents (answer found at rank 1144 instead of top 10)

**Proposed Solution**:
Train embeddings with semantic similarity objectives:

1. **Semantic similarity pairs from ReVo**:
   - Use synonym relations: (hund, best) should be close
   - Use antonym relations: (bon, mal) should be far apart
   - Use hypernym/hyponym: (hund, besto) hierarchical distance

2. **Contrastive learning from corpus**:
   - Words appearing in similar contexts should have similar embeddings
   - Use co-occurrence statistics from parsed corpus
   - Skip-gram or CBOW style objectives

3. **Multi-task learning**:
   - Keep morphological composition objective (Stage 1)
   - Add semantic similarity objective
   - Balance both with weighted loss

**Training Data Available**:
- ✓ ReVo semantic relations: 2543 synonym roots, 300 antonym roots
- ✓ 4.3M parsed sentences with word contexts
- ✓ Existing morphological training pairs

**Success Criteria**:
- Root embedding quality improves (see test_embedding_quality.py)
- Pre-filter ranks relevant docs in top 100-500 (not 1000+)
- AST-aware retriever accuracy improves from 12% → 30%+

**Implementation Steps**:
1. Create semantic similarity training pairs from ReVo
2. Create contrastive pairs from corpus co-occurrence
3. Design multi-task loss function
4. Train new embedding model (Stage 1.5)
5. Rebuild HNSW index with new embeddings
6. Re-run benchmark to measure improvement

**Related**: Task #63 (parent task - improve AST retrieval accuracy)
