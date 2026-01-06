---
id: 77
title: Refactor to separate linguistic and topical embedding models
state: closed
created: '2026-01-06T03:16:31.553666Z'
labels:
- architecture
- 'priority: high'
- refactoring
priority: high
---
## Problem

Current DualRootEmbeddings forces same vocabulary for both embeddings, but:
- Proper nouns (Napoleono, Parizo) only need topical embeddings
- Function words (kaj, de, la) need neither (handled by AST)
- Content words (hund, bel) need both
- Technical terms may only need topical

Forcing same vocab wastes storage and doesn't match semantic reality.

## Solution

Separate models with independent vocabularies:

**1. Two independent embedding models:**
- `LinguisticEmbeddings` - 64d, trained on ReVo + semantic relations (~15K roots)
- `TopicalEmbeddings` - 64d, trained on skip-gram pairs (~77K roots)

**2. Smart combiner for inference:**
- Loads both models
- Looks up each root in both vocabularies
- Returns available embeddings (may be 64d, 128d, or even 0d)
- Handles missing gracefully

**3. Benefits:**
- Vocabulary independence (linguistic stable, topical grows)
- Efficient storage (don't store useless embeddings)
- Clear semantics (presence in vocab = has that type of meaning)
- Better for new words (add to topical only)

## Implementation Plan

1. Create `LinguisticEmbeddings` (simple nn.Embedding wrapper)
2. Create `TopicalEmbeddings` (simple nn.Embedding wrapper)
3. Create `HybridEmbeddings` (smart combiner)
4. Update training scripts (train separately)
5. Update retrieval pipeline
6. Deprecate DualRootEmbeddings

## Files to Create/Modify

- NEW: `klareco/embeddings/linguistic_embeddings.py`
- NEW: `klareco/embeddings/topical_embeddings.py`
- NEW: `klareco/embeddings/hybrid_embeddings.py`
- MODIFY: `scripts/training/train_topical_embeddings.py` (rename from train_dual)
- DEPRECATE: `klareco/embeddings/dual_root_embeddings.py`
