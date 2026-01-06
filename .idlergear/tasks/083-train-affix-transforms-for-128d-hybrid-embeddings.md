---
id: 83
title: Train affix transforms for 128d hybrid embeddings
state: open
created: '2026-01-06T05:51:56.973333Z'
labels:
- enhancement
- embeddings
- future
priority: medium
---
Affix transforms currently work only with 64d linguistic embeddings. Need to train new transforms for 128d hybrid embeddings.

**Current status:**
- Affix transforms: 64d → 64d (linguistic only)
- Hybrid embeddings: 128d (64d linguistic + 64d topical)

**Temporary solution:**
- Hybrid mode: Skip affix transforms (use raw root embeddings)
- Legacy mode: Continue using 64d transforms

**Future task:**
Train new affix transformation matrices:
1. Prepare training data with affix pairs from corpus
2. Train low-rank transforms (128d → 128d)
3. Update SlotBasedIndexer to apply transforms in hybrid mode

**Priority:** Medium (can improve quality but not blocking)
