---
id: 69
title: Update CompositionalEmbedding to support dual roots
state: closed
created: '2026-01-05T23:02:35.971811Z'
labels:
- enhancement
- 'priority: high'
priority: high
---
**Phase 1: Architecture - Enable dual embedding mode in compositional system**

## Goal
Add `use_dual_roots` parameter to CompositionalEmbedding to support both single (64d) and dual (128d) root embeddings.

## Implementation

**File:** `klareco/embeddings/compositional.py` (MODIFY)

**Changes:**
1. Add `use_dual_roots: bool = False` parameter to `__init__`
2. Conditionally create DualRootEmbeddings or single Embedding
3. Add `root_mode` parameter to `forward()` method
4. Update root dimension calculation (64d vs 128d)
5. Adjust projection layer if needed

**Key logic:**
```python
if use_dual_roots:
    self.root_embed = DualRootEmbeddings(len(root_vocab), 64)
    self.root_dim = 128  # combined mode
else:
    self.root_embed = nn.Embedding(len(root_vocab), 64)
    self.root_dim = 64
```

**Backward compatibility:**
- Default `use_dual_roots=False` preserves existing behavior
- Existing code works without changes
- Can load old checkpoints

**Testing:**
- Unit tests with `use_dual_roots=True`
- Unit tests with `use_dual_roots=False` (existing behavior)
- Test all root modes: linguistic, topical, combined
- Test composition with affixes works correctly
- Integration test: load old model, switch to new, verify equivalence

## Acceptance Criteria
- [ ] `use_dual_roots` parameter added
- [ ] Dual mode creates 128d root embeddings
- [ ] Single mode still works (backward compatible)
- [ ] Forward pass handles `root_mode` parameter
- [ ] All existing tests still pass
- [ ] New tests for dual mode pass

## Dependencies
- **Blocks:** Training script (#71), retrieval integration (#73-75)
- **Depends on:** DualRootEmbeddings class (#68)

## Estimated Effort
3-4 hours

## References
Design doc Section 1.2
