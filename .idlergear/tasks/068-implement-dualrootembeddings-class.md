---
id: 68
title: Implement DualRootEmbeddings class
state: closed
created: '2026-01-05T23:02:19.930306Z'
labels:
- enhancement
- 'priority: high'
priority: high
---
**Phase 1: Architecture - Core dual embedding model**

## Goal
Create new `DualRootEmbeddings` class with two independent 64d embeddings (linguistic + topical).

## Implementation

**File:** `klareco/embeddings/dual_root_embeddings.py` (NEW)

**Class structure:**
```python
class DualRootEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64)
    def forward(self, indices, mode='combined') -> Tensor
    def get_normalized(self, indices, mode='combined') -> Tensor  
    def similarity(self, idx1, idx2, mode='combined', weights=(0.5, 0.5)) -> Tensor
```

**Key features:**
- Two independent `nn.Embedding` layers (linguistic, topical)
- Mode selection: 'linguistic', 'topical', 'combined'
- Combined mode returns 128d (concat of both)
- Weighted similarity computation
- L2 normalization support

**Testing:**
- Unit tests for forward pass (all modes)
- Unit tests for similarity computation
- Test shape assertions (64d vs 128d)
- Test backward compatibility

## Acceptance Criteria
- [ ] Class implemented with all methods
- [ ] Forward pass works for all 3 modes
- [ ] Similarity computation handles weights correctly
- [ ] Unit tests pass (>90% coverage)
- [ ] Can save/load checkpoints

## Dependencies
None (foundational)

## Estimated Effort
4-6 hours

## References
See design doc: `.idlergear/reference/dual-parallel-embeddings-implementation-design.md` Section 1.1
