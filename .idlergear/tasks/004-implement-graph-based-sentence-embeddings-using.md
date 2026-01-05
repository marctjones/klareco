---
id: 4
title: Implement graph-based sentence embeddings using TreeLSTM/GNN
state: open
created: '2026-01-02T07:57:13.384849Z'
labels:
- enhancement
- research
- neural
priority: medium
---
Implement graph-based sentence embeddings using TreeLSTM or GCN.

**Status**: Research needed - evaluate TreeLSTM vs GCN approaches

**Goal**: AST-aware sentence embeddings that capture syntactic structure

**Current approach**: Mean pooling of word embeddings (loses structure)

**Proposed**: 
- TreeLSTM: Processes AST bottom-up
- GCN: Graph convolution over AST edges

**Benefits**:
- Syntax-aware similarity
- Better handling of word order variations
- Compositional understanding

**Next steps**:
1. Research TreeLSTM implementations
2. Prototype on small dataset
3. Compare with baseline mean pooling
