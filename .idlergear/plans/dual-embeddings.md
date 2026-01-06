---
name: dual-embeddings
title: Dual Embeddings Implementation (Stage 1.5)
state: active
created: '2026-01-05T23:19:29.949289Z'
---
# Dual Embeddings Implementation Plan

**Goal:** Improve retrieval accuracy from 12% → 30%+ by adding topical embeddings alongside linguistic embeddings.

**Timeline:** 3 weeks (15 working days)

---

## Milestone 1: Dual Embeddings Core (Week 1)

**Deliverable:** Working dual embedding architecture with training data ready

**Tasks:**
- [x] Task #68: Implement DualRootEmbeddings class
- [ ] Task #69: Update CompositionalEmbedding to support dual roots  
- [ ] Task #70: Create topical training data from corpus

**Success Metrics:**
- Can create and load dual embeddings
- Forward pass works for all modes
- 50M+ topical training pairs extracted
- All tests pass

**Dependencies:**
- #68 and #70 can run in parallel (no deps)
- #69 depends on #68

---

## Milestone 2: Trained Dual Model (Week 2)

**Deliverable:** Trained dual embedding model with good intrinsic metrics

**Tasks:**
- [ ] Task #71: Implement dual embeddings training script
- [ ] Task #72: Evaluate dual embeddings quality

**Success Metrics:**
- Linguistic correlation >0.85
- Topical correlation >0.65
- t-SNE shows topical clustering
- Manual inspection passes

**Decision Point:**
If evaluation fails → retrain with adjusted hyperparameters
If evaluation passes → proceed to integration

**Dependencies:**
- #71 depends on #68, #70
- #72 depends on #71

---

## Milestone 3: Retrieval Integration (Week 3)

**Deliverable:** AST-aware retriever using dual embeddings with 25-30%+ accuracy

**Tasks:**
- [ ] Task #73: Update SlotBasedIndexer for dual embeddings
- [ ] Task #74: Update HNSW retriever for dual embeddings
- [ ] Task #75: Add adaptive weighting to AST-aware retriever
- [ ] Task #76: Benchmark dual embeddings on Q&A dataset

**Success Metrics:**
- HNSW index rebuilt with 128d embeddings
- Adaptive weighting works
- Benchmark accuracy ≥25% (target 30%)
- Explainability shows 3 components

**Dependencies:**
- #73 depends on #68, #71
- #74 depends on #73
- #75 depends on #74
- #76 depends on #75

---

## Critical Path

```
#68 (DualRootEmbeddings)
  ↓
#69 (CompositionalEmbedding) ──┐
                                ↓
#70 (Topical data) ────────────→ #71 (Training)
                                  ↓
                                #72 (Evaluation)
                                  ↓
                                #73 (SlotIndexer)
                                  ↓
                                #74 (HNSW)
                                  ↓
                                #75 (AST retriever)
                                  ↓
                                #76 (Benchmark)
```

**Total Duration:** ~42-55 hours of work over 3 weeks

---

## Risk Mitigation

**Risk 1:** Topical embeddings don't improve retrieval
- Mitigation: Evaluation gate at Milestone 2
- Fallback: Try different window sizes/loss weights

**Risk 2:** Training takes too long
- Mitigation: Use sequential training (faster)
- Fallback: Train on subset of corpus first

**Risk 3:** Accuracy doesn't reach 25%
- Mitigation: Try different embedding modes/weights
- Fallback: Corpus coverage audit (Task #66)

---

## Success Definition

**Must have:**
- Dual embeddings train without errors
- Retrieval works with all modes
- Accuracy improves over baseline (12%)

**Should have:**
- Accuracy reaches 25%+
- Adaptive weighting outperforms fixed weights
- Explainability working

**Nice to have:**
- Accuracy reaches 30%+
- Users can configure weights
- Migration tool for easy upgrade

---

## Integration with Overall Project

**Current Status:**
- Stage 1: Root embeddings ✅ (single embedding trained)
- Stage 2: Compositional system ✅ (working)
- **Stage 1.5: Dual embeddings** ← THIS PLAN
- Stage 3: Retrieval 🚧 (AST-aware at 12%, needs improvement)

**After Dual Embeddings:**
- Task #65: Test larger prefilter_n (easier with better embeddings)
- Task #66: Audit Q&A corpus coverage
- Task #67: Keyword fallback
- Stage 4: Reasoning (benefits from better retrieval)

**Why Stage 1.5?**
This extends Stage 1 without breaking Stage 2, fixes the retrieval bottleneck identified in AST-aware testing, and maintains backward compatibility.
