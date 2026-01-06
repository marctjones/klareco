---
id: 76
title: Benchmark dual embeddings on Q&A dataset
state: open
created: '2026-01-05T23:04:30.977696Z'
labels:
- evaluation
- 'priority: medium'
priority: medium
---
**Phase 6: Evaluation - Measure retrieval improvement with dual embeddings**

## Goal
Run comprehensive benchmark to compare dual embeddings vs single embedding baseline.

## Implementation

**Benchmark configurations:**

1. **Baseline (existing):**
   - Single 64d embedding (current model)
   - Accuracy: 12% (6/50 found)

2. **Dual - Linguistic only:**
   - Use only linguistic embedding (64d)
   - Test if linguistic alone works

3. **Dual - Topical only:**
   - Use only topical embedding (64d)
   - Test if topical alone helps

4. **Dual - Combined (50/50):**
   - Use both with equal weighting
   - Fixed weights, no adaptation

5. **Dual - Adaptive:**
   - Query-type-based weighting
   - Different weights per question type

**Commands:**
```bash
# Baseline (already done)
# Results: 12% accuracy

# Linguistic only
python scripts/benchmark_qa_enhanced.py \
  --retriever ast \
  --embedding-mode linguistic \
  --output results/dual_linguistic.jsonl

# Topical only
python scripts/benchmark_qa_enhanced.py \
  --retriever ast \
  --embedding-mode topical \
  --output results/dual_topical.jsonl

# Combined 50/50
python scripts/benchmark_qa_enhanced.py \
  --retriever ast \
  --embedding-mode combined \
  --output results/dual_combined.jsonl

# Adaptive
python scripts/benchmark_qa_enhanced.py \
  --retriever ast \
  --embedding-mode adaptive \
  --output results/dual_adaptive.jsonl
```

**Analysis:**
- Compare accuracy across all configurations
- Analyze per-question-type performance
- Check if topical helps factual questions
- Check if linguistic helps definition questions
- Identify failure patterns

**Success criteria:**
- **Must achieve:** 25%+ accuracy (baseline 12%)
- **Target:** 30%+ accuracy
- **Stretch:** 40%+ accuracy

## Acceptance Criteria
- [ ] All 5 configurations benchmarked
- [ ] Accuracy comparison table generated
- [ ] Per-question-type analysis complete
- [ ] Adaptive outperforms fixed 50/50
- [ ] At least 25% accuracy achieved
- [ ] Report written with findings

## Dependencies
- **Depends on:** AST retriever update (#75)

## Estimated Effort
6-8 hours (including analysis)

## References
Design doc Section 5, Phase 6

## Deliverable
Comprehensive report showing:
- Accuracy by configuration
- Per-question-type breakdown
- Example successes vs failures
- Recommendation for production config
