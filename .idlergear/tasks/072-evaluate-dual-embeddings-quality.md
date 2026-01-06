---
id: 72
title: Evaluate dual embeddings quality
state: open
created: '2026-01-05T23:03:27.924977Z'
labels:
- evaluation
- 'priority: high'
priority: high
---
**Phase 4: Evaluation - Validate dual embeddings before integration**

## Goal
Intrinsic evaluation to verify both linguistic and topical embeddings have good quality before integrating into retrieval.

## Implementation

**File:** `scripts/training/evaluate_dual_embeddings.py` (NEW)

**Evaluation metrics:**

1. **Linguistic correlation** (existing metric)
   - Test on ReVo validation pairs
   - Target: >0.85 (maintain current quality)

2. **Topical correlation** (new metric)
   - Test on held-out topical pairs
   - Target: >0.65

3. **Manual inspection**
   - Check semantic clusters: "Esperanto", "Zamenhof", "lingvo", "internacia"
   - Check topical proximity: "Fundamento" + "1905"
   - Print nearest neighbors for key words

4. **t-SNE visualization**
   - Plot linguistic embeddings
   - Plot topical embeddings
   - Check if topical shows topic clustering

5. **Probe tests**
   - Query: "Words related to Esperanto history"
   - Expected cluster: Fundamento, Zamenhof, 1887, 1905, lingvo
   - Measure precision@10

**Output:**
- Correlation scores report
- t-SNE plots saved to `results/embeddings/`
- Nearest neighbors JSON
- Pass/fail decision

## Acceptance Criteria
- [ ] Script evaluates both embeddings separately
- [ ] Linguistic correlation >0.85
- [ ] Topical correlation >0.65
- [ ] t-SNE plots generated
- [ ] Manual inspection shows expected clustering
- [ ] Probe tests show topical grouping
- [ ] Decision: proceed to integration or retrain

## Dependencies
- **Blocks:** Retrieval integration (#73-75)
- **Depends on:** Trained dual model (#71)

## Estimated Effort
4-6 hours

## References
Design doc Section 6 (Open Questions - Q6)

## Decision Point
If metrics fail:
- Adjust hyperparameters (window size, loss weights)
- Retrain with different strategy
- Do NOT proceed to integration
