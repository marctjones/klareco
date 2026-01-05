---
id: 32
title: Research optimal Wikipedia tier weight for retrieval
state: open
created: '2026-01-05T01:09:47.308890Z'
labels:
- research
- retrieval
priority: low
---
## Problem

Wikipedia is currently assigned tier 6 with weight 0.5 in `build_enhanced_corpus.py`:

```python
TIER_MAP = {
    'fundamento_ekzercaro': 1,        # weight 3.0
    'fundamenta_krestomatio': 2,      # weight 2.5
    'gerda_malaperis': 3,             # weight 2.0
    # ... tiers 4-5 undefined
    'wikipedia': 6,                    # weight 0.5
}
```

**Questions**:
1. Is tier 6 (weight 0.5) appropriate for Wikipedia?
2. Should Wikipedia have higher priority than tier 6?
3. How does Wikipedia quality compare to Gutenberg books?

## Research Tasks

1. **Evaluate Wikipedia quality**:
   - Parse rate on sample (1000 articles)
   - Average sentence length
   - Vocabulary coverage
   - Grammar complexity

2. **Compare to other sources**:
   - Gutenberg books: tier assignment and quality
   - What are tiers 4-5? (currently undefined)

3. **Test different weights**:
   - Try weights: 0.3, 0.5 (current), 0.7, 1.0
   - Measure retrieval accuracy on test queries
   - Measure diversity of results

## Expected Outcome

Recommendation for optimal Wikipedia tier/weight based on:
- Content quality metrics
- Retrieval performance
- Result diversity

## Impact

- Better calibrated retrieval scoring
- More relevant results for factual queries
- Balanced representation across sources
