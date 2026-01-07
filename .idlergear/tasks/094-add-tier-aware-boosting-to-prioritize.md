---
id: 94
title: Add tier-aware boosting to prioritize authoritative sources
state: open
created: '2026-01-06T22:08:50.189012Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Problem

All sources are weighted equally in retrieval, but authoritative sources (Fundamento, ReVo) should rank higher than general Wikipedia content for factual questions about Esperanto.

## Current Tier System

From slot_index, documents have tier metadata:
- Tier 1: Fundamento (most authoritative)
- Tier 2: ReVo dictionary
- Tier 3: Curated content
- Tier 5: Gutenberg books
- Tier 6: Wikipedia (largest but least authoritative)

## Evidence

Evaluation results show all top results come from tier 6 (Wikipedia):
```json
"source": {
  "tier": 6,
  "name": "wikipedia",
  "weight": 0.5
}
```

For "Kiam aperis la Fundamento?", the answer from Fundamento itself (tier 1) should rank higher than Wikipedia mentions.

## Proposed Solution

Add tier-based boosting in reranking:

```python
TIER_BOOST = {
    1: 1.5,   # Fundamento - 50% boost
    2: 1.3,   # ReVo - 30% boost
    3: 1.2,   # Curated - 20% boost
    5: 1.0,   # Gutenberg - neutral
    6: 0.9,   # Wikipedia - 10% penalty (or neutral)
}

def _apply_tier_boost(self, candidates):
    for score, doc in candidates:
        tier = doc.get('source', {}).get('tier', 6)
        boost = TIER_BOOST.get(tier, 1.0)
        yield (score * boost, doc)
```

## Files to Modify

- `klareco/rag/ast_aware_retriever.py`: Add tier boosting in search pipeline
- Could also apply to other slot retrievers

## Considerations

- Tier boosting should be optional/configurable
- May want different boosts for different question types
- Don't boost too aggressively (Wikipedia has valid answers too)

## Expected Impact

- Authoritative answers rank higher
- Better precision for Esperanto-specific factual questions

## Acceptance Criteria

- [ ] Tier boost factors defined
- [ ] Boosting applied in reranking stage
- [ ] Configurable on/off
- [ ] Evaluation shows improvement for factual questions
