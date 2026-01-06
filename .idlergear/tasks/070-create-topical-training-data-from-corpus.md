---
id: 70
title: Create topical training data from corpus
state: closed
created: '2026-01-05T23:02:53.022297Z'
labels:
- data
- 'priority: high'
priority: high
---
**Phase 2: Data Preparation - Extract skip-gram pairs for topical embeddings**

## Goal
Extract co-occurrence training pairs from 4.3M corpus documents for training topical embeddings.

## Implementation

**File:** `scripts/data/prepare_topical_pairs.py` (NEW - create `scripts/data/` directory)

**Algorithm:**
1. **Pass 1:** Count root frequencies in corpus
2. **Pass 2:** Extract skip-gram pairs
   - For each document, extract roots from AST
   - Filter roots by frequency (min=5 occurrences)
   - Create pairs within window (center, context)
   - Window size: 5 (start with this, experiment later)
3. **Negative sampling:** Add negative pairs (ratio 5:1)
4. **Save:** JSON format with pairs + metadata

**Output format:**
```json
{
  "pairs": [
    ["fundament", "esperant", 1.0],  // positive
    ["fundament", "hund", 0.0],      // negative
    ...
  ],
  "root_freq": {"fundament": 1234, ...},
  "window_size": 5,
  "negative_ratio": 5,
  "total_pairs": 50000000
}
```

**Performance:**
- Process ~4.3M documents
- Extract ~10M positive pairs (estimate)
- Generate ~50M total pairs with negatives
- Save to `data/training/topical_skipgram_pairs.json`

**Validation:**
- Check frequency distribution (Zipf's law)
- Verify window extraction logic
- Sample pairs manually to verify quality
- Check negative sampling is truly random

## Acceptance Criteria
- [ ] Script extracts skip-gram pairs from corpus
- [ ] Frequency filtering works (min=5)
- [ ] Window size configurable (default=5)
- [ ] Negative sampling ratio 5:1
- [ ] Output file created with 50M+ pairs
- [ ] Validation checks pass
- [ ] Script has progress bar for long processing

## Dependencies
- **Blocks:** Training script (#71)
- **Depends on:** Corpus exists (already built)

## Estimated Effort
6-8 hours (including validation)

## References
Design doc Section 2.1

## Notes
Can run in parallel with #68 (no dependencies)
