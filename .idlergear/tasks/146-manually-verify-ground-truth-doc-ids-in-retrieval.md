---
id: 146
title: Manually verify ground truth doc IDs in retrieval benchmark
state: open
created: '2026-01-08T15:28:36.212660Z'
labels:
- benchmark
- documentation
priority: medium
---
## Problem
The expected_doc_ids in `data/benchmarks/retrieval_benchmark_v1.json` were auto-generated from `get_occurrences()`. They need manual verification to ensure:

1. Documents actually contain the queried concept (not just the root in a different context)
2. Documents are good representative answers to the query
3. Sample size is appropriate (currently 5 per query)

## Verification needed for each tier
- **Tier 1**: Verify docs actually describe the entity (e.g., "elefanto" docs describe elephants)
- **Tier 2**: Verify both roots appear in meaningful conjunction
- **Tier 3**: Verify synonym-only docs are actually relevant
- **Tier 4**: Verify role assignments are correct (e.g., "kur" is actually the verb)
- **Tier 5**: Document why these are inference-required (already noted)

## Process
1. Read each expected doc
2. Confirm relevance to query
3. Replace irrelevant docs with better examples
4. Consider adding more expected docs (10-20 per query)

## File
`data/benchmarks/retrieval_benchmark_v1.json`
