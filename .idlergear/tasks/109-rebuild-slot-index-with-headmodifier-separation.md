---
id: 109
title: Rebuild slot index with HEAD/MODIFIER separation
state: open
created: '2026-01-07T00:10:14.055742Z'
labels:
- enhancement
- retrieval
priority: medium
---
## Goal

Rebuild the slot index with the new HEAD/MODIFIER separated format after parser and indexer changes are complete.

## Prerequisites

- [ ] #106 (parser compound word fix) - MUST be complete
- [ ] #107 (slot indexer update) - MUST be complete

## Steps

1. Verify parser correctly extracts compound modifiers
2. Verify indexer correctly separates HEAD from MODIFIER
3. Run full index rebuild: `./scripts/build_slot_index.sh --fresh`
4. Verify new index format
5. Run benchmark to measure improvement

## Estimated Time

- Index build: 4-5 hours (4.4M documents)
- Can run overnight

## Verification

```bash
# Check a known compound word document
grep "Esperanto-klubo" data/indexes/slot_hybrid/slot_index.jsonl | head -1 | jq '.slots'

# Expected:
# {
#   "OBJ_HEAD": [klub embedding],
#   "OBJ_MOD": [esperant embedding],
#   ...
# }
```

## Acceptance Criteria

- [ ] Index rebuilt with new format
- [ ] HEAD/MODIFIER separation verified
- [ ] Retrieval benchmark run
- [ ] Results documented

## Related

Parent: #105 (HEAD/MODIFIER distinction in retrieval)
