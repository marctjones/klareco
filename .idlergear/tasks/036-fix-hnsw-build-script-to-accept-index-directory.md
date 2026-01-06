---
id: 36
title: Fix HNSW build script to accept index directory argument
state: open
created: '2026-01-05T15:22:16.857323Z'
labels:
- bug
- 'priority: medium'
---
## Problem
`./scripts/build_hnsw_index.sh` fails with:
```
Unknown option: data/indexes/slot_verified
```

The script doesn't accept the index directory as a positional argument.

## Current Behavior
Script expects different arguments format.

## Expected Behavior
Should accept index directory like:
```bash
./scripts/build_hnsw_index.sh data/indexes/slot_verified
```

## Fix Required
Update `scripts/build_hnsw_index.sh` to:
- Accept index directory as first positional arg
- Match the pattern used by other build scripts
- Work with `build_verified_indexes.sh`

## Impact
Medium - Prevents automated HNSW index building
