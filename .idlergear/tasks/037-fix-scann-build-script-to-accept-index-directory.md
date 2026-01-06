---
id: 37
title: Fix ScaNN build script to accept index directory argument
state: open
created: '2026-01-05T15:22:17.034377Z'
labels:
- bug
- 'priority: medium'
---
## Problem
`./scripts/build_scann_index.sh` fails with:
```
Unknown option: data/indexes/slot_verified
```

The script doesn't accept the index directory as a positional argument.

## Current Behavior
Script expects different arguments format.

## Expected Behavior
Should accept index directory like:
```bash
./scripts/build_scann_index.sh data/indexes/slot_verified
```

## Fix Required
Update `scripts/build_scann_index.sh` to:
- Accept index directory as first positional arg
- Match the pattern used by other build scripts
- Work with `build_verified_indexes.sh`

## Impact
Medium - Prevents automated ScaNN index building
