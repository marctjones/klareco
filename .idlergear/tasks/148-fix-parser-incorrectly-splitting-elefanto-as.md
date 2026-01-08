---
id: 148
title: Fix parser incorrectly splitting "elefanto" as elef+ant suffix
state: closed
created: '2026-01-08T15:35:17.508481Z'
labels:
- bug
- parser
priority: medium
---
## Problem
The morphology analyzer is over-aggressively finding suffixes. It parses "elefanto" as:
- radiko: `elef`
- sufiksoj: `['ant']` (active participle)

This is wrong. "elefanto" should be:
- radiko: `elefant`
- sufiksoj: `[]`

## Root Cause
`elefant` is in `KNOWN_ROOTS` but NOT in `PROTECTED_SUFFIX_ROOTS`.

The parser sees `-ant-` suffix and splits it because `elef` is also a valid root in `KNOWN_ROOTS`.

## Fix
Add `-ant` look-alikes to `PROTECTED_SUFFIX_ROOTS` in `klareco/parser.py`:

```python
# -ant look-alikes (-ant means "active participle present")
"elefant", "gigant", "infant", "pedant", "diamant", "briliant", "merkant",
"dilettant", "konsultant", "protestant", "ignorant", "elegant", "arrogant",
```

This is the same pattern used for other suffix look-alikes like `-id`, `-et`, `-il`, etc.

## Impact on Retrieval
- `elef` has 1,206 docs in index
- `elefant` has 73 docs in index

## Files
- `klareco/parser.py` - add to `PROTECTED_SUFFIX_ROOTS` set around line 592

## Related Words to Test
- elefanto → should be elefant+o
- giganto → should be gigant+o  
- infanto → should be infant+o
- pedanto → should be pedant+o
- merkanto → should be merkant+o (merchant)
