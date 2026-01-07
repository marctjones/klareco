---
id: 106
title: Fix parser compound word handling to capture modifier components
state: open
created: '2026-01-07T00:10:12.296297Z'
labels:
- bug
- parser
priority: high
---
## Problem

The parser correctly identifies the HEAD of compound words but does NOT capture the modifier components in `priskriboj`.

**Current behavior:**
```python
parse("Li fondis Esperanto-klubon.")
# objekto.kerno.radiko = "klub" ✓ (correct HEAD)
# objekto.priskriboj = []  ✗ (missing "esperant" modifier!)
```

**Expected behavior:**
```python
parse("Li fondis Esperanto-klubon.")
# objekto.kerno.radiko = "klub" (HEAD)
# objekto.priskriboj = [{"radiko": "esperant", "tipo": "kunmetaĵo"}]  # compound modifier
```

## Another Bug Found

```python
parse("La Esperanto-Asocio estas granda.")
# subjekto.kerno.radiko = "esperant"  ✗ WRONG! Should be "asoci"
```

The parser is incorrectly treating "Esperanto" as the head instead of "Asocio".

## Root Cause

Compound word parsing in `parser.py` needs to:
1. Split hyphenated words correctly
2. Identify the rightmost component as HEAD
3. Store left components as compound modifiers (kunmetaĵoj)

## Acceptance Criteria

- [ ] Hyphenated compounds split correctly: "Esperanto-klubo" → HEAD="klub", MOD=["esperant"]
- [ ] Rightmost component is always HEAD: "hundo-domo" → HEAD="dom", MOD=["hund"]
- [ ] Multi-part compounds work: "ferro-vojo-stacio" → HEAD="staci", MOD=["fer", "voj"]
- [ ] Modifier stored in priskriboj or new kunmetaĵoj field
- [ ] Tests added for compound word parsing

## Related

Parent: #105 (HEAD/MODIFIER distinction in retrieval)
