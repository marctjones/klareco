---
id: 22
title: "Fix parser conditional (kondi\u0109a) sentence detection"
state: closed
created: '2026-01-05T00:12:56.342421Z'
labels:
- bug
- parser
priority: low
---
**Problem**: Parser incorrectly labels conditional sentences as `deklaro` instead of `kondiĉa`.

**Test case**:
```python
parse("Se mi havus tempon, mi venus.")
# Expected: fraztipo='kondiĉa'
# Actual: fraztipo='deklaro'
```

**Detection rules needed**:
- Sentence starts with "Se" + conditional mood verb (havus, venus, estus)
- Main clause uses conditional mood (-us ending)

**Why important**: 
- Conditionals express hypotheticals, not facts
- May need different retrieval scoring (TBD after fix)
- Already affects ~60-80% of conditional sentences in corpus

**Dependencies**: None
**Priority**: Medium (P2)
**Labels**: bug, parser
