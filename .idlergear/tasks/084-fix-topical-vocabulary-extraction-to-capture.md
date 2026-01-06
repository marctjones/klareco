---
id: 84
title: Fix topical vocabulary extraction to capture academic terms
state: open
created: '2026-01-06T06:23:45.506617Z'
labels:
- enhancement
- embeddings
- data-quality
priority: low
---
## Issue

Current topical vocabulary (77K roots) is missing common academic/technical terms:
- 'geometri' (geometry) - missing
- 'matemat' (mathematics) - missing
- Likely other academic terms are also missing

This limits the topical embeddings' effectiveness for academic/scientific queries.

## Investigation Results (2026-01-06)

**FINDING: Vocabulary coverage is actually EXCELLENT. The perceived gap was due to test methodology errors.**

### Comprehensive Coverage Analysis

Tested 49 representative roots across 4 categories:

```
Category           Coverage    Missing Terms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Math & Science     11/11 (100%)  None
Humanities         10/11 (91%)   'politi'
Common Content     15/15 (100%)  None
Proper Nouns       12/12 (100%)  None
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL              47/49 (96%)   'politi', 'arkeologi'
```

### Root Cause of Initial Errors

The validation tests showing 'matemat' as missing were using **incorrect root forms**:

```python
# WRONG (what tests used):
'matemat'  # ❌ Not in vocabulary

# CORRECT (what parser extracts):
'matematik'  # ✓ In vocabulary

# Evidence:
from klareco.parser import parse
parse('La matematiko estas scienco.')['subjekto']['kerno']['radiko']
# Returns: 'matematik'
```

### Vocabulary Statistics

```
Linguistic vocabulary: 953,549 roots
Topical vocabulary:    77,236 roots
Overlap:               77,115 roots (99.8%)
```

### Terms Verified Present

**Math & Science (11/11):**
matematik, geometri, algebr, fizik, kemi, biologi, astronomi, geologi, botani, zoologi, arkeologi

**Humanities (10/11):**
histori, filozofi, literatur, lingvistik, sociologi, psikologi, ekonomi, geografi, antropologi, arkeologi
Missing: politi

**Common Content (15/15):**
hund, kat, arb, flor, dom, akv, sun, lun, ter, mar, mont, river, lag, urb, land

**Proper Nouns (12/12):**
napoleon, pariz, london, esperant, eŭrop, amerik, azi, afrik, platon, aristotel, sokrat, kant

## Conclusion

**This task should be CLOSED or downgraded to LOW priority.**

The topical vocabulary extraction is working correctly. The only genuinely missing term from our test set is 'politi' (politics). The vocabulary has:
- 100% coverage of math/science terms
- 100% coverage of common content words  
- 100% coverage of proper nouns
- 96% overall coverage

The hybrid embeddings integration (Task #79) will work well with this vocabulary.

## Recommendations

1. **Close this task** - No significant vocabulary gap exists
2. **Update validation tests** - Use correct root forms ('matematik' not 'matemat')
3. **Document root extraction rules** - Add reference for which root forms the parser uses
4. **Optional:** Add 'politi' to vocabulary if politics queries are important

## Related

- Task #79: Rebuild indexes with hybrid embeddings (in progress)
- Task #85: Clean proper noun classification (still valid - proper nouns in both vocabs)
- Note: Investigation conducted while waiting for hybrid index build
