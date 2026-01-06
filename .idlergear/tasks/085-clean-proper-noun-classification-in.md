---
id: 85
title: Clean proper noun classification in topical/linguistic vocabularies
state: open
created: '2026-01-06T06:24:07.508498Z'
labels:
- bug
- embeddings
- data-quality
priority: medium
---
## Issue

Some proper nouns exist in BOTH linguistic and topical vocabularies when they should be topical-only:
- **pariz** (Paris) - in both vocabs
- **eŭrop** (Europe) - in both vocabs

This causes:
- Semantic inconsistency in embeddings
- Lower similarity scores than expected (pariz↔london: 0.307 instead of >0.4)
- Confusion about which embedding to use in hybrid mode

## Expected Behavior

**Proper nouns should be:**
- ✅ In topical vocab ONLY (geographic, person names, etc.)
- ❌ NOT in linguistic vocab (which should focus on morphological patterns)

**Content words should be:**
- ✅ In both vocabs (hund, kat, bel, kur, etc.)
- Different purposes: linguistic = morphology, topical = semantics

## Current State (from validation)

```
Root       Linguistic?  Topical?   Expected
pariz      ✓ YES        ✓ YES      Topical only!
napoleon   ✗ NO         ✓ YES      ✓ Correct
esperant   ✗ NO         ✓ YES      ✓ Correct
eŭrop      ✓ YES        ✓ YES      Topical only!
```

## Root Cause

Linguistic vocabulary was built from corpus analysis without filtering proper nouns. Common proper nouns (cities, countries) appeared frequently enough to be included.

## Proposed Solution

### Option 1: Post-process vocabularies (Quick fix)

```python
# Remove proper nouns from linguistic vocab
import json

# Load vocabs
with open('data/vocabularies/root_vocab.json') as f:
    ling_vocab = json.load(f)

with open('data/vocabularies/topical_vocab.json') as f:
    top_vocab = json.load(f)

# Identify proper nouns (in topical but not content words)
proper_nouns = identify_proper_nouns(top_vocab)

# Remove from linguistic
for noun in proper_nouns:
    if noun in ling_vocab['root_to_idx']:
        del ling_vocab['root_to_idx'][noun]
        
# Save cleaned vocab
```

### Option 2: Re-train with proper filtering (Proper fix)

1. Create proper noun detection function
2. Filter during vocabulary extraction
3. Re-train linguistic embeddings (11K → ~10.5K roots)
4. Keep topical embeddings as-is

### Option 3: Smart hybrid mode (Workaround)

Update HybridEmbeddings to:
- Detect proper nouns automatically
- Use topical-only for proper nouns
- Use both for content words

## Impact

**Before fix:**
- pariz↔london: 0.307 (too low)

**After fix (expected):**
- pariz↔london: >0.4 (both use pure topical embeddings)
- Better semantic consistency
- Clearer vocabulary boundaries

## Recommendation

Start with **Option 1** (quick fix):
- Minimal disruption
- No retraining needed
- Can validate improvement quickly

Then move to **Option 2** for next model version.

## Related

- Note #80: Topical model validation results
- Overlap: 5,817 roots (52.3% of linguistic vocab)
- Should reduce overlap to ~40% (content words only)
