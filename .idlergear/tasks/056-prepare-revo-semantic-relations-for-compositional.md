---
id: 56
title: Prepare ReVo semantic relations for compositional embedding training
state: open
created: '2026-01-05T16:30:59.860837Z'
labels:
- data-prep
- embeddings
priority: high
---
## Problem
ReVo provides root-level semantic relations (e.g., "humor ≈ humur"), but our embeddings are **compositional** - we embed roots, prefixes, and suffixes separately, then combine them.

**Mismatch**: ReVo has word-level pairs, we need root-level pairs that respect morphology.

## Example Issues

### Issue 1: Full words vs roots
ReVo pair: `"meksik" ≈ "meksiki"` (Mexico country vs Mexican person)

**Problem**: These aren't roots, they're full words with:
- Root: "meksik"
- Suffix: none vs "-i" (person from place)

**What we actually need**: Recognize these are the SAME root with different suffixes, not synonyms.

### Issue 2: Morphological variants
ReVo pair: `"humor" ≈ "humur"`

**Question**: Are these:
- Different roots that are synonyms?
- Same root with spelling variants?
- One is archaic/non-standard?

We need to **decompose into roots** before using as training data.

### Issue 3: Compositional meaning
ReVo pair: `"krei" ≈ "establi"` (create ≈ establish)

**Compositional implications**:
- Should `"kre"` and `"establ"` roots be similar?
- What about derived forms: `"kreinto"` (creator) vs `"establinto"`?
- Should similarity propagate through composition?

## Required Data Preparation

### Step 1: Parse ReVo words into morphemes

For each word in ReVo pairs:
```python
def decompose_word(word):
    ast = parse(word)
    return {
        'root': ast['radiko'],
        'prefix': ast.get('prefikso'),
        'suffixes': ast.get('sufiksoj', []),
        'ending': ast['vortspeco']
    }
```

**Example**:
- Input: `"malkrei"` (uncreate/destroy)
- Output: `{root: "kre", prefix: "mal", suffixes: [], ending: "i"}`

### Step 2: Filter out morphological variants

Remove pairs that are NOT semantic synonyms but morphological variants:
```python
def is_morphological_variant(word1, word2):
    m1 = decompose_word(word1)
    m2 = decompose_word(word2)
    
    # Same root, different affixes = variant, not synonym
    if m1['root'] == m2['root']:
        return True  # REMOVE from synonym pairs
    
    return False
```

**Examples to remove**:
- `"meksik" ≈ "meksiki"` (same root)
- `"krei" ≈ "kreado"` (same root, different ending)

### Step 3: Create root-level synonym pairs

Extract true root-level synonyms:
```python
synonym_roots = []
for (word1, word2) in revo_synonyms:
    m1 = decompose_word(word1)
    m2 = decompose_word(word2)
    
    if m1['root'] != m2['root']:  # Different roots
        # True synonym
        synonym_roots.append((m1['root'], m2['root']))
```

**Example output**:
- `("kre", "establ")` ← from "krei ≈ establi"
- `("humor", "humur")` ← if these are different roots

### Step 4: Handle compositional consistency

If `root1 ≈ root2`, then derived forms should also be similar:
```python
# If "kre" ≈ "establ", then we expect:
# - "krei" ≈ "establi" (verb forms)
# - "kreado" ≈ "establado" (noun forms)
# - "kreinto" ≈ "establinto" (agent nouns)
# - "malkrei" ≈ "malestabli" (negated forms)

def generate_compositional_pairs(root1, root2):
    pairs = []
    
    # Generate with same affixes
    for ending in ['i', 'o', 'a', 'e']:
        pairs.append((root1 + ending, root2 + ending))
    
    for suffix in ['ad', 'ant', 'int', 'it', 'ont']:
        pairs.append((root1 + suffix + 'o', root2 + suffix + 'o'))
    
    for prefix in ['mal', 'ek', 're']:
        pairs.append((prefix + root1 + 'i', prefix + root2 + 'i'))
    
    return pairs
```

**Why this matters**: Our embeddings are compositional. If we train root similarity, it should transfer to all derived forms.

### Step 5: Validate against corpus

Check if generated pairs actually exist in corpus:
```python
def validate_pair(word1, word2, corpus_vocab):
    if word1 not in corpus_vocab:
        return False  # Can't use for training
    if word2 not in corpus_vocab:
        return False
    return True
```

Only use pairs where both words appear in our corpus.

## Output Files

Create prepared datasets:

1. `data/training/revo_root_synonyms.json`
```json
{
  "metadata": {
    "source": "ReVo semantic relations",
    "processing": "Filtered for root-level synonyms only",
    "total_pairs": XXX
  },
  "pairs": [
    {"root1": "kre", "root2": "establ"},
    {"root1": "humor", "root2": "humur"}
  ]
}
```

2. `data/training/revo_compositional_pairs.json`
```json
{
  "metadata": {
    "source": "Generated from root synonyms",
    "description": "All valid compositional variants",
    "total_pairs": XXX
  },
  "pairs": [
    {"word1": "krei", "word2": "establi", "root_pair": ["kre", "establ"]},
    {"word1": "kreado", "word2": "establado", "root_pair": ["kre", "establ"]},
    ...
  ]
}
```

3. `data/training/revo_filtered_out.json` (for analysis)
```json
{
  "morphological_variants": [
    {"word1": "meksik", "word2": "meksiki", "reason": "same root, different suffix"}
  ],
  "missing_from_corpus": [
    {"word1": "rara", "word2": "rara2", "reason": "word2 not in vocabulary"}
  ]
}
```

## Implementation

Create script: `scripts/prepare_revo_for_training.py`

```bash
python scripts/prepare_revo_for_training.py \
  --revo data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
  --corpus data/corpus/unified_corpus.jsonl \
  --output data/training/
```

## Success Criteria
- Extract 500-1000 true root-level synonym pairs
- Generate 3000-5000 compositional variant pairs
- All pairs validated against corpus vocabulary
- Document filtered-out pairs for analysis

## Effort
6-8 hours (parsing + validation + generation)

## Dependencies
- ReVo semantic relations (already downloaded)
- Unified corpus for vocabulary
- Parser for morphological decomposition

## Blocks
- Task #55 (contrastive training needs this data)
- Task #54 (evaluation should use filtered data too)
