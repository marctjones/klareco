---
id: 61
title: Prepare ReVo semantic relations for affix training validation
state: open
created: '2026-01-05T16:46:44.848716Z'
labels:
- data-prep
- embeddings
- affixes
priority: high
---
## Goal
Use ReVo's semantic relation data (antonyms, hypernyms, part_of, etc.) to validate and improve affix training data from Task #59.

## The Opportunity
ReVo provides 173 **antonym pairs** and other semantic relations that can:
1. Validate mal- prefix transformations
2. Validate hypernym/hyponym relationships for improved understanding
3. Provide gold-standard test cases for affix evaluation

**Currently unused**: This valuable data is not being leveraged in Tasks #59-60.

## Available ReVo Relations

From `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`:
- **173 antonym pairs**: e.g., "kaŝ ↔ montr" (hide ↔ show), "sub ↔ super" (under ↔ above)
- **3,351 hypernym pairs**: e.g., "hundo → besto" (dog → animal)
- **1,098 hyponym pairs**: inverse of hypernyms
- **598 part_of pairs**: e.g., "ringo → fingro" (ring → finger)
- **2,141 has_part pairs**: inverse of part_of

## Use Cases for Affix Training

### Use Case 1: Validate mal- Prefix with Antonyms

**Problem**: How do we know if corpus-extracted mal- pairs are true opposites?

**Solution**: Cross-reference with ReVo antonyms.

```python
def validate_mal_pairs_with_revo(corpus_mal_pairs, revo_antonyms):
    """
    Check if corpus mal- pairs match ReVo's antonym data.
    
    Example:
        Corpus: "bona" → "malbona"
        ReVo antonyms: Check if ("bon", "malbon") is listed
    """
    validated = []
    corpus_only = []
    
    for (base, negated) in corpus_mal_pairs:
        # Parse to roots
        base_root = parse_root(base)
        negated_root = parse_root(negated)  # Should be same root
        
        # Check if this base has a ReVo antonym
        revo_antonyms_for_base = find_revo_antonyms(base_root, revo_antonyms)
        
        if revo_antonyms_for_base:
            # Gold standard: ReVo confirms this is an antonym relationship
            validated.append({
                'base': base,
                'negated': negated,
                'status': 'confirmed_by_revo',
                'revo_antonyms': revo_antonyms_for_base
            })
        else:
            # Only in corpus, not in ReVo
            corpus_only.append({
                'base': base,
                'negated': negated,
                'status': 'corpus_only'
            })
    
    return validated, corpus_only
```

**Example findings**:
- "kaŝi" (hide) ↔ "montri" (show) - ReVo antonyms
- Check if "malkaŝi" (reveal) is used in corpus
- If yes: mal-kaŝi should be similar to "montri"

### Use Case 2: Find Missing mal- Pairs

**Discovery**: ReVo antonyms where one side has mal- prefix:

```python
def find_mal_antonyms_in_revo(revo_antonyms):
    """
    Find ReVo antonym pairs where one word is mal-X.
    
    Example: If ReVo has "nova ↔ malnova", extract this as
    a mal- transformation example.
    """
    mal_pairs = []
    
    for (word1, word2) in revo_antonyms:
        if word2.startswith('mal'):
            # word2 is negation of something
            base = word2[3:]  # Remove 'mal'
            if base == word1:
                mal_pairs.append({
                    'base': word1,
                    'negated': word2,
                    'source': 'revo_antonym'
                })
        elif word1.startswith('mal'):
            base = word1[3:]
            if base == word2:
                mal_pairs.append({
                    'base': word2,
                    'negated': word1,
                    'source': 'revo_antonym'
                })
    
    return mal_pairs
```

**Expected output**: 20-50 mal- pairs directly from ReVo antonyms.

### Use Case 3: Antonym Test Cases for Affix Evaluation

Use ReVo antonyms as **gold standard test cases** for Task #58:

```python
def create_antonym_test_cases(revo_antonyms):
    """
    Convert ReVo antonyms to test cases for mal- evaluation.
    
    For Task #58: Test if mal- correctly reverses polarity.
    """
    test_cases = []
    
    for (word1, word2) in revo_antonyms:
        # These should have LOW similarity
        test_cases.append({
            'word1': word1,
            'word2': word2,
            'relationship': 'antonym',
            'expected_similarity': '<0.3',
            'source': 'revo'
        })
    
    return test_cases
```

**Example test**: 
- sim("kaŝi", "montri") should be <0.3 (hide vs show)
- If our embeddings show sim > 0.5, they're broken

### Use Case 4: Hypernym/Hyponym for Hierarchical Understanding

**Potential use**: Train embeddings to understand is-a relationships.

```python
def extract_hypernym_hierarchy(revo_hypernyms):
    """
    Build semantic hierarchy from ReVo hypernyms.
    
    Example:
        hundo → besto → vivaĵo
        (dog → animal → living thing)
    """
    hierarchy = {}
    
    for (specific, general) in revo_hypernyms:
        if specific not in hierarchy:
            hierarchy[specific] = []
        hierarchy[specific].append(general)
    
    return hierarchy
```

**Training signal**: 
- emb("hundo") should be "near but more specific than" emb("besto")
- Could use this for improved retrieval (search "besto" finds "hundo")

### Use Case 5: Part-of Relations for Compositional Understanding

**Example pairs**: 
- "ringo → fingro" (ring → finger)
- "tegmento → domo" (roof → house)

**Could use for**: Understanding part-whole relationships in queries.

## Data Preparation Tasks

### Task 1: Parse ReVo Antonyms

```python
def parse_revo_antonyms_for_mal():
    """
    Extract and categorize ReVo antonyms.
    
    Categories:
    1. mal- pairs: word1 ↔ mal-word1
    2. Lexical antonyms: bona ↔ malbona (not mal- based)
    3. Other opposites: sub ↔ super
    """
    pass
```

### Task 2: Cross-Reference with Corpus

```python
def cross_reference_corpus_and_revo(corpus_pairs, revo_antonyms):
    """
    Find overlap and gaps between corpus and ReVo.
    
    Output:
    - Both: Confirmed antonyms
    - Corpus only: Need validation
    - ReVo only: Missing from corpus (rare words?)
    """
    pass
```

### Task 3: Generate Augmented Training Data

Combine corpus + ReVo for richer training:

```python
def augment_affix_training_data(corpus_pairs, revo_relations):
    """
    Create enhanced training dataset.
    
    Priority tiers:
    - P0: Confirmed by both corpus and ReVo (highest confidence)
    - P1: Corpus only with high frequency
    - P2: ReVo only (add even if not in corpus, for rare words)
    """
    pass
```

## Output Files

### 1. ReVo Antonym Pairs for mal-
`data/training/revo_mal_antonyms.json`

```json
{
  "metadata": {
    "source": "ReVo antonym relations",
    "total_pairs": XXX,
    "mal_prefix_pairs": XXX,
    "lexical_antonyms": XXX
  },
  "mal_pairs": [
    {
      "base": "nova",
      "negated": "malnova",
      "base_root": "nov",
      "in_corpus": true,
      "source": "revo_antonym"
    }
  ],
  "lexical_antonyms": [
    {
      "word1": "kaŝi",
      "word2": "montri",
      "relationship": "antonym",
      "use_for_testing": true
    }
  ]
}
```

### 2. Cross-Reference Report
`data/training/corpus_revo_crossref.json`

```json
{
  "mal_pairs": {
    "confirmed_by_both": 50,
    "corpus_only": 4950,
    "revo_only": 30,
    "total": 5030
  },
  "confirmed_examples": [
    {"base": "bona", "negated": "malbona", "sources": ["corpus", "revo"]}
  ],
  "needs_validation": [
    {"base": "X", "negated": "malX", "source": "corpus_only"}
  ]
}
```

### 3. Gold Standard Test Cases
`data/training/revo_antonym_tests.json`

```json
{
  "antonym_tests": [
    {
      "word1": "kaŝi",
      "word2": "montri",
      "expected_similarity": "<0.3",
      "test_type": "lexical_antonym",
      "source": "revo"
    }
  ]
}
```

## Implementation

Create script: `scripts/prepare_revo_affix_data.py`

```bash
python scripts/prepare_revo_affix_data.py \
  --revo data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
  --corpus-pairs data/training/affix_pairs.json \
  --output data/training/
```

## Integration Points

This data will be used by:

1. **Task #59** (Affix data prep): 
   - Use ReVo antonyms to validate mal- pairs
   - Add ReVo-only pairs as P2 training data

2. **Task #58** (Affix evaluation):
   - Use ReVo antonyms as gold standard test cases
   - Test if model correctly identifies opposites

3. **Task #60** (Affix training):
   - Prioritize confirmed pairs (corpus + ReVo) in training
   - Use ReVo antonyms for harder negative examples

## Expected Statistics

From 173 ReVo antonym pairs:
- **mal- prefix pairs**: 20-40 pairs (e.g., "nova ↔ malnova")
- **Lexical antonyms**: 100-130 pairs (e.g., "kaŝi ↔ montri")
- **Confirmed with corpus**: 50-80 pairs
- **ReVo-only rare words**: 20-30 pairs

## Success Criteria
- Extract all 173 antonym pairs from ReVo
- Categorize by type (mal- vs lexical)
- Cross-reference with corpus pairs (find overlap)
- Generate 100+ gold standard test cases
- Augment affix training data with ReVo confirmations

## Effort
4-6 hours (extraction + cross-referencing + formatting)

## Dependencies
- ReVo semantic relations (already downloaded)
- Task #59 corpus pairs (for cross-referencing)

## Blocks
- Task #58 (affix evaluation - needs gold standard tests)
- Task #59 (affix data prep - needs validation data)
- Task #60 (affix training - needs prioritized training data)
