---
id: 59
title: Prepare affix training data from corpus patterns
state: open
created: '2026-01-05T16:40:53.466263Z'
labels:
- data-prep
- embeddings
- affixes
priority: high
---
## Goal
Extract training examples from the corpus that demonstrate clear semantic transformations caused by affixes, to improve affix embeddings.

## The Problem
Current affix embeddings may not capture semantic transformations well because they were trained on co-occurrence patterns, not explicit transformation examples.

**We need**: Pairs showing the same root with/without affixes to learn what each affix DOES semantically.

## Examples of Good Affix Transformations

### 1. Negation Prefix (mal-)
Clear semantic reversal:
```
san/a (healthy) → malsan/a (sick)
bon/a (good) → malbon/a (bad)
grand/a (big) → malgrand/a (small)
```

**Training signal**: These pairs should have OPPOSITE meanings but same root.

### 2. Place Suffix (-ej)
Consistent "place where" transformation:
```
lern/i (to learn) → lern/ej/o (school - place where one learns)
labor/i (to work) → labor/ej/o (workplace)
kuir/i (to cook) → kuir/ej/o (kitchen)
```

**Training signal**: The -ej form should cluster with other "place" words.

### 3. Agent Suffix (-ist, -ant, -int)
Person who does the action:
```
art/o (art) → art/ist/o (artist)
kant/i (to sing) → kant/ist/o (singer)
labor/i (to work) → labor/ant/o (worker - currently working)
labor/i (to work) → labor/int/o (worker - worked in past)
```

**Training signal**: The -ist/-ant forms should cluster with "person/profession" words.

### 4. Causative Suffix (-ig)
Make/cause to be X:
```
san/a (healthy) → san/ig/i (to heal - make healthy)
pura/a (clean) → pur/ig/i (to clean - make clean)
mort/a (dead) → mort/ig/i (to kill - make dead)
```

**Training signal**: -ig adds causative/transitive meaning.

### 5. Size Modifiers (-et, -eg)
Clear semantic scaling:
```
dom/o (house) → dom/et/o (cottage) / dom/eg/o (mansion)
pluv/o (rain) → pluv/et/o (drizzle) / pluv/eg/o (downpour)
rid/i (laugh) → rid/et/i (chuckle) / rid/eg/i (guffaw)
```

**Training signal**: -et and -eg are opposite transformations.

## Data Extraction Strategy

### Step 1: Find Root Pairs in Corpus

For each affix, find corpus sentences containing both root and root+affix:

```python
def extract_affix_pairs(corpus, affix, affix_type):
    """
    Find sentences where both root and root+affix appear.
    
    Args:
        corpus: Parsed corpus with ASTs
        affix: Affix to search for (e.g., "mal", "ej", "ist")
        affix_type: "prefix" or "suffix"
    
    Returns:
        List of (root, root+affix, context) tuples
    """
    pairs = []
    
    for doc in corpus:
        # Parse to get roots and affixes
        words = extract_words_from_ast(doc['ast'])
        
        roots_seen = set()
        affixed_words = {}
        
        for word in words:
            root = word['radiko']
            roots_seen.add(root)
            
            if affix_type == "prefix" and word.get('prefikso') == affix:
                affixed_words[root] = word['text']
            elif affix_type == "suffix" and affix in word.get('sufiksoj', []):
                affixed_words[root] = word['text']
        
        # Find roots that appear both with and without affix
        for root in roots_seen:
            if root in affixed_words:
                pairs.append({
                    'root': root,
                    'base_form': root + 'i',  # Or actual form from corpus
                    'affixed_form': affixed_words[root],
                    'context': doc['text'],
                    'affix': affix
                })
    
    return pairs
```

### Step 2: Validate Semantic Relationships

Not all root+affix pairs are valid training examples. Filter by:

```python
def validate_affix_pair(root, affixed_word, affix, affix_type):
    """Check if this is a valid semantic transformation."""
    
    # 1. Check both words exist in vocabulary
    if not (root in vocab and affixed_word in vocab):
        return False, "missing from vocab"
    
    # 2. Check frequency (need enough examples)
    if word_frequency[root] < 10 or word_frequency[affixed_word] < 10:
        return False, "too rare"
    
    # 3. Check semantic relationship makes sense
    if affix == "mal":
        # Should be opposites (checked via ReVo antonyms if available)
        if (root, affixed_word) in revo_antonyms:
            return True, "confirmed antonym"
        # Or check corpus contexts suggest opposition
    
    if affix == "ej":
        # Should relate to place
        # Check if affixed_word appears near place words in corpus
        pass
    
    return True, "validated"
```

### Step 3: Generate Triplets for Contrastive Learning

For each valid pair, create training triplets:

```python
def generate_affix_triplets(affix_pairs, affix, affix_type):
    """
    Create triplets for contrastive learning.
    
    For prefix (e.g., mal-):
        (root_emb, affixed_emb, target_relationship)
    
    For suffix (e.g., -ej):
        (root_emb, affixed_emb, target_relationship)
    """
    triplets = []
    
    for pair in affix_pairs:
        root = pair['root']
        affixed_form = pair['affixed_form']
        
        if affix == "mal":
            # Negation: should push embeddings apart
            triplets.append({
                'anchor': root,
                'positive': None,  # No positive example
                'negative': affixed_form,  # These should be OPPOSITE
                'affix': affix,
                'relationship': 'antonym'
            })
        
        elif affix in ["ej", "ist", "ant"]:
            # Transformation: both related but different semantic role
            # Need to train affix vector to do the transformation
            triplets.append({
                'root': root,
                'affixed': affixed_form,
                'affix': affix,
                'relationship': 'transformation',
                'target_cluster': get_cluster_target(affix)  # e.g., "place" for -ej
            })
    
    return triplets
```

### Step 4: Extract Affix Context Patterns

Find what contexts each affix typically appears in:

```python
def extract_affix_contexts(corpus, affix):
    """
    Find typical contexts for affixed words.
    
    E.g., for -ej:
        - "iris al la [lernejo]" (went to the school)
        - "laboris en [fabriko/laborejo]" (worked in factory/workplace)
    
    This helps train affix embeddings to cluster appropriately.
    """
    contexts = []
    
    for doc in corpus:
        for word in doc['words']:
            if has_affix(word, affix):
                # Extract context window (±5 words)
                context = get_context_window(doc, word, window=5)
                contexts.append({
                    'word': word['text'],
                    'root': word['radiko'],
                    'affix': affix,
                    'context': context
                })
    
    return contexts
```

## Output Files

### 1. Affix Pair Dataset
`data/training/affix_pairs.json`

```json
{
  "mal": {
    "prefix": true,
    "semantic_type": "negation",
    "total_pairs": XXX,
    "pairs": [
      {
        "root": "san",
        "base_form": "sana",
        "affixed_form": "malsana",
        "contexts": ["La malsana hundo...", "Ŝi fariĝis sana..."],
        "validated": true
      }
    ]
  },
  "ej": {
    "suffix": true,
    "semantic_type": "place",
    "total_pairs": XXX,
    "pairs": [...]
  },
  "ist": {...},
  "ig": {...},
  "et": {...},
  "eg": {...}
}
```

### 2. Affix Transformation Triplets
`data/training/affix_triplets.json`

```json
{
  "mal_negation": [
    {
      "root": "san",
      "affixed": "malsan",
      "relationship": "antonym",
      "training_signal": "push_apart"
    }
  ],
  "ej_place": [
    {
      "root": "lern",
      "affixed": "lernejo",
      "relationship": "transformation",
      "target_cluster": "place_words",
      "cluster_examples": ["domo", "urbo", "lando", "loko"]
    }
  ]
}
```

### 3. Affix Context Patterns
`data/training/affix_contexts.json`

```json
{
  "ej": {
    "common_patterns": [
      "al la [WORD]",
      "en la [WORD]",
      "de la [WORD]"
    ],
    "example_contexts": [...]
  }
}
```

## Implementation

Create script: `scripts/prepare_affix_training_data.py`

```bash
python scripts/prepare_affix_training_data.py \
  --corpus data/corpus/unified_corpus.jsonl \
  --output data/training/ \
  --affixes mal,ej,ist,ig,et,eg,ant,int,ul,ad
```

## Expected Statistics

From 4.2M sentence corpus:
- **mal-** pairs: 5,000-10,000 (very common)
- **-ej** pairs: 1,000-3,000
- **-ist** pairs: 2,000-4,000
- **-ig** pairs: 3,000-5,000
- **-et/-eg** pairs: 500-1,500

## Success Criteria
- Extract 15,000+ valid affix transformation pairs
- Each major affix has 500+ examples
- All pairs validated (both words in vocab, sufficient frequency)
- Context patterns identified for each affix

## Effort
8-10 hours (extraction + validation + formatting)

## Dependencies
- Unified corpus with ASTs
- Parser for morphological decomposition
- ReVo antonyms (for mal- validation)

## Blocks
- Task #59 (affix improvement training)
