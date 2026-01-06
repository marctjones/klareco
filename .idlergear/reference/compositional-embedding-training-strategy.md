---
id: 1
title: Compositional Embedding Training Strategy
created: '2026-01-05T19:43:10.790637Z'
updated: '2026-01-05T19:43:10.790652Z'
---
# Compositional Embedding Training Strategy

This document describes how to evaluate and improve Klareco's compositional embeddings using the ReVo thesaurus and corpus-extracted patterns.

## Overview

Klareco uses **compositional embeddings** where words are built from:
- **Root embeddings** (64d, learned): Semantic meaning - e.g., "hund" (dog)
- **Affix embeddings** (8d each, learned): Transformations - e.g., "mal-" (opposite), "-ej" (place)
- **Grammatical features** (16d, deterministic): Case, number, tense - programmatic, not learned

**Total**: 128d per word, but only ~500K parameters to learn (roots + affixes).

## Key Architectural Principle

**Train components separately at the correct level**:
1. **Roots**: Train semantic similarity between root embeddings
2. **Affixes**: Train transformation functions (what each affix DOES)
3. **Never train full composed words**: Would break compositionality

## Data Sources

### ReVo (Reta Vortaro) Thesaurus
- Location: `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`
- **1,943 synonym pairs**: "humor ≈ humur", "krei ≈ establi"
- **173 antonym pairs**: "kaŝ ↔ montr" (hide ↔ show), "sub ↔ super"
- **3,351 hypernym pairs**: "hundo → besto" (dog → animal)
- **1,098 hyponym pairs**: Inverse of hypernyms
- **598 part_of pairs**: "ringo → fingro" (ring → finger)

### Unified Corpus
- Location: `data/corpus/unified_corpus.jsonl`
- **4.2M parsed sentences** with AST annotations
- Extract affix transformation patterns: sentences with both "san" and "malsan"

## ROOT Embedding Training

### Evaluation (Task #54)

**Script**: `scripts/evaluate_embeddings_with_revo.py`

**Tests**:
1. **Synonym similarity**: For 1,943 synonym pairs, compute cosine similarity
   - Target: Mean similarity >0.7
   - Bad: <0.5 (not capturing synonyms)

2. **Baseline comparison**: Compare synonym pairs vs random word pairs
   - Target: Gap >0.2
   - Bad: <0.1 (synonyms not distinctive)

3. **Antonym discrimination**: 173 antonym pairs should have LOW similarity
   - Target: Mean <0.4
   - Bad: >0.6 (can't distinguish opposites)

**Output**: `benchmark_results/embeddings/revo_evaluation_TIMESTAMP.json`

### Data Preparation (Task #56)

**Script**: `scripts/prepare_revo_for_training.py`

**Steps**:
1. **Parse ReVo words**: Decompose into roots/affixes using parser
   - Input: "meksik" ≈ "meksiki"
   - Analysis: Same root "meksik", different suffix (none vs "-i")
   - Decision: REMOVE (morphological variant, not synonym)

2. **Extract root-level synonyms**: Only different roots
   - Input: "krei" ≈ "establi"
   - Roots: "kre" ≈ "establ"
   - Output: Root synonym pair

3. **Generate compositional pairs**: If roots are synonyms, all derived forms should be
   - If "kre" ≈ "establ", then:
     - "krei" ≈ "establi" (verbs)
     - "kreado" ≈ "establado" (nouns)
     - "malkrei" ≈ "malestabli" (negated)
     - etc.

4. **Validate against corpus**: Only use pairs where both words exist

**Output**:
- `data/training/revo_root_synonyms.json`: 500-1000 root pairs
- `data/training/revo_compositional_pairs.json`: 3000-5000 word pairs

### Training Design (Task #57)

**Key decision**: Train at ROOT level only, not word level

**WRONG approach**:
```python
# BAD: Trains composed embeddings, doesn't generalize
loss = triplet_loss(embed("krei"), embed("establi"), embed("random"))
```

**CORRECT approach**:
```python
# GOOD: Trains roots, automatically transfers to all forms
root1 = "kre"
root2 = "establ"
loss = triplet_loss(root_emb[root1], root_emb[root2], root_emb[random_root])
```

**Why this works**:
- Root similarity → All derived forms similar (compositionality)
- Affixes stay frozen (deterministic meanings)
- Generalizes to unseen combinations

### Training Implementation (Task #55)

**Script**: `scripts/train_embeddings_contrastive.py`

**Architecture**:
```python
class CompositionalContrastiveLearner:
    def __init__(self, base_model):
        self.trainable = [base_model.root_embeddings]  # Only roots
        # Freeze affixes
        for param in base_model.affix_embeddings.parameters():
            param.requires_grad = False
```

**Loss function**: Triplet loss
```python
loss = max(0, margin + sim(anchor, negative) - sim(anchor, positive))
```

**Training data**:
- Positive pairs: ReVo root synonyms
- Negative samples: Random roots from vocabulary

**Expected improvement**:
- Mean synonym similarity: 0.55 → >0.75
- Synonym vs random gap: 0.10 → >0.25

**Output**: `models/root_embeddings/contrastive_model.pt`

## AFFIX Embedding Training

### Evaluation (Task #58)

**Script**: `scripts/evaluate_affix_semantics.py`

**Tests by affix type**:

1. **mal- (negation)**: Should reverse polarity
   - Test: sim("bona", "malbona") 
   - Target: <0.3 (opposites)
   - Consistency: mal_vector std <0.15

2. **-ej (place)**: Should cluster with place words
   - Test: sim("lernejo", place_cluster_center)
   - Target: >0.7
   - Consistency: ej_vector std <0.15

3. **-ist (agent)**: Should cluster with person/profession words
   - Test: sim("artisto", person_cluster_center)
   - Target: >0.7

4. **-ig (causative)**: Should add "make/cause" meaning
   - Test: emb("sanigi") ≈ emb("sana") + ig_vector
   - Target: Transformation loss <0.2

5. **-et/-eg (size)**: Should be opposite transformations
   - Test: sim(et_vector, eg_vector)
   - Target: <-0.3 (opposite directions)

**Output**: `benchmark_results/embeddings/affix_evaluation_TIMESTAMP.json`

### Data Preparation - Corpus Patterns (Task #59)

**Script**: `scripts/prepare_affix_training_data.py`

**Strategy**: Find sentences with BOTH root and root+affix

**Example for mal-**:
```python
# Find documents with both "sana" and "malsana"
# Extract: {root: "san", base: "sana", affixed: "malsana", context: "..."}
```

**Validation**:
- Both words in vocabulary
- Sufficient frequency (>10 occurrences)
- For mal-: Cross-check with ReVo antonyms

**Expected output**:
- mal-: 5,000-10,000 pairs
- -ej: 1,000-3,000 pairs
- -ist: 2,000-4,000 pairs
- -ig: 3,000-5,000 pairs
- -et/-eg: 500-1,500 pairs

**Files**:
- `data/training/affix_pairs.json`
- `data/training/affix_triplets.json`
- `data/training/affix_contexts.json`

### Data Preparation - ReVo Validation (Task #61)

**Script**: `scripts/prepare_revo_affix_data.py`

**Use ReVo antonyms to validate and augment**:

1. **Find mal- pairs in ReVo**: Check if antonym pairs have mal- prefix
   - Example: If "nova ↔ malnova" in ReVo antonyms
   - Extract as confirmed mal- transformation

2. **Cross-reference with corpus**:
   - P0: Confirmed by both corpus AND ReVo (highest quality)
   - P1: Corpus only (high frequency)
   - P2: ReVo only (rare but expert-validated)

3. **Gold standard test cases**: Use all 173 antonym pairs for evaluation
   - Test: sim("kaŝi", "montri") should be low (hide ↔ show)

**Files**:
- `data/training/revo_mal_antonyms.json`
- `data/training/corpus_revo_crossref.json`
- `data/training/revo_antonym_tests.json`

### Training Implementation (Task #60)

**Script**: `scripts/train_affix_embeddings.py`

**Key insight**: Different affixes need different loss functions

**1. Negation (mal-)**: Polarity reversal loss
```python
def negation_loss(root_emb, affixed_emb, mal_vector):
    # Transformation accuracy
    predicted = root_emb + mal_vector
    transform_loss = 1 - sim(predicted, affixed_emb)
    
    # Opposites should be dissimilar
    polarity_loss = max(0, sim(root_emb, affixed_emb) - 0.3)
    
    return transform_loss + 0.5 * polarity_loss
```

**2. Place (-ej)**: Cluster targeting loss
```python
def place_suffix_loss(root_emb, affixed_emb, ej_vector, place_cluster):
    # Transformation accuracy
    predicted = root_emb + ej_vector
    transform_loss = 1 - sim(predicted, affixed_emb)
    
    # Cluster with place words
    cluster_loss = 1 - sim(affixed_emb, place_cluster)
    
    # Root should NOT be near places
    separation_loss = max(0, sim(root_emb, place_cluster) - 0.4)
    
    return transform_loss + 0.3 * cluster_loss + 0.2 * separation_loss
```

**3. Size (-et/-eg)**: Opposite transformation loss
```python
def size_modifier_loss(root, diminutive, augmentative, et_vec, eg_vec):
    # Both transformations should work
    et_loss = 1 - sim(root + et_vec, diminutive)
    eg_loss = 1 - sim(root + eg_vec, augmentative)
    
    # Vectors should be opposite
    opposite_loss = max(0, sim(et_vec, eg_vec) + 0.5)
    
    return et_loss + eg_loss + 0.5 * opposite_loss
```

**Architecture**:
```python
class AffixTransformationModel:
    def __init__(self, base_model):
        # Learnable affix vectors
        self.affix_vectors = nn.ParameterDict({
            'mal': nn.Parameter(torch.randn(8)),
            'ej': nn.Parameter(torch.randn(8)),
            'ist': nn.Parameter(torch.randn(8)),
            # ... etc
        })
        
        # Freeze roots during affix training
        for param in base_model.root_embeddings.parameters():
            param.requires_grad = False
```

**Expected improvement**:
- mal- reversal: 0.65 → <0.3
- -ej clustering: 0.45 → >0.7
- Transformation consistency: 0.35 → <0.15

**Output**: `models/affix_embeddings/trained_affixes.pt`

## Implementation Sequence

### Phase 1: Baselines (Measure current state)
1. Run Task #54: Evaluate root embeddings
2. Run Task #58: Evaluate affix transformations

### Phase 2: Data Preparation (Can run in parallel)
3. Run Task #56: Prepare ReVo roots
4. Run Task #59: Prepare corpus affix patterns
5. Run Task #61: Prepare ReVo affix validation

### Phase 3: Training
6. Implement Task #57: Design root contrastive learning
7. Run Task #55: Train roots
8. Run Task #60: Train affixes

### Phase 4: Re-evaluation
9. Re-run Task #54: Measure root improvement
10. Re-run Task #58: Measure affix improvement

## Success Criteria

### Root Embeddings
- ✅ Mean synonym similarity >0.75
- ✅ Synonym vs random gap >0.25
- ✅ Compositional transfer works (derived forms similar)
- ✅ No embedding collapse (mean_sim <0.5)

### Affix Embeddings
- ✅ mal- polarity reversal: sim <0.3
- ✅ -ej place clustering: sim >0.7
- ✅ Transformation consistency: std <0.15
- ✅ Morphological tests still pass

## Validation: Compositional Transfer

**Critical test**: If root training works, ALL derived forms should improve

```python
def test_compositional_transfer(model, root1, root2):
    """
    If roots "kre" and "establ" are trained as synonyms,
    test that ALL derived forms are also similar.
    """
    tests = {
        'verb': (root1 + 'i', root2 + 'i'),
        'noun': (root1 + 'o', root2 + 'o'),
        'adjective': (root1 + 'a', root2 + 'a'),
        'negated': ('mal' + root1 + 'i', 'mal' + root2 + 'i'),
        'place': (root1 + 'ejo', root2 + 'ejo'),
        'agent': (root1 + 'isto', root2 + 'isto'),
    }
    
    for form_type, (word1, word2) in tests.items():
        sim = cosine_similarity(model.embed(word1), model.embed(word2))
        assert sim > 0.7, f"{form_type} failed: {word1} ≈ {word2} = {sim}"
```

**Expected**: ALL forms >0.7 similarity if root training worked.

## Related Tasks

- Tasks #49-53: AST-aware retrieval (prioritized over embeddings)
- Tasks #40-48: Deep embedding investigation (blocked pending AST retrieval results)
- Task #39: Q&A accuracy investigation (showed need for both approaches)

## References

- VISION.md: Klareco's deterministic processing philosophy
- CLAUDE.md: Compositional embedding architecture details
- Note #63: Complete embedding improvement strategy
