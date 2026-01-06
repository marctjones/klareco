---
id: 54
title: Evaluate embedding quality using ReVo synonym pairs
state: open
created: '2026-01-05T16:28:12.533620Z'
labels:
- enhancement
- evaluation
- embeddings
priority: high
---
## Goal
Use the 1,943 synonym pairs from ReVo (Reta Vortaro) to evaluate how well the current compositional embeddings capture semantic similarity.

## Available Data
- File: `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`
- Contains 1,943 synonym pairs: "fik ≈ koit", "humor ≈ humur", etc.
- Also has antonyms (173), hypernyms (3,351), hyponyms (1,098), part_of (598), has_part (2,141)

## Test Design

### Test 1: Synonym Similarity Scores
For each synonym pair (A, B):
- Compute cosine similarity between embeddings: `sim(emb(A), emb(B))`
- Expected: High similarity (>0.7) for good embeddings

**Metrics**:
- Mean synonym similarity (should be >0.7)
- Std deviation (should be low, indicating consistency)
- Distribution histogram

### Test 2: Synonym vs Random Baseline
For each synonym pair (A, B):
- Compute `sim(A, B)` (true synonym)
- Sample 100 random words C from vocabulary
- Compute `sim(A, C)` for each random word
- Measure gap: `mean(sim(A, B)) - mean(sim(A, random))`

**Expected**: Significant gap (>0.2) between synonyms and random words

### Test 3: Antonym Discrimination
For each antonym pair (A, B):
- Compute `sim(emb(A), emb(B))`
- Expected: Lower than synonyms but not negative (cosine similarity range [0, 1])

**Comparison**:
- Mean synonym similarity vs mean antonym similarity
- Should see clear separation

### Test 4: Hypernym Hierarchy (Optional)
For hypernym pairs (A, B) where B is more general:
- Check if embeddings show hierarchical structure
- Example: "hundo" (hyponym) vs "besto" (hypernym)

## Implementation

Create script: `scripts/evaluate_embeddings_with_revo.py`

```python
def load_semantic_relations():
    # Load ReVo relations
    
def load_embeddings():
    # Load compositional embedding model
    
def evaluate_synonyms():
    # Test 1 & 2
    
def evaluate_antonyms():
    # Test 3
    
def evaluate_hypernyms():
    # Test 4
    
def main():
    # Run all tests, generate report
```

## Output

Generate report: `benchmark_results/embeddings/revo_evaluation_TIMESTAMP.json`

```json
{
  "synonym_test": {
    "mean_similarity": 0.XX,
    "std_similarity": 0.XX,
    "pairs_tested": 1943,
    "high_quality_pairs": XXX,  // sim > 0.7
    "medium_quality_pairs": XXX,  // 0.5 < sim < 0.7
    "low_quality_pairs": XXX  // sim < 0.5
  },
  "baseline_test": {
    "mean_synonym_sim": 0.XX,
    "mean_random_sim": 0.XX,
    "gap": 0.XX,
    "effect_size": "small/medium/large"
  },
  "antonym_test": {
    "mean_similarity": 0.XX,
    "synonym_antonym_gap": 0.XX
  },
  "failures": [
    {"pair": ["word1", "word2"], "similarity": 0.XX, "reason": "..."}
  ]
}
```

## Success Criteria
- Mean synonym similarity >0.7
- Synonym vs random gap >0.2
- Synonym vs antonym gap >0.1
- <10% of synonym pairs with sim <0.5

## Effort
4-6 hours (script + analysis)

## Dependencies
- Current compositional embedding model (`models/root_embeddings/`)
- ReVo semantic relations (already downloaded)
