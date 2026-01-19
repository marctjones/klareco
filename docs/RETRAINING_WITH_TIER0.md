# Retraining Models with Tier0 Corpus and Semantic Relations

## Overview

This document describes how to retrain Klareco models to leverage:
1. **Tier0 corpus**: 23,873 authoritative Esperanto sentences (weight=10.0)
2. **ReVo semantic relations**: 1,982 curated relations (synonyms, antonyms, hypernyms, etc.)
3. **ConceptNet relations**: 16,769 multilingual knowledge graph edges

The retraining pipeline updates both Stage 1 (root embeddings) and M1 (selectional preferences) models to benefit from higher-quality data and explicit semantic structure.

## What Changed

### Stage 1 Root Embeddings Training

**Modified script**: `scripts/train_root_embeddings.py`

**New capabilities**:
- **Tier0 co-occurrence pairs**: Extracts root co-occurrence from tier0 corpus ASTs (weight=15.0, target similarity 0.4-0.95)
- **Corrected ReVo path**: Now points to `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`
- **Semantic relation integration**: Uses ReVo synonyms, antonyms, hypernyms for high-quality similarity targets

**New parameters**:
```bash
--tier0-corpus PATH          # Path to tier0 corpus (default: data/enhanced_corpus/corpus_with_tier0.jsonl)
--revo-relations PATH        # Path to ReVo semantic relations (fixed path)
```

**Training data composition**:
| Source | Pairs | Weight | Similarity Range | Purpose |
|--------|-------|--------|------------------|---------|
| Tier0 co-occurrence | ~5-10K | 15.0 | 0.4-0.95 | High-quality real usage |
| ReVo curated relations | ~2K | 2.0-8.0 | 0.1-0.75 | Explicit semantic structure |
| Ekzercaro co-occurrence | ~20K | 10.0 | 0.3-0.9 | Foundational examples |
| ReVo Jaccard definitions | ~50K | 2.0-5.0 | 0.4-0.8 | Definition similarity |
| Fundamento translations | ~5K | 5.0 | 0.5-0.95 | Translation equivalence |
| Semantic clusters | ~10K | 5.0-6.0 | 0.0-0.45 | Category structure |

### M1 Selectional Preferences Training

**Modified approach**: Regenerate training data from tier0-enhanced corpus

**Data generation**:
```bash
python scripts/prepare_m1_training_data.py \
  --corpus data/enhanced_corpus/corpus_full_with_tier0.jsonl \
  --output data/training/m1_with_tier0 \
  --max-triples 200000
```

**Benefits**:
- Tier0 examples provide authoritative verb-object selectional preferences
- Higher parse quality (99.99% success rate vs ~92% for general corpus)
- Literary and grammar sources give natural, correct usage patterns

## Quick Start

### Full Retraining (Both Models)

```bash
./scripts/retrain_with_tier0.sh
```

This will:
1. Regenerate M1 training data from tier0-enhanced corpus (10-15 minutes)
2. Retrain Stage 1 root embeddings with tier0 + semantic relations (~2-4 hours)
3. Retrain M1 selectional model with new embeddings (~30-60 minutes)

### Partial Retraining

```bash
# Only retrain Stage 1 (root embeddings)
./scripts/retrain_with_tier0.sh --stage1-only

# Only retrain M1 (skip data generation + Stage 1)
./scripts/retrain_with_tier0.sh --m1-only --skip-data

# Retrain M1 with existing data
./scripts/retrain_with_tier0.sh --skip-data --stage1-only
```

## Manual Steps

### Step 1: Regenerate M1 Training Data

```bash
python scripts/prepare_m1_training_data.py \
  --corpus data/enhanced_corpus/corpus_full_with_tier0.jsonl \
  --output data/training/m1_with_tier0 \
  --max-triples 200000 \
  --negatives-per-positive 1
```

**Output**:
- `data/training/m1_with_tier0/train.jsonl` (~140K examples)
- `data/training/m1_with_tier0/val.jsonl` (~20K examples)
- `data/training/m1_with_tier0/test.jsonl` (~20K examples)

### Step 2: Retrain Stage 1 Root Embeddings

```bash
python scripts/train_root_embeddings.py \
  --tier0-corpus data/enhanced_corpus/corpus_with_tier0.jsonl \
  --revo-relations data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
  --output-dir models/root_embeddings_tier0 \
  --epochs 100 \
  --patience 15 \
  --fresh
```

**Output**:
- `models/root_embeddings_tier0/best_model.pt` (best validation model)
- `models/root_embeddings_tier0/checkpoint.pt` (latest checkpoint)
- `logs/training/train_root_embeddings_*.log` (training log)

**Expected improvements**:
- Better semantic clustering (tier0 co-occurrence provides clean signal)
- Stronger synonym/antonym separation (explicit ReVo relations)
- Reduced embedding collapse (semantic cluster constraints + tier0 weight)

### Step 3: Retrain M1 Selectional Model

```bash
python scripts/train_m1_selectional.py \
  --stage1-model models/root_embeddings_tier0/best_model.pt \
  --data-dir data/training/m1_with_tier0 \
  --output-dir models/m1_selectional_tier0 \
  --epochs 50 \
  --patience 10 \
  --fresh
```

**Output**:
- `models/m1_selectional_tier0/best_model.pt`
- `logs/training/m1_training_*.log`

**Expected improvements**:
- Better handling of literary constructions (tier0 includes Alice in Wonderland, etc.)
- More accurate verb-object compatibility from authoritative grammar examples
- Improved performance on PMEG/Lingvaj Respondoj test cases

## Evaluation

### Stage 1 Quality Tests

```bash
# Run comprehensive embedding quality tests
pytest tests/test_stage1_model_quality.py -v

# Key metrics:
# - Root similarity accuracy: >85%
# - No embedding collapse: mean_sim < 0.5
# - Cluster separation: gap > 0.03
# - Fundamento coverage: 100%
```

### M1 Quality Tests

```bash
# Validate selectional preferences
python scripts/validate_m1_extensive.py \
  --model models/m1_selectional_tier0/best_model.pt

# Expected accuracy: >85% on validation set
```

### Manual Semantic Query Tests

```python
import kuzu

db = kuzu.Database('data/indexes/kuzu_index/kuzu.db')
conn = kuzu.Connection(db)

# Test semantic relations
result = conn.execute("""
    MATCH (r:Root {root: 'bona'})-[:REVO_SYNONYM]->(s:Root)
    RETURN s.root
""")
# Should return semantically related roots

# Test tier0 corpus retrieval
result = conn.execute("""
    MATCH (s:Sentence)-[:HAS_ROOT]->(r:Root {root: 'hundo'})
    WHERE s.tier = 0
    RETURN s.text LIMIT 5
""")
# Should return high-quality sentences about dogs
```

## Model Comparison

After retraining, compare old vs new models:

| Metric | Stage 1 (old) | Stage 1 (tier0) | Expected Gain |
|--------|---------------|-----------------|---------------|
| Synonym accuracy | ~80% | ~88% | +8% |
| Embedding collapse (mean_sim) | 0.45 | 0.35 | -0.10 (better) |
| Cluster gap | 0.02 | 0.05 | +0.03 |
| ReVo coverage | 0% (no data) | 62% | +62% |

| Metric | M1 (old) | M1 (tier0) | Expected Gain |
|--------|----------|------------|---------------|
| Validation accuracy | ~82% | ~87% | +5% |
| Literary examples (tier0) | N/A | High quality | New capability |
| Grammar violations | ~20% false negatives | ~10% | -10% (better) |

## Troubleshooting

### Issue: Training runs out of memory

**Solution**: Reduce batch size or max triples
```bash
python scripts/train_root_embeddings.py --batch-size 64  # Default: 128
python scripts/prepare_m1_training_data.py --max-triples 100000  # Default: 200000
```

### Issue: Embeddings still collapsing

**Symptoms**: High average similarity (>0.7), low cluster gap (<0.02)

**Solutions**:
1. Increase semantic cluster negative weight (edit script line ~545: change `weights.append(5.0)` to `8.0`)
2. Add more hard negatives: `--hard-negatives` flag
3. Reduce learning rate: `--learning-rate 0.0005`

### Issue: M1 model predicting all zeros or ones

**Symptoms**: Low score variance (<0.05), stuck predictions

**Solutions**:
1. Check data balance (should be ~50/50 positive/negative)
2. Reduce hidden layer size: `--hidden-dim 64` (default: 128)
3. Increase dropout: `--dropout 0.2` (default: 0.1)
4. Add more training data: `--max-triples 300000`

## Files Modified

**Scripts**:
- `scripts/train_root_embeddings.py` - Added tier0 corpus support
- `scripts/retrain_with_tier0.sh` - New wrapper for full retraining pipeline

**Data**:
- `data/enhanced_corpus/corpus_with_tier0.jsonl` - Tier0 corpus (23K sentences)
- `data/raw/eo/dictionaries/revo/revo_semantic_relations.json` - ReVo relations (1,982)
- Kuzu database now includes 18,509 semantic edges

**Models** (new outputs):
- `models/root_embeddings_tier0/` - Retrained Stage 1 embeddings
- `models/m1_selectional_tier0/` - Retrained M1 selectional model

## Next Steps After Retraining

1. **Evaluate Quality**: Run tests and compare metrics vs baseline
2. **Test Semantic Retrieval**: Query Kuzu with new embeddings
3. **Update Production**: If quality improves, replace production models
4. **Retrain Stage 2**: Use new embeddings for grammatical model training
5. **RAG Integration**: Update retriever to use semantic relations

## References

- **Tier0 Integration**: See `docs/SEMANTIC_KNOWLEDGE_GRAPH.md`
- **ReVo Pipeline**: See `scripts/process_revo.sh`
- **ConceptNet Loading**: See `scripts/load_conceptnet_to_kuzu.py`
- **Model Architecture**: See `CLAUDE.md` section on compositional embeddings
