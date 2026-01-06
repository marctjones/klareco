# Embedding Improvement Plan

**Created**: 2026-01-05
**Status**: Ready to implement
**Priority**: HIGH (but after AST-aware retrieval - Tasks #49-53)

## Quick Summary

We have 8 tasks to evaluate and improve Klareco's compositional embeddings using the ReVo thesaurus (1,943 synonyms, 173 antonyms) and corpus patterns (4.2M sentences).

**Key insight**: Train roots and affixes SEPARATELY at the correct level to preserve compositionality.

## Task Overview

### ROOT Embeddings (4 tasks)

| # | Task | Type | Effort | Output |
|---|------|------|--------|--------|
| **#54** | Evaluate root quality | Evaluation | 4-6h | Baseline metrics |
| **#56** | Prepare ReVo roots | Data Prep | 6-8h | 500-1000 root pairs |
| **#57** | Design root training | Design | 4-6h | Architecture spec |
| **#55** | Train roots (contrastive) | Training | 8-12h | Improved model |

**Expected improvement**: Synonym similarity 0.55 → >0.75

### AFFIX Embeddings (4 tasks)

| # | Task | Type | Effort | Output |
|---|------|------|--------|--------|
| **#58** | Evaluate affix transformations | Evaluation | 4-6h | Baseline metrics |
| **#59** | Prepare corpus affix patterns | Data Prep | 8-10h | 15,000+ pairs |
| **#61** | Prepare ReVo affix validation | Data Prep | 4-6h | Gold standard tests |
| **#60** | Train affixes (transformation-aware) | Training | 10-14h | Improved model |

**Expected improvement**: mal- reversal 0.65 → <0.3, -ej clustering 0.45 → >0.7

## Implementation Phases

### Phase 1: Baselines (Run first)
```bash
# Measure current state
python scripts/evaluate_embeddings_with_revo.py --output baseline_roots.json
python scripts/evaluate_affix_semantics.py --output baseline_affixes.json
```

**Tasks**: #54, #58
**Time**: 8-12 hours
**Output**: Know what needs fixing

### Phase 2: Data Preparation (Run in parallel)
```bash
# ROOT data
python scripts/prepare_revo_for_training.py \
  --revo data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
  --corpus data/corpus/unified_corpus.jsonl \
  --output data/training/

# AFFIX data (corpus patterns)
python scripts/prepare_affix_training_data.py \
  --corpus data/corpus/unified_corpus.jsonl \
  --output data/training/ \
  --affixes mal,ej,ist,ig,et,eg,ant,int,ul,ad

# AFFIX data (ReVo validation)
python scripts/prepare_revo_affix_data.py \
  --revo data/raw/eo/dictionaries/revo/revo_semantic_relations.json \
  --corpus-pairs data/training/affix_pairs.json \
  --output data/training/
```

**Tasks**: #56, #59, #61
**Time**: 18-24 hours
**Output**: Training-ready datasets

### Phase 3: Training
```bash
# ROOT training
python scripts/train_embeddings_contrastive.py \
  --base-model models/root_embeddings/best_model.pt \
  --root-pairs data/training/revo_root_synonyms.json \
  --output models/root_embeddings/contrastive_model.pt

# AFFIX training
python scripts/train_affix_embeddings.py \
  --base-model models/root_embeddings/contrastive_model.pt \
  --affix-pairs data/training/affix_pairs.json \
  --revo-validation data/training/revo_mal_antonyms.json \
  --output models/affix_embeddings/trained_affixes.pt
```

**Tasks**: #57, #55, #60
**Time**: 22-32 hours
**Output**: Improved embedding models

### Phase 4: Re-evaluation
```bash
# Measure improvements
python scripts/evaluate_embeddings_with_revo.py \
  --model models/root_embeddings/contrastive_model.pt \
  --output improved_roots.json

python scripts/evaluate_affix_semantics.py \
  --model models/affix_embeddings/trained_affixes.pt \
  --output improved_affixes.json
```

**Tasks**: Re-run #54, #58
**Time**: 8-12 hours
**Output**: Improvement report

## Total Effort

- **Minimum**: 56 hours (7 days @ 8hr/day)
- **Maximum**: 80 hours (10 days @ 8hr/day)
- **Realistic**: 68 hours (8.5 days @ 8hr/day)

## Data Sources

### ReVo Thesaurus
- **Location**: `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`
- **Downloaded**: 2026-01-05
- **Contents**:
  - 1,943 synonym pairs
  - 173 antonym pairs
  - 3,351 hypernym pairs
  - 1,098 hyponym pairs
  - 598 part_of pairs

### Unified Corpus
- **Location**: `data/corpus/unified_corpus.jsonl`
- **Size**: 4.2M parsed sentences with ASTs
- **Use**: Extract affix transformation patterns

## Key Architectural Principles

### 1. Train at the Correct Level
```python
# WRONG: Train full words (breaks compositionality)
loss = triplet_loss(embed("krei"), embed("establi"), ...)

# RIGHT: Train roots only (transfers to all forms)
loss = triplet_loss(root_emb["kre"], root_emb["establ"], ...)
```

### 2. Freeze Components During Training
- **Root training**: Freeze affixes (they're deterministic)
- **Affix training**: Freeze roots (only learn transformations)

### 3. Validate Compositional Transfer
If roots improve, ALL derived forms must improve:
- "krei" ≈ "establi" (verbs)
- "kreado" ≈ "establado" (nouns)
- "malkrei" ≈ "malestabli" (negated)

## Success Criteria

### ROOT Embeddings
- ✅ Mean synonym similarity >0.75
- ✅ Synonym vs random gap >0.25
- ✅ Compositional transfer works
- ✅ No embedding collapse (mean <0.5)

### AFFIX Embeddings
- ✅ mal- polarity reversal: sim <0.3
- ✅ -ej place clustering: sim >0.7
- ✅ Transformation consistency: std <0.15
- ✅ All affix tests pass (>80%)

## Strategic Context

**This plan is CONDITIONAL on AST-aware retrieval results** (Tasks #49-53):

- **If AST retrieval achieves >60% accuracy**: Embedding improvements become lower priority
- **If AST retrieval achieves <60% accuracy**: Implement this embedding improvement plan

**Rationale**: Focus on deterministic AST-aware methods first (aligns with Klareco vision). Only invest in learned embeddings if deterministic approaches aren't sufficient.

## Documentation

- **Reference**: `.idlergear/reference/compositional-embedding-training-strategy.md`
- **Strategy Note**: `.idlergear/notes/063.md`
- **Related Tasks**: #54-61 in `.idlergear/tasks/`
- **Vision Alignment**: See `VISION.md` - maximize deterministic processing

## Next Steps

1. **Wait for AST retrieval results** (Tasks #49-53)
2. **If needed**: Run Phase 1 (baselines) to measure current state
3. **Based on baselines**: Decide whether to proceed with full plan
4. **If proceeding**: Run Phases 2-4 sequentially

---

**Last Updated**: 2026-01-05
**Created By**: Claude Code session analyzing Q&A benchmark failures
