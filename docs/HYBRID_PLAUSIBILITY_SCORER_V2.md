# Hybrid Plausibility Scorer v2.0 - Training Report

## Executive Summary

Successfully trained a hybrid plausibility scorer achieving **56.77% F1 score**, a +41% improvement over the initial 40.15% baseline. The model uses **172D word representations** (128D learned root embeddings + 44D deterministic features) to judge the semantic plausibility of Esperanto SVO triples.

## Architecture

**Model**: HybridPlausibilityScorer
- **Input**: 172D per word (subject, verb, object) = 516D concatenated
- **MLP**: 516D → 256D → 128D → 1 (sigmoid output)
- **Parameters**: 165,377 trainable (MLP only, embeddings frozen)

**Word Encoder**: HybridWordEncoder (frozen)
- Root embedding: 128D (learned from unified root embedder)
- Affix features: 22D (deterministic from affix semantics)
- Lexicon features: 22D (deterministic from ROOT_LEXICON)
- Total: 172D output per word

## Training Data Evolution

### Iteration 1: Baseline (31K examples)
- **Dataset**: 31,358 examples (11,358 pos, 20,000 neg)
- **Positive ratio**: 36.2%
- **Result**: F1 = 40.15%
- **Issues**: 84% unknown animacy (only 95 roots in lexicon)

### Iteration 2: Lexicon Expansion (73K examples)
- **Dataset**: 73,324 examples (23,324 pos, 50,000 neg)
- **Positive ratio**: 31.8%
- **Result**: F1 = 35.82% (WORSE!)
- **Root cause**: Verb dominance - "est" (10.7%) and "hav" (9.0%) dominating training signal

### Iteration 3: Verb Filtering (68K examples)
- **Dataset**: 68,657 examples (18,657 pos, 50,000 neg)
- **Positive ratio**: 27.2% (too low!)
- **Result**: F1 = 30.49% (WORST!)
- **Root cause**: Class imbalance (too few positives)

### Iteration 4: Optimal Balance (43K examples) ✓
- **Dataset**: 43,157 examples (18,657 pos, 24,500 neg)
- **Positive ratio**: 43.2% ✓
- **Result**: F1 = 56.77% (BEST!)
- **Success factors**:
  - Balanced class distribution (~40-45% positive)
  - Filtered verbs >5% frequency (removes "est", "hav")
  - 100% semantic coverage (all roots in 500-root lexicon)
  - 341 unique verbs, 35 subject types, 35 object types

## Key Learnings

1. **Class balance matters more than dataset size**
   - 43K with 43% positive ratio > 73K with 32% positive ratio

2. **Verb frequency filtering is critical**
   - Filter verbs appearing >5% prevents copula/auxiliary dominance
   - Top verbs in final dataset: pov (3.4%), fond (2.1%), ricev (2.1%)

3. **Semantic coverage is essential**
   - Filtering to 500-root lexicon ensures 100% valid semantic features
   - No "unknown" animacy values that confuse the model

4. **Optimal positive ratio: 40-45%**
   - Too low (<30%): model learns to predict "implausible" too often
   - Too high (>50%): insufficient negative examples for discrimination

## Dataset Specifications

**Final Training Data**: `data/plausibility_training_word_level_final/`

```
Total examples: 43,157
├─ Training: 38,841 (90%)
└─ Validation: 4,316 (10%)

Positive examples: 18,657 (43.2%)
├─ Source: wikipedia_sentences (corpus triples)
└─ Coverage: 100% (all roots in 500-root lexicon)

Negative examples: 24,500 (56.8%)
├─ Affix-aware type swaps: 9,800 (40%)
├─ Type mismatches: 9,855 (40%)
└─ Animacy violations: 4,845 (20%)

Quality filters:
├─ Verb frequency: <5% of corpus
├─ Confidence threshold: ≥0.0 (all triples)
└─ No proper nouns (filtered by lexicon)
```

## Model Performance

**Best Model**: `models/hybrid_plausibility_word_level_final/best_model.pt`

```
Epoch 8/10:
├─ Val F1: 56.77%
├─ Precision: 57.40%
└─ Recall: 56.16%

Training time: ~3 minutes (CPU)
Model size: 643 KB
```

## Usage

```python
# Load model
import torch
from train_plausibility_scorer_word_level import (
    HybridPlausibilityScorer, HybridWordEncoder
)

checkpoint = torch.load("models/hybrid_plausibility_word_level_final/best_model.pt")
# ... (see training script for full initialization)

# Score a triple
score = model(subject_words, verb_words, object_words)
plausibility = torch.sigmoid(score)  # 0-1 probability
```

## Next Steps

1. **Integrate into AST pipeline** (Task #4)
   - Add plausibility filtering to SVO extraction
   - Use scores to rank or filter retrieved triples

2. **Evaluate on downstream tasks**
   - Question answering accuracy
   - Retrieval precision/recall impact

3. **Potential improvements**
   - Train on larger lexicon (>500 roots)
   - Add verb selectional preferences
   - Multi-task learning (plausibility + entailment)

## Files

### Training Data
- `data/semantic_types/svo_triples_word_level_filtered.jsonl` - Filtered corpus (23K triples)
- `data/plausibility_training_word_level_final/` - Final balanced dataset (43K examples)
- `scripts/generate_plausibility_training_data_word_level.py` - Data generation script

### Models
- `models/hybrid_plausibility_word_level_final/best_model.pt` - Trained model (F1=56.77%)
- `models/hybrid_plausibility_word_level_final/training.log` - Full training log

### Code
- `scripts/train_plausibility_scorer_word_level.py` - Training script with model definition
- `scripts/filter_svo_triples_by_lexicon.py` - Lexicon filtering script

### Lexicon
- `klareco/morphology/root_lexicon_v3_full.py` - 500-root lexicon
- `klareco/morphology/affix_semantics.py` - Deterministic affix semantics

## References

- See `CLAUDE.md` for architectural principles
- See `docs/MODEL_INVENTORY.md` for model versioning
- See `.claude/rules/idlergear.md` for development workflow

---

**Last Updated**: 2026-03-23  
**Author**: Claude Sonnet 4.5 (with Marc)  
**Version**: 2.0 (Hybrid Word-Level)
