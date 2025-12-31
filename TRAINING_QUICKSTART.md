# Training Quickstart

**Last Updated**: December 2025

This is the definitive guide to training Klareco models. For full details, see `TRAINING_PLAN_V3.md`.

---

## Current Status

| Stage | Status | Script |
|-------|--------|--------|
| Training corpus | **COMPLETE** | `scripts/create_training_corpus.py` |
| Stage 1: Root embeddings | **COMPLETE** | `scripts/training/train_root_embeddings.py` |
| Stage 1: Affix transforms V2 | **COMPLETE** | `scripts/training/train_affix_transforms_v2.py` |
| Stage 1: Corpus index | **COMPLETE** | `scripts/index_corpus_compositional.py` |
| Stage 2+: Grammatical/Discourse | Next | - |

### Stage 1 Results

- **Root embeddings**: 11,121 roots × 64d = 712K params
  - Correlation: 0.8871 | Accuracy: 97.98%
  - Model: `models/root_embeddings/best_model.pt`

- **Affix transforms V2**: 12 prefixes + 29 suffixes (~21K params)
  - Anti-collapse: mal_mean_sim = -0.03 (target < 0.5)
  - Model: `models/affix_transforms_v2/best_model.pt`

- **Corpus index**: 4.38M sentences
  - Index: `data/corpus_index_compositional/`

---

## Quick Start

### 1. Verify Prerequisites

```bash
# Check training data exists
ls -la data/training/*.jsonl
# Should show: authoritative_training.jsonl, literature_training.jsonl, general_training.jsonl

# Check ReVo definitions exist
ls -la data/revo/revo_definitions_with_roots.json
# Should show: 11MB file with 10,766 dictionary entries

# Check Fundamento roots exist
ls -la data/vocabularies/fundamento_roots.json
```

### 2. Run Training

```bash
# Run the full training pipeline (in a separate terminal)
./scripts/run_fundamento_training.sh

# Or run with fresh start (ignores checkpoints)
./scripts/run_fundamento_training.sh --fresh

# Monitor progress
tail -f logs/training/fundamento_training_*.log
```

### 3. Evaluate

```bash
# After training completes
python scripts/training/evaluate_embeddings.py \
    --model models/root_embeddings/best_model.pt
```

---

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: SEMANTIC MODEL (~733K params) ✓ COMPLETE           │
│                                                             │
│  Phase 1: ROOT EMBEDDINGS ✓                                 │
│  ├── 11,121 roots × 64d = 712K params                       │
│  ├── Correlation: 0.8871 | Accuracy: 97.98%                 │
│  └── Model: models/root_embeddings/best_model.pt            │
│                                                             │
│  Phase 2: AFFIX TRANSFORMS V2 ✓                             │
│  ├── 12 prefixes + 29 suffixes (~21K params)                │
│  ├── Low-rank transformations (rank=8)                      │
│  ├── Anti-collapse: mal_mean_sim = -0.03                    │
│  └── Model: models/affix_transforms_v2/best_model.pt        │
│                                                             │
│  Phase 3: CORPUS INDEX ✓                                    │
│  ├── 4.38M sentences indexed                                │
│  └── Index: data/corpus_index_compositional/                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: GRAMMATICAL MODEL (~52K params) ← NEXT             │
│  Learn semantic effects of tense, mood, negation, etc.      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: DISCOURSE MODEL (~100K params) - FUTURE            │
│  Coreference, discourse relations, multi-sentence context   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Tier | Source | Weight | Entries | Location |
|------|--------|--------|---------|----------|
| 1 | Fundamento Ekzercaro | 100x | 904 | `data/training/authoritative_training.jsonl` |
| 2 | Fundamenta Krestomatio | 50x | 14,123 | `data/training/authoritative_training.jsonl` |
| 3 | Gerda Malaperis | 20x | 1,530 | `data/training/authoritative_training.jsonl` |
| 4 | Reta Vortaro (ReVo) | 5x | 10,766 | `data/revo/revo_definitions_with_roots.json` |
| 5 | Gutenberg Literature | 2x | 19,796 | `data/training/literature_training.jsonl` |
| 6 | Wikipedia | 1x | 1,947,337 | `data/training/general_training.jsonl` |

**Note**: Tier weights ensure authoritative sources dominate the training signal despite having fewer examples.

---

## Key Principles

### 1. Function Word Exclusion

Function words (kaj, de, en, la, mi, etc.) are **excluded** from embedding training. They are handled by the deterministic AST layer. Including them causes embedding collapse.

### 2. Content Words Only

Only content words get learned embeddings:
- Nouns: hundo, tablo, domo
- Verbs: kuri, manĝi, ami
- Adjectives: bela, granda, rapida
- Adverbs (content): rapide, bone

### 3. Staged Training

Each stage is **frozen** before the next begins:
1. Train root embeddings → freeze
2. Train affix embeddings → freeze
3. Integrate with corpus → freeze
4. Train grammatical transforms → freeze
5. etc.

### 4. Fundamento-Centered

Training is weighted heavily toward Zamenhof's original works. The Fundamento Ekzercaro has 100x weight compared to Wikipedia.

---

## Troubleshooting

### Embedding Collapse

**Symptom**: All embeddings become similar (>0.9 cosine similarity)

**Causes**:
- Function words included in training
- Missing negative sampling
- Learning rate too high

**Fix**: Check that `FUNCTION_WORDS` filter is applied in training script.

### Low Parse Rate

**Symptom**: Many sentences filtered out during corpus creation

**Check**: `data/training/training_corpus_metadata.json` shows filter stats

**Fix**: May need to improve parser before retraining

### Checkpoint Issues

**Symptom**: Training restarts from scratch

**Check**: Look for `checkpoint.pt` in output directory

**Fix**: Use `--fresh` flag intentionally, or check disk space

---

## Files Reference

### Training Scripts
```
scripts/
├── run_fundamento_training.sh      # Main training runner
├── run_full_training.sh            # Full pipeline runner
├── create_training_corpus.py       # Create training data from corpus
└── training/
    ├── train_root_embeddings.py    # Stage 1 Phase 1
    ├── train_affix_embeddings.py   # Stage 1 Phase 2
    ├── train_sentence_encoder.py   # Stage 1 Phase 3
    ├── evaluate_embeddings.py      # Evaluation
    ├── extract_fundamento_uv.py    # Extract Fundamento roots
    └── extract_ekzercaro.py        # Extract Ekzercaro sentences
```

### Data Files
```
data/
├── training/                       # Training-ready filtered data
│   ├── authoritative_training.jsonl   # Tier 1-3 (16,557 entries)
│   ├── literature_training.jsonl      # Tier 5 (19,796 entries)
│   ├── general_training.jsonl         # Tier 6 (1,947,337 entries)
│   └── training_corpus_metadata.json  # Stats and weights
├── revo/
│   └── revo_definitions_with_roots.json  # ReVo dictionary (10,766 entries)
├── vocabularies/
│   ├── fundamento_roots.json       # Fundamento UV roots
│   └── affix_vocabulary.json       # Esperanto affixes
└── corpus/
    └── unified_corpus.jsonl        # Full parsed corpus (4.2M entries)
```

### Model Outputs
```
models/
├── root_embeddings/
│   ├── best_model.pt              # 11,121 roots × 64d
│   └── training.log               # Training progress
└── affix_transforms_v2/
    ├── best_model.pt              # 41 affixes, low-rank transforms
    └── training.log               # Training progress

data/
└── corpus_index_compositional/
    ├── embeddings.npy             # 4.38M sentence embeddings
    ├── sentences.jsonl            # Sentence metadata
    └── faiss.index                # FAISS index for retrieval
```

---

## What's NOT Part of Current Training

These approaches were tried and **removed**:

| Old Approach | Why Removed |
|--------------|-------------|
| Tatoeba sentence similarity | Caused embedding collapse, wrong abstraction level |
| Plena Vortaro (PV) | OCR artifacts, replaced by ReVo |
| Cross-lingual training | Violates Esperanto-first principle |
| Whole-word embeddings | Can't generalize to unseen word forms |

---

## Next Steps (Stage 2)

Stage 1 is complete. Next steps:

1. **Design** Stage 2 grammatical transforms (negation, tense, mood)
2. **Create** minimal pairs training data for grammatical features
3. **Train** grammatical transform models
4. **Evaluate** impact on retrieval quality

See `TRAINING_PLAN_V3.md` for the complete roadmap.

## Testing Stage 1 Models

```bash
# Test affix transforms
python scripts/test_affix_v2.py

# Interactive root embedding demo
python scripts/demo_root_embeddings.py -i

# Test RAG with compositional index
python scripts/demo_rag_compositional.py -i
```
