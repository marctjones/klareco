# Training Quickstart

**Last Updated**: December 2024

This is the definitive guide to training Klareco models. For full details, see `TRAINING_PLAN_V3.md`.

---

## Current Status

| Stage | Status | Script |
|-------|--------|--------|
| Training corpus | **READY** | `scripts/create_training_corpus.py` |
| Stage 1: Root embeddings | **READY TO RUN** | `scripts/training/train_root_embeddings.py` |
| Stage 1: Affix embeddings | Ready | `scripts/training/train_affix_embeddings.py` |
| Stage 2+: Grammatical/Discourse | Blocked by Stage 1 | - |

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
│ STAGE 1: SEMANTIC MODEL (~333K params)                      │
│                                                             │
│  Phase 1: ROOT EMBEDDINGS                                   │
│  ├── Input: Fundamento roots + ReVo definitions + Ekzercaro │
│  ├── Method: Contrastive learning with semantic clusters    │
│  ├── Output: 64d embeddings for ~5K content word roots      │
│  └── Script: train_root_embeddings.py                       │
│                                                             │
│  Phase 2: AFFIX EMBEDDINGS                                  │
│  ├── Input: Root embeddings (frozen) + affix pairs          │
│  ├── Method: Learn transformation vectors (mal-, -et-, etc) │
│  ├── Output: 64d transformation vectors for ~50 affixes     │
│  └── Script: train_affix_embeddings.py                      │
│                                                             │
│  Phase 3: CORPUS INTEGRATION                                │
│  ├── Input: Frozen root+affix embeddings + training corpus  │
│  ├── Method: Fine-tune with corpus co-occurrence            │
│  └── Script: train_sentence_encoder.py                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: GRAMMATICAL MODEL (~52K params) - FUTURE           │
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
└── root_embeddings/
    ├── best_model.pt              # Best checkpoint
    ├── checkpoint.pt              # Latest checkpoint
    └── training.log               # Training progress
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

## Next Steps After Training

1. **Evaluate** root embeddings with semantic tests
2. **Train** affix embeddings (Stage 1 Phase 2)
3. **Build** FAISS index with trained embeddings
4. **Test** retrieval quality improvement

See `TRAINING_PLAN_V3.md` for the complete roadmap.
