# Klareco Scripts Guide

## Pipeline Overview

The Klareco data pipeline has 7 logical stages:

```
ACQUIRE  → Download raw data (Gutenberg, Wikipedia)
CLEAN    → Clean/normalize text (remove headers, markup)
EXTRACT  → Extract sentences + metadata (JSONL)
PARSE    → Parse to ASTs (unified corpus)
INDEX    → Build FAISS indexes
TRAIN    → Train embedding models
VALIDATE → Validate quality
```

### Master Script

```bash
./scripts/pipeline.sh              # Run full pipeline
./scripts/pipeline.sh --from clean # Start from clean stage
./scripts/pipeline.sh --only index # Run only index stage
```

---

## Naming Convention

Scripts follow the pattern: `<stage>_<target>.py` or `<stage>_<target>.sh`

| Stage | Prefix | Example |
|-------|--------|---------|
| Acquire | `acquire_` | `acquire_gutenberg.py` |
| Clean | `clean_` | `clean_gutenberg.py`, `clean_all.sh` |
| Extract | `extract_` | `extract_wikipedia.py`, `extract_all.sh` |
| Parse | `parse_` | `parse_corpus.py`, `parse_corpus.sh` |
| Index | `index_` | `index_compositional.py`, `index_compositional.sh` |
| Train | `train_` | `train_roots.sh`, `train_affixes.sh` |
| Validate | `validate_` | `validate_stage1.py`, `validate_all.sh` |
| Demo | `demo_` | `demo_rag.py`, `demo_pipeline.py` |

Other categories:
- `analyze_*.py` - Analysis/inspection (read-only)
- `benchmark_*.py` - Performance benchmarks
- `util_*.py` - Shared utilities

---

## Stage-by-Stage Reference

### Stage 1: ACQUIRE

Download raw data sources.

```bash
# Download Gutenberg Esperanto texts
python scripts/acquire_gutenberg.py

# Download FastText embeddings (reference)
python scripts/acquire_fasttext.py
```

**Output:** `data/raw/eo/gutenberg/`, `data/raw/eo/wikipedia/`

### Stage 2: CLEAN

Clean and normalize text files.

```bash
# Clean all sources
./scripts/clean_all.sh

# Or individually:
python scripts/clean_gutenberg.py --input data/raw/eo/gutenberg --output data/cleaned/eo
python scripts/clean_revo.py
```

**Output:** `data/cleaned/eo/`

### Stage 3: EXTRACT

Extract sentences with metadata (JSONL format).

```bash
# Extract all sources
./scripts/extract_all.sh

# Or individually:
./scripts/extract_wikipedia.sh   # Wikipedia articles (~2-3 hours)
./scripts/extract_gutenberg.sh   # Books (~5-10 minutes)
```

**Output:** `data/extracted/wikipedia_sentences.jsonl`, `data/extracted/books_sentences.jsonl`

### Stage 4: PARSE

Parse sentences to ASTs and build unified corpus.

```bash
# Build unified corpus with ASTs
./scripts/parse_corpus.sh

# Or with options:
./scripts/parse_corpus.sh --min-parse-rate 0.5
```

**Output:** `data/corpus/unified_corpus.jsonl`

### Stage 5: INDEX

Build FAISS indexes for retrieval.

```bash
# Build compositional embeddings index
./scripts/index_compositional.sh

# Start fresh (ignore checkpoint)
./scripts/index_compositional.sh --fresh
```

**Output:** `data/indexes/compositional/`

### Stage 6: TRAIN

Train embedding models.

```bash
# Train root embeddings
./scripts/train_roots.sh

# Train affix transforms
./scripts/train_affixes.sh

# Full training pipeline
./scripts/train_full.sh
```

**Output:** `models/root_embeddings/best_model.pt`, `models/affix_transforms_v2/best_model.pt`

### Stage 7: VALIDATE

Validate quality of corpus, vocabulary, and models.

```bash
# Run all validation
./scripts/validate_all.sh

# Or individually:
python scripts/validate_vocabulary.py
python scripts/validate_corpus.py
python scripts/validate_stage1.py  # Requires trained model
```

---

## Demo Scripts

Interactive demos to test the pipeline:

```bash
# RAG demo (retrieval + answering)
python scripts/demo_rag.py --interactive
python scripts/demo_rag.py "Kio estas la Unu Ringo?"

# Full pipeline demo
python scripts/demo_pipeline.py

# Embeddings demo
python scripts/demo_embeddings.py
```

---

## Long-Running Scripts

These scripts take significant time and should be run in a separate terminal:

| Script | Duration | Description |
|--------|----------|-------------|
| `extract_wikipedia.sh` | 2-3 hours | Extract Wikipedia |
| `extract_all.sh` | 2-3 hours | All extraction |
| `parse_corpus.sh` | 1-2 hours | Build corpus with ASTs |
| `index_compositional.sh` | 30-60 min | Build FAISS index |
| `train_roots.sh` | 1-2 hours | Train root embeddings |
| `train_affixes.sh` | 30-60 min | Train affix transforms |

### Running in Background

```bash
# Option 1: tmux (recommended)
tmux new -s pipeline
./scripts/pipeline.sh
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t pipeline

# Option 2: nohup
nohup ./scripts/train_roots.sh &

# Option 3: screen
screen -S training
./scripts/train_roots.sh
# Detach: Ctrl+A, then D
```

### Monitoring Progress

```bash
# Follow training logs
tail -f logs/training/root_training_*.log

# Check extraction progress
wc -l data/extracted/wikipedia_sentences.jsonl

# Check corpus size
wc -l data/corpus/unified_corpus.jsonl
```

---

## Checkpointing

All long-running scripts support checkpointing:

```bash
# Resume from checkpoint (default)
./scripts/train_roots.sh

# Start fresh (ignore checkpoint)
./scripts/train_roots.sh --fresh
```

Checkpoint files are saved atomically (write to `.tmp`, then rename) to prevent corruption.

---

## Archive

Obsolete/superseded scripts are in `scripts/archive/`. These are kept for reference but should not be used.

---

## Quick Reference

```bash
# Full pipeline
./scripts/pipeline.sh

# Stage by stage
./scripts/clean_all.sh
./scripts/extract_all.sh
./scripts/parse_corpus.sh
./scripts/index_compositional.sh
./scripts/train_roots.sh
./scripts/train_affixes.sh
./scripts/validate_all.sh

# Demo
python scripts/demo_rag.py -i
```
