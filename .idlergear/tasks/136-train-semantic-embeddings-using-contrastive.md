---
id: 136
title: Train semantic embeddings using contrastive learning on synonym pairs
state: open
created: '2026-01-08T00:07:02.527837Z'
labels:
- enhancement
- training
- high-priority
priority: high
---
## Problem
Current topical embeddings don't capture semantic relationships well:
- esperant ↔ lingv: 0.008 similarity (should be ~0.8)
- fond ↔ kre: 0.26 similarity (should be ~0.9)
- zamenhof ↔ esperant: 0.18 similarity (should be ~0.7)

They were trained with skip-gram which captures distributional similarity, not semantic similarity.

## Solution
Train with contrastive learning using our curated semantic relations.

## Training Scripts (CREATED)

The following scripts have been created and are ready to use:

### 1. Data Preparation
```bash
python scripts/prepare_semantic_training_data.py
```
- Extracts synonym pairs from SemanticRelationDB
- Generates triplets: (anchor, positive=synonym, negative=random)
- Output: `data/training/semantic_triplets.jsonl`

### 2. Training Script
```bash
python scripts/training/train_semantic_embeddings.py
```
- Triplet margin loss training
- Checkpoints every epoch
- Early stopping
- Resume support

### 3. Shell Wrapper (RECOMMENDED)
```bash
./scripts/train_semantic_embeddings.sh           # Normal run (resumes if checkpoint exists)
./scripts/train_semantic_embeddings.sh --fresh   # Start fresh
./scripts/train_semantic_embeddings.sh --include-hypernyms  # Include hypernym pairs
```

Features:
- Prepares data automatically if needed
- Logs to `logs/training/semantic_training_*.log`
- Saves checkpoints to `models/semantic_embeddings/`
- Detects GPU/MPS for acceleration

## Training Data (already available)
- SemanticRelationDB: 2,598 synonym sets
- Curated synonyms: 50+ verified pairs
- ReVo hypernyms: 2,794 relations (optional)
- ReVo hyponyms: 425 relations (optional)

## Training Approach
- Triplet loss: (anchor, positive=synonym, negative=random)
- Loss = max(0, margin - sim(a,p) + sim(a,n))
- Margin = 0.5 (synonyms should be 0.5+ more similar than randoms)
- 20 epochs, early stopping patience=5

## Validation
- Test on held-out synonym pairs
- Measure similarity on known pairs
- Target: synonyms have >0.7 similarity

## TO RUN
In a separate terminal:
```bash
cd /home/marc/Projects/klareco
./scripts/train_semantic_embeddings.sh
```

## Impact
Better semantic embeddings will improve:
- Synonym expansion for root-based retrieval
- Concept-based BM25 scoring (IDF on concepts)
- Topical prefiltering for semantic matching

## Related Tasks
- Task #55: Alternative contrastive learning design
- Task #56: ReVo data preparation (more thorough)
- Task #57: Compositional embedding design
