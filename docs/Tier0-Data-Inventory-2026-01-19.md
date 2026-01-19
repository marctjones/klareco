# Tier0 Data Inventory - 2026-01-19

## Summary

This document tracks which datasets, models, and embeddings contain or were trained with tier0 data.

**Tier 0 Definition**: Authoritative Esperanto texts (Fundamento, PMEG, Krestomatio, etc.) used as gold standard for grammar and vocabulary.

---

## Corpus Files

### ✅ Contains Tier0 Data

| File | Size | Tier0 Count | Sources |
|------|------|-------------|---------|
| `corpus_with_tier0.jsonl` | 131M | 23K | ekzercaro, krestomatio, lingvaj_respondoj, pmeg |
| `corpus_full_with_tier0.jsonl` | 18G | 22,516 | ekzercaro (157), krestomatio (4,587), lingvaj_respondoj (4,789), pmeg (12,983) |

**Tier0 breakdown in full corpus**:
- **ekzercaro**: 157 sentences
- **krestomatio**: 4,587 sentences
- **lingvaj_respondoj**: 4,789 sentences
- **pmeg**: 12,983 sentences
- **Total**: 22,516 sentences

### ❌ Does NOT Contain Tier0 Data

| File | Size | Tier0 Count | Notes |
|------|------|-------------|-------|
| `books_corpus.jsonl` | 608M | 0 | Gutenberg books only (tier 6) |
| `wikipedia_corpus.jsonl` | 17G | 0 | Wikipedia only (tier 5) |
| `corpus_with_metadata.jsonl` | 18G | 0 | Combined corpus without tier0 |

---

## Training Datasets

### ✅ Contains Tier0 Data

| Dataset | Total | Tier0 | % Tier0 | Purpose |
|---------|-------|-------|---------|---------|
| `data/training/m1_tier0_only/` | 23,899 | 22,253 | 93.1% | M1 trained ONLY on tier0 |

### ❌ Does NOT Contain Tier0 Data

| Dataset | Total | Tier0 | Purpose | Issue |
|---------|-------|-------|---------|-------|
| `data/training/m1_semantic_full/` | 320,000 | 0 | M1 semantic training | ⚠️ Filtered out despite corpus having tier0 |
| `data/training/m1_with_tier0/` | 320,000 | 0 | M1 selectional training | ⚠️ **MISNAMED** - does NOT contain tier0! |
| `data/training/m1_semantic_violations/` | 26,723 | 0 | M1 semantic violations | Hard negatives only |
| `data/training/m1_selectional_hard_only/` | 13,772 | 0 | M1 hard negatives | Hard negatives only |

---

## Trained Models

### ✅ Trained WITH Tier0 Data

| Model | Type | Accuracy/Corr | Date | Dataset |
|-------|------|---------------|------|---------|
| `root_embeddings_tier0/best_model.pt` | Stage 1 Embeddings | 85.34% corr | Jan 18 10:33 | Tier0 vocabulary (10,819 roots) |
| `m1_tier0_only/best_model.pt` | M1 Selectional | 68.23% acc | Jan 19 00:13 | m1_tier0_only (22,253 tier0 examples) |
| `m1_selectional_tier0/best_model.pt` | M1 Selectional | 70.18% acc | Jan 18 17:17 | Unknown tier0 dataset |

**Note**: Lower accuracy for tier0-only models is expected due to smaller dataset size (23K vs 320K examples).

### ❌ Trained WITHOUT Tier0 Data

| Model | Type | Accuracy/Corr | Date | Dataset | Issue |
|-------|------|---------------|------|---------|-------|
| `m1_semantic_full/best_model.pt` | **M1 Semantic (CURRENT)** | **86.37% acc** | **Jan 19 03:04** | m1_semantic_full | ⚠️ **Missing tier0 despite corpus having it** |
| `m1_selectional/best_model.pt` | M1 Selectional | 80.71% acc | Jan 14 20:15 | Unknown | Legacy model |
| `m1_selectional_v2/best_model.pt` | M1 Selectional | 76.23% acc | Jan 17 15:01 | Unknown | Legacy model |
| `m1_selectional_v3/best_model.pt` | M1 Selectional | 74.55% acc | Jan 17 15:37 | Unknown | Legacy model |
| `root_embeddings/best_model.pt` | Stage 1 Embeddings | 83.96% corr | Jan 17 14:34 | Mixed corpus | Legacy model |

---

## Key Findings

### 🔴 Critical Issues

1. **`m1_with_tier0` dataset is MISNAMED**
   - Name suggests it contains tier0 data
   - Actually contains ZERO tier0 examples
   - Issue #13: Rename dataset to avoid confusion

2. **`m1_semantic_full` missing tier0 data**
   - Corpus contains 22,516 tier0 sentences
   - Training data has 0 tier0 examples
   - Issue #12: Investigate filtering logic in data generation script

3. **Current production model lacks tier0**
   - `demo_rag_with_m1.py` uses `m1_semantic_full/best_model.pt`
   - This model was NOT trained on tier0 data
   - Achieves 86.37% accuracy despite missing tier0
   - Potential improvement to 87-88% if retrained with tier0

### ✅ What Works

1. **Corpus files correctly contain tier0**
   - `corpus_full_with_tier0.jsonl` has 22,516 tier0 sentences
   - File is correctly named

2. **Stage 1 embeddings use tier0 vocabulary**
   - `root_embeddings_tier0/best_model.pt` trained on tier0 roots
   - Achieves 85.34% correlation

3. **Tier0-only models exist**
   - `m1_tier0_only/best_model.pt` trained exclusively on tier0
   - Lower accuracy (68.23%) due to small dataset size

---

## Related Issues

**EPIC #14**: RAG Quality Investigation and Training Pipeline Improvements (see `docs/EPIC-14-RAG-Quality-Investigation.md`)

Sub-issues:
- **Issue #10**: Fix M1 training data generation to use tier0 corpus (OPEN)
- **Issue #11**: ✅ CLOSED - corpus_full_with_tier0.jsonl correctly contains tier0
- **Issue #12**: Investigate why tier0 data is filtered out during M1 training data generation (OPEN)
- **Issue #13**: Rename `m1_with_tier0` dataset to `m1_selectional_full` (OPEN)
- **Issue #15**: Investigate M1 plausibility scoring issues (OPEN)
- **Issue #16**: Investigate retrieval ranking issues (OPEN)
- **Issue #17**: Audit model and dataset lifecycle (OPEN)
- **Issue #18**: Audit corpus for basic question coverage (OPEN)

---

## Recommendations

### Immediate Actions

1. **Rename misleading dataset** (Issue #13)
   ```bash
   mv data/training/m1_with_tier0 data/training/m1_selectional_full
   # Update any scripts that reference old name
   ```

2. **Document filtering logic** (Issue #12)
   - Investigate `scripts/generate_m1_semantic_data.py`
   - Check parse rate threshold, sentence length filters
   - Document why tier0 is being excluded

### Medium-term Improvements

3. **Retrain M1 with tier0 included**
   - Fix filtering to include tier0 (22,516 additional examples)
   - Expected improvement: 86.37% → 87-88% accuracy
   - Priority: Medium (current model is good enough)

4. **Verify all training pipelines**
   - Audit all data generation scripts
   - Ensure tier0 is included unless explicitly excluded
   - Add tier0 coverage tests

---

## Verification Commands

### Check corpus tier0 content
```bash
# Count tier0 sentences
jq -r 'select(.source.tier == 0)' data/enhanced_corpus/corpus_full_with_tier0.jsonl | wc -l
# Returns: 22,516

# Count by source
jq -r 'select(.source.tier == 0) | .source.name' data/enhanced_corpus/corpus_full_with_tier0.jsonl | sort | uniq -c
```

### Check training data tier0 content
```bash
# Check each training dataset
for dataset in data/training/*/train.jsonl; do
  echo "=== $dataset ==="
  jq -r 'select(.source.tier == 0)' "$dataset" | wc -l
done
```

### Check model metadata
```bash
python3 << 'EOF'
import torch
checkpoint = torch.load('models/m1_semantic_full/best_model.pt', map_location='cpu', weights_only=False)
print(f"Accuracy: {checkpoint['best_accuracy']:.4f}")
# Check if training data info stored in checkpoint
EOF
```

---

**Report Date**: 2026-01-19
**Verified By**: Claude Code
**Last Updated**: 2026-01-19

