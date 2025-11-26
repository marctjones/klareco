# Archived Files - RAG Improvement Project

**Date**: 2025-11-13
**Project**: RAG System Improvement (Corpus Segmentation + GNN Retraining)

---

## Archived Models

### `models/tree_lstm_old/`
- **Original**: `models/tree_lstm/`
- **Date Archived**: 2025-11-13
- **Reason**: Replaced with retrained GNN (10x more training data)
- **Details**:
  - Model: checkpoint_epoch_12.pt
  - Training data: 5,495 pairs from fragmented corpus
  - Accuracy: 98.9%
  - Embedding dim: 512

**Why Archived**: Model was trained on fragmented corpus (line-based splitting). New model trained on properly segmented sentences (20,985 sentences, 58,355 pairs).

---

## Archived Data

### `data/corpus_index_old/`
- **Original**: `data/corpus_index/`
- **Date Archived**: 2025-11-13
- **Reason**: Built from fragmented corpus (49K line fragments)
- **Details**:
  - 49,066 fragments (line-based splitting)
  - Model: checkpoint_epoch_12.pt (old)
  - Results: Sentence fragments, incomplete text

**Why Archived**: Corpus was split by lines instead of sentences, resulting in unusable fragments like "kaj li estis konten-" (cut mid-word).

### `data/corpus_index_old_gutenberg/`
- **Date Archived**: 2025-11-12
- **Reason**: Even older index from earlier experiments
- **Can be deleted**: Yes (no longer needed)

### `data/training_pairs/`
- **Date Archived**: Implicitly (replaced by training_pairs_v2)
- **Reason**: Too small (5,495 pairs), built from fragments
- **Details**:
  - 495 positive pairs
  - 5,000 negative pairs
  - Total: 5,495 pairs
  - Quality: Fragment-based

**Why Archived**: Replaced by `data/training_pairs_v2/` (58,355 pairs from proper sentences).

---

## Current (Active) Files

### Models
- **`models/tree_lstm/`** - NEW retrained model (20 epochs, 58K pairs)

### Data
- **`data/corpus_sentences.jsonl`** - Properly segmented corpus (20,985 sentences)
- **`data/corpus_index/`** - Index built from proper sentences
- **`data/training_pairs_v2/`** - New training data (58,355 pairs)

---

## Cleanup Recommendations

**Safe to delete** (after validation):
1. `data/corpus_index_old_gutenberg/` - Very old, superseded
2. `data/training_pairs/` - Replaced by training_pairs_v2
3. `data/corpus_index_old/` - Keep for comparison for now, can delete after validation

**Keep for comparison**:
1. `models/tree_lstm_old/` - Keep to run comparison tests
2. `data/corpus_index_old/` - Keep for before/after comparison

**Timeline for deletion**: After running `scripts/compare_models.py` and validating improvement, can delete old indexes and training data (estimated: after 2025-11-20).

---

## What Changed

**Problem**: Corpus was split by lines, creating fragments:
```
"kaj li estis konten-"  ← CUT MID-WORD
"Mitrandiro estis, mi"  ← INCOMPLETE
```

**Solution**: Proper sentence segmentation:
```
"Estis oficiale anoncite, ke Sam iros al Boklando \"por servi al s-ro Frodo kaj
 prizorgi ties ĝardeneton\": aranĝo, kiun aprobis la Avulo, kvankam tio ne
 konsolis lin rilate Lobelian, kiel estontan nabarinon."
```

**Impact**:
- Corpus: 49K fragments → 21K sentences (57% reduction, higher quality)
- Training: 5.5K pairs → 58K pairs (10.6x increase)
- Model: 12 epochs → 20 epochs
- Results: Complete sentences instead of fragments
