# Next Steps - After GNN Training Completes

**Date**: 2025-11-13
**Current Status**: GNN training in progress (20 epochs, ~10-13 hours)

---

## Immediate Actions (When Training Finishes)

### 1. Verify Training Completed Successfully

```bash
# Check if final model exists
ls -lh models/tree_lstm/checkpoint_epoch_20.pt

# Check training log for completion
tail -50 /tmp/retrain_gnn.log

# Look for:
# - "TRAINING COMPLETE"
# - "checkpoint_epoch_20.pt"
# - Final accuracy > 99%
```

**Expected**: Model file ~10-15MB, final training accuracy >99%

---

### 2. Re-index Corpus with New Model

```bash
./reindex_with_new_model.sh
```

**What this does**:
1. Checks for new model (checkpoint_epoch_20.pt)
2. Archives current index → `data/corpus_index_before_retrain_TIMESTAMP`
3. Re-indexes 20,985 sentences with new model
4. Creates embeddings that match new model weights

**Expected time**: ~5 minutes

**Expected output**:
```
✓ Found new model: models/tree_lstm/checkpoint_epoch_20.pt
Archiving old index → data/corpus_index_before_retrain_20251113_210000
✓ Old index archived
Re-indexing corpus with new GNN model...
  Corpus: data/corpus_sentences.jsonl (20,985 sentences)
  Model:  models/tree_lstm/checkpoint_epoch_20.pt
  Output: data/corpus_index/
...
✓ Re-indexing complete!
```

---

### 3. Compare Old vs New Model Performance

```bash
python scripts/compare_models.py
```

**What this tests**:
- 4 standard queries on both old and new models
- Stage 1 candidate counts
- Top-3 scores for each query
- Improvement metrics

**Expected output**:
```
======================================================================
Testing: OLD MODEL (5.5K pairs, 12 epochs)
======================================================================
Query: Kiu estas Frodo? (Who is Frodo?)
  Stage 1: 87 candidates
  Top 3 scores: 0.845, 0.823, 0.801

======================================================================
Testing: NEW MODEL (58K pairs, 20 epochs)
======================================================================
Query: Kiu estas Frodo? (Who is Frodo?)
  Stage 1: 87 candidates
  Top 3 scores: 0.912, 0.889, 0.867

======================================================================
COMPARISON SUMMARY
======================================================================
Kiu estas Frodo? (Who is Frodo?):
  Stage 1 candidates: 87 → 87
  Top score: 0.845 → 0.912
  ✅ Improved by 0.067
```

**What to look for**:
- ✅ Higher top scores (better semantic understanding)
- ✅ Better ranking (most relevant result at #1)
- ✅ All queries show improvement or maintain quality

---

### 4. Test Individual Queries

```bash
# Test with various queries
python scripts/quick_query.py "Kiu estas Frodo?"
python scripts/quick_query.py "Kiu estas Gandalfo?"
python scripts/quick_query.py "Kio estas hobito?"

# Show Stage 1 details
python scripts/quick_query.py "Kiu estas Frodo?" --show-stage1
```

**What to verify**:
- ✅ All results are COMPLETE SENTENCES (not fragments)
- ✅ Results are semantically relevant
- ✅ No "kaj li estis konten-" type fragments
- ✅ Stage 1 finds reasonable candidates (50-200)
- ✅ Stage 2 reranks effectively

**Example good result**:
```
Query: Kiu estas Frodo?

Results:
1. [Score: 0.912] "Estis oficiale anoncite, ke Sam iros al Boklando
   \"por servi al s-ro Frodo kaj prizorgi ties ĝardeneton\": aranĝo,
   kiun aprobis la Avulo, kvankam tio ne konsolis lin rilate Lobelian,
   kiel estontan nabarinon."
   Source: la_hobito.txt

2. [Score: 0.889] "Frodo estis la filo de Drogo Sacvil kaj Primula..."
   Source: la_hobito.txt
```

---

### 5. Run Two-Stage Demo

```bash
python scripts/demo_two_stage.py
```

**What this shows**:
- Stage 1 keyword filtering
- Stage 2 semantic reranking
- Complete pipeline with timing

**Expected**: Fast Stage 1 (<50ms), accurate Stage 2 results

---

## Validation Checklist

After completing steps 1-5, verify:

- [ ] Training completed successfully (20/20 epochs)
- [ ] Final model exists: `models/tree_lstm/checkpoint_epoch_20.pt`
- [ ] Training accuracy: >99%
- [ ] Re-indexing completed: `data/corpus_index/` exists
- [ ] Index contains 20,985 embeddings
- [ ] All test queries return COMPLETE sentences (no fragments)
- [ ] Comparison shows improvement over old model
- [ ] Stage 1 finds reasonable candidates (50-200 per query)
- [ ] Stage 2 reranks results effectively
- [ ] No errors in retrieval pipeline

---

## If Everything Looks Good

### 6. Create Final Summary Report

Create `RAG_IMPROVEMENT_FINAL_REPORT.md` with:
- ✅ Confirmed improvements (comparison results)
- ✅ Example query results (before/after)
- ✅ Performance metrics (timing, accuracy)
- ✅ Validation results

---

### 7. Commit Changes

See commit message in `COMMIT_MESSAGE.md` (to be created).

**Files to commit**:

**New scripts**:
- `scripts/segment_corpus_sentences.py`
- `scripts/generate_training_pairs.py`
- `scripts/convert_training_data.py`
- `scripts/compare_models.py`
- `retrain_gnn.sh`
- `reindex_with_new_model.sh`

**New data** (large files - may need .gitignore or Git LFS):
- `data/corpus_sentences.jsonl` (20,985 sentences)
- `data/training_pairs_v2/*.jsonl` (58,355 pairs)
- `data/corpus_index/` (FAISS index)
- `models/tree_lstm/checkpoint_epoch_20.pt` (new model)

**Documentation**:
- `IMPROVEMENT_GUIDE.md`
- `RAG_IMPROVEMENT_SUMMARY.md`
- `RAG_IMPROVEMENT_WORK_SUMMARY.md`
- `RESULTS_COMPARISON.md`
- `TWO_STAGE_IMPLEMENTATION_SUMMARY.md`
- `NEXT_STEPS.md` (this file)
- `.local_backups/ARCHIVED_FILES.md`

**Modified files**:
- Any retriever or expert changes (check `git status`)

---

## Troubleshooting

### Training didn't complete
```bash
# Check if interrupted
ps aux | grep train_tree_lstm

# If interrupted, just re-run (auto-resumes from last checkpoint)
./retrain_gnn.sh

# Check which epoch it resumed from
tail -100 /tmp/retrain_gnn.log | grep "Resumed from epoch"
```

### Re-indexing fails
```bash
# Check model exists
ls -lh models/tree_lstm/checkpoint_epoch_20.pt

# Run manually with debug output
python scripts/index_corpus.py \
    --corpus data/corpus_sentences.jsonl \
    --output data/corpus_index \
    --model models/tree_lstm/checkpoint_epoch_20.pt \
    --batch-size 32
```

### Retrieval results still poor
```bash
# Compare with old model
python scripts/compare_models.py

# Check if using correct index
ls -lh data/corpus_index/

# Check corpus quality
head -5 data/corpus_sentences.jsonl | jq .
```

### Results show fragments (not complete sentences)
**This should NOT happen** - means corpus wasn't properly segmented.

```bash
# Verify corpus
head -20 data/corpus_sentences.jsonl | jq -r '.text' | head -20

# Should see complete sentences, not:
# - "kaj li estis konten-"
# - "Mitrandiro estis, mi"

# If fragments exist, re-run segmentation:
python scripts/segment_corpus_sentences.py
```

---

## Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Training complete | 0 min | Assuming it finished |
| Re-index corpus | 5 min | 20,985 sentences |
| Run comparison | 2 min | 4 queries × 2 models |
| Test queries | 5 min | Manual testing |
| Validation | 10 min | Check all results |
| Final report | 15 min | Document findings |
| Commit | 5 min | Git add/commit |
| **Total** | **42 min** | After training completes |

---

## Success Criteria

**RAG improvement is successful if**:

1. ✅ All retrieved results are COMPLETE SENTENCES
2. ✅ No fragments like "kaj li estis konten-"
3. ✅ New model shows equal or better semantic scores than old
4. ✅ Stage 1 finds relevant candidates (not everything or nothing)
5. ✅ Stage 2 reranks effectively (best result at top)
6. ✅ Query latency remains acceptable (<200ms total)
7. ✅ Training accuracy >99%
8. ✅ No errors in retrieval pipeline

---

## Long-term Improvements (Future)

After validating this improvement:

1. **BM25 scoring** - Replace binary keyword matching with BM25
2. **Query expansion** - Add synonym/paraphrase generation
3. **Cross-encoder Stage 3** - Final reranking with cross-attention
4. **Character name mapping** - Handle English ↔ Esperanto names
5. **Result caching** - Cache common queries
6. **Multi-hop retrieval** - Chain retrievals for complex questions

See `IMPROVEMENT_GUIDE.md` section "Future Improvements" for details.

---

## Questions?

Check documentation:
- **Implementation details**: `IMPROVEMENT_GUIDE.md`
- **Architecture**: `TWO_STAGE_IMPLEMENTATION_SUMMARY.md`
- **Archived files**: `.local_backups/ARCHIVED_FILES.md`
- **Work summary**: `RAG_IMPROVEMENT_WORK_SUMMARY.md`
- **Before/after**: `RESULTS_COMPARISON.md`
