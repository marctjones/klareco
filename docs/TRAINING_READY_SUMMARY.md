# Training Ready Summary - 2026-03-08

## ✅ What We Completed Today

### 1. Database Classification (COMPLETE)
- ✅ Classified 1.2M Radiko nodes in 2 minutes
- ✅ Propagated to 77.9M Vorto nodes in 6.5 minutes
- ✅ All 6 validation tests passed
- ✅ Optimized from 100 minutes → 8.5 minutes total

**Key Achievement:** Found and eliminated SKIP/LIMIT bottleneck by using COPY FROM + single JOIN UPDATE.

### 2. Tier-Filtered Vocabulary (COMPLETE)
- ✅ Generated 76,352 clean roots from classified database
- ✅ Excluded tier0 function words (187 words) ← **Critical fix for embedding collapse**
- ✅ Excluded tier5 garbage (1M parse failures)
- ✅ Validated: No function words, all expected content words present

**Distribution:**
- Tier 1a (Unua Libro): 750 roots (1.0%)
- Tier 1b (Fundamento): 1,401 roots (1.8%)
- Tier 2 (ReVo): 7,646 roots (10.0%)
- Tier 3 (Corpus): 66,555 roots (87.2%)

### 3. Training Script Updates (COMPLETE)
- ✅ Updated `scripts/train_roots.sh` to use tier-filtered vocabulary
- ✅ Created `scripts/generate_tier_filtered_vocabulary.py`
- ✅ Created `scripts/validate_tier_filtered_training.py`
- ✅ All validation checks pass

---

## 🎯 Ready to Train

Your training infrastructure is now ready with proper tier filtering!

### Start Training Now:

```bash
# Option 1: Train with tier-filtered vocabulary (RECOMMENDED)
./scripts/train_roots.sh --fresh

# Option 2: Validate setup first
python scripts/validate_tier_filtered_training.py

# Option 3: Check what will be trained
python -c "
import json
with open('data/vocabularies/tier_filtered_roots.json') as f:
    vocab = json.load(f)
print(f'Will train on {len(vocab):,} roots')
print(f'Excludes: tier0 function words, tier5 garbage')
"
```

---

## 📊 Expected Improvements

### Before (Old Vocabulary)
- ~10-18K roots (may have included function words)
- **Embedding collapse risk** from tier0 function words
- Training on parse failures (tier5 garbage)
- No provenance tracking

### After (Tier-Filtered Vocabulary)
- **76,352 clean roots**
- **No function words** (tier0 excluded) ← Prevents embedding collapse
- **No garbage** (tier5 excluded)
- **Provenance tracked** (unua_libro, fundamento, revo, korpuso)
- **Frequency weighted** (ofteco available)

### Quality Metrics to Watch

After retraining, validate:

```bash
# 1. Check vocabulary excludes function words
python -c "
import torch
checkpoint = torch.load('models/root_embeddings/best_model.pt')
vocab = checkpoint['idx_to_root']
function_words = ['mi', 'kaj', 'la', 'de', 'mal', 'iĝ']
found = [w for w in function_words if w in vocab.values()]
print('Function words in trained model:', found if found else 'None (GOOD!)')
"

# 2. Check no embedding collapse
python -c "
import torch
checkpoint = torch.load('models/root_embeddings/best_model.pt')
embeddings = checkpoint['model_state_dict']['embeddings.weight']
cos_sim = torch.nn.functional.cosine_similarity(
    embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2
)
mean_sim = (cos_sim.sum() - cos_sim.diagonal().sum()) / (cos_sim.numel() - len(cos_sim))
print(f'Mean pairwise similarity: {mean_sim:.4f}')
print('Status:', 'GOOD' if mean_sim < 0.5 else 'EMBEDDING COLLAPSE!')
"

# 3. Run full test suite
python -m pytest tests/test_embedding_quality.py -v
```

---

## 📁 Files Created/Updated

### New Files
- `data/vocabularies/tier_filtered_roots.json` (6.8 MB) ← **Use this for training**
- `data/vocabularies/tier_filtered_stats.json` - Vocabulary statistics
- `scripts/generate_tier_filtered_vocabulary.py` - Vocabulary generator
- `scripts/validate_tier_filtered_training.py` - Training validation
- `scripts/classify_roots_copy_from.py` - **FASTEST classifier (use this)**
- `scripts/validate_classification.py` - Classification validation
- `docs/V2.1_DATABASE_CLASSIFICATION_COMPLETE.md` - Full classification report
- `docs/NEXT_STEPS_AFTER_CLASSIFICATION.md` - Action plan
- `docs/WHY_CLASSIFICATION_WAS_SLOW.md` - Performance analysis
- `docs/TRAINING_READY_SUMMARY.md` - This file

### Updated Files
- `scripts/train_roots.sh` - Now uses tier-filtered vocabulary by default

### Database
- `data/indexes/v2.1_kuzu_index_full` - Fully classified with tier/source/frequency

---

## 🔍 Troubleshooting

### If training fails with "vocabulary not found":
```bash
python scripts/generate_tier_filtered_vocabulary.py --kuzu data/indexes/v2.1_kuzu_index_full
```

### If you want different tier filtering:
```bash
# Only tier1 + tier2 (core + technical)
python scripts/generate_tier_filtered_vocabulary.py \
  --kuzu data/indexes/v2.1_kuzu_index_full \
  --tiers 1a,1b,2 \
  --output data/vocabularies/tier_1-2_only.json

# With minimum frequency threshold
python scripts/generate_tier_filtered_vocabulary.py \
  --kuzu data/indexes/v2.1_kuzu_index_full \
  --min-ofteco 10 \
  --output data/vocabularies/tier_frequent.json
```

### If you want to use old vocabulary (not recommended):
```bash
./scripts/train_roots.sh --fresh --vocab old
```

---

## 🎓 Key Lessons Learned

### Performance Optimization
1. **SKIP/LIMIT with large offsets is O(n)** - Use cursors, ID ranges, or fetch-all
2. **Kuzu's COPY FROM is 100x faster** than individual INSERTs/UPDATEs
3. **Single UPDATE with JOIN** beats 1M individual transactions
4. **String manipulation overhead can dominate** - The "fast" UNWIND was 3.6x slower!

### Training Data Quality
5. **Function words cause embedding collapse** - Must exclude tier0 from semantic training
6. **Garbage data hurts models** - Must exclude tier5 parse failures
7. **Provenance matters** - Tracking source (unua_libro, revo, etc.) aids debugging
8. **Frequency weighting improves learning** - ofteco enables curriculum learning

### Architecture Decisions
9. **Classification at database level** - Single source of truth for all training
10. **Tier system scales** - Easy to add/remove tiers without code changes
11. **Validation is critical** - Automated checks catch issues early

---

## 📈 Success Metrics

Track these after retraining:

| Metric | Before | Target | Measure |
|--------|--------|--------|---------|
| Vocabulary size | ~10-18K | ~76K | ✓ 76,352 |
| Function words included | Unknown | 0 | ✓ 0 |
| Mean embedding similarity | Unknown | <0.5 | Run after training |
| Fundamento coverage | Unknown | >95% | 100% (all tier1a/1b) |
| Parse failure data | Included | 0% | ✓ 0% |

---

## 🚀 Next Actions (Priority Order)

### 1. Train Root Embeddings (TODAY)
```bash
./scripts/train_roots.sh --fresh
```

**Expected:** ~2-4 hours on CPU, ~76K vocabulary, no embedding collapse

### 2. Validate Model Quality (AFTER TRAINING)
```bash
python scripts/validate_tier_filtered_training.py
python -m pytest tests/test_embedding_quality.py -v
```

### 3. Update M1 Training (THIS WEEK)
- Regenerate M1 training data with tier filtering
- Retrain M1 with clean embeddings
- Compare accuracy vs old model

### 4. Update Documentation (THIS WEEK)
- Update README.md with new vocabulary counts
- Update CLAUDE.md with tier filtering notes
- Create Wiki page on tier system

---

## 💡 Future Enhancements

### Curriculum Learning (Next Month)
- Phase 1: Train on high-frequency words (ofteco > P90)
- Phase 2: Add medium-frequency (P50-P90)
- Phase 3: Add rare words (P10-P50)

### Tier-Specific Evaluation (Next Month)
- Evaluate embeddings separately by tier
- Track improvement per tier
- Identify weak tiers for focused training

### Proper Name Embeddings (Later)
- Consider separate entity embeddings for tier4
- Different architecture for named entities
- Link to entity knowledge graph

---

## ✅ Checklist: Ready to Train?

- [x] Database classified (1.2M Radiko, 77.9M Vorto)
- [x] Tier-filtered vocabulary generated (76K roots)
- [x] Function words excluded (tier0)
- [x] Garbage excluded (tier5)
- [x] Training script updated
- [x] Validation passing
- [ ] ← **YOU ARE HERE** → Run `./scripts/train_roots.sh --fresh`

---

## 🎉 Summary

You've successfully completed the v2.1 database classification and set up tier-filtered training infrastructure!

**The big win:** By excluding tier0 function words, you've likely fixed the embedding collapse issue (Issue #479) that was blocking progress.

**What changed:**
- Old: ~10-18K vocabulary (may have included function words causing collapse)
- New: **76K clean vocabulary** (no function words, no garbage, provenance tracked)

**Ready to train!** 🚀

Run this to start:
```bash
./scripts/train_roots.sh --fresh
```

Monitor training logs in `logs/training/root_training_*.log`

Good luck! 🍀
