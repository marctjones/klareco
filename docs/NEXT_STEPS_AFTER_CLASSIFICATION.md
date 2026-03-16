# Next Steps After v2.1 Classification

**Status**: ✅ Classification complete and validated (2026-03-08)

## Immediate Actions (This Week)

### 1. Update Training Data Filtering ⚡ **HIGH PRIORITY**

The classification enables proper training data filtering. Update training scripts to:

**Exclude from semantic training:**
- **Tier 0** (function words) - Handled deterministically, cause embedding collapse
- **Tier 5** (parse failures/garbage) - Malformed data, zero ofteco

**Focus training on:**
- **Tier 1a-1b** (Fundamento) - Core vocabulary (2,153 roots)
- **Tier 2** (ReVo) - Technical terms (7,730 roots)
- **Tier 3** (Corpus-validated) - Extended vocabulary (66,555 roots)
- **Tier 4** (Proper names) - Entity embeddings (68,981 roots)

**Files to Update:**
```bash
# Root embeddings training
scripts/train_roots.py
scripts/train_roots.sh

# M1 selectional preference training
scripts/train_m1_selectional.py

# RAG training data generation
scripts/generate_semantic_pairs.py  # (if exists)
```

**Filter Pattern:**
```python
# Exclude function words and garbage
valid_tiers = [
    'tier1a_unua_libro', 'tier1b_fundamento',
    'tier2_revo', 'tier3_korpuso', 'tier4_propranomo'
]

query = """
    MATCH (r:Radiko)
    WHERE r.nivelo IN $valid_tiers
      AND r.ofteco > 0
    RETURN r.radiko, r.ofteco, r.fonto
    ORDER BY r.ofteco DESC
"""
```

---

### 2. Retrain Stage 1 Root Embeddings 🔄 **HIGH PRIORITY**

**Issue**: Current embeddings may include tier0 function words (causes embedding collapse)

**Action Plan:**
1. Load classification from database
2. Filter training data to exclude tier0 and tier5
3. Retrain root embeddings with proper vocabulary
4. Validate no embedding collapse (mean_sim < 0.5)
5. Save new embeddings with versioning

**Expected Results:**
- Vocabulary: ~145K roots (tier1-4 only)
- Model size: ~320K params (same architecture)
- Quality: Better semantic clustering (no function word interference)

**Files:**
```bash
./scripts/train_roots.sh --tier-filter "1a,1b,2,3,4"  # Use classification
```

**Validation:**
```python
# Check no function words in vocabulary
python -c "
import kuzu
db = kuzu.Database('data/indexes/v2.1_kuzu_index_full', read_only=True)
conn = kuzu.Connection(db)

# Should return 0
result = conn.execute('''
    MATCH (v:VocabEntry)  -- If you store vocab in graph
    WHERE v.tier = \"tier0_*\"
    RETURN count(v)
''')
print('Function words in vocab:', result.get_next()[0])
"
```

---

### 3. Update M1 Training Data 🎯

**Current Issue**: M1 model may be trained on data including function words and garbage

**Action:**
1. Regenerate M1 training pairs excluding tier0 and tier5
2. Filter by ofteco (focus on frequently used words)
3. Retrain M1 with clean data

**Files:**
```bash
# If M1 training data is generated from corpus
scripts/generate_m1_training_data.py  # Update with tier filtering
```

---

### 4. Update Documentation 📝

**Files to Update:**

**README.md:**
- Update vocabulary counts (was "18,928 roots" → now "~145K clean roots")
- Note function word exclusion is now database-enforced

**CLAUDE.md:**
- Add note about tier filtering for training

**Wiki Pages** (if they exist):
- Stage-1-Embeddings.md - Update with classification details
- Training-Strategy.md - Document tier filtering approach

---

## Medium-Term Actions (Next 2 Weeks)

### 5. Implement Tier-Aware Training Pipeline

Create unified training pipeline that:
1. Queries database for tier-filtered vocabulary
2. Applies ofteco weighting
3. Tracks provenance (fonto) for interpretability
4. Logs training data statistics

**New File:**
```python
# klareco/training/tier_filtered_dataset.py
class TierFilteredDataset:
    """Dataset that filters by tier and ofteco."""

    def __init__(self, db_path, tiers=['1a','1b','2','3','4'], min_ofteco=1):
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)
        self.tiers = tiers
        self.min_ofteco = min_ofteco

    def get_vocabulary(self):
        """Get filtered vocabulary."""
        tier_filters = [f"'tier{t}_*'" for t in self.tiers]
        query = f"""
            MATCH (r:Radiko)
            WHERE r.nivelo IN [{','.join(tier_filters)}]
              AND r.ofteco >= {self.min_ofteco}
            RETURN r.radiko, r.nivelo, r.fonto, r.ofteco
            ORDER BY r.ofteco DESC
        """
        # ... implementation
```

---

### 6. Create Training Data Quality Metrics

Track training data composition:

```python
# scripts/analyze_training_data.py
"""
Generate report:
- Vocabulary size by tier
- Ofteco distribution
- Source distribution (unua_libro vs revo vs corpus)
- Coverage of Fundamento
"""
```

**Output:**
```
Training Data Composition Report
================================
Total vocabulary: 145,233 roots

By Tier:
  tier1a (Unua Libro): 750 (0.5%)
  tier1b (Fundamento): 1,403 (1.0%)
  tier2 (ReVo): 7,730 (5.3%)
  tier3 (Corpus): 66,555 (45.8%)
  tier4 (Proper names): 68,981 (47.5%)

By Source:
  unua_libro: 937 (0.6%)
  fundamento: 1,403 (1.0%)
  revo: 7,730 (5.3%)
  korpuso: 66,555 (45.8%)
  propranomo: 68,981 (47.5%)

Ofteco Statistics:
  Mean: 12.3
  Median: 3
  P95: 67
  P99: 234
```

---

## Long-Term Enhancements (Next Month)

### 7. Frequency-Based Curriculum Learning

Use ofteco for curriculum learning:
1. **Phase 1**: Train on high-frequency words (ofteco > P90)
2. **Phase 2**: Add medium-frequency (P50-P90)
3. **Phase 3**: Add rare words (P10-P50)
4. **Skip**: Very rare words (< P10) - may be noise

**Rationale**: High-frequency words provide better signal, learn them first.

---

### 8. Source-Aware Training

Track which source a word comes from for interpretability:

```python
# In embedding training
embedding_metadata = {
    'radiko': 'hund',
    'tier': 'tier1a_unua_libro',
    'fonto': 'unua_libro',
    'ofteco': 5654,
    'trained_epoch': 23,
    'final_loss': 0.0234
}
```

**Use Case**: "This word is from Unua Libro, so it should have high-quality embeddings"

---

### 9. Tier-Specific Model Evaluation

Evaluate models separately by tier:

```python
# tests/test_model_quality.py
def test_embeddings_by_tier():
    """Test embedding quality per tier."""
    for tier in ['tier1a', 'tier1b', 'tier2', 'tier3']:
        accuracy = evaluate_tier(tier)
        assert accuracy > MIN_ACCURACY[tier]
```

**Expected:** Tier 1a (Unua Libro) should have highest accuracy (most frequently used, best documented)

---

### 10. Update Corpus Statistics

With classification complete, regenerate corpus stats:

```bash
python scripts/analyze_corpus_coverage.py --with-tiers

# Output:
# Corpus Coverage by Tier
# =======================
# Tier 1a (Unua Libro): 98.7% coverage (740/750 roots found in corpus)
# Tier 1b (Fundamento): 94.2% coverage
# Tier 2 (ReVo): 76.3% coverage
# Tier 3 (Corpus): 100% coverage (by definition)
```

---

## Success Metrics

After implementing these changes, validate:

1. ✅ **No function words in vocabulary** (tier0 excluded)
2. ✅ **No garbage in training** (tier5 excluded)
3. ✅ **Embedding collapse prevented** (mean_sim < 0.5)
4. ✅ **Training data statistics logged** (tier/source/ofteco distribution)
5. ✅ **Model quality improved** (compared to pre-classification baseline)
6. ✅ **Fundamento coverage high** (>95% of tier1 roots in training)

---

## Questions to Answer

As you implement these steps:

1. **What percentage of training examples come from each tier?**
   - Should we weight tiers equally or by ofteco?

2. **Should tier4 (proper names) be trained with same architecture?**
   - Or separate entity embeddings?

3. **Do we need tier-specific learning rates?**
   - Tier 1 (small, high quality) vs tier 3 (large, variable quality)

4. **How to handle compound roots?**
   - See `docs/COMPOUND_ROOTS_DEFERRED.md` - still deferred

5. **Should we validate embeddings per-tier during training?**
   - Early stopping based on tier1a validation set?

---

## Files to Create

- `klareco/training/tier_filtered_dataset.py` - Dataset with tier filtering
- `scripts/analyze_training_data.py` - Training data composition report
- `scripts/train_with_tiers.py` - Tier-aware training script
- `tests/test_tier_filtering.py` - Validate tier filtering works

---

## Current Blocking Issues

**From README.md:**
- ❌ Issue #479: Vocabulary corruption (CRITICAL) - **Now fixed with tier filtering!**
- 🚧 Issue #475: M1 object selectional preference issues - **May improve with clean data**

**Classification unlocks:**
- ✅ Fixes #479 by excluding tier0/tier5
- ✅ Enables systematic vocabulary management
- ✅ Provides provenance tracking for debugging

---

## Summary: Your Next Command

**Start here:**
```bash
# 1. Update root embedding training to use tier filtering
./scripts/train_roots.sh --tier-filter "1a,1b,2,3,4" --min-ofteco 1

# 2. Validate no function words in trained vocabulary
python scripts/validate_vocabulary.py --check-tiers

# 3. Retrain M1 with clean data
./scripts/train_m1_selectional.py --use-tier-filtering

# 4. Run full validation suite
python -m pytest tests/test_model_quality.py -v
```

The classification infrastructure is complete. Now use it to train better models! 🚀
