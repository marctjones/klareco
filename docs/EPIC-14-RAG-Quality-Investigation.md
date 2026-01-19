# EPIC #14: RAG Quality Investigation and Training Pipeline Improvements

**Status**: Open
**Created**: 2026-01-19
**Priority**: High

---

## Problem Statement

The RAG demo (`demo_rag_with_m1.py`) shows significant quality issues when answering basic questions about Esperanto:

### Example Query Results (Jan 19, 2026)

**Query 1: "Kiu fondis Esperanton?" (Who founded Esperanto?)**
- ❌ No results directly answer "Zamenhof"
- Top result is a long Volapük comparison that doesn't mention the founder
- M1 plausibility scores: 0.81-0.98 (marked "plausible" despite being irrelevant)

**Query 2: "Kio estas Esperanto?" (What is Esperanto?)**
- ❌ Results don't explain what Esperanto IS
- Top result: Same Volapük comparison
- Result #2: Technical grammar details about consonants
- Result #3: Index page

**Query 3: "Kie naskiĝis Zamenhof?" (Where was Zamenhof born?)**
- ❌ Only 7 results (vs 10 for other queries)
- Results don't contain the answer
- Results #2-5 are duplicates
- M1 scores: 0.72-0.74

### Root Issues Identified

1. **Tier0 data missing from M1 training** (0 out of 320K examples)
2. **M1 over-scores irrelevant answers** (0.7-0.9 for poor matches)
3. **Retrieval ranking prioritizes wrong results**
4. **Corpus may lack direct answers** to basic questions
5. **No clear model/dataset lifecycle policy**

---

## Epic Objectives

1. **Understand why tier0 was filtered out** during training data generation
2. **Fix M1 plausibility scoring** to reject irrelevant answers
3. **Improve retrieval ranking** to surface best answers first
4. **Audit corpus coverage** for fundamental questions
5. **Establish model/dataset lifecycle** policy (keep/delete/recreate)

---

## Sub-Issues

### #12: Investigate Tier0 Filtering (Bug)

**Status**: Open
**Created**: 2026-01-19

**Problem**:
- Corpus contains 22,516 tier0 sentences
- M1 training data has 0 tier0 examples
- Data filtered out during generation

**Investigation Areas**:
- Parse rate threshold (tier0 may parse differently)
- Sentence length filters
- Triple extraction logic (may fail on tier0 patterns)
- Explicit tier filtering in generation script

**Files to Check**:
- `scripts/generate_m1_semantic_data.py`
- `scripts/prepare_m1_training_data_semantic.py`
- `scripts/m1_generate_semantic_data.sh`

**Expected Outcome**: Identify exact filtering logic preventing tier0 inclusion

---

### #15: Investigate M1 Plausibility Scoring (Bug)

**Status**: Open
**Created**: 2026-01-19

**Problem**:
- M1 gives high scores (0.7-0.9) to answers that don't answer the question
- Example: "Kiu fondis Esperanton?" gets Volapük comparison scored 0.98

**Investigation Areas**:
- What does M1 actually measure? (semantic similarity vs. answer relevance)
- Is M1 scoring subject-verb-object plausibility OR question-answer matching?
- Training data quality - were training examples actual Q&A pairs?
- Model architecture - does it understand question semantics?

**Test Cases**:
```python
# Should score HIGH
("Kiu fondis Esperanton?", "Zamenhof fondis Esperanton.")  # Direct answer

# Should score LOW
("Kiu fondis Esperanton?", "Volapuek aperis tiam...")  # Irrelevant
```

**Expected Outcome**: Understand M1's actual capabilities and limitations

---

### #16: Investigate Retrieval Ranking (Bug)

**Status**: Open
**Created**: 2026-01-19

**Problem**:
- Better answers exist but aren't ranked at top
- Irrelevant results consistently appear first

**Investigation Areas**:
- Retrieval algorithm: two-stage (structural + neural)
- Stage 1: How are candidate docs selected?
- Stage 2: How does M1 rerank candidates?
- Are both stages working correctly?
- Is the issue in retrieval OR in M1 filtering?

**Files to Check**:
- `klareco/rag/kuzu_inverted_index.py` (retrieval logic)
- `klareco/rag/retriever.py` (two-stage pipeline)
- `klareco/models/m1_inference.py` (M1 scoring)

**Expected Outcome**: Identify which stage fails to rank good answers highly

---

### #18: Audit Corpus for Basic Question Coverage (Enhancement)

**Status**: Open
**Created**: 2026-01-19

**Problem**:
- We don't know if corpus contains direct answers to fundamental questions
- Retrieval can't find answers that don't exist

**Investigation Method**:
```bash
# Search corpus for answer patterns
jq -r '.text' corpus_full_with_tier0.jsonl | grep -i "zamenhof fond"
jq -r '.text' corpus_full_with_tier0.jsonl | grep -i "esperanto estas"
jq -r '.text' corpus_full_with_tier0.jsonl | grep -i "naskiĝis"
```

**Questions to Check**:
1. "Kiu fondis Esperanton?" → "Zamenhof fondis..."
2. "Kio estas Esperanto?" → "Esperanto estas lingvo..."
3. "Kie naskiĝis Zamenhof?" → "Zamenhof naskiĝis en Bjalistoko..."
4. "Kiam fondiĝis Esperanto?" → "Esperanto fondiĝis en 1887..."

**Expected Outcome**:
- List of fundamental questions
- Coverage report (which answers exist in corpus)
- Identify gaps in corpus content

---

### #17: Audit Model and Dataset Lifecycle (Documentation)

**Status**: Open
**Created**: 2026-01-19

**Problem**:
- Multiple model versions exist (m1_selectional, v2, v3, semantic_full)
- Multiple training datasets (some misnamed)
- No clear policy on what to keep vs. delete

**Audit Tasks**:

#### 1. Document Current State
Create inventory table:

| Model/Dataset | Date | Accuracy | Tier0? | Status | Action |
|---------------|------|----------|--------|--------|--------|
| `m1_semantic_full` | Jan 19 | 86.37% | ❌ | Production | Retrain with tier0 |
| `m1_selectional` | Jan 14 | 80.71% | ❌ | Legacy | Delete |
| `m1_selectional_v2` | Jan 17 | 76.23% | ❌ | Legacy | Delete |
| `m1_selectional_v3` | Jan 17 | 74.55% | ❌ | Legacy | Delete |
| `m1_tier0_only` | Jan 19 | 68.23% | ✅ | Experimental | Keep |
| `root_embeddings_tier0` | Jan 18 | 85.34% | ✅ | Production | Keep |
| `m1_with_tier0` (dataset) | Jan 19 | N/A | ❌ | **MISNAMED** | Rename (#13) |

#### 2. Establish Lifecycle Policy

**Models to KEEP**:
- Latest production models (currently in use)
- Tier0-based models (for comparison)
- Best-performing model in each category

**Models to DELETE**:
- Legacy versions superseded by better models
- Broken models (e.g., `best_model.BROKEN.pt`)
- Models with accuracy < 70% (unless experimental)

**Datasets to KEEP**:
- Latest training data for each model type
- Tier0-only datasets (authoritative)
- Test/validation sets

**Datasets to DELETE**:
- Obsolete training data
- Misnamed datasets (after renaming)
- Duplicate datasets

**Models/Datasets to RECREATE**:
- `m1_semantic_full` → retrain with tier0 included
- `m1_with_tier0` → recreate with actual tier0 data (or rename)

#### 3. Cleanup Script

Create `scripts/cleanup_models.sh`:
```bash
#!/bin/bash
# Archive legacy models
mkdir -p models/archive
mv models/m1_selectional/best_model.pt models/archive/
mv models/m1_selectional_v2/best_model.pt models/archive/
mv models/m1_selectional_v3/best_model.pt models/archive/

# Rename misnamed datasets
mv data/training/m1_with_tier0 data/training/m1_selectional_full

# Document what was kept
scripts/generate_model_inventory.py > docs/Model-Inventory.md
```

**Expected Outcome**:
- Clear inventory of all models/datasets
- Documented lifecycle policy
- Cleanup script to remove obsolete artifacts
- Updated references in code/docs

---

## Timeline and Dependencies

### Phase 1: Investigation (Weeks 1-2)
- [x] Task #12: Investigate tier0 filtering (Week 1)
- [ ] Task #15: Investigate M1 scoring (Week 1)
- [ ] Task #16: Investigate retrieval ranking (Week 1)
- [ ] Task #18: Audit corpus coverage (Week 2)

### Phase 2: Fixes (Weeks 3-4)
- [ ] Task #10: Fix M1 data generation to include tier0 (Week 3)
- [ ] Task #13: Rename misnamed datasets (Week 3)
- [ ] Fix M1 scoring issues (depends on #15) (Week 3)
- [ ] Fix retrieval ranking (depends on #16) (Week 4)

### Phase 3: Improvements (Weeks 5-6)
- [ ] Task #17: Model/dataset lifecycle cleanup (Week 5)
- [ ] Retrain M1 with tier0 data (Week 5)
- [ ] Add missing corpus content (depends on #18) (Week 6)
- [ ] Validate improvements with test queries (Week 6)

---

## Success Metrics

### Before (Current State)
- M1 accuracy: 86.37% (without tier0)
- Query 1 answer quality: ❌ Poor (irrelevant top results)
- Query 2 answer quality: ❌ Poor (no direct answer)
- Query 3 answer quality: ❌ Poor (no answer, duplicates)
- Tier0 in training: 0 / 320K examples (0%)

### After (Target State)
- M1 accuracy: 87-88% (with tier0)
- Query 1 answer quality: ✅ Good (Zamenhof in top 3)
- Query 2 answer quality: ✅ Good (definition in top 3)
- Query 3 answer quality: ✅ Good (birthplace in top 3)
- Tier0 in training: 22K / 342K examples (~6%)
- Models/datasets: Documented and cleaned up

---

## Related Documentation

- `docs/Model-Verification-2026-01-19.md` - Model verification findings
- `docs/Tier0-Data-Inventory-2026-01-19.md` - Complete tier0 audit
- `docs/Helsinki-NLP-Integration.md` - Translation layer (display only)
- `docs/Purity-Guarantee.md` - Klareco Pure Esperanto architecture

---

## Notes

**Key Insight**: The RAG system has THREE separate quality issues:
1. **Training data** (tier0 missing) → affects M1 scoring
2. **M1 model** (over-scores poor answers) → affects filtering
3. **Retrieval** (ranks poorly) → affects candidate selection

All three must be fixed for good end-to-end quality.

**Priority Order**:
1. Fix tier0 filtering (#12, #10) - foundation issue
2. Fix M1 scoring (#15) - critical for answer quality
3. Fix retrieval ranking (#16) - multiplies M1 improvements
4. Audit corpus (#18) - may reveal missing content
5. Lifecycle cleanup (#17) - prevents future confusion

---

**Epic Owner**: TBD
**Last Updated**: 2026-01-19
