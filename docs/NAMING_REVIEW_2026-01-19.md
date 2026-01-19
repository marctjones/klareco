# Naming Consistency Review - 2026-01-19

**Status**: CRITICAL ISSUES FOUND - DO NOT RUN PIPELINE YET
**Reviewer**: Claude Sonnet 4.5 (as requested by user)

---

## 🚨 CRITICAL Issues (MUST FIX FIRST)

### 1. TIER 2 DISAPPEARS AFTER REBUILD
**Severity**: BLOCKER

**Problem**:
- OLD corpus: Fundamento Krestomatio = tier 2 (6,316 examples)
- NEW extracted sources: Krestomatio = tier 0
- Tier priority training script: `priority_tiers=[0, 2]` expects tier 2

**What will happen**:
1. Corpus rebuild: Krestomatio becomes tier 0 (correct!)
2. Tier 2 completely disappears from corpus
3. Training script crashes or produces unexpected results

**Evidence**:
```bash
# OLD corpus
$ jq -r 'select(.source.tier == 2) | .source.name' corpus_with_metadata.jsonl | uniq
fundamenta_krestomatio  # 6,316 examples

# NEW extracted sources
$ jq '{tier}' data/extracted/eo/tier0_filtered/literary/krestomatio_sentences.jsonl | head -1
{"tier": 0}  # Moved to tier 0!
```

**Fix Options**:

**Option A (RECOMMENDED)**: Remove tier 2 from training, keep Krestomatio as tier 0
```python
# In prepare_m1_training_data_tier_priority.py
- priority_tiers: List[int] = [0, 2]
+ priority_tiers: List[int] = [0]

# In train_m1_semantic_tier_priority.sh
- --priority-tiers 0 2 \
+ --priority-tiers 0 \
```
**Rationale**: Krestomatio IS authoritative, belongs in tier 0

**Option B**: Move Krestomatio back to tier 2
- Update extraction scripts to mark as tier 2
- Rebuild extracted sources
**Rationale**: Maintain compatibility with old corpus

### 2. TIER NUMBERING GAPS
**Severity**: MODERATE (confusing but not breaking)

**Problem**: Claim "tiers 0-6" but only use 0, (2), 5, 6

**Reality**:
- Tier 0: Authoritative grammar (PMEG, Krestomatio, Lingvaj Respondoj)
- Tier 1: **UNDEFINED** (never used)
- Tier 2: Fundamento Krestomatio (will disappear - see Issue #1)
- Tier 3: **UNDEFINED** (never used)
- Tier 4: **UNDEFINED** (never used)
- Tier 5: Wikipedia
- Tier 6: Gutenberg

**Actually used after rebuild**: 0, 5, 6 (3 tiers)

**Fix**: Document the actual system
```markdown
Klareco uses a 3-tier quality system:
- Tier 0: Authoritative (PMEG, Krestomatio, Lingvaj Respondoj) ~22K sentences
- Tier 5: Encyclopedic (Wikipedia) ~3.8M sentences
- Tier 6: Literary (Gutenberg) ~380K sentences

Note: Tiers 1-4 are reserved for future data sources
```

### 3. TWO DIFFERENT "TIER" SYSTEMS
**Severity**: MINOR (confusing terminology)

**Problem**: "Tier" used for two unrelated concepts:
1. **Data quality tiers** (0-6): Corpus/training
2. **Documentation tiers** (1-3): Wiki-Migration-Plan.md

**Evidence**:
```
Wiki-Migration-Plan.md:
- "Tier 1: Educational content" ← Documentation organization
- "Tier 2: Operational guides"
- "Tier 3: Session notes"
```

**Fix**: Rename documentation tiers to "categories"
```markdown
- Category 1: Educational content (→ Wiki)
- Category 2: Operational guides (stay in docs/)
- Category 3: Session notes (→ Discussions)
```

---

## ⚠️ MODERATE Issues (should fix)

### 4. STAGE vs MODEL NUMBERING
**Severity**: MODERATE (confusing but works)

**Problem**: Mixed terminology for model numbering

**Found**:
- Stage 0: Parser (deterministic, no model)
- Stage 1: Root embeddings (692K params)
- M0: Unclear - is this Stage 1?
- M1: Selectional preferences (838K params)
- M2, M3: Referenced but not implemented

**Inconsistency**: Why "Stage" for embeddings but "M" for selectional?

**Fix**: Document the relationship
```markdown
## Model Naming Convention

Klareco uses "Stage N" for training phases and "MX" for reasoning models:

- **Stage 0**: Parser (deterministic, no training)
- **Stage 1**: Root embeddings (foundation for all models)
- **M1**: Selectional preferences (S-V-O plausibility)
- **M2**: Grammatical patterns (planned)
- **M3**: Discourse coherence (planned)

Note: Stage 1 is NOT called "M0" because it's not a reasoning model
```

### 5. MISLEADING DATASET NAME
**Severity**: MINOR (already tracked as Issue #13)

**File**: `data/training/m1_with_tier0/`
**Claim**: Name suggests it contains tier0
**Reality**: Contains ZERO tier0 (0/320K examples)

**Fix**: Rename (already tracked)
```bash
mv data/training/m1_with_tier0 data/training/m1_selectional_no_tier0
```

---

## Minor Issues

### 6. Tier Capitalization
Not critical, just inconsistent:
- "Tier0", "tier0", "Tier 0" all used
- Pick one: recommend "tier 0" (lowercase with space)

---

## Recommendations

### BEFORE Running Corpus Rebuild

1. **Fix tier priority script** (CRITICAL):
   ```bash
   # Edit prepare_m1_training_data_tier_priority.py
   # Change: priority_tiers: List[int] = [0, 2]
   # To: priority_tiers: List[int] = [0]

   # Edit train_m1_semantic_tier_priority.sh
   # Change: --priority-tiers 0 2
   # To: --priority-tiers 0
   ```

2. **Test on small subset** (CRITICAL):
   ```bash
   # Create test corpus with 1000 sentences
   head -1000 data/extracted/wikipedia_sentences.jsonl > /tmp/test_corpus.jsonl

   # Test corpus builder
   python scripts/build_unified_corpus.py --output /tmp/test_corpus_built.jsonl --fresh

   # Check tiers present
   jq -r '.source.tier' /tmp/test_corpus_built.jsonl | sort | uniq -c
   ```

3. **Update documentation**:
   - Document actual 3-tier system (0, 5, 6)
   - Explain tier 1-4 are reserved
   - Clarify Stage vs M numbering

### AFTER Corpus Rebuild

4. **Rename misnamed datasets**:
   ```bash
   mv data/training/m1_with_tier0 data/training/m1_selectional_no_tier0
   ```

5. **Update Wiki-Migration-Plan.md**:
   - Change "Tier 1/2/3" → "Category 1/2/3"

---

## Overall Assessment

**Ready for production**: **NO - CRITICAL BLOCKER**

**Blocking Issue**: Tier 2 removal will break tier priority training script

**Time to fix**: 15-30 minutes (edit 2 files, test)

**After fixes**: READY FOR PRODUCTION

---

## Action Items

### Immediate (before rebuild)

- [ ] **CRITICAL**: Edit `prepare_m1_training_data_tier_priority.py`
  - Change `priority_tiers=[0, 2]` to `priority_tiers=[0]`

- [ ] **CRITICAL**: Edit `train_m1_semantic_tier_priority.sh`
  - Change `--priority-tiers 0 2` to `--priority-tiers 0`

- [ ] **CRITICAL**: Test corpus builder on small subset
  - Verify tiers 0, 5, 6 present
  - Verify tier 2 absent (expected)

- [ ] **Document**: Update tier system docs
  - Clarify 3-tier system (0, 5, 6)
  - Note tiers 1-4 reserved

### After rebuild

- [ ] Rename `m1_with_tier0` → `m1_selectional_no_tier0`
- [ ] Update Wiki-Migration-Plan.md ("Category" not "Tier")
- [ ] Commit documentation updates

---

**Review Date**: 2026-01-19
**Status**: CRITICAL ISSUES FOUND
**Next Step**: Fix tier 2 references BEFORE running corpus rebuild
