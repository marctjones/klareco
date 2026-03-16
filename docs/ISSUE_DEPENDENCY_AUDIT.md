# Issue Dependency Audit - 2026-03-09

## Summary

**Status**: ❌ **INCOMPLETE** - Missing sub-issues and dependencies not set up correctly

### Problems Found:

1. ❌ **CLI Epics missing sub-issues** - Epics #640-642 have no sub-issues
2. ❌ **Dependencies not set up** - #617-636 should be blocked by #637 and #641
3. ❌ **Missing deterministic model issues** - No issues for M2.1/M2.2 rule systems
4. ❌ **Epic #637 incomplete** - Only 2 of 4 commands have issues

## Issue Structure Analysis

### Epic #616: Tier-Filtered Model Retraining ✅ (Has Sub-Issues, Missing Dependencies)

**Status**: Sub-issues exist (#617-636) but dependencies NOT set up correctly

| Issue | Title | Dependencies Set? | Should Block |
|-------|-------|------------------|--------------|
| #617 | Root 1: Generate training data | ❌ No | Should be blocked by #641 (data pipeline) |
| #618 | Root 2: Train embeddings | ❌ No | Should be blocked by #617, #637 (inspection tools) |
| #619 | Root 3: Test quality | ❌ No | Should be blocked by #618, #637 (inspection tools) |
| #620 | Root 4: Integration | ❌ No | Should be blocked by #619 |
| #621 | M1 1: Generate training data | ❌ No | Should be blocked by #620, #641 (data pipeline) |
| #622 | M1 2: Train model | ❌ No | Should be blocked by #621, #637 (inspection tools) |
| #623 | M1 3: Test accuracy | ❌ No | Should be blocked by #622, #637 (inspection tools) |
| #624 | M1 4: Integration | ❌ No | Should be blocked by #623 |
| #625 | Entity 1: Generate training data | ❌ No | Should be blocked by #641 (data pipeline) |
| #626 | Entity 2: Train classifier | ❌ No | Should be blocked by #625, #637 (inspection tools) |
| #627 | Entity 3: Test accuracy | ❌ No | Should be blocked by #626 |
| #628 | Entity 4: Integration | ❌ No | Should be blocked by #627 |
| #629 | Reranker 1: Generate training data | ❌ No | Should be blocked by #641 (data pipeline) |
| #630 | Reranker 2: Train model | ❌ No | Should be blocked by #629, #637 (inspection tools) |
| #631 | Reranker 3: Test quality | ❌ No | Should be blocked by #630 |
| #632 | Reranker 4: Integration | ❌ No | Should be blocked by #631 |

**Missing**: No issues for Phase 0 dependencies!

### Phase 0: AST Infrastructure ✅ (Has Issues, Missing Dependencies)

| Issue | Title | Dependencies Set? | Should Block |
|-------|-------|------------------|--------------|
| #633 | Phase 0.1: Design ASTAnnotator | ❌ No | None |
| #634 | Phase 0.2: Implement base class | ❌ No | Should be blocked by #633 |
| #635 | Phase 0.3: Update M0 parser | ❌ No | Should be blocked by #634 |
| #636 | Phase 0.4: Validation tests | ❌ No | Should be blocked by #635 |

**Status**: ✅ Issues exist, ❌ Dependencies not set

### Epic #637: CLI Phase 1 - Inspection Tools ⚠️ (Incomplete)

**Status**: Epic exists, only 2 of 4 commands have issues

| Command | Issue | Status |
|---------|-------|--------|
| `klareco inspect ast` | #638 | ✅ Exists |
| `klareco inspect annotations` | ❌ Missing | ❌ Not created |
| `klareco inspect tensor` | #639 | ✅ Exists |
| `klareco annotate text` | ❌ Missing | ❌ Not created |

**Missing**: 2 sub-issues

### Epic #640: CLI Phase 2 - Training Lifecycle ❌ (No Sub-Issues)

**Status**: Epic exists, NO sub-issues created

| Command | Issue | Status |
|---------|-------|--------|
| `klareco train roots` | ❌ Missing | ❌ Not created |
| `klareco train m1` | ❌ Missing | ❌ Not created |
| `klareco train entity` | ❌ Missing | ❌ Not created |
| `klareco train reranker` | ❌ Missing | ❌ Not created |
| `klareco train status` | ❌ Missing | ❌ Not created |

**Missing**: 5 sub-issues

### Epic #641: CLI Phase 3 - Data Pipeline ❌ (No Sub-Issues)

**Status**: Epic exists, NO sub-issues created

| Script | Issue | Status |
|--------|-------|--------|
| `data/export_roots_v2.1.py` | ❌ Missing | ❌ Not created |
| `data/export_m1_triples_v2.1.py` | ❌ Missing | ❌ Not created |
| `data/export_entity_labels_v2.1.py` | ❌ Missing | ❌ Not created |
| `data/export_reranker_pairs_v2.1.py` | ❌ Missing | ❌ Not created |
| `data/validate_training_data.py` | ❌ Missing | ❌ Not created |
| `data/stats_training_data.py` | ❌ Missing | ❌ Not created |

**Missing**: 6 sub-issues

### Epic #642: CLI Phase 4 - Pipeline Debugging ❌ (No Sub-Issues)

**Status**: Epic exists, NO sub-issues created

| Command | Issue | Status |
|---------|-------|--------|
| `klareco inspect pipeline` | ❌ Missing | ❌ Not created |
| `klareco query explain` | ❌ Missing | ❌ Not created |
| `klareco test pipeline` | ❌ Missing | ❌ Not created |
| `klareco test model` | ❌ Missing | ❌ Not created |

**Missing**: 4 sub-issues

### Deterministic Models ❌ (No Issues At All!)

**Status**: NO issues created for deterministic rule systems

| Component | Issue | Status |
|-----------|-------|--------|
| M2.1 Taxonomy - ReVo loader | ❌ Missing | ❌ Not created |
| M2.1 Taxonomy - ConceptNet loader | ❌ Missing | ❌ Not created |
| M2.1 Taxonomy - Affix rules | ❌ Missing | ❌ Not created |
| M2.1 Taxonomy - Fallback model | ❌ Missing | ❌ Not created |
| M2.2 Coreference - Grammar matching | ❌ Missing | ❌ Not created |
| M2.2 Coreference - Recency heuristics | ❌ Missing | ❌ Not created |
| M2.2 Coreference - Disambiguation model | ❌ Missing | ❌ Not created |
| Entity Classifier - Tier1-2 deterministic | ❌ Missing | ❌ Not created |
| Entity Classifier - Tier3 learned | ❌ Missing | Already covered by #625-628 |

**Missing**: 7-8 sub-issues for deterministic models

## Correct Dependency Chain

### Critical Path (Must Complete in Order)

```
Phase 0: AST Infrastructure (#633-636)
  └─ BLOCKS Epic #637: Inspection Tools (#638-639 + missing)
      └─ BLOCKS Epic #641: Data Pipeline (missing sub-issues)
          └─ BLOCKS Epic #616: Model Training (#617-636)
              └─ ENABLES Epic #640: Training CLI (missing sub-issues)
                  └─ ENABLES Epic #642: Debugging CLI (missing sub-issues)
```

### Parallel Tracks (Can Work Simultaneously)

**Track 1: CLI Infrastructure**
```
Epic #637 (Inspection)
  ↓
Epic #640 (Training CLI)
  ↓
Epic #642 (Debugging CLI)
```

**Track 2: Model Training**
```
Phase 0 (AST)
  ↓
Epic #641 (Data Pipeline)
  ↓
Epic #616 (Model Training)
  Phase 1: Root Embeddings
  Phase 2: M1 Selectional
  Phase 3: Entity Classifier
  Phase 4: Reranker
```

**Track 3: Deterministic Models** (Can start after Phase 0)
```
Phase 0 (AST)
  ↓
M2.1 Taxonomy (90% deterministic)
M2.2 Coreference (80% deterministic)
Entity Tier1-2 (100% deterministic)
```

## What Needs to Be Created

### Immediate Priority (Week 1)

1. **Complete Epic #637** (Missing 2 issues):
   - [ ] `klareco inspect annotations`
   - [ ] `klareco annotate text`

2. **Set up Phase 0 dependencies**:
   - [ ] Add "Blocked by" comments to #634-636

3. **Set up Epic #616 dependencies**:
   - [ ] Add "Blocked by #637" to #618, #619, #622, #623, #626, #630
   - [ ] Add "Blocked by #641" to #617, #621, #625, #629

### High Priority (Week 2)

4. **Create Epic #641 sub-issues** (6 issues):
   - [ ] Create data export script for roots
   - [ ] Create data export script for M1 triples
   - [ ] Create data export script for entity labels
   - [ ] Create data export script for reranker pairs
   - [ ] Create data validation script
   - [ ] Create data stats script

### Medium Priority (Week 3-4)

5. **Create Epic #640 sub-issues** (5 issues):
   - [ ] Implement `klareco train roots`
   - [ ] Implement `klareco train m1`
   - [ ] Implement `klareco train entity`
   - [ ] Implement `klareco train reranker`
   - [ ] Implement `klareco train status`

6. **Create deterministic model issues** (7-8 issues):
   - [ ] M2.1: Load ReVo definitions
   - [ ] M2.1: Load ConceptNet relations
   - [ ] M2.1: Implement affix rules
   - [ ] M2.1: Train fallback model
   - [ ] M2.2: Implement grammar matching
   - [ ] M2.2: Implement recency heuristics
   - [ ] M2.2: Train disambiguation model
   - [ ] Entity: Implement tier1-2 deterministic

### Lower Priority (Month 2)

7. **Create Epic #642 sub-issues** (4 issues):
   - [ ] Implement `klareco inspect pipeline`
   - [ ] Implement `klareco query explain`
   - [ ] Implement `klareco test pipeline`
   - [ ] Implement `klareco test model`

## Total Missing Issues

| Category | Missing | Priority |
|----------|---------|----------|
| Epic #637 sub-issues | 2 | 🔴 Critical |
| Epic #641 sub-issues | 6 | 🔴 Critical |
| Epic #640 sub-issues | 5 | 🟡 High |
| Epic #642 sub-issues | 4 | 🟢 Medium |
| Deterministic models | 7-8 | 🟡 High |
| **Total** | **24-25** | |

## Dependency Setup Needed

| Issue Range | Add Dependency | Priority |
|-------------|----------------|----------|
| #634-636 | Blocked by previous Phase 0 issue | 🔴 Critical |
| #618, #619, #622, #623, #626, #630 | Blocked by #637 (inspection tools) | 🔴 Critical |
| #617, #621, #625, #629 | Blocked by #641 (data pipeline) | 🔴 Critical |
| All Epic #616 issues | Blocked by #633-636 (Phase 0) | 🔴 Critical |

## Recommended Actions

### This Session (Next 30 minutes)

1. ✅ Create missing Epic #637 sub-issues (2 issues)
2. ✅ Create Epic #641 sub-issues (6 issues)
3. ⚠️ Add dependency comments to Epic #616 issues

### Next Session

4. Create Epic #640 sub-issues (5 issues)
5. Create deterministic model issues (7-8 issues)
6. Create Epic #642 sub-issues (4 issues)
7. Set up all dependency chains properly

## GitHub Issue Features for Dependencies

**GitHub doesn't have native "blocked by" relationships**, but we can use:

1. **Task lists in issue body**:
   ```markdown
   ## Dependencies
   - [ ] Blocked by #637 (Inspection Tools)
   - [ ] Blocked by #641 (Data Pipeline)
   ```

2. **Comments**:
   ```
   ⚠️ BLOCKED: This issue cannot be started until #637 (Inspection Tools) is complete.
   ```

3. **Labels**:
   - `blocked` label
   - `ready-to-start` label (remove `blocked`, add this when unblocked)

4. **Milestones**:
   - Group related issues into milestones
   - Shows progress visually

## Conclusion

**Answer to "Do we have issues for everything?"**: ❌ **NO**

- Missing: 24-25 issues
- Missing: Proper dependency setup for existing issues

**Answer to "Are dependencies set up correctly?"**: ❌ **NO**

- Phase 0 → Inspection → Data → Training dependency chain NOT set up
- Individual issue blocking NOT documented

**Recommendation**: Create missing issues and set up dependencies ASAP to prevent:
- Working on blocked issues too early
- Confusion about what can be started when
- Wasted effort on tasks that will need rework
