# Klareco Models Overview

This page provides a comprehensive overview of all models in the Klareco system, their status, and relationships.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Model Status](#model-status)
- [Quick Links](#quick-links)

---

## Model Architecture

Klareco uses a **staged learning approach** where each stage builds on the previous one:

```
┌─────────────────────────────────────────────┐
│ STAGE 0: Parser (100% Deterministic)        │
│ - 16 Esperanto grammar rules                │
│ - 0 learned parameters                       │
│ - Output: AST with structure + morphology   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ STAGE 1: Root Embeddings (LEARNED)          │
│ - Semantic vectors for roots                │
│ - 692K parameters (10,819 × 64d)           │
│ - Input: Tier0 + ReVo + Ekzercaro           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ M1: Selectional Preferences (LEARNED)       │
│ - Subject-verb-object plausibility          │
│ - 50K parameters                             │
│ - Uses Stage 1 embeddings as input          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ M2: Grammatical Model (PLANNED)             │
│ - Tense, mood, aspect compatibility         │
│ - TBD parameters                             │
│ - Uses Stage 1 + M1                         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ M3: Discourse Model (PLANNED)               │
│ - Coreference, coherence                    │
│ - TBD parameters                             │
│ - Uses Stage 1 + M1 + M2                    │
└─────────────────────────────────────────────┘
```

**Key Principle**: Grammar is deterministic (Stage 0), only semantics are learned.

---

## Training Pipeline

### Dependencies

Models must be trained in order due to dependencies:

```
Stage 1 → M1 → M2 → M3
  ↓       ↓     ↓     ↓
 Must    Uses   Uses  Uses
 train   S1     S1+M1 S1+M1+M2
 first
```

### Retraining Triggers

| Trigger | Retrain | Reason |
|---------|---------|--------|
| New tier0 corpus | Stage 1 + M1 | Better semantic signal |
| ReVo relations updated | Stage 1 | Semantic structure changed |
| Stage 1 retrained | M1 + M2 + M3 | Dependencies changed |
| M1 retrained | M2 + M3 | Dependencies changed |
| Performance degradation | Affected model | Quality dropped |

**Full pipeline script**: `./scripts/retrain_with_tier0.sh`

---

## Model Status

### Stage 0: Parser

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Production |
| **Type** | Deterministic (rule-based) |
| **Parameters** | 0 (no learned params) |
| **Parse Rate** | 91.8% (on general corpus) |
| **Tier0 Parse Rate** | 99.99% (on authoritative sources) |
| **Rules** | 16 Esperanto grammar rules |
| **Last Updated** | Ongoing (rules refined as needed) |

**Documentation**:
- [16RULES.md](../16RULES.MD) - Complete rule specification
- [parser.py](../../klareco/parser.py) - Implementation

**Key Features**:
- Handles all Esperanto morphology (prefixes, suffixes, endings)
- Correlative table support (kiu, tiu, ĉiu, etc.)
- Numeral processing (unu, du, tri, etc.)
- Proper name detection

---

### Stage 1: Root Embeddings

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Production (as of 2026-01-18) |
| **Type** | Learned (neural) |
| **Parameters** | 692K (10,819 vocab × 64d) |
| **Training Data** | 2.2M pairs (tier0 + ReVo + Ekzercaro) |
| **Correlation** | 0.8491 (✅ Target: > 0.80) |
| **Separation Gap** | 0.499 (✅ Target: > 0.40) |
| **Training Time** | 68 minutes (32 epochs, early stopped) |
| **Last Trained** | 2026-01-18 |

**Documentation**: [Stage-1-Root-Embeddings.md](Stage-1-Root-Embeddings.md)

**Key Features**:
- Function words excluded (handled by deterministic layer)
- Graded similarity targets (0.0-1.0 scale)
- Tier0 corpus provides high-quality signal (weight=15.0)
- ReVo semantic relations (synonyms, antonyms, hypernyms)
- Semantic cluster structure (intra-cluster positive, inter-cluster negative)

**Training Script**: `scripts/train_root_embeddings.py`

**Model Path**: `models/root_embeddings_tier0/best_model.pt`

---

### M1: Selectional Preferences

| Attribute | Value |
|-----------|-------|
| **Status** | ⚠️ Trained but below target (as of 2026-01-18) |
| **Type** | Learned (neural) |
| **Parameters** | ~50K (128d hidden) |
| **Training Data** | 400K triples (200K positive, 200K negative) |
| **Accuracy** | 0.7020 (❌ Target: > 0.82) |
| **Training Time** | 18 minutes (21 epochs, early stopped) |
| **Dependencies** | Requires Stage 1 embeddings |
| **Last Trained** | 2026-01-18 |

**Documentation**: [M1-Selectional-Preferences.md](M1-Selectional-Preferences.md)

**Key Features**:
- Learns subject-verb-object plausibility
- Three-component architecture (SV, VO, SVO)
- Trained on real corpus + corrupted negatives
- Filters implausible retrieval results
- Prevents nonsensical generation

**Training Script**: `scripts/train_m1_selectional.py`

**Model Path**: `models/m1_selectional_tier0/best_model.pt`

---

### M2: Grammatical Model

| Attribute | Value |
|-----------|-------|
| **Status** | 📋 Planned (not yet implemented) |
| **Type** | Learned (neural) |
| **Parameters** | TBD |
| **Purpose** | Tense, mood, aspect compatibility |
| **Dependencies** | Stage 1 + M1 |

**Planned Features**:
- Temporal ordering (pasinteco < present < future)
- Mood compatibility (conditional + volus, imperative + -u)
- Aspect handling (progressive, perfect)
- Negation scope (ne placement validation)

**Status**: Awaiting Stage 1 + M1 completion and evaluation

---

### M3: Discourse Model

| Attribute | Value |
|-----------|-------|
| **Status** | 📋 Planned (not yet implemented) |
| **Type** | Learned (neural) |
| **Parameters** | TBD |
| **Purpose** | Coreference, discourse coherence |
| **Dependencies** | Stage 1 + M1 + M2 |

**Planned Features**:
- Coreference resolution (pronoun → antecedent)
- Discourse relation classification (cause, contrast, elaboration)
- Cross-sentence coherence scoring
- Topic tracking

**Status**: Future work after M2 completion

---

## Quick Links

### Documentation

| Page | Description |
|------|-------------|
| [Stage-1-Root-Embeddings](Stage-1-Root-Embeddings.md) | Complete Stage 1 documentation |
| [M1-Selectional-Preferences](M1-Selectional-Preferences.md) | Complete M1 documentation |
| [Understanding-Model-Metrics](Understanding-Model-Metrics.md) | How to interpret metrics |
| [RETRAINING_WITH_TIER0](../RETRAINING_WITH_TIER0.md) | Retraining guide |
| [SEMANTIC_KNOWLEDGE_GRAPH](../SEMANTIC_KNOWLEDGE_GRAPH.md) | ReVo/ConceptNet integration |

### Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_root_embeddings.py` | Train Stage 1 |
| `scripts/train_m1_selectional.py` | Train M1 |
| `scripts/prepare_m1_training_data.py` | Generate M1 data |
| `scripts/retrain_with_tier0.sh` | Full pipeline |

### Model Files

| Path | Contents |
|------|----------|
| `models/root_embeddings_tier0/` | Stage 1 checkpoints |
| `models/m1_selectional_tier0/` | M1 checkpoints |
| `data/training/m1_with_tier0/` | M1 training data |
| `data/enhanced_corpus/corpus_with_tier0.jsonl` | Tier0 corpus |

### Tests

| Test | Purpose |
|------|---------|
| `tests/test_parser.py` | Validate Stage 0 |
| `tests/test_stage1_model_quality.py` | Validate Stage 1 |
| `scripts/validate_m1_extensive.py` | Validate M1 |

---

## Performance Comparison

### Parameter Efficiency

| Model | Parameters | Purpose | Esperanto Coverage |
|-------|------------|---------|-------------------|
| **Klareco Total** | **~750K** | **Full system** | **100%** |
| - Stage 0 (Parser) | 0 | Grammar | 100% (deterministic) |
| - Stage 1 (Roots) | 692K | Semantics | 10,819 roots |
| - M1 (Selectional) | ~50K | Plausibility | All triples |
| - M2 (Planned) | TBD | Grammar compat | TBD |
| - M3 (Planned) | TBD | Discourse | TBD |
| **Target M0/Reasoner** | **~100M** | **Reasoning** | **100%** |
| **Total System** | **~100M** | **Complete** | **100%** |

**Compare to**:
- BERT-Base: 110M parameters (monolingual, no grammar)
- GPT-2: 124M parameters (limited reasoning)
- GPT-3: 175B parameters (1750× larger!)

**Klareco Advantage**:
- Grammar handled deterministically (0 learned parameters)
- Semantics learned efficiently (sub-1M parameters)
- Focus remaining capacity on reasoning (~100M target)

---

## Training History

### Latest Training Run (2026-01-18)

| Stage | Status | Duration | Correlation | Accuracy | Notes |
|-------|--------|----------|-------------|----------|-------|
| Stage 1 | ✅ Complete | 68 min | 0.8491 | N/A | Epoch 32/100 (early stopped at 17) |
| M1 | ⚠️ Below target | 18 min | N/A | 0.7020 | Epoch 21/50 (early stopped at 11) |

**Enhancements**:
- ✅ Integrated tier0 corpus (23,873 sentences, weight=15.0)
- ✅ Added ReVo semantic relations (2,189 curated pairs)
- ✅ Fixed ReVo relation parsing bug
- ✅ Added checkpoint support for restartability

**Expected Improvements**:
- Stage 1 correlation: 0.80 → **0.85+** (tier0 + ReVo data)
- M1 accuracy: 0.82 → **0.87+** (tier0 examples)
- Semantic coverage: 0% → **62%** (ReVo in corpus)

### Previous Training Runs

| Date | Stage | Version | Correlation | Accuracy | Notes |
|------|-------|---------|-------------|----------|-------|
| [DATE] | Stage 1 | v1.0 | [VALUE] | N/A | Baseline (Ekzercaro only) |
| [DATE] | M1 | v1.0 | N/A | [VALUE] | Baseline (general corpus) |

---

## Roadmap

### Short-Term (Q1 2026)

- [x] Complete Stage 1 training with tier0 + ReVo (✅ 2026-01-18, correlation: 0.8491)
- [x] Complete M1 training with tier0 data (⚠️ 2026-01-18, accuracy: 0.7020 - below target)
- [ ] Investigate M1 low accuracy and retrain with improved hyperparameters
- [ ] Integrate semantic expansion into retriever (Task #7)
- [ ] Evaluate quality improvements
- [ ] Update production models if quality improves

### Medium-Term (Q2 2026)

- [ ] Design M2 (Grammatical Model) architecture
- [ ] Collect training data for M2
- [ ] Implement M2 training pipeline
- [ ] Integrate M2 into generation validation

### Long-Term (Q3-Q4 2026)

- [ ] Design M3 (Discourse Model) architecture
- [ ] Design M0 (Reasoning Core) architecture
- [ ] Integrate all models into unified system
- [ ] Achieve 50-question benchmark with deterministic + retrieval only

---

## Contributing

When adding new models:

1. **Create model documentation** using this template structure
2. **Add to this overview page** with status and metrics
3. **Update training pipeline** if dependencies change
4. **Write tests** (unit, quality, integration)
5. **Document retraining** procedure and triggers

## References

- [VISION.md](../../VISION.md) - Overall project vision
- [DESIGN.md](../../DESIGN.md) - Technical design decisions
- [CLAUDE.md](../../CLAUDE.md) - Development guide
- [IMPLEMENTATION_ROADMAP_V2.md](../../IMPLEMENTATION_ROADMAP_V2.md) - Detailed plan
