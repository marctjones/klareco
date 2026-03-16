# Complete System Implementation - Issue Tracking

**DATE**: 2026-03-09
**STATUS**: All issues created, ready for implementation
**RELATED**: docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md

---

## Epic

**#654: EPIC - Implement Comprehensive Semantic Ontology for Schema-Based Summarization**

All issues below are part of this epic.

---

## Foundation & Infrastructure

### Pure Esperanto Design
- **#664**: Pure Esperanto Semantic Ontology (Self-Reflective Capability)
  - All classifications in Esperanto: kreado, movo, persono
  - Enables self-querying and learning from Esperanto sources

- **#665**: Tier System 2.0 (Frequency × Semantic Priority)
  - Three-dimensional word properties:
    1. Foundational status (Fundamento, dictionary, neologism)
    2. Frequency tier (0/1/2/3)
    3. Semantic class (verba_klaso, aspekta_klaso)
    4. Schema importance (graveco_biografia, graveco_difina, graveco_okazaĵa)

### Semantic Schema
- **#655**: Design and implement Kuzu semantic schema (Layers 1-4)
  - Layer 1: Leksika (50-100 verb classes, 80-120 noun classes)
  - Layer 2: Kadra (FrameNet-style semantic frames)
  - Layer 3: Diskursa (RST relations, information status)
  - Layer 4: Skema (biographical, definitional, event schemas)

---

## Data Annotation (Phase 1-2)

### Phase 1: Core Vocabulary (200 roots)
- **#656**: Classify core vocabulary: 200 roots (Phase 1)
  - Highest priority: Fundamento + high-frequency + schema-important
  - Annotate with: verba_klaso, aspekta_klaso, graveco_biografia, etc.
  - Target: 75% corpus coverage

- **#658**: Bootstrap classification from ReVo/Fundamento/WordNet
  - Semi-automatic classification using existing resources
  - Human validation required

### Phase 2: Extended Coverage (500 roots)
- **#659**: Expand classification to 500 roots (Phase 2)
  - Target: 90% corpus coverage
  - Decides if Model 5 (Unknown Root Classifier) is needed

---

## Learned Models

### Existing Models (Need Fixes)
- **#479**: Fix Stage 1 vocabulary corruption - retrain with Tier 5 only (CLOSED)
  - Root Embeddings (320K params)
  - Status: Closed, may need reopening for clean retrain

- **#475**: Improve M1 object selectional preference learning (OPEN)
  - M1 Selectional Preference (10M params)
  - Status: Has issues with object selectional, needs debugging

### New Models (To Be Trained)
- **#666**: Phase 0 - Validate deterministic baseline before training models
  - 2-week validation phase
  - Test deterministic pipeline on 10 queries
  - Success criteria: 75%+ quality to proceed

- **#667**: Train Model 3 - Importance Adjuster (2M params)
  - Fine-tunes deterministic importance scores
  - Handles: query relevance, redundancy, frequency-based surprise
  - Training data: 5,000-10,000 examples (active learning)
  - Timeline: Phase 2, Week 3-4

- **#668**: Train Model 4 - Reranker (5M params)
  - Reranks RAG retrieval results by semantic relevance
  - Improves retrieval quality beyond BM25
  - Training data: 10,000-20,000 examples
  - Timeline: Phase 2, Week 1-2

- **#669**: Train Model 5 - Unknown Root Classifier (500K params, OPTIONAL)
  - Predicts semantic classes for unknown roots
  - Only train if Phase 2 annotations achieve <90% coverage
  - Training data: 2,000-5,000 examples (hold-out strategy)
  - Timeline: Phase 2, Week 7-8 (if needed)

---

## Deterministic Pipeline Components

### Semantic Enrichment
- **#657**: Implement semantic lookup in fact extraction pipeline
  - Look up semantic classes in Kuzu
  - Fallback to Unknown Root Classifier (if trained)
  - 95% deterministic (lookup), 5% learned (unknown roots)

### Schema Classification
- **#660**: Implement frame semantics (top 20 frames)
  - Semantic frames: Kreado, Movo, Biografio, etc.
  - Thematic roles: Aganto, Paciento, Temo, Spertanto

### Discourse Analysis
- **#661**: Implement RST discourse relation detection
  - Detect discourse markers: ĉar, sed, do, tamen, kvankam
  - Pattern matching for structural relations
  - Classify facts as nucleus vs satellite
  - 95% deterministic (markers), 5% ambiguity

### Importance Scoring
- **#670**: Implement deterministic importance formula (schema + RST + novelty)
  - Factor 1 (40%): Schema slot importance
  - Factor 2 (25%): RST role (nucleus vs satellite)
  - Factor 3 (15%): Information status (given vs new)
  - Factor 4 (10%): Centrality (entity mention frequency)
  - Factor 5 (10%): Sentence position
  - Output: importance score 0.0-1.0

- **#662**: Integrate schema-based fact ranking with semantic annotations
  - Combines all deterministic factors
  - Optional: Add Model 3 (Importance Adjuster) adjustment
  - Hybrid: 80% deterministic, 20% learned

### Fact Selection
- **#671**: Implement fact selection with diversity constraints
  - Select top-N facts by importance
  - Ensure coverage of high-importance schema slots
  - Avoid redundant paraphrases
  - Balance entity representation

### Sentence Generation
- **#672**: Implement AST-based fact clustering for sentence synthesis
  - Group facts into 3-5 clusters (future sentences)
  - Strategy depends on schema type (biographical/definitional/event)
  - Check syntactic compatibility for fusion

- **#673**: Implement deterministic sentence synthesis from fact clusters
  - Fusion strategies:
    1. Single fact → single sentence
    2. Same subject → relative clause (kiu)
    3. Same subject+verb → coordination (kaj, aŭ)
    4. Causal/temporal → subordination (ĉar, kiam, post kiam)
  - Construct grammatically correct Esperanto ASTs
  - Use existing deparser for linearization

### Source Citations
- **#674**: Add source citations to generated summaries
  - Track provenance of facts through entire pipeline
  - Generate citations showing which Wikipedia sentences contributed
  - Support multiple formats: inline footnotes, Wikipedia-style, academic
  - Enable fact-checking and trustworthiness
  - Display source quality (ORO/ARĜENTO/BRONZO)

- **#675**: Add CLI commands for citation lookup and verification
  - `klareco cite N` - Look up citation by number
  - `klareco source Document:ID` - Query source sentence with context
  - `klareco verify` - Validate all citations in summary
  - Interactive mode for exploration
  - Search within documents

---

## Testing & Validation

- **#663**: Test and validate on summarization benchmarks
  - Question answering: 90%+ simple, 80%+ complex
  - Summarization: 85%+ quality (biographical, definitional, event)
  - Explainability: 100% deterministic decisions traceable
  - Citations: 100% of facts traceable to original sources (#674)
  - Coverage: 90-95% with 500-1000 root annotations
  - Speed: 150-250ms per summary

---

## Implementation Timeline

### Phase 0: Validation (2 weeks)
- #666 (Validate deterministic baseline)

### Phase 1: Foundation (8 weeks)
- Week 1-2: #664, #665 (Pure Esperanto design)
- Week 3: #479 (Retrain root embeddings if needed)
- Week 4-5: #656, #658 (200 roots annotation)
- Week 6-7: #655, #657, #660, #661, #670, #671, #672, #673, #674 (Deterministic pipeline + citations)
- Week 8: Evaluate baseline (#666) + CLI citation tools (#675)

### Phase 2: Learned Models (8 weeks)
- Week 1-2: #668 (Train Reranker)
- Week 3-4: #667 (Train Importance Adjuster)
- Week 5-6: #659 (500 roots annotation)
- Week 7-8: #669 (Optional: Train Unknown Root Classifier)

### Phase 3: Optimization (4 weeks)
- Week 1-2: #475 (Fix M1 Selectional)
- Week 3: Hyperparameter tuning
- Week 4: #663 (Final evaluation)

**Total: 22 weeks (5.5 months)**

---

## Dependency Graph

```
#654 (EPIC)
├── #664, #665 (Design: Pure Esperanto + Tiers)
│   └── #655 (Kuzu semantic schema)
│       ├── #656, #658 (200 roots annotation)
│       │   └── #666 (Phase 0 validation)
│       │       └── Phase 1 deterministic pipeline:
│       │           ├── #657 (Semantic lookup)
│       │           ├── #660 (Frame semantics)
│       │           ├── #661 (RST detection)
│       │           ├── #670 (Importance formula)
│       │           ├── #662 (Integration)
│       │           ├── #671 (Fact selection)
│       │           ├── #672 (Fact clustering)
│       │           └── #673 (Sentence synthesis)
│       └── Phase 2 learned models:
│           ├── #668 (Reranker)
│           ├── #667 (Importance Adjuster)
│           ├── #659 (500 roots annotation)
│           └── #669 (Optional: Unknown Root Classifier)
├── #475 (Fix M1 Selectional)
└── #663 (Final testing & validation)
```

---

## Coverage Summary

### ✅ Complete Coverage

**All 5 models**:
1. Root Embeddings (320K) - #479 (closed, may need reopening)
2. M1 Selectional (10M) - #475 (open)
3. Importance Adjuster (2M) - #667 (open)
4. Reranker (5M) - #668 (open)
5. Unknown Root Classifier (500K) - #669 (open, optional)

**All pipeline components**:
- Semantic enrichment - #657
- Schema classification - #660, #655
- RST detection - #661
- Importance scoring - #670, #662, #667
- Fact selection - #671
- Fact clustering - #672
- Sentence synthesis - #673

**All phases**:
- Phase 0: Validation - #666
- Phase 1: Foundation - #664, #665, #655-#662, #670-#673
- Phase 2: Learned Models - #667-#669
- Phase 3: Optimization - #475, #663

### No Gaps!

Every component from `docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md` has a corresponding GitHub issue.

---

## Quick Reference

| Component | Issue | Status | Priority |
|-----------|-------|--------|----------|
| **Epic** | #654 | Open | - |
| **Design** | #664, #665 | Open | Critical (start first) |
| **Schema** | #655 | Open | Critical (foundation) |
| **200 roots** | #656, #658 | Open | High (Phase 1) |
| **500 roots** | #659 | Open | Medium (Phase 2) |
| **Validation** | #666 | Open | Critical (decision point) |
| **Deterministic pipeline** | #657, #660-#662, #670-#675 | Open | High (Phase 1) |
| **Model 3** | #667 | Open | Medium (Phase 2) |
| **Model 4** | #668 | Open | Medium (Phase 2) |
| **Model 5** | #669 | Open | Low (optional) |
| **M1 fix** | #475 | Open | Medium (Phase 3) |
| **Testing** | #663 | Open | High (final validation) |

---

## Next Steps

1. **Start with design** (#664, #665) - Pure Esperanto terminology
2. **Implement schema** (#655) - Kuzu database structure
3. **Run Phase 0** (#666) - Validate deterministic baseline (2 weeks)
4. **Decision point**: If baseline ≥75%, proceed to Phase 1; else, revise design

**All issues are tracked and ready for implementation! 🎉**
