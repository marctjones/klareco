# Klareco Current Plan

**Last updated**: 2026-01-17

---

## The Plan: Multi-Model Semantic Architecture

Klareco builds **small, specialized semantic models** (5-50M params each) that work together on top of a deterministic AST foundation. This approach maximizes explainability and efficiency by keeping grammar 100% deterministic and focusing learned parameters on semantic understanding.

**Philosophy**: Use AST structure + specialized models + RAG, NOT a monolithic LLM.

---

## Architecture Overview

```
INPUT: "Kiu manĝas viandon?"
    ↓
┌─────────────────────────────────────────────┐
│  M0: Deterministic Layer (0 params)         │
│  - Parser (16 Esperanto grammar rules)      │
│  - AST generation                           │
│  - Morpheme decomposition                   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Stage 1: Root Embeddings (~320K params)    │
│  - 64D embeddings for content words only    │
│  - Function words excluded (deterministic)  │
│  - Trained on 4.2M sentence corpus          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  M1: Selectional Preference (10M params)    │
│  - Subject-verb-object compatibility        │
│  - Rejects implausible triples              │
│  - Example: "ideo manĝas viandon" = invalid │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  M2: Taxonomic + Discourse (40-50M params)  │
│  - Taxonomic: IS-A relationships (10M)      │
│  - Discourse: Passage coherence (30-50M)    │
│  - Used for retrieval reranking             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  M3: Orchestration & Integration            │
│  - Multi-model coordination                 │
│  - RAG with Kuzu graph database             │
│  - AST-first retrieval                      │
└─────────────────────────────────────────────┘
    ↓
OUTPUT: "Homo manĝas viandon" + explanation
```

**Total learned parameters**: ~60-70M (vs 1B+ for typical LLMs)

---

## Current Status (2026-01-17)

### ✅ M0: Parser & AST Foundation (COMPLETE)

**Status**: Production-ready
**Components**:
- Parser: 16 deterministic Esperanto grammar rules
- AST structure: Explicit roles (subjekto, verbo, objekto, aliaj)
- Morpheme decomposition: radiko, prefikso, sufiksoj, vortspeco
- Parse rate: 91.8% on 4.2M sentence corpus
- Deparser: AST → text reconstruction (deterministic)

**Key files**:
- `klareco/parser.py` - Core parser implementation
- `klareco/deparser.py` - AST to text
- `klareco/ast_to_graph.py` - AST to PyG graph conversion

---

### ✅ Stage 1: Root Embeddings (COMPLETE - needs retrain)

**Status**: Trained, needs retrain due to vocabulary corruption (Issue #479)
**Architecture**: 320K parameters
- Root embeddings: 64D (10,819 roots)
- Function words excluded (handled deterministically)
- Compositional: Generalizes to unseen word combinations

**Current issues**:
- **#479**: Vocabulary corruption - retrain with Tier 5 only (CRITICAL)
- Need clean vocabulary (18,928 roots from Tier 2-5)

**Key files**:
- `klareco/embeddings/compositional.py` - Compositional embedding model
- `models/root_embeddings/best_model.pt` - Trained model (needs retrain)
- `scripts/train_roots.sh` - Training script
- `data/vocabularies/clean_roots_tier2-5.json` - Expanded vocabulary

---

### 🚧 M1: Selectional Preference Model (IN PROGRESS)

**Status**: Model trained, validation issues found
**Architecture**: 10M parameters
- Subject-verb-object compatibility scoring
- Trained on hard negatives (selectional violations)
- Accuracy: 80.2% overall, 83% plausible detection

**Current issues**:
- **#475**: Object selectional preference not working (ACTIVE)
- **#479**: Blocked by Stage 1 vocabulary corruption
- Need selectional-aware hard negative generation

**Key files**:
- `scripts/train_m1_selectional.py` - Training script
- `scripts/prepare_m1_training_data_hard_negatives.py` - Data generation
- `data/training/m1_selectional_hard_only/` - Training data (17K examples)
- `tests/test_m1_model_quality.py` - Quality tests

**Related issues**: #442, #475, #476, #477, #478

---

### ❌ M2: Taxonomic + Discourse Models (TODO)

**Status**: Not started
**Target**: 40-50M parameters total

**M2.1: Taxonomic Model** (10M params) - Issue #443
- Pure IS-A relationships (hundo IS-A besto)
- Training: ReVo hypernyms + Fundamento
- Remove co-occurrence data (use pure taxonomy)
- Used for: Query expansion, semantic clustering

**M2.2: Discourse Coherence Model** (30-50M params) - Issue #444
- Sentence-level coherence scoring for passage ranking
- Training: Adjacent sentences = coherent, random = incoherent
- Critical for retrieval reranking
- Status: Needs document structure verification

**Key files** (to be created):
- `scripts/train_taxonomic.py` - Taxonomic training
- `scripts/train_discourse.py` - Discourse training
- `data/dictionaries/revo_semantic_relations.json` - Source for taxonomy

---

### ❌ M3: Orchestration & Integration (TODO)

**Status**: Research phase
**Target**: No learned parameters (orchestration logic)

**Components**:
- Multi-model coordination (Issue #449)
- Kuzu graph database for retrieval (ACTIVE - 5.2GB index)
- AST-first retrieval pipeline
- Hybrid retrieval (deterministic + semantic)

**Current infrastructure**:
- ✅ Kuzu inverted index (`klareco/rag/kuzu_inverted_index.py`)
- ✅ AST-aware retriever (`klareco/rag/ast_aware_retriever.py`)
- ❌ Multi-model orchestrator (TODO)
- ❌ Integrated pipeline (TODO)

**Key files**:
- `klareco/rag/kuzu_inverted_index.py` - Kuzu-backed retrieval
- `klareco/rag/ast_aware_retriever.py` - AST-first retrieval
- `data/indexes/kuzu_index/` - Graph database (5.2 GB)

**Related issues**: #449, #453 (Epic)

---

## Key Principles

### 1. Decomposable Contributions (Core Thesis)

**Explainability doesn't require zero learned parameters—it requires decomposable contributions.**

Every prediction can be traced to its sources:
- What came from deterministic rules? (grammar, morphology)
- What came from learned models? (semantic similarity, selectional preference)
- What evidence was retrieved? (citation trails)

A prediction might be "77% deterministic rule (mal- means opposite), 23% learned adjustment (context: moral judgment)."

### 2. Function Word Exclusion Principle

**Function words** (kaj, de, en, la, mi) are grammatical, not semantic:
- Handled by deterministic AST layer, NOT learned embeddings
- Including them causes embedding collapse (all words become similar)
- Only **content words** (hundo, tablo, legi, bela) get learned embeddings

This is a core architectural decision, not a workaround.

### 3. AST-First Pipeline

Everything operates on structured Abstract Syntax Trees, not raw text:

```
Text → Parser (rules) → AST → Semantic Models → AST → Deparser → Text
       └─ deterministic     └─ learned          └─ deterministic
```

---

## What's IN SCOPE

### Current Focus (Next 2-4 weeks)
1. **Fix Stage 1 vocabulary corruption** (Issue #479) - CRITICAL
2. **Fix M1 selectional preference** (Issue #475) - HIGH PRIORITY
3. **Build M2 Taxonomic Model** (Issue #443) - NEXT
4. **Build M2 Discourse Model** (Issue #444) - NEXT

### Near-term (1-2 months)
- M3 orchestration design (Issue #449)
- Multi-model integration
- End-to-end pipeline validation
- Demo system showing all models working together

### Medium-term (3-6 months)
- Additional M2 models (thematic roles, syntagmatic, attribute-noun)
- RAG integration with all models
- Comprehensive evaluation suite
- Performance optimization

---

## What's OUT OF SCOPE

**Not building** (see Issue #452):
- ❌ Coreference resolution (use rules + discourse coherence)
- ❌ Sentiment analysis (use lexicon-based)
- ❌ Metaphor detection (use selectional preference flags)
- ❌ Pragmatics/speech acts (use rules)
- ❌ World knowledge model (use RAG instead - Issue #451)
- ❌ Common sense reasoning (use retrieval + selectional preference)
- ❌ Vision/multimodal
- ❌ Speech/phonology (use deterministic rules)
- ❌ Translation models (Esperanto-only)
- ❌ Monolithic LLM training

---

## Data Assets

### Core Corpus (35 GB)
- **4.2M parsed sentences** with ASTs
- Wikipedia: 4.2M sentences (95% parse quality ≥0.7)
- Books: 108 Gutenberg texts
- Enhanced with metadata, parse statistics
- Location: `data/enhanced_corpus/corpus_with_metadata.jsonl`
- Status: Complete (Issue #28)

### Raw Sources (558 MB)
- Wikipedia dump (348 MB, Nov 2024 snapshot)
- Gutenberg books (20 MB, 108 files)
- Fundamento de Esperanto (371 KB)
- ReVo dictionary database (144 MB)
- See: `data/raw/README.md`

### Kuzu Graph Database (5.2 GB)
- AST-first retrieval infrastructure
- O(1) root lookups, graph traversal for synonyms
- Sentence adjacency for context retrieval
- Location: `data/indexes/kuzu_index/`
- Status: Active, core retrieval system

---

## Development Workflow

### Priority Order
1. **CRITICAL**: Fix Stage 1 vocabulary (#479) - Blocks M1
2. **HIGH**: Fix M1 selectional preference (#475) - Core model
3. **NEXT**: Build M2 models (#443, #444) - Complete semantic portfolio
4. **THEN**: M3 orchestration (#449) - Bring it all together

### Testing Strategy
- Code tests: Implementation correctness (`tests/test_*.py`)
- Data quality tests: Training data validation
- Model quality tests: Performance metrics (`tests/test_m1_model_quality.py`)
- Regression tests: Prevent quality degradation

See: `docs/TESTING_REFERENCE.md`

---

## Success Metrics

### Stage 1 (Root Embeddings)
- ✅ Vocabulary coverage: 18,928 roots (Tier 2-5)
- ✅ Fundamento coverage: 100%
- ✅ No embedding collapse: mean_sim < 0.5
- ✅ ReVo correlation: >0.75
- ⚠️ Status: NEEDS RETRAIN (#479)

### M1 (Selectional Preference)
- ✅ Overall accuracy: >80%
- ✅ Plausible detection: >85%
- ✅ Implausible detection: >70%
- ⚠️ Object selectional: FAILING (#475)

### M2 (Taxonomic + Discourse)
- ❌ Taxonomic: >90% coherent clusters (TODO)
- ❌ Discourse: >70% adjacent vs random distinction (TODO)

### M3 (Integration)
- ❌ End-to-end Q&A: >50% partial match (TODO)
- ❌ Multi-model coordination working (TODO)

---

## Key Design Decisions

### Why Multi-Model vs Monolithic?
- **Modularity**: Each model is independent, trainable, testable
- **Explainability**: Know which model contributed to decision
- **Efficiency**: 60-70M params total vs 1B+ for LLMs
- **Updatable**: Replace/upgrade individual models
- **Esperanto-optimized**: Leverage explicit grammar

### Why Kuzu Database?
- O(1) root lookups (hash-based)
- Native graph traversal for semantic relations
- Memory-efficient (vs loading full corpus)
- Speed: Fast structural pattern matching

### Why AST-First?
- Grammar is free (deterministic)
- Learned models focus on semantics only
- Explainable intermediate representation
- Compositionality built in

---

## Architecture Decisions

**Parser**: Deterministic rules (16 Esperanto grammar rules)
**Stage 1**: Learned root embeddings (320K params, function words excluded)
**M1-M2**: Small specialized models (10-50M params each)
**M3**: Orchestration logic (0 learned params)
**Retrieval**: Kuzu graph database + AST matching
**Output**: Deterministic deparser (grammatically perfect by construction)

**Total learned**: ~60-70M parameters
**Total deterministic**: Grammar, morphology, linearization

---

## Related Documentation

- `VISION.md` - Long-term architecture vision
- `DESIGN.md` - Technical design decisions
- `CLAUDE.md` - Development guidelines for Claude
- `README.md` - Project overview and usage
- `16RULES.MD` - Esperanto grammar specification
- `data/DATA_ORGANIZATION.md` - Data directory structure
- `docs/TESTING_REFERENCE.md` - Testing strategy
- `docs/MODEL_TRAINING_REFERENCE.md` - Training guidelines

**Wiki**:
- Decomposable Contributions Principle
- Function Word Exclusion Principle
- Multi-Model Semantic Architecture (Epic #453)

---

## Quick Reference: Active Issues

**CRITICAL**:
- #479: Fix Stage 1 vocabulary corruption - retrain with Tier 5 only

**HIGH PRIORITY**:
- #475: Improve M1 object selectional preference learning
- #470: Complete test suite implementation

**M2 MODELS** (TODO):
- #442: Build Selectional Preference Model (M1 - in progress)
- #443: Retrain Taxonomic Model with pure taxonomy
- #444: Build Discourse Coherence Model

**M3 INTEGRATION** (TODO):
- #449: Research multi-model orchestration
- #453: Epic - Multi-Model Semantic Architecture
- #468: Stage 1B - Hybrid deterministic + learned affix transforms

---

## Revision History

- **2026-01-17**: Initial creation - defined M0/M1/M2/M3 structure
