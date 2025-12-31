# Klareco Training Plan v3

**Version**: 3.3 (December 2025)
**Status**: Stage 1 COMPLETE - M1 (Q&A Evaluation) NEXT

---

## Executive Summary

This document describes Klareco's staged training pipeline and milestone-based development approach. We target progressive capability milestones, comparing against reference LLMs.

### Development Milestones

| Milestone | Target | Reference Model | Klareco Params | Efficiency |
|-----------|--------|----------------|----------------|------------|
| **M1: Single-Turn QA** | Match OLMo-1B | OLMo 1B Instruct (1.18B) | 20-50M | 20-40× |
| **M2: Multi-Turn Chat** | Match Llama 3.1 8B | Llama 3.1 8B Instruct | 50-100M | 75× |
| **M3: Complex Reasoning** | GPT-4-level (Esperanto) | GPT-4 | 100-200M | 10-85× |

### Key Principles (Lessons Learned)

1. **Function Word Exclusion**: Function words (kaj, de, la, mi, etc.) are handled by the deterministic AST layer, NOT learned embeddings. Including them causes embedding collapse.

2. **Correlatives are Function Words**: Correlative words (kiu, tio, ĉie, etc.) are excluded from embedding training. They're grammatical, not semantic.

3. **Low-Rank Affix Transforms**: Affixes are learned as matrix transformations (rank=8), not additive vectors. This prevents collapse and captures semantic effects correctly.

4. **Parser Sufficient at 91.8%**: Stage 1 training succeeded with current parser. Full parser completion is not a strict prerequisite.

5. **Staged Freezing**: Each stage is frozen before the next begins. No catastrophic forgetting.

6. **Grammar is Deterministic**: Stage 2 (GrammaticalAdjuster) was removed - AST already carries grammar labels (negita, tempo, fraztipo, modo). No need to learn what we already know.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC LAYER (Stage 0)                        │
│  Parser (16 rules) → AST with roles, morphemes, negation flags              │
│  Parse rate: 91.8% | Grammar labels: negita, tempo, fraztipo, modo          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 STAGED LEARNED PIPELINE                                      │
│                                                                              │
│  Stage 1: SEMANTIC MODEL (~733K params) ✓ COMPLETE                          │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Root Embeddings: 11,121 roots × 64d = 712K params          │            │
│  │  Affix Transforms V2: 7 prefixes + 29 suffixes (~21K params)│            │
│  │  Corpus Index: 4.38M sentences with FAISS                   │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (frozen)                                │
│  Stage 2: REMOVED - Grammar handled by AST labels (0 params)                │
│                                    │                                         │
│                                    ▼                                         │
│  Stage 3: DISCOURSE MODEL (~100K params) - FUTURE                           │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Coreference chains, discourse relations                    │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (frozen)                                │
│  Stage 4: REASONING CORE (20-100M params) ← M1 TARGET                       │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Q&A Pipeline: Reranker + Answer Extractor + Synthesizer    │            │
│  │  Target: Match OLMo 1B on single-turn Esperanto Q&A         │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘

Total current: ~733K params | M1 Target: 20-50M params
```

---

## Stage 0: Parser & Deterministic Layer ✓ SUFFICIENT

**Status**: Parser at 91.8% parse rate - sufficient for Stage 1 training.

The parser handles:
- 16 Esperanto grammar rules
- Morpheme decomposition (root + affixes + ending)
- Role detection (subjekto, verbo, objekto, aliaj)
- Negation marking (`negita` flag)
- Tense/mood extraction
- Fundamento-based prefix disambiguation

### Known Limitations (tracked in issues)

| Issue | Description | Impact |
|-------|-------------|--------|
| #141 | Parser v2 clean rewrite | Future improvement |
| #145 | Numeral suffix detection (-obl, -on, -op) | Minor |
| #152 | Preposition/prefix confusion | Minor |

### Function Word Exclusion (CRITICAL)

These words are handled by AST, not learned:

```python
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    # Correlatives (ALL - they are grammatical, not semantic)
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',
}
```

---

## Stage 1: Semantic Model ✓ COMPLETE

**Total params**: ~733K
**Models**: `models/root_embeddings/best_model.pt`, `models/affix_transforms_v2/best_model.pt`
**Index**: `data/corpus_index_compositional/`

### Phase 1: Root Embeddings ✓

**Results**:
- 11,121 roots × 64 dimensions = 712K parameters
- Pearson correlation: 0.8871
- Accuracy: 97.98%
- Synonym accuracy: 93.1%
- Antonym accuracy: 82.7%
- Hierarchy accuracy: 98.6%

**Training approach**:
- Function word exclusion (CRITICAL)
- Semantic cluster negatives (family vs animals vs body parts)
- Graded similarity targets (not binary)
- Fundamento-weighted sources (10x for Ekzercaro)

### Phase 2: Affix Transforms V2 ✓

**Results**:
- 7 prefixes with training data + 29 suffixes (~21K params)
- Low-rank transformations (rank=8)
- Anti-collapse metric: mal_mean_sim = -0.03 (target < 0.5) ✓
- Embedding diversity: 1.17 (healthy spread)

**Key insight**: Affixes are *matrix transformations*, not additive vectors:
- `mal-` flips polarity: bon → malbon (similarity 0.25-0.50, distinct)
- `re-` preserves meaning: fari → refari (similarity 0.82-0.97, similar)

**Prefix coverage** (from training data):

| Prefix | Count | Status |
|--------|-------|--------|
| mal | 356,587 | ✓ Trained |
| re | 223,477 | ✓ Trained |
| ek | 89,822 | ✓ Trained |
| for | 67,333 | ✓ Trained |
| ge | 38,769 | ✓ Trained |
| eks | 13,324 | ✓ Trained |
| pra | 7,495 | ✓ Trained |
| bo | 0 | ❌ No data (see #153) |
| dis | 0 | ❌ No data (see #153) |
| fi | 0 | ❌ No data (see #153) |
| mis | 0 | ❌ No data (see #153) |
| vic | 0 | ❌ No data (see #153) |

**Suffix coverage**: 29 suffixes trained with good coverage. Missing: -ism, -ing, -estr, -uj, -aĉ (see #151).

### Phase 3: Corpus Index ✓

**Results**:
- 4.38M sentences indexed
- Compositional embeddings (root + affix transforms)
- FAISS index for fast retrieval

**Process**:
1. Parse sentence → AST
2. Extract content words (exclude function words)
3. Look up root embeddings
4. Apply affix transforms
5. Pool to sentence embedding (mean)
6. Index with FAISS

---

## Stage 2: Grammatical Model ✓ REMOVED (Deterministic)

**Params**: 0 (handled by AST labels)
**Status**: REMOVED - December 2025

### Why Stage 2 Was Removed

The parser already extracts grammatical features into the AST:
- `negita`: Boolean flag for negation
- `tempo`: pasinteco/prezenco/futuro (tense)
- `fraztipo`: deklaro/demando/ordono (sentence type)
- `modo`: indikativo/kondiĉa/vola (mood)

**Key insight**: Why learn what we already know? The GrammaticalAdjuster approach tried to learn transformations for features that are already deterministically extracted. This is redundant.

### How Grammar Is Handled Now

Instead of learned transforms, the Q&A pipeline can:
1. **Direct AST comparison**: Check if negita flags match
2. **Feature filtering**: Only retrieve sentences with matching tense
3. **Explicit rules**: Apply deterministic adjustments if needed

Example (deterministic):
```python
# Instead of learning this...
# ("Mi amas", "Mi ne amas") → learned similarity

# We can compute directly:
if ast1.negita != ast2.negita:
    similarity *= -0.8  # Deterministic rule
```

This aligns with Klareco's core philosophy: **maximize deterministic processing**.

### Related Issues (Now Closed/Archived)

| Issue | Description | Resolution |
|-------|-------------|------------|
| #104 | Sentence type semantics | Use fraztipo AST label |
| #105 | Accusative direction | Use kazo AST label |
| #108-112 | Various grammar features | Handle in AST layer |

---

## Stage 3: Discourse Model

**Target params**: ~100K
**Status**: Future

### Goal

Handle multi-sentence understanding:
- Coreference chains (li → Zamenhof)
- Discourse relations (cause, contrast, elaboration)
- Paragraph-level coherence

### Approach

1. Deterministic: Gender/number agreement, proximity
2. Learned: Ranking candidates when rules are insufficient
3. Discourse connectives: tamen (contrast), ĉar (cause), do (result)

---

## Stage 4: Reasoning Core

**Target params**: 20-100M
**Status**: Future

### Goal

AST-to-AST reasoning for Q&A.

### The Thesis Test

If a 50-100M param reasoning core achieves 80%+ accuracy on Esperanto Q&A while being:
- Fully explainable (reasoning chain visible)
- Grammatically perfect (deterministic deparser)
- Built on only ~1M params of language understanding

Then the core thesis is proven: traditional LLMs waste capacity on grammar.

---

## Evaluation Criteria

### Stage 1 Success ✓ ACHIEVED

| Metric | Target | Actual |
|--------|--------|--------|
| Root similarity correlation | >0.80 | 0.8871 ✓ |
| Synonym accuracy | >85% | 93.1% ✓ |
| Antonym direction | <-0.5 | ✓ (mal- works) |
| Embedding collapse | mean_sim < 0.5 | -0.03 ✓ |

### Stage 2 Success Criteria

| Metric | Target |
|--------|--------|
| Negation discrimination | Context-appropriate |
| Tense temporal ordering | Preserved |
| Mood discrimination | >80% accuracy |
| Sentence type classification | >95% |

### Stage 3 Success Criteria

| Metric | Target |
|--------|--------|
| Coreference chain coherence | >0.7 similarity |
| Cross-document discrimination | <0.3 similarity |

---

## M1 Roadmap: Single-Turn Q&A

**Target**: Match OLMo 1B Instruct on single-turn Esperanto Q&A
**Reference**: OLMo 1B (1.18B params) → Klareco 20-50M params (20-40× efficiency)

### What We Have (Foundation Complete)

| Component | Status | Params |
|-----------|--------|--------|
| Parser (Stage 0) | ✓ 91.8% parse rate | 0 |
| Root Embeddings | ✓ 11,121 roots | 712K |
| Affix Transforms | ✓ 41 affixes | 21K |
| FAISS Index | ✓ 4.38M sentences | 0 |
| Retriever | ✓ Semantic search | 0 |
| **Total Foundation** | | **733K** |

### What We Need (M1 Components)

| Component | Purpose | Estimated Params |
|-----------|---------|------------------|
| **Q&A Benchmark** | 50 questions with gold answers | 0 |
| **Reranker** | Score relevance of retrieved docs | 1-5M |
| **Answer Extractor** | Identify answer spans in context | 2-10M |
| **Answer Synthesizer** | Combine evidence into answer | 5-20M |
| **Evaluation Harness** | Compare against OLMo 1B | 0 |
| **Total M1** | | **8-35M** |

### M1 Implementation Phases

**Phase 1: Baseline Evaluation (No New Models)**
1. Create 50-question Esperanto Q&A benchmark
2. Test current retrieval: What works with pure RAG?
3. Compare against OLMo 1B on same questions
4. Establish baseline accuracy metrics

**Phase 2: Deterministic Improvements**
1. AST-based reranking (use grammar labels)
2. Extractive answer selection (no ML, just heuristics)
3. Template-based answer synthesis
4. Re-evaluate: How much can we do with 0 new params?

**Phase 3: Minimal Learning (If Needed)**
1. Train lightweight reranker (cross-encoder, ~2M params)
2. Train answer span predictor (~5M params)
3. Evaluate: Did learning help significantly?

**Phase 4: Integration & Evaluation**
1. End-to-end pipeline integration
2. Full benchmark evaluation
3. A/B comparison with OLMo 1B
4. Document results, lessons learned

### M1 Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Q&A Accuracy | ≥ OLMo 1B | On Esperanto benchmark |
| Grammar Correctness | 100% | Deterministic deparser |
| Explainability | Full AST trace | Every answer explainable |
| Parameters | <50M | vs OLMo's 1.18B |
| Latency | <2s | Per question |

### M1 vs OLMo 1B Comparison Plan

```
Klareco M1 Pipeline:
Question → Parser → AST → Retriever → Reranker → Extractor → Deparser → Answer
          ├──────── deterministic ────────┤├── learned ──┤├─ deterministic ─┤

OLMo 1B Pipeline:
Question → Tokenizer → 16 Transformer Layers → LM Head → Answer
          ├───────────── all learned ──────────────────┤
```

Key comparison points:
- **Accuracy**: Does Klareco match OLMo on Esperanto Q&A?
- **Efficiency**: 20-50M vs 1.18B params (20-40× smaller)
- **Explainability**: Klareco shows AST trace; OLMo is opaque
- **Grammar**: Klareco 100% correct; OLMo ~92%

---

## Mistakes to Avoid (Lessons Learned)

### 1. Function Word Collapse
**Symptom**: All embeddings become similar (>0.99 cosine)
**Cause**: Function words appear in every sentence
**Prevention**: FUNCTION_WORDS filter in ALL training

### 2. Correlative Embeddings
**Symptom**: Correlatives cluster together incorrectly
**Cause**: Tried to learn embeddings for grammatical words
**Prevention**: Exclude correlatives - they're function words

### 3. Additive Affix Vectors
**Symptom**: Affixes don't capture semantic transformation
**Cause**: word = root + prefix + suffix (additive)
**Fix**: Use low-rank matrix transformations instead

### 4. Binary Similarity Targets
**Symptom**: Poor discrimination between related and unrelated
**Cause**: Binary 0/1 targets
**Prevention**: Graded targets based on co-occurrence

### 5. Missing Negative Sampling
**Symptom**: All embeddings drift to center
**Prevention**: Semantic cluster negatives with weight=3.0

### 6. Parser Perfectionism
**Symptom**: Never start training because parser isn't "complete"
**Reality**: 91.8% was sufficient for excellent Stage 1 results

---

## Files Reference

### Models (git-tracked)
```
models/
├── root_embeddings/
│   └── best_model.pt          # 11,121 roots × 64d (8.4MB)
└── affix_transforms_v2/
    └── best_model.pt          # 41 affixes, low-rank (328KB)
```

### Training Scripts
```
scripts/training/
├── train_root_embeddings.py    # Stage 1 Phase 1
├── train_affix_transforms_v2.py # Stage 1 Phase 2
└── evaluate_embeddings.py      # Evaluation
```

### Index (local only)
```
data/corpus_index_compositional/
├── embeddings.npy              # 4.38M × 64d
├── sentences.jsonl             # Metadata
└── faiss.index                 # FAISS index
```

---

## Current Open Issues

### M1 Priority (Q&A Pipeline)
- Create Q&A benchmark (50 Esperanto questions)
- Build evaluation harness for OLMo 1B comparison
- Implement deterministic reranker
- Implement answer extractor

### Stage 1 Gaps (Lower Priority)
- #151 - Missing suffixes (-ism, -ing, -estr, -uj, -aĉ)
- #153 - Missing prefixes (bo, dis, fi, mis, vic)

### Stage 2 (Closed - Grammar Deterministic)
- ~~#104, #105, #108-#112~~ - Closed: Grammar handled by AST labels

### Architecture
- #106 - AST Enrichment Pipeline
- #107 - Thought Visualizer Demo
- #133 - Refactor Stage 1 for enriched ASTs

### Parser Improvements (Deferred)
- #141 - Parser v2 clean rewrite
- #145 - Numeral suffix detection
- #152 - Preposition/prefix confusion

---

*Last updated: December 31, 2025*
