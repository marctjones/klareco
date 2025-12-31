# Klareco Training Plan v3

**Version**: 3.2 (December 2025)
**Status**: Stage 1 COMPLETE - Stage 2 NEXT

---

## Executive Summary

This document describes Klareco's staged training pipeline. Each stage is trained independently and frozen before the next begins.

### Key Principles (Lessons Learned)

1. **Function Word Exclusion**: Function words (kaj, de, la, mi, etc.) are handled by the deterministic AST layer, NOT learned embeddings. Including them causes embedding collapse.

2. **Correlatives are Function Words**: Correlative words (kiu, tio, ĉie, etc.) are excluded from embedding training. They're grammatical, not semantic.

3. **Low-Rank Affix Transforms**: Affixes are learned as matrix transformations (rank=8), not additive vectors. This prevents collapse and captures semantic effects correctly.

4. **Parser Sufficient at 91.8%**: Stage 1 training succeeded with current parser. Full parser completion is not a strict prerequisite.

5. **Staged Freezing**: Each stage is frozen before the next begins. No catastrophic forgetting.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC LAYER (Stage 0)                        │
│  Parser (16 rules) → AST with roles, morphemes, negation flags              │
│  Parse rate: 91.8% | Handles: S/V/O roles, tense, negation, correlatives    │
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
│  Stage 2: GRAMMATICAL MODEL (~52K params) ← NEXT                            │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Negation, tense, mood, sentence type transforms            │            │
│  │  Minimal pairs training approach                            │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (frozen)                                │
│  Stage 3: DISCOURSE MODEL (~100K params)                                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Coreference chains, discourse relations                    │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (frozen)                                │
│  Stage 4: REASONING CORE (20-100M params) - FUTURE                          │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  AST-to-AST reasoning transformer                           │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘

Total pre-reasoning: ~885K params
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

## Stage 2: Grammatical Model ← NEXT

**Target params**: ~52K
**Status**: Not started

### Goal

Learn semantic effects of grammatical features that are detected deterministically but have semantic content.

### Grammatical Transforms

| Feature | Params | Training Approach |
|---------|--------|-------------------|
| Negation | 4K | Minimal pairs: "Mi amas" vs "Mi ne amas" |
| Tense | 8K | Temporal ordering: past < present < future |
| Mood | 8K | Factual vs hypothetical discrimination |
| Sentence type | 8K | Statement vs question vs command |
| Direction | 4K | Motion vs location (accusative) |
| Comparison | 4K | pli/plej/ol scalar ordering |
| Aspect | 4K | ek- (inchoative), -ad- (continuative) |
| Focus particles | 8K | nur, eĉ, ankaŭ, ja |
| Evidentiality | 4K | verŝajne, certe, eble |

### Training Data Required

Minimal pairs for each grammatical feature:

```python
# Negation - context-dependent, not simple flip
("Mi amas vin", "Mi ne amas vin", similarity=-0.8)
("Estas bone", "Ne estas malbone", similarity=0.6)  # Litotes

# Tense - temporal ordering
("Li venas", "Li venis", similarity=0.7)
("Li venis", "Li venos", similarity=0.4)

# Mood - factual vs hypothetical
("Li venas", "Li venus", similarity=0.3)  # Very different!

# Sentence type
("Li venas", "Ĉu li venas?", similarity=0.5)
```

### Open Issues for Stage 2

| Issue | Description |
|-------|-------------|
| #104 | Sentence type semantic effects |
| #105 | Accusative direction semantics |
| #108 | Comparison semantics (pli/plej/ol) |
| #109 | Aspect semantics (ek-, -ad-) |
| #110 | Focus particle semantics |
| #111 | Evidentiality markers |
| #112 | Possessive semantics |

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

### Stage 1 Gaps
- #151 - Missing suffixes (-ism, -ing, -estr, -uj, -aĉ)
- #153 - Missing prefixes (bo, dis, fi, mis, vic)

### Stage 2 Design
- #104, #105, #108-#112 - Grammatical feature semantics

### Architecture
- #106 - AST Enrichment Pipeline
- #107 - Thought Visualizer Demo
- #133 - Refactor Stage 1 for enriched ASTs

### Parser Improvements
- #141 - Parser v2 clean rewrite
- #145 - Numeral suffix detection
- #152 - Preposition/prefix confusion

---

*Last updated: December 2025*
