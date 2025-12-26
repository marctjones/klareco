# Klareco Training Plan v3

**Version**: 3.0 (December 2024)
**Status**: Redesign incorporating lessons learned from v2 training

---

## Executive Summary

This plan redesigns Klareco's training pipeline based on critical lessons learned:

1. **Function Word Collapse**: High-frequency function words caused all embeddings to collapse
2. **Language Layers**: Semantics alone is insufficient - we need discourse and pragmatics
3. **Parser Gaps**: AST must capture more linguistic phenomena before training
4. **Deterministic First**: Maximize what's handled by rules before any learning

### Key Changes from v2

| Aspect | v2 Approach | v3 Approach |
|--------|-------------|-------------|
| Function words | Included in training | **Excluded** - handled by AST |
| Negation | Ignored | **Detected** deterministically, **effect learned** |
| Grammatical features | Frozen one-hot | **Detected** deterministically, **semantic effect learned** |
| Sentence scope | Individual sentences | **Discourse-aware** multi-sentence |
| Coreference | Not handled | **Hybrid** - rules + learned ranking |
| Compound words | Treated as single root | **Decomposed** when possible |
| Training phases | Linear | **Prerequisites enforced** |

### Critical Insight: Grammatical Features Have Semantics

We previously conflated "can be detected by rules" with "has no semantic content."

**Wrong**: Freeze tense/mood/sentence-type as one-hot features
**Right**: Detect deterministically, but LEARN their semantic effect

See Issue #101 and Wiki [[Grammatical-Features-Semantic-Audit]] for details.

---

## Architecture Overview: AST Enrichment Pipeline (#106)

The key architectural insight is that AST serves as "thought" that passes between models, accumulating semantic meaning at each stage. Each model reads AST annotations and writes new ones.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETERMINISTIC LAYER (Phase 0)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Parser    │  │  Negation   │  │ Coreference │  │  Discourse  │        │
│  │  (16 rules) │  │   Marker    │  │   (rules)   │  │ Connectives │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│                        ┌─────────────────────┐                              │
│                        │    ENHANCED AST     │ ← Deterministic annotations   │
│                        │  - Roles (S/V/O)    │                              │
│                        │  - Negation flags   │                              │
│                        │  - Coref chains     │                              │
│                        │  - Discourse rels   │                              │
│                        │  - Sentence type    │                              │
│                        └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 STAGED LEARNED PIPELINE (AST as "Thought")                   │
│                                                                              │
│  Each model: reads AST → learns semantic effects → writes to AST slots      │
│  Models trained independently, then frozen, then composed                   │
│                                                                              │
│  Stage 1: SEMANTIC MODEL (~333K params)                                     │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Root Embeddings (5K×64d) + Affix Embeddings (50×64d)       │            │
│  │  → Writes: word_embedding, sentence_embedding               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (freeze, pass AST)                      │
│  Stage 2: GRAMMATICAL MODEL (~52K params)                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Dedicated transforms per feature (see Phase 4):            │            │
│  │  - Negation: 4K, Tense: 8K, Mood: 8K, Sentence type: 8K    │            │
│  │  - Direction: 4K, Comparison: 4K, Aspect: 4K               │            │
│  │  - Focus particles: 8K, Evidentiality: 4K, Possessive: 4K  │            │
│  │  → Writes: transformed_embedding + interpretable labels     │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (freeze, pass AST)                      │
│  Stage 3: DISCOURSE MODEL (~100K params)                                    │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  Cross-sentence attention, coreference chain scoring        │            │
│  │  → Writes: discourse_context, coref_scores                  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                    │                                         │
│                                    ▼ (freeze, pass AST)                      │
│  Stage 4: REASONING CORE (20-100M params)                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  AST-to-AST reasoning transformer                           │            │
│  │  → Writes: answer_ast, reasoning_chain                      │            │
│  └─────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘

Total pre-reasoning: ~485K params (vs 110M BERT)
With reasoning core: 20-100M params
```

### AST Slot Architecture

Each model reads from and writes to designated AST slots:

```python
# AST "thought" structure with semantic slots
{
    # Deterministic (from parser)
    'tipo': 'frazo',
    'subjekto': {...},
    'verbo': {...},
    'objekto': {...},
    'negita': True,
    'fraztipo': 'demando',

    # Semantic Model output (Stage 1)
    'semantic': {
        'word_embeddings': [...],      # Per-word vectors
        'sentence_embedding': [...],    # Pooled sentence vector
    },

    # Grammatical Model output (Stage 2)
    'grammatical': {
        'transformed_embedding': [...], # After applying feature transforms
        'negation_effect': 'reversal',  # Interpretable label
        'tense_effect': 'past_reference',
    },

    # Discourse Model output (Stage 3)
    'discourse': {
        'context_embedding': [...],     # With discourse context
        'coref_chain_id': 3,            # Which entity chain
        'discourse_role': 'evidence',   # Role in argument
    },

    # Reasoning Core output (Stage 4)
    'reasoning': {
        'answer_confidence': 0.85,
        'reasoning_chain': [...],       # Steps taken
        'retrieved_evidence': [...],    # Supporting ASTs
    },
}
```

### Why Staged Pipeline?

1. **Independent Training**: Each model trained separately with clear objectives
2. **Interpretability**: Can inspect "thought" at each stage (#107)
3. **Modularity**: Can improve one stage without retraining others
4. **Debugging**: Easy to identify where understanding breaks down
5. **Tiny Models**: Each stage stays small because AST handles structure

---

## Phase 0: Parser & AST Completion (PREREQUISITE)

**Goal**: Complete all deterministic AST features before any training.

**Why First**: Training on incomplete ASTs wastes compute and requires retraining when AST changes.

### 0.1 Parser Bug Fixes (High Priority)

| Issue | Fix | Status |
|-------|-----|--------|
| #89 | Preposition 'por' not recognized | Pending |
| #90 | Adverb root extraction incorrect | Pending |
| #91 | Mood/tense structure inconsistent | Pending |
| #85 | Parser artifacts in morpheme analysis | Pending |

### 0.2 AST Enhancements (High Priority)

| Issue | Feature | Deterministic? |
|-------|---------|----------------|
| #78 | Negation marking (`negita` flag) | **Yes** - implemented |
| #87 | Sentence type (question/command/statement) | **Yes** |
| #76 | Correlative decomposition (ki-/ti-/ĉi-/neni-/i-) | **Yes** |
| #80 | Compound word decomposition | **Mostly** - some learning for rare cases |
| #84 | Participle tense/voice structure | **Yes** |
| #88 | Elision handling (l', hund') | **Yes** |

### 0.3 Discourse Features (Medium Priority)

| Issue | Feature | Approach |
|-------|---------|----------|
| #93 | Discourse connectives (tamen, do, ĉar) | **100% Deterministic** |
| #92 | Coreference resolution | **60% Deterministic** (gender/number/proximity) + **40% Learned** (ranking) |
| #94 | Deixis marking (mi, nun, ĉi tie) | **100% Deterministic** |

### 0.4 Corpus Reparsing (#86)

**Blocked by**: All Phase 0 issues above

After all parser changes:
1. Reparse entire corpus with enhanced parser
2. Generate new ASTs with all features
3. Validate parse rates remain high (>90%)
4. Archive old corpus, use new for training

---

## Phase 1: Root Embedding Training

**Goal**: Learn semantic meaning of content word roots only.

### 1.1 Function Word Exclusion (CRITICAL)

**The Function Word Exclusion Principle**: Function words are handled by the deterministic AST layer, not learned embeddings.

```python
FUNCTION_WORDS = {
    # Conjunctions
    'kaj', 'aŭ', 'sed', 'nek', 'do', 'tamen', 'ĉar', 'ke', 'se',
    # Prepositions
    'al', 'de', 'en', 'el', 'kun', 'per', 'por', 'pri', 'sen', 'sur', 'sub', 'ĉe', 'tra', 'ĉirkaŭ',
    # Pronouns (handled by coreference)
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'si', 'oni',
    # Correlatives (handled by decomposition)
    'kiu', 'kio', 'kia', 'kie', 'kiel', 'kiam', 'kiom', 'kial',
    'tiu', 'tio', 'tia', 'tie', 'tiel', 'tiam', 'tiom', 'tial',
    'ĉiu', 'ĉio', 'ĉia', 'ĉie', 'ĉiel', 'ĉiam', 'ĉiom', 'ĉial',
    'neniu', 'nenio', 'nenia', 'nenie', 'neniel', 'neniam', 'neniom', 'nenial',
    'iu', 'io', 'ia', 'ie', 'iel', 'iam', 'iom', 'ial',
    # Copula/common verbs
    'est', 'far', 'hav', 'pov', 'dev', 'vol', 'deb',
    # Particles
    'la', 'ne', 'tre', 'nur', 'ankaŭ', 'eĉ', 'ja', 'jen', 'jes', 'plej', 'pli', 'tro',

    # NOTE: Numbers (unu, du, tri, etc.) are NOT excluded - they carry semantic content
}
```

**Why This Matters**: Including function words caused embedding collapse where all content words became similar (0.99+ cosine similarity).

### 1.2 Training Data Sources

| Source | Weight | Pairs Type |
|--------|--------|------------|
| Ekzercaro (Fundamento) | 10.0 | Co-occurrence in Zamenhof's examples |
| ReVo definitions | 5.0 | Definition overlap similarity |
| Corpus co-occurrence | 1.0 | Same-sentence co-occurrence |

### 1.3 Semantic Cluster Negatives

Explicitly push apart unrelated semantic categories:

```python
SEMANTIC_CLUSTERS = {
    'family': ['patr', 'matr', 'fil', 'frat', 'edz', 'av', 'nev', 'onkl', 'nep'],
    'animals': ['hund', 'kat', 'bird', 'fiŝ', 'ĉeval', 'bov', 'ŝaf', 'kok', 'leon'],
    'body': ['kap', 'man', 'brak', 'okul', 'buŝ', 'nas', 'orel', 'kor', 'pied', 'fingr'],
    'time': ['tag', 'nokt', 'hor', 'jar', 'monat', 'semajn', 'minut', 'sekund'],
    'places': ['dom', 'urb', 'land', 'lok', 'ĉambr', 'strat', 'vilaĝ', 'mont', 'mar'],
    'food': ['pan', 'lakt', 'viand', 'frukt', 'legom', 'suk', 'vin', 'kaĉ'],
    'nature': ['arb', 'flor', 'herb', 'sun', 'lun', 'stel', 'nub', 'pluv', 'vent'],
}

# Cross-cluster pairs are negative with weight=3.0
```

### 1.4 Graded Similarity Targets

Not binary (similar/not similar) but graded:

```python
# Based on co-occurrence frequency
target = min(0.5 + 0.1 * cooccurrence_count, 0.9)

# Based on definition overlap
target = jaccard_similarity(def_roots_a, def_roots_b)
```

### 1.5 Monitoring Metrics (Prevent Collapse)

| Metric | Target | Alarm |
|--------|--------|-------|
| Mean embedding norm | 0.05-0.15 | >0.20 = collapse |
| Cluster separation | >0.0 | <0.0 = collapse |
| Negative pair similarity | ~0.0 | >0.5 = collapse |
| patr↔tabl similarity | <0.1 | >0.5 = collapse |

---

## Phase 2: Affix Embedding Training

**Goal**: Learn semantic transformation vectors for affixes.

### 2.1 Affix Types

| Type | Examples | Training Approach |
|------|----------|-------------------|
| Semantic transformers | mal-, -et-, -eg-, -ej-, -ul-, -il- | Learn transformation vectors |
| Grammatical markers | -o, -a, -e, -as, -is, -n, -j | **Frozen** - one-hot features |

### 2.2 Training by Semantic Function (#79)

Group affixes by what they do, not by shared roots:

```python
AFFIX_SEMANTIC_GROUPS = {
    'opposite': ['mal-'],
    'degree': ['-et-', '-eg-'],
    'person': ['-ul-', '-ist-', '-an-'],
    'place': ['-ej-', '-uj-'],
    'tool': ['-il-'],
    'abstract': ['-ec-', '-aĵ-'],
    'action': ['-ad-', 'ek-', 're-'],
    'possibility': ['-ebl-', '-ind-', '-end-'],
    'causative': ['-ig-', '-iĝ-'],
}
```

### 2.3 Antonym Handling (#82)

Special case for `mal-`:
- `mal-X` should be in opposite direction from `X`
- Train with explicit antonym pairs
- Verify: `cos(bona, malbona) < -0.5`

---

## Phase 3: Corpus Integration

**Goal**: Refine embeddings with usage patterns (low weight).

### 3.1 Prerequisites
- Phase 0 complete (AST enhancements)
- Phase 1 complete (root embeddings)
- Corpus reparsed with new AST (#86)

### 3.2 Function Word Filter (CRITICAL)

```python
def extract_roots_from_ast(ast):
    """Extract only content word roots."""
    roots = []
    for word in ast.words:
        if word.root not in FUNCTION_WORDS:
            roots.append(word.root)
    return roots
```

### 3.3 Negation-Aware Pairs

Sentences with negation should NOT be similar to non-negated versions:

```python
# "Mi amas vin" and "Mi ne amas vin" should be DISSIMILAR
# Use AST negita flag to detect and weight accordingly

if ast1.negita != ast2.negita:
    # Penalize similarity
    target = 0.0  # or negative
```

### 3.4 Source Weighting

```python
source_weights = {
    'fundamento': 10.0,
    'revo': 5.0,
    'wikipedia': 1.0,
    'gutenberg': 1.5,
}
```

---

## Phase 4: Sentence Encoding

**Goal**: Build sentence embeddings from enhanced ASTs.

### 4.1 Content Words Only

```python
def extract_content_words(ast):
    """Extract words for embedding, excluding function words."""
    words = []
    for word in ast.words:
        if word.root not in FUNCTION_WORDS:
            words.append(word)
    return words
```

### 4.2 Grammatical Feature Transformations (NEW - #101)

**Key Insight**: Grammatical features have semantic content that should be LEARNED, not frozen.

```python
class GrammaticalTransformers(nn.Module):
    """Learn semantic effect of grammatical features."""

    def __init__(self, dim):
        # Negation (#78) - context-dependent, not simple flip
        self.negation = nn.Linear(dim, dim)

        # Tense (#102) - temporal semantics
        self.tense = nn.ModuleDict({
            'pasinteco': nn.Linear(dim, dim),
            'prezenco': nn.Identity(),  # Baseline
            'futuro': nn.Linear(dim, dim),
        })

        # Mood (#103) - modality
        self.mood = nn.ModuleDict({
            'indikativo': nn.Identity(),  # Baseline
            'kondicionalo': nn.Linear(dim, dim),  # Hypothetical
            'imperativo': nn.Linear(dim, dim),    # Command
        })

        # Sentence type (#104)
        self.sentence_type = nn.ModuleDict({
            'aserto': nn.Identity(),    # Statement baseline
            'demando': nn.Linear(dim, dim),  # Question
            'ordono': nn.Linear(dim, dim),   # Command
        })

    def forward(self, embedding, ast):
        # Apply transformations based on AST features
        if ast.get('negita'):
            embedding = self.negation(embedding)

        if tense := ast.get('verbo', {}).get('tempo'):
            if tense in self.tense:
                embedding = self.tense[tense](embedding)

        if mood := ast.get('verbo', {}).get('modo'):
            if mood in self.mood:
                embedding = self.mood[mood](embedding)

        if sent_type := ast.get('fraztipo'):
            if sent_type in self.sentence_type:
                embedding = self.sentence_type[sent_type](embedding)

        return embedding
```

### 4.3 Training Pairs for Grammatical Semantics

```python
# Negation pairs - context-dependent similarity
("Mi amas vin", "Mi ne amas vin", similarity=-0.8)
("Estas bone", "Ne estas malbone", similarity=0.6)  # Litotes

# Tense pairs - temporal ordering
("Li venas", "Li venis", similarity=0.7)
("Li venis", "Li venos", similarity=0.4)

# Mood pairs - factual vs hypothetical
("Li venas", "Li venus", similarity=0.3)  # Very different!

# Sentence type pairs
("Li venas", "Ĉu li venas?", similarity=0.5)
("Venu!", "Li venas", similarity=0.3)
```

### 4.4 Role-Aware Attention

Use AST roles (subjekto, verbo, objekto) for attention weights:

```python
ROLE_WEIGHTS = {
    'subjekto': 1.0,
    'verbo': 1.5,      # Verbs are semantically central
    'objekto': 1.0,
    'aliaj': 0.5,      # Modifiers less important
}
```

### 4.5 Correlative Prefix Embeddings (#76)

Decompose correlatives and learn prefix embeddings:

```python
CORRELATIVE_PREFIXES = {
    'ki': 'question',      # Who, what, where (seeks info)
    'ti': 'demonstrative', # That, there, then (points)
    'ĉi': 'universal',     # Every, all, always (quantifies all)
    'neni': 'negative',    # No one, nothing, never (empty set)
    'i': 'indefinite',     # Some, somewhere, sometime
}

# Learn embeddings for prefixes
prefix_embeddings = nn.Embedding(5, prefix_dim)

# Correlative = prefix + suffix
correlative_emb = prefix_embeddings[prefix] + suffix_embeddings[suffix]
```

### 4.6 Accusative Direction (#105)

Learn motion vs location semantics:

```python
class DirectionTransformer(nn.Module):
    def forward(self, prep_emb, noun_emb, is_direction):
        combined = prep_emb + noun_emb
        if is_direction:  # Accusative with preposition
            return self.motion_transform(combined)
        return combined  # Static location
```

---

## Phase 5: Discourse Integration (NEW)

**Goal**: Handle multi-sentence understanding.

### 5.1 Coreference Chain Training (#92)

After deterministic coreference resolution:

```python
# Sentences in same coreference chain should be similar
# "Zamenhof kreis Esperanton. Li naskiĝis en 1859."
# → li=Zamenhof, so sentences are about same entity

if share_coreference_chain(ast1, ast2):
    target_similarity = 0.7  # Related but not identical
```

### 5.2 Discourse Relation Features (#93)

Use connective-marked relations:

```python
DISCOURSE_RELATIONS = {
    'contrast': ['tamen', 'sed', 'malgraŭ'],    # A but B
    'cause': ['ĉar', 'pro tio ke'],              # A because B
    'result': ['do', 'tial', 'sekve'],           # A therefore B
    'addition': ['kaj', 'ankaŭ', 'krome'],       # A and B
}

# Sentences with cause/result relation should embed close
# Sentences with contrast relation should embed differently
```

---

## Phase 6: Reasoning Core (Future)

**Goal**: Learn AST-to-AST transformations for Q&A.

### 6.1 Prerequisites
- Phases 0-5 complete
- Evaluation metrics passing
- Retrieval quality validated

### 6.2 Architecture Options

1. **Graph Transformer**: ASTs as graphs, learn transformations
2. **Seq2Seq on Linearized ASTs**: Smaller vocabulary, explicit structure
3. **Neuro-Symbolic Hybrid**: Neural selection + symbolic composition

### 6.3 Training Data
- AST pairs from parallel sentences
- Synthetic reasoning examples
- Human-annotated reasoning chains

---

## Evaluation Criteria

### Phase 1 Success (Root Embeddings)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Root similarity accuracy | >85% | Test pairs (patr↔matr similar, patr↔tabl dissimilar) |
| Cluster separation | >0.03 | Mean intra-cluster - mean inter-cluster |
| Negative similarity | <0.1 | Mean similarity of random pairs |
| Antonym direction | <-0.5 | cos(X, malX) for known antonyms |

### Phase 4 Success (Sentence Encoding + Grammatical Semantics)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Retrieval MRR@10 | >0.6 | Held-out Q&A pairs |
| **Negation discrimination** | Context-appropriate | Graded test pairs |
| **Tense ordering** | Preserved | Past < Present < Future similarity |
| **Mood discrimination** | >80% | Indicative vs conditional accuracy |
| **Sentence type** | >95% | Classification + embedding effect |
| **Correlative quantification** | Logical consistency | Reasoning test set |
| **Direction vs location** | >80% | Motion phrase discrimination |

### Phase 5 Success (Discourse)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Coreference chain coherence | >0.7 | Same-chain similarity |
| Cross-document discrimination | <0.3 | Different-doc similarity |

---

## Implementation Order (Staged Pipeline)

The pipeline is trained in stages, with each stage frozen before the next begins.
This aligns with the AST Enrichment Pipeline architecture (#106).

```
PHASE 0: PARSER COMPLETION (BLOCKER FOR ALL)
├── Fix parser bugs (#89, #90, #91, #85)
├── Add AST features (#78✓, #87, #76, #80, #84, #88)
├── Add grammatical features for semantic learning:
│   ├── tempo (tense) - already present
│   ├── modo (mood) - fix inconsistency (#91)
│   ├── fraztipo (sentence type) - add (#87)
│   ├── korelativo_prefikso (correlative prefix) - add (#76)
│   └── direkto (direction) - add (#105)
├── Add discourse features (#93, #92-rules, #94)
├── Reparse corpus (#86)
└── Build Thought Visualizer Demo (#107) - develop alongside

═══════════════════════════════════════════════════════════════
STAGE 1: SEMANTIC MODEL (~333K params)
═══════════════════════════════════════════════════════════════

PHASE 1: ROOT EMBEDDINGS
├── Apply FUNCTION_WORDS filter
├── Add SEMANTIC_CLUSTERS negatives
├── Use graded similarity targets
├── Monitor for collapse
└── Validate with test pairs

PHASE 2: AFFIX EMBEDDINGS
├── Group by semantic function
├── Train transformation vectors
├── Handle mal- antonyms specially
└── Freeze grammatical affixes

PHASE 3: CORPUS INTEGRATION
├── Requires: Phase 0 + Phase 1 complete
├── Apply function word filter
├── Use negation-aware pairs
├── Weight by source authority
└── Monitor for collapse

→ FREEZE Stage 1 models
→ Verify with Thought Visualizer: AST now has 'semantic' slot

═══════════════════════════════════════════════════════════════
STAGE 2: GRAMMATICAL MODEL (~52K params)
═══════════════════════════════════════════════════════════════

PHASE 4: GRAMMATICAL FEATURE TRANSFORMS
├── Requires: Stage 1 frozen
├── Read embeddings from AST 'semantic' slot
├── Train dedicated transforms for each feature (#101):
│   ├── Negation transformation (learned, not flip) (#78) - 4K params
│   ├── Tense transformation (#102) - 8K params
│   ├── Mood transformation (#103) - 8K params
│   ├── Sentence type transformation (#104) - 8K params
│   ├── Direction transformation (#105) - 4K params
│   ├── Comparison transformation (#108) - 4K params
│   ├── Aspect transformation (#109) - 4K params
│   ├── Focus particle transformation (#110) - 8K params
│   ├── Evidentiality transformation (#111) - 4K params
│   └── Possessive transformation (#112) - 4K params
│   Total Stage 2: ~52K params (was ~32K)
├── Correlative prefix embeddings (#76) - included in Stage 1
├── Create minimal pairs for each grammatical feature
├── Joint training with separate loss terms per feature
├── Write to AST 'grammatical' slot
└── Evaluation metrics for each feature

→ FREEZE Stage 2 model
→ Verify with Thought Visualizer: AST now has 'grammatical' slot

═══════════════════════════════════════════════════════════════
STAGE 3: DISCOURSE MODEL (~100K params)
═══════════════════════════════════════════════════════════════

PHASE 5: DISCOURSE INTEGRATION
├── Requires: Stage 2 frozen
├── Read from AST 'semantic' + 'grammatical' slots
├── Coreference chain training (#92)
├── Discourse relation features (#93)
├── Multi-sentence coherence
├── Write to AST 'discourse' slot
└── Paragraph-level retrieval validation

→ FREEZE Stage 3 model
→ Verify with Thought Visualizer: AST now has 'discourse' slot

═══════════════════════════════════════════════════════════════
STAGE 4: REASONING CORE (20-100M params) - FUTURE
═══════════════════════════════════════════════════════════════

PHASE 6: REASONING
├── Requires: Stages 1-3 frozen
├── Read from all AST slots
├── AST-to-AST architecture
├── Q&A training data
├── Write to AST 'reasoning' slot
└── Evaluation benchmark

→ Verify with Thought Visualizer: Complete pipeline traceable
```

### Staged Training Benefits

1. **Clear Checkpoints**: Each stage has definable success criteria
2. **No Catastrophic Forgetting**: Frozen stages don't degrade
3. **Interpretable Progress**: Thought Visualizer shows what each stage adds
4. **Efficient Debugging**: If Stage 3 fails, Stages 1-2 are known-good
5. **Parameter Efficiency**: Total ~485K params before reasoning core

---

## Mistakes to Avoid (Lessons Learned)

### 1. Function Word Collapse

**Symptom**: All embeddings become similar (>0.99 cosine)
**Cause**: Function words appear in every sentence
**Prevention**: FUNCTION_WORDS filter in ALL training phases

### 2. Binary Similarity Targets

**Symptom**: Poor discrimination between somewhat-similar and dissimilar
**Cause**: Binary 0/1 targets
**Prevention**: Graded targets based on co-occurrence/overlap

### 3. Missing Negative Sampling

**Symptom**: All embeddings drift toward center
**Cause**: No explicit push-apart signal
**Prevention**: Semantic cluster negatives with high weight

### 4. Training Before Parser Complete

**Symptom**: Need to retrain after parser changes
**Cause**: AST changes invalidate training data
**Prevention**: Complete Phase 0 before any training

### 5. Ignoring Negation

**Symptom**: "X" and "ne X" have identical embeddings
**Cause**: `ne` excluded as function word
**Prevention**: Negita flag in AST, transformation in encoder

### 6. Numbers as Function Words (#83)

**Symptom**: Quantity lost in embeddings
**Cause**: Numbers incorrectly excluded
**Prevention**: Numbers are content words, keep in training

### 7. Compound Word Literalism (#80)

**Symptom**: vaporŝipo = vapor + ŝipo exactly
**Cause**: Naive additive composition
**Prevention**: Compounds get emergent embeddings, composition is initialization only

### 8. Freezing Grammatical Features (#101) - NEW

**Symptom**: Tense, mood, sentence type have no effect on embeddings
**Cause**: Treated as frozen one-hot features
**Wrong assumption**: "Detectable by rules" = "no semantic content"
**Prevention**: Detect deterministically, but LEARN semantic effect

### 9. Simple Negation Flip (#78) - UPDATE

**Symptom**: "ne malbone" treated as opposite of "malbone"
**Cause**: Deterministic sign flip ignores context
**Prevention**: Learn negation transformation, use graded training pairs

---

## Related Documentation

- [[Function-Word-Exclusion-Principle]] - Why function words are excluded
- [[Root-Embedding-Training-Lessons-Learned]] - Detailed lessons from v2
- [[Fundamento-Centered-Training]] - Authority hierarchy
- [[Grammatical-Features-Semantic-Audit]] - Which grammar features have semantics
- Wiki meta-issues: #95 (Semantics), #96 (Pragmatics), #97 (Discourse), #98 (Morphology), #99 (Syntax)

---

## GitHub Issues

### Architecture
#106 (AST Enrichment Pipeline) - Models as thought accumulators
#107 (Thought Visualizer Demo) - Show AST state after each stage

### Parser Prerequisites (Phase 0)
#76, #80, #84, #85, #87, #88, #89, #90, #91

### Training Infrastructure
#78 (Negation) - detection in parser, needs learned transformation
#79 (Affix semantics)
#81 (Weak training signal)
#82 (Antonyms)
#83 (Numbers) - fixed

### Grammatical Feature Semantics (#101 meta-issue)
#102 (Tense semantics)
#103 (Mood semantics)
#104 (Sentence type semantics)
#105 (Accusative direction semantics)
#108 (Comparison semantics - pli/plej/ol)
#109 (Aspect semantics - ek-, -ad-)
#110 (Focus particle semantics - nur, eĉ, ankaŭ, ja)
#111 (Evidentiality markers)
#112 (Possessive semantics)

### Discourse Layer
#92 (Coreference)
#93 (Discourse connectives)
#94 (Deixis)

### Blocked
#86 (Reparse corpus) - blocked by all Phase 0 issues

---

*Last updated: December 2024*
