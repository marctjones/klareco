# Klareco Semantic Learning Strategy

**Version**: 2.0 (December 2024)
**Status**: Revised strategy incorporating lessons learned

---

## Executive Summary

**Key Insight**: Esperanto's regular grammar fundamentally changes what we need to learn.

| Component | Traditional LLMs | Klareco |
|-----------|-----------------|---------|
| Grammar | Learned (70M+ params) | **Deterministic** (0 params) |
| Morphology | Learned | **Deterministic** (parser) |
| Semantic Roles | Learned | **FREE** (accusative markers) |
| Tokenization | BPE (50K tokens) | **Morphological** (~3K roots + 100 affixes) |
| Training Signal | Next-token / MLM | **Contrastive on AST pairs** |

**What we have**: 4.2M sentences with ASTs, 95% parse quality, semantic roles extracted
**What we need to learn**: Root semantics (~5K roots) and affix transformations (~50 affixes)

---

## Core Principles

### 1. Grammar is Deterministic - Stop Trying to Learn It

Our MLM experiment proved this: loss *increased* (6.96 → 7.28) because:
- Grammar endings are 100% predictable from rules
- Neural networks fight against deterministic patterns
- Capacity wasted on what's already known

**Rule**: Never use training objectives that reward predicting grammar.

### 2. AST Is Already Annotated - Exploit It

Every sentence in our corpus has:
```python
{
    'subjekto': {'radiko': 'hund', 'kazo': 'nominativo'},  # AGENT
    'verbo': {'radiko': 'kur', 'tempo': 'prezenco'},       # ACTION
    'objekto': {'radiko': 'pilk', 'kazo': 'akuzativo'},    # PATIENT
}
```

This is **free semantic role labeling** from grammar markers. Use it.

### 3. Zamenhof's Design Was Optimal

~900 essential roots + ~50 affixes = complete language coverage
- Compression-optimal (MDL principle)
- Grounding-friendly (concrete roots → derived abstractions)
- Compositional (meaning builds systematically)

---

## Revised Architecture

### What's Deterministic (0 learned parameters)

```
Text → Parser (16 rules) → AST with roles
              ↑
              │ Morpheme decomposition
              │ Case/tense/number extraction
              │ Semantic role assignment
              │ (ALL FREE)
```

**Implementation**: Already done in `klareco/parser.py`

### What's Learned: Semantic Embeddings Only

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNED COMPONENTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │  Root Embeddings │    │  Affix Embeddings│               │
│  │  ~5K roots × 64d │    │  ~50 affixes × 64d              │
│  │  = 320K params   │    │  = 3.2K params   │               │
│  └────────┬─────────┘    └────────┬─────────┘               │
│           │                        │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                      │
│           ┌──────────────────────┐                          │
│           │  Composition Function │                          │
│           │  word = f(root, affixes, features)              │
│           └──────────┬───────────┘                          │
│                      │                                       │
│                      ▼                                       │
│           ┌──────────────────────┐                          │
│           │  AST Encoder (GNN)   │                          │
│           │  ~1-5M params        │                          │
│           └──────────┬───────────┘                          │
│                      │                                       │
│                      ▼                                       │
│           ┌──────────────────────┐                          │
│           │  Reasoning Core      │                          │
│           │  20-100M params      │                          │
│           └──────────────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

What's FROZEN (deterministic features):
- Grammatical endings as one-hot: register_buffer('endings', ...)
- Case features: [nominative, accusative]
- Tense features: [present, past, future, conditional]
- Number features: [singular, plural]
```

---

## Training Strategy: Three Phases

### Phase 1: Root Semantic Grounding (Current Focus)

**Objective**: Learn what roots *mean*, not how they're used grammatically.

**Strategy**: Contrastive learning on AST pairs from document context

```python
# Positive pairs: sentences from same article
ast_a = parse("Aristotelo estis filozofo")
ast_b = parse("Li instruis en Ateno")
# similarity(embed(ast_a), embed(ast_b)) → HIGH

# Negative pairs: sentences from different articles
ast_c = parse("La hundo kuras rapide")
# similarity(embed(ast_a), embed(ast_c)) → LOW
```

**Why this works for Esperanto**:
- AST structure gives us roots directly (no BPE confusion)
- Article boundaries provide semantic grouping signal
- Grammar is stripped away - learning pure semantics

**Data source**: Wikipedia articles provide natural semantic boundaries
- 4.2M sentences with article_id metadata
- Document-level coherence as training signal

**Loss function**:
```python
def contrastive_loss(anchor, positive, negatives, margin=0.2):
    pos_sim = cosine_similarity(anchor, positive)
    neg_sims = [cosine_similarity(anchor, neg) for neg in negatives]
    loss = max(0, margin - pos_sim + max(neg_sims))
    return loss
```

### Phase 2: Affix Semantic Transformations

**Objective**: Learn how affixes modify root meanings.

**Strategy**: Triplet learning on morphological pairs

```python
# Anchor: "bona" (good, adj)
# Positive: "malbona" (bad, adj) - with mal- transformation
# Negative: "rapida" (fast, adj) - unrelated

# Learn that embed("malbona") ≈ transform(embed("bon"), embed("mal-"))
```

**Esperanto affixes to learn** (semantic, not grammatical):

| Affix | Meaning | Example |
|-------|---------|---------|
| mal- | opposite | bona → malbona |
| -et | diminutive | domo → dometo |
| -eg | augmentative | domo → domego |
| -ul | person | riĉa → riĉulo |
| -ej | place | lerni → lernejo |
| -il | tool | kombilo |
| re- | again | reveni |
| ek- | begin | ekiri |

**Grammatical affixes are FROZEN** (not learned):
- -o, -a, -e, -i (POS markers)
- -as, -is, -os, -us, -u (tense/mood)
- -n (accusative)
- -j (plural)

### Phase 3: Prototype Grounding (Future)

**Objective**: Ground abstract concepts via Zamenhof's derivational system.

**Strategy**: Hierarchical embedding initialization

```
Level 0: Functional words (frozen, one-hot-like)
  - la (the), kaj (and), aŭ (or), sed (but)

Level 1: Concrete roots (learned from co-occurrence)
  - hundo, kato, domo, akvo, varma, malvarma

Level 2: Abstract derivations (computed from Level 1 + affixes)
  - varmeco = varm- (grounded) + -ec- (abstractness) + -o
  - malvarmeco = mal- + varm- + -ec- + -o
```

**Zamenhof's ~900 essential roots** become grounding primitives.
Everything else derives compositionally.

---

## What NOT to Do (Lessons Learned)

### Avoid These Training Strategies

| Strategy | Why It Fails for Esperanto |
|----------|---------------------------|
| **MLM (BERT)** | Predicting grammar is trivial; loss increases |
| **Causal LM (GPT)** | Word order is flexible; learns redundant patterns |
| **Distillation** | Imports biases from English-centric models |
| **Denoising** | Reconstruction is deterministic via grammar |
| **Rotation prediction** | Word order is free in Esperanto |

### Signs Your Training Is Wrong

1. **Loss increasing** → Model fighting deterministic patterns
2. **Learning grammar rules** → Wasted capacity
3. **Memorizing word forms** → Not using compositionality
4. **Ignoring AST structure** → Missing free information

---

## Implementation Plan

### Immediate (Phase 1.1): Root Embeddings

```python
# In compositional.py - separate learned from frozen

class RootSemanticEmbedding(nn.Module):
    def __init__(self, num_roots=5000, embed_dim=64):
        super().__init__()
        # LEARNED: Root meanings
        self.root_embed = nn.Embedding(num_roots, embed_dim)

        # FROZEN: Grammatical features
        self.register_buffer('ending_features',
            torch.eye(17))  # 17 endings
        self.register_buffer('case_features',
            torch.tensor([[1, 0], [0, 1]]))  # nom/acc
        self.register_buffer('tense_features',
            torch.eye(5))  # as/is/os/us/u
```

### Training Script Changes

```python
# DON'T: Predict next token or masked token
# DO: Contrastive learning on document context

class ContrastiveASTTrainer:
    def training_step(self, batch):
        # Parse sentences (deterministic, no gradients)
        with torch.no_grad():
            asts = [parser.parse(s) for s in batch['text']]

        # Extract roots from AST
        root_ids = extract_roots(asts)  # Deterministic

        # Embed roots (gradients flow here)
        embeddings = self.root_embed(root_ids)

        # Contrastive loss on document pairs
        anchor = embeddings[batch['anchor_idx']]
        positive = embeddings[batch['positive_idx']]
        negatives = embeddings[batch['negative_idx']]

        return contrastive_loss(anchor, positive, negatives)
```

### Metrics to Track

1. **Root similarity accuracy**: Do semantically similar roots cluster?
2. **Affix transformation accuracy**: Does mal- consistently negate?
3. **Retrieval quality**: Do sentence embeddings enable good RAG?
4. **Zero grammar learning**: Verify no grammatical patterns in embeddings

---

## Parameter Budget

| Component | Parameters | Notes |
|-----------|------------|-------|
| Root embeddings | 320K | 5K roots × 64d |
| Affix embeddings | 3.2K | 50 affixes × 64d |
| Grammatical features | 0 | Frozen buffers |
| AST encoder (GNN) | 1-5M | 4-6 layers, 128-256 hidden |
| Reasoning core | 20-100M | AST-to-AST transformer |
| **Total** | **21-105M** | vs. 7B+ for LLMs |

**Thesis test**: If this architecture achieves 80%+ accuracy on Q&A while being fully explainable, the thesis is proven.

---

## Success Criteria

1. **Embeddings capture semantics, not grammar**
   - Similar roots cluster (hund ~ kat, bon ~ bel)
   - Grammar-different forms identical (hundo = hundon = hundoj)

2. **Affixes transform consistently**
   - mal-X is always opposite direction from X
   - -ej always means "place of"

3. **Retrieval works on AST similarity**
   - Question AST retrieves relevant context ASTs
   - Not fooled by surface form differences

4. **End-to-end Q&A**
   - Answer 50+ questions using deterministic + retrieval
   - Add reasoning core, measure improvement
   - Target: 80%+ accuracy with <100M learned params

---

## Related Documentation

- [[Why-AST-Changes-Everything]] - The paradigm shift
- [[Embedding-Training-Strategies]] - Strategy evaluation
- [[Neuro-Symbolic-Architecture]] - Architecture details
- [[Deterministic-vs-Learned-Boundaries]] - Design philosophy

---

*Last updated: December 2024*
