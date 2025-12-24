# Compositional Embeddings for Esperanto

**Core Insight**: Esperanto morphology is COMPOSITIONAL and REGULAR, so embeddings should be too!

Traditional word embeddings learn "dog" and "dogs" as separate vectors with no explicit relationship. This wastes dimensions and data.

Esperanto embeddings should be: **root + prefix + suffix + ending = word meaning**

---

## Esperanto Morpheme Structure

Every Esperanto word follows this pattern:
```
[PREFIX*] + ROOT + [SUFFIX*] + ENDING + [j] + [n]

Examples:
  hundo     =              hund + o
  hundoj    =              hund + o + j
  malhundo  = mal +        hund + o          (anti-dog?)
  hundejo   =              hund + ej + o     (dog place/kennel)
  rehundigi = re  +        hund + ig + i     (to re-dogify)
```

---

## Morpheme Inventory (Deterministic!)

From the parser, we have:

### Affixes (Fixed, Small Set)
- **7 prefixes**: mal, re, ge, eks, ek, pra, for
- **31 suffixes**: ul, ej, in, et, ad, ig, iĝ, ism, ist, ar, aĉ, aĵ, ebl, end, ec, eg, em, er, estr, id, il, ind, ing, uj, um, ant, int, ont, at, it, ot
- **11 endings**: as, is, os, us, u, i, o, a, e, j, n

**Total: 49 morphemes with KNOWN, FIXED meanings**

### Grammatical Features (Deterministic!)
- **Case**: 2 values (nominative, accusative)
- **Number**: 2 values (singular, plural)
- **Tense**: 4 values (present, past, future, conditional)
- **Mood**: 2 values (infinitive, imperative)
- **POS**: 4 values (noun, adjective, adverb, verb)

**Total: 14 categorical values**

### Roots (Learned!)
- **~10,000-15,000 roots** in full Esperanto vocabulary
- But our corpus has **~3,000-5,000 unique roots**
- These are the ONLY things we need to learn!

---

## Proposed Embedding Architecture

### Compositional Design

Instead of one big embedding table:
```python
# TRADITIONAL (wasteful!)
embeddings = nn.Embedding(vocab_size=50000, embed_dim=256)
# Learns "hundo", "hundoj", "malhundoj" as separate, unrelated vectors
```

Use separate, composable embeddings:
```python
# COMPOSITIONAL (efficient!)
root_embeddings    = nn.Embedding(5000, 64)   # Semantic content (learned)
prefix_embeddings  = nn.Embedding(7, 8)       # Fixed meanings (learned or programmatic)
suffix_embeddings  = nn.Embedding(31, 8)      # Fixed meanings (learned or programmatic)
ending_features    = Fixed(11, 8)             # Grammar only (programmatic!)
grammar_features   = Fixed(14, 8)             # POS/case/tense (programmatic!)

# Compose them!
def embed_word(word_ast):
    root_vec    = root_embeddings[word_ast['radiko']]      # 64 dims
    prefix_vec  = sum(prefix_embeddings[p] for p in prefixes)  # 8 dims
    suffix_vec  = sum(suffix_embeddings[s] for s in suffixes)  # 8 dims
    ending_vec  = encode_ending(word_ast['ending'])        # 8 dims (programmatic!)
    grammar_vec = encode_grammar(word_ast)                 # 8 dims (programmatic!)

    # Concatenate or add
    return torch.cat([root_vec, prefix_vec, suffix_vec, ending_vec, grammar_vec])
    # Total: 64 + 8 + 8 + 8 + 8 = 96 dimensions
```

---

## Dimension Allocation Strategy

### Option 1: Concatenation (Recommended)

**Total Dimensions: 96**

| Component | Dims | Type | Reasoning |
|-----------|------|------|-----------|
| **Root** | 64 | Learned | Semantic meaning (dog, cat, see, run) |
| **Prefix** | 8 | Learned | Modifiers (mal=opposite, re=again, etc.) |
| **Suffix** | 8 | Learned | Derivations (ej=place, ul=person, etc.) |
| **Ending** | 8 | **Programmatic** | POS marker (o/a/e/i/as/is/os) |
| **Grammar** | 8 | **Programmatic** | Case/number/tense |

**Why this split?**
- **Root (64 dims)**: Needs richness for semantic similarity (dog ≈ cat ≠ idea)
- **Affixes (8 dims each)**: Simple transformations, less complexity needed
- **Grammar (16 dims total)**: Deterministic features, can be binary-encoded

**Programmatic Dimensions: 16 out of 96 (17%)** ✅

### Option 2: Addition (More Compact)

**Total Dimensions: 64**

All components share the same 64-dimensional space:
```python
embedding = root_vec + prefix_vec + suffix_vec + ending_vec + grammar_vec
```

**Tradeoff**:
- ✅ More compact (64 vs 96 dims)
- ❌ Components interfere with each other
- ❌ Harder to disentangle root meaning from grammar

**Recommendation**: Use concatenation for cleaner separation.

---

## Programmatic vs Learned Dimensions

### Fully Programmatic (No Learning): Endings + Grammar

**Endings** (8 dimensions, one-hot style):
```python
ENDING_VECTORS = {
    'o':  [1, 0, 0, 0, 0, 0, 0, 0],  # Noun
    'a':  [0, 1, 0, 0, 0, 0, 0, 0],  # Adjective
    'e':  [0, 0, 1, 0, 0, 0, 0, 0],  # Adverb
    'i':  [0, 0, 0, 1, 0, 0, 0, 0],  # Infinitive
    'as': [0, 0, 0, 0, 1, 0, 0, 0],  # Present tense
    'is': [0, 0, 0, 0, 0, 1, 0, 0],  # Past tense
    'os': [0, 0, 0, 0, 0, 0, 1, 0],  # Future tense
    'us': [0, 0, 0, 0, 0, 0, 0, 1],  # Conditional
}

def encode_ending(ending: str) -> torch.Tensor:
    """Deterministic encoding of grammatical ending."""
    return torch.tensor(ENDING_VECTORS.get(ending, [0]*8))
```

**Grammar** (8 dimensions, binary encoding):
```python
def encode_grammar(word_ast: dict) -> torch.Tensor:
    """Encode grammatical features as binary vector."""
    vec = [0.0] * 8

    # Case (1 bit)
    vec[0] = 1.0 if word_ast.get('kazo') == 'akuzativo' else 0.0

    # Number (1 bit)
    vec[1] = 1.0 if word_ast.get('nombro') == 'pluralo' else 0.0

    # Tense (2 bits, one-hot for 4 values)
    tempo = word_ast.get('tempo')
    if tempo == 'prezenco':
        vec[2], vec[3] = 0.0, 0.0
    elif tempo == 'pasinteco':
        vec[2], vec[3] = 0.0, 1.0
    elif tempo == 'futuro':
        vec[2], vec[3] = 1.0, 0.0
    elif tempo == 'kondiĉa':
        vec[2], vec[3] = 1.0, 1.0

    # Mood (1 bit, for infinitive/imperative)
    modo = word_ast.get('modo')
    vec[4] = 1.0 if modo == 'imperativo' else 0.0

    # Reserved for future features
    vec[5] = 0.0
    vec[6] = 0.0
    vec[7] = 0.0

    return torch.tensor(vec)
```

**Benefits**:
- ✅ Zero parameters for these dimensions
- ✅ Perfect consistency (same grammar = same encoding)
- ✅ Interpretable (can read meaning from vector)
- ✅ No training data needed

### Semi-Programmatic (Learned with Constraints): Affixes

**Affixes have fixed meanings, but learned representations:**

```python
# Initialize with semantic priors, then allow fine-tuning
prefix_embeddings = nn.Embedding(7, 8)

# Initialize "mal" with negative values (opposite)
prefix_embeddings.weight.data[PREFIX_TO_ID['mal']] = torch.tensor([-1, -1, 0, 0, 0, 0, 0, 0])

# Initialize "re" with repetition/return values
prefix_embeddings.weight.data[PREFIX_TO_ID['re']] = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0])

# Then train, but with small learning rate (frozen or slow)
```

**Constraint Options**:
1. **Frozen**: Fix these embeddings, never train (fully deterministic)
2. **Slow learning**: Train with 10x smaller learning rate
3. **Regularized**: Add loss term to keep close to initialization

**Recommendation**: Start frozen, unfreeze later if needed.

### Fully Learned: Roots

**Roots** (64 dimensions, learned from data):
```python
root_embeddings = nn.Embedding(num_roots=5000, embed_dim=64)

# Initialize randomly or with pre-training
# Then train on corpus to learn semantic similarity
```

**This is the ONLY large learned component!**

---

## Dimension Count Analysis

### Vocabulary Size Comparison

**Traditional LLM**:
```
Vocab: 50,000 words (including "dog", "dogs", "dog's", "doggy", etc.)
Dimensions: 256-768
Parameters: 50,000 × 256 = 12.8M parameters
```

**Klareco Compositional**:
```
Roots: 5,000
Prefixes: 7
Suffixes: 31
Endings: 11 (programmatic!)
Grammar: 14 features (programmatic!)

Learned parameters:
  Roots:    5,000 × 64 = 320,000
  Prefixes:     7 × 8  =      56
  Suffixes:    31 × 8  =     248
  Total:                 320,304 parameters

Programmatic parameters: 0

Total: ~320K vs 12.8M = 40x smaller! 🎉
```

### Effective Vocabulary

With composition, we can represent:
```
5,000 roots × 7 prefixes × 31 suffixes × 11 endings × 2 cases × 2 numbers
= 5,000 × 7 × 31 × 11 × 2 × 2
= ~950,000 unique word forms!

From 320K parameters!
```

---

## Optimal Dimension Counts

### Minimum (Fast, Small)

| Component | Dims | Params | Notes |
|-----------|------|--------|-------|
| Root | 32 | 160K | Minimal semantic space |
| Prefix | 4 | 28 | Very compact |
| Suffix | 4 | 124 | Very compact |
| Ending | 4 | 0 | Programmatic |
| Grammar | 4 | 0 | Programmatic |
| **Total** | **48** | **160K** | Ultra-lightweight |

**Use case**: Fast retrieval, embedded devices, proof-of-concept

### Recommended (Balanced)

| Component | Dims | Params | Notes |
|-----------|------|--------|-------|
| Root | 64 | 320K | Good semantic richness |
| Prefix | 8 | 56 | Room for nuance |
| Suffix | 8 | 248 | Room for nuance |
| Ending | 8 | 0 | Programmatic |
| Grammar | 8 | 0 | Programmatic |
| **Total** | **96** | **320K** | Sweet spot |

**Use case**: Production system (THIS ONE!)

### Maximum (Rich Semantics)

| Component | Dims | Params | Notes |
|-----------|------|--------|-------|
| Root | 128 | 640K | Very rich semantics |
| Prefix | 16 | 112 | Rare, but possible |
| Suffix | 16 | 496 | Rare, but possible |
| Ending | 16 | 0 | Programmatic |
| Grammar | 16 | 0 | Programmatic |
| **Total** | **192** | **640K** | High quality |

**Use case**: When semantic similarity is critical

---

## Comparison with Current System

### Current Tree-LSTM Encoder

Looking at your retriever code:
```python
state_dict = checkpoint.get("model_state_dict", checkpoint)
vocab_size = state_dict["embed.weight"].shape[0]
embed_dim = state_dict["embed.weight"].shape[1]
```

You're likely using:
- Vocab size: ~10,000-20,000 (whole words)
- Embed dim: ~128-256 (mixed semantic + grammar)

### Proposed Compositional Encoder

```python
class CompositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_roots: int = 5000,
        root_dim: int = 64,
        affix_dim: int = 8,
        grammar_dim: int = 8,
        ending_dim: int = 8
    ):
        super().__init__()

        # Learned components
        self.root_embed = nn.Embedding(num_roots, root_dim)
        self.prefix_embed = nn.Embedding(7, affix_dim)
        self.suffix_embed = nn.Embedding(31, affix_dim)

        # Programmatic components (no nn.Embedding!)
        self.ending_dim = ending_dim
        self.grammar_dim = grammar_dim

        # Total output dimension
        self.output_dim = root_dim + affix_dim * 2 + ending_dim + grammar_dim

    def forward(self, word_ast: dict) -> torch.Tensor:
        """
        Compose word embedding from AST.

        Args:
            word_ast: Parsed word AST with radiko, prefikso, sufiksoj, etc.

        Returns:
            Composed embedding (96 dims)
        """
        # 1. Root (learned, 64 dims)
        root_id = self.root_to_id[word_ast['radiko']]
        root_vec = self.root_embed(root_id)

        # 2. Prefixes (learned, 8 dims)
        prefix = word_ast.get('prefikso')
        if prefix:
            prefix_id = self.prefix_to_id[prefix]
            prefix_vec = self.prefix_embed(prefix_id)
        else:
            prefix_vec = torch.zeros(self.affix_dim)

        # 3. Suffixes (learned, 8 dims)
        suffixes = word_ast.get('sufiksoj', [])
        if suffixes:
            suffix_vecs = [self.suffix_embed(self.suffix_to_id[s]) for s in suffixes]
            suffix_vec = torch.stack(suffix_vecs).sum(dim=0)  # Sum multiple suffixes
        else:
            suffix_vec = torch.zeros(self.affix_dim)

        # 4. Ending (programmatic, 8 dims)
        ending_vec = self._encode_ending(word_ast)

        # 5. Grammar (programmatic, 8 dims)
        grammar_vec = self._encode_grammar(word_ast)

        # Concatenate all components
        return torch.cat([root_vec, prefix_vec, suffix_vec, ending_vec, grammar_vec])

    def _encode_ending(self, word_ast: dict) -> torch.Tensor:
        """Programmatic encoding of grammatical ending."""
        # ... (see earlier example)

    def _encode_grammar(self, word_ast: dict) -> torch.Tensor:
        """Programmatic encoding of case/number/tense."""
        # ... (see earlier example)
```

**Benefits over whole-word embeddings**:
- ✅ 40x fewer parameters (320K vs 12.8M)
- ✅ Handles unseen word forms (compositional generalization)
- ✅ Explicit grammar encoding (16 dims deterministic)
- ✅ Shared root semantics across forms ("hund" same in "hundo", "hundoj", "hundejo")
- ✅ Interpretable (can inspect what each component contributes)

---

## Forced Dimensions Strategy

### Which Dimensions to Force (Make Programmatic)?

**Tier 1: Fully Deterministic (Force These!)** ✅

1. **Case** (1 dim, binary):
   - Dimension 65: 0 = nominative, 1 = accusative
   - Rationale: Purely grammatical, no semantic content

2. **Number** (1 dim, binary):
   - Dimension 66: 0 = singular, 1 = plural
   - Rationale: Purely grammatical, no semantic content

3. **Tense** (2 dims, one-hot):
   - Dimensions 67-68: 00=present, 01=past, 10=future, 11=conditional
   - Rationale: Temporal information is discrete and deterministic

4. **Part of Speech** (2 dims, one-hot):
   - Dimensions 69-70: 00=noun, 01=adjective, 10=adverb, 11=verb
   - Rationale: Category, not meaning

**Total forced: 6 dimensions** (out of 96 = 6.25%)

**Tier 2: Semi-Forced (Initialize, Then Allow Learning)** 🔄

5. **Prefixes** (8 dims):
   - Initialize "mal" with negative values
   - Initialize "re" with repetition pattern
   - Allow fine-tuning during training

6. **Suffixes** (8 dims):
   - Initialize "ej" with location pattern
   - Initialize "ul" with person pattern
   - Allow fine-tuning during training

**Tier 3: Fully Learned** 📚

7. **Roots** (64 dims):
   - Pure semantic content
   - Learn from data

### How to Force Dimensions

**Method 1: Fixed Masks (Recommended for Grammar)**

```python
class CompositionalEmbedding(nn.Module):
    def forward(self, word_ast):
        # ... get learned components ...

        # Create programmatic mask
        grammar_vec = torch.zeros(8)
        grammar_vec[0] = 1.0 if word_ast.get('kazo') == 'akuzativo' else 0.0
        grammar_vec[1] = 1.0 if word_ast.get('nombro') == 'pluralo' else 0.0
        # ... encode tense ...

        # Concatenate (forced dims are never trained!)
        return torch.cat([learned_part, grammar_vec])
```

**Method 2: Frozen Embeddings (For Affixes)**

```python
class CompositionalEmbedding(nn.Module):
    def __init__(self, ...):
        self.prefix_embed = nn.Embedding(7, 8)

        # Initialize with semantic priors
        self._initialize_prefixes()

        # Freeze (don't train)
        self.prefix_embed.weight.requires_grad = False

    def _initialize_prefixes(self):
        # "mal" = opposite (negative pattern)
        self.prefix_embed.weight.data[0] = torch.tensor([-1, -1, 0, 0, 0, 0, 0, 0])

        # "re" = again (repetition pattern)
        self.prefix_embed.weight.data[1] = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0])
        # ...
```

---

## Free vs Constrained Dimensions

### Recommended Split (96 total)

| Type | Dims | % | Component | Training |
|------|------|---|-----------|----------|
| **Fully Free** | 64 | 67% | Roots | Learned |
| **Semi-Free** | 16 | 17% | Affixes | Initialized, then learned |
| **Fully Constrained** | 16 | 17% | Grammar | Programmatic, never trained |

**Rationale**:
- Roots carry semantic content → need freedom to learn
- Affixes have known meanings → initialize, but allow refinement
- Grammar is deterministic → hard-code, save parameters

### Alternative: More Constraints (For Smaller Models)

| Type | Dims | % | Component | Training |
|------|------|---|-----------|----------|
| **Fully Free** | 32 | 44% | Roots | Learned |
| **Semi-Free** | 8 | 11% | Affixes | Frozen with priors |
| **Fully Constrained** | 32 | 44% | Grammar + POS | Programmatic |

**Use case**: Very small model, limited data

---

## Implementation Plan

### Phase 1: Build Compositional Embedder

```python
# klareco/embeddings/compositional.py (NEW)

class CompositionalEmbedding(nn.Module):
    """
    Compositional word embeddings for Esperanto.

    Decomposes words into: root + prefix + suffix + ending + grammar
    Only roots are fully learned; grammar is programmatic.
    """
    # ... (see full implementation above)
```

### Phase 2: Extract Root Vocabulary

```python
# scripts/extract_root_vocabulary.py (NEW)

def extract_roots_from_corpus(corpus_path: Path) -> dict:
    """Extract all unique roots from parsed corpus."""
    roots = set()

    for entry in read_corpus(corpus_path):
        ast = entry['ast']
        for word in extract_all_words(ast):
            if 'radiko' in word:
                roots.add(word['radiko'])

    return {root: i for i, root in enumerate(sorted(roots))}

# Build root vocabulary
root_vocab = extract_roots_from_corpus('data/corpus_with_sources_v2.jsonl')
print(f"Found {len(root_vocab)} unique roots")

# Save
with open('data/root_vocabulary.json', 'w') as f:
    json.dump(root_vocab, f, ensure_ascii=False, indent=2)
```

### Phase 3: Replace Tree-LSTM Embedder

```python
# In models/tree_lstm.py

class TreeLSTMEncoder(nn.Module):
    def __init__(self, ...):
        # OLD: Single embedding table
        # self.embed = nn.Embedding(vocab_size, embed_dim)

        # NEW: Compositional embeddings
        self.embed = CompositionalEmbedding(
            num_roots=5000,
            root_dim=64,
            affix_dim=8,
            grammar_dim=8,
            ending_dim=8
        )
```

### Phase 4: Test & Compare

```python
# tests/test_compositional_embeddings.py

def test_same_root_similar():
    """Words with same root should be similar."""
    embedder = CompositionalEmbedding(...)

    hundo_ast = parse("hundo")
    hundoj_ast = parse("hundoj")

    hundo_vec = embedder(hundo_ast)
    hundoj_vec = embedder(hundoj_ast)

    # Should be similar (same root, different grammar)
    similarity = cosine_similarity(hundo_vec, hundoj_vec)
    assert similarity > 0.9  # Very similar!

def test_opposite_prefix():
    """'mal' prefix should flip meaning."""
    bona_ast = parse("bona")  # good
    malbona_ast = parse("malbona")  # bad

    bona_vec = embedder(bona_ast)
    malbona_vec = embedder(malbona_ast)

    # Should be dissimilar (opposite meanings)
    similarity = cosine_similarity(bona_vec, malbona_vec)
    assert similarity < 0.5
```

---

## Expected Improvements

### Parameter Efficiency
- **Current**: ~10K vocab × 128 dims = 1.28M params
- **Compositional**: 5K roots × 64 dims + 38 affixes × 8 dims = 320K params
- **Reduction**: 75% fewer parameters! 🎉

### Generalization
- **Current**: Unseen word forms → OOV → bad embedding
- **Compositional**: Unseen forms composed from known parts → good embedding!
  - Example: Never saw "rehundejo" (re-dogification-place)
  - But can compose: re + hund + ej + o = reasonable embedding!

### Interpretability
- **Current**: Embedding dims are black box
- **Compositional**:
  - Dims 0-63: Root semantics
  - Dims 64-71: Prefix modifications
  - Dims 72-79: Suffix derivations
  - Dims 80-87: POS/ending (one-hot)
  - Dims 88-95: Grammar (case/number/tense)

Can probe each component separately!

---

## Summary & Recommendations

### Optimal Configuration

**96-dimensional compositional embeddings**:
- 64 dims: Root semantics (learned, 320K params)
- 16 dims: Affixes (frozen or slow-learning, 304 params)
- 16 dims: Grammar + POS (programmatic, 0 params)

**Total: ~320K parameters (vs 1.28M for whole-word)**

### Forced Dimensions

**Force these 16 dimensions** (17% of total):
1. Case (1 dim): Binary nominative/accusative
2. Number (1 dim): Binary singular/plural
3. Tense (3 dims): One-hot for 4 tenses
4. Mood (2 dims): One-hot for infinitive/imperative
5. POS (4 dims): One-hot for noun/adj/adv/verb
6. Reserved (5 dims): Future features

**Allow learning for 80 dimensions** (83% of total):
- Roots (64 dims): Full semantic space
- Affixes (16 dims): Initialized with priors, then fine-tuned

### Next Steps

1. **Extract root vocabulary** from corpus (~2 hours)
2. **Implement CompositionalEmbedding** class (~4 hours)
3. **Test composition** (same root → similar) (~2 hours)
4. **Integrate into Tree-LSTM** (~2 hours)
5. **Retrain with compositional embeddings** (~overnight)
6. **Compare performance** vs whole-word embeddings

**Total: 1-2 days of work for 75% parameter reduction!** 🚀

This is the Esperanto advantage in action - explicit morphology = free dimensions!
