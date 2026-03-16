# Root Embeddings Model - Complete Design

**Status**: Design updated - AST-aware semantic pairing (2026-03-10)
**Type**: Learned component (~2.9M params with 96D embeddings)
**Purpose**: Semantic similarity for content roots
**Phase**: Phase 1

---

## 🔄 IMPORTANT: Design Update (2026-03-10)

### What Changed

**OLD APPROACH (Lines 172-201)**: Positional window-based skip-gram
- Extract flat list of content roots from sentence
- Create pairs using 5-word positional window
- **Problem**: Learns distributional similarity (co-occurrence) which mixes semantic AND grammatical patterns
- Example: (hund, manĝ) is a grammatical relationship (subject-verb), not semantic

**NEW APPROACH (AST-Aware Semantic Pairing)**:
- Use AST structure to identify semantic relationships
- Create pairs based on syntactic roles that indicate semantic similarity
- **Benefit**: Learns PURE semantic similarity, not grammar (AST already knows grammar!)

### Why This Matters

```python
# Example sentence: "La granda hundo manĝas la malgrandan katon"
#                   (The big dog eats the small cat)

# OLD: Positional window creates these pairs:
pairs_old = [
    (grand, hund),   # ✅ SEMANTIC - adjective modifies noun
    (grand, manĝ),   # ❌ NOISE - adjective doesn't relate to distant verb
    (hund, manĝ),    # ❌ GRAMMAR - subject-verb (AST already knows this!)
    (manĝ, malgrand), # ❌ NOISE - verb doesn't relate to distant adjective
    (malgranda, kat), # ✅ SEMANTIC - adjective modifies noun
]
# Result: 40% useful pairs, 60% noise/grammar

# NEW: AST-aware creates only semantic pairs:
pairs_new = [
    (grand, hund),      # ✅ Modifier-head relationship
    (malgranda, kat),   # ✅ Modifier-head relationship
    (hund, kat),        # ✅ Semantic arguments (both participants in eating event)
]
# Result: 100% semantic pairs, 0% grammar/noise
```

**Key Insight**: We don't want embeddings to learn that "hund" and "manĝ" are related because that's a GRAMMATICAL relationship (subject-verb). The AST already captures this! We want embeddings to learn that "hund" and "kat" are similar because they're both ANIMALS (semantic category).

---

## 🎯 What This Model Does

### Purpose
**Capture semantic similarity between Esperanto content roots to enable:**
1. Semantic search: Find sentences with synonyms/related words
2. Improved retrieval: Rank by meaning, not just keyword match
3. Unknown root handling: Generalize to roots not in database

### Example
```
Query: "Kio estas planlingvo?"  (What is a planned language?)
Without embeddings: Only finds sentences with "planlingvo"
With embeddings: Also finds "artefarita lingvo" (artificial language),
                 "internacia lingvo" (international language)
                 → 40% more relevant results!
```

---

## 🏗️ Model Architecture

### Design Choice: Skip-gram with Negative Sampling

```python
class RootEmbedding(nn.Module):
    """
    Skip-gram model for root embeddings.

    Architecture: Simple but effective for semantic similarity.
    """

    def __init__(self, vocab_size=5000, embedding_dim=64):
        super().__init__()

        # Main embedding table (what we'll use after training)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Context embedding table (training only)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_root, context_root):
        """
        Predict if context_root appears near center_root.

        center_root: Root we're learning about (e.g., "hund")
        context_root: Nearby root in sentence (e.g., "bojl")

        Returns: Similarity score (higher = more likely to co-occur)
        """
        center_embed = self.embeddings(center_root)      # [batch, 64]
        context_embed = self.context_embeddings(context_root)  # [batch, 64]

        # Dot product = similarity
        similarity = (center_embed * context_embed).sum(dim=1)
        return similarity
```

### Why This Architecture?

**Skip-gram** learns: "Words that appear in similar contexts have similar meanings"

**Example**:
```
Sentence 1: "La hundo bojlas." (The dog barks)
Sentence 2: "La kato miaŭas." (The cat meows)
Sentence 3: "La hundo kuras." (The dog runs)

Training pairs:
- (hund, bojl) → high similarity
- (hund, kur) → high similarity
- (kat, miaŭ) → high similarity

After training:
- hund ≈ kat (both animals that make sounds)
- bojl ≈ miaŭ (both animal sounds)
- hund ≈ kur (dogs run)
```

### Model Size (UPDATED 2026-03-10)

**Production Configuration**:

| Component | Size |
|-----------|------|
| Vocabulary | 15,171 roots (Fundamento + validated corpus) |
| Embedding dimension | 96 dimensions per root |
| Total parameters | 15,171 × 96 × 2 = **2,912,832 params** |
| Model file size | ~11.4 MB (still tiny!) |
| Training time | ~4 hours on CPU (optimized) |

**Why 96 dimensions?** (Updated from 64D)
- Appropriate for larger vocabulary (15K vs 5K)
- Better semantic nuance capture
- Still fast to train with optimized settings
- Standard for mid-size word2vec models

**Why 15K vocabulary?** (Updated from 5K)
- Covers 100% Fundamento (official Esperanto)
- Covers 99%+ corpus content
- Clean (no garbage/noise)
- Reasonable parameter count (~3M)

### Production Training Settings (OPTIMIZED 2026-03-10)

**Dataset Optimization**:
```python
# OLD: Use all available pairs (95M pairs = 2,042 per root)
# → Massive overkill! Word2vec literature: 50-100 pairs optimal

# NEW: Smart sampling based on literature
{
    'dataset_fraction': 0.10,           # Use 10% of data
    'subsample_threshold': 1e-3,        # Mikolov et al. 2013 subsampling
    'cross_sentence_weight': 0.5,       # Discourse relationships
    'window_size': 5,                   # (only for cross-sentence fallback)

    # Result: ~50-200 pairs per root (adaptive based on frequency)
    # Matches Word2vec best practices!
}
```

**Training Optimization**:
```python
{
    'embedding_dim': 96,                # Appropriate for 15K vocab
    'batch_size': 512,                  # Stable convergence
    'epochs': 10,                       # Full convergence
    'learning_rate': 0.025,             # Standard for skip-gram
    'negative_samples': 5,              # Per positive pair
    'patience': 3,                      # Early stopping
    'min_delta': 0.001,                 # Loss improvement threshold
    'collapse_threshold': 0.7,          # Detect embedding collapse

    # Memory optimization
    'num_workers': 0,                   # For large datasets (>80M pairs)
    'use_persistent': False,            # Avoid memory duplication
}
```

**Why These Settings?**
1. **Dataset size**: 10% sampling + subsampling = ~50-200 pairs/root (optimal per literature)
2. **Batch size**: 512 balances speed and stability
3. **No OOM**: num_workers=0 prevents memory duplication with large datasets
4. **Training time**: ~4 hours on CPU (acceptable for production)
5. **Quality**: Matches Word2vec best practices (not overkill)

---

## 📊 Training Data Preparation

### What We Need

**Input**: 5.4M sentences from Kuzu database (already available!)

**Output**: Co-occurrence pairs (center_root, context_root, label)

### Step 1: Extract Content Roots from Corpus

**What deterministic parser does**:
```python
# For each sentence in corpus:
sentence = "La granda hundo kuras rapide."
ast = parse(sentence)  # Uses deterministic parser

# Extract roots from AST:
roots = []
for word in ast['words']:
    root = word['radiko']
    pos = word['vortspeco']

    # Only keep CONTENT roots (not function words!)
    if pos in ['substantivo', 'verbo', 'adjektivo']:
        roots.append(root)

# Result: ['grand', 'hund', 'kur', 'rapid']
# Excluded: 'la' (function word)
```

**Critical**: We **only embed content roots**, not function words!

**Why exclude function words** (see DETERMINISTIC_LIMITATIONS.md):
- Function words are grammatical, not semantic
- Including them causes embedding collapse (all words become similar)
- Function words handled by deterministic AST layer

### Step 2: Build Vocabulary (UPDATED 2026-03-10)

**Production Vocabulary Strategy**: Clean semantic roots only, no garbage

```python
from collections import Counter

# Count root frequencies across all 5.4M sentences
root_counts = Counter()
for sentence in corpus:
    roots = extract_content_roots(sentence)  # From Step 1
    root_counts.update(roots)

# Filter out garbage before selecting
def is_clean_root(root):
    """
    Filter garbage: pure digits, alphanumeric codes, etc.

    KEEP:
    - Fundamento roots (official Esperanto: 2,171 roots)
    - Validated high-frequency corpus roots
    - Number WORDS: unu, du, cent, mil (for quantifier grounding)

    REMOVE:
    - Pure digits: 007, 42, 1984
    - Alphanumeric codes: 13r, abc123, x2y
    """
    # Pure digits → deterministic parsing, no embedding
    if root.isdigit():
        return False

    # Alphanumeric codes
    if any(c.isdigit() for c in root) and any(c.isalpha() for c in root):
        return False

    return True

# Build clean vocabulary
fundamento_roots = load_fundamento_roots()  # 2,171 official roots
corpus_roots = [root for root, count in root_counts.most_common(20000)
                if is_clean_root(root)]

# Combine: Fundamento (all) + top corpus roots (validated)
vocabulary = list(set(fundamento_roots) | set(corpus_roots[:13000]))
# Total: ~15,171 clean semantic roots

# Create root → ID mapping
root_to_id = {root: i for i, root in enumerate(vocabulary)}
```

**Production Vocabulary (15,171 roots)**:
```python
{
    # Official Esperanto (Fundamento: 2,171 roots)
    'est': 0,      # "to be"
    'hav': 1,      # "to have"
    'far': 2,      # "to do/make"

    # High-frequency corpus (13,000 validated roots)
    'hom': 3,      # "person"
    'tag': 4,      # "day"
    'dinosaŭr': 5000,  # Less frequent but semantic

    # Number words (for quantifier grounding)
    'unu': 100,    # "one" (semantic, not "1" digit)
    'cent': 101,   # "hundred"
    'mil': 102,    # "thousand"
    'mult': 103,   # "many" (learns magnitude from cent, mil)
    'kelk': 104,   # "some" (learns from unu, du, tri)

    # What's EXCLUDED:
    # ✗ Pure digits: 007, 42 → parsed deterministically
    # ✗ Alphanumeric: 13r, abc123 → noise/codes
    # ✗ Function words: la, de, en → handled by AST
}
```

**Why This Vocabulary?**
1. **Quality over size**: 15K clean > 33K bloated
2. **Fundamento coverage**: 100% of official Esperanto
3. **No garbage**: 0% digits/codes (was 8.3% in old vocab)
4. **Number grounding**: mult/kelk learn magnitude through co-occurrence with number words
5. **99%+ coverage**: Covers virtually all semantic content in corpus

### Step 3: Generate Training Pairs (AST-Aware Semantic Pairing)

**NEW APPROACH**: Use AST structure to create semantically meaningful pairs instead of positional windows.

```python
def generate_semantic_pairs_from_ast(sentence_ast):
    """
    Generate (center, context) pairs using AST structure.

    Creates pairs based on SEMANTIC relationships, not positional proximity.

    Args:
        sentence_ast: Parsed AST from deterministic parser

    Returns:
        Semantic pairs: [(modifier, head), (arg1, arg2), ...]
    """
    pairs = []

    # 1. MODIFIER-HEAD RELATIONSHIPS (adjectives/adverbs modifying nouns/verbs)
    #    These indicate semantic similarity - modifiers describe properties
    for phrase in extract_phrases(sentence_ast):
        if phrase['tipo'] == 'vortgrupo':
            head = phrase['kerno']['radiko']

            # Add adjective-noun pairs
            for modifier in phrase['priskriboj']:
                if modifier['vortspeco'] == 'adjektivo':
                    modifier_root = modifier['radiko']
                    pairs.append((modifier_root, head, 1.0))  # Strong semantic link

    # 2. SEMANTIC ARGUMENTS (participants in same event)
    #    Subject and object of same verb are often semantically related
    #    (both animals, both people, both objects, etc.)
    if 'subjekto' in sentence_ast and 'objekto' in sentence_ast:
        subj_root = extract_head_root(sentence_ast['subjekto'])
        obj_root = extract_head_root(sentence_ast['objekto'])

        # Don't pair verb with args (that's grammar!)
        # But DO pair arguments with each other (semantic!)
        if subj_root and obj_root:
            pairs.append((subj_root, obj_root, 0.8))  # Moderate semantic link

    # 3. CROSS-SENTENCE DISCOURSE (adjacent sentences)
    #    Content roots from neighboring sentences show topical coherence
    #    Weight = 0.5 (weaker than intra-sentence relationships)

    # 4. COMPOUND ROOT COMPONENTS
    #    For compound words like "fiŝhundo" (seal = fish-dog)
    #    The components are semantically related
    for word in extract_all_words(sentence_ast):
        if has_multiple_roots(word):
            roots = extract_component_roots(word)
            for i, root1 in enumerate(roots):
                for root2 in roots[i+1:]:
                    pairs.append((root1, root2, 0.9))  # Strong semantic link

    return pairs
```

**Example - Semantic vs Positional**:
```
Sentence: "La granda hundo rapide manĝas la malgrandan katon"
          (The big dog quickly eats the small cat)

AST Structure:
- subjekto: vortgrupo
    - kerno: hundo (substantivo)
    - priskriboj: [granda (adjektivo)]
- verbo: manĝas
    - modifiers: [rapide (adverbo)]
- objekto: vortgrupo
    - kerno: katon (substantivo)
    - priskriboj: [malgrandan (adjektivo)]

Generated SEMANTIC pairs (NEW):
1. (grand, hund)      weight=1.0  # Modifier-head: adjective describes noun
2. (rapid, manĝ)      weight=1.0  # Modifier-head: adverb describes verb
3. (malgrand, kat)    weight=1.0  # Modifier-head: adjective describes noun
4. (hund, kat)        weight=0.8  # Semantic arguments: both animals, participants in event

What we DON'T create (filtering out grammar):
✗ (hund, manĝ)  - Subject-verb is GRAMMAR, not semantic similarity
✗ (kat, manĝ)   - Object-verb is GRAMMAR, not semantic similarity
✗ (grand, manĝ) - No syntactic relationship (positional noise)
```

**Why This Works Better**:
```
OLD (positional window):
- Creates ~50-100 pairs per sentence
- 40% semantic pairs + 60% grammar/noise
- Model wastes capacity learning grammar
- Needs massive dataset to separate signal from noise

NEW (AST-aware):
- Creates ~10-20 pairs per sentence (more selective)
- 100% semantic pairs, 0% grammar
- Model focuses ALL capacity on semantics
- Needs 5x less data for same quality
```

### Step 4: Negative Sampling

**Problem**: We need negative examples too!

```python
def generate_negative_samples(center_root, num_negatives=5):
    """
    Generate random roots that DON'T appear with center_root.

    This teaches the model: "hund and aeroplan don't co-occur"
    """
    negative_roots = []

    for _ in range(num_negatives):
        # Sample random root from vocabulary
        random_root = random.choice(vocabulary)
        negative_roots.append(random_root)

    return negative_roots
```

**Example**:
```
Positive pair: (hund, bojl) → label = 1 (they co-occur)

Negative pairs (random samples):
- (hund, aeroplan) → label = 0 (dog doesn't co-occur with airplane)
- (hund, matematiko) → label = 0
- (hund, oceano) → label = 0
```

### Final Training Data Format

```python
# Each training example:
{
    'center_root_id': 123,      # ID of center root
    'context_root_id': 456,     # ID of context root
    'label': 1,                 # 1 = positive, 0 = negative
}

# Total training examples:
# - 5.4M sentences
# - ~10 roots per sentence (content only)
# - ~50 pairs per sentence (window_size=5)
# - 5 negative samples per positive
# → ~1.6 BILLION training pairs!
```

---

## 🔄 Training Pipeline

### Complete Data Pipeline

```
Step 1: Extract Roots from Corpus (Deterministic Parser)
├─ Input: 5.4M sentences in Kuzu
├─ Process: Parse each sentence, extract content roots only
└─ Output: List of root sequences
    Example: [['hund', 'bojl'], ['kat', 'miaŭ'], ...]
    Time: ~2 hours (parallel processing)

Step 2: Build Vocabulary
├─ Input: All extracted roots
├─ Process: Count frequencies, keep top 5,000
└─ Output: vocabulary.json
    Example: {'est': 0, 'hav': 1, ...}
    Time: ~10 minutes

Step 3: Generate Training Pairs
├─ Input: Root sequences + vocabulary
├─ Process: Skip-gram pairs + negative sampling
└─ Output: training_pairs.pt (PyTorch tensor file)
    Size: ~6 GB (1.6B pairs × 4 bytes)
    Time: ~4 hours

Step 4: Train Model
├─ Input: training_pairs.pt
├─ Process: SGD with skip-gram + negative sampling loss
└─ Output: root_embeddings_64d.pt
    Size: ~1.3 MB (320K params)
    Time: ~2 days on GPU, ~1 week on CPU
```

### Training Script (Simplified)

```python
# scripts/train_root_embeddings.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Load training pairs
train_data = load_training_pairs('data/training/skipgram_pairs.pt')
dataloader = DataLoader(train_data, batch_size=1024, shuffle=True)

# Initialize model
model = RootEmbedding(vocab_size=5000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss function: Binary Cross Entropy (predict co-occurrence)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(10):
    for batch in dataloader:
        center_ids = batch['center_root_id']
        context_ids = batch['context_root_id']
        labels = batch['label']

        # Forward pass
        scores = model(center_ids, context_ids)
        loss = criterion(scores, labels.float())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Save embeddings
torch.save(model.embeddings.state_dict(),
           'models/root_embeddings_64d.pt')
```

---

## 🔍 What Semantic Meaning Will It Capture?

### Types of Semantic Relations

**1. Synonymy** (Similar meaning)
```
planlingvo ≈ artefarita_lingvo ≈ konstruita_lingvo
(planned language ≈ artificial language ≈ constructed language)

Similarity: 0.85+
```

**2. Hypernymy/Hyponymy** (Category relationships)
```
animal ≈ hund, kat, bird  (animal is category of dog, cat, bird)
hund ≈ kat  (both animals)

Similarity: 0.70-0.80
```

**3. Meronymy** (Part-whole relationships)
```
arb ≈ branĉ, foli, radik  (tree has branches, leaves, roots)

Similarity: 0.60-0.75
```

**4. Functional Similarity** (Used in similar contexts)
```
kuir ≈ bak, frit, rost  (all cooking methods)
lern ≈ stud, kompren, sci  (all learning-related)

Similarity: 0.65-0.80
```

**5. Antonymy** (Opposites - often appear in similar contexts!)
```
grand ≈ malgrand  (big ≈ small - both describe size)
bon ≈ malbona  (good ≈ bad)

Similarity: 0.50-0.65 (lower but still related)
```

### Example Embedding Space (After Training)

```
Visualizing 64D embeddings in 2D (using t-SNE):

           kat •
       hund •    • bird
           • animal

   kuir •  • bak
       • frit

grand •          • malgrand
      •          •
      bon    malbon

Observations:
- Animals cluster together
- Cooking verbs cluster together
- Antonyms are nearby (same semantic domain)
```

---

## 🎯 Relationship with Deterministic Components

### What Deterministic Parser Provides

**1. Content Root Extraction**
```python
# Deterministic parser identifies content roots
sentence = "La granda hundo rapide kuras"
ast = parser.parse(sentence)

# Parser tells us:
- "la" = function word (artikolo) → SKIP ❌
- "granda" = content word (adjektivo) → KEEP ✅ root='grand'
- "hundo" = content word (substantivo) → KEEP ✅ root='hund'
- "rapide" = content word (adverbo) → KEEP ✅ root='rapid'
- "kuras" = content word (verbo) → KEEP ✅ root='kur'

# Embedding model only sees: ['grand', 'hund', 'rapid', 'kur']
```

**2. Morphological Decomposition**
```python
# Parser decomposes compound words
word = "rehundejo"  # dog shelter again
ast = parser.parse_word(word)

# Parser provides:
- prefikso: 're'      → SKIP (grammatical)
- radiko: 'hund'      → KEEP ✅ (content)
- sufikso: 'ej'       → SKIP (grammatical)
- finalo: 'o'         → SKIP (grammatical)

# Embedding model only sees: 'hund'
```

**3. Part-of-Speech Filtering**
```python
# Parser identifies word types
def should_embed(word_ast):
    pos = word_ast['vortspeco']

    # Content words → embed
    if pos in ['substantivo', 'verbo', 'adjektivo', 'adverbo']:
        return True

    # Function words → skip
    if pos in ['pronomo', 'prepozicio', 'konjunkcio', 'artikolo']:
        return False

    # Correlatives → depends on semantic load
    if pos == 'korelativo':
        # 'kio', 'kiu' → semantic (embed)
        # 'ĉio', 'ĉiu' → grammatical (skip)
        return is_semantic_correlative(word_ast)
```

### What Embedding Model Provides Back

**1. Semantic Similarity Search**
```python
# In retriever, instead of keyword match:
query_roots = ['planlingvo']  # From parser
query_embedding = model.embeddings(query_roots)

# Find similar roots
similar_roots = find_similar(query_embedding, top_k=10)
# Returns: ['artefarita_lingvo', 'internacia_lingvo', 'helplingvo', ...]

# Expand retrieval with synonyms!
```

**2. Unknown Root Handling**
```python
# For roots not in 50 annotated:
unknown_root = 'dinosaŭr'

# Can't query database (no annotation)
# But embedding knows semantic properties!
embedding = model.embeddings('dinosaŭr')

# Find nearest neighbors
neighbors = find_similar(embedding)
# → ['reptilio', 'best', 'antikv'] (reptile, beast, ancient)

# Infer properties from neighbors
# → probably: substantiva_klaso='organism', graveco_hav=0.3
```

---

## 🏭 Specialized Models for Different Word Types?

### Current Design: Single Model for All Content Roots

**Why unified model?**
1. **Simplicity**: One model, one training pipeline
2. **Transfer learning**: Verbs and nouns inform each other
3. **Small vocabulary**: 5K roots fits comfortably in 320K params

**Example of cross-POS learning**:
```
Sentence: "La kuracisto kuracas la malsanulon"
          (The doctor heals the patient)

Model learns:
- kuracist (noun) ≈ kurac (verb) → both medical
- malsanul (noun) ≈ kurac (verb) → patient needs healing
→ Cross-part-of-speech semantics!
```

### Future: Specialized Models (Phase 2/3)?

**If we expand to 100K+ vocabulary**, might benefit from specialization:

**Option A: Separate Models by POS**
```python
class SpecializedEmbeddings:
    noun_embeddings = Embedding(30000, 64)  # 30K nouns
    verb_embeddings = Embedding(10000, 64)  # 10K verbs
    adj_embeddings = Embedding(5000, 64)    # 5K adjectives
```

**Pros**:
- Verbs can have different semantic dimensions (tense, aspect)
- Nouns can capture taxonomies better

**Cons**:
- More complex training
- Can't capture cross-POS relations
- **Not needed for Phase 1**

**Option B: Compositional Models**
```python
# For compound roots like "fiŝo-hundo" (seal = fish-dog)
embedding = compose(fish_embedding, dog_embedding)
```

**Verdict for Phase 1**: **Single unified model sufficient**
- Only 5K roots (manageable)
- Cross-POS learning is beneficial
- Can revisit in Phase 2 if needed

---

## 📥 Model Input/Output Specification

### Input During Training

```python
# Training example (one pair)
{
    'center_root_id': torch.tensor([123]),   # int64
    'context_root_id': torch.tensor([456]),  # int64
    'label': torch.tensor([1.0]),            # float32 (1=positive, 0=negative)
}

# Batch of 1024 examples
{
    'center_root_id': torch.tensor([123, 456, 789, ...]),  # [1024]
    'context_root_id': torch.tensor([456, 789, 234, ...]), # [1024]
    'label': torch.tensor([1.0, 0.0, 1.0, ...]),           # [1024]
}
```

### Output After Training

```python
# Load trained embeddings
embeddings = torch.load('models/root_embeddings_64d.pt')

# Query single root
hund_embedding = embeddings['hund']  # shape: [64]
# → array([0.23, -0.45, 0.12, ..., 0.67])  # 64 numbers

# Query multiple roots
roots = ['hund', 'kat', 'bird']
root_ids = [root_to_id[r] for r in roots]
batch_embeddings = embeddings[root_ids]  # shape: [3, 64]

# Compute similarity
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(
    embeddings['hund'].unsqueeze(0),
    embeddings['kat'].unsqueeze(0)
)
# → 0.78 (high similarity - both animals)
```

### Integration with Retriever

```python
# Before (Phase 0): Keyword matching
def retrieve(query):
    keywords = extract_roots(query)  # ['planlingvo']
    return query_database(keywords)  # Only exact matches

# After (Phase 1): Semantic search
def retrieve(query):
    keywords = extract_roots(query)  # ['planlingvo']

    # Expand with semantic neighbors
    expanded = []
    for keyword in keywords:
        embedding = model.embeddings[keyword]
        similar_roots = find_similar(embedding, top_k=5)
        expanded.extend(similar_roots)

    # Now search for: ['planlingvo', 'artefarita_lingvo',
    #                  'konstruita_lingvo', 'helplingvo', ...]
    return query_database(expanded)  # 40% more results!
```

---

## 🧪 Evaluation Strategy

### How We'll Know If It Works

**1. Synonym Detection**
```python
# Test set: Known synonym pairs
test_pairs = [
    ('planlingvo', 'artefarita_lingvo', 1),  # Synonyms
    ('hund', 'kat', 0.7),                     # Related
    ('hund', 'aeroplan', 0.1),                # Unrelated
]

for root1, root2, expected_similarity in test_pairs:
    predicted = cosine_similarity(
        embeddings[root1],
        embeddings[root2]
    )

    assert abs(predicted - expected) < 0.2  # Within threshold
```

**2. Analogy Task**
```python
# Test: "hund is to bojl as kat is to ___?"
# Expected: miaŭ

result = embeddings['kat'] + (embeddings['bojl'] - embeddings['hund'])
nearest = find_nearest_neighbor(result)

assert nearest == 'miaŭ'  # Cat says "meow"
```

**3. Retrieval Quality**
```python
# Run 10 test queries with embeddings
queries = ["Kio estas planlingvo?", "Rakontu pri Zamenhof", ...]

for query in queries:
    results_without_embeddings = retrieve_baseline(query)
    results_with_embeddings = retrieve_semantic(query)

    # Measure relevance (human evaluation)
    baseline_relevance = evaluate(results_without_embeddings)
    semantic_relevance = evaluate(results_with_embeddings)

    improvement = semantic_relevance - baseline_relevance
    print(f"Improvement: {improvement:.1%}")

# Target: 40% average improvement
```

---

## 📋 Implementation Checklist

### Data Preparation (4-6 hours)
- [ ] Extract content roots from 5.4M sentences (parser)
- [ ] Build vocabulary (top 5K frequent roots)
- [ ] Generate skip-gram training pairs
- [ ] Add negative samples (5 per positive)
- [ ] Save training data (~6 GB file)

### Model Training (2 days GPU, 1 week CPU)
- [ ] Implement skip-gram model architecture
- [ ] Set up training loop with Adam optimizer
- [ ] Train for 10 epochs
- [ ] Monitor loss convergence
- [ ] Save trained embeddings

### Evaluation (1-2 hours)
- [ ] Test on synonym pairs
- [ ] Test on analogy tasks
- [ ] Measure retrieval improvement
- [ ] Human evaluation on 10 test queries

### Integration (2-3 hours)
- [ ] Load embeddings into retriever
- [ ] Implement semantic expansion
- [ ] Update retrieval ranking
- [ ] Test end-to-end pipeline

**Total estimated time**: ~1 week (mostly training time)

---

## 🎯 Success Criteria

**Minimum viable**:
- [ ] Synonym similarity > 0.7
- [ ] Unrelated words similarity < 0.3
- [ ] Retrieval improvement > 20%

**Target goals**:
- [ ] Synonym similarity > 0.85
- [ ] Analogy accuracy > 70%
- [ ] Retrieval improvement > 40%

---

## 💡 Key Design Decisions

### Why Skip-gram (Not CBOW)?
- Skip-gram better for rare words
- Skip-gram captures more semantic nuances
- CBOW faster but less accurate for small datasets

### Why 64D (Not 128D or 300D)?
- 64D sufficient for 5K vocabulary
- Faster training
- Smaller model size
- Can expand to 128D later if needed

### Why Content Roots Only?
- Function words are grammatical, not semantic
- Including them causes embedding collapse
- Deterministic parser handles function words perfectly

### Why Self-Supervised?
- No manual annotation needed
- Scales to full corpus (5.4M sentences)
- Co-occurrence is reliable signal for semantics

---

## 🚀 Implementation Status (2026-03-10)

### What's Documented (This Design)

✅ **AST-aware semantic pairing approach** - Fully specified
- Modifier-head relationships (adjective-noun)
- Semantic arguments (subject-object pairs)
- Cross-sentence discourse
- Compound root components
- Weighted by relationship type

✅ **Production vocabulary** - Created and validated
- 15,171 clean semantic roots
- 100% Fundamento coverage
- 0% garbage (no digits, codes)
- Located: `data/vocabularies/production_semantic_roots_15k.json`

✅ **Optimized training settings** - Tested and validated
- 96D embeddings, 2.9M params
- 10% dataset sampling (~50-200 pairs/root)
- 512 batch size, 10 epochs
- ~4 hours training time on CPU
- Memory-safe (num_workers=0 for large datasets)

### What's Implemented (Current Code)

❌ **Extraction script** - Uses OLD positional window approach
- Current: `scripts/extract_embedding_training_pairs.py` (lines 143-193, 437-453)
- Gets roots by AST role but creates flat list
- Generates pairs with 5-word positional window
- **Needs rewrite** to use AST structure for semantic pairing

✅ **Training script** - Ready for new data
- Current: `scripts/train_root_embeddings_skipgram_v2_1.py`
- Supports dataset sampling, memory optimization, checkpointing
- Works with new settings (96D, 512 batch, etc.)
- **No changes needed** - just feed it better pairs!

✅ **Pipeline script** - Ready for new extraction
- Current: `scripts/train_phase1_embeddings_fast.sh`
- Uses production vocabulary (15K roots)
- Configured with optimized settings
- **Will work once extraction is fixed**

### Next Steps

**1. Rewrite Extraction (HIGH PRIORITY)**
File: `scripts/extract_embedding_training_pairs.py`

Changes needed:
```python
# REPLACE: Lines 437-453 (positional window pairing)
for i, target in enumerate(target_roots):
    start = max(0, i - window_size)
    end = min(len(target_roots), i + window_size + 1)
    for j in range(start, end):
        if i != j:
            pairs.append((target, context, 1.0))

# WITH: AST-aware semantic pairing
def extract_semantic_pairs_from_ast(db, frazo_id):
    # Get AST structure from database
    ast = query_sentence_ast(db, frazo_id)

    # 1. Modifier-head pairs (adjektivo-substantivo)
    # 2. Semantic arguments (subjekto-objekto)
    # 3. Cross-sentence discourse (adjacent sentences)
    # 4. Compound root components

    return pairs_with_weights
```

**2. Validate New Pairs**
- Extract 1,000 sample pairs
- Manual review: Are they semantic relationships?
- Expected: 100% semantic, 0% grammar

**3. Train Production Model**
- Run: `./scripts/train_phase1_embeddings_fast.sh --fresh`
- Expected time: ~4 hours
- Expected output: `models/root_embeddings_phase1_fast/root_embeddings_best.pt`

**4. Evaluate Embeddings**
- Test synonym similarity (hund ≈ kat > 0.7)
- Test unrelated words (hund ≈ aeroplan < 0.3)
- Test quantifier grounding (mult ≈ cent, mil)

### Estimated Timeline

- **Day 1**: Rewrite extraction script (4-6 hours)
- **Day 1**: Validate sample pairs (1 hour)
- **Day 1-2**: Train production model (~4 hours, can run overnight)
- **Day 2**: Evaluate embeddings (2 hours)
- **Day 2**: Integration testing (2 hours)

**Total**: 1-2 days to production-ready embeddings

---

## 📝 Design Decisions Summary

**Key principles**:
1. **AST-aware pairing** - Use structure, not position
2. **Semantic-only** - No grammar (AST already knows!)
3. **Clean vocabulary** - No garbage, quality over size
4. **Literature-based** - Follow Word2vec best practices (50-100 pairs/root)
5. **Production-ready** - Optimized for CPU, reasonable time (~4 hours)

**Files to read for full context**:
- This design doc: `docs/ROOT_EMBEDDINGS_DESIGN.md`
- Current extraction: `scripts/extract_embedding_training_pairs.py`
- Current training: `scripts/train_root_embeddings_skipgram_v2_1.py`
- Pipeline: `scripts/train_phase1_embeddings_fast.sh`
- Vocabulary: `data/vocabularies/production_semantic_roots_15k.json`

