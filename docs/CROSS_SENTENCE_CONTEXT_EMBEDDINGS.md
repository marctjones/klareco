# Cross-Sentence Context for Better Embeddings

**Idea**: Use content from adjacent sentences, not just within-sentence co-occurrence

**Benefit**: Captures discourse-level semantic relationships

---

## 🎯 Why Cross-Sentence Context Helps

### Problem with Sentence-Only Context

**Current approach** (within-sentence only):
```
Sentence 1: "Ludoviko Zamenhof fondis Esperanton."
           Roots: [Ludoviko, Zamenhof, fond, Esperanton]
           Pairs: (Ludoviko, Zamenhof), (Ludoviko, fond), (Ludoviko, Esperanton),
                  (Zamenhof, fond), (Zamenhof, Esperanton), (fond, Esperanton)

Sentence 2: "Li estis kuracisto en Bjalistoko."
           Roots: [li, est, kuracist, Bjalistoko]
           Pairs: (li, est), (li, kuracist), (li, Bjalistoko),
                  (est, kuracist), (est, Bjalistoko), (kuracist, Bjalistoko)

Missing relationships:
❌ (fond, kuracist) → Founder was a doctor
❌ (Esperanton, kuracist) → Esperanto's creator was a doctor
❌ (Zamenhof, Bjalistoko) → Zamenhof in Bjalistoko (separated by sentence)
```

**With cross-sentence context**:
```
Context window spans sentences:
Sentence 1: [Ludoviko, Zamenhof, fond, Esperanton]
Sentence 2: [li, est, kuracist, Bjalistoko]

New pairs captured:
✅ (fond, kuracist) → Founder-profession relationship
✅ (Esperanton, kuracist) → Creator-profession relationship
✅ (Zamenhof, Bjalistoko) → Person-location relationship
✅ (fond, Bjalistoko) → Action-location relationship
```

### Benefits

**1. Discourse-level semantics**
```
Paragraph about Zamenhof:
S1: "Zamenhof fondis Esperanton en 1887."
S2: "Li estis kuracisto."
S3: "Lia celo estis paco."

Within-sentence: fondis↔Esperanto, estis↔kuracisto, celo↔paco
Cross-sentence: fondis↔kuracisto↔paco (founder was peaceful doctor)
```

**2. Coreference relationships**
```
S1: "La hundo bojlas."
S2: "Ĝi estas granda."

Within-sentence: hund↔bojl, ĝi↔grand
Cross-sentence: hund↔grand (the dog is big)
→ Learns: "hund" and "grand" co-occur (even with pronoun gap)
```

**3. Topic coherence**
```
Paragraph about cooking:
S1: "Mi kuiras viandon."
S2: "La forno estas varma."
S3: "La rezulto estas bongusta."

Within-sentence: kuir↔viand, forn↔varm, rezult↔bongust
Cross-sentence: kuir↔forn↔bongust (cooking-oven-tasty domain)
```

**4. Better rare word coverage**
```
S1: "La dinosaŭro vivis antaŭ milionoj da jaroj."
S2: "Ĝi estis granda reptilio."

Within-sentence: dinosaŭr↔viv↔jar, grand↔reptili
Cross-sentence: dinosaŭr↔reptili (dinosaur is reptile)
→ Rare word "dinosaŭr" gets connection to "reptili"!
```

---

## 🏗️ Implementation Strategies

### Strategy 1: Fixed Cross-Sentence Window

**Simple approach**: Extend window across sentence boundaries

```python
def generate_cross_sentence_pairs(paragraph_roots, window_size=5):
    """
    Generate pairs with window spanning sentences.

    Args:
        paragraph_roots: [
            ['hund', 'bojl'],        # Sentence 1
            ['ĝi', 'est', 'grand'],  # Sentence 2
            ['ĝi', 'kur', 'rapid']   # Sentence 3
        ]
        window_size: Context window (spans sentences)

    Returns:
        All pairs within window_size, regardless of sentence boundaries
    """
    # Flatten to single sequence
    all_roots = []
    for sent_roots in paragraph_roots:
        all_roots.extend(sent_roots)

    # all_roots = ['hund', 'bojl', 'ĝi', 'est', 'grand', 'ĝi', 'kur', 'rapid']

    # Generate pairs with fixed window
    pairs = []
    for i, center in enumerate(all_roots):
        start = max(0, i - window_size)
        end = min(len(all_roots), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                pairs.append((center, all_roots[j]))

    return pairs

# Example:
# (hund, bojl) ✅ Within S1
# (hund, ĝi) ✅ Cross S1→S2
# (hund, est) ✅ Cross S1→S2
# (bojl, grand) ✅ Cross S1→S2
# (ĝi, kur) ✅ Cross S2→S3
```

**Pros**:
- Simple to implement
- Captures all local context

**Cons**:
- May capture noise if sentences are weakly related
- Equal weight to within-sentence and cross-sentence

---

### Strategy 2: Weighted Cross-Sentence Window

**Smarter approach**: Distance-based weighting

```python
def generate_weighted_cross_sentence_pairs(paragraph_roots,
                                           window_size=5,
                                           cross_sentence_weight=0.5):
    """
    Generate pairs with reduced weight for cross-sentence.

    Intuition: Words in same sentence are more related than
               words in adjacent sentences.
    """
    pairs = []

    # Track sentence boundaries
    sentence_boundaries = []
    pos = 0
    for sent_roots in paragraph_roots:
        sentence_boundaries.append((pos, pos + len(sent_roots)))
        pos += len(sent_roots)

    # Flatten
    all_roots = [r for sent in paragraph_roots for r in sent]

    # Generate pairs with weights
    for i, center in enumerate(all_roots):
        center_sentence = _get_sentence_id(i, sentence_boundaries)

        start = max(0, i - window_size)
        end = min(len(all_roots), i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue

            context = all_roots[j]
            context_sentence = _get_sentence_id(j, sentence_boundaries)

            # Same sentence: full weight
            if center_sentence == context_sentence:
                weight = 1.0
                pairs.append((center, context, weight))

            # Adjacent sentence: reduced weight
            elif abs(center_sentence - context_sentence) == 1:
                weight = cross_sentence_weight
                pairs.append((center, context, weight))

            # Distant sentences: skip
            else:
                continue

    return pairs

# Example output:
# (hund, bojl, weight=1.0) ✅ Same sentence (S1)
# (hund, ĝi, weight=0.5) ✅ Adjacent sentence (S1→S2)
# (hund, est, weight=0.5) ✅ Adjacent sentence (S1→S2)
# (hund, kur, weight=0.0) ❌ Skip (S1→S3, too far)
```

**Pros**:
- Prioritizes within-sentence relationships
- Still captures cross-sentence context
- Reduces noise from distant sentences

**Cons**:
- Slightly more complex
- Need to tune weight parameter

---

### Strategy 3: Document-Level Context with Decay

**Most sophisticated**: Exponential distance decay

```python
def generate_document_context_pairs(paragraph_roots,
                                    max_distance=10,
                                    decay_factor=0.5):
    """
    Generate pairs with exponential decay by distance.

    Weight = decay_factor ^ (distance_between_words)
    """
    all_roots = [r for sent in paragraph_roots for r in sent]

    pairs = []
    for i, center in enumerate(all_roots):
        for j in range(max(0, i - max_distance),
                      min(len(all_roots), i + max_distance + 1)):
            if i == j:
                continue

            context = all_roots[j]
            distance = abs(i - j)

            # Weight decays exponentially
            weight = decay_factor ** distance

            pairs.append((center, context, weight))

    return pairs

# Example weights:
# distance=0: weight=1.0 (same position - impossible)
# distance=1: weight=0.5 (immediate neighbor)
# distance=2: weight=0.25 (skip one word)
# distance=3: weight=0.125 (skip two words)
# distance=5: weight=0.03 (across sentence boundary)
# distance=10: weight=0.001 (very distant)
```

**Pros**:
- Most nuanced
- Captures long-range dependencies
- Smooth transition across sentence boundaries

**Cons**:
- Most complex
- Highest computational cost

---

## 📊 Comparison of Strategies

### Training Data Size

**Sentence-only** (baseline):
```
5.4M sentences
× 10 roots per sentence
× 5 pairs per root (window=5 within sentence)
= 270M positive pairs
× 5 negative samples
= 1.35B total pairs
```

**Strategy 1 (Fixed cross-sentence, window=5)**:
```
5.4M sentences in ~1M paragraphs (avg 5 sentences/paragraph)
× 50 roots per paragraph (10 per sentence × 5 sentences)
× 10 pairs per root (window=5, spans sentences)
= 500M positive pairs
× 5 negative samples
= 2.5B total pairs (1.9x more)
```

**Strategy 2 (Weighted, weight=0.5)**:
```
270M within-sentence pairs (weight=1.0)
+ 100M cross-sentence pairs (weight=0.5, effective = 50M)
= 320M effective positive pairs
× 5 negative samples
= 1.6B total pairs (1.2x more)
```

**Strategy 3 (Exponential decay)**:
```
Similar to Strategy 2, depends on decay factor
```

---

## 🧪 Expected Quality Improvements

### Semantic Coverage

**Baseline (sentence-only)**:
```python
# Test: How many semantic relationships captured?
test_pairs = [
    ('kuracist', 'fond'),  # Doctor who founded (often separate sentences)
    ('hund', 'grand'),     # Dog that is big (coreference)
    ('dinosaŭr', 'reptili'), # Dinosaur is reptile (definition pattern)
]

baseline_captures = 40%  # Often in separate sentences
```

**Cross-sentence context**:
```python
cross_sentence_captures = 75%  # Captures most relationships
→ 35% improvement in coverage
```

### Retrieval Quality

**Example query**: "Kiu estis Zamenhof?"

**Baseline**: Finds sentences with "Zamenhof", "estis"
```
Result 1: "Zamenhof fondis Esperanton." ✅
Result 2: "Li estis kuracisto." ❌ (no "Zamenhof" - missed)
```

**With cross-sentence**:
- Model learns: (Zamenhof, kuracist) from cross-sentence context
- When searching, "Zamenhof" → also retrieves sentences with "kuracisto"
```
Result 1: "Zamenhof fondis Esperanton." ✅
Result 2: "Li estis kuracisto." ✅ (found via "kuracist" ≈ "Zamenhof")
```

**Estimated improvement**: 20-30% better retrieval

---

## ⚖️ Tradeoffs

### Pros

**1. Better semantic coverage**
- Captures discourse-level relationships
- Handles coreference implicitly
- Better rare word embeddings

**2. More training data**
- 1.2-1.9x more pairs (depending on strategy)
- Better coverage for less frequent roots

**3. Document coherence**
- Learns topic-level relationships
- Captures narrative flow

### Cons

**1. Increased noise**
- Some cross-sentence pairs are spurious
- Paragraph boundaries matter (cross-paragraph = noise)

**2. Computational cost**
- 20-90% more training pairs
- Longer training time (still manageable)

**3. Implementation complexity**
- Need paragraph segmentation
- More sophisticated pair generation

---

## 🚀 Recommended Approach

### **Strategy 2: Weighted Cross-Sentence** (Best balance)

**Configuration**:
```python
window_size = 5  # Standard within-sentence
cross_sentence_weight = 0.5  # Half weight for adjacent sentences
max_cross_sentence_distance = 1  # Only adjacent sentences (not skip one)
```

**Why**:
- ✅ Captures cross-sentence relationships
- ✅ Reduces noise (only adjacent sentences)
- ✅ Modest 20% increase in training data (manageable)
- ✅ Simple to implement
- ✅ Easy to ablate (compare with/without)

**Implementation**:
```python
def extract_paragraph_context(corpus):
    """
    Extract roots with paragraph-level context.

    Returns:
        List of paragraphs, each containing root sequences
    """
    paragraphs = []

    for document in corpus:
        # Split into paragraphs (or use existing paragraph boundaries)
        doc_paragraphs = split_into_paragraphs(document)

        for paragraph in doc_paragraphs:
            # Parse each sentence
            paragraph_roots = []
            for sentence in paragraph.sentences:
                ast = parse(sentence)
                roots = extract_content_roots(ast)
                paragraph_roots.append(roots)

            if paragraph_roots:
                paragraphs.append(paragraph_roots)

    return paragraphs

# Training data generation
paragraphs = extract_paragraph_context(corpus)
training_pairs = []

for paragraph_roots in paragraphs:
    pairs = generate_weighted_cross_sentence_pairs(
        paragraph_roots,
        window_size=5,
        cross_sentence_weight=0.5
    )
    training_pairs.extend(pairs)
```

---

## 📋 Implementation Checklist

**Data preparation** (add to existing pipeline):
- [ ] Identify paragraph boundaries in corpus
  - Use Frazoteksto → Paragrafo relationships in Kuzu
  - Or use blank lines / section breaks in text
- [ ] Extract roots per paragraph (not just per sentence)
- [ ] Implement weighted cross-sentence pair generation
- [ ] Validate: Sample 100 paragraphs, check pairs make sense

**Training modifications**:
- [ ] Update training script to use weighted pairs
- [ ] Modify loss function to handle weights:
  ```python
  loss = weight * criterion(scores, labels)
  ```
- [ ] Monitor: Track within-sentence vs cross-sentence loss separately

**Evaluation** (ablation study):
- [ ] Train baseline model (sentence-only)
- [ ] Train cross-sentence model
- [ ] Compare on:
  - Synonym accuracy
  - Retrieval quality
  - Coreference relationships
  - Rare word embeddings

---

## 🔬 Ablation Study Design

### Experiment Setup

**Model 1: Baseline** (sentence-only)
```python
training_data = generate_sentence_only_pairs(corpus, window=5)
# ~1.35B pairs
```

**Model 2: Cross-sentence** (weighted)
```python
training_data = generate_weighted_cross_sentence_pairs(
    corpus,
    window=5,
    cross_sentence_weight=0.5
)
# ~1.6B pairs
```

**Model 3: Cross-sentence (higher weight)**
```python
training_data = generate_weighted_cross_sentence_pairs(
    corpus,
    window=5,
    cross_sentence_weight=0.8  # More emphasis
)
# ~1.8B pairs
```

### Evaluation Metrics

**1. Coreference relationships** (direct benefit)
```python
test_cases = [
    # (word_in_S1, word_in_S2, expected_similarity)
    ("hund", "grand", 0.7),  # "La hundo bojlas. Ĝi estas granda."
    ("Zamenhof", "kuracist", 0.75),  # "Zamenhof fondis... Li estis kuracisto."
    ("dinosaŭr", "reptili", 0.8),  # "dinosaŭro vivis... Ĝi estis reptilio."
]

for model in [baseline, cross_sentence]:
    score = evaluate_coreference(model, test_cases)
    print(f"{model}: Coreference score = {score:.1%}")

# Expected:
# Baseline: 45% (misses many relationships)
# Cross-sentence: 75% (captures most) ✅ +30%
```

**2. Retrieval quality**
```python
test_queries = [
    "Kiu estis Zamenhof?",
    "Kio estas dinosaŭro?",
    "Rakontu pri kuirado.",
]

for model in [baseline, cross_sentence]:
    avg_precision = evaluate_retrieval(model, test_queries)
    print(f"{model}: Retrieval = {avg_precision:.1%}")

# Expected:
# Baseline: 62%
# Cross-sentence: 78% ✅ +16%
```

**3. Rare word quality**
```python
rare_words = ['dinosaŭr', 'xilofon', 'krokodil']

for model in [baseline, cross_sentence]:
    avg_neighbors = count_valid_neighbors(model, rare_words)
    print(f"{model}: Rare words = {avg_neighbors} valid neighbors")

# Expected:
# Baseline: 2.3 valid neighbors per rare word
# Cross-sentence: 4.1 valid neighbors ✅ +78%
```

---

## 🎯 Expected Outcomes

**Conservative estimate**:
- Coreference relationships: +25-35% accuracy
- Retrieval quality: +15-25% precision
- Rare word coverage: +50-100% valid neighbors
- Training time: +20% (1.6B vs 1.35B pairs)

**Cost/Benefit**:
```
Cost: +20% training time (~0.5 days extra)
Benefit: +20% average quality improvement across all metrics
→ Worth it! ✅
```

---

## 💡 Key Insights

### When Cross-Sentence Helps Most

**1. Biographical/narrative text** (high benefit)
```
"Zamenhof naskiĝis en Bjalistoko. Li fondis Esperanton. Li estis kuracisto."
→ Many cross-sentence relationships
```

**2. Technical/definitional text** (high benefit)
```
"La dinosaŭro estis reptilio. Ĝi vivis antaŭ milionoj da jaroj."
→ Definition spans sentences
```

**3. Dialogue text** (medium benefit)
```
"—Kie vi loĝas? —Mi loĝas en Parizo."
→ Q&A relationships across sentences
```

**4. Lists/enumerations** (low benefit)
```
"Unue, mi kuiras. Due, mi manĝas. Trie, mi ripozas."
→ Weak semantic relationships between list items
```

### When to Use Paragraph Boundaries

**Include**:
- ✅ Consecutive sentences in same paragraph
- ✅ Sentences separated by 1-2 sentences
- ✅ Within same section/topic

**Exclude**:
- ❌ Cross-paragraph boundaries (topic shifts)
- ❌ Cross-section boundaries
- ❌ Cross-document boundaries

**Why**: Topic coherence breaks at paragraph boundaries

---

## ✅ Recommendation

**YES, implement cross-sentence context with Strategy 2 (weighted)**

**Why**:
1. ✅ Significant quality improvement (+20% average)
2. ✅ Modest computational cost (+20% training time)
3. ✅ Captures important discourse relationships
4. ✅ Particularly helps coreference and rare words
5. ✅ Kuzu database already has paragraph structure (Paragrafo nodes)

**Implementation priority**: **HIGH** (do this before training)

**Next steps**:
1. Query Kuzu for paragraph-level root sequences
2. Implement weighted cross-sentence pair generation
3. Run ablation study (baseline vs cross-sentence)
4. Use best model for Phase 1

**Want me to implement the paragraph-aware data extraction from Kuzu?**

