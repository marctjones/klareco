# Embedding Training Data Requirements & Quality

**Critical Questions Answered**:
1. How much training data do we need?
2. How much per word?
3. Do we need opposites?
4. How to ensure quality?
5. When does training stop?
6. What if embeddings collapse?

---

## 📊 Training Data Requirements

### How Much Total Data?

**Rule of thumb**: ~1000 training examples per vocabulary item

**For our 5,000 root vocabulary**:
- **Minimum**: 5 million training pairs
- **Recommended**: 50 million training pairs
- **Our corpus**: 1.6 billion training pairs ✅ (300x minimum!)

**Why we have so much**:
```python
5.4M sentences in corpus
× ~10 content roots per sentence (filtered from ~15 total words)
× ~5 context pairs per root (window_size=5)
× 5 negative samples per positive
= ~1.35 billion training pairs

Plus data augmentation (different window positions) = ~1.6B total
```

**Conclusion**: We have **abundant** training data (300x more than minimum)

---

### How Much Data Per Word?

**Frequency distribution matters!**

**Common words** (top 100):
```python
Root: 'est' (to be)
Frequency in corpus: ~500,000 occurrences
Training pairs: ~2.5 million positive + 12.5 million negative = 15M total
Status: ✅✅✅ Excellent coverage
```

**Medium frequency words** (100-1000):
```python
Root: 'hund' (dog)
Frequency in corpus: ~5,000 occurrences
Training pairs: ~25,000 positive + 125,000 negative = 150K total
Status: ✅✅ Good coverage
```

**Rare words** (1000-5000):
```python
Root: 'dinosaŭr' (dinosaur)
Frequency in corpus: ~200 occurrences
Training pairs: ~1,000 positive + 5,000 negative = 6K total
Status: ✅ Minimum coverage (but enough!)
```

**Critical threshold**: **~1,000 training pairs minimum per word**

**Below threshold**:
- <100 occurrences: Embedding will be noisy/unreliable
- <10 occurrences: Embedding essentially random
- Solution: Don't include in vocabulary (use fallback)

**How we handle this**:
```python
# Build vocabulary with frequency cutoff
root_counts = count_all_roots(corpus)

# Only keep roots with sufficient data
vocabulary = []
for root, count in root_counts.most_common():
    if count >= 100:  # Minimum 100 corpus occurrences
        vocabulary.append(root)

    if len(vocabulary) >= 5000:
        break

# Result: All 5,000 roots have ≥100 occurrences
# → Each has ≥500 training pairs ✅
```

---

## 🎯 Do We Need Opposites for Distance?

**Short answer**: No! Negative sampling creates distance automatically.

### How Negative Sampling Works

**Positive examples** (words that co-occur):
```python
Sentence: "La granda hundo bojlas."
Positive pairs:
- (grand, hund) → label=1  # "big dog" - they appear together
- (hund, bojl) → label=1   # "dog barks" - they appear together
```

**Negative examples** (words that DON'T co-occur):
```python
Random sampling:
- (hund, aeroplan) → label=0  # dog + airplane (never together)
- (hund, matematiko) → label=0  # dog + mathematics (never together)
- (grand, oceano) → label=0  # big + ocean (rarely together)
```

**What the model learns**:
```python
# Positive examples → pull embeddings CLOSER
loss_positive = -log(sigmoid(dot(hund, bojl)))  # Minimize when similar

# Negative examples → push embeddings APART
loss_negative = -log(1 - sigmoid(dot(hund, aeroplan)))  # Minimize when dissimilar

# Total loss = minimize both
```

**Result**: Model automatically creates distance between unrelated words!

### What About Antonyms?

**Interesting case**: Antonyms often appear in similar contexts!

```python
Sentence 1: "La granda hundo kuras."
Sentence 2: "La malgranda hundo kuras."

Co-occurrence:
- grand → hund, kur
- malgrand → hund, kur

Result: grand ≈ malgrand (similarity ~0.6)
```

**This is actually CORRECT semantically!**
- "grand" and "malgrand" are in the same semantic domain (size)
- They're used in similar contexts (describing objects)
- Similarity captures "semantic relatedness", not just "meaning identity"

**If we want to distinguish antonyms**, we'd need:
1. Explicit antonym annotations (not in corpus naturally)
2. Contrastive learning: Pull antonyms apart explicitly
3. **Not needed for Phase 1** - semantic relatedness is sufficient for retrieval

**Bottom line**: Negative sampling handles distance automatically. No need to manually encode opposites.

---

## ✅ How to Create Quality Training Data

### Quality Factors

**1. Clean Root Extraction** (Deterministic Parser)
```python
# Good: Parse correctly
sentence = "La hundo kuras rapide"
roots = ['hund', 'kur', 'rapid']  # ✅ All content roots

# Bad: Include function words
sentence = "La hundo kuras rapide"
roots = ['la', 'hund', 'kur', 'rapid']  # ❌ 'la' is function word
→ Causes embedding collapse!
```

**Quality check**:
```python
def validate_root_extraction():
    sample_sentences = corpus.sample(1000)

    for sent in sample_sentences:
        roots = extract_content_roots(sent)

        # Check: No function words
        function_words = {'la', 'de', 'en', 'kaj', 'aŭ', 'sed', 'pri'}
        assert not any(r in function_words for r in roots)

        # Check: All roots valid
        assert all(is_valid_esperanto_root(r) for r in roots)

        # Check: Reasonable length
        assert 2 <= len(roots) <= 20  # Typical sentence
```

**2. Appropriate Window Size**
```python
# Too small (window=1): Only immediate neighbors
sentence_roots = ['hund', 'grand', 'bojl', 'laŭt']
pairs_window1 = [(hund, grand), (grand, bojl), (bojl, laŭt)]
→ Misses: (hund, bojl), (hund, laŭt)  # Too restrictive!

# Too large (window=20): Unrelated words
sentence_roots = ['hund', 'bojl', 'laŭt', 'park', 'tago', 'sol', ...]
pairs_window20 = [(hund, sol), (bojl, tago), ...]  # Unrelated!
→ Noise: dog rarely relates to sun in meaningful way

# Optimal (window=5): Balance
sentence_roots = ['hund', 'grand', 'bojl', 'laŭt', 'park']
pairs_window5 = [(hund, bojl), (hund, laŭt), (bojl, park)]
→ Captures meaningful relationships ✅
```

**Quality check**:
```python
# Test different window sizes
for window_size in [2, 5, 10]:
    pairs = generate_pairs(corpus, window_size)

    # Sample and manually inspect
    sample = random.sample(pairs, 100)

    # Count "makes sense" pairs
    sensible = count_sensible_pairs(sample)
    print(f"Window {window_size}: {sensible}% sensible")

# Expected results:
# Window 2: 95% sensible (but misses relationships)
# Window 5: 85% sensible (good balance) ✅
# Window 10: 60% sensible (too much noise)
```

**3. Balanced Negative Sampling**
```python
# Too few negatives (ratio=1:1)
positive_pairs = 1000
negative_pairs = 1000
→ Model doesn't learn to distinguish unrelated words well

# Too many negatives (ratio=1:20)
positive_pairs = 1000
negative_pairs = 20000
→ Model overly cautious, underestimates similarities

# Optimal (ratio=1:5)
positive_pairs = 1000
negative_pairs = 5000
→ Good balance ✅ (standard in word2vec)
```

**4. Vocabulary Quality Control**
```python
def validate_vocabulary(vocabulary):
    issues = []

    for root in vocabulary:
        # Check 1: Valid Esperanto root
        if not is_valid_root(root):
            issues.append(f"{root}: Invalid root")

        # Check 2: Sufficient frequency
        count = corpus_frequency(root)
        if count < 100:
            issues.append(f"{root}: Only {count} occurrences (min 100)")

        # Check 3: Not a function word
        if root in FUNCTION_WORDS:
            issues.append(f"{root}: Function word (should exclude)")

        # Check 4: Not a number
        if root.isdigit():
            issues.append(f"{root}: Number (should exclude)")

    return issues
```

---

## 🛑 Training Stopping Criteria

### Early Stopping (Validation Loss)

**How it works**:
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience  # Epochs to wait
        self.min_delta = min_delta  # Minimum improvement
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, validation_loss):
        # Is validation loss improving?
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False  # Keep training ✅

        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training 🛑
            return False
```

**In practice**:
```python
# Training loop
early_stopping = EarlyStopping(patience=3)

for epoch in range(100):  # Max 100 epochs
    train_loss = train_one_epoch(model, train_data)
    val_loss = validate(model, validation_data)

    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    if early_stopping.should_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break  # 🛑

# Example output:
# Epoch 0: Train=0.6931, Val=0.6920
# Epoch 1: Train=0.5234, Val=0.5298  # Improving ✅
# Epoch 2: Train=0.4123, Val=0.4201  # Improving ✅
# Epoch 3: Train=0.3456, Val=0.3501  # Improving ✅
# Epoch 4: Train=0.2987, Val=0.3498  # Slight improvement ✅
# Epoch 5: Train=0.2634, Val=0.3503  # Validation worse ⚠️ (count=1)
# Epoch 6: Train=0.2301, Val=0.3509  # Still worse ⚠️ (count=2)
# Epoch 7: Train=0.2045, Val=0.3512  # Still worse ⚠️ (count=3)
# Early stopping at epoch 7 🛑
```

**Why we need this**: Prevents overfitting!

**Overfitting signs**:
- Training loss keeps decreasing
- Validation loss stops improving or increases
- Model memorizes training data but doesn't generalize

---

## 🚨 Embedding Collapse Detection

### What is Embedding Collapse?

**Problem**: All embeddings become too similar (lose distinctiveness)

```python
# Healthy embeddings (diverse)
embeddings['hund'] = [0.8, -0.3, 0.1, ...]   # Different
embeddings['kat']  = [0.7, -0.2, 0.2, ...]   # Slightly similar
embeddings['aeroplan'] = [-0.2, 0.9, -0.4, ...]  # Very different

cosine_similarity(hund, kat) = 0.78      # Similar (both animals) ✅
cosine_similarity(hund, aeroplan) = 0.05  # Different (dog vs plane) ✅

# Collapsed embeddings (all similar)
embeddings['hund'] = [0.5, 0.3, 0.2, ...]   # Similar
embeddings['kat']  = [0.5, 0.3, 0.2, ...]   # Too similar!
embeddings['aeroplan'] = [0.5, 0.3, 0.2, ...]  # Too similar!

cosine_similarity(hund, kat) = 0.99       # TOO similar ❌
cosine_similarity(hund, aeroplan) = 0.95  # TOO similar ❌
```

### Causes of Collapse

**1. Including function words**:
```python
# Bad: Include 'la', 'de', 'en' (function words)
roots = ['la', 'hund', 'de', 'kat', 'en', 'dom']

# These appear everywhere!
# 'la' appears with EVERY noun
# 'de' appears with EVERY possessive
# → Model learns: "everything is similar to everything" ❌
```

**2. Too high learning rate**:
```python
# Bad: lr=0.1 (too high)
optimizer = Adam(lr=0.1)
→ Large updates push all embeddings toward average

# Good: lr=0.001 (standard)
optimizer = Adam(lr=0.001)
→ Gradual updates preserve differences ✅
```

**3. Insufficient negative samples**:
```python
# Bad: Only 1 negative per positive
positive: (hund, bojl) → label=1
negative: (hund, random1) → label=0
→ Model doesn't learn strong distinctions

# Good: 5 negatives per positive
positive: (hund, bojl) → label=1
negatives: (hund, random1), (hund, random2), ... (5 total)
→ Model learns: "hund is NOT like most random things" ✅
```

### Collapse Detection

```python
class CollapseDetector:
    def __init__(self, threshold=0.8):
        self.threshold = threshold  # Mean similarity threshold

    def check_collapse(self, embeddings):
        # Sample 1000 random pairs
        pairs = sample_random_pairs(embeddings, n=1000)

        # Compute similarities
        similarities = []
        for emb1, emb2 in pairs:
            sim = cosine_similarity(emb1, emb2)
            similarities.append(sim)

        mean_sim = np.mean(similarities)

        # Healthy: mean ~0.2-0.4 (mostly unrelated)
        # Collapsed: mean >0.7 (everything similar)

        if mean_sim > self.threshold:
            return True, mean_sim  # Collapsed! 🚨
        else:
            return False, mean_sim  # Healthy ✅
```

**In training loop**:
```python
collapse_detector = CollapseDetector(threshold=0.7)

for epoch in range(100):
    train_one_epoch(model, train_data)

    # Check for collapse every 5 epochs
    if epoch % 5 == 0:
        collapsed, mean_sim = collapse_detector.check_collapse(
            model.embeddings.weight.data
        )

        print(f"Epoch {epoch}: Mean similarity = {mean_sim:.3f}")

        if collapsed:
            print(f"🚨 COLLAPSE DETECTED! Mean sim = {mean_sim:.3f}")
            print("Stopping training and reverting to previous checkpoint")
            model.load_state_dict(best_checkpoint)
            break  # 🛑

# Example output:
# Epoch 0: Mean similarity = 0.05  # Random initialization
# Epoch 5: Mean similarity = 0.23  # Learning ✅
# Epoch 10: Mean similarity = 0.31  # Healthy ✅
# Epoch 15: Mean similarity = 0.38  # Still good ✅
# Epoch 20: Mean similarity = 0.42  # Acceptable ✅
# Epoch 25: Mean similarity = 0.73  # 🚨 COLLAPSE!
# Stopping training and reverting to epoch 20 checkpoint
```

### Preventing Collapse

**1. Filter function words** (CRITICAL)
```python
FUNCTION_WORDS = {
    'la', 'de', 'al', 'en', 'sur', 'sub',  # Prepositions/articles
    'kaj', 'aŭ', 'sed', 'ĉar',             # Conjunctions
    'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni',   # Pronouns
    'estas', 'estis', 'estos',             # Copula variations
}

def extract_content_roots(sentence_ast):
    roots = []
    for word in sentence_ast['words']:
        root = word['radiko']
        pos = word['vortspeco']

        # Filter function words
        if root in FUNCTION_WORDS:
            continue  # Skip! ✅

        # Keep only content words
        if pos in ['substantivo', 'verbo', 'adjektivo', 'adverbo']:
            roots.append(root)

    return roots
```

**2. Use appropriate learning rate**
```python
# Start with standard rate
optimizer = Adam(model.parameters(), lr=0.001)

# Reduce if training is unstable
if mean_similarity_increasing_too_fast:
    optimizer = Adam(model.parameters(), lr=0.0001)
```

**3. Sufficient negative sampling**
```python
# Standard ratio: 5 negatives per positive
num_negatives = 5  # ✅ Proven to work well
```

**4. Monitor and checkpoint**
```python
# Save checkpoint every epoch
if val_loss < best_val_loss:
    torch.save(model.state_dict(), 'checkpoint_best.pt')
    best_val_loss = val_loss

# Can revert if collapse detected
if collapse_detected:
    model.load_state_dict(torch.load('checkpoint_best.pt'))
```

---

## 📈 Training Monitoring Dashboard

### What to Track

```python
class TrainingMonitor:
    def log_metrics(self, epoch):
        metrics = {
            # Loss metrics
            'train_loss': train_loss,
            'val_loss': val_loss,

            # Collapse metrics
            'mean_similarity': self.compute_mean_similarity(),
            'std_similarity': self.compute_std_similarity(),

            # Quality metrics (sample evaluation)
            'synonym_accuracy': self.test_synonyms(),
            'analogy_accuracy': self.test_analogies(),

            # Training health
            'gradient_norm': self.compute_gradient_norm(),
            'learning_rate': self.get_current_lr(),
        }

        # Warning triggers
        if metrics['mean_similarity'] > 0.7:
            print("⚠️  WARNING: Possible collapse")

        if metrics['gradient_norm'] < 0.0001:
            print("⚠️  WARNING: Vanishing gradients")

        if metrics['gradient_norm'] > 10.0:
            print("⚠️  WARNING: Exploding gradients")

        return metrics
```

**Healthy training output**:
```
Epoch 0: Loss=0.693 | MeanSim=0.05 | SynAcc=10% | GradNorm=2.3
Epoch 1: Loss=0.523 | MeanSim=0.18 | SynAcc=35% | GradNorm=1.8
Epoch 2: Loss=0.412 | MeanSim=0.27 | SynAcc=58% | GradNorm=1.2
Epoch 3: Loss=0.346 | MeanSim=0.33 | SynAcc=71% | GradNorm=0.9
Epoch 4: Loss=0.299 | MeanSim=0.38 | SynAcc=79% | GradNorm=0.7
Epoch 5: Loss=0.265 | MeanSim=0.42 | SynAcc=84% | GradNorm=0.5
Epoch 6: Loss=0.241 | MeanSim=0.44 | SynAcc=87% | GradNorm=0.4
Epoch 7: Loss=0.235 | MeanSim=0.45 | SynAcc=88% | GradNorm=0.4
Early stopping at epoch 7 (validation not improving) ✅
```

---

## ✅ Quality Checklist

**Before training**:
- [ ] Function words filtered out
- [ ] Vocabulary ≥100 occurrences per root
- [ ] Window size = 5 (or tested optimal)
- [ ] Negative sampling ratio = 1:5
- [ ] Validation set held out (10% of data)

**During training**:
- [ ] Early stopping configured (patience=3)
- [ ] Collapse detection active (check every 5 epochs)
- [ ] Checkpointing enabled (save best model)
- [ ] Metrics logged (loss, similarity, quality)

**After training**:
- [ ] Mean similarity < 0.5 (no collapse)
- [ ] Synonym accuracy > 70%
- [ ] Analogy accuracy > 50%
- [ ] Validation loss converged
- [ ] Manual inspection: 20 random embeddings make sense

---

## 🎯 Summary

| Question | Answer |
|----------|--------|
| **How much data total?** | 1.6B pairs ✅ (300x minimum) |
| **How much per word?** | Min 1K pairs, we have 500-15M per word ✅ |
| **Need opposites?** | No, negative sampling handles distance ✅ |
| **Quality training data?** | Filter function words, window=5, ratio 1:5 ✅ |
| **Training stops when?** | Early stopping: 3 epochs no improvement ✅ |
| **Collapse detection?** | Yes, monitor mean similarity < 0.5 ✅ |

**We have excellent conditions for training**:
- ✅ Abundant data (1.6B pairs)
- ✅ Good coverage (all roots ≥100 occurrences)
- ✅ Deterministic parser filters function words
- ✅ Standard hyperparameters (proven to work)
- ✅ Safety mechanisms (early stop + collapse detection)

**Expected outcome**: High-quality embeddings in ~7-10 epochs (2 days GPU)

