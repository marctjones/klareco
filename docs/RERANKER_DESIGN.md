# Reranker Architecture Design

## Overview

A lightweight query-document relevance reranker for Klareco's AST-aware retrieval system.

**Goal**: Learn to score (query, document) pairs by relevance, focusing on query intent understanding.

**Key principle**: Use existing compositional embeddings (no new encoding parameters), only learn interaction/scoring layer (~10-20M params).

---

## Architecture

### Input Representation

```python
# Both query and document are parsed ASTs
query_ast = parse("Kio estas hundo?")
doc_ast = parse("Hundo estas dombesto...")

# Use existing CompositionalEmbedding (320K params, already trained)
query_emb = encode_ast(query_ast)    # 128d vector
doc_emb = encode_ast(doc_ast)        # 128d vector
```

### Interaction Features

Combine query and document representations to capture relevance patterns:

```python
# Core features
features = [
    query_emb,                           # 128d - query representation
    doc_emb,                             # 128d - document representation
    query_emb * doc_emb,                 # 128d - element-wise product (interaction)
    abs(query_emb - doc_emb),           # 128d - absolute difference
]

# Additional AST-based features (optional)
structural_features = [
    question_type_encoding,              # 8d - WHO/WHAT/WHERE/WHEN/WHY/HOW
    has_shared_verb,                     # 1d - query and doc share verb root
    has_shared_subject,                  # 1d - query and doc share subject
    root_overlap_ratio,                  # 1d - % of query roots in doc
    doc_has_definition_pattern,          # 1d - "X estas Y" pattern (for "Kio estas X?")
]

total_features = concat(features, structural_features)  # ~520d
```

### Scoring Network

Small MLP for relevance scoring:

```python
class RelevanceScorer(nn.Module):
    def __init__(self, input_dim=520, hidden_dim=256):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 520 -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),  # 256 -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),         # 256 -> 128
            nn.ReLU(),
            nn.Linear(128, 1),                  # 128 -> 1 (relevance score)
            nn.Sigmoid()                        # [0, 1] relevance
        )

    def forward(self, features):
        return self.layers(features)
```

**Parameter count**: ~180K params (tiny!)

---

## Training Data Generation

### Strategy 1: Automatic Pattern Mining (Highest Quality)

Mine query-answer pairs from corpus based on patterns:

**Definition Queries** ("Kio estas X?"):
```python
# Find sentences with "X estas Y" pattern
patterns = [
    "X estas Y",           # X is Y
    "X, kiu estas Y",     # X, which is Y
    "Y nomata X",          # Y called X
]

# Example mined pairs:
query: "Kio estas hundo?"
doc+: "Hundo estas dombesto kun lojala karaktero" (score: 1.0)
doc-: "La hundoj havas utilajn funkciojn" (score: 0.3)
```

**Factual Queries** ("Kiu fondis X?"):
```python
# Find subject-verb-object triples matching query
patterns = [
    "Subjekto Verbo Objekton",
    "Objekto estas Verbo-ita de Subjekto",
]

# Example:
query: "Kiu fondis Esperanton?"
doc+: "Zamenhof fondis Esperanton en 1887" (score: 1.0)
doc-: "Esperanto estas planlingvo" (score: 0.2)
```

### Strategy 2: BM25-Based Pseudo-Labeling

Use current retriever scores as weak labels:

```python
# For each query in corpus:
# - Top-5 results: positive examples (score: 0.7-1.0)
# - Random samples: negative examples (score: 0.0-0.3)
# - Middle results: medium relevance (score: 0.3-0.7)

# Add noise to avoid overfitting to BM25:
relevance_score = bm25_score * random.uniform(0.8, 1.2)
```

### Strategy 3: Synthetic Query Generation

Generate queries from corpus sentences:

```python
# From: "Hundo estas dombesto"
# Generate:
# - "Kio estas hundo?" (definition)
# - "Ĉu hundo estas dombesto?" (yes/no)
# - "Kia estas hundo?" (description)

def generate_definition_query(sentence_ast):
    subject = extract_subject(sentence_ast)
    if sentence_ast['verbo']['radiko'] == 'est':
        return f"Kio estas {subject}?"
```

### Data Split

```
Training:   80% of generated pairs (~50K pairs)
Validation: 10% (~6K pairs)
Test:       10% (~6K pairs, held out for evaluation)
```

---

## Loss Function

### Option 1: Pointwise (Binary Cross-Entropy)

Simple classification approach:

```python
loss = BCELoss(
    predicted_relevance,
    target_relevance  # [0.0, 1.0]
)
```

**Pros**: Simple, works well
**Cons**: Doesn't directly optimize ranking

### Option 2: Pairwise (Margin Ranking Loss)

Learn relative ordering:

```python
# Given query q, positive doc d+, negative doc d-
score_pos = model(q, d+)
score_neg = model(q, d-)

loss = max(0, margin - (score_pos - score_neg))
```

**Pros**: Directly optimizes ranking
**Cons**: Requires paired data

### Option 3: Listwise (Softmax Cross-Entropy)

Rank entire result list:

```python
# Given query q and docs [d1, d2, ..., dk] with relevance [r1, r2, ..., rk]
scores = [model(q, di) for di in docs]
loss = CrossEntropyLoss(softmax(scores), relevance_distribution)
```

**Recommendation**: Start with **Option 1 (BCE)** for simplicity, switch to **Option 2 (pairwise)** if needed.

---

## Training Procedure

```python
# Hyperparameters
batch_size = 256
learning_rate = 1e-3
epochs = 20
early_stopping_patience = 3

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        query_ast, doc_ast, relevance = batch

        # Encode (frozen compositional embedding)
        with torch.no_grad():
            query_emb = compositional_emb.encode_ast(query_ast)
            doc_emb = compositional_emb.encode_ast(doc_ast)

        # Build features
        features = build_features(query_emb, doc_emb, query_ast, doc_ast)

        # Score
        predicted = model(features)

        # Loss
        loss = bce_loss(predicted, relevance)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Evaluation Metrics

### Offline Metrics (Test Set)

1. **NDCG@5**: Normalized Discounted Cumulative Gain
   - Measures ranking quality
   - Target: NDCG@5 > 0.75

2. **MRR**: Mean Reciprocal Rank
   - Average rank of first relevant result
   - Target: MRR > 0.60

3. **Precision@1**: Is top result relevant?
   - Target: P@1 > 0.70

### Online Metrics (Manual Evaluation)

Create test queries for each question type:

```python
test_queries = [
    # Definition queries (Kio estas X?)
    "Kio estas hundo?",
    "Kio estas Esperanto?",
    "Kio estas scienco?",

    # Who queries (Kiu...)
    "Kiu fondis Esperanton?",
    "Kiu verkis Hamleto?",

    # Where queries (Kie...)
    "Kie loĝas tigro?",
    "Kie naskiĝis Zamenhof?",

    # When queries (Kiam...)
    "Kiam naskiĝis Zamenhof?",

    # Why queries (Kial...)
    "Kial la ĉielo estas blua?",
]
```

**Manual labels**: For each query, annotate top-10 results as:
- 0 = Not relevant
- 1 = Somewhat relevant
- 2 = Highly relevant

---

## Implementation Plan

### Phase 1: Data Preparation (1-2 days)

```bash
# 1. Mine definition pairs from corpus
python scripts/generate_reranker_training_data.py \
    --corpus data/enhanced_corpus/corpus_with_metadata.jsonl \
    --output data/training/reranker/ \
    --strategy definition_mining \
    --num_samples 30000

# 2. Generate synthetic queries
python scripts/generate_reranker_training_data.py \
    --strategy synthetic_queries \
    --num_samples 20000

# 3. Combine datasets
python scripts/merge_training_data.py \
    --output data/training/reranker/train.jsonl
```

### Phase 2: Model Training (1 day)

```bash
# Train reranker
python scripts/train_reranker.py \
    --train-data data/training/reranker/train.jsonl \
    --val-data data/training/reranker/val.jsonl \
    --compositional-model models/root_embeddings/best_model.pt \
    --output models/reranker/ \
    --batch-size 256 \
    --epochs 20
```

### Phase 3: Integration & Evaluation (1 day)

```bash
# Integrate into retriever
# Add reranking step in ast_aware_retriever.py

# Evaluate
python scripts/evaluate_reranker.py \
    --test-queries data/evaluation/test_queries.json \
    --reranker models/reranker/best_model.pt

# Compare baseline vs reranker
python scripts/demo_ast_retriever.py "Kio estas hundo?" --top-k 5  # baseline
python scripts/demo_ast_retriever.py "Kio estas hundo?" --fallback rerank --top-k 5  # with reranker
```

---

## File Structure

```
klareco/
├── models/
│   └── reranker.py                      # RelevanceScorer model
├── training/
│   ├── reranker_data_generator.py       # Data mining & generation
│   └── reranker_trainer.py              # Training loop
├── rag/
│   └── ast_aware_retriever.py           # Integration point

scripts/
├── generate_reranker_training_data.py   # CLI for data generation
├── train_reranker.py                    # CLI for training
└── evaluate_reranker.py                 # Evaluation script

data/
├── training/
│   └── reranker/
│       ├── train.jsonl                  # Training pairs
│       ├── val.jsonl                    # Validation pairs
│       └── test.jsonl                   # Test pairs
└── evaluation/
    └── test_queries.json                # Manual evaluation queries

models/
└── reranker/
    ├── best_model.pt                    # Trained reranker
    ├── config.json                      # Model config
    └── training.log                     # Training metrics
```

---

## Expected Results

### Before Reranker (Current)
```
Query: "Kio estas hundo?"
1. [8.079] "...rusa havas nomon por virhundo..." (naming conventions) ❌
2. [8.079] "...afrika natura hundo..." (wild dogs) ❌
3. [7.886] "...ĉashundoj, leporhundoj..." (dog breeds) ❌
```

### After Reranker (Expected)
```
Query: "Kio estas hundo?"
1. [0.95] "Hundo estas dombesto kun lojala karaktero..." (definition) ✓
2. [0.88] "Hundo apartenas al familio Canidae..." (taxonomy) ✓
3. [0.72] "Hundoj estas uzataj por ĉasado kaj gardo..." (uses) ✓
```

---

## Parameter Budget

```
Existing (frozen):
  CompositionalEmbedding: 320K params

New (trainable):
  RelevanceScorer MLP:    180K params

Total learned params:     500K params  ✓ (stays well under 1M!)
```

This aligns perfectly with the "Pure Esperanto AI" thesis - we're learning **reasoning** (what makes a document relevant), not grammar!

---

## Next Steps

1. Create data generation script (`generate_reranker_training_data.py`)
2. Implement RelevanceScorer model (`klareco/models/reranker.py`)
3. Create training script (`train_reranker.py`)
4. Generate training data (~50K pairs)
5. Train for 20 epochs (~1 hour on GPU, 3 hours on CPU)
6. Integrate into retriever
7. Evaluate on test queries

**Estimated time**: 3-4 days total

Want me to start implementing? I can begin with the data generation script.
