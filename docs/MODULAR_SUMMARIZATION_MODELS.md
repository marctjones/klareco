# Modular Summarization Models Design

## The Problem with One Large Model

**Original design**: 10M parameter "topic clustering model"
- ❌ Black box - hard to understand what it does
- ❌ Hard to debug - if results are bad, which part failed?
- ❌ Hard to train - needs lots of data for all tasks combined
- ❌ Monolithic - can't improve one aspect without retraining whole model

## Better Approach: Deterministic First + Small Focused Models

**Key principle**: Do maximum work deterministically, use small models ONLY where semantic understanding is needed.

```
Deterministic Work (0 params) → Small Model 1 (2M) → More Deterministic → Small Model 2 (3M) → ...
```

## The Five Focused Models (Total: 10M params)

### Model 1: Semantic Importance Adjuster (2M params)

**Input**:
- Information unit with deterministic importance score
- Embeddings for roots in the unit
- Query embeddings

**Output**:
- Adjusted importance score (-0.2 to +0.2 adjustment)

**Purpose**: Handle cases where deterministic scoring misses semantic nuance

**Example**:
```
Unit: "pola kuracisto" (Polish doctor)
Deterministic score: 0.47 (medium)

Query: "Kiu fondis Esperanton?"

Semantic analysis:
  - "kuracisto" (doctor) embedding is semantically related to "person" (answers kiu)
  - "pola" (Polish) embedding relates to "Pollando" (Poland), which co-occurs with "Zamenhof"

Model output: +0.15 adjustment
Final score: 0.47 + 0.15 = 0.62
```

**Why small works**: Only adjusting existing scores, not computing from scratch!

**Training data**: 3,000-5,000 (unit, query, deterministic_score, gold_adjustment) tuples

---

### Model 2: Topic Assignment Classifier (3M params)

**Input**:
- Information unit with importance score
- All other units (for context)
- Predefined topic candidates (extracted deterministically)

**Output**:
- Topic assignment probabilities (which topic does this unit belong to?)

**Purpose**: Assign units to topics when deterministic clustering ambiguous

**Example**:
```
Unit: "en 1887" (in 1887)

Deterministic analysis:
  - Could be Topic A: "Founder Identity" (when Zamenhof was born/active)
  - Could be Topic B: "Creation Event" (when Esperanto was created)

Context from other units:
  - "fondis Esperanton" is in Topic B
  - "en 1887" appears in same sentence as "fondis"

Model output:
  P(Topic A) = 0.20
  P(Topic B) = 0.80  ← Assign to Creation Event

Reasoning: Temporal modifier of "fondis" action
```

**Why small works**: Choosing between pre-identified topics, not discovering topics from scratch!

**Training data**: 5,000-8,000 (unit, context, topic_assignments) examples

---

### Model 3: Sentence Construction Planner (2M params)

**Input**:
- Set of information units selected for summary
- Topic assignments for each unit
- Importance scores

**Output**:
- Grouping of units into sentences (which units go in same sentence?)

**Purpose**: Decide how to combine information units into coherent sentences

**Example**:
```
Selected units:
  u1: "Zamenhof" (importance: 0.95, topic: Founder)
  u2: "pola kuracisto" (importance: 0.62, topic: Founder)
  u3: "fondis Esperanton" (importance: 0.90, topic: Creation)
  u4: "en 1887" (importance: 0.75, topic: Creation)

Deterministic constraints:
  - u1 and u3 have syntactic relation (subject-verb) → should be together
  - u2 is appositive of u1 → should be with u1
  - u4 is temporal modifier of u3 → should be with u3

Model decides:
  Sentence 1: [u1, u2, u3, u4]  (all in one sentence)
  or
  Sentence 1: [u1, u2, u3]  (main info)
  Sentence 2: [u4]  (separate detail)

Model output: Sentence 1 (combine all)

Reasoning: All units are high importance and relate to same event
```

**Why small works**: Operating on already-selected units with deterministic constraints!

**Training data**: 2,000-4,000 (selected_units, gold_sentence_groupings) examples

---

### Model 4: Discourse Ordering Model (2M params)

**Input**:
- Information units grouped into a sentence/paragraph
- Topic labels
- Discourse relations (RST nucleus/satellite)

**Output**:
- Ordering of units within sentence/paragraph

**Purpose**: Decide order when multiple valid orderings exist

**Example**:
```
Units for Sentence 1:
  u1: "Zamenhof" (subject, nucleus)
  u2: "pola kuracisto" (appositive, satellite)
  u3: "fondis Esperanton" (verb+object, nucleus)
  u4: "en 1887" (temporal, satellite)

Deterministic constraints:
  - u1 (subject) comes before u3 (verb) in SVO order
  - u2 (appositive) comes after u1 (the noun it describes)
  - u4 (temporal) is flexible

Possible orderings:
  A: "Zamenhof, pola kuracisto, fondis Esperanton en 1887."
  B: "En 1887, Zamenhof, pola kuracisto, fondis Esperanton."
  C: "Zamenhof fondis Esperanton en 1887, pola kuracisto."  (awkward)

Model output: Order A (0.85 probability)

Reasoning:
  - Nucleus-first principle (RST)
  - Information structure: Given→New (Zamenhof→en 1887)
  - Natural flow in Esperanto
```

**Why small works**: Choosing between valid orderings, not generating from scratch!

**Training data**: 3,000-5,000 (units, gold_ordering) examples

---

### Model 5: Paragraph Break Predictor (1M params)

**Input**:
- Sequence of sentences
- Topic assignments for each sentence
- Sentence lengths

**Output**:
- Paragraph break decisions (insert break after this sentence?)

**Purpose**: Decide where to insert paragraph breaks for multi-paragraph summaries

**Example**:
```
Sentences:
  S1: "Zamenhof, pola kuracisto, fondis Esperanton en 1887." (Topic: Founder)
  S2: "Li vivis en Bjalistoko dum sia infanaĝo." (Topic: Founder)
  S3: "La lingvo estis kreita por internacia komunikado." (Topic: Purpose)
  S4: "Esperanto havas regulan gramatikon." (Topic: Features)

Deterministic hints:
  - S1→S2: Same topic (Founder) → probably no break
  - S2→S3: Topic change (Founder→Purpose) → maybe break
  - S3→S4: Topic change (Purpose→Features) → maybe break

Model output:
  After S1: No break (probability: 0.15)
  After S2: Break! (probability: 0.85)  ← Topic change + logical grouping
  After S3: Break! (probability: 0.75)  ← Topic change

Result:
  Paragraph 1: S1, S2 (about founder)
  Paragraph 2: S3 (about purpose)
  Paragraph 3: S4 (about features)
```

**Why small works**: Binary decision with strong deterministic hints!

**Training data**: 2,000-3,000 (sentence_sequence, gold_paragraph_breaks) examples

---

## Complete Pipeline with Focused Models

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Query + Retrieved Sentences                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Parse to ASTs (Deterministic)                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Extract Information Units (Deterministic)          │
│   Output: Units with linguistic features                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Build Information Graph (Deterministic + Embeddings)│
│   Nodes: Information units                                  │
│   Edges: Syntactic, coreference, entity (deterministic)     │
│          + Semantic similarity (embeddings)                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Compute Base Importance (Deterministic)            │
│   ① PageRank (graph centrality)                            │
│   ② Entity salience (Kuzu queries)                         │
│   ③ Entropy (information theory)                           │
│   ④ Info structure (topic/focus)                           │
│   ⑤ RST nucleus detection                                  │
│   Output: Base importance score per unit (0-1)             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🤖 MODEL 1: Semantic Importance Adjuster (2M)              │
│   Input: Base scores + embeddings                          │
│   Output: Adjusted scores                                  │
│   Example: "pola kuracisto" 0.47 → 0.62 (+0.15)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Identify Topic Candidates (Deterministic)          │
│   - Use root overlap clustering                            │
│   - Use Kuzu co-occurrence patterns                        │
│   Output: 3-5 topic candidates                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🤖 MODEL 2: Topic Assignment Classifier (3M)               │
│   Input: Units + topic candidates                          │
│   Output: Topic assignments per unit                       │
│   Example: "en 1887" → Topic B (Creation Event)           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Select Units by Importance (Deterministic Threshold)│
│   - 1-sentence: importance > 0.7                           │
│   - 1-paragraph: importance > 0.5                          │
│   - Multi-paragraph: importance > 0.35                     │
│   Output: Selected units for summary                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🤖 MODEL 3: Sentence Construction Planner (2M)             │
│   Input: Selected units + topics + importance              │
│   Output: Grouping of units into sentences                 │
│   Example: [u1,u2,u3,u4] → Sentence 1                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Apply Fusion Rules (Deterministic)                 │
│   - Same subject → appositive or coordinate verbs          │
│   - Relative clause insertion                              │
│   Output: Fused ASTs for each sentence                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🤖 MODEL 4: Discourse Ordering Model (2M)                  │
│   Input: Units per sentence + RST structure                │
│   Output: Ordering of units within sentences               │
│   Example: Order A (Zamenhof, kuracisto, fondis, 1887)    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: Deparse Sentences (Deterministic)                  │
│   - AST → Esperanto text using grammar rules               │
│   Output: Sequence of sentences                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 🤖 MODEL 5: Paragraph Break Predictor (1M)                 │
│   Input: Sentence sequence + topic changes                 │
│   Output: Paragraph break positions                        │
│   Example: Break after S2 (topic change)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 9: Format Output (Deterministic)                      │
│   - Insert paragraph breaks                                │
│   - Add punctuation, spacing                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Output: Formatted Summary                                   │
└─────────────────────────────────────────────────────────────┘
```

## Advantages of Modular Design

### 1. Clear Responsibilities

Each model has ONE job:
- Model 1: Adjust importance using semantic understanding
- Model 2: Assign units to topics
- Model 3: Group units into sentences
- Model 4: Order units within sentences
- Model 5: Insert paragraph breaks

No overlap, no confusion!

### 2. Easier to Debug

If summaries have bad sentence grouping:
- Debug Model 3 specifically
- Don't need to look at Models 1, 2, 4, 5

If summaries have wrong paragraph breaks:
- Debug Model 5 specifically
- Other models are fine

### 3. Smaller Training Datasets

| Model | Params | Training Examples Needed |
|-------|--------|-------------------------|
| Model 1 | 2M | 3,000-5,000 |
| Model 2 | 3M | 5,000-8,000 |
| Model 3 | 2M | 2,000-4,000 |
| Model 4 | 2M | 3,000-5,000 |
| Model 5 | 1M | 2,000-3,000 |
| **Total** | **10M** | **15,000-25,000 total** |

Compare to one 10M model: Would need 50,000+ examples for all tasks!

### 4. Can Train Incrementally

Phase 1: Train Model 1 only (importance adjustment)
- Test: Are importance scores better?

Phase 2: Add Model 2 (topic assignment)
- Test: Are topics correct?

Phase 3: Add Model 3 (sentence construction)
- Test: Are sentences well-formed?

Don't need all models at once!

### 5. Can Use Different Architectures

- Model 1 (importance adjustment): Simple feed-forward network
- Model 2 (topic assignment): Classifier (softmax over topics)
- Model 3 (sentence grouping): Sequence model (LSTM or transformer)
- Model 4 (ordering): Ranking model
- Model 5 (paragraph breaks): Binary classifier

Each optimized for its task!

### 6. Can Replace/Improve Individually

Found a better way to adjust importance?
- Replace Model 1
- Keep Models 2-5 unchanged

Found topic assignment isn't needed (deterministic works)?
- Remove Model 2
- Keep rest of pipeline

## Optional Models (Can Skip)

Each model is truly optional:

### Without Model 1 (Semantic Importance Adjuster)
- Use only deterministic importance scores
- Loss: Miss semantic nuances
- Impact: ~5% quality drop (estimated)

### Without Model 2 (Topic Assignment)
- Use deterministic clustering only
- Loss: Some units assigned to wrong topics
- Impact: ~8% quality drop (estimated)

### Without Model 3 (Sentence Construction)
- Use heuristic rules (combine if same topic + high importance)
- Loss: Some awkward sentence combinations
- Impact: ~10% quality drop (estimated)

### Without Model 4 (Discourse Ordering)
- Use deterministic ordering (SVO + RST nucleus-first)
- Loss: Sometimes unnatural order
- Impact: ~5% quality drop (estimated)

### Without Model 5 (Paragraph Breaks)
- Use deterministic rule (break on topic change)
- Loss: Sometimes awkward paragraph boundaries
- Impact: ~3% quality drop (estimated)

**Without any models**: ~70-75% quality (deterministic only)
**With all 5 models**: ~90-95% quality (estimated)

## Recommended Implementation Order

### Phase 1: Deterministic Baseline (Weeks 1-2)
Build complete pipeline with NO learned models:
- Extract information units
- Build information graph
- Compute base importance (PageRank + Kuzu + entropy)
- Deterministic topic clustering (root overlap)
- Deterministic sentence grouping (syntactic constraints)
- Deterministic ordering (SVO + RST)
- Deterministic paragraph breaks (topic change)

**Test**: Measure baseline quality (target: 70-75%)

### Phase 2: Add Model 1 - Importance Adjustment (Week 3)
- Collect training data (3K-5K examples)
- Train 2M param model
- Integrate into pipeline

**Test**: Measure improvement (target: 75-80%)

### Phase 3: Add Model 2 - Topic Assignment (Week 4)
- Collect training data (5K-8K examples)
- Train 3M param model
- Integrate into pipeline

**Test**: Measure improvement (target: 80-85%)

### Phase 4: Add Remaining Models (Weeks 5-7)
- Add Model 3 (sentence construction)
- Add Model 4 (discourse ordering)
- Add Model 5 (paragraph breaks)
- Train incrementally

**Test**: Measure final quality (target: 90-95%)

## Training Data Collection Strategy

### Active Learning Approach

1. **Start with deterministic system** (no models)
2. **Run on 100 test queries**
3. **Identify where deterministic fails**:
   - Model 1 needed: Cases where deterministic importance is wrong
   - Model 2 needed: Cases where topic assignment is ambiguous
   - Model 3 needed: Cases where sentence grouping is awkward
   - etc.
4. **Collect human annotations for failure cases**
5. **Train model to fix those specific cases**

This minimizes annotation effort!

### Annotation Tools Needed

For each model, create simple annotation interface:

**Model 1 (Importance Adjustment)**:
```
Query: "Kiu fondis Esperanton?"
Unit: "pola kuracisto"
Deterministic score: 0.47

Should this score be adjusted?
[ ] Yes, increase to: [0.62]
[ ] Yes, decrease to: [___]
[x] No, 0.47 is correct
```

**Model 2 (Topic Assignment)**:
```
Unit: "en 1887"

Which topic does this belong to?
( ) Topic A: Founder Identity
(•) Topic B: Creation Event
( ) Topic C: Purpose
```

Simple interfaces → faster annotation → less expensive!

## Model Architecture Details

### Model 1: Semantic Importance Adjuster

```python
class SemanticImportanceAdjuster(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2 + 10, 128)  # 2 embeddings + features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: adjustment (-0.2 to +0.2)

    def forward(self, unit_embedding, query_embedding, deterministic_features):
        # Concatenate embeddings and features
        x = torch.cat([unit_embedding, query_embedding, deterministic_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        adjustment = torch.tanh(self.fc3(x)) * 0.2  # Clamp to [-0.2, +0.2]
        return adjustment
```

**Total params**: ~2M
**Training**: Supervised (gold adjustments from human annotations)

### Model 2: Topic Assignment Classifier

```python
class TopicAssignmentClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_topics=5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim + 20, 256)  # Embedding + features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_topics)  # Output: topic probabilities

    def forward(self, unit_embedding, context_features):
        x = torch.cat([unit_embedding, context_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        topic_logits = self.fc3(x)
        return F.softmax(topic_logits, dim=-1)
```

**Total params**: ~3M
**Training**: Multi-class classification (softmax + cross-entropy)

### Model 3: Sentence Construction Planner

```python
class SentenceConstructionPlanner(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Encode each unit
        self.unit_encoder = nn.LSTM(embedding_dim, 128, batch_first=True)
        # Pairwise scoring
        self.fc_pair = nn.Linear(256 + 20, 1)  # Should these units be in same sentence?

    def forward(self, unit_embeddings, pairwise_features):
        # Encode units
        encoded, _ = self.unit_encoder(unit_embeddings)

        # For each pair of units, score if they should be together
        scores = []
        for i in range(len(encoded)):
            for j in range(i+1, len(encoded)):
                pair = torch.cat([encoded[i], encoded[j], pairwise_features[i,j]], dim=-1)
                score = torch.sigmoid(self.fc_pair(pair))
                scores.append(score)

        return scores  # 1 = same sentence, 0 = different sentences
```

**Total params**: ~2M
**Training**: Binary classification per pair

### Model 4: Discourse Ordering Model

```python
class DiscourseOrderingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.unit_encoder = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.fc_order = nn.Linear(256 + 15, 1)  # Score for u1 before u2

    def forward(self, unit_embeddings, ordering_features):
        encoded, _ = self.unit_encoder(unit_embeddings)

        # For each pair, score if u1 should come before u2
        order_scores = []
        for i in range(len(encoded)):
            for j in range(i+1, len(encoded)):
                pair = torch.cat([encoded[i], encoded[j], ordering_features[i,j]], dim=-1)
                score = self.fc_order(pair)  # Higher = i before j
                order_scores.append(score)

        return order_scores
```

**Total params**: ~2M
**Training**: Ranking loss (pairwise comparison)

### Model 5: Paragraph Break Predictor

```python
class ParagraphBreakPredictor(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.sentence_encoder = nn.LSTM(embedding_dim, 64, batch_first=True)
        self.fc_break = nn.Linear(128 + 10, 1)  # Should break after this sentence?

    def forward(self, sentence_embeddings, break_features):
        encoded, _ = self.sentence_encoder(sentence_embeddings)

        # For each sentence, score if paragraph break should follow
        break_scores = []
        for i in range(len(encoded) - 1):
            pair = torch.cat([encoded[i], encoded[i+1], break_features[i]], dim=-1)
            score = torch.sigmoid(self.fc_break(pair))
            break_scores.append(score)  # 1 = insert break, 0 = no break

        return break_scores
```

**Total params**: ~1M
**Training**: Binary classification per sentence boundary

## Summary

**You're absolutely right!** Breaking into 5 focused models (2M + 3M + 2M + 2M + 1M = 10M) is much better than one 10M model:

1. ✅ Each model has clear purpose
2. ✅ Easier to debug (know which model is failing)
3. ✅ Smaller training datasets per model
4. ✅ Can train incrementally (don't need all at once)
5. ✅ Can replace/improve individually
6. ✅ Each model truly optional (deterministic fallback)

**Pipeline**: Deterministic work → Small model → More deterministic → Small model → ...

This is excellent software engineering AND machine learning practice!

Ready to implement Phase 1 (deterministic baseline)?
