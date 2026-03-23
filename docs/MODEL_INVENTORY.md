# Klareco Model Inventory

Complete list of all deep learning models in the Klareco architecture, organized by capability.

## Model Overview

| Model Name | Parameters | Status | Purpose | Used By |
|------------|------------|--------|---------|---------|
| **RootEmbedder** | 500K | 🔄 In Progress | Semantic similarity between roots | All capabilities |
| **MorphemeComposer** | 500K | 🔄 In Progress | Combine root + affix semantics | Text generation, retrieval |
| **PlausibilityFilter** | 2M | 📝 Planned (#687) | Filter implausible facts | Reasoning, QA |
| **RelevanceRanker** | 5M | 📝 Planned (#686) | Rank retrieved facts | Retrieval, QA |
| **ASTEncoder** | 8M | ✅ Exists | Encode AST to vector | Generation, chat, intent |
| **NodePredictor** | 12M | 📝 Planned (#692) | Predict next AST node | Text generation |
| **IntentClassifier** | 5M | 📝 Optional (#693) | Classify instruction intent | Instruction following |
| **DiscourseClassifier** | 10M | 📝 Optional (#694) | Classify turn relations | Multi-turn chat |

**Total (Minimal)**: 21M params (RootEmbedder, MorphemeComposer, ASTEncoder, NodePredictor)
**Total (Full)**: 43M params (All models)
**Traditional LLM**: 70B-175B params (1,600x - 8,300x larger!)

## Foundation Models (Required by All)

### RootEmbedder (500K params)
- **Size**: 40K roots × 64 dims = 500K params (after compression)
- **Architecture**: Embedding lookup table
- **Training**: Contrastive learning on corpus semantic pairs
- **Training Data**: ~500K semantic pairs from AST-annotated corpus
- **Purpose**: Capture semantic similarity between Esperanto roots
- **Use Cases**:
  - Query expansion (expand "hundo" to "besto", "mamulo")
  - Semantic similarity scoring
  - Synonym detection
- **Training Script**: `scripts/train_roots.sh`
- **Status**: 🔄 In Progress (#685)
- **Dependencies**: None (first model to train)
- **Example**:
  ```python
  embed("hund") ≈ embed("best")  # dog ≈ animal
  embed("bel") ≈ embed("plag")   # beautiful ≈ pleasant
  ```

### MorphemeComposer (500K params)
- **Size**: 500K params
  - Prefix embeddings: 16 × 8 dims
  - Suffix embeddings: 32 × 8 dims
  - Combination MLP: 256 → 128
  - Reuses RootEmbedder root embeddings
- **Architecture**: Root + affix composition via learned MLP
- **Training**: Predict word meaning from morphemes
- **Purpose**: Handle unseen word combinations
- **Use Cases**:
  - Generalize to novel constructions ("rehundejo" = re+hund+ej+o)
  - Predict semantics of compounds
  - Zero-shot understanding of new affixed words
- **Training Script**: `scripts/train_affixes.sh`
- **Status**: 🔄 In Progress
- **Dependencies**: RootEmbedder (root embeddings)
- **Example**:
  ```python
  embed("rehundejo") = combine(
      embed("re"),    # prefix: again
      embed("hund"),  # root: dog
      embed("ej"),    # suffix: place
      embed("o")      # ending: noun
  )
  # Result: "dog kennel" semantics from never-seen-before word
  ```

## Retrieval Models (Support Reasoning & QA)

### PlausibilityFilter (2M params)
- **Size**: 2M params
- **Architecture**: SVO triple classifier
- **Training**: Binary classification on plausible/implausible triples
- **Purpose**: Filter semantically implausible facts
- **Use Cases**:
  - Remove nonsense from retrieval results
  - Improve reasoning accuracy
  - Reduce hallucinations
- **Training Script**: `scripts/train_m1.sh` (to be created)
- **Status**: 📝 Planned (#687)
- **Dependencies**: RootEmbedder (root embeddings)
- **Example**:
  ```python
  plausible("Zamenhof founded Esperanto") → 0.95 ✓
  implausible("Table founded language") → 0.02 ✗
  ```

### RelevanceRanker (5M params)
- **Size**: 5M params
- **Architecture**: Cross-encoder (query + fact → relevance score)
- **Training**: Pairwise ranking on query-fact pairs
- **Purpose**: Rank retrieved facts by relevance to query
- **Use Cases**:
  - Improve retrieval precision
  - Order facts for reasoning
  - Prioritize most relevant evidence
- **Training Script**: `scripts/train_reranker.sh` (to be created)
- **Status**: 📝 Planned (#686)
- **Dependencies**: RootEmbedder (root embeddings)
- **Example**:
  ```python
  query = "Kiu fondis Esperanton?"
  rerank([
      ("Zamenhof founded Esperanto", 0.98),  # Most relevant
      ("Esperanto is international", 0.45),  # Less relevant
  ])
  ```

## Generation Models

### ASTEncoder (8M params)
- **Size**: 8M params
- **Architecture**: Child-Sum TreeLSTM
- **Training**: Part of downstream tasks (NodePredictor, IntentClassifier, DiscourseClassifier)
- **Purpose**: Encode AST structure into dense vector
- **Use Cases**:
  - Context encoding for generation
  - AST similarity computation
  - Intent classification input
  - Discourse relation input
- **Training Script**: Trained jointly with downstream models
- **Status**: ✅ Exists (in `klareco/models/tree_lstm.py`)
- **Dependencies**: MorphemeComposer (compositional embeddings for node features)
- **Example**:
  ```python
  ast = parse("Mi amas la hundon")
  vector = tree_lstm_encoder(ast)  # 256-dim context vector
  ```

### NodePredictor (12M params)
- **Size**: 12M params
- **Architecture**: Multi-head classifier
  - Root predictor: 8M params (40K vocab)
  - Word type: 0.5M params (10 classes)
  - Case: 0.5M params (5 classes)
  - Number: 0.5M params (2 classes)
  - Tense: 0.5M params (6 classes)
  - Affixes: 2M params (multi-label)
- **Training**: Next-node prediction on parsed corpus
- **Purpose**: Predict next AST node given partial AST context
- **Use Cases**:
  - Text completion
  - Paraphrasing
  - Question generation
  - Abstractive summarization
- **Training Script**: `scripts/train_ast_generator.sh` (to be created)
- **Status**: 📝 Planned (#692)
- **Dependencies**: ASTEncoder (TreeLSTM encoder)
- **Example**:
  ```python
  partial_ast = parse("Mi amas la hun")
  next_node = predictor(partial_ast)
  # Predicts: {radiko: "hund", kazo: "akuzativo", ...}
  # Deparse: "hundon"
  ```

## Instruction & Chat Models (Optional)

### IntentClassifier (5M params, optional)
- **Size**: 5M params
- **Architecture**: TreeLSTM encoder + small MLP classifier
- **Training**: Instruction-intent pairs (6-10 classes)
- **Purpose**: Classify ambiguous instructions (fallback only!)
- **Coverage**: Only 15% of instructions (85% handled by patterns)
- **Use Cases**:
  - Disambiguate unclear instructions
  - Avoid clarification questions
  - Automatic intent routing
- **Training Script**: `scripts/train_intent_classifier.sh` (to be created)
- **Status**: 📝 Optional (#693)
- **Dependencies**: ASTEncoder (TreeLSTM encoder)
- **Alternative**: Use deterministic patterns + clarification questions (0 params)
- **Example**:
  ```python
  # Pattern matching handles most cases (0 params):
  classify("Respondu: Kiu...") → extractive_qa  # Verb pattern

  # Learned model for ambiguous cases only:
  classify("Pri Zamenhof...") → summarization   # Ambiguous, needs learning
  ```

### DiscourseClassifier (10M params, optional)
- **Size**: 10M params
- **Architecture**: Dual TreeLSTM encoder + relation classifier
- **Training**: Multi-turn dialogues with relation annotations (6 classes)
- **Purpose**: Classify discourse relation between turns
- **Coverage**: Improves response quality (basic chat works without it!)
- **Use Cases**:
  - Better response generation
  - Acknowledge topic shifts
  - Handle corrections appropriately
- **Training Script**: `scripts/train_discourse_model.sh` (to be created)
- **Status**: 📝 Optional (#694)
- **Dependencies**: ASTEncoder (TreeLSTM encoder)
- **Alternative**: Basic chat works with 0 params (deterministic coreference + entity tracking)
- **Example**:
  ```python
  # Deterministic chat (0 params):
  - Resolve pronouns: li → Zamenhof (gender/number matching)
  - Track entities: {Zamenhof, Esperanto}
  - Maintain AST history

  # Learned model improves quality:
  classify(current="Kial?", previous="Zamenhof fondis...") → elaboration
  # AI knows to provide more detail, not just answer literally
  ```

## Reasoning Models

### NONE! Reasoning is 100% Deterministic (0 params)

All reasoning components are rule-based:
- **Query parsing**: AST traversal rules
- **Fact extraction**: SVO triple extraction rules
- **Inference engine**: 10 first-order logic rules
- **AST Trail**: Provenance logging (data structure)

The only learned components used in reasoning are retrieval models (PlausibilityFilter, RelevanceRanker), which filter and rank facts before deterministic inference.

**Inference Rules** (0 params, 100% deterministic):
1. Transitivity
2. Temporal ordering
3. Temporal extrema (first/last)
4. Property inheritance
5. Spatial containment
6. Negation handling
7. Causality chains
8. Set membership
9. Comparison transitivity
10. Counting

See #695 for full details.

## Training Order

```
Stage 0: Parser (deterministic, 0 params) ✅
  ↓
Stage 1: RootEmbedder Root Embeddings (500K) 🔄
  ↓
Stage 2: MorphemeComposer Compositional Embeddings (500K) 🔄
  ↓
Stage 3: PlausibilityFilter Selectional Preference (2M) 📝
  ↓
Stage 4: RelevanceRanker Reranker (5M) 📝
  ↓
Stage 5: ASTEncoder TreeLSTM + NodePredictor Next-Node (8M + 12M) 📝
  ↓
Stage 6 (Optional): IntentClassifier Intent Classifier (5M) 📝
  ↓
Stage 7 (Optional): DiscourseClassifier Discourse Classifier (10M) 📝
```

## Configuration Options

### Minimal Configuration (21M params)
**Recommended for initial implementation**

- RootEmbedder: Root Embeddings (500K) ✓
- MorphemeComposer: Compositional Embeddings (500K) ✓
- ASTEncoder: TreeLSTM Encoder (8M) ✓
- NodePredictor: Next-Node Predictor (12M) ✓
- **TOTAL: 21M params**

**Capabilities**:
- ✓ Text generation (NodePredictor)
- ✓ Instruction following (deterministic patterns, 85% coverage)
- ✓ Multi-turn chat (deterministic coreference, 90% coverage)
- ✓ Symbolic reasoning (deterministic inference, 100% coverage)

**Missing**:
- Advanced retrieval filtering (no PlausibilityFilter, RelevanceRanker)
- Ambiguous intent disambiguation (uses clarification instead)
- Discourse quality (basic but functional)

### Standard Configuration (28M params)
**Recommended for production**

Minimal + Retrieval models:
- PlausibilityFilter: Selectional Preference (2M) ✓
- RelevanceRanker: Reranker (5M) ✓
- **TOTAL: 28M params**

**Improvements**:
- Better retrieval precision (RelevanceRanker reranking)
- Fewer hallucinations (PlausibilityFilter filtering)
- More accurate reasoning

### Full Configuration (43M params)
**Recommended for maximum quality**

Standard + Optional models:
- IntentClassifier: Intent Classifier (5M) ✓
- DiscourseClassifier: Discourse Classifier (10M) ✓
- **TOTAL: 43M params**

**Improvements**:
- Automatic intent disambiguation (no clarification questions)
- Better chat response quality
- Smoother conversation flow

## Comparison to Traditional LLMs

| Configuration | Klareco Params | vs GPT-3 (175B) | vs LLaMA-2-70B | vs GPT-4 (~1.8T) |
|---------------|----------------|-----------------|----------------|------------------|
| **Minimal** | 21M | 8,333x smaller | 3,333x smaller | 85,714x smaller |
| **Standard** | 28M | 6,250x smaller | 2,500x smaller | 64,286x smaller |
| **Full** | 43M | 4,070x smaller | 1,628x smaller | 41,860x smaller |

## Model Cards

Each model should have a detailed model card documenting:
- Architecture details
- Training data requirements
- Hyperparameters
- Evaluation metrics
- Known limitations
- Ethical considerations

See `models/*/MODEL_CARD.md` (to be created during training).

## Training Data Requirements

| Model | Training Data | Source | Size |
|-------|---------------|--------|------|
| RootEmbedder | Semantic pairs | AST-annotated corpus | ~500K pairs |
| MorphemeComposer | Compositional examples | Corpus word decompositions | ~200K examples |
| PlausibilityFilter | SVO plausibility | Generated + annotated | ~100K triples |
| RelevanceRanker | Query-fact pairs | QA dataset | ~50K pairs |
| NodePredictor | Next-node prediction | Parsed corpus | ~1M ASTs |
| IntentClassifier | Instruction-intent | Manual annotation | ~10K instructions |
| DiscourseClassifier | Multi-turn dialogues | Dialogues + annotation | ~5K dialogues |

Total annotation effort: ~10K instructions + ~5K dialogues (~15K human annotations)

Compare to LLM pretraining: Trillions of tokens (100,000x less data!)

## Evaluation Metrics

| Model | Metrics | Target |
|-------|---------|--------|
| RootEmbedder | Synonym accuracy, clustering | >85% |
| MorphemeComposer | Composition accuracy | >80% |
| PlausibilityFilter | Plausibility precision/recall | >90% |
| RelevanceRanker | Ranking accuracy (NDCG) | >0.85 |
| NodePredictor | Generation coherence, grammaticality | 100% grammatical, >80% coherent |
| IntentClassifier | Intent classification accuracy | >95% |
| DiscourseClassifier | Discourse relation accuracy | >80% |

## Storage Requirements

| Model | Disk Size | Memory (Inference) |
|-------|-----------|-------------------|
| RootEmbedder | ~2MB | ~5MB |
| MorphemeComposer | ~2MB | ~5MB |
| PlausibilityFilter | ~8MB | ~20MB |
| RelevanceRanker | ~20MB | ~50MB |
| ASTEncoder | ~32MB | ~80MB |
| NodePredictor | ~48MB | ~120MB |
| IntentClassifier | ~20MB | ~50MB |
| DiscourseClassifier | ~40MB | ~100MB |
| **Total (Full)** | **~172MB** | **~430MB** |

Compare to LLaMA-2-70B: ~140GB (814x larger!)

## Inference Speed (Target)

| Model | Latency | Throughput |
|-------|---------|------------|
| RootEmbedder | <1ms | 100K lookups/sec |
| MorphemeComposer | <1ms | 50K compositions/sec |
| PlausibilityFilter | <5ms | 1K triples/sec |
| RelevanceRanker | <10ms | 100 rankings/sec |
| ASTEncoder+NodePredictor | <50ms | 20 generations/sec |
| IntentClassifier | <5ms | 200 classifications/sec |
| DiscourseClassifier | <10ms | 100 relations/sec |

End-to-end latency target: <500ms per query

Compare to GPT-3 API: ~1-5 seconds (2-10x faster!)

## Next Steps

1. Complete RootEmbedder, MorphemeComposer training (in progress)
2. Implement PlausibilityFilter, RelevanceRanker for retrieval improvements
3. Implement ASTEncoder+NodePredictor for text generation
4. Evaluate minimal configuration (21M params)
5. Decide if IntentClassifier, DiscourseClassifier are needed (compare to deterministic baseline)
6. Publish model cards for each trained model
7. Benchmark against traditional LLMs
