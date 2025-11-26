# Phase 3: GNN Encoder & RAG System - Design Document

**Date:** 2025-11-11
**Status:** 🚧 In Progress
**Goal:** Build semantic search over Esperanto AST corpus using Graph Neural Networks

---

## Executive Summary

Phase 3 implements the "Librarian" - a retrieval-augmented generation (RAG) system that uses Graph Neural Networks (GNNs) to encode AST structure into semantic embeddings. This enables similarity search over a 547MB Esperanto corpus based on MEANING, not just surface text.

**Key Innovation:** Unlike traditional RAG systems that embed text, we embed AST STRUCTURE, capturing explicit grammatical relationships (subject-verb-object, modification, case marking, etc.).

---

## Architecture Overview

### Dual-Track Approach

Following Opus's recommendation, we build **two systems in parallel**:

#### Track A: Baseline RAG (Simple Embeddings)
- **Week 1-2**
- Deparse ASTs → normalized Esperanto text
- Embed with `sentence-transformers` (distiluse-base-multilingual)
- FAISS similarity search
- **Goal:** Working RAG quickly, establish baseline

#### Track B: GNN Encoder (Structural Embeddings)
- **Week 3-6**
- Convert ASTs → graph representation
- Train Tree-LSTM or GAT on AST structure
- Contrastive learning (similar ASTs → similar embeddings)
- **Goal:** Beat baseline by 10%+, justify structural encoding

**Decision Point:** Compare both approaches on Q&A benchmark. Ship the winner to Phase 4.

---

## Why GNN Over Simple Embeddings?

### AST Structure Captures Semantics Text Misses

**Example:**
```esperanto
"La hundo vidas la katon" (The dog sees the cat)
vs
"La katon vidas la hundo" (Same words, different meaning due to case)
```

**Sentence embeddings:** These look very similar (same words, similar order)

**GNN on AST:** Completely different structures:
```
AST 1:
subjekto: [hund] → verbo: [vid] → objekto: [kat-n]

AST 2:
aliaj: [kat-n] → verbo: [vid] → subjekto: [hund]
```

GNN learns: `-n` suffix = accusative = object, explicit subject-verb-object relationships.

### What GNN Learns That Text Embeddings Cannot

1. **Case marking semantics** (`-n` = accusative = object)
2. **Affix transformations** (`mal-` = opposite, `-ig` = causative)
3. **Structural similarity** (same subject-verb-object pattern)
4. **Compositional meaning** (adjective modifies noun, not verb)

---

## GNN Architecture Design

### Option 1: Child-Sum Tree-LSTM (RECOMMENDED)

**Architecture:**
```
TreeLSTM(
  node_features: 128d (morpheme embedding)
  hidden_size: 256d
  num_layers: 2
  output: 512d sentence embedding
)
```

**Why Tree-LSTM:**
- ASTs are TREES, not arbitrary graphs
- More parameter-efficient than general GNNs
- Well-studied architecture (Tai et al., 2015)
- Natural fit: recursively compose child embeddings

**How it works:**
```python
def tree_lstm_cell(node, children):
    # Aggregate child hidden states
    h_children = sum([child.hidden for child in children])

    # LSTM gates
    i = sigmoid(W_i @ [node.features, h_children])  # input gate
    f = [sigmoid(W_f @ [node.features, child.hidden]) for child in children]  # forget gates
    o = sigmoid(W_o @ [node.features, h_children])  # output gate
    u = tanh(W_u @ [node.features, h_children])  # cell input

    # Cell state
    c = i * u + sum([f_j * child.cell for f_j, child in zip(f, children)])

    # Hidden state
    h = o * tanh(c)

    return h, c
```

**Advantages:**
- Bottom-up composition (leaves → root)
- Captures long-range dependencies
- Proven on constituency parsing tasks

**Disadvantages:**
- Requires ordered children (we need to define order for AST nodes)
- No explicit attention (can't visualize which nodes matter most)

### Option 2: Graph Attention Network (GAT) (ALTERNATIVE)

**Architecture:**
```
GAT(
  node_features: 128d
  hidden_channels: 256d
  num_layers: 3
  heads: 4 (multi-head attention)
  output: 512d
)
```

**Why GAT:**
- Attention mechanism: learns which AST nodes are important
- More flexible than Tree-LSTM
- Can handle any graph structure (not just trees)

**How it works:**
```python
def gat_layer(node, neighbors):
    # Attention scores
    α = softmax([
        attention(node.features, neighbor.features)
        for neighbor in neighbors
    ])

    # Weighted aggregation
    h = sum([α_i * W @ neighbor.features for α_i, neighbor in zip(α, neighbors)])

    return h
```

**Advantages:**
- Interpretable (attention weights show which nodes matter)
- Flexible (works on any graph)
- State-of-the-art on many graph tasks

**Disadvantages:**
- More parameters (slower training)
- Might be overkill for tree-structured data

### Option 3: Graph Transformer (FUTURE)

- Most flexible, most expensive
- Use if Tree-LSTM and GAT insufficient
- Defer to Phase 3B if needed

---

## Decision: Start with Tree-LSTM

**Rationale:**
1. Our ASTs are ALWAYS trees → use tree-specific model
2. Simpler architecture → faster iteration
3. If insufficient, upgrade to GAT
4. Document: "We chose Tree-LSTM because ASTs are trees"

---

## Training Strategy

### Task: Contrastive Learning

**Goal:** Similar ASTs → similar embeddings

**Approach:**
```python
# Positive pairs: ASTs with similar semantics
# - Same root words, different word order
# - Same meaning, different tense
# - Paraphrases

# Negative pairs: ASTs with different semantics
# - Different verbs
# - Different subjects/objects
# - Opposite meanings (mal- prefix)

loss = contrastive_loss(
    anchor=encode(ast1),
    positive=encode(ast2_similar),
    negative=encode(ast3_different)
)
```

### Data Augmentation

Generate positive pairs by:
1. **Word order permutation** (Esperanto allows flexible order)
2. **Tense transformation** (present → past, meaning similar)
3. **Synonym substitution** (hund → best, both mean animal)
4. **Case variation** (nominative → accusative, structural difference but related)

Generate negative pairs by:
1. **Random ASTs** from corpus
2. **Antonym substitution** (bon → malbon)
3. **Subject/object swap** (changes meaning)

### Training Data Size

- **PoC:** 10K AST pairs (100K ASTs from corpus)
- **Full training:** 100K+ AST pairs (547MB corpus = ~1M ASTs)
- **Validation:** 10K held-out pairs
- **Test:** 5K held-out pairs

### Hyperparameters

```python
TREE_LSTM_CONFIG = {
    "node_features": 128,  # morpheme embedding dimension
    "hidden_size": 256,
    "num_layers": 2,
    "output_dim": 512,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    "warmup_steps": 1000,
}
```

---

## Node Feature Design

### What Features to Encode for Each AST Node?

**Option A: Morpheme-Level Features (RECOMMENDED)**
```python
node_features = {
    "radiko": embed("hund"),           # root word embedding (128d)
    "vortspeco": onehot("substantivo"), # POS tag (10d)
    "nombro": onehot("singularo"),      # number (2d)
    "kazo": onehot("akuzativo"),        # case (2d)
    "prefikso": embed("mal") or zeros,  # prefix embedding (64d)
    "sufiksoj": mean([embed(s) for s in suffixes]),  # suffix embeddings (64d)
}
# Total: 128 + 10 + 2 + 2 + 64 + 64 = 270d
```

**Option B: Learned Features (ALTERNATIVE)**
- Pass raw morpheme strings through learned embedding layer
- Let model learn optimal representation
- More flexible, requires more data

**Decision: Start with Option A** (hand-crafted features), easier to debug and interpret.

---

## Edge Type Design

### What Edge Types to Define in AST Graph?

**Syntactic Edges:**
1. `has_subject` (frazo → subjekto)
2. `has_verb` (frazo → verbo)
3. `has_object` (frazo → objekto)
4. `has_modifier` (frazo → aliaj)
5. `modifies` (adjektivo → substantivo)
6. `has_article` (vortgrupo → artikolo)

**Morphological Edges:**
1. `has_root` (vorto → radiko)
2. `has_prefix` (vorto → prefikso)
3. `has_suffix` (vorto → sufikso)

**Total: 9 edge types** (can use separate GNN layer for each type, or embed edge type as feature)

---

## Baseline RAG Implementation

### Architecture

```
Input Query (any language)
    ↓
FrontDoor (translate to Esperanto)
    ↓
Parser (text → AST)
    ↓
Deparser (AST → normalized Esperanto text)
    ↓
sentence-transformers.encode(text)
    ↓
FAISS.search(embedding, top_k=5)
    ↓
Retrieved contexts (top-5 similar sentences)
```

### Components

1. **Corpus indexing:**
   ```python
   from sentence_transformers import SentenceTransformer
   import faiss

   model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

   # Parse corpus
   texts = [deparse(parse(sentence)) for sentence in corpus]

   # Embed
   embeddings = model.encode(texts, show_progress_bar=True)

   # Index
   index = faiss.IndexFlatL2(embeddings.shape[1])
   index.add(embeddings)
   ```

2. **Retrieval:**
   ```python
   def retrieve(query, k=5):
       # Translate to Esperanto
       eo_query = front_door.process(query)

       # Parse and deparse (normalize)
       ast = parse(eo_query)
       normalized = deparse(ast)

       # Embed and search
       query_embedding = model.encode([normalized])
       distances, indices = index.search(query_embedding, k)

       return [corpus[i] for i in indices[0]]
   ```

### Evaluation Metrics

**For both baseline and GNN:**

1. **Retrieval Accuracy:**
   - Precision@5: % of top-5 results that are relevant
   - Recall@5: % of relevant docs found in top-5
   - MRR (Mean Reciprocal Rank): 1/rank of first relevant doc

2. **Semantic Similarity:**
   - Cosine similarity between query and retrieved docs
   - Should be high for semantically similar, low for dissimilar

3. **Structural Similarity (GNN only):**
   - Tree edit distance between query AST and retrieved AST
   - Measure: Do similar AST structures have similar embeddings?

**Benchmark Dataset:**
- 1000 Esperanto Q&A pairs
- Format: (question, answer, [relevant_corpus_sentences])
- Source: Generate from corpus or use existing Esperanto Q&A

---

## Implementation Plan

### Week 1: Baseline RAG + Corpus Parsing

**Days 1-2: Corpus Parsing**
```bash
python scripts/parse_corpus_to_asts.py \
  --input data/clean_corpus/ \
  --output data/ast_corpus/ \
  --format jsonl
```
- Parse 547MB corpus → ~1M AST JSON files
- Store as JSONL (one AST per line for streaming)
- Log parsing statistics (success rate, unknown words)

**Days 3-4: Baseline RAG**
```bash
python scripts/build_baseline_rag.py \
  --corpus data/ast_corpus/ \
  --index data/faiss_index/ \
  --model distiluse-base-multilingual-cased-v2
```
- Deparse ASTs → normalized Esperanto text
- Embed with sentence-transformers
- Build FAISS index
- Implement retrieval function

**Days 5-7: Evaluation Benchmark**
```bash
python scripts/create_qa_benchmark.py \
  --corpus data/ast_corpus/ \
  --output data/qa_benchmark.json \
  --num_pairs 1000
```
- Generate or curate 1000 Q&A pairs
- Evaluate baseline RAG (Precision@5, Recall@5, MRR)
- Document baseline performance

### Week 2: AST → Graph Conversion

**Days 8-10: Graph Format Design**
```python
# klareco/ast_to_graph.py
def ast_to_pyg_graph(ast):
    """Convert AST to PyTorch Geometric Data object"""
    nodes = []  # List of node feature tensors
    edges = []  # List of (src, dst, edge_type) tuples

    # Traverse AST, extract nodes and edges
    # Return: Data(x=nodes, edge_index=edges, edge_type=...)
```

**Days 11-14: Training Data Preparation**
```bash
python scripts/prepare_gnn_training_data.py \
  --ast_corpus data/ast_corpus/ \
  --output data/gnn_training/ \
  --augment \
  --num_pairs 10000
```
- Convert 100K ASTs → graph format
- Generate positive/negative pairs
- Data augmentation (word order, tense, synonyms)
- Save as PyG dataset

### Week 3-4: Tree-LSTM Implementation

**Days 15-21: Model Implementation**
```python
# klareco/models/tree_lstm.py
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, node_features=270, hidden_size=256, output_dim=512):
        # Implement Tree-LSTM architecture

    def forward(self, ast_graph):
        # Bottom-up traversal, compose embeddings
        # Return: sentence embedding (512d)
```

**Days 22-28: Training Loop**
```bash
python scripts/train_tree_lstm.py \
  --data data/gnn_training/ \
  --config configs/tree_lstm.yaml \
  --output models/tree_lstm_v1/
```
- Implement contrastive loss
- Training loop with validation
- Log to WandB or TensorBoard
- Save checkpoints

### Week 5: Evaluation & Comparison

**Days 29-32: GNN RAG Pipeline**
```python
# Integrate Tree-LSTM into RAG
def retrieve_with_gnn(query, k=5):
    ast = parse(query)
    graph = ast_to_graph(ast)
    embedding = tree_lstm.encode(graph)

    distances, indices = gnn_index.search(embedding, k)
    return [corpus[i] for i in indices[0]]
```

**Days 33-35: Head-to-Head Comparison**
```bash
python scripts/evaluate_rag_systems.py \
  --baseline baseline_index/ \
  --gnn gnn_index/ \
  --benchmark data/qa_benchmark.json
```

**Metrics to compare:**
| Metric | Baseline | Tree-LSTM | Winner |
|--------|----------|-----------|--------|
| Precision@5 | ? | ? | ? |
| Recall@5 | ? | ? | ? |
| MRR | ? | ? | ? |
| Structural Similarity | N/A | ? | - |

**Decision:** If Tree-LSTM beats baseline by 10%+, ship to Phase 4. Otherwise, analyze why and iterate.

### Week 6: Integration & Documentation

**Days 36-40: RAG Integration**
```python
# klareco/rag.py
class RAGSystem:
    def __init__(self, encoder_type="tree_lstm"):
        # Load encoder and index

    def retrieve(self, query, k=5):
        # End-to-end retrieval

    def add_to_index(self, text):
        # Add new documents dynamically
```

**Days 41-42: Documentation**
- Update DESIGN.md (mark Phase 3 complete)
- Write PHASE3_RESULTS.md (performance report)
- Update README.md (add RAG usage examples)
- Commit and push

---

## Success Criteria

### Minimum Viable Product (MVP):
- ✅ Baseline RAG working (sentence-transformers + FAISS)
- ✅ Tree-LSTM encoder trained
- ✅ Evaluation benchmark created
- ✅ GNN beats baseline on at least ONE metric

### Stretch Goals:
- 🎯 GNN beats baseline by 10%+ on Precision@5
- 🎯 Structural similarity correlates with semantic similarity
- 🎯 Retrieval latency <100ms (encoding + search)
- 🎯 AST corpus fully parsed (95%+ success rate)

---

## Risks & Mitigations

### Risk 1: GNN Doesn't Beat Baseline
**Probability:** Medium
**Impact:** High (invalidates structural encoding hypothesis)
**Mitigation:**
- Analyze failure modes (where does GNN fail?)
- Try GAT or Graph Transformer
- Investigate: Are sentence-transformers already capturing structure implicitly?

### Risk 2: Training Data Insufficient
**Probability:** Low
**Impact:** Medium (model doesn't converge)
**Mitigation:**
- More aggressive data augmentation
- Pre-train on unsupervised task (AST reconstruction)
- Use transfer learning from existing graph models

### Risk 3: Corpus Parsing Fails
**Probability:** Low (we're at 95.4%)
**Impact:** Medium (fewer training examples)
**Mitigation:**
- Graceful degradation already handles this
- Use partially parsed ASTs (mark unknown nodes)
- Expand vocabulary incrementally

### Risk 4: Integration Complexity
**Probability:** Medium
**Impact:** Low (delays Phase 4)
**Mitigation:**
- Keep RAG interface simple (query → top-K docs)
- Abstract encoder behind common interface
- Extensive integration testing

---

## Next Steps (Immediate)

1. **Create `scripts/parse_corpus_to_asts.py`** - Parse 547MB corpus
2. **Create `scripts/build_baseline_rag.py`** - Implement baseline RAG
3. **Create `scripts/create_qa_benchmark.py`** - Build evaluation dataset
4. **Create `klareco/ast_to_graph.py`** - AST → graph conversion
5. **Create `klareco/models/tree_lstm.py`** - Tree-LSTM implementation

---

## References

1. Tai et al. (2015) - "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
2. Veličković et al. (2017) - "Graph Attention Networks"
3. Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
4. PyTorch Geometric Documentation - https://pytorch-geometric.readthedocs.io/

---

**Status:** Ready to begin implementation
**Next:** Create corpus parsing script (`scripts/parse_corpus_to_asts.py`)
