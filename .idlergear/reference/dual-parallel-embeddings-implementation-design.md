---
id: 1
title: Dual Parallel Embeddings - Implementation Design
created: '2026-01-05T22:46:52.612444Z'
updated: '2026-01-05T22:46:52.612458Z'
---
# Dual Parallel Embeddings - Implementation Design

## Executive Summary

Design for implementing Option 2: Two independent 64d embeddings per root (linguistic + topical).

**Goals:**
- Improve retrieval accuracy from 12% → 30-40%
- Maintain explainability (show linguistic vs topical contribution)
- Preserve backward compatibility with existing system

**Timeline:** 2-3 weeks
- Week 1: Architecture + data preparation
- Week 2: Training implementation
- Week 3: Integration + evaluation

---

## 1. Architecture Changes

### 1.1 New Class: `DualRootEmbeddings`

**File:** `klareco/embeddings/dual_root_embeddings.py` (NEW)

```python
class DualRootEmbeddings(nn.Module):
    """
    Two independent embeddings per root:
    - Linguistic: semantic relations (ReVo + Ekzercaro)
    - Topical: document context (corpus co-occurrence)
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Two independent embedding layers
        self.linguistic = nn.Embedding(vocab_size, embedding_dim)
        self.topical = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with different strategies
        nn.init.normal_(self.linguistic.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.topical.weight, mean=0.0, std=0.5)
    
    def forward(self, indices: torch.Tensor, mode: str = 'combined') -> torch.Tensor:
        """
        Args:
            indices: Root indices [batch_size]
            mode: 'linguistic' | 'topical' | 'combined'
        
        Returns:
            embeddings: [batch_size, embed_dim] or [batch_size, 2*embed_dim]
        """
        if mode == 'linguistic':
            return self.linguistic(indices)  # 64d
        elif mode == 'topical':
            return self.topical(indices)      # 64d
        else:  # combined
            ling = self.linguistic(indices)
            topic = self.topical(indices)
            return torch.cat([ling, topic], dim=-1)  # 128d
    
    def get_normalized(self, indices: torch.Tensor, mode: str = 'combined'):
        """Get L2-normalized embeddings."""
        emb = self.forward(indices, mode)
        return F.normalize(emb, dim=-1)
    
    def similarity(self, idx1: torch.Tensor, idx2: torch.Tensor, 
                   mode: str = 'combined', 
                   weights: Tuple[float, float] = (0.5, 0.5)) -> torch.Tensor:
        """
        Compute weighted similarity.
        
        Args:
            weights: (linguistic_weight, topical_weight)
        """
        if mode == 'linguistic':
            emb1 = self.get_normalized(idx1, 'linguistic')
            emb2 = self.get_normalized(idx2, 'linguistic')
            return (emb1 * emb2).sum(dim=-1)
        elif mode == 'topical':
            emb1 = self.get_normalized(idx1, 'topical')
            emb2 = self.get_normalized(idx2, 'topical')
            return (emb1 * emb2).sum(dim=-1)
        else:  # weighted combination
            ling_sim = self.similarity(idx1, idx2, 'linguistic')
            topic_sim = self.similarity(idx1, idx2, 'topical')
            return weights[0] * ling_sim + weights[1] * topic_sim
```

### 1.2 Update: `CompositionalEmbedding`

**File:** `klareco/embeddings/compositional.py` (MODIFY)

```python
class CompositionalEmbedding(nn.Module):
    def __init__(
        self,
        root_vocab: Dict[str, int],
        prefix_vocab: Dict[str, int],
        suffix_vocab: Dict[str, int],
        embed_dim: int = 128,
        use_dual_roots: bool = True,  # NEW parameter
        composition_method: str = 'sum',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.use_dual_roots = use_dual_roots
        self.embed_dim = embed_dim
        
        if use_dual_roots:
            # Root: 128d (64d linguistic + 64d topical)
            self.root_embed = DualRootEmbeddings(len(root_vocab), 64)
            self.root_dim = 128
        else:
            # Backward compatibility: 64d single embedding
            self.root_embed = nn.Embedding(len(root_vocab), 64)
            self.root_dim = 64
        
        # Affixes stay the same
        self.prefix_embed = nn.Embedding(len(prefix_vocab), 16)
        self.suffix_embed = nn.Embedding(len(suffix_vocab), 16)
        self.ending_embed = nn.Embedding(len(ENDINGS), 16)
        
        # Projection to final embed_dim
        total_dim = self.root_dim + 16 + 16 + 16  # root + prefix + suffix + ending
        if total_dim != embed_dim:
            self.projection = nn.Linear(total_dim, embed_dim)
        else:
            self.projection = None
    
    def forward(self, parsed_words: List[Dict], 
                root_mode: str = 'combined') -> torch.Tensor:
        """
        Args:
            root_mode: For dual roots - 'linguistic', 'topical', or 'combined'
        """
        embeddings = []
        
        for word in parsed_words:
            # Root embedding (64d or 128d depending on use_dual_roots)
            root_idx = self.root_vocab[word['radiko']]
            if self.use_dual_roots:
                root_emb = self.root_embed(root_idx, mode=root_mode)
            else:
                root_emb = self.root_embed(root_idx)
            
            # Affixes (same as before)
            # ... rest of composition logic
            
        return embeddings
```

---

## 2. Training Pipeline

### 2.1 Data Preparation: Topical Pairs

**File:** `scripts/data/prepare_topical_pairs.py` (NEW)

```python
def extract_skipgram_pairs(
    corpus_path: Path,
    window_size: int = 5,
    min_frequency: int = 5,
    output_path: Path
):
    """
    Extract skip-gram training pairs from corpus.
    
    For each sentence:
        For each center word:
            Create pairs with context words within window
    
    Args:
        window_size: Context window (5 = 2 words each side)
        min_frequency: Filter rare roots
    """
    pairs = []
    root_freq = defaultdict(int)
    
    # Pass 1: Count frequencies
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            ast = doc.get('ast')
            if not ast:
                continue
            
            roots = extract_roots_from_ast(ast)
            for root in roots:
                root_freq[root] += 1
    
    # Pass 2: Extract pairs
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            ast = doc.get('ast')
            if not ast:
                continue
            
            roots = extract_roots_from_ast(ast)
            # Filter by frequency
            roots = [r for r in roots if root_freq[r] >= min_frequency]
            
            # Skip-gram pairs
            for i, center in enumerate(roots):
                start = max(0, i - window_size)
                end = min(len(roots), i + window_size + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    context = roots[j]
                    
                    # Positive pair: (center, context, 1.0)
                    pairs.append((center, context, 1.0))
    
    # Negative sampling
    pairs_with_negatives = add_negative_samples(pairs, root_freq, ratio=5)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump({
            'pairs': pairs_with_negatives,
            'root_freq': dict(root_freq),
            'window_size': window_size,
        }, f)
```

### 2.2 Training: Joint Multi-Task

**File:** `scripts/training/train_dual_embeddings.py` (NEW)

```python
def train_dual_embeddings(
    linguistic_pairs: List[Tuple],  # ReVo + Ekzercaro
    topical_pairs: List[Tuple],     # Corpus skip-gram
    vocab: Dict[str, int],
    epochs: int = 100,
    linguistic_weight: float = 0.5,
    topical_weight: float = 0.5,
):
    """
    Joint training of both embeddings.
    """
    model = DualRootEmbeddings(len(vocab), embedding_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Training step
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Linguistic loss (on linguistic embeddings only)
            ling_loss = compute_linguistic_loss(
                model, batch['linguistic_pairs'], mode='linguistic'
            )
            
            # Topical loss (on topical embeddings only)
            topic_loss = compute_topical_loss(
                model, batch['topical_pairs'], mode='topical'
            )
            
            # Combined loss
            loss = linguistic_weight * ling_loss + topical_weight * topic_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation
        ling_corr = evaluate_linguistic(model, val_linguistic_pairs)
        topic_corr = evaluate_topical(model, val_topical_pairs)
        
        logger.info(f"Epoch {epoch}: Ling={ling_corr:.3f}, Topic={topic_corr:.3f}")

def compute_linguistic_loss(model, pairs, mode='linguistic'):
    """MSE + margin loss on linguistic embeddings."""
    root1, root2, target_sim, weight = pairs
    
    pred_sim = model.similarity(root1, root2, mode=mode)
    
    # MSE regression
    mse_loss = ((pred_sim - target_sim) ** 2 * weight).mean()
    
    # Margin loss for negatives
    negatives = target_sim < 0.3
    if negatives.any():
        margin_violation = F.relu(pred_sim[negatives] - 0.3)
        margin_loss = (margin_violation ** 2).mean()
    else:
        margin_loss = 0.0
    
    return mse_loss + 0.5 * margin_loss

def compute_topical_loss(model, pairs, mode='topical'):
    """Skip-gram negative sampling loss on topical embeddings."""
    center, context, label = pairs  # label: 1.0 (positive) or 0.0 (negative)
    
    center_emb = model.get_normalized(center, mode=mode)
    context_emb = model.get_normalized(context, mode=mode)
    
    # Dot product similarity
    sim = (center_emb * context_emb).sum(dim=-1)
    
    # BCE loss (binary classification)
    loss = F.binary_cross_entropy_with_logits(sim, label)
    
    return loss
```

### 2.3 Alternative: Sequential Training

```python
# Phase 1: Train linguistic (or load existing)
linguistic_model = train_linguistic_only(...)
# Save: models/dual_embeddings/linguistic.pt

# Phase 2: Train topical separately
topical_model = train_topical_only(...)
# Save: models/dual_embeddings/topical.pt

# Phase 3: Combine into DualRootEmbeddings
dual_model = DualRootEmbeddings(vocab_size, 64)
dual_model.linguistic.load_state_dict(linguistic_model.state_dict())
dual_model.topical.load_state_dict(topical_model.state_dict())
```

---

## 3. Storage Format

### 3.1 Checkpoint Structure

```python
checkpoint = {
    'epoch': 100,
    'model_state_dict': {
        'linguistic.weight': torch.Tensor([11121, 64]),
        'topical.weight': torch.Tensor([11121, 64]),
    },
    'optimizer_state_dict': {...},
    
    # Metadata
    'embedding_dim': 64,
    'vocab_size': 11121,
    'model_type': 'dual',  # vs 'single' for backward compat
    
    # Vocabularies
    'root_to_idx': {...},
    'idx_to_root': {...},
    
    # Training info
    'linguistic_correlation': 0.89,
    'topical_correlation': 0.75,
    'linguistic_loss': 0.015,
    'topical_loss': 0.023,
}
```

### 3.2 Migration from Single to Dual

```python
def migrate_single_to_dual(single_path: Path, dual_path: Path):
    """
    Convert existing single embedding to dual format.
    
    Strategy: Use single embedding as BOTH linguistic and topical initially,
    then fine-tune topical with corpus data.
    """
    # Load single
    single_ckpt = torch.load(single_path)
    single_weights = single_ckpt['model_state_dict']['embeddings.weight']
    
    # Create dual
    dual_model = DualRootEmbeddings(vocab_size=11121, embedding_dim=64)
    
    # Initialize both with same weights
    dual_model.linguistic.weight.data = single_weights.clone()
    dual_model.topical.weight.data = single_weights.clone()
    
    # Save
    torch.save({
        'model_state_dict': dual_model.state_dict(),
        'model_type': 'dual',
        'migrated_from': str(single_path),
        ...
    }, dual_path)
```

---

## 4. Retrieval Integration

### 4.1 Slot-Based Indexer Updates

**File:** `klareco/rag/slot_indexer.py` (MODIFY)

```python
class SlotBasedIndexer:
    def __init__(
        self,
        root_model_path: Path,
        use_dual: bool = True,
        embedding_mode: str = 'combined'  # 'linguistic' | 'topical' | 'combined'
    ):
        self.use_dual = use_dual
        self.embedding_mode = embedding_mode
        
        # Load model
        if use_dual:
            self.root_model = load_dual_embeddings(root_model_path)
        else:
            self.root_model = load_single_embeddings(root_model_path)
    
    def embed_slots(self, slots: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Embed SUBJ/VERB/OBJ slots.
        
        For dual embeddings, use specified mode.
        """
        slot_embeddings = {}
        
        for slot_name, roots in slots.items():
            if not roots:
                slot_embeddings[slot_name] = np.zeros(self.embedding_dim)
                continue
            
            # Get root embeddings (64d or 128d depending on mode)
            root_indices = [self.root_to_idx[r] for r in roots]
            root_embs = self.root_model.forward(
                torch.tensor(root_indices),
                mode=self.embedding_mode
            )
            
            # Average
            slot_embeddings[slot_name] = root_embs.mean(dim=0).numpy()
        
        return slot_embeddings
```

### 4.2 Query-Time Weighting

**File:** `klareco/rag/ast_aware_retriever.py` (MODIFY)

```python
class ASTAwareRetriever:
    def __init__(
        self,
        index_path: Path,
        embedding_weights: Dict[str, float] = None,  # NEW
    ):
        self.embedding_weights = embedding_weights or {
            'linguistic': 0.5,
            'topical': 0.5
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = 'auto',
        prefilter_n: int = 500,
        embedding_mode: str = 'adaptive',  # NEW
    ):
        """
        Args:
            embedding_mode:
                - 'linguistic': Use only linguistic embeddings
                - 'topical': Use only topical embeddings  
                - 'combined': Use weighted combination
                - 'adaptive': Choose based on query type
        """
        # Classify query
        query_ast = parse(query)
        classification = self.question_classifier.classify(query, query_ast)
        
        # Adaptive weighting based on query type
        if embedding_mode == 'adaptive':
            if classification['type'] == 'definition':
                # Definitions need linguistic similarity
                mode = 'combined'
                weights = {'linguistic': 0.7, 'topical': 0.3}
            elif classification['type'] in ['who', 'where', 'when']:
                # Factual questions need topical context
                mode = 'combined'
                weights = {'linguistic': 0.3, 'topical': 0.7}
            else:
                # Default: balanced
                mode = 'combined'
                weights = {'linguistic': 0.5, 'topical': 0.5}
        else:
            mode = embedding_mode
            weights = self.embedding_weights
        
        # Pre-filter with chosen mode
        if self.prefilter_retriever:
            prefilter_results = self.prefilter_retriever.search(
                query,
                top_k=prefilter_n,
                embedding_mode=mode,
                embedding_weights=weights
            )
        
        # ... rest of retrieval
```

### 4.3 Explainability Output

```python
def explain_retrieval(self, query: str, doc: Dict, score: float) -> Dict:
    """
    Decompose match score into components.
    """
    query_ast = parse(query)
    doc_ast = doc.get('ast') or parse(doc['text'])
    
    # AST pattern match (deterministic)
    pattern_score = self.pattern_matcher.match(query_ast, doc_ast).score
    
    # Linguistic similarity (learned)
    query_roots = extract_roots(query_ast)
    doc_roots = extract_roots(doc_ast)
    ling_score = self.compute_similarity(query_roots, doc_roots, mode='linguistic')
    
    # Topical similarity (learned)
    topic_score = self.compute_similarity(query_roots, doc_roots, mode='topical')
    
    return {
        'total_score': score,
        'components': {
            'ast_pattern': {
                'score': pattern_score,
                'weight': 0.4,
                'contribution': 0.4 * pattern_score,
                'explanation': 'SUBJ-VERB-OBJ structure matches'
            },
            'linguistic': {
                'score': ling_score,
                'weight': 0.3,
                'contribution': 0.3 * ling_score,
                'explanation': f'Semantic relations: {similar_pairs}'
            },
            'topical': {
                'score': topic_score,
                'weight': 0.3,
                'contribution': 0.3 * topic_score,
                'explanation': f'Shared context: {shared_topics}'
            }
        }
    }
```

---

## 5. Implementation Phases

### Phase 1: Architecture (Week 1, Days 1-3)
**Goal:** Dual embeddings work in isolation

- [ ] Create `DualRootEmbeddings` class
- [ ] Update `CompositionalEmbedding` with `use_dual_roots` flag
- [ ] Write unit tests for dual embedding forward pass
- [ ] Implement checkpoint save/load for dual format
- [ ] Migration script: single → dual

**Deliverable:** Can load/save dual embeddings, forward pass works

### Phase 2: Data Preparation (Week 1, Days 4-5)
**Goal:** Topical training data ready

- [ ] Implement `prepare_topical_pairs.py`
- [ ] Extract skip-gram pairs from 4.3M corpus (window=5)
- [ ] Add negative sampling (ratio=5:1)
- [ ] Validate data quality (frequency distribution, pair statistics)
- [ ] Save to `data/training/topical_skipgram_pairs.json`

**Deliverable:** ~50M topical training pairs

### Phase 3: Training (Week 2, Days 1-4)
**Goal:** Train dual embeddings

**Option A: Joint Training**
- [ ] Implement `train_dual_embeddings.py`
- [ ] Multi-task loss: linguistic + topical
- [ ] Train for 100 epochs
- [ ] Monitor both correlations

**Option B: Sequential Training** (RECOMMENDED for first attempt)
- [ ] Migrate existing single embedding to dual
- [ ] Freeze linguistic, train only topical
- [ ] 50 epochs topical training
- [ ] Fine-tune both together for 20 epochs

**Deliverable:** `models/dual_embeddings/best_model.pt`

### Phase 4: Evaluation (Week 2, Day 5)
**Goal:** Validate embeddings work

- [ ] Intrinsic eval: linguistic correlation (should stay ~0.89)
- [ ] Intrinsic eval: topical correlation (target >0.70)
- [ ] Embedding visualization: t-SNE plots
- [ ] Check topical clustering (Esperanto-related words cluster?)

### Phase 5: Retrieval Integration (Week 3, Days 1-3)
**Goal:** Use dual embeddings in retrieval

- [ ] Update `SlotBasedIndexer` to support dual embeddings
- [ ] Rebuild HNSW index with combined (128d) embeddings
- [ ] Update `ASTAwareRetriever` with embedding_mode parameter
- [ ] Implement adaptive weighting

**Deliverable:** Retrieval works with dual embeddings

### Phase 6: Benchmark (Week 3, Days 4-5)
**Goal:** Measure improvement

- [ ] Run Q&A benchmark with linguistic-only
- [ ] Run Q&A benchmark with topical-only
- [ ] Run Q&A benchmark with combined (50/50)
- [ ] Run Q&A benchmark with adaptive weighting
- [ ] Compare: baseline (12%) vs dual (target 30%+)

**Deliverable:** Performance report, decision on best configuration

---

## 6. Open Questions & Decisions

### Q1: Joint vs Sequential Training?

**Recommendation: Sequential (Phase 1)**
- Start with existing linguistic embedding (already good)
- Add topical separately
- Fine-tune together if needed
- Faster iteration, less risk

**Alternative: Joint from scratch**
- If sequential doesn't work well
- Full retraining with both objectives
- Takes longer but may be more optimal

### Q2: Skip-gram Window Size?

**Test these:**
- window=3: Tight context (sentence-level)
- window=5: Medium context (paragraph-level) ← START HERE
- window=10: Broad context (document-level)

**Decision:** Run quick experiment with each, measure topical correlation.

### Q3: Negative Sampling Ratio?

**Options:**
- 1:1 (one negative per positive) - too easy
- 5:1 (five negatives per positive) ← STANDARD
- 10:1 (ten negatives) - harder, may work better

**Decision:** Start with 5:1 (standard in Word2Vec)

### Q4: Embedding Dimension?

**Current:** 64d linguistic + 64d topical = 128d combined

**Alternative:** Could use 32d + 32d = 64d total (smaller)
- Pros: Same total size as current
- Cons: Less capacity per embedding

**Recommendation:** Stick with 64d+64d. Memory cost is negligible (1.4MB vs 712KB).

### Q5: Query-Time Weighting Strategy?

**Options:**
1. **Fixed 50/50** - simple, no tuning needed
2. **Per-query-type** - different weights for definition vs factual
3. **Learned weights** - small MLP predicts weights from query
4. **User-configurable** - let users adjust via UI

**Recommendation:** Start with #2 (per-query-type), evaluate if #3 needed.

### Q6: How to Validate Topical Embeddings?

**Tests:**
1. Manual inspection: Are "Esperanto", "Zamenhof", "lingvo" close?
2. Retrieval probe: Does "Fundamento" retrieve docs with "1905"?
3. Clustering: Do topic-related words cluster in t-SNE?
4. Benchmark: Does it improve Q&A accuracy?

**Decision:** Run all four before integrating.

---

## 7. Risk Mitigation

### Risk 1: Topical training data is noisy
**Mitigation:** 
- Filter by root frequency (min=5)
- Use window size to control context scope
- Validate with manual inspection

### Risk 2: Dual embeddings don't improve retrieval
**Mitigation:**
- Evaluate intrinsically before integrating
- Try different weighting strategies
- Fall back to single embedding if needed

### Risk 3: Training takes too long
**Mitigation:**
- Use sequential training (faster)
- Start with subset of corpus (1M docs)
- Scale up if it works

### Risk 4: Backward compatibility breaks
**Mitigation:**
- Keep `use_dual_roots=False` option
- Migration script for seamless upgrade
- Extensive testing on existing functionality

---

## 8. Success Metrics

### Must Have (Phase 1-5)
- ✅ Dual embeddings load/save correctly
- ✅ Topical training data extracted (50M+ pairs)
- ✅ Model trains without errors
- ✅ Linguistic correlation stays >0.85
- ✅ Topical correlation reaches >0.65

### Should Have (Phase 6)
- ✅ Q&A benchmark accuracy improves 12% → 25%+
- ✅ Explainability shows linguistic vs topical breakdown
- ✅ Adaptive weighting outperforms fixed 50/50

### Nice to Have
- ✅ Accuracy reaches 30%+
- ✅ Topic clustering visible in t-SNE
- ✅ Users can adjust weights interactively

---

## 9. File Structure

```
klareco/
├── embeddings/
│   ├── compositional.py           # MODIFY: add use_dual_roots
│   └── dual_root_embeddings.py    # NEW: DualRootEmbeddings class
├── rag/
│   ├── slot_indexer.py            # MODIFY: support dual embeddings
│   └── ast_aware_retriever.py     # MODIFY: adaptive weighting

scripts/
├── data/
│   └── prepare_topical_pairs.py   # NEW: extract skip-gram pairs
├── training/
│   └── train_dual_embeddings.py   # NEW: joint/sequential training
└── migration/
    └── single_to_dual.py           # NEW: migration tool

models/
└── dual_embeddings/                # NEW: dual model checkpoints
    ├── best_model.pt
    ├── linguistic.pt               # Optional: separate saves
    └── topical.pt

data/
└── training/
    └── topical_skipgram_pairs.json # NEW: topical training data
```

---

## 10. Timeline Summary

**Week 1: Foundation**
- Days 1-3: Architecture + unit tests
- Days 4-5: Data preparation

**Week 2: Training**
- Days 1-4: Train dual embeddings (sequential approach)
- Day 5: Intrinsic evaluation

**Week 3: Integration**
- Days 1-3: Retrieval integration + rebuild indexes
- Days 4-5: Benchmark evaluation + analysis

**Total: 15 working days (~3 weeks)**

**Checkpoints:**
- End of Week 1: Can load/use dual embeddings, topical data ready
- End of Week 2: Trained model with good correlations
- End of Week 3: Benchmark shows improvement (target: 25%+ accuracy)

---

## 11. Next Steps

After design approval:

1. **Create implementation tasks** for each phase
2. **Set up development branch** (`feature/dual-embeddings`)
3. **Write unit tests** before implementation (TDD)
4. **Implement Phase 1** (architecture)
5. **Review and iterate** before proceeding to Phase 2

**Decision needed:** Joint vs sequential training strategy?
**Decision needed:** Skip-gram window size (3, 5, or 10)?
**Decision needed:** Target accuracy threshold to consider successful?
