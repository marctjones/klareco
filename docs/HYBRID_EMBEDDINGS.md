# Hybrid Root Embeddings

## Overview

Klareco uses a **hybrid approach** for root embeddings that combines two complementary models:

1. **Production Model** (positional window, 6,719 roots)
   - Excellent semantic clustering (0.342 coherence)
   - Broad vocabulary coverage
   - Learned from 34M co-occurrence pairs

2. **AST-Only Model** (structural, 2,369 roots)
   - Systematic antonym detection (76.5% negative rate)
   - AST-grounded (authoritative Fundamento roots)
   - Learned from 35K AST-aware pairs

**Result**: Best-of-both-worlds quality (90/100) with zero additional training cost.

## Performance Comparison

| Model | Score | Antonyms | Clustering | Vocabulary |
|-------|-------|----------|------------|------------|
| Production | 50.0/100 | 20/100 ❌ | 80/100 ✓ | 6,719 |
| AST-Only | 90.0/100 | 100/100 ✓✓ | 80/100 ✓ | 2,369 |
| **Hybrid** | **90.0/100** | **100/100** ✓✓ | **80/100** ✓ | **7,843** ✓✓ |

**Key Insight**: Hybrid matches AST-Only quality while providing **3.3x more vocabulary coverage** (7,843 vs 2,369 roots).

## How It Works

### Three-Tier Selection Strategy

The hybrid embedder intelligently chooses which model to use based on query type:

```
Query: similarity(root1, root2)
  │
  ├─ TIER 1: Antonym pair (mal-)?
  │   └─ YES → Use AST model (explicit negation)
  │
  ├─ TIER 2: Both in Fundamento?
  │   └─ YES → Use AST model (authoritative)
  │
  └─ TIER 3: General similarity
      └─ Use Production model (better clustering)
```

### Model Selection Examples

| Query | Model Used | Rationale |
|-------|-----------|-----------|
| `am` vs `malam` | **AST** | Antonym pair (mal-) |
| `hund` vs `kat` | **AST** | Both in Fundamento |
| `hipopotam` vs `elefant` | **Production** | Rare roots (coverage) |
| Nearest neighbors (clustering) | **Production** | Better clustering |
| Nearest neighbors (structural) | **AST** | AST-grounded relationships |

## Usage

### Basic Example

```python
from klareco.embeddings import load_hybrid_embedder

# Load hybrid embedder (uses default paths)
hybrid = load_hybrid_embedder()

# Compute similarity
sim, source = hybrid.similarity("am", "malam")
print(f"Similarity: {sim:.3f} (from {source})")
# Output: Similarity: -0.390 (from AST-antonym)

# Find nearest neighbors (clustering)
neighbors = hybrid.nearest_neighbors("hund", k=5, use_clustering=True)
for neighbor, sim, source in neighbors:
    print(f"  {neighbor}: {sim:.3f} ({source})")
```

### Custom Model Paths

```python
from klareco.embeddings import HybridRootEmbedder

hybrid = HybridRootEmbedder(
    production_path="path/to/production/model.pt",
    ast_path="path/to/ast/model.pt",
    device="cuda"  # or "cpu"
)
```

### Integration with CompositionalEmbedding

```python
from klareco.embeddings import CompositionalEmbedding, load_hybrid_embedder

# Load hybrid embedder
hybrid = load_hybrid_embedder()

# Use hybrid vocabulary
compositional = CompositionalEmbedding(
    root_vocab=hybrid.root_to_idx,
    prefix_vocab=prefix_vocab,
    suffix_vocab=suffix_vocab,
    embed_dim=128
)

# Note: For full integration, root embeddings would need to be
# replaced with hybrid.get_embedding() calls
```

## Command-Line Demo

```bash
# Basic demo
python scripts/demo_hybrid_embeddings.py

# Interactive mode
python scripts/demo_hybrid_embeddings.py --interactive
```

### Interactive Commands

```
hybrid> sim am malam
  am vs malam: -0.390 (AST-antonym)

hybrid> neighbors hund 5
  Top 5 neighbors for 'hund' (clustering):
    bird: 0.384
    kok: 0.380
    dorm: 0.372
    naz: 0.346
    griz: 0.346

hybrid> structural bel 5
  Top 5 neighbors for 'bel' (structural):
    konversaci: 0.572
    poet: 0.514
    kandel: 0.495
    novjar: 0.475
    vals: 0.468

hybrid> info hund
  Root: 'hund'
    In Production: ✓
    In AST: ✓
    In Unified: ✓
    Unified index: 3245

hybrid> stats
  Vocabulary Coverage:
    Total roots: 7,843
    Production: 6,719
    AST: 2,369
    Overlap: 1,245 (18.5%)
```

## Technical Details

### Vocabulary Coverage

- **Total unique roots**: 7,843
- **Production vocabulary**: 6,719 roots
  - Learned from 34M co-occurrence pairs
  - 5.4M sentences across all sources
- **AST vocabulary**: 2,369 roots
  - Learned from 35K AST-aware pairs
  - 23K GOLD sentences (authoritative)
- **Overlap**: 1,245 roots (18.5%)
- **Production-only**: 5,474 roots (rare/additional)
- **AST-only**: 1,124 roots (Fundamento-specific)

### Antonym Detection

The AST model systematically learns antonyms through the `mal-` prefix:

```python
# Training includes negative similarity pairs:
("am", "malam", -0.7)    # love vs hate
("bon", "malbon", -0.7)  # good vs bad
("alt", "malalt", -0.7)  # high vs low
```

**Results**:
- Mean similarity: **-0.194** (negative, correct!)
- Negative rate: **76.5%** (most antonym pairs are negative)
- Tested on 85 antonym pairs

### Semantic Clustering

The Production model learns distributional semantics from co-occurrence:

**Cluster Coherence**:
- Colors: 0.498 (excellent)
- Animals: 0.153 (good)
- Actions: 0.152 (good)
- **Overall**: 0.267 (strong clustering)

### Model Specifications

| Metric | Production | AST-Only |
|--------|-----------|----------|
| **Training Pairs** | 34M | 35K |
| **Vocabulary** | 6,719 | 2,369 |
| **Embedding Dim** | 128D | 64D |
| **Parameters** | 1.7M | 303K |
| **Model Size** | 20 MB | 3.6 MB |
| **Training Time** | 2-3 hours | 30 min |

### Memory Requirements

- **Production model**: ~20 MB
- **AST model**: ~4 MB
- **Total (both loaded)**: ~24 MB
- **Unified vocabulary**: Minimal overhead

## Advantages Over Single Models

### 1. Zero Training Cost
- Both models already trained
- No additional compute required
- Immediate deployment

### 2. Best-of-Both-Worlds Quality
- Antonym detection: **100/100** (from AST)
- Semantic clustering: **80/100** (from Production)
- Vocabulary coverage: **7,843 roots** (combined)

### 3. Explainability
- Source tracking: Know which model provided each result
- AST model: Traceable to AST structure
- Production model: Learned from corpus statistics

### 4. Flexibility
- Can update either model independently
- Can add additional specialized models
- Adjustable selection strategy

### 5. Architectural Alignment
- Uses AST model for Fundamento (authoritative)
- Maintains AST-first architecture principles
- Falls back to Production only when needed

## Comparison to Enhanced Model

We tried training an "Enhanced" model that combined AST + co-occurrence data. **It failed** (66.7/100):

| Approach | Score | Antonyms | Clustering | Training |
|----------|-------|----------|------------|----------|
| Enhanced (trained) | 66.7/100 ❌ | 80/100 | **40/100** ❌ | Required |
| **Hybrid (combined)** | **90.0/100** ✓ | **100/100** ✓ | **80/100** ✓ | **Not required** ✓ |

**Why Enhanced Failed**:
- Co-occurrence (72%) drowned out AST signal (28%)
- Insufficient co-occurrence data (91K << 34M)
- Clustering degraded: 0.253 → 0.188

**Why Hybrid Succeeds**:
- Chooses right model for each query
- Keeps AST and Production signals separate
- No training required, no risk of degradation

## Future Enhancements

### 1. Lazy Loading
Currently both models are loaded into memory. Could implement:
- Load only when needed
- Cache frequently-queried embeddings
- Unload unused model after timeout

### 2. Confidence Weighting
Instead of binary selection, could blend both models:
```python
# Weighted combination based on confidence
emb = alpha * ast_emb + (1 - alpha) * prod_emb
```

### 3. Additional Specialized Models
Could add more models for specific purposes:
- Technical vocabulary model
- Proper names model
- Domain-specific models (medical, legal, etc.)

### 4. Adaptive Selection
Learn which model works best for which queries:
- Track query success rate per model
- Adjust selection heuristics over time
- User feedback integration

## Implementation Files

- **Module**: `klareco/embeddings/hybrid.py`
- **Demo**: `scripts/demo_hybrid_embeddings.py`
- **Tests**: `tests/test_hybrid_embeddings.py` (to be created)
- **Documentation**: This file

## References

- Production model: `models/root_embeddings_phase1_fast/root_embeddings_best.pt`
- AST-Only model: `models/root_embeddings_fundamento_ast/root_embeddings_best.pt`
- Training scripts:
  - `scripts/train_root_embeddings.sh` (Production)
  - `scripts/train_fundamento_ast_embeddings.sh` (AST-Only)
- Evaluation: `/tmp/three_way_model_comparison.md`

## See Also

- [Compositional Embeddings](../klareco/embeddings/compositional.py) - Main embedding interface
- [AST Structure](16RULES.MD) - Esperanto grammar rules
- [Vision](VISION.md) - AST-first architecture principles
- [Training Roadmap](IMPLEMENTATION_ROADMAP_V2.md) - Model training stages
