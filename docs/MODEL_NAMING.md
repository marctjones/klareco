# Model Naming Convention

This document defines the naming convention for all Klareco deep learning models, replacing generic labels (M0.1, M1, etc.) with meaningful, purpose-driven names.

## Naming Principles

1. **Descriptive**: Name reflects the model's purpose
2. **Concise**: Short enough to use in conversation
3. **Memorable**: Easy to remember and distinguish
4. **Consistent**: Follow common ML naming patterns

## Model Name Mappings

| Old Label | New Name | Purpose | Mnemonic |
|-----------|----------|---------|----------|
| **M0.1** | **RootEmbedder** | Embed roots semantically | Embeds **roots** |
| **M0.2** | **MorphemeComposer** | Compose morphemes into words | **Composes morphemes** |
| **M1** | **PlausibilityFilter** | Filter implausible SVO triples | Filters by **plausibility** |
| **M2** | **RelevanceRanker** | Rank facts by query relevance | Ranks by **relevance** |
| **M3** | **ASTEncoder** | Encode AST structure to vector | Encodes **ASTs** |
| **M4** | **NodePredictor** | Predict next AST node | Predicts **nodes** |
| **M5** | **IntentClassifier** | Classify instruction intent | Classifies **intent** |
| **M6** | **DiscourseClassifier** | Classify discourse relations | Classifies **discourse** |

## Detailed Model Descriptions

### Foundation Models

#### RootEmbedder (formerly M0.1)
```python
# klareco/models/root_embedder.py
class RootEmbedder(nn.Module):
    """
    Embeds Esperanto roots into semantic space.

    Purpose: Capture semantic similarity between roots (e.g., hund ≈ best)
    Size: 40K roots × 64 dims = 500K params
    Training: Contrastive learning on semantic pairs
    Use: Query expansion, synonym detection, semantic similarity
    """
```

**File locations**:
- Model: `klareco/models/root_embedder.py`
- Weights: `models/root_embedder/best_model.pt`
- Training: `scripts/train_root_embedder.sh`
- Tests: `tests/test_root_embedder.py`

---

#### MorphemeComposer (formerly M0.2)
```python
# klareco/models/morpheme_composer.py
class MorphemeComposer(nn.Module):
    """
    Composes root + affix embeddings into word semantics.

    Purpose: Handle novel word combinations (e.g., rehundejo = re+hund+ej+o)
    Size: 500K params (affix embeddings + combination MLP)
    Training: Predict word semantics from morpheme composition
    Use: Zero-shot understanding of unseen words
    """
```

**File locations**:
- Model: `klareco/models/morpheme_composer.py`
- Weights: `models/morpheme_composer/best_model.pt`
- Training: `scripts/train_morpheme_composer.sh`
- Tests: `tests/test_morpheme_composer.py`

---

### Retrieval Models

#### PlausibilityFilter (formerly M1)
```python
# klareco/models/plausibility_filter.py
class PlausibilityFilter(nn.Module):
    """
    Filters semantically implausible SVO triples.

    Purpose: Remove nonsense facts (e.g., "table founded language")
    Size: 2M params (SVO triple classifier)
    Training: Binary classification on plausible/implausible triples
    Use: Improve reasoning accuracy, reduce hallucinations
    """
```

**File locations**:
- Model: `klareco/models/plausibility_filter.py`
- Weights: `models/plausibility_filter/best_model.pt`
- Training: `scripts/train_plausibility_filter.sh`
- Tests: `tests/test_plausibility_filter.py`

---

#### RelevanceRanker (formerly M2)
```python
# klareco/models/relevance_ranker.py
class RelevanceRanker(nn.Module):
    """
    Ranks retrieved facts by query relevance.

    Purpose: Prioritize most relevant facts for reasoning
    Size: 5M params (cross-encoder)
    Training: Pairwise ranking on query-fact pairs
    Use: Improve retrieval precision, order facts for reasoning
    """
```

**File locations**:
- Model: `klareco/models/relevance_ranker.py`
- Weights: `models/relevance_ranker/best_model.pt`
- Training: `scripts/train_relevance_ranker.sh`
- Tests: `tests/test_relevance_ranker.py`

---

### Generation Models

#### ASTEncoder (formerly M3)
```python
# klareco/models/ast_encoder.py
class ASTEncoder(nn.Module):
    """
    Encodes AST structure into dense vector.

    Purpose: Capture AST context for downstream tasks
    Size: 8M params (Child-Sum TreeLSTM)
    Training: Trained jointly with downstream models
    Use: Generation, intent classification, discourse modeling
    """
```

**File locations**:
- Model: `klareco/models/ast_encoder.py` (already exists as `tree_lstm.py`)
- Weights: Trained jointly with downstream models
- Tests: `tests/test_ast_encoder.py`

---

#### NodePredictor (formerly M4)
```python
# klareco/models/node_predictor.py
class NodePredictor(nn.Module):
    """
    Predicts next AST node for text generation.

    Purpose: Generate grammatically correct Esperanto text
    Size: 12M params (multi-head classifier)
    Training: Next-node prediction on parsed corpus
    Use: Text completion, paraphrasing, abstractive generation
    """
```

**File locations**:
- Model: `klareco/models/node_predictor.py`
- Weights: `models/node_predictor/best_model.pt`
- Training: `scripts/train_node_predictor.sh`
- Tests: `tests/test_node_predictor.py`

---

### Instruction & Chat Models (Optional)

#### IntentClassifier (formerly M5)
```python
# klareco/models/intent_classifier.py
class IntentClassifier(nn.Module):
    """
    Classifies instruction intent (QA, summarization, etc.).

    Purpose: Route ambiguous instructions to appropriate expert
    Size: 5M params (ASTEncoder + MLP classifier)
    Training: Instruction-intent pairs (6-10 classes)
    Use: Automatic task routing (fallback for deterministic patterns)
    """
```

**File locations**:
- Model: `klareco/models/intent_classifier.py`
- Weights: `models/intent_classifier/best_model.pt`
- Training: `scripts/train_intent_classifier.sh`
- Tests: `tests/test_intent_classifier.py`

**Note**: Optional - 85% of instructions handled by deterministic patterns

---

#### DiscourseClassifier (formerly M6)
```python
# klareco/models/discourse_classifier.py
class DiscourseClassifier(nn.Module):
    """
    Classifies discourse relations between conversation turns.

    Purpose: Improve multi-turn chat response quality
    Size: 10M params (dual ASTEncoder + relation classifier)
    Training: Multi-turn dialogues with relation annotations (6 classes)
    Use: Better conversational flow, acknowledge topic shifts
    """
```

**File locations**:
- Model: `klareco/models/discourse_classifier.py`
- Weights: `models/discourse_classifier/best_model.pt`
- Training: `scripts/train_discourse_classifier.sh`
- Tests: `tests/test_discourse_classifier.py`

**Note**: Optional - 90% of chat works with deterministic coreference

---

## Usage in Code

### Old way (deprecated):
```python
from klareco.models.tree_lstm import TreeLSTMEncoder

m0_1 = load_model("models/root_embeddings/best_model.pt")
m1 = PlausibilityFilter()
```

### New way:
```python
from klareco.models import RootEmbedder, ASTEncoder, PlausibilityFilter

root_embedder = RootEmbedder.load("models/root_embedder/best_model.pt")
plausibility_filter = PlausibilityFilter.load("models/plausibility_filter/best_model.pt")
```

## Directory Structure

```
klareco/
└── models/
    ├── __init__.py                 # Import all models
    ├── root_embedder.py            # RootEmbedder (M0.1)
    ├── morpheme_composer.py        # MorphemeComposer (M0.2)
    ├── plausibility_filter.py      # PlausibilityFilter (M1)
    ├── relevance_ranker.py         # RelevanceRanker (M2)
    ├── ast_encoder.py              # ASTEncoder (M3, rename from tree_lstm.py)
    ├── node_predictor.py           # NodePredictor (M4)
    ├── intent_classifier.py        # IntentClassifier (M5)
    └── discourse_classifier.py     # DiscourseClassifier (M6)

models/  (trained weights)
├── root_embedder/
│   ├── best_model.pt
│   └── MODEL_CARD.md
├── morpheme_composer/
│   ├── best_model.pt
│   └── MODEL_CARD.md
├── plausibility_filter/
│   ├── best_model.pt
│   └── MODEL_CARD.md
├── relevance_ranker/
│   ├── best_model.pt
│   └── MODEL_CARD.md
├── node_predictor/
│   ├── best_model.pt
│   └── MODEL_CARD.md
├── intent_classifier/
│   ├── best_model.pt
│   └── MODEL_CARD.md
└── discourse_classifier/
    ├── best_model.pt
    └── MODEL_CARD.md

scripts/
├── train_root_embedder.sh          # Train RootEmbedder
├── train_morpheme_composer.sh      # Train MorphemeComposer
├── train_plausibility_filter.sh    # Train PlausibilityFilter
├── train_relevance_ranker.sh       # Train RelevanceRanker
├── train_node_predictor.sh         # Train NodePredictor
├── train_intent_classifier.sh      # Train IntentClassifier
└── train_discourse_classifier.sh   # Train DiscourseClassifier
```

## Configuration Options with New Names

### Minimal (21M params)
```python
MINIMAL_CONFIG = {
    'foundation': ['RootEmbedder', 'MorphemeComposer'],
    'generation': ['ASTEncoder', 'NodePredictor'],
}
```

### Standard (28M params)
```python
STANDARD_CONFIG = {
    'foundation': ['RootEmbedder', 'MorphemeComposer'],
    'retrieval': ['PlausibilityFilter', 'RelevanceRanker'],
    'generation': ['ASTEncoder', 'NodePredictor'],
}
```

### Full (43M params)
```python
FULL_CONFIG = {
    'foundation': ['RootEmbedder', 'MorphemeComposer'],
    'retrieval': ['PlausibilityFilter', 'RelevanceRanker'],
    'generation': ['ASTEncoder', 'NodePredictor'],
    'instruction': ['IntentClassifier'],
    'chat': ['DiscourseClassifier'],
}
```

## Training Pipeline with New Names

```
Stage 0: Parser (deterministic) ✅
  ↓
Stage 1: RootEmbedder (500K) 🔄
  ↓
Stage 2: MorphemeComposer (500K) 📝
  ↓
Stage 3: PlausibilityFilter (2M) 📝
  ↓
Stage 4: RelevanceRanker (5M) 📝
  ↓
Stage 5: ASTEncoder + NodePredictor (8M + 12M) 📝
  ↓
Stage 6 (Optional): IntentClassifier (5M) 📝
  ↓
Stage 7 (Optional): DiscourseClassifier (10M) 📝
```

## Migration Guide

### Renaming Checklist

For each model, update:
- [ ] Model class name in `klareco/models/*.py`
- [ ] File name: `klareco/models/*.py`
- [ ] Import in `klareco/models/__init__.py`
- [ ] Training script: `scripts/train_*.sh`
- [ ] Model weights directory: `models/*/`
- [ ] Test file: `tests/test_*.py`
- [ ] Documentation references
- [ ] Issue references (update titles/descriptions)

### Example Migration

**RootEmbedder (M0.1)**:
1. Rename file: `klareco/embeddings/compositional.py` → `klareco/models/root_embedder.py`
2. Rename class: `CompositionalEmbedding` → `RootEmbedder`
3. Rename script: `scripts/train_roots.sh` → `scripts/train_root_embedder.sh`
4. Rename directory: `models/root_embeddings/` → `models/root_embedder/`
5. Update imports throughout codebase
6. Update issue #685 title: "Train RootEmbedder for semantic similarity"

## Benefits of New Names

1. **Self-documenting**: `RootEmbedder` is clearer than `M0.1`
2. **Easier communication**: "The RelevanceRanker improves precision" vs "M2 improves precision"
3. **Better code navigation**: File names match model purpose
4. **Reduced cognitive load**: No need to remember M1=selectional, M2=reranker, etc.
5. **Professional naming**: Follows industry standards

## Backward Compatibility

During migration, maintain aliases:
```python
# klareco/models/__init__.py
from .root_embedder import RootEmbedder
from .morpheme_composer import MorphemeComposer
# ... other models

# Backward compatibility aliases (deprecated)
M0_1 = RootEmbedder  # Deprecated: use RootEmbedder
M0_2 = MorphemeComposer  # Deprecated: use MorphemeComposer
M1 = PlausibilityFilter  # Deprecated: use PlausibilityFilter
# ... etc
```

## Next Steps

1. Update `MODEL_INVENTORY.md` with new names
2. Update `MODEL_ISSUES_REFERENCE.md` with new names
3. Update `DETERMINISTIC_VS_LEARNED.md` with new names
4. Update GitHub issue titles to use new names
5. Create migration script to rename files/directories
6. Update training scripts with new names
7. Update tests with new names
