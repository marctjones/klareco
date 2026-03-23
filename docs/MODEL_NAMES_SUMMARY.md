# Model Names Summary

Quick reference showing the new meaningful names for all Klareco models.

## Name Mappings

| Old Label | New Name | One-Line Purpose |
|-----------|----------|------------------|
| M0.1 | **RootEmbedder** | Embeds roots semantically (hund ≈ best) |
| M0.2 | **MorphemeComposer** | Composes morphemes into words (re+hund+ej+o) |
| M1 | **PlausibilityFilter** | Filters implausible SVO triples |
| M2 | **RelevanceRanker** | Ranks facts by query relevance |
| M3 | **ASTEncoder** | Encodes AST structure to vector |
| M4 | **NodePredictor** | Predicts next AST node |
| M5 | **IntentClassifier** | Classifies instruction intent |
| M6 | **DiscourseClassifier** | Classifies discourse relations |

## Why the Change?

**Before**: "Train M1 model" - What is M1?
**After**: "Train PlausibilityFilter" - Immediately clear!

## Usage Examples

### Documentation
✅ **Good**: "The RootEmbedder captures semantic similarity between roots"
❌ **Old**: "M0.1 captures semantic similarity between roots"

### Code
✅ **Good**:
```python
from klareco.models import RootEmbedder, PlausibilityFilter

embedder = RootEmbedder.load("models/root_embedder/best_model.pt")
filter = PlausibilityFilter.load("models/plausibility_filter/best_model.pt")
```

❌ **Old**:
```python
m0_1 = load_model("models/root_embeddings/best_model.pt")
m1 = load_model("models/m1/best_model.pt")
```

### Conversation
✅ **Good**: "We need to train the PlausibilityFilter to reduce hallucinations"
❌ **Old**: "We need to train M1 to reduce hallucinations"

## Updated Documentation

All documentation has been updated to use the new names:
- ✅ `docs/MODEL_NAMING.md` - Detailed naming specification
- ✅ `docs/MODEL_INVENTORY.md` - Model catalog
- ✅ `docs/MODEL_ISSUES_REFERENCE.md` - Issue mapping
- ✅ `docs/DETERMINISTIC_VS_LEARNED.md` - Architectural analysis

## Configuration Names

### Minimal (21M params)
- RootEmbedder
- MorphemeComposer
- ASTEncoder
- NodePredictor

### Standard (28M params)
Minimal +
- PlausibilityFilter
- RelevanceRanker

### Full (43M params)
Standard +
- IntentClassifier
- DiscourseClassifier

## Quick Reference Card

```
Foundation (Required):
├─ RootEmbedder (500K)      - Semantic similarity
└─ MorphemeComposer (500K)   - Morpheme composition

Retrieval (Standard):
├─ PlausibilityFilter (2M)   - Filter implausible facts
└─ RelevanceRanker (5M)      - Rank by relevance

Generation (Required):
├─ ASTEncoder (8M)           - Encode AST context
└─ NodePredictor (12M)       - Predict next node

Optional (Full):
├─ IntentClassifier (5M)     - Classify intent (or use patterns)
└─ DiscourseClassifier (10M) - Classify discourse (or use rules)
```

## Training Order

```
1. RootEmbedder (500K)
2. MorphemeComposer (500K)
3. PlausibilityFilter (2M)
4. RelevanceRanker (5M)
5. ASTEncoder + NodePredictor (8M + 12M)
6. IntentClassifier (5M, optional)
7. DiscourseClassifier (10M, optional)
```

## File Locations

### Code
- `klareco/models/root_embedder.py`
- `klareco/models/morpheme_composer.py`
- `klareco/models/plausibility_filter.py`
- `klareco/models/relevance_ranker.py`
- `klareco/models/ast_encoder.py` (rename from `tree_lstm.py`)
- `klareco/models/node_predictor.py`
- `klareco/models/intent_classifier.py`
- `klareco/models/discourse_classifier.py`

### Weights
- `models/root_embedder/best_model.pt`
- `models/morpheme_composer/best_model.pt`
- `models/plausibility_filter/best_model.pt`
- `models/relevance_ranker/best_model.pt`
- `models/node_predictor/best_model.pt`
- `models/intent_classifier/best_model.pt`
- `models/discourse_classifier/best_model.pt`

### Training Scripts
- `scripts/train_root_embedder.sh`
- `scripts/train_morpheme_composer.sh`
- `scripts/train_plausibility_filter.sh`
- `scripts/train_relevance_ranker.sh`
- `scripts/train_node_predictor.sh`
- `scripts/train_intent_classifier.sh`
- `scripts/train_discourse_classifier.sh`

## Migration Notes

- Old M0.1-M6 labels are deprecated but remain in existing issues for now
- New code should use meaningful names exclusively
- Update GitHub issue titles in future milestone
- Add backward compatibility aliases during transition period

## Benefits

✅ **Self-documenting**: Name tells you what it does
✅ **Easier onboarding**: New contributors understand immediately
✅ **Better conversations**: "PlausibilityFilter" > "M1"
✅ **Professional**: Industry-standard naming
✅ **Memorable**: Purpose-based names stick in memory
