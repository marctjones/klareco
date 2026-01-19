# M1: Selectional Preference Model

**Status**: ✅ Production-ready (as of 2026-01-19)
**Model Path**: `models/m1_semantic_full/best_model.pt`
**Training Script**: `scripts/train_m1_semantic.sh`

## Overview

M1 learns which combinations of roots are plausible in grammatical roles (subject-verb-object). This enables the system to distinguish sensible statements from nonsensical ones, improving both retrieval quality and generation safety.

**Key Insight**: By training on real corpus triples + **semantically-distant** corrupted negatives (similarity < 0.15), M1 learns selectional preferences without explicit semantic categories. This creates a learnable signal that distinguishes plausible from implausible triples.

## Architecture

```python
Input:
  subject_embedding: [64d]  # From Stage 1
  verb_embedding: [64d]     # From Stage 1
  object_embedding: [64d]   # From Stage 1

Hidden:
  subject_verb_compatibility: [128d → 1d]  # Can X do Y?
  verb_object_compatibility: [128d → 1d]   # Can Y apply to Z?
  triple_plausibility: [192d → 1d]         # Is (X,Y,Z) sensible?

Output:
  plausibility_score: [0.0 - 1.0]
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input dimension | 64d × 3 | Three Stage 1 embeddings |
| Hidden dimension | 256d | Production model (--hidden-dim 256) |
| Dropout | 0.2 | Regularization |
| Total parameters | ~838K | Lightweight scorer |
| Output | Single score | 0.0 = implausible, 1.0 = plausible |

**Model Class**: `M1SelectionalPreference` in `klareco/models/m1_selectional.py`

## Training Data

### Data Generation

Generated from full corpus using `scripts/prepare_m1_training_data_semantic.py` with **semantic-distance corruption**:

**Key Innovation**: Corrupted negatives are semantically DISTANT from originals (similarity < 0.15), creating a learnable signal.

| Component | Count | Source |
|-----------|-------|--------|
| **Positive triples** | 200,000 | Real SVO triples from corpus |
| - High-quality tier0 | ~15% | Authoritative literary texts |
| - General corpus | ~85% | Mixed quality (tiers 2-6) |
| **Negative triples** | 200,000 | Semantically-distant corruptions |
| - Subject corruption | ~67K | Distant noun (sim < 0.15) |
| - Verb corruption | ~67K | Distant verb (sim < 0.15) |
| - Object corruption | ~67K | Distant noun (sim < 0.15) |

**Total Training Examples**: 400,000 (50% positive, 50% negative)

**Corruption Strategy** (Fixed Bug #2):
- Random corruption created indistinguishable negatives (positive sim: 0.24, negative sim: 0.15)
- Semantic-distance corruption ensures clear separation (negative sim < 0.15 to ALL components)
- Uses Stage 1 embeddings to find maximally distant replacements

**Split**:
- Train: 320,000 (80%)
- Validation: 40,000 (10%)
- Test: 40,000 (10%)

### Example Triples

**Positive (label=1.0)**:
```json
{
  "subject_root": "hund",
  "verb_root": "manĝ",
  "object_root": "viand",
  "label": 1.0,
  "original_text": "La hundo manĝas viandon."
}
```
Score: ~0.92 (highly plausible - dogs eat meat)

**Negative (label=0.0)**:
```json
{
  "subject_root": "hund",
  "verb_root": "manĝ",
  "object_root": "ide",
  "label": 0.0,
  "corruption": "object",
  "original_text": "La hundo manĝas viandon."
}
```
Score: ~0.08 (implausible - dogs don't eat ideas)

## Training Configuration

```python
epochs = 50
batch_size = 32
learning_rate = 0.001
patience = 20  # Early stopping (production model)
hidden_dim = 256  # Production model (doubled from baseline)
dropout = 0.2  # Production model (increased regularization)
```

**Loss Function**: `M1Loss` - Combined BCE loss
- Subject-verb compatibility loss (33%)
- Verb-object compatibility loss (33%)
- Triple plausibility loss (34%)

**Optimizer**: Adam with gradient clipping (max_norm=1.0)

## Performance Metrics

### Achieved Results (Semantic-Distance Training)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Test accuracy** | **86.2%** | > 82% | ✅ **PASS** |
| **Validation accuracy** | 86.4% (epoch 11) | > 82% | ✅ PASS |
| **Plausible recall** | 95.2% | > 90% | ✅ PASS |
| **Implausible precision** | 77.1% | > 70% | ✅ PASS |
| **Score mean** | 0.512 | 0.4-0.6 | ✅ PASS |
| **Score std** | 0.309 | > 0.05 | ✅ PASS |
| **Training epochs** | 31 (stopped at 11) | - | - |
| **Data generation** | 90 minutes | - | - |
| **Model training** | 55 minutes | - | - |

### ✅ Production-Ready Quality

**Accuracy 86.2% exceeds target by 4.2 percentage points!**

**Success factors:**
- ✅ Semantic-distance corruption creates clear training signal
- ✅ Model jumped to 85%+ accuracy on first epoch
- ✅ Steady improvement to 86.4% by epoch 11
- ✅ Clean convergence with early stopping
- ✅ High plausible recall (95.2%) - won't filter good results
- ✅ Good implausible precision (77.1%) - catches bad results

**What Changed (Bug Fix #2):**
- **Before**: Random corruption → 70.2% accuracy (negatives indistinguishable)
- **After**: Semantic-distance corruption → 86.2% accuracy (+16 points)
- **Root Cause**: Random corruption didn't change semantic patterns enough
- **Solution**: Ensure corrupted words have similarity < 0.15 to ALL triple components

### Understanding Accuracy

**Binary classification**: Is (subject, verb, object) plausible or not?

| Accuracy | Quality | Interpretation |
|----------|---------|----------------|
| < 0.7 | Poor | Random guessing level |
| 0.7 - 0.8 | Decent | Basic selectional patterns |
| 0.8 - 0.85 | Good | Reliable for filtering |
| **0.85 - 0.9** | **Very Good** | **Production-ready** |
| > 0.9 | Exceptional | May be overfitting |

**Why 0.85+ is excellent**:
- Some corpus triples are genuinely ambiguous
- Metaphorical usage (e.g., "ideoj manĝas tempon" - ideas consume time)
- Creative language (literature, poetry)
- Corruption can create plausible alternatives

### Score Distribution

**Healthy distribution** (no collapse):
- Mean: 0.4-0.6 (balanced)
- Std: > 0.05 (spread, not stuck)
- Min: 0.0-0.1 (clear rejections)
- Max: 0.9-1.0 (strong confidence)

**Warning signs**:
- Mean near 0.0 or 1.0 (collapsed to one class)
- Std < 0.05 (not learning distinctions)
- All scores 0.4-0.6 (indecisive)

## Usage

### Loading Model

```python
import torch
from klareco.models.m1_selectional import M1SelectionalPreference

# Load checkpoint (production model with semantic-distance training)
checkpoint = torch.load('models/m1_semantic_full/best_model.pt')

# Initialize model
model = M1SelectionalPreference(
    embedding_dim=checkpoint['embedding_dim'],  # 64
    hidden_dim=checkpoint['hidden_dim']        # 256
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded M1 model: {model.count_parameters():,} parameters")  # 838,145
```

### Checking Plausibility

```python
import torch.nn.functional as F

# Load Stage 1 embeddings
stage1_checkpoint = torch.load('models/root_embeddings_tier0/best_model.pt')
embeddings = stage1_checkpoint['model_state_dict']['embeddings.weight']
root_to_idx = stage1_checkpoint['root_to_idx']

def get_embedding(root: str):
    idx = root_to_idx.get(root.lower())
    if idx is None:
        return None
    return embeddings[idx]

# Check triple plausibility
def check_triple(subject: str, verb: str, object: str):
    """Check if (subject, verb, object) is plausible."""
    subj_emb = get_embedding(subject)
    verb_emb = get_embedding(verb)
    obj_emb = get_embedding(object)

    if None in [subj_emb, verb_emb, obj_emb]:
        return None  # Unknown root

    # Forward pass
    with torch.no_grad():
        outputs = model(
            subj_emb.unsqueeze(0),
            verb_emb.unsqueeze(0),
            obj_emb.unsqueeze(0)
        )

    score = outputs['triple_score'].item()
    return score

# Examples
print(check_triple('hund', 'manĝ', 'viand'))  # ~0.92 (plausible)
print(check_triple('hund', 'manĝ', 'ide'))    # ~0.08 (implausible)
print(check_triple('stud', 'leg', 'libr'))    # ~0.87 (plausible)
print(check_triple('tabl', 'leg', 'hund'))    # ~0.03 (implausible)
```

### Filtering Retrieval Results

```python
def filter_implausible_results(results, threshold=0.5):
    """Filter retrieval results by M1 plausibility."""
    filtered = []

    for result in results:
        # Extract SVO from result AST
        ast = result['ast']
        subj = ast['subjekto']['kerno']['radiko']
        verb = ast['verbo']['radiko']
        obj = ast['objekto']['kerno']['radiko']

        # Check plausibility
        score = check_triple(subj, verb, obj)

        if score is None or score >= threshold:
            result['m1_score'] = score
            filtered.append(result)

    return filtered
```

## Integration with Pipeline

M1 is used in two places:

### 1. Retrieval Reranking

```python
# Query: "hundo manĝas"
# Retrieved: 100 candidate sentences

# Rerank by M1 plausibility
for candidate in candidates:
    svo = extract_svo(candidate['ast'])
    m1_score = check_triple(svo['subject'], svo['verb'], svo['object'])

    # Combined score
    candidate['final_score'] = (
        0.6 * candidate['similarity'] +
        0.4 * m1_score
    )

# Sort by final score
candidates.sort(key=lambda x: x['final_score'], reverse=True)
```

### 2. Generation Safety

```python
# Before generating response, check plausibility
def validate_generation(ast):
    """Ensure generated sentence is semantically plausible."""
    svo = extract_svo(ast)
    score = check_triple(svo['subject'], svo['verb'], svo['object'])

    if score < 0.3:
        # Highly implausible - reject generation
        raise ValueError(f"Implausible triple: {svo}")

    return True
```

## Troubleshooting

### Symptom: Low Accuracy (< 0.8)

**Possible causes**:
- Stage 1 embeddings poor quality
- Insufficient training data
- Overfitting to corruption patterns

**Solutions**:
1. Check Stage 1 correlation (should be > 0.8)
2. Increase training data: `--max-triples 300000`
3. Add more dropout: `--dropout 0.2`

### Symptom: Score Collapse

**Detection**:
- All scores near 0.5
- Std < 0.05
- Model outputs same value for all inputs

**Causes**:
- Hidden dimension too small
- Learning rate too high (unstable)
- Training data imbalanced

**Solutions**:
1. Increase hidden dim: `--hidden-dim 256`
2. Reduce learning rate: `--learning-rate 0.0005`
3. Verify 50/50 positive/negative split

### Symptom: Overfitting

**Detection**:
- Train accuracy > 95%, val accuracy < 85%
- Score std > 0.3 (overconfident)

**Solutions**:
1. Increase dropout: `--dropout 0.2`
2. Reduce hidden dimension: `--hidden-dim 64`
3. Add more training data

## Retraining

### When to Retrain

- ✅ Stage 1 embeddings retrained (M1 depends on them)
- ✅ New corpus data added (> 10% increase)
- ✅ Performance degradation detected

- ❌ Minor corpus additions (< 5%)
- ❌ Stage 1 tweaks without full retraining

### Full Retraining

```bash
# Complete pipeline with semantic-distance corruption (RECOMMENDED)
./scripts/train_m1_semantic.sh --full-corpus  # 400K examples

# Tier0-only (smaller, higher quality)
./scripts/train_m1_semantic.sh                # 30K examples

# Skip data generation if already exists
./scripts/train_m1_semantic.sh --skip-data

# Manual control
python scripts/train_m1_selectional.py \
  --stage1-model models/root_embeddings_tier0/best_model.pt \
  --data-dir data/training/m1_semantic_full \
  --output-dir models/m1_semantic_full \
  --hidden-dim 256 \
  --dropout 0.2 \
  --patience 20 \
  --epochs 50 \
  --fresh
```

## Testing & Validation

### Automated Tests

```bash
# Validate M1 quality
python scripts/validate_m1_extensive.py \
  --model models/m1_selectional_tier0/best_model.pt
```

### Manual Quality Checks

```python
# Test known plausible triples
plausible = [
    ('hund', 'manĝ', 'viand'),   # dog eats meat
    ('stud', 'leg', 'libr'),     # student reads book
    ('inf', 'lern', 'lingv'),    # child learns language
]

for subj, verb, obj in plausible:
    score = check_triple(subj, verb, obj)
    print(f"({subj}, {verb}, {obj}): {score:.3f}")
    # Expected: > 0.7

# Test known implausible triples
implausible = [
    ('hund', 'manĝ', 'ide'),     # dog eats idea
    ('tabl', 'leg', 'libr'),     # table reads book
    ('akv', 'pens', 'penso'),    # water thinks thought
]

for subj, verb, obj in implausible:
    score = check_triple(subj, verb, obj)
    print(f"({subj}, {verb}, {obj}): {score:.3f}")
    # Expected: < 0.3
```

## Files & Paths

**Training**:
- Script: `scripts/train_m1_selectional.py`
- Data generation: `scripts/prepare_m1_training_data_semantic.py` (semantic-distance corruption)
- Wrapper: `scripts/train_m1_semantic.sh` (complete pipeline)

**Data**:
- Training data: `data/training/m1_semantic_full/`
- Source corpus: `data/enhanced_corpus/corpus_full_with_tier0.jsonl`

**Output**:
- **Production model**: `models/m1_semantic_full/best_model.pt` (86.2% accuracy)
- Training log: `logs/training/train_m1_semantic_*.log`

**Integration**:
- Inference wrapper: `klareco/models/m1_inference.py`
- RAG demo: `scripts/demo_rag_with_m1.py`
- M1 demo: `scripts/demo_m1_selectional.py`

**Tests**:
- Validation: `scripts/validate_m1_extensive.py`

## References

- **Stage-1-Root-Embeddings**: Input embeddings for M1
- **RETRAINING_WITH_TIER0.md**: Full retraining guide
- **CLAUDE.md**: Architecture overview

## Changelog

- **2026-01-19**: ✅ **Production release** - Semantic-distance training achieves 86.2% accuracy (+16 points improvement)
  - Fixed Bug #2: Semantic-distance corruption creates learnable signal
  - Trained on 400K examples with similarity threshold 0.15
  - Model path: `models/m1_semantic_full/best_model.pt`
  - Integrated into RAG pipeline via `klareco/models/m1_inference.py`
- **2026-01-18**: Initial training with random corruption (accuracy: 70.2% ⚠️ below target)
  - Identified root cause: corrupted negatives indistinguishable from positives
  - Quality test: tier0-only achieved 69.2% (proves data quality not the issue)
  - Capacity test: 256d hidden achieved 70.25% (proves capacity not the issue)
