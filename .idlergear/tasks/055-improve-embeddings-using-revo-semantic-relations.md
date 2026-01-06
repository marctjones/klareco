---
id: 55
title: Improve embeddings using ReVo semantic relations (contrastive learning)
state: open
created: '2026-01-05T16:28:46.962529Z'
labels:
- enhancement
- training
- embeddings
priority: high
---
## Goal
Use the 9,304 semantic relation pairs from ReVo to improve embedding quality through contrastive learning - train embeddings to pull synonyms closer and push antonyms/random words apart.

## Motivation
Current embeddings may not capture semantic similarity well because they were trained only on co-occurrence patterns. ReVo provides explicit human-curated semantic judgments (synonyms, antonyms, etc.) that we can use as supervision signal.

## Available Data
From `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`:
- 1,943 synonym pairs (positive examples)
- 173 antonym pairs (negative examples)
- 3,351 hypernym pairs (could use for hierarchy)
- 1,098 hyponym pairs
- Total: 9,304 semantic relations

## Approach: Contrastive Fine-Tuning

### Loss Function
Use **triplet loss** with margin:

```
L(anchor, positive, negative) = max(0, margin + sim(anchor, positive) - sim(anchor, negative))
```

For each synonym pair (A, B):
- **Anchor**: embedding of word A
- **Positive**: embedding of word B (synonym)
- **Negative**: embedding of random word C OR antonym

**Goal**: Make `sim(A, B) > sim(A, C) + margin`

### Training Strategy

**Option 1: Fine-tune entire model**
- Load existing compositional embedding model
- Add contrastive loss to training objective
- Fine-tune for 10-20 epochs on ReVo pairs

**Option 2: Learn adjustment layer**
- Keep base embeddings frozen
- Add small learned transformation: `emb' = W * emb + b`
- Train only W and b on contrastive objective
- Preserves morphological structure from base model

**Recommended**: Option 2 (safer, faster)

### Data Preparation

Create training pairs:
```python
def create_triplets(synonym_pairs, antonym_pairs, vocab):
    triplets = []
    
    # Positive triplets from synonyms
    for (A, B) in synonym_pairs:
        # Sample hard negative (random word)
        C = random.choice(vocab)
        triplets.append((A, B, C))
    
    # Negative triplets from antonyms
    for (A, B) in antonym_pairs:
        # Sample positive (random synonym if available)
        # OR use A again with margin
        triplets.append((A, A, B))  # Push antonyms apart
    
    return triplets
```

### Implementation

Create script: `scripts/train_embeddings_contrastive.py`

```python
class ContrastiveLearner:
    def __init__(self, base_embedding_model):
        self.base_model = base_embedding_model
        # Option 2: Add small adjustment layer
        self.adjustment = nn.Linear(128, 128)
        
    def forward(self, word):
        base_emb = self.base_model.embed(word)
        adjusted_emb = self.adjustment(base_emb)
        return F.normalize(adjusted_emb)  # L2 normalize
        
    def triplet_loss(self, anchor, positive, negative, margin=0.2):
        sim_pos = F.cosine_similarity(anchor, positive)
        sim_neg = F.cosine_similarity(anchor, negative)
        loss = F.relu(margin + sim_neg - sim_pos)
        return loss.mean()
        
    def train_epoch(self, triplets):
        for (A, B, C) in triplets:
            emb_A = self.forward(A)
            emb_B = self.forward(B)
            emb_C = self.forward(C)
            
            loss = self.triplet_loss(emb_A, emb_B, emb_C)
            loss.backward()
            optimizer.step()
```

### Training Configuration

```python
{
  "base_model": "models/root_embeddings/best_model.pt",
  "learning_rate": 1e-4,
  "batch_size": 128,
  "epochs": 20,
  "margin": 0.2,
  "negative_samples_per_positive": 3,
  "validation_split": 0.1,
  "early_stopping_patience": 3
}
```

### Validation

Split ReVo pairs into train/val:
- 90% train (1,748 synonym pairs)
- 10% val (195 synonym pairs)

**Validation metrics**:
- Mean synonym similarity (should increase)
- Synonym vs random gap (should widen)
- Check for embedding collapse (mean sim <0.5)

### Output

Save improved model: `models/root_embeddings/contrastive_model.pt`

Log training: `logs/training/contrastive_TIMESTAMP.log`

Save metrics: `benchmark_results/embeddings/contrastive_training_TIMESTAMP.json`

## Expected Improvement

**Before** (hypothetical current state):
- Mean synonym similarity: ~0.55
- Synonym vs random gap: ~0.10

**After** (target):
- Mean synonym similarity: >0.75 (+0.20)
- Synonym vs random gap: >0.25 (+0.15)

## Risks

1. **Overfitting**: Only 1,943 synonym pairs, could overfit
   - Mitigation: Use validation split, early stopping
   
2. **Embedding collapse**: All embeddings become similar
   - Mitigation: Monitor mean similarity, add regularization
   
3. **Breaking morphology**: Fine-tuning could break affix composition
   - Mitigation: Use Option 2 (adjustment layer), keep base frozen

## Success Criteria
- Task #54 evaluation shows mean synonym similarity >0.75
- No embedding collapse (mean_sim <0.5)
- Synonym vs random gap >0.25
- Morphological tests still pass (mal- consistency, etc.)

## Effort
8-12 hours (implementation + training + validation)

## Dependencies
- Task #54 (run evaluation first to establish baseline)
- Current compositional embedding model
- ReVo semantic relations (already downloaded)

## Related
- This is an alternative to Tasks #40-48 (embedding investigation)
- If this succeeds, may not need full investigation
- Aligns with "improve then evaluate" workflow
