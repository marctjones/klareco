---
id: 57
title: Design contrastive learning to respect compositional embeddings
state: open
created: '2026-01-05T16:31:46.442004Z'
labels:
- design
- embeddings
- training
priority: high
---
## Problem
Standard contrastive learning treats words as atomic units, but our embeddings are **compositional** - built from roots, prefixes, suffixes, and grammatical features.

**We need to train at the RIGHT level** to preserve compositionality.

## Wrong Approach (Standard Contrastive Learning)

```python
# BAD: Train on full word embeddings
emb_krei = model.embed("krei")
emb_establi = model.embed("establi")
loss = triplet_loss(emb_krei, emb_establi, emb_random)
```

**Problem**: This trains the COMPOSED embedding, not the components. Doesn't generalize to:
- "kreado" vs "establado"
- "malkrei" vs "malestabli"
- "kreinto" vs "establinto"

## Correct Approach (Compositional Contrastive Learning)

### Strategy 1: Root-Level Contrastive Learning

Train only root embeddings, keep affix embeddings frozen:

```python
# GOOD: Train at root level
root1 = "kre"  # from "krei"
root2 = "establ"  # from "establi"

# Get root embeddings (64d)
emb_root1 = model.root_embeddings[root1]
emb_root2 = model.root_embeddings[root2]
emb_random_root = model.root_embeddings[random_root]

# Contrastive loss on ROOTS only
loss = triplet_loss(emb_root1, emb_root2, emb_random_root)
```

**Benefit**: Root similarity automatically transfers to ALL derived forms through composition.

### Strategy 2: Compositional Consistency Loss

Enforce that synonym roots produce similar embeddings REGARDLESS of affixes:

```python
# If "kre" ≈ "establ", then for ANY affixes (prefix, suffix, ending):
# compose(kre, affixes) ≈ compose(establ, same_affixes)

def compositional_consistency_loss(root1, root2, model):
    losses = []
    
    # Test with various affixes
    for ending in ['i', 'o', 'a', 'e']:
        word1 = root1 + ending
        word2 = root2 + ending
        
        emb1 = model.embed(word1)  # Compositional embedding
        emb2 = model.embed(word2)
        
        # These should be similar
        losses.append(1 - cosine_similarity(emb1, emb2))
    
    for prefix in ['mal', 'ek', 're']:
        word1 = prefix + root1 + 'i'
        word2 = prefix + root2 + 'i'
        
        emb1 = model.embed(word1)
        emb2 = model.embed(word2)
        
        losses.append(1 - cosine_similarity(emb1, emb2))
    
    return mean(losses)
```

**Benefit**: Explicitly trains for compositional consistency.

### Strategy 3: Multi-Level Contrastive Learning

Train at BOTH root level AND composed level:

```python
def multi_level_loss(word1, word2, random_word, model):
    # Decompose into components
    m1 = decompose(word1)  # {root: "kre", prefix: None, suffix: [], ending: "i"}
    m2 = decompose(word2)  # {root: "establ", prefix: None, suffix: [], ending: "i"}
    m_rand = decompose(random_word)
    
    # Loss 1: Root-level contrastive
    root_loss = triplet_loss(
        model.root_embeddings[m1['root']],
        model.root_embeddings[m2['root']],
        model.root_embeddings[m_rand['root']]
    )
    
    # Loss 2: Composed word-level contrastive
    composed_loss = triplet_loss(
        model.embed(word1),
        model.embed(word2),
        model.embed(random_word)
    )
    
    # Combined loss
    return 0.7 * root_loss + 0.3 * composed_loss
```

**Benefit**: Trains roots while also ensuring full compositions work.

## Recommended Approach

**Use Strategy 1 (Root-Level Only)** because:

1. **Simplest**: Only train root embeddings, keep affixes frozen
2. **Most generalizable**: Root similarity transfers to ALL derived forms
3. **Preserves morphology**: Affix meanings stay fixed (mal-, -ej, -ist, etc.)
4. **Aligned with architecture**: Our model already separates roots from affixes

## Implementation Changes Needed

### In `scripts/train_embeddings_contrastive.py`:

```python
class CompositionalContrastiveLearner:
    def __init__(self, base_embedding_model):
        self.base_model = base_embedding_model
        
        # ONLY train root embeddings
        self.trainable_params = [self.base_model.root_embeddings]
        
        # Freeze affix embeddings
        for param in self.base_model.prefix_embeddings.parameters():
            param.requires_grad = False
        for param in self.base_model.suffix_embeddings.parameters():
            param.requires_grad = False
        # Grammatical features already frozen (deterministic)
        
    def forward(self, root):
        # Get root embedding (64d)
        root_emb = self.base_model.root_embeddings[root]
        return F.normalize(root_emb)
        
    def triplet_loss(self, root1, root2, random_root, margin=0.2):
        emb1 = self.forward(root1)
        emb2 = self.forward(root2)
        emb_rand = self.forward(random_root)
        
        sim_pos = F.cosine_similarity(emb1, emb2)
        sim_neg = F.cosine_similarity(emb1, emb_rand)
        
        loss = F.relu(margin + sim_neg - sim_pos)
        return loss
        
    def train_epoch(self, root_pairs):
        for (root1, root2) in root_pairs:
            # Sample random root
            random_root = sample_random(self.vocab_roots)
            
            loss = self.triplet_loss(root1, root2, random_root)
            loss.backward()
            
            # Only update root embeddings
            self.optimizer.step()
```

### Changes to training data loader:

```python
# Load root pairs (from Task #56)
with open('data/training/revo_root_synonyms.json') as f:
    data = json.load(f)
    root_pairs = [(p['root1'], p['root2']) for p in data['pairs']]

# Train on ROOTS, not full words
for epoch in range(num_epochs):
    for (root1, root2) in root_pairs:
        # Train root-level similarity
        loss = model.triplet_loss(root1, root2, random_root)
        loss.backward()
        optimizer.step()
```

## Validation: Test Compositional Transfer

After training, verify that root similarity transfers:

```python
def test_compositional_transfer(model, root1, root2):
    # Test that if roots are similar, derived forms are too
    
    results = {}
    
    # Test base forms
    for ending in ['i', 'o', 'a', 'e']:
        word1 = root1 + ending
        word2 = root2 + ending
        sim = cosine_similarity(model.embed(word1), model.embed(word2))
        results[f'{ending}_form'] = sim
    
    # Test negated forms
    word1 = 'mal' + root1 + 'i'
    word2 = 'mal' + root2 + 'i'
    results['negated'] = cosine_similarity(model.embed(word1), model.embed(word2))
    
    # Test agent nouns
    word1 = root1 + 'isto'
    word2 = root2 + 'isto'
    results['agent'] = cosine_similarity(model.embed(word1), model.embed(word2))
    
    return results

# Example test
results = test_compositional_transfer(model, "kre", "establ")
# Expected: ALL forms should have high similarity (>0.7)
# {
#   'i_form': 0.82,  # krei ≈ establi
#   'o_form': 0.81,  # kreo ≈ establo
#   'a_form': 0.80,  # krea ≈ establa
#   'negated': 0.79,  # malkrei ≈ malestabli
#   'agent': 0.78    # kreisto ≈ establisto
# }
```

## Critical Design Decisions

### Decision 1: Train roots only, freeze affixes
**Rationale**: Affix meanings are deterministic (mal- always means opposite, -ej always means place). Only root meanings are learned from data.

### Decision 2: Use root-level triplet loss
**Rationale**: Simpler than multi-level, generalizes better, preserves compositionality.

### Decision 3: Validate compositional transfer
**Rationale**: Must verify that training roots actually improves all derived forms.

## Success Criteria
- Root-level synonym similarity >0.75
- Compositional transfer: derived forms also >0.7
- Affix embeddings unchanged (frozen)
- Morphological tests still pass (mal- consistency)

## Effort
4-6 hours (redesign + implementation + validation)

## Dependencies
- Task #56 (needs root-level synonym pairs)
- Current compositional embedding model architecture

## Blocks
- Task #55 (contrastive training implementation)
