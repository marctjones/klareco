---
id: 60
title: Train affix embeddings using transformation-aware contrastive learning
state: open
created: '2026-01-05T16:41:49.175744Z'
labels:
- training
- embeddings
- affixes
priority: high
---
## Goal
Improve affix embeddings to correctly capture semantic transformations (negation, place, causative, etc.) using prepared affix pairs from Task #59.

## The Problem
Current affix embeddings may not correctly transform root meanings because they were trained on generic co-occurrence patterns, not explicit semantic transformation examples.

**We need**: Affix embeddings that perform consistent, meaningful transformations.

## Training Strategy

### Approach: Transformation-Aware Contrastive Learning

Unlike root synonyms (Task #57), affixes have **directional transformations**:
- `mal-`: reverses polarity (good → bad)
- `-ej`: adds place semantics (learn → school)
- `-ig`: adds causative semantics (healthy → heal)

We need to train affix embeddings to PERFORM these transformations consistently.

## Loss Functions by Affix Type

### 1. Negation Prefix (mal-) - Polarity Reversal Loss

```python
def negation_loss(root_emb, affixed_emb, mal_vector):
    """
    Train mal- to reverse semantic polarity.
    
    Expected: emb(malbona) ≈ emb(bona) + mal_vector
    But: sim(bona, malbona) should be LOW (opposites)
    """
    # Transform root with mal-
    predicted_affixed = root_emb + mal_vector
    
    # Loss 1: Transformation should match actual affixed embedding
    transform_loss = 1 - cosine_similarity(predicted_affixed, affixed_emb)
    
    # Loss 2: Opposites should be dissimilar
    polarity_loss = max(0, cosine_similarity(root_emb, affixed_emb) - 0.3)
    
    # Loss 3: Consistency - mal- should do same transformation across roots
    # (measured via variance of mal_vector across examples)
    
    return transform_loss + 0.5 * polarity_loss
```

### 2. Place Suffix (-ej) - Cluster Targeting Loss

```python
def place_suffix_loss(root_emb, affixed_emb, ej_vector, place_cluster_center):
    """
    Train -ej to transform to place semantics.
    
    Expected: emb(lernejo) ≈ emb(lern) + ej_vector
    And: emb(lernejo) should cluster near other place words
    """
    # Transform root with -ej
    predicted_affixed = root_emb + ej_vector
    
    # Loss 1: Transformation accuracy
    transform_loss = 1 - cosine_similarity(predicted_affixed, affixed_emb)
    
    # Loss 2: Cluster with place words
    cluster_loss = 1 - cosine_similarity(affixed_emb, place_cluster_center)
    
    # Loss 3: Root should NOT be near place cluster (only affixed form)
    separation_loss = max(0, cosine_similarity(root_emb, place_cluster_center) - 0.4)
    
    return transform_loss + 0.3 * cluster_loss + 0.2 * separation_loss
```

### 3. Agent Suffix (-ist, -ant, -int) - Person Cluster Loss

```python
def agent_suffix_loss(root_emb, affixed_emb, ist_vector, person_cluster_center):
    """
    Train -ist to transform to profession/person semantics.
    
    Expected: emb(artisto) ≈ emb(art) + ist_vector
    And: emb(artisto) should cluster near other person/profession words
    """
    # Similar to place suffix
    predicted_affixed = root_emb + ist_vector
    
    transform_loss = 1 - cosine_similarity(predicted_affixed, affixed_emb)
    cluster_loss = 1 - cosine_similarity(affixed_emb, person_cluster_center)
    
    return transform_loss + 0.3 * cluster_loss
```

### 4. Causative Suffix (-ig) - Transitivity Loss

```python
def causative_suffix_loss(root_emb, affixed_emb, ig_vector):
    """
    Train -ig to add causative/transitive semantics.
    
    Expected: emb(sanigi) ≈ emb(sana) + ig_vector
    """
    predicted_affixed = root_emb + ig_vector
    
    transform_loss = 1 - cosine_similarity(predicted_affixed, affixed_emb)
    
    # Could add: cluster with other causative verbs
    
    return transform_loss
```

### 5. Size Modifiers (-et, -eg) - Opposite Transformation Loss

```python
def size_modifier_loss(root_emb, diminutive_emb, augmentative_emb, et_vector, eg_vector):
    """
    Train -et and -eg to be opposite transformations.
    
    Expected: 
        emb(dometo) ≈ emb(domo) + et_vector
        emb(domego) ≈ emb(domo) + eg_vector
        et_vector and eg_vector should be opposite directions
    """
    # Transformation losses
    et_transform = 1 - cosine_similarity(root_emb + et_vector, diminutive_emb)
    eg_transform = 1 - cosine_similarity(root_emb + eg_vector, augmentative_emb)
    
    # Opposite direction loss
    # et_vector and eg_vector should have negative cosine similarity
    opposite_loss = max(0, cosine_similarity(et_vector, eg_vector) + 0.5)
    
    return et_transform + eg_transform + 0.5 * opposite_loss
```

## Training Architecture

### Option 1: Train Affix Vectors Directly (Recommended)

Treat each affix as a learnable transformation vector:

```python
class AffixTransformationModel(nn.Module):
    def __init__(self, base_embedding_model, affix_dim=8):
        super().__init__()
        self.base_model = base_embedding_model
        
        # Learnable affix vectors
        self.affix_vectors = nn.ParameterDict({
            'mal': nn.Parameter(torch.randn(affix_dim)),
            'ej': nn.Parameter(torch.randn(affix_dim)),
            'ist': nn.Parameter(torch.randn(affix_dim)),
            'ig': nn.Parameter(torch.randn(affix_dim)),
            'et': nn.Parameter(torch.randn(affix_dim)),
            'eg': nn.Parameter(torch.randn(affix_dim)),
            # ... other affixes
        })
        
        # Freeze root embeddings during affix training
        for param in self.base_model.root_embeddings.parameters():
            param.requires_grad = False
    
    def forward(self, root, affix):
        # Get root embedding
        root_emb = self.base_model.root_embeddings[root]  # 64d
        
        # Get affix transformation vector
        affix_vec = self.affix_vectors[affix]  # 8d
        
        # Compose (concatenate)
        composed = torch.cat([root_emb, affix_vec], dim=-1)  # 72d total
        
        # Or: Add transformation (if same dimensions)
        # composed = root_emb + self.expand(affix_vec)
        
        return composed
```

### Option 2: Learn Affix Transformation Functions

Use small neural networks to transform root embeddings:

```python
class AffixTransformationNetwork(nn.Module):
    def __init__(self, root_dim=64):
        super().__init__()
        
        # Separate transformation network for each affix
        self.mal_transform = nn.Linear(root_dim, root_dim)
        self.ej_transform = nn.Linear(root_dim, root_dim)
        # ... etc
    
    def apply_affix(self, root_emb, affix):
        if affix == 'mal':
            return self.mal_transform(root_emb)
        elif affix == 'ej':
            return self.ej_transform(root_emb)
        # ... etc
```

**Recommended**: Option 1 (simpler, more interpretable)

## Training Procedure

```python
def train_affix_embeddings(affix_pairs, base_model, num_epochs=20):
    """
    Train affix vectors using transformation-aware contrastive learning.
    """
    model = AffixTransformationModel(base_model)
    optimizer = torch.optim.Adam(model.affix_vectors.parameters(), lr=1e-3)
    
    # Pre-compute cluster centers for place/person words
    place_cluster = compute_cluster_center(['domo', 'urbo', 'lando', 'loko', ...])
    person_cluster = compute_cluster_center(['homo', 'viro', 'virino', 'persono', ...])
    
    for epoch in range(num_epochs):
        for affix_type in ['mal', 'ej', 'ist', 'ig', 'et', 'eg']:
            pairs = affix_pairs[affix_type]
            
            for pair in pairs:
                root = pair['root']
                affixed_form = pair['affixed_form']
                
                # Get embeddings
                root_emb = base_model.embed_root(root)
                affixed_emb = base_model.embed(affixed_form)  # Ground truth
                
                # Get affix vector
                affix_vec = model.affix_vectors[affix_type]
                
                # Compute loss (depends on affix type)
                if affix_type == 'mal':
                    loss = negation_loss(root_emb, affixed_emb, affix_vec)
                elif affix_type == 'ej':
                    loss = place_suffix_loss(root_emb, affixed_emb, affix_vec, place_cluster)
                elif affix_type == 'ist':
                    loss = agent_suffix_loss(root_emb, affixed_emb, affix_vec, person_cluster)
                # ... etc
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # Validation
        if epoch % 5 == 0:
            validate_affix_consistency(model)
```

## Validation Tests

After training, run Task #58 tests to verify:

```python
def validate_affix_training(model):
    """Run all affix semantic tests from Task #58."""
    
    # Test 1: mal- reverses polarity
    mal_results = test_negation_prefix(model)
    assert mal_results['mean_similarity'] < 0.3, "mal- not reversing polarity"
    
    # Test 2: -ej clusters with places
    ej_results = test_place_suffix(model)
    assert ej_results['cluster_similarity'] > 0.7, "-ej not clustering with places"
    
    # Test 3: Consistency
    consistency_results = test_affix_consistency(model)
    assert consistency_results['mal_std'] < 0.15, "mal- vector inconsistent"
    
    return {
        'negation': mal_results,
        'place': ej_results,
        'consistency': consistency_results
    }
```

## Output

Save improved model: `models/affix_embeddings/trained_affixes.pt`

Log training: `logs/training/affix_training_TIMESTAMP.log`

Save metrics: `benchmark_results/embeddings/affix_training_TIMESTAMP.json`

## Expected Improvement

**Before** (hypothetical):
- mal- reversal: sim(bona, malbona) = 0.65 (too similar)
- -ej cluster: 0.45 (weak association with places)
- Affix vector std: 0.35 (inconsistent)

**After** (target):
- mal- reversal: sim(bona, malbona) < 0.3 (clear opposition)
- -ej cluster: >0.7 (strong place association)
- Affix vector std: <0.15 (consistent transformations)

## Success Criteria
- Task #58 tests show >80% pass rate
- Affix transformations consistent (std < 0.15)
- Cluster targeting works (similarity > 0.7)
- Compositional forms inherit transformations

## Effort
10-14 hours (implementation + training + validation)

## Dependencies
- Task #59 (affix training data)
- Task #58 (evaluation framework to test results)
- Current compositional embedding model

## Related
- Complements Task #57 (root contrastive learning)
- Together these improve both roots AND affixes
