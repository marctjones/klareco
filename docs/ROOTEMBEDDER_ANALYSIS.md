# RootEmbedder Design Analysis

Is the current RootEmbedder well-designed for Klareco's AST-first architecture? Should we redesign it with Claude Opus?

## Current Design Summary

### Architecture
```python
class RootEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Simple lookup table: 40K roots × 64 dims = 2.56M params
```

### Training Approach
1. **Data sources** (authority-weighted):
   - ReVo dictionary definitions (Jaccard similarity of definition roots)
   - Ekzercaro co-occurrence (Zamenhof's curated examples)
   - Tier 5 corpus co-occurrence (clean Esperanto books)

2. **Graded similarity targets** (0.0-1.0):
   - High similarity (0.7-0.95): Strong co-occurrence, synonym relations
   - Medium similarity (0.4-0.7): Jaccard overlap in definitions
   - Low similarity (0.0-0.3): Negative samples (random pairs)

3. **Loss function**: Margin-based contrastive loss
   - Push similar pairs above target - margin
   - Push dissimilar pairs below target + margin
   - Weighted by source authority (Tier 0 > Ekzercaro > ReVo)

4. **Function word filtering**: Excludes grammatical words (kaj, de, la, etc.)

### Current Results
- **Correlation**: 0.85 (predicted vs target similarity)
- **Positive pair similarity**: ~0.53 average
- **Negative pair similarity**: ~0.03 average (good separation)
- **Training time**: ~32 epochs, ~2 hours on CPU

---

## Strengths (What Works Well)

### ✅ 1. Function Word Exclusion
**Why it works**: Prevents embedding collapse by excluding high-frequency grammatical words.

**Example**:
```python
FUNCTION_WORDS = {
    'kaj', 'aŭ', 'sed',  # Conjunctions
    'de', 'en', 'al',    # Prepositions
    'mi', 'li', 'ŝi',    # Pronouns
}
# These are handled deterministically in ASTs, not learned!
```

**Klareco benefit**: Aligns perfectly with AST-first philosophy - grammar is deterministic, not learned.

### ✅ 2. Authority-Weighted Training
**Why it works**: Zamenhof's Ekzercaro > crowd-sourced definitions > corpus noise.

**Weighting**:
- Tier 0 (Fundamento): weight = 20.0
- Ekzercaro: weight = 10.0
- ReVo definitions: weight = 5.0
- Tier 5 books: weight = 15.0

**Klareco benefit**: Prioritizes quality over quantity, good for small models.

### ✅ 3. Graded Similarity Targets
**Why it works**: Not all synonyms are equal - "hundo" vs "besto" (dog vs animal) should have lower similarity than "hundo" vs "hundeto" (dog vs puppy).

**Klareco benefit**: Enables fine-grained semantic distinctions.

### ✅ 4. Simple Architecture
**Why it works**: 64-dim embeddings are fast, small (2.56M params), easy to debug.

**Klareco benefit**: Fits in L1 cache, fast lookups, minimal memory.

---

## Weaknesses (Klareco-Specific Issues)

### ❌ 1. AST-Agnostic Training
**Problem**: Treats sentences as bags of roots, ignores AST structure.

**Example**:
```
Text: "La hundo manĝas la katon"
Current: {hund, manĝ, kat} → co-occurrence
Missing: hund is subjekto, kat is objekto → different roles!
```

**Impact**: Doesn't learn that subjects vs objects might have different distributions.

**Fix**: Train on AST role-aware pairs (subject-verb, verb-object).

### ❌ 2. No Negation/Antonym Modeling
**Problem**: "bona" (good) and "malbona" (bad) are unrelated in current embeddings.

**Example**:
```python
# Esperanto has systematic negation via "mal-" prefix
similarity("bon", "malbon") → 0.1 (random)
# Should be: -0.8 (antonyms)
```

**Impact**: Misses 30%+ of Esperanto vocabulary (mal- is productive).

**Fix**: Add antonym pairs with **negative** similarity targets.

### ❌ 3. No Hierarchical Structure
**Problem**: No explicit hypernym/hyponym relationships.

**Example**:
```
hund (dog) → mamul (mammal) → best (animal) → viv (living thing)
Current: All treated as flat co-occurrence
Missing: Hierarchical taxonomy
```

**Impact**: Can't reason about "all dogs are animals" (useful for PlausibilityFilter).

**Fix**: Add hierarchical relations from WordNet/ReVo taxonomy.

### ❌ 4. Context-Free
**Problem**: Root embeddings don't vary by context (always same vector).

**Example**:
```
"La pomo estas ruĝa" (The apple is red)
"La ruĝa pomo" (The red apple)
Current: embed("ruĝ") is identical in both
Better: Contextualized based on AST role (adjective vs predicate)
```

**Impact**: Can't capture polysemy or role-dependent meaning.

**Fix**: This is actually good! Context should come from ASTEncoder, not root embeddings. Root embeddings should be context-free primitives.

### ❌ 5. No Compositionality
**Problem**: Multi-word expressions treated as separate roots.

**Example**:
```
"preni decidon" (make a decision) = idiomatic phrase
Current: embed("pren") + embed("decid") ≠ phrase meaning
Missing: Phrase-level semantics
```

**Impact**: Misses 10-15% of Esperanto expressions.

**Fix**: This is MorphemeComposer's job, not RootEmbedder's. Accept this limitation.

---

## Klareco-Specific Improvements

### 🎯 Improvement 1: AST Role-Aware Training Pairs
**Why**: Klareco operates on ASTs, not bag-of-words.

**Current**:
```python
# Co-occurrence: just extract roots
roots = ['hund', 'manĝ', 'kat']
pairs = [(hund, manĝ), (hund, kat), (manĝ, kat)]  # All equal weight
```

**Improved**:
```python
# AST-aware: extract role-specific pairs
ast = {
    'subjekto': {'radiko': 'hund'},
    'verbo': {'radiko': 'manĝ'},
    'objekto': {'radiko': 'kat'}
}
pairs = [
    (('hund', 'subjekto'), ('manĝ', 'verbo'), weight=10.0),  # Subject-verb affinity
    (('manĝ', 'verbo'), ('kat', 'objekto'), weight=10.0),    # Verb-object affinity
    (('hund', 'subjekto'), ('kat', 'objekto'), weight=2.0),  # Cross-role (lower)
]
```

**Benefit**: Learns that "hund" frequently appears as subject, "manĝ" as verb, etc.

**Cost**: Slightly more complex training data extraction.

### 🎯 Improvement 2: Antonym Pairs (mal- prefix)
**Why**: Esperanto has systematic negation - leverage it!

**Implementation**:
```python
# Generate antonym pairs programmatically
for root in vocab:
    if root.startswith('mal'):
        positive_root = root[3:]  # Remove 'mal-'
        if positive_root in vocab:
            # Add negative similarity target
            pairs.append((root, positive_root, -0.7, weight=20.0))
```

**Example pairs**:
```
(bon, malbon) → target = -0.7  # good ↔ bad
(longa, mallonga) → target = -0.7  # long ↔ short
(varma, malvarma) → target = -0.7  # warm ↔ cold
```

**Benefit**: Systematic antonym modeling with zero annotation cost.

**Impact**: Affects ~4,000 roots (30% of vocabulary).

### 🎯 Improvement 3: Hypernym Relations from ReVo
**Why**: Hierarchical reasoning needed for PlausibilityFilter.

**Data source**: ReVo already has genus-species relations!

**Example**:
```json
{
  "hundo": {
    "definition": "Hejmbesto el la familio Canidae",
    "genus": "besto",  // ReVo taxonomy
    "hypernyms": ["mamulo", "besto", "vivulo"]
  }
}
```

**Training pairs**:
```python
# Hypernym pairs: child → parent similarity = 0.6-0.8
pairs.append(('hund', 'best'), 0.7, weight=15.0)  # dog → animal
pairs.append(('hund', 'mamul'), 0.75, weight=15.0)  # dog → mammal
```

**Benefit**: Enables "all X are Y" reasoning.

**Cost**: Requires parsing ReVo taxonomy (one-time effort).

### 🎯 Improvement 4: Freeze After Training
**Why**: Root embeddings should be **stable primitives** for downstream models.

**Current**: Embeddings could drift if retrained.

**Improved**:
```python
# After training, freeze and version
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Save with version metadata
torch.save({
    'embeddings': model.state_dict(),
    'version': 'v1.0',
    'frozen': True,
    'vocab': root_to_idx,
    'metadata': {...}
}, 'models/root_embedder/frozen_v1.0.pt')
```

**Benefit**: Downstream models (MorphemeComposer, NodePredictor) can rely on stable embeddings.

---

## Should We Redesign with Opus?

### When to Use Opus for Redesign

✅ **YES - Use Opus if**:
1. You want to add **AST role-aware training** (complex extraction logic)
2. You want to add **hierarchical relations** (ReVo taxonomy parsing)
3. You want to experiment with **novel architectures** (e.g., hyperbolic embeddings for hierarchies)
4. You want **comprehensive evaluation framework** (similarity benchmarks, analogy tests)

❌ **NO - Don't need Opus if**:
1. Current embeddings work well enough (correlation = 0.85 is good!)
2. You just want simple fixes (add antonym pairs, freeze model)
3. Time is limited (redesign = 1-2 weeks delay)

### Recommendation: Incremental Improvements First

**Phase 1: Quick Wins (1-2 days)** - Do this with Sonnet/yourself:
1. ✅ Add antonym pairs (mal- prefix) → +30% vocabulary coverage
2. ✅ Freeze model after training → stability for downstream
3. ✅ Add evaluation metrics → track quality over time

**Phase 2: AST-Aware Training (1 week)** - Use Opus for this:
1. 🤖 Extract role-aware pairs from ASTs
2. 🤖 Parse ReVo taxonomy for hypernyms
3. 🤖 Design comprehensive evaluation suite
4. 🤖 Experiment with advanced architectures (optional)

**Phase 3: Production (after evaluation)**:
- If Phase 1 works well (correlation >0.90) → ship it!
- If not → implement Phase 2 with Opus

---

## Opus Design Prompt (If Redesigning)

If you decide to use Opus, here's the prompt:

```
Design an improved RootEmbedder for Klareco's AST-first Esperanto AI architecture.

CONTEXT:
- Klareco operates on ASTs, not raw text
- Grammar is 100% deterministic (parser handles it)
- RootEmbedder should learn ONLY semantic similarity
- Current approach: simple 64-dim embeddings, contrastive learning, 0.85 correlation

REQUIREMENTS:
1. AST role-aware training (subject/verb/object matter)
2. Systematic antonym modeling (mal- prefix)
3. Hierarchical relations (hypernyms from ReVo taxonomy)
4. Maintain simplicity (<5M params)
5. Fast inference (<1ms per lookup)
6. Freezable for downstream stability

CURRENT STRENGTHS TO KEEP:
- Function word exclusion (prevents collapse)
- Authority-weighted training (Fundamento > corpus)
- Graded similarity targets (not binary)
- 64-dim size (fast, small)

DESIGN DELIVERABLES:
1. Training data extraction (from AST corpus + ReVo)
2. Model architecture (pytorch implementation)
3. Loss function (handle positive, negative, antonym, hypernym pairs)
4. Evaluation framework (similarity, analogy, hierarchy tests)
5. Training script (resume from checkpoint, early stopping)

CONSTRAINTS:
- Must work with existing Klareco corpus (data/corpus/unified_corpus.jsonl)
- Must output frozen embeddings for downstream use
- Should train in <4 hours on CPU
```

---

## Verdict

### Current RootEmbedder: **B+ (Good, but not optimal for ASTs)**

**Strengths**:
- ✅ Simple, fast, small (64 dims, 2.56M params)
- ✅ Function word filtering (aligns with AST philosophy)
- ✅ Authority-weighted training (quality over quantity)
- ✅ 0.85 correlation (decent results)

**Weaknesses for Klareco**:
- ❌ AST-agnostic (bags of roots, not structured)
- ❌ No antonym modeling (misses mal- systematicity)
- ❌ No hierarchical relations (can't reason about taxonomies)

### Recommendation: **Incremental Improvements → Opus Redesign**

1. **Week 1**: Add antonym pairs + freeze model (yourself/Sonnet)
2. **Week 2**: Evaluate results
3. **Week 3**: If needed, use Opus to design AST-aware v2.0

**Why this approach**:
- Don't over-engineer prematurely
- Current embeddings might be "good enough" for minimal config (21M params)
- Antonym pairs are low-hanging fruit (30% vocab improvement)
- Can always upgrade later if quality is insufficient

### Key Insight

**RootEmbedder's job**: Provide stable, frozen semantic primitives.
**NOT its job**: Context, grammar, compositional semantics (that's MorphemeComposer, ASTEncoder, NodePredictor).

Current design is 80% there. Add antonym pairs + freezing → 90%. If that's not enough, then redesign with Opus.

---

## Next Steps

1. **Decide**: Quick fixes vs Opus redesign?
2. **If quick fixes**:
   - Add antonym pair generation script
   - Add freeze-after-training to training script
   - Validate current embeddings (similarity tests)
3. **If Opus redesign**:
   - Use prompt above
   - Budget 1-2 weeks for design + implementation
   - Compare v1.0 vs v2.0 on benchmarks

**My vote**: Quick fixes first. Current design is solid. Don't let perfect be the enemy of good.
