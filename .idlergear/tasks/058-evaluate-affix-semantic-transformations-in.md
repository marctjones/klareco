---
id: 58
title: Evaluate affix semantic transformations in current compositional model
state: open
created: '2026-01-05T16:39:58.738278Z'
labels:
- evaluation
- embeddings
- affixes
priority: high
---
## Goal
Test whether our current affix embeddings correctly capture semantic transformations that affixes apply to roots.

## The Question
When we add an affix to a root, does the composed embedding capture the semantic change?

**Example**: 
- Root: "san" (healthy)
- With "mal-": "malsan" (unhealthy/sick) - should be semantically OPPOSITE
- With "-ej": "sanej" (health center/clinic) - should be semantically related to PLACE
- With "-ig": "sanig" (to make healthy/heal) - should be semantically CAUSATIVE

## Test Categories

### Test 1: Negation Prefix (mal-)
**Expected behavior**: `mal-` should reverse semantic polarity

Test pairs:
```python
test_pairs = [
    ("bona", "malbona"),      # good → bad
    ("granda", "malgranda"),  # big → small
    ("riĉa", "malriĉa"),      # rich → poor
    ("rapida", "malrapida"),  # fast → slow
    ("varma", "malvarma"),    # warm → cold
    ("nova", "malnova"),      # new → old
    ("juna", "maljuna"),      # young → old
    ("facila", "malfacila"),  # easy → difficult
    ("sana", "malsana"),      # healthy → sick
    ("proksima", "malproksima")  # near → far
]
```

**Metrics**:
- Cosine similarity should be LOW (opposite meaning)
- Vector arithmetic: `emb("malbona") ≈ emb("bona") + mal_vector`
- Consistency: Is `mal_vector` similar across all applications?

**Success criteria**:
- Mean similarity < 0.3 (opposite meanings)
- Vector consistency: std(mal_vector) < 0.15

---

### Test 2: Place Suffix (-ej)
**Expected behavior**: `-ej` should transform to "place where X happens"

Test cases:
```python
test_cases = [
    ("lern", "lernejo"),     # learn → school
    ("labor", "laborejo"),   # work → workplace
    ("preĝ", "preĝejo"),     # pray → church
    ("kuir", "kuirejo"),     # cook → kitchen
    ("kurac", "kuracejo"),   # cure → hospital
    ("vend", "vendejo"),     # sell → store
    ("lav", "lavejo"),       # wash → laundry
    ("dorm", "dormejo"),     # sleep → bedroom
    ("trink", "trinkejo"),   # drink → bar/pub
    ("san", "sanej")         # healthy → clinic
]
```

**Metrics**:
- Does `emb("lernejo")` cluster with other places?
- Vector arithmetic: `emb("lernejo") ≈ emb("lern") + ej_vector`
- Is `ej_vector` consistent across roots?

**Success criteria**:
- `ej_vector` has high cosine similarity (>0.8) across applications
- Composed embeddings cluster near ground-truth "place" words

---

### Test 3: Agent Suffix (-ist)
**Expected behavior**: `-ist` should transform to "person who does X professionally"

Test cases:
```python
test_cases = [
    ("art", "artisto"),      # art → artist
    ("musik", "muzikisto"),  # music → musician
    ("scienc", "sciencisto"), # science → scientist
    ("kant", "kantisto"),    # sing → singer
    ("labor", "laboristo"),  # work → worker
    ("juĝ", "juĝisto"),      # judge → judge (profession)
    ("politic", "politikisto"),  # politics → politician
    ("sport", "sportisto"),  # sport → athlete
    ("journal", "ĵurnalisto"),  # journalism → journalist
    ("instru", "instruisto")    # teach → teacher
]
```

**Metrics**:
- Does `emb("artisto")` cluster with other professions?
- Vector arithmetic: `emb("artisto") ≈ emb("art") + ist_vector`

---

### Test 4: Causative Suffix (-ig)
**Expected behavior**: `-ig` should transform to "make/cause to be X"

Test cases:
```python
test_cases = [
    ("san", "sanigi"),       # healthy → heal (make healthy)
    ("pura", "purigi"),      # clean → clean (make clean)
    ("libera", "liberigi"),  # free → liberate (make free)
    ("varma", "varmigi"),    # warm → heat (make warm)
    ("riĉa", "riĉigi"),      # rich → enrich (make rich)
    ("bela", "beligi"),      # beautiful → beautify
    ("fort", "fortigi"),     # strong → strengthen
    ("mola", "moligi"),      # soft → soften
    ("klar", "klarigi"),     # clear → clarify
    ("facila", "faciligi")   # easy → facilitate (make easy)
]
```

**Metrics**:
- Vector arithmetic: `emb("sanigi") ≈ emb("sana") + ig_vector`
- Is the transformation consistent?

---

### Test 5: Diminutive Suffix (-et)
**Expected behavior**: `-et` should transform to "small version of X"

Test cases:
```python
test_cases = [
    ("domo", "dometo"),      # house → cottage
    ("kato", "kateto"),      # cat → kitten
    ("hundo", "hundeto"),    # dog → puppy
    ("libro", "libreto"),    # book → booklet
    ("ŝip", "ŝipeto"),       # ship → boat
    ("tablo", "tableto"),    # table → small table
    ("arb", "arbeto"),       # tree → shrub
    ("urb", "urbeto"),       # city → town
    ("riv", "riveto"),       # river → stream
    ("mont", "monteto")      # mountain → hill
]
```

---

### Test 6: Augmentative Suffix (-eg)
**Expected behavior**: `-eg` should transform to "large version of X"

Test cases:
```python
test_cases = [
    ("domo", "domego"),      # house → mansion
    ("pluv", "pluvego"),     # rain → downpour
    ("vento", "ventego"),    # wind → gale
    ("rido", "ridego"),      # laugh → guffaw
    ("varmo", "varmego"),    # warmth → heat
    ("krio", "kriego"),      # cry → scream
]
```

**Expected**: `-eg` vector should be opposite direction from `-et` vector

---

## Implementation

Create script: `scripts/evaluate_affix_semantics.py`

```python
class AffixSemanticEvaluator:
    def __init__(self, compositional_model, corpus_vocab):
        self.model = compositional_model
        self.vocab = corpus_vocab
        
    def test_negation_prefix(self):
        """Test mal- reverses polarity"""
        results = []
        mal_vectors = []
        
        for (base, negated) in test_pairs:
            emb_base = self.model.embed(base)
            emb_neg = self.model.embed(negated)
            
            # Similarity (should be low)
            sim = cosine_similarity(emb_base, emb_neg)
            
            # Extract mal- transformation vector
            mal_vec = emb_neg - emb_base
            mal_vectors.append(mal_vec)
            
            results.append({
                'base': base,
                'negated': negated,
                'similarity': float(sim),
                'expected': 'low (<0.3)'
            })
        
        # Check consistency of mal- vector
        mal_consistency = self.vector_consistency(mal_vectors)
        
        return {
            'pairs': results,
            'mean_similarity': np.mean([r['similarity'] for r in results]),
            'mal_vector_consistency': mal_consistency
        }
    
    def vector_consistency(self, vectors):
        """Measure how consistent a transformation vector is"""
        # Compute mean vector
        mean_vec = np.mean(vectors, axis=0)
        
        # Compute similarity of each to mean
        sims = [cosine_similarity(v, mean_vec) for v in vectors]
        
        return {
            'mean_similarity_to_mean': np.mean(sims),
            'std_similarity': np.std(sims)
        }
    
    def test_place_suffix(self):
        """Test -ej creates place semantics"""
        # Similar pattern...
        
    def test_agent_suffix(self):
        """Test -ist creates profession semantics"""
        # Similar pattern...
        
    def generate_report(self):
        """Run all tests and generate comprehensive report"""
        pass
```

## Output

Generate report: `benchmark_results/embeddings/affix_evaluation_TIMESTAMP.json`

```json
{
  "negation_prefix_mal": {
    "mean_similarity": 0.XX,
    "expected": "<0.3",
    "status": "pass/fail",
    "mal_vector_consistency": {
      "mean_similarity_to_mean": 0.XX,
      "std": 0.XX
    },
    "failures": [
      {"base": "bona", "negated": "malbona", "similarity": 0.XX, "issue": "too similar"}
    ]
  },
  "place_suffix_ej": {
    "ej_vector_consistency": 0.XX,
    "expected": ">0.8",
    "status": "pass/fail"
  },
  "agent_suffix_ist": {...},
  "causative_suffix_ig": {...},
  "diminutive_suffix_et": {...},
  "augmentative_suffix_eg": {...}
}
```

## Success Criteria

For EACH affix type:
- Transformation is consistent (vector std < 0.15)
- Semantic change captured correctly
- Vector arithmetic works: `word_with_affix ≈ root + affix_vector`

## Expected Findings

**If affixes work well**:
- mal- vector consistently reverses polarity
- -ej vector consistently adds "place" semantics
- Affix vectors are reusable across different roots

**If affixes DON'T work well**:
- High variance in transformation vectors
- Semantic changes not captured
- Need to improve affix training (→ Task #58)

## Effort
4-6 hours (test implementation + analysis)

## Dependencies
- Current compositional embedding model
- Corpus vocabulary (to check word existence)

## Blocks
- Task #58 (affix improvement - if this evaluation shows poor performance)
