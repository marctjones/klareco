# Coreference Resolution Design (Phase 1.4)

**Date**: 2026-03-09
**Status**: Design phase
**Component**: Phase 1 Learned Component #2

---

## Overview

**Goal**: Resolve pronouns and referring expressions to their antecedents

**Example**:
```
"Ludoviko Zamenhof fondis Esperanton. Li estis kuracisto."
                                        ↓
                                    Zamenhof
```

**Impact**:
- 60% of sentences contain pronouns
- Critical for fact extraction quality
- Enables better entity resolution in summaries

---

## Current Limitation

**Phase 0/1 behavior**:
- Extracts fact: `{'subject': 'Li', 'predicate': 'est', 'object': 'kuracisto'}`
- Summary: "Li estis kuracisto." (pronoun not resolved)
- **Problem**: User doesn't know who "Li" refers to

**Desired behavior**:
- Resolve: Li → Zamenhof
- Extract fact: `{'subject': 'Zamenhof', 'predicate': 'est', 'object': 'kuracisto'}`
- Summary: "Zamenhof estis kuracisto." (clear entity reference)

---

## Architecture Design

### Input
- Paragraph of sentences (context for resolution)
- AST for each sentence (deterministic parse)
- Compositional embeddings (from Phase 1.3)

### Output
- Coreference chains: [(sent_idx, word_idx, entity_id), ...]
- Example: [(0, 0, 'E1'), (1, 0, 'E1')] → "Zamenhof" and "Li" refer to same entity E1

### Model
- **Type**: Neural coreference resolver
- **Size**: ~10M parameters (small by modern standards)
- **Architecture**:
  - Encoder: BiLSTM over compositional embeddings
  - Mention detector: Identify potential mentions (nouns, pronouns, proper names)
  - Pairwise scorer: Score all mention pairs for coreference
  - Decoder: Cluster mentions into entities

---

## Training Data Requirements

### Annotation Needed
- **1,000 annotated documents** (~50K sentences)
- Each document: Coreference chains marked
- Example annotation:
  ```
  Sentence 0: "Ludoviko Zamenhof fondis Esperanton."
              └─E1─────────┘              └─E2──────┘

  Sentence 1: "Li estis kuracisto."
              └E1┘

  Sentence 2: "Ĝi estas planlingvo."
              └E2┘
  ```

### Annotation Format
```json
{
  "document_id": "doc_001",
  "sentences": [
    {"id": 0, "text": "Ludoviko Zamenhof fondis Esperanton."},
    {"id": 1, "text": "Li estis kuracisto."}
  ],
  "entities": [
    {
      "entity_id": "E1",
      "mentions": [
        {"sent_idx": 0, "start": 0, "end": 17, "text": "Ludoviko Zamenhof", "type": "proper_name"},
        {"sent_idx": 1, "start": 0, "end": 2, "text": "Li", "type": "pronoun"}
      ]
    }
  ]
}
```

### Annotation Cost
- **Time**: 2-3 minutes per document
- **Total**: ~50 hours (1 week at 7 hours/day)
- **Tool**: Can use existing annotation tools (Brat, INCEpTION)

---

## Training Strategy

### Stage 1: Mention Detection (2 weeks)
- Train mention detector on annotated mentions
- Identify: nouns, pronouns, proper names
- Precision target: >95% (recall can be lower)

### Stage 2: Pairwise Scoring (2 weeks)
- Train pairwise scorer on mention pairs
- Features:
  - Embeddings from Phase 1.3 (semantic similarity)
  - AST features (syntactic role: subject/object)
  - Distance features (sentences apart)
  - Gender/number agreement (from deterministic morphology)
- Score: P(mention_i corefers with mention_j)

### Stage 3: Clustering (1 week)
- Implement clustering algorithm (e.g., agglomerative clustering)
- Use pairwise scores to build entity clusters
- Evaluation: CoNLL F1 score

---

## Integration with Fact Extractor

### Current Pipeline
```
Sentence → Parser → AST → Fact Extractor → Facts
```

### With Coreference
```
Paragraph → Parser → ASTs → Coreference Resolver → Resolved ASTs
              ↓                      ↓
         Embeddings            Entity Mapping
              ↓                      ↓
          Fact Extractor ← (Replace pronouns with entities)
              ↓
            Facts
```

### Implementation
```python
# In fact_extractor.py

def extract_facts_with_coreference(paragraph_asts, coreference_chains):
    """
    Extract facts with pronoun resolution.

    Args:
        paragraph_asts: List of ASTs for paragraph
        coreference_chains: Dict mapping (sent_idx, word_idx) → entity_id

    Returns:
        List of facts with resolved entity references
    """
    facts = []

    for sent_idx, ast in enumerate(paragraph_asts):
        # Extract fact as normal
        fact = extract_fact_from_ast(ast)

        # Resolve pronouns
        if is_pronoun(fact['subject']):
            entity_id = coreference_chains.get((sent_idx, subject_word_idx))
            if entity_id:
                # Find canonical name for entity (first mention or most descriptive)
                fact['subject'] = get_canonical_name(entity_id, coreference_chains)

        # Same for object
        if is_pronoun(fact['object']):
            entity_id = coreference_chains.get((sent_idx, object_word_idx))
            if entity_id:
                fact['object'] = get_canonical_name(entity_id, coreference_chains)

        facts.append(fact)

    return facts
```

---

## Leveraging Deterministic Components

### What Coreference Model Learns
- ✅ **Semantic compatibility**: "kuracisto" corefers with "viro" (plausible)
- ✅ **Discourse patterns**: Pronouns typically refer to recent mentions
- ✅ **Entity salience**: Subject mentions more likely to be referred to

### What's Deterministic (Reduces Model Complexity)
- ✅ **Gender agreement**: Parser extracts from morphology
  - "Li" → masculine (from ending)
  - "Ŝi" → feminine
  - "Ĝi" → neuter
  - Model doesn't need to learn this!

- ✅ **Number agreement**: Parser provides
  - "ili" → plural (from ending)
  - "li" → singular
  - Model just checks agreement

- ✅ **Syntactic role**: AST provides
  - Subject pronouns resolve to subject mentions (higher probability)
  - Object pronouns resolve to object mentions

**This is the hybrid advantage**: Model focuses on semantics and discourse, not grammar.

---

## Expected Quality

### Baseline (Without Coreference)
- Pronoun facts: 60% of sentences
- Resolved: 0%
- **Quality**: Poor (facts with "Li", "Ŝi", "Ĝi", "ili")

### With Coreference (Phase 1.4)
- Mention detection: >95% precision
- Pairwise accuracy: >85%
- End-to-end CoNLL F1: >75%
- **Quality**: Much better (most pronouns resolved)

### Literature Benchmarks
- English coreference: 75-80% CoNLL F1 (state-of-art)
- Esperanto simpler: No gender ambiguity, explicit morphology
- **Expected**: 75-80% CoNLL F1 (competitive with English)

---

## Training Timeline

| Stage | Task | Duration |
|-------|------|----------|
| **1. Annotation** | Annotate 1K documents | 1 week (50 hours) |
| **2. Mention Detection** | Train mention detector | 2 weeks |
| **3. Pairwise Scoring** | Train pairwise scorer | 2 weeks |
| **4. Clustering** | Implement clustering | 1 week |
| **5. Integration** | Integrate with fact extractor | 1 week |
| **6. Evaluation** | Test and iterate | 1 week |
| **Total** | | **8 weeks** |

---

## Cost-Benefit Analysis

### Cost
- **Annotation**: 50 hours (1 week)
- **Training**: 5 weeks implementation + 1 week evaluation
- **Total**: 6-8 weeks

### Benefit
- **Impact**: 60% of sentences have pronouns
- **Quality improvement**: Facts become entity-grounded (not pronoun-based)
- **User experience**: Summaries more clear and informative
- **Downstream**: Better for reasoning and question answering

### Recommendation
- **Priority**: High (60% of sentences affected)
- **Complexity**: Moderate (10M params, manageable)
- **Dependencies**: Requires Phase 1.3 embeddings (for semantic scoring)

---

## Alternative: Rule-Based Coreference (Faster, Lower Quality)

### Simpler Approach
- Use deterministic rules only:
  - Pronouns resolve to nearest matching gender/number
  - Subject pronouns → recent subject mentions
  - Proper names take precedence
- **No learning required**

### Trade-offs
- ✅ **Faster**: 1 week implementation (no annotation needed)
- ✅ **Explainable**: Clear rules
- ❌ **Lower quality**: ~50-60% accuracy (vs 75-80% learned)
- ❌ **No semantic understanding**: "Li estis kuracisto" resolved to any recent masculine mention

### Recommendation
- **Use as baseline**: Implement rule-based first
- **Then train learned model**: Measure improvement over baseline
- **This follows the hybrid philosophy**: Deterministic first, learned for remaining complexity

---

## Next Steps

### Option A: Train Learned Model (8 weeks)
1. Annotate 1,000 documents with coreference chains
2. Train mention detector (2 weeks)
3. Train pairwise scorer (2 weeks)
4. Implement clustering and integration (2 weeks)
5. Evaluate and iterate (1 week)

### Option B: Rule-Based First (1 week)
1. Implement gender/number agreement rules
2. Implement recency-based resolution
3. Test on Phase 0 summaries
4. Measure accuracy (likely 50-60%)
5. Then decide: Is learned model worth 8 weeks?

### Recommendation
**Start with Option B (rule-based)**, then:
- If quality sufficient (>70%) → Done, move to Phase 2
- If quality insufficient (<70%) → Implement learned model (Option A)

This follows the "deterministic first, learn only what's necessary" philosophy.

---

## Summary

**Coreference resolution** is critical for Phase 1 quality:
- 60% of sentences have pronouns
- Learned model: 75-80% accuracy (8 weeks)
- Rule-based: 50-60% accuracy (1 week)

**Recommendation**:
1. Implement rule-based first (1 week)
2. Evaluate quality
3. If insufficient, train learned model (8 weeks)

**Aligns with hybrid philosophy**: Try deterministic first, learn only if needed.

---

**Last Updated**: 2026-03-09
**Status**: Design complete, awaiting user decision
**Next**: Implement rule-based baseline OR train learned model?
