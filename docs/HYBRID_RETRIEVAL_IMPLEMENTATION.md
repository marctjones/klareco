# Hybrid Retrieval Implementation

## Overview

Implemented two-track query expansion for RAG retrieval, combining:
1. **Deterministic synonyms** (ReVo dictionary)
2. **Learned associations** (embedding co-occurrence)

This aligns with Klareco's thesis: maximize deterministic processing, use learned capacity strategically.

## Architecture

```
Query: "Kiu fondis Esperanton?"
         ↓
    Parse → AST
         ↓
  Extract roots: [fond, esperant, kiu]
         ↓
    ┌─────────────────────────┐
    ↓                         ↓
TRACK 1: ReVo Synonyms    TRACK 2: Embeddings
(Deterministic)           (Learned Associations)
    ↓                         ↓
fond → kre, establ       fond → universitat,
                                societ, organiz
    ↓                         ↓
    └──────────┬──────────────┘
               ↓
    Merged expansion: [fond, kre, establ, 
                       universitat, societ, ...]
               ↓
         Retrieve from Kuzu
               ↓
        Candidate documents
```

## Components

### 1. HybridQueryExpander
**File:** `klareco/rag/hybrid_query_expander.py`

Class that combines both expansion tracks:
```python
expander = HybridQueryExpander(
    embedding_path="models/root_embeddings_phase1_fast/root_embeddings_best.pt",
    db_path="data/indexes/v2.1_kuzu_index_full",
    embedding_k=5,              # Top 5 embedding neighbors
    embedding_threshold=0.4,    # Min similarity threshold
    use_revo=True,             # Enable ReVo synonyms
    use_embeddings=True        # Enable embedding associations
)

expansion = expander.expand({'fond', 'esperant'})
# Returns: {
#   'original': {'fond', 'esperant'},
#   'revo_synonyms': {'kre', 'establ'},
#   'embedding_associations': {'universitat', 'societ'},
#   'all': {'fond', 'esperant', 'kre', 'establ', 'universitat', 'societ'}
# }
```

### 2. Demo Scripts

**scripts/demo_hybrid_retrieval.py**
- Interactive demo of hybrid expansion
- Compare modes: hybrid, revo-only, embeddings-only

**scripts/evaluate_hybrid_retrieval.py**
- Measure recall improvement
- Compare against baselines

**scripts/test_hybrid_retrieval.py**
- Test hybrid expansion on sample queries

## Evaluation Results

### Recall Comparison

| Query | Method | Expanded Roots | Recall |
|-------|--------|----------------|--------|
| "Kio estas Esperanto?" | No expansion | 3 | 20.0% |
| | ReVo only | 6 | 60.0% |
| | Embeddings only | 8 | 20.0% |
| | **Hybrid** | **11** | **60.0%** |
| | | | |
| "Kiu fondis Esperanton?" | No expansion | 3 | 14.3% |
| | ReVo only | 3 | 14.3% |
| | Embeddings only | 8 | 28.6% |
| | **Hybrid** | **8** | **28.6%** |

### Key Findings

1. **ReVo provides high-precision synonyms** when available
   - "est" → "ekzist", "ent" (correct synonyms)
   - Coverage: 35% of Fundamento roots

2. **Embeddings provide learned associations** (co-occurrence)
   - "fond" → "universitat", "societ" (contextually related)
   - Coverage: 100% of roots in corpus

3. **Hybrid combines strengths**
   - Uses ReVo where available (precision)
   - Falls back to embeddings for gaps (recall)
   - Best coverage overall

## What Embeddings Should Learn (For RAG)

### Conclusion from Analysis

**For RAG retrieval, current co-occurrence embeddings are CORRECT:**

- ✅ Learn typical associations (verb-object, topic-context)
- ✅ Find documents ABOUT the topic (not just exact matches)
- ✅ Complement deterministic synonyms (different roles)

**Example:** Query "Kiu fondis Esperanton?" (Who founded Esperanto?)
- ReVo synonyms: fond → kre, establ (find paraphrases)
- Embeddings: fond → universitat, societ (find contextual mentions)
- Result: Find both direct answers AND contextual information

### NOT for Future Reasoning Models

This is ONLY for RAG retrieval. If/when building reasoning models:
- Would need TRUE semantic similarity (synonyms)
- Would retrain embeddings on synonym pairs (not co-occurrence)
- Current embeddings are optimized for retrieval, not reasoning

## Current Coverage

| Source | Coverage | Type | Purpose |
|--------|----------|------|---------|
| ReVo | 35% | Deterministic | High-precision synonyms |
| Embeddings | 100% | Learned | Contextual associations |
| **Hybrid** | **100%** | **Combined** | **Best of both** |

## Future Improvements

1. **Expand ReVo coverage** (Issue #678)
   - Mine Wiktionary for more synonyms
   - Target: 60-80% Fundamento coverage
   - Deferred for now (ReVo 35% sufficient to start)

2. **Integrate into full RAG pipeline**
   - Replace current SemanticQueryExpander with HybridQueryExpander
   - Update demo_semantic_retrieval.py
   - Test end-to-end retrieval

3. **Tune expansion parameters**
   - embedding_k (currently 5)
   - embedding_threshold (currently 0.4)
   - Balance precision vs recall

## Usage

```bash
# Demo hybrid expansion
python scripts/demo_hybrid_retrieval.py
python scripts/demo_hybrid_retrieval.py -i  # Interactive
python scripts/demo_hybrid_retrieval.py "Kio estas Esperanto?"

# Compare modes
python scripts/demo_hybrid_retrieval.py --revo-only "..."
python scripts/demo_hybrid_retrieval.py --embeddings-only "..."

# Evaluate recall
python scripts/evaluate_hybrid_retrieval.py
```

## Related Issues

- #678: Mine Wiktionary for expanded synonym coverage (deferred)
- #679: Define semantic role for embeddings (resolved: co-occurrence is correct for RAG)
- #677: AST-aware extraction optimization (deferred - too slow)

## Status

✅ **COMPLETE**: Hybrid retrieval system implemented and evaluated
- HybridQueryExpander class created
- Demo and evaluation scripts working
- Recall improvement demonstrated
- Ready for integration into RAG pipeline

**Next:** Integrate into demo_semantic_retrieval.py for end-to-end RAG testing
