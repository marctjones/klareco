# Klareco Vision: Esperanto-Native AI

**Core Thesis**: Traditional LLMs waste capacity learning grammar. By making Esperanto grammar 100% deterministic, we focus all learned parameters on reasoning—achieving comparable capabilities with 50-100x fewer parameters.

## Architecture

```
Text → Parser (0 params) → AST → Embeddings (733K) → RAG → Reasoning Core (20-100M) → Linearizer (0 params) → Text
```

## What's Deterministic vs Learned

| Component | Parameters | Notes |
|-----------|------------|-------|
| Parser (16 rules) | 0 | Grammar + semantic roles |
| Morphology | 0 | Decomposition is deterministic |
| **Stage 1: Semantic Model** | **733K** | **✓ COMPLETE** |
| - Root embeddings | 712K | 11,121 roots × 64d |
| - Affix transforms V2 | 21K | Low-rank transformations |
| Stage 2: Grammatical | ~52K | Negation, tense, mood (NEXT) |
| Stage 3: Discourse | ~100K | Coreference, coherence |
| Reasoning core | 20-100M | The actual goal (FUTURE) |
| Linearizer | 0 | AST → text |

## Current Status

### ✅ **Milestone M1: COMPLETE** (Dec 31, 2025)

Evaluation vs OLMo 1B on 50-question benchmark:
- **Partial match**: 20% (Klareco) vs 8% (OLMo) = **2.5x better**
- **Latency**: 690ms vs 38,329ms = **56x faster**
- **Parameters**: 733K vs 1.18B = **1,600x smaller**

**Key Insight**: Retrieval is the bottleneck (35% recall). Grammar/embeddings work well.

### 🎯 **Milestone M2: IN PROGRESS**

**Focus**: Improve retrieval corpus to 80% recall, establish OLMo baselines for fair comparison.

**Current blockers**:
- Retrieval recall: 35% → target 80%
- Missing entities: Zamenhof, Fundamento, Esperanto history
- Data quality issues (4 files with x-notation, Wikipedia cleanup needed)

### 📦 **Production Components**

- **Parser**: 16 Esperanto rules, 91.8% parse rate
- **Corpus**: 4.38M sentences indexed with compositional embeddings
- **Two-stage retrieval**: Structural filtering + FAISS semantic search
- **EnrichedAST**: Combines parser AST + trained embeddings
- **Extractive Q&A**: Template-based answering from retrieved context

## Success Criteria

If 50-100M params achieves 80%+ on Esperanto Q&A while being:
- Fully explainable (AST trail)
- Grammatically perfect (by construction)
- Hallucination-free (grounded in corpus)

**→ The thesis is proven.**

## Why This Matters

- **Smaller models** = more accessible AI
- **Explainable** = see exactly how conclusions reached
- **No hallucination** = grounded in corpus
- **Linguistic insight** = proves value of structured representation
- **M1 proves concept**: 733K params competitive with 1.18B params

## Roadmap

- ✅ Stage 1: Semantic model (root + affix embeddings) - COMPLETE
- 🔲 Stage 2: Grammatical model (negation, tense, mood) - NEXT
- 🔲 Stage 3: Discourse model (coreference, coherence)
- 🔲 Stage 4: Reasoning core (20-100M params)
- 🔲 M2: 80% retrieval recall
- 🔲 M3: Multi-hop reasoning
- 🔲 M4: AST-constrained generation