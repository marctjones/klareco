# Klareco Vision: Esperanto-Native AI

**The Big Idea**: Build a smaller, more efficient AI system by leveraging Esperanto's regular grammar - offloading linguistic structure to deterministic rules and focusing learned parameters on reasoning and world knowledge.

---

## The Core Insight

Traditional LLMs waste enormous capacity learning:
- **Grammar** - Esperanto grammar is regular and known
- **Morphology** - We encode this in compositional embeddings
- **Syntax** - Our parser already extracts this
- **Token prediction** - "hund" + "o" is trivial in Esperanto

**What if we factor these out?**

The remaining "reasoning core" only needs to learn:
- World knowledge (facts and relationships)
- Reasoning patterns (cause/effect, counterfactuals, analogies)
- Discourse structure (how ideas connect)
- Pragmatics (what's relevant, what to emphasize)

---

## Architecture Comparison

### Traditional LLM
```
Input: "Kiu estas Frodo?"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monolithic LLM                           │
│                                                             │
│   Token → Token → Token → Token → Token → Token ...        │
│                                                             │
│   Learns EVERYTHING: grammar, meaning, facts, reasoning    │
│   Parameters: 7B - 1.7T                                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Output: "Frodo estas hobito..."
```

### Esperanto-Native Architecture
```
Input: "Kiu estas Frodo?"
         │
         ▼
┌──────────────────┐
│ Parser (rules)   │  ← No learning, deterministic
│ Text → AST       │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Compositional    │  ← Minimal learning (~500K params)
│ Embeddings       │     Morpheme composition
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ RAG Retrieval    │  ← Retrieves relevant ASTs from corpus
│ (semantic index) │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Reasoning Core   │  ← THE KEY INNOVATION
│ (learned model)  │     Operates on ASTs, not tokens
│ AST → AST        │     Maybe 100M-500M params?
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Linearizer       │  ← No learning, deterministic
│ AST → Text       │
└──────────────────┘
         │
         ▼
Output: "Frodo estas hobito..."
```

---

## The Reasoning Core: Key Research Question

This is the novel component. Instead of predicting tokens, it:
- **Input**: Query AST + Retrieved Context ASTs
- **Output**: Response AST (structured meaning)
- **Learns**: How to compose, transform, and generate semantic structures

### Possible Architectures

#### Option A: Graph Neural Network
```
ASTs as graphs → Graph Transformer → New graph
- Nodes are semantic units (with compositional embeddings)
- Edges are grammatical relationships
- Model learns graph transformations
```

#### Option B: Sequence-to-Sequence on Linearized ASTs
```
"( frazo ( subj Frodo ) ( pred estas ) ( obj hobito ) )"
                    ↓
              Small Transformer
                    ↓
"( frazo ( subj Frodo ) ( pred portis ) ( obj Ringo ) )"

- Much smaller vocabulary than words
- Structure is explicit, model focuses on semantics
```

#### Option C: Neuro-Symbolic Hybrid
```
┌─────────────────────────┐
│ Neural: Selection       │  ← Which facts? What emphasis?
│ (learned)               │     Creativity, relevance
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ Symbolic: Composition   │  ← How to combine?
│ (rules)                 │     Grammar, agreement, structure
└─────────────────────────┘
```

---

## Why This Might Work

### Esperanto's Advantages
1. **Regular morphology** - No exceptions, fully compositional
2. **Predictable syntax** - SVO but flexible, clear markers
3. **Explicit grammar** - Case, number, tense all marked
4. **Agglutinative** - Meaning built from pieces

### Efficiency Gains
| Model | Parameters | Why |
|-------|-----------|-----|
| GPT-4 | ~1.7T | Learns everything from scratch |
| Llama-7B | 7B | Still learns all grammar |
| **Klareco Reasoning Core** | 100M-500M? | Only learns reasoning |

The hypothesis: **Clean structural input → fewer parameters needed**

---

## What Each Component Knows

| Component | What It Knows | Learned? |
|-----------|---------------|----------|
| Parser | Esperanto grammar rules | No (rule-based) |
| Compositional Embeddings | Word meanings, morpheme semantics | Yes (~500K params) |
| RAG Retrieval | Which facts are relevant | Partially (index + embeddings) |
| **Reasoning Core** | How to combine facts, answer questions | Yes (100M-500M params?) |
| Linearizer | How to express AST as text | No (rule-based) |

---

## Capabilities and Limitations

### What It Could Do Well
- Factual Q&A grounded in corpus
- Explain entities based on retrieved facts
- Compare things with known properties
- Grammatically correct output (by construction)
- Explainable reasoning (AST transformations visible)
- No hallucination (grounded in retrieved facts)

### What It Would Struggle With (Initially)
- Open-ended creative writing (limited to recombining known facts)
- Complex multi-hop reasoning chains
- World knowledge not in corpus
- Nuance and style (output may be formulaic)

### Future Possibilities
- Train on synthetic reasoning examples
- Learn to generate novel AST structures (creativity)
- Multi-hop reasoning through iterative retrieval
- Style transfer at linearization stage

---

## Roadmap to Vision

### Current Status (November 2025)
- [x] Parser: Morpheme-aware, 16 rules
- [x] Corpus: 5.3M sentences with ASTs
- [x] Compositional Embeddings: Training now
- [x] RAG: Two-stage retrieval working
- [ ] Vocabulary expansion system: Just built

### Phase 1: Complete RAG System (Current)
- Finish compositional embedding training
- Integrate enhanced retrieval
- Build demo showing semantic role disambiguation
- **Goal**: Best-in-class Esperanto RAG

### Phase 2: AST Trail & Explainability
- Track all transformations through pipeline
- Visualize reasoning steps
- **Goal**: Fully transparent retrieval

### Phase 3: Rule-Based Reasoning (Weeks 4-6)
- Implement basic reasoning patterns as rules
- Question type → Retrieval strategy → Composition
- **Goal**: Working Q&A without neural reasoning

### Phase 4: Neural Reasoning Core (Weeks 7-12)
- Design AST-to-AST architecture
- Create training data (AST transformation pairs)
- Train small model (start with 50M params)
- **Goal**: Learn to compose answers from retrieved ASTs

### Phase 5: Evaluation & Iteration
- Compare against LLM baselines
- Identify capability gaps
- Iterate on architecture
- **Goal**: Prove or refine the thesis

### Phase 6: Generative Capabilities
- Train reasoning core on creative examples
- Enable generation beyond retrieved facts
- Style and fluency improvements
- **Goal**: Creative Esperanto generation

---

## Open Research Questions

1. **How small can the reasoning core be?**
   - Hypothesis: 100M-500M params sufficient for basic Q&A
   - Need to test empirically

2. **What's the right AST representation for the model?**
   - Linearized S-expressions?
   - Graph structure directly?
   - Hybrid?

3. **How to handle creativity/generation?**
   - Can we learn to synthesize novel AST structures?
   - Or is creativity fundamentally different?

4. **Does this generalize beyond Esperanto?**
   - Could apply to other regular languages
   - Could be a preprocessing layer for irregular languages

5. **What's the training data?**
   - AST-to-AST pairs from parallel sentences?
   - Synthetic reasoning examples?
   - Human-annotated reasoning chains?

---

## Why This Matters

If this works:
- **Smaller models** - More accessible AI
- **Explainable** - See exactly how conclusions reached
- **No hallucination** - Grounded in corpus
- **Linguistic insight** - Proves value of structured representation
- **Esperanto validation** - Shows the language design was right

If it doesn't work fully:
- **Hybrid approach** - AST preprocessing for traditional LLMs
- **Specialized tools** - Best Esperanto NLP toolkit
- **Research contribution** - Data on what structure buys you

---

## Next Conversation Topics

When ready to move beyond RAG:
1. Design the AST-to-AST training data format
2. Prototype graph transformer on small examples
3. Create evaluation benchmark for reasoning
4. Compare minimal neural vs rule-based composition

---

*This document captures the long-term vision. Current work is tracked in `IMPLEMENTATION_ROADMAP_V2.md`.*
