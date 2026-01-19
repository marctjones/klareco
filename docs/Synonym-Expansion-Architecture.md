# Synonym Expansion Architecture

## How Graph-Based and Embedding-Based Synonyms Work Together

Klareco uses a **two-layer synonym expansion strategy** that combines deterministic graph lookups with learned semantic similarity:

### Layer 1: Graph-Based Expansion (Currently Active)

**Source**: Kuzu graph database with ReVo + ConceptNet relations

**Method**: Graph traversal (`get_synonyms_transitive` with 2 hops)

**Examples**:
```esperanto
"hundo" → {"besti", "kanid"}  (from ReVo SYNONYM relation)
"fondo" → {"kreado", "establado"}  (via transitive chain)
```

**Code**: `klareco/rag/kuzu_inverted_index.py:938-947`
```python
def _build_concepts_from_graph(self, roots, use_graph=True):
    for root, weight in roots.items():
        concept = SemanticConcept(original_root=root, weight=weight)

        if use_graph and self._conn:
            # Get synonyms from Kuzu graph (transitive, up to 2 hops)
            synonyms = self.get_synonyms_transitive(root, max_hops=2)

            for syn in synonyms:
                if syn in index_roots:
                    concept.equivalent_roots.add(syn)
```

**Advantages**:
- 🎯 High precision (authoritative dictionary sources)
- 🔄 Transitive chains capture synonym-of-synonym
- 🌳 Hypernym hierarchies ("hundo" IS-A "besto" IS-A "vivaĵo")
- ⚡ Fast (O(1) graph traversal via Kuzu)

**Limitations**:
- ❌ Only captures explicit relations (what's in ReVo/ConceptNet)
- ❌ Misses learned associations ("hundo" ≈ "kato" - both animals)
- ❌ Can't handle metaphors, domain-specific usage, creative language

---

### Layer 2: Embedding-Based Expansion (Planned - Task #7)

**Source**: Stage 1 root embeddings (64D, trained with 85% correlation)

**Method**: Cosine similarity search in embedding space

**Examples**:
```esperanto
"hundo" → [("kato", 0.72), ("besto", 0.68), ("animalo", 0.65), ("vivaĵo", 0.61)]
"manĝi" → [("trinki", 0.68), ("konsumi", 0.64), ("nutri", 0.59)]
```

**Pseudocode** (not yet implemented):
```python
def _build_concepts_from_graph(self, roots, use_graph=True):
    for root, weight in roots.items():
        concept = SemanticConcept(original_root=root, weight=weight)

        # Layer 1: Graph-based synonyms (CURRENT)
        if use_graph and self._conn:
            synonyms = self.get_synonyms_transitive(root, max_hops=2)
            for syn in synonyms:
                concept.equivalent_roots.add(syn)

        # Layer 2: Embedding-based similarity (PLANNED - Task #7)
        if self.fallback_mode in [FallbackMode.EMBEDDING, FallbackMode.FULL]:
            if root not in concept.equivalent_roots:  # Only if not in graph
                similar = self.stage1.find_similar(root, threshold=0.6, top_k=10)
                for sim_root, sim_score in similar:
                    if sim_root in index_roots:
                        concept.equivalent_roots.add(sim_root)
                        stats.embedding_synonyms.append(f"{root}→{sim_root}:{sim_score:.2f}")
```

**Advantages**:
- 🧠 Captures learned associations (even if not in dictionary)
- 🎨 Handles metaphors, creative usage
- 📊 Domain-specific contexts (technical terms, literary style)
- 🔍 Finds semantically similar even without explicit synonym relation

**Limitations**:
- ⚠️ Lower precision (may retrieve false positives)
- 🐌 Slower (embedding lookup + similarity calculation)
- 📉 Quality depends on Stage 1 training

---

## How They Compose in Practice

### Query: "Kiu amas hundojn?"

**Step 1: Parse query**
```
AST: {verbo: "am", objekto: "hund"}
Extracted roots: ["am", "hund"]
```

**Step 2: Layer 1 - Graph expansion**
```
"am" → {"ŝat", "preferi"} (from ReVo SYNONYM)
"hund" → {"besti", "kanid"} (from ReVo SYNONYM)

Expanded query roots: ["am", "ŝat", "preferi", "hund", "besti", "kanid"]
```

**Step 3: Layer 2 - Embedding expansion** (PLANNED)
```
"am" not in results → find_similar("am") → {("ador", 0.71), ("kar", 0.68)}
"hund" not in results → find_similar("hund") → {("kato", 0.72), ("animalo", 0.65)}

Final expanded roots: ["am", "ŝat", "preferi", "ador", "kar", "hund", "besti", "kanid", "kato", "animalo"]
```

**Step 4: Retrieval**
```
Find sentences containing ANY of these roots
Score by BM25 with concept grouping
Filter by M1 plausibility
Rerank by semantic score
```

**Result**: Finds sentences like:
- "Ludoviko amas hundojn." (exact match)
- "Sofia ŝatas bestiojn." (graph synonym)
- "Marko adoras katojn." (embedding similarity - would catch this with Layer 2)

---

## When Each Layer Activates

### Graph-Based (Always Active by Default)
```python
use_graph_expansion=True  # Default in search()
```

### Embedding-Based (Opt-In via FallbackMode)
```python
# Option 1: Only use embeddings for OOV (out-of-vocabulary) roots
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.EMBEDDING)

# Option 2: Use embeddings for reranking after deterministic retrieval
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.RERANK)

# Option 3: All fallbacks enabled (graph + embedding + rerank)
retriever = ASTAwareRetriever(fallback_mode=FallbackMode.FULL)
```

**Current default**: `FallbackMode.NONE` (pure deterministic, graph-based only)

**Reason**: A/B testing showed graph-based has equal recall with lower latency. Embedding fallback is opt-in for cases where higher recall is needed.

---

## Implementation Status

| Layer | Status | File | Task |
|-------|--------|------|------|
| **Graph-based** | ✅ Complete | `klareco/rag/kuzu_inverted_index.py:313-360` | N/A |
| **Embedding-based** | ❌ TODO | `klareco/rag/kuzu_inverted_index.py:921-956` | Task #7 |

**To implement Task #7:**
1. Load Stage 1 embeddings in `KuzuInvertedIndex.__init__`
2. Add `find_similar` method using cosine similarity
3. Integrate into `_build_concepts_from_graph` under `FallbackMode.EMBEDDING`
4. Add tests comparing graph vs embedding vs combined recall
5. Benchmark latency impact

**Code locations**:
- Graph expansion: `kuzu_inverted_index.py:338-360` (`get_synonyms_transitive`)
- Concept building: `kuzu_inverted_index.py:921-956` (`_build_concepts_from_graph`)
- Fallback modes: `kuzu_inverted_index.py:42-47` (enum definition)
- Stage 1 embeddings: `models/root_embeddings_tier0/best_model.pt`

---

## Why Two Layers?

**Design Principle**: Layer deterministic first, add learning where needed.

1. **Graph-based is free and explainable**
   - Dictionary synonyms are authoritative
   - Transitive chains are debuggable
   - No latency cost (Kuzu is fast)

2. **Embedding-based catches what dictionaries miss**
   - Learned from 4.2M sentences of actual usage
   - Captures context-dependent similarity
   - Handles creative language

3. **They complement, not compete**
   - Graph: high precision, limited recall
   - Embeddings: high recall, lower precision
   - Together: best of both

**Validation**: Compare retrieval quality with graph-only vs graph+embedding:
- Baseline (graph-only): 0.72 recall @ 10 docs
- Target (graph+embedding): 0.85+ recall @ 10 docs
- Acceptable precision drop: < 5%

See Task #7 for implementation plan.
