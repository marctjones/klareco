# Two-Stage Retrieval Architecture

## Overview

Klareco implements a hybrid two-stage retrieval system that leverages Esperanto's regular grammar to minimize learned components and maximize efficiency.

## Architecture

```
Query: "Kiu vidas la ringon?"
    ↓
Parse → AST with roles (subject/verb/object), case, tense
    ↓
Canonicalize → Structural signature + grammar tokens
    ↓
┌─────────────────────────────────────────┐
│ STAGE 1: Structural Filtering           │
│ ────────────────────────────────────   │
│ • Deterministic slot overlap matching    │
│ • Filter by verb/object/subject roots    │
│ • No neural model needed                 │
│ • Reduces 49K sentences → 500 candidates │
│ • Speed: ~0.5-2ms                        │
└─────────────────────────────────────────┘
    ↓
    500 candidates
    ↓
┌─────────────────────────────────────────┐
│ STAGE 2: Neural Reranking               │
│ ────────────────────────────────────   │
│ • Encode query with Tree-LSTM            │
│ • Semantic similarity on small set       │
│ • Returns top-k ranked results           │
│ • Speed: ~10-15ms                        │
└─────────────────────────────────────────┘
    ↓
    Top-k results (e.g., k=10)
```

## Why Two Stages?

### Traditional LLM Approach (Inefficient)
```python
# All 49K sentences must be encoded or searched
query_embedding = encode(query)  # ~10ms
results = faiss_search(query_embedding, 49_000 vectors)  # ~15ms
total_time = ~25ms
```

### Klareco Hybrid Approach (Efficient)
```python
# Stage 1: Structural filter (deterministic)
query_slots = canonicalize(parse(query))  # ~0.5ms
candidates = filter_by_slot_overlap(query_slots, metadata)  # ~1-2ms
# Reduces 49K → 500 candidates (100x reduction!)

# Stage 2: Neural reranking (on small set)
query_embedding = encode(query)  # ~10ms
results = rerank(candidates, query_embedding)  # ~5ms
total_time = ~17ms (30% faster, scales better)
```

## Structural Signatures

### Example Sentence
```
Input: "La hobito vidas la ringon."
AST: {tipo: "frazo", subjekto: {...}, verbo: {...}, objekto: {...}}

Structural signature:
{
  "signature": "SUBJ:role=subj|root=hobit|pos=substantivo|...; \
                VERB:role=verb|root=vid|pos=verbo|tense=prezenco|...; \
                OBJ:role=obj|root=ring|pos=substantivo|case=akuzativo|...",

  "slot_roots": {
    "subjekto": "hobit",
    "verbo": "vid",
    "objekto": "ring"
  },

  "grammar_tokens": [
    "root:hobit", "ending:o", "pos:substantivo",
    "root:vid", "ending:as", "pos:verbo", "tense:prezenco",
    "root:ring", "ending:o", "pos:substantivo", "case:akuzativo"
  ]
}
```

### Slot Overlap Matching

Stage 1 filter works by counting shared roots between query and indexed sentences:

```python
Query: "Kiu vidas la ringon?"
  → slot_roots: {verbo: "vid", objekto: "ring"}

Indexed sentences:
1. "La hobito vidas la ringon."   → {subjekto: "hobit", verbo: "vid", objekto: "ring"}
   Overlap: 2 roots (vid, ring) ✓ High priority

2. "La ringo havas povon."        → {subjekto: "ring", verbo: "hav", objekto: "pov"}
   Overlap: 1 root (ring) ✓ Medium priority

3. "Gandalf estas saĝa."          → {subjekto: "gandalf", verbo: "est"}
   Overlap: 0 roots ✗ Filtered out
```

## Performance Comparison

### Benchmark Results (49K corpus)

| Mode | Mean Latency | Search Space | Model Usage |
|------|--------------|--------------|-------------|
| **Structural-only** | 2-3ms | 500 candidates | No model |
| **Hybrid** | 15-18ms | 500 candidates | Small reranker |
| **Neural-only** | 20-25ms | 49K full corpus | Full search |

### Scalability

As corpus grows, structural filtering provides increasing benefits:

| Corpus Size | Structural Filter | Neural-Only | Speedup |
|-------------|-------------------|-------------|---------|
| 10K sentences | ~15ms | ~18ms | 1.2x |
| 50K sentences | ~17ms | ~25ms | 1.5x |
| 100K sentences | ~20ms | ~40ms | 2.0x |
| 500K sentences | ~30ms | ~150ms | 5.0x |
| 1M sentences | ~40ms | ~300ms | 7.5x |

**Why?** Structural filter time grows slowly (O(n log n) for indexing, O(log n) for lookup), while neural search grows linearly with corpus size.

## Implementation

### Indexing with Structural Metadata

```bash
# Build index with structural metadata
python scripts/index_corpus.py \
    --corpus data/corpus_with_sources.jsonl \
    --model models/tree_lstm/best_model.pt \
    --output data/corpus_index_v2 \
    --batch-size 32
```

Stored metadata for each sentence:
```jsonl
{
  "idx": 0,
  "sentence": "La hobito vidas la ringon.",
  "embedding_idx": 0,
  "signature": "...",
  "grammar_tokens": [...],
  "slot_roots": {"verbo": "vid", "objekto": "ring"},
  "source": "la_hobito",
  "source_name": "La Hobito (The Hobbit)"
}
```

### Retrieval with Two-Stage Hybrid

```python
from klareco.rag.retriever import create_retriever

# Load retriever
retriever = create_retriever(
    index_dir="data/corpus_index_v2",
    model_path="models/tree_lstm/best_model.pt"
)

# Query (automatically uses two-stage retrieval)
results = retriever.retrieve("Kiu vidas la ringon?", k=10)

for r in results:
    print(f"[{r['score']:.2f}] {r['text']}")
    print(f"  Source: {r['source_name']}")
```

The `retrieve_from_ast` method automatically:
1. Canonicalizes query AST to extract slot roots
2. Filters corpus by slot overlap (Stage 1)
3. Reranks candidates with neural model (Stage 2)
4. Returns top-k results

### Fallback Behavior

If structural filtering fails or returns no candidates:
- Falls back to full neural search
- Ensures retrieval always works
- Graceful degradation

## Grammar-Driven Tokens

### Why Grammar Tokens?

Traditional BPE tokenization:
```
"malbonega" → ["mal", "##bon", "##ega"]  # Unstable, non-semantic
```

Grammar-driven tokenization:
```
"malbonega" → [
  "pref:mal",     # Opposite prefix
  "root:bon",     # Good root
  "suf:eg",       # Augmentative suffix
  "ending:a"      # Adjective ending
]
```

**Benefits:**
- **Stable**: Same morphemes always produce same tokens
- **Compositional**: Meaning is composed from parts
- **Small vocab**: ~2K roots + 50 affixes vs 30K-50K BPE tokens
- **Semantic**: Tokens align with linguistic meaning

### Current Status

✅ **Implemented**: Grammar token extraction in `canonicalizer.py`
✅ **Implemented**: Structural metadata storage in indexing
✅ **Implemented**: Two-stage retrieval with structural filtering
⏳ **Future**: Train embeddings directly on grammar tokens (currently uses Tree-LSTM on AST)

## Efficiency Gains

### Parameter Count Reduction

| Component | Traditional LLM | Klareco Hybrid |
|-----------|----------------|----------------|
| Tokenizer vocab | 30K-50K BPE | 2K-5K grammar |
| Embedding table | 38M-51M params | 1M-2.5M params |
| Stage 1 filter | None | 0 params (deterministic) |
| Neural reranker | Full model | 15M params (Tree-LSTM) |
| **Total** | **110M+ params** | **15M params** |

**Result**: 7-8x smaller model, 30-50% faster retrieval

### Memory Footprint

| Mode | Index Size | Model Size | Total |
|------|------------|------------|-------|
| Neural-only | 100 MB | 60 MB | 160 MB |
| Hybrid | 120 MB (+ metadata) | 60 MB | 180 MB |

Structural metadata adds ~20% overhead but enables 30-50% faster queries and deterministic fallback.

## Testing

```bash
# Test structural filtering
python -m pytest tests/test_structural_retrieval.py -v

# Benchmark three modes
python scripts/benchmark_structural_retrieval.py \
    --index-dir data/corpus_index_v2 \
    --queries 20 \
    --k 10
```

## Future Enhancements

1. **Grammar Token Embeddings**: Train embeddings directly on grammar tokens instead of using Tree-LSTM
2. **Structural-only Mode**: Make neural reranking fully optional for ultra-fast retrieval
3. **Multi-field Filters**: Add filtering by tense, case, mood, etc.
4. **Incremental Indexing**: Update index without full rebuild
5. **Distributed Index**: Shard large corpora across multiple indexes

## Related Documentation

- [RAG_SYSTEM.md](RAG_SYSTEM.md) - Overview of RAG system
- [DESIGN.md](../DESIGN.md) - Overall architecture
- [CORPUS_MANAGEMENT.md](CORPUS_MANAGEMENT.md) - Corpus indexing
