---
id: 124
title: Evaluate ChromaDB for unified embedding + metadata storage
state: open
created: '2026-01-07T02:58:34.249227Z'
labels:
- enhancement
- retrieval
- infrastructure
priority: medium
---
## Motivation

Currently we have fragmented storage:
- HNSW index for embedding search (`slot_hybrid.hnsw`)
- JSONL for documents + slots + features (`slot_index.jsonl`)
- Numpy offset cache for seeking (`slot_index.offsets.npy`)

This requires multiple lookups per query and custom code for metadata filtering.

## Proposed Solution

ChromaDB is a purpose-built vector database for RAG that could unify all three:

```python
# Current (fragmented)
labels, distances = hnsw_index.knn_query(embedding)  # Get IDs
doc = _get_document(id)  # Seek in 30GB JSONL
slots = doc['slots']     # Parse JSON

# With ChromaDB (unified)
results = collection.query(
    query_embeddings=[embedding],
    n_results=100,
    where={"fraztipo": "deklaro"},  # Metadata filtering!
    include=["documents", "metadatas", "embeddings"]
)
```

## Benefits

1. **Unified storage** - embeddings, text, and metadata in one place
2. **Metadata filtering** - filter by `fraztipo`, `negita`, `tempo` at query time
3. **No custom offset caching** - ChromaDB handles indexing
4. **Persistence** - built-in disk persistence
5. **Slot filtering** - could filter by slot presence (has_SUBJ, has_VERB)

## Challenges

1. **Scale**: 4.4M vectors × 128d = ~2.2GB embeddings alone
2. **Migration**: Need to convert existing JSONL → ChromaDB
3. **Slot embeddings**: ChromaDB stores one embedding per doc, but we have 4 (full + SUBJ/VERB/OBJ)
   - Option A: Store only full_embedding, recompute slot similarity in reranking
   - Option B: Create separate collections for each slot type
4. **Performance**: Need to benchmark vs current HNSW

## Tasks

1. [ ] Install ChromaDB and test with small subset (10K docs)
2. [ ] Design schema for storing slots as metadata vs separate collections
3. [ ] Write migration script from slot_index.jsonl
4. [ ] Benchmark query latency vs current HNSW implementation
5. [ ] Implement ChromaDBSlotRetriever if benchmarks are favorable

## Alignment with Architecture

- **Deterministic**: ChromaDB is just storage - our deterministic AST analysis remains unchanged
- **Memory efficient**: ChromaDB handles disk-backed storage
- **Simplification**: Removes need for custom JSONL seeking and offset caching
