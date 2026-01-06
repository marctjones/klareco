---
id: 14
title: Add FAISS ID-based filtering for tier constraints
state: open
created: '2026-01-04T00:37:12.087870Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Problem

Current tier filtering happens **after** FAISS retrieval in Python:

```python
# Stage 1: FAISS retrieves top-500
faiss_results = faiss_index.search(query_emb, 500)

# Stage 2: Filter by tier in Python
authoritative_results = [
    r for r in faiss_results 
    if metadata[r.index].get('tier', 6) <= 3
]

# Stage 3: Slot reranking
final_results = slot_rerank(authoritative_results)
```

**Problems**:
- ❌ **Wasteful** - FAISS searches all tiers, then we discard non-matching
- ❌ **Less accurate** - Can only rerank from 500 candidates (limited by FAISS top-k)
- ❌ **No structural filter** - Tier is deterministic metadata, not learned

## FAISS Solution: ID Selectors

**FAISS Documentation**:
> "Ignore a subset of index vectors according to a predicate on the vector ids"

**FAISS provides**:
```python
# Create selector for IDs to search
selector = faiss.IDSelectorArray(valid_ids)

# Search with filtering
scores, indices = index.search(query_emb, k, params=selector)
```

## Proposed Solution

### Tier-Based Filtering

```python
class TierFilteredRetriever:
    """Retriever with FAISS-native tier filtering."""
    
    def __init__(self, index_path: Path, indexer: SlotBasedIndexer):
        self.index = faiss.read_index(str(index_path / "faiss_index.bin"))
        self.metadata = self._load_metadata(index_path / "metadata.jsonl")
        
        # Pre-compute tier ID selectors
        self._tier_selectors = self._create_tier_selectors()
    
    def _create_tier_selectors(self) -> Dict[str, faiss.IDSelectorArray]:
        """Create FAISS ID selectors for tier filtering."""
        selectors = {}
        
        # Authoritative sources (tier 0-3)
        authoritative_ids = np.array([
            i for i, meta in enumerate(self.metadata)
            if meta.get('tier', 6) <= 3
        ], dtype=np.int64)
        selectors['authoritative'] = faiss.IDSelectorArray(authoritative_ids)
        
        # Fundamento only (tier 1)
        fundamento_ids = np.array([
            i for i, meta in enumerate(self.metadata)
            if meta.get('tier') == 1
        ], dtype=np.int64)
        selectors['fundamento'] = faiss.IDSelectorArray(fundamento_ids)
        
        # All (no filtering)
        selectors['all'] = None
        
        return selectors
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        tier_filter: str = 'all',  # 'all', 'authoritative', 'fundamento'
        faiss_top_n: int = 500,
    ):
        """
        Search with tier filtering applied during FAISS search.
        
        Args:
            tier_filter: Which tier selector to use
                - 'all': Search all tiers (no filtering)
                - 'authoritative': Only tier 0-3 (Fundamento, ReVo, curated)
                - 'fundamento': Only tier 1 (Fundamento)
        """
        # Get query embedding
        query_emb = self._embed_query(query)
        
        # Get selector
        selector = self._tier_selectors.get(tier_filter)
        
        # Search with filtering
        if selector is not None:
            # FAISS applies filter during search (efficient!)
            params = faiss.SearchParameters()
            params.sel = selector
            scores, indices = self.index.search(query_emb, faiss_top_n, params=params)
        else:
            # No filtering
            scores, indices = self.index.search(query_emb, faiss_top_n)
        
        # Continue with slot reranking...
        return self._slot_rerank(indices, scores, top_k)
```

## Benefits

**Performance**:
- ✅ **Faster** - FAISS skips non-matching vectors during search
- ✅ **Larger candidate pool** - Can retrieve top-1000 authoritative instead of top-500 all

**Accuracy**:
- ✅ **Better recall** - More candidates to rerank from
- ✅ **Authoritative bias** - Prefer Fundamento/ReVo over Wikipedia

**Architecture**:
- ✅ **Combines deterministic + learned** - Tier (0 params) + embeddings (733K params)
- ✅ **Aligns with Klareco philosophy** - Maximize deterministic processing

## Use Cases

### 1. Grammar Questions → Fundamento Only
```python
# Query: "Kio estas la akuzativo?"
results = retriever.search(query, tier_filter='fundamento')
# Only searches official Fundamento grammar rules
```

### 2. Factual Questions → Authoritative Sources
```python
# Query: "Kiu kreis Esperanton?"
results = retriever.search(query, tier_filter='authoritative')
# Searches Fundamento + ReVo + curated texts, skips general Wikipedia
```

### 3. General Questions → All Sources
```python
# Query: "Kio estas la Hobito?"
results = retriever.search(query, tier_filter='all')
# Searches entire corpus including literature
```

## Advanced: Custom Selectors

```python
def search_with_custom_filter(
    self,
    query: str,
    filter_fn: Callable[[Dict], bool],
    top_k: int = 10,
):
    """Search with custom metadata filter."""
    # Create selector from filter function
    valid_ids = np.array([
        i for i, meta in enumerate(self.metadata)
        if filter_fn(meta)
    ], dtype=np.int64)
    
    selector = faiss.IDSelectorArray(valid_ids)
    
    # Search with custom filter
    params = faiss.SearchParameters()
    params.sel = selector
    scores, indices = self.index.search(query_emb, top_k, params=params)
    
    return indices, scores

# Usage:
results = retriever.search_with_custom_filter(
    query,
    lambda m: m.get('source', {}).get('type') == 'literature',  # Books only
)
```

## Implementation Notes

**FAISS ID Selector Types**:
1. `IDSelectorArray(ids)` - Whitelist of valid IDs
2. `IDSelectorRange(imin, imax)` - Range of IDs
3. `IDSelectorBatch(ids)` - For batch queries
4. `IDSelectorNot(selector)` - Negation
5. `IDSelectorAnd(sel1, sel2)` - Intersection
6. `IDSelectorOr(sel1, sel2)` - Union

**Performance Consideration**:
- Pre-compute selectors at load time (not per query)
- Store as `self._tier_selectors` dict
- ID arrays must be sorted for best performance

## Files to Change

- `klareco/rag/slot_retriever_faiss.py` - Add tier filtering
- `klareco/rag/retriever.py` - Add to base Retriever too
- `scripts/demo_slot_retrieval.py` - Demo tier filtering

## References

- FAISS Wiki: "Pre-filtering and post-filtering"
- FAISS SearchParameters: https://github.com/facebookresearch/faiss/blob/main/faiss/impl/IDSelector.h

## Acceptance Criteria

- [ ] IDSelectorArray implemented for tier filtering
- [ ] Pre-computed tier selectors (authoritative, fundamento, all)
- [ ] search() accepts tier_filter parameter
- [ ] Benchmark shows improved accuracy for authoritative-only queries
- [ ] Demo shows tier filtering examples
- [ ] Custom filter function supported
