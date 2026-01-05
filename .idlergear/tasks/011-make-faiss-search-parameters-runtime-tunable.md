---
id: 11
title: Make FAISS search parameters runtime-tunable (nprobe, efSearch)
state: open
created: '2026-01-04T00:37:09.587063Z'
labels:
- enhancement
- M2
- faiss
priority: high
---
## Problem

Current implementation hardcodes search parameters:

```python
# Set once at search time, not configurable by caller
self.faiss_index.nprobe = min(32, self.faiss_index.nlist)
```

**Issues**:
- ❌ Can't A/B test different nprobe values without code changes
- ❌ Can't tune per-query (e.g., higher accuracy for critical queries)
- ❌ No way to adjust speed/accuracy tradeoff in production

## FAISS Best Practice

**FAISS Documentation**:
> "nprobe is always a way of adjusting the tradeoff between speed and accuracy"
> "Parameters are adjustable at runtime without rebuilding the index"

## Proposed Solution

### Add search parameters to retriever API

```python
class FAISSSlotRetriever:
    def __init__(self, ..., default_search_params: Optional[Dict] = None):
        """
        Args:
            default_search_params: Default FAISS search parameters
                - nprobe: IVF cells to search (default: 32)
                - efSearch: HNSW search depth (default: 16)
        """
        self.default_search_params = default_search_params or {
            'nprobe': 32,
            'efSearch': 16,
        }
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        faiss_top_n: int = 500,
        slot_weight: float = 0.6,
        full_weight: float = 0.4,
        **search_params,  # Allow runtime override
    ) -> List[Tuple[float, Dict]]:
        """
        Args:
            search_params: Override default FAISS parameters
                - nprobe: Number of IVF cells to search
                - efSearch: HNSW search depth
        """
        # Merge default + runtime params
        params = {**self.default_search_params, **search_params}
        
        # Apply to FAISS index
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = params.get('nprobe', 32)
        
        if hasattr(self.faiss_index, 'quantizer'):
            quantizer = self.faiss_index.quantizer
            if hasattr(quantizer, 'hnsw'):
                quantizer.hnsw.efSearch = params.get('efSearch', 16)
        
        # Continue with search...
```

### Usage Examples

```python
# Default parameters
results = retriever.search("Kiu kreis Esperanton?")

# Higher accuracy for critical query
results = retriever.search("Kiu kreis Esperanton?", nprobe=64, efSearch=32)

# Faster search for bulk processing
results = retriever.search(query, nprobe=16, efSearch=8)

# A/B testing different configurations
for nprobe in [16, 32, 64, 128]:
    results = retriever.search(query, nprobe=nprobe)
    measure_accuracy(results)
```

## Expected Benefits

- ✅ **Experimentation**: Tune parameters without rebuilding index
- ✅ **Production flexibility**: Adjust speed/accuracy per use case
- ✅ **Per-query tuning**: High accuracy for important queries, fast for bulk
- ✅ **A/B testing**: Compare configurations easily

## Files to Change

- `klareco/rag/slot_retriever_faiss.py:187-274` (search method)
- `klareco/rag/slot_retriever_multifaiss.py` (if applicable)
- `scripts/demo_slot_retrieval.py` (add parameter examples)

## Related

- Issue #178: Experiment with top_k and rerank_top_n (this adds nprobe/efSearch tuning)

## Acceptance Criteria

- [ ] `search()` method accepts **search_params kwargs
- [ ] nprobe parameter applied to IVF indexes
- [ ] efSearch parameter applied to HNSW quantizers
- [ ] Default parameters documented
- [ ] Demo script shows parameter tuning examples
- [ ] No breaking changes to existing API
