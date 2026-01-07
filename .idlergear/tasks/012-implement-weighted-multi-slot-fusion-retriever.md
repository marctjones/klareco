---
id: 12
title: Implement weighted multi-slot fusion retriever (replace intersection with fusion)
state: closed
created: '2026-01-04T00:37:10.220134Z'
labels:
- faiss
- future
- superseded
priority: low
---
## Problem

Current `MultiFAISSSlotRetriever` uses **intersection** of slot results:

```python
# Only returns documents that match ALL slots
common_doc_ids = set(subj_results) & set(verb_results) & set(obj_results)
```

**Results**:
- ❌ **75% recall** (worst of all retrievers)
- ❌ **MRR=0.56** (poor ranking)
- ❌ Too strict - misses valid partial matches

**Why intersection fails**:
- Query "Kiu kreis Esperanton?" has no explicit subject (it's a question word)
- Document "Zamenhof kreis Esperanton" should match on VERB+OBJ only
- Intersection requires all 3 slots → miss!

## Proposed Solution: Weighted Fusion

**Instead of intersection (AND), use weighted sum (OR with scoring)**:

```python
# Query each slot index independently
subj_results = subj_index.search(query_slots['SUBJ'], k=100)
verb_results = verb_index.search(query_slots['VERB'], k=100)
obj_results = obj_index.search(query_slots['OBJ'], k=100)

# Fuse results with slot weights
from collections import defaultdict
scores = defaultdict(float)

for score, doc_id in subj_results:
    scores[doc_id] += 0.3 * score  # SUBJ weight

for score, doc_id in verb_results:
    scores[doc_id] += 0.4 * score  # VERB weight (highest)

for score, doc_id in obj_results:
    scores[doc_id] += 0.3 * score  # OBJ weight

# Return top-k by combined score
final_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

## Expected Improvements

**Accuracy**:
- 75% → 85-90% recall (match current FAISS performance)
- Better MRR (documents can score on 1, 2, or 3 slots)

**Speed**:
- **1-2ms latency** (current MultiFAISS: 1.0ms, FAISS: 5.1ms)
- 3× faster than current FAISSSlotRetriever!
- Parallel slot queries (can use threading)

**Explainability**:
- Can show which slots contributed to match
- "Matched on VERB (0.85) and OBJ (0.72), missing SUBJ"

## Implementation Plan

### 1. Create `FusedMultiSlotRetriever` class

```python
class FusedMultiSlotRetriever:
    """Weighted fusion of per-slot FAISS indexes."""
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
    ):
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,  # Verbs most discriminative
            'OBJ': 0.3,
        }
        
        # Load separate indexes (reuse MultiFAISS logic)
        self._load_slot_indexes()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        slot_k: int = 100,  # Candidates per slot
        fusion_method: str = 'weighted_sum',
    ) -> List[Tuple[float, Dict]]:
        """
        Multi-slot retrieval with weighted fusion.
        
        Args:
            query: Query text
            top_k: Final results to return
            slot_k: Candidates to retrieve per slot
            fusion_method: 'weighted_sum', 'max', 'min'
        """
        # Parse query and extract slots
        query_slots = self._parse_query(query)
        
        # Query each slot index
        results_by_slot = {}
        for slot, emb in query_slots.items():
            if emb is not None and slot in self.slot_indexes:
                results_by_slot[slot] = self.slot_indexes[slot].search(emb, slot_k)
        
        # Fuse results
        if fusion_method == 'weighted_sum':
            scores = self._weighted_sum_fusion(results_by_slot)
        elif fusion_method == 'max':
            scores = self._max_fusion(results_by_slot)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Return top-k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def _weighted_sum_fusion(self, results_by_slot):
        """Combine slot scores with weights."""
        scores = defaultdict(float)
        
        for slot, results in results_by_slot.items():
            weight = self.slot_weights.get(slot, 1.0)
            for score, doc_id in results:
                scores[doc_id] += weight * score
        
        return scores
    
    def explain_match(self, query: str, doc_id: int) -> Dict:
        """Show which slots contributed to match."""
        # ... return per-slot scores
```

### 2. Alternative Fusion Methods

```python
def _max_fusion(self, results_by_slot):
    """Use maximum score across slots."""
    scores = {}
    for slot, results in results_by_slot.items():
        for score, doc_id in results:
            scores[doc_id] = max(scores.get(doc_id, 0), score)
    return scores

def _rrf_fusion(self, results_by_slot):
    """Reciprocal Rank Fusion (RRF)."""
    scores = defaultdict(float)
    k = 60  # RRF constant
    
    for slot, results in results_by_slot.items():
        for rank, (_, doc_id) in enumerate(results, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    
    return scores
```

## Optimization: Parallel Slot Queries

```python
from concurrent.futures import ThreadPoolExecutor

def search_parallel(self, query: str, top_k: int = 10, slot_k: int = 100):
    """Query slot indexes in parallel."""
    query_slots = self._parse_query(query)
    
    def query_slot(slot, emb):
        return slot, self.slot_indexes[slot].search(emb, slot_k)
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(query_slot, slot, emb)
            for slot, emb in query_slots.items()
            if emb is not None and slot in self.slot_indexes
        ]
        
        results_by_slot = {
            future.result()[0]: future.result()[1]
            for future in futures
        }
    
    # Fuse and return
    scores = self._weighted_sum_fusion(results_by_slot)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

## Files to Create

- `klareco/rag/slot_retriever_fused.py` - New FusedMultiSlotRetriever class
- `scripts/benchmark_fused_retriever.py` - Benchmark script

## Files to Modify

- `scripts/demo_slot_retrieval.py` - Add fusion demo

## Expected Benchmark Results

| Retriever | Recall@10 | Latency | Speedup |
|-----------|-----------|---------|---------|
| FAISSSlot (current best) | 85% | 5.1ms | 1× |
| MultiFAISS (intersection) | 75% | 1.0ms | 5× (but bad accuracy) |
| **Fused (proposed)** | **85-90%** | **1-2ms** | **3-5×** |

## Acceptance Criteria

- [ ] FusedMultiSlotRetriever class implemented
- [ ] Weighted sum fusion working
- [ ] Parallel slot queries implemented
- [ ] Benchmark shows 85%+ recall at <2ms latency
- [ ] Explain match shows per-slot contributions
- [ ] No regression vs current FAISSSlotRetriever accuracy
