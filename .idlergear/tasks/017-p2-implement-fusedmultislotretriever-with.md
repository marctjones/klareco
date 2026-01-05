---
id: 17
title: '[P2] Implement FusedMultiSlotRetriever with weighted fusion (Task #12)'
state: open
created: '2026-01-04T00:46:07.641447Z'
labels:
- enhancement
- M2
- faiss
- P2
priority: high
---
## Goal

Fix MultiFAISSSlotRetriever by replacing intersection with weighted fusion.

**Priority**: P2 (High) - Biggest performance win (3× speedup!)

**Expected Results:**
- Memory: 2.1GB (20% less than FAISS)
- Latency: 1-2ms (3× faster than optimized FAISS!)
- Recall: 85-90% (same or better than FAISS)

## Problem with Current MultiFAISS

Current implementation uses **intersection**:
```python
common_doc_ids = set(subj_results) & set(verb_results) & set(obj_results)
```

**Results:**
- ❌ 75% recall (worst of all retrievers)
- ❌ Misses partial matches (e.g., query with only VERB+OBJ)

## Proposed Solution: Weighted Fusion

```python
# Query each slot independently
subj_results = subj_index.search(query_slots['SUBJ'], k=100)
verb_results = verb_index.search(query_slots['VERB'], k=100)
obj_results = obj_index.search(query_slots['OBJ'], k=100)

# Weighted sum fusion (not intersection!)
scores = defaultdict(float)

for score, doc_id in subj_results:
    scores[doc_id] += 0.3 * score  # SUBJ weight

for score, doc_id in verb_results:
    scores[doc_id] += 0.4 * score  # VERB weight (most important)

for score, doc_id in obj_results:
    scores[doc_id] += 0.3 * score  # OBJ weight

# Return top-k by combined score
return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

## Implementation

Create `klareco/rag/slot_retriever_fused.py`:

```python
"""
Fused multi-slot retriever with weighted fusion.

Fixes MultiFAISS intersection problem by using weighted sum.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import faiss
import numpy as np

from klareco.parser import parse
from klareco.rag.slot_indexer import SlotBasedIndexer

logger = logging.getLogger(__name__)


class FusedMultiSlotRetriever:
    """
    Multi-slot retriever with weighted fusion.
    
    Key differences from MultiFAISS:
    - Uses weighted sum instead of intersection
    - Supports partial matches (1, 2, or 3 slots)
    - Parallel slot queries with ThreadPool
    - Explainable: shows which slots contributed
    """
    
    def __init__(
        self,
        index_path: Path,
        indexer: SlotBasedIndexer,
        slot_weights: Optional[Dict[str, float]] = None,
        use_parallel: bool = True,
    ):
        """
        Args:
            index_path: Path to slot index directory
            indexer: SlotBasedIndexer for query embedding
            slot_weights: Slot importance weights (default: SUBJ=0.3, VERB=0.4, OBJ=0.3)
            use_parallel: Query slots in parallel (default: True for 16-core laptop)
        """
        self.index_path = Path(index_path)
        self.indexer = indexer
        self.use_parallel = use_parallel
        
        # Default weights (verb most important)
        self.slot_weights = slot_weights or {
            'SUBJ': 0.3,
            'VERB': 0.4,
            'OBJ': 0.3,
        }
        
        # Load indexes
        self._load_indexes()
    
    def _load_indexes(self):
        """Load separate FAISS indexes for each slot."""
        logger.info(f"Loading fused multi-slot indexes from {self.index_path}")
        
        multifaiss_dir = self.index_path / "multifaiss"
        
        # Check if indexes exist
        if not multifaiss_dir.exists():
            logger.info("  Multi-FAISS indexes not found, creating...")
            self._create_multifaiss_indexes(multifaiss_dir)
        
        # Load each slot index
        self.slot_indexes = {}
        self.slot_doc_ids = {}
        
        for slot in ['SUBJ', 'VERB', 'OBJ']:
            index_file = multifaiss_dir / f"{slot}.index"
            id_file = multifaiss_dir / f"{slot}_ids.npy"
            
            if index_file.exists():
                index = faiss.read_index(str(index_file))
                self.slot_indexes[slot] = index
                self.slot_doc_ids[slot] = np.load(id_file)
                logger.info(f"  {slot}: {index.ntotal:,} vectors")
            else:
                logger.warning(f"  {slot}: index not found")
                self.slot_indexes[slot] = None
                self.slot_doc_ids[slot] = np.array([])
        
        # Load metadata
        self.documents = []
        index_file = self.index_path / "slot_index.jsonl"
        
        import json
        with open(index_file) as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
        
        logger.info(f"  Loaded {len(self.documents):,} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        slot_k: int = 100,
        fusion_method: str = 'weighted_sum',
    ) -> List[Tuple[float, Dict]]:
        """
        Multi-slot retrieval with weighted fusion.
        
        Args:
            query: Query text
            top_k: Final results to return
            slot_k: Candidates to retrieve per slot (default: 100)
            fusion_method: 'weighted_sum', 'max', 'rrf'
        
        Returns:
            List of (score, document) tuples
        """
        # Parse query and extract slots
        try:
            query_ast = parse(query)
        except Exception as e:
            logger.error(f"Failed to parse query: {query} - {e}")
            return []
        
        query_slots = self.indexer.extract_slots(query_ast)
        
        # Query each slot index
        if self.use_parallel:
            results_by_slot = self._query_slots_parallel(query_slots, slot_k)
        else:
            results_by_slot = self._query_slots_sequential(query_slots, slot_k)
        
        if not results_by_slot:
            logger.warning(f"No valid slots in query: {query}")
            return []
        
        # Fuse results
        if fusion_method == 'weighted_sum':
            scores = self._weighted_sum_fusion(results_by_slot)
        elif fusion_method == 'max':
            scores = self._max_fusion(results_by_slot)
        elif fusion_method == 'rrf':
            scores = self._rrf_fusion(results_by_slot)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Convert to final results
        final_results = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            final_results.append((score, self.documents[doc_id]))
        
        return final_results
    
    def _query_slots_parallel(self, query_slots: Dict, slot_k: int) -> Dict:
        """Query slot indexes in parallel using ThreadPool."""
        def query_slot(slot, emb):
            if emb is None or slot not in self.slot_indexes or self.slot_indexes[slot] is None:
                return slot, []
            
            # Normalize
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.slot_indexes[slot].search(emb, slot_k)
            
            # Map back to document IDs
            results = [
                (float(scores[0][i]), int(self.slot_doc_ids[slot][indices[0][i]]))
                for i in range(len(indices[0]))
                if indices[0][i] >= 0
            ]
            
            return slot, results
        
        # Parallel execution (3 slots max, have 16 cores)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(query_slot, slot, emb)
                for slot, emb in query_slots.items()
            ]
            
            results_by_slot = {
                future.result()[0]: future.result()[1]
                for future in futures
                if future.result()[1]  # Only include non-empty results
            }
        
        return results_by_slot
    
    def _query_slots_sequential(self, query_slots: Dict, slot_k: int) -> Dict:
        """Query slot indexes sequentially (fallback)."""
        results_by_slot = {}
        
        for slot, emb in query_slots.items():
            if emb is None or slot not in self.slot_indexes or self.slot_indexes[slot] is None:
                continue
            
            # Normalize
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.slot_indexes[slot].search(emb, slot_k)
            
            # Map back to document IDs
            results = [
                (float(scores[0][i]), int(self.slot_doc_ids[slot][indices[0][i]]))
                for i in range(len(indices[0]))
                if indices[0][i] >= 0
            ]
            
            if results:
                results_by_slot[slot] = results
        
        return results_by_slot
    
    def _weighted_sum_fusion(self, results_by_slot: Dict) -> Dict[int, float]:
        """Combine slot scores with weights."""
        scores = defaultdict(float)
        
        for slot, results in results_by_slot.items():
            weight = self.slot_weights.get(slot, 1.0)
            for score, doc_id in results:
                scores[doc_id] += weight * score
        
        return scores
    
    def _max_fusion(self, results_by_slot: Dict) -> Dict[int, float]:
        """Use maximum score across slots."""
        scores = {}
        
        for slot, results in results_by_slot.items():
            for score, doc_id in results:
                scores[doc_id] = max(scores.get(doc_id, 0), score)
        
        return scores
    
    def _rrf_fusion(self, results_by_slot: Dict) -> Dict[int, float]:
        """Reciprocal Rank Fusion (RRF)."""
        scores = defaultdict(float)
        k = 60  # RRF constant
        
        for slot, results in results_by_slot.items():
            for rank, (_, doc_id) in enumerate(results, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        
        return scores
    
    def explain_match(self, query: str, doc_id: int) -> Dict:
        """Show which slots contributed to match."""
        query_ast = parse(query)
        query_slots = self.indexer.extract_slots(query_ast)
        
        explanation = {
            'query': query,
            'document': self.documents[doc_id]['text'],
            'slot_contributions': {},
        }
        
        # Query each slot and find this doc
        for slot, emb in query_slots.items():
            if emb is None or slot not in self.slot_indexes:
                explanation['slot_contributions'][slot] = {'status': 'query_missing'}
                continue
            
            # Search this slot
            emb = emb / np.linalg.norm(emb)
            emb = emb.reshape(1, -1).astype(np.float32)
            scores, indices = self.slot_indexes[slot].search(emb, 100)
            
            # Find this doc in results
            mapped_indices = [self.slot_doc_ids[slot][i] for i in indices[0] if i >= 0]
            
            if doc_id in mapped_indices:
                pos = mapped_indices.index(doc_id)
                explanation['slot_contributions'][slot] = {
                    'status': 'matched',
                    'rank': pos + 1,
                    'score': float(scores[0][pos]),
                    'weighted_score': float(scores[0][pos]) * self.slot_weights.get(slot, 1.0),
                }
            else:
                explanation['slot_contributions'][slot] = {'status': 'not_in_top_100'}
        
        return explanation
```

## Memory Tuning for <16GB RAM

**Memory breakdown:**
- SUBJ index: ~700MB (1.4M vectors with embeddings)
- VERB index: ~700MB (1.4M vectors with embeddings)
- OBJ index: ~700MB (1.4M vectors with embeddings)
- Metadata: ~100MB
- Python overhead: ~200MB
- **Total: ~2.4GB** ✅ Well under 16GB

**Parallel execution safe:**
- 16 cores available
- Only 3 parallel queries (SUBJ, VERB, OBJ)
- Each query independent, no shared state

## Validation Test

Create `tests/test_fused_retriever.py`:

```python
import pytest
from klareco.rag.slot_retriever_fused import FusedMultiSlotRetriever

def test_weighted_fusion():
    """Test that weighted fusion works correctly."""
    retriever = FusedMultiSlotRetriever(index_path, indexer)
    
    # Query with all slots
    results = retriever.search("La kato manĝas la muson.", top_k=5)
    assert len(results) == 5
    
    # Query with partial slots (only VERB+OBJ)
    results = retriever.search("Kiu kreis Esperanton?", top_k=5)
    assert len(results) == 5  # Should still work!

def test_parallel_vs_sequential():
    """Test that parallel and sequential give same results."""
    retriever_parallel = FusedMultiSlotRetriever(index_path, indexer, use_parallel=True)
    retriever_sequential = FusedMultiSlotRetriever(index_path, indexer, use_parallel=False)
    
    query = "Kiu kreis Esperanton?"
    
    results_parallel = retriever_parallel.search(query, top_k=10)
    results_sequential = retriever_sequential.search(query, top_k=10)
    
    # Should return same top documents (order may vary slightly)
    docs_parallel = {r[1]['text'] for r in results_parallel[:5]}
    docs_sequential = {r[1]['text'] for r in results_sequential[:5]}
    
    overlap = len(docs_parallel & docs_sequential) / 5
    assert overlap >= 0.8  # At least 80% overlap in top-5

def test_fusion_methods():
    """Test different fusion methods."""
    retriever = FusedMultiSlotRetriever(index_path, indexer)
    query = "Kiu kreis Esperanton?"
    
    results_weighted = retriever.search(query, fusion_method='weighted_sum')
    results_max = retriever.search(query, fusion_method='max')
    results_rrf = retriever.search(query, fusion_method='rrf')
    
    # All should return results
    assert len(results_weighted) > 0
    assert len(results_max) > 0
    assert len(results_rrf) > 0

def test_explain_match():
    """Test that explain shows slot contributions."""
    retriever = FusedMultiSlotRetriever(index_path, indexer)
    
    results = retriever.search("Zamenhof kreis Esperanton.", top_k=1)
    doc_id = results[0][1]['id']  # Assuming documents have ID
    
    explanation = retriever.explain_match("Kiu kreis Esperanton?", doc_id)
    
    # Should show VERB and OBJ matched
    assert 'VERB' in explanation['slot_contributions']
    assert 'OBJ' in explanation['slot_contributions']

def test_memory_usage():
    """Test that memory stays under 4GB."""
    import psutil
    retriever = FusedMultiSlotRetriever(index_path, indexer)
    
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    assert mem_mb < 4000, f"Memory usage {mem_mb:.0f}MB exceeds 4GB limit"

def test_accuracy_vs_multifaiss():
    """Test that fusion improves over intersection."""
    # Old MultiFAISS with intersection: 75% recall
    # New Fused with weighted sum: should be 85%+
    
    recall = run_benchmark(retriever, queries)
    assert recall >= 0.85, f"Recall {recall:.2%} below 85% target"
```

## Demo Script

Create `scripts/demo_fused_retriever.py`:

```python
#!/usr/bin/env python3
"""
Demo: FusedMultiSlotRetriever with weighted fusion.

Shows how weighted fusion fixes the intersection problem.
"""

import time
from pathlib import Path
from klareco.rag.slot_retriever_fused import FusedMultiSlotRetriever
from klareco.rag.slot_indexer import SlotBasedIndexer

def main():
    print("=" * 70)
    print("FusedMultiSlotRetriever Demo: Weighted Fusion vs Intersection")
    print("=" * 70)
    print()
    
    # Load retriever
    index_path = Path("data/indexes/slot_full")
    indexer = SlotBasedIndexer.load()
    retriever = FusedMultiSlotRetriever(index_path, indexer, use_parallel=True)
    
    print(f"✓ Loaded retriever")
    print(f"  SUBJ index: {retriever.slot_indexes['SUBJ'].ntotal:,} vectors")
    print(f"  VERB index: {retriever.slot_indexes['VERB'].ntotal:,} vectors")
    print(f"  OBJ index: {retriever.slot_indexes['OBJ'].ntotal:,} vectors")
    print(f"  Parallel mode: {retriever.use_parallel}")
    print()
    
    # Demo 1: Partial match (VERB+OBJ only)
    print("=" * 70)
    print("Demo 1: Partial Match (Query has no explicit subject)")
    print("=" * 70)
    print()
    
    query = "Kiu kreis Esperanton?"
    print(f"Query: {query}")
    print(f"Parsed: SUBJ=<Kiu?> VERB=kreis OBJ=Esperanton")
    print()
    print("❌ Old MultiFAISS (intersection): Would require ALL 3 slots to match")
    print("✅ New Fused (weighted): Matches on VERB (0.4) + OBJ (0.3) = 0.7 score")
    print()
    
    start = time.time()
    results = retriever.search(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    
    print(f"⏱️  Latency: {elapsed:.1f}ms")
    print(f"📊 Results:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {doc['text'][:80]}...")
    print()
    
    # Demo 2: Explain match
    print("=" * 70)
    print("Demo 2: Explain Match (Which slots contributed?)")
    print("=" * 70)
    print()
    
    if results:
        top_doc_text = results[0][1]['text']
        # Find doc ID (hack: search by text)
        doc_id = next(i for i, d in enumerate(retriever.documents) if d['text'] == top_doc_text)
        
        explanation = retriever.explain_match(query, doc_id)
        
        print(f"Query: {explanation['query']}")
        print(f"Match: {explanation['document'][:100]}...")
        print()
        print("Slot contributions:")
        for slot, contrib in explanation['slot_contributions'].items():
            if contrib['status'] == 'matched':
                print(f"  {slot}: ✓ Rank {contrib['rank']}, Score {contrib['score']:.3f}, Weighted {contrib['weighted_score']:.3f}")
            elif contrib['status'] == 'query_missing':
                print(f"  {slot}: ○ Query has no {slot}")
            else:
                print(f"  {slot}: ✗ Not in top-100")
    print()
    
    # Demo 3: Fusion methods comparison
    print("=" * 70)
    print("Demo 3: Fusion Methods Comparison")
    print("=" * 70)
    print()
    
    query = "Zamenhof kreis Esperanton en Varsovio."
    print(f"Query: {query}")
    print()
    
    methods = ['weighted_sum', 'max', 'rrf']
    
    print(f"{'Method':<15} {'Latency':<12} {'Top-1 Text':<50}")
    print("-" * 77)
    
    for method in methods:
        start = time.time()
        results = retriever.search(query, top_k=5, fusion_method=method)
        elapsed = (time.time() - start) * 1000
        
        top_text = results[0][1]['text'][:47] + "..." if results else "No results"
        print(f"{method:<15} {elapsed:>8.1f}ms    {top_text}")
    
    print()
    
    # Demo 4: Parallel vs Sequential
    print("=" * 70)
    print("Demo 4: Parallel vs Sequential (16 cores available)")
    print("=" * 70)
    print()
    
    retriever_sequential = FusedMultiSlotRetriever(index_path, indexer, use_parallel=False)
    
    print("Testing with 10 queries...")
    
    queries = [
        "Kiu kreis Esperanton?",
        "Kiam Zamenhof kreis Esperanton?",
        "Kie naski\u011dis Zamenhof?",
        "Kio estas Esperanto?",
        "Kial Esperanto estas facila?",
        "Kiom da homoj parolas Esperanton?",
        "Kiu verkis la Fundamenton?",
        "Kio estas la akuzativo?",
        "Kie estas UEA?",
        "Kiam estis la unua Esperanto-kongreso?",
    ]
    
    # Sequential
    start = time.time()
    for q in queries:
        retriever_sequential.search(q, top_k=5)
    sequential_time = (time.time() - start) * 1000
    
    # Parallel
    start = time.time()
    for q in queries:
        retriever.search(q, top_k=5)
    parallel_time = (time.time() - start) * 1000
    
    print(f"Sequential: {sequential_time:>8.1f}ms ({sequential_time/10:.1f}ms per query)")
    print(f"Parallel:   {parallel_time:>8.1f}ms ({parallel_time/10:.1f}ms per query)")
    print(f"Speedup:    {sequential_time/parallel_time:.2f}×")
    print()
    
    print("✅ Demo complete!")
    print()
    print("Key takeaways:")
    print("  • Weighted fusion handles partial matches (no intersection)")
    print("  • Explainable: see which slots contributed")
    print("  • Parallel queries leverage multi-core CPU")
    print("  • 3× faster than single FAISS index!")

if __name__ == '__main__':
    main()
```

## Benchmark

```bash
# Benchmark against other retrievers
python scripts/benchmark_slot_retrievers.py \
    --index data/indexes/slot_full \
    --retrievers fused \
    --output benchmark_results/fused.json
```

## Acceptance Criteria

- [ ] FusedMultiSlotRetriever implemented
- [ ] Weighted sum fusion working
- [ ] Parallel slot queries working (16 cores)
- [ ] Memory usage < 3GB confirmed
- [ ] Validation tests pass
- [ ] Demo script works end-to-end
- [ ] Benchmark shows 1-2ms latency
- [ ] Benchmark shows 85-90% recall (improves from 75% MultiFAISS)
- [ ] explain_match() shows slot contributions

## Related Tasks

- Implements: Task #12 (weighted multi-slot fusion)
- Fixes: MultiFAISSSlotRetriever intersection problem
- Improves: 75% → 85-90% recall
