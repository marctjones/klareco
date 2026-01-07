---
id: 112
title: "Add tests for slot retriever variants (8 files, 10-15% \u2192 80%)"
state: open
created: '2026-01-07T00:12:01.010755Z'
labels:
- testing
- coverage
priority: medium
---
## Goal

Add comprehensive tests for all 8 slot retriever implementations.

## Files to Cover

| File | Current | Target | Lines |
|------|---------|--------|-------|
| `slot_retriever.py` (base) | ~20% | 85% | ~100 |
| `slot_retriever_faiss.py` | 14% | 80% | 148 |
| `slot_retriever_hnsw.py` | 14% | 80% | 162 |
| `slot_retriever_hybrid.py` | 15% | 80% | 150 |
| `slot_retriever_mmap.py` | 11% | 80% | 166 |
| `slot_retriever_multifaiss.py` | 12% | 80% | 175 |
| `slot_retriever_scann.py` | 14% | 80% | 150 |
| `slot_retriever_sqlite.py` | 12% | 80% | 157 |

## Test Categories

### Unit Tests
- `test_slot_similarity()` - slot matching logic
- `test_feature_bonus()` - negation, tense matching
- `test_partial_match_bonus()` - question handling
- `test_prefilter()` - candidate selection
- `test_rerank()` - final ordering

### Integration Tests
- `test_search_basic()` - end-to-end search
- `test_search_with_slots()` - slot-based matching
- `test_search_compound_words()` - HEAD/MOD handling
- `test_search_questions()` - question word handling

### Mock Strategy
- Use small test index (100 documents)
- Mock FAISS/HNSW/ScaNN for unit tests
- Real index for integration tests (marked slow)

## Test File Structure

```
tests/
  rag/
    test_slot_retriever_base.py      # Base class tests
    test_slot_retriever_faiss.py     # FAISS-specific
    test_slot_retriever_hnsw.py      # HNSW-specific
    test_slot_retriever_hybrid.py    # Hybrid-specific
    test_slot_retriever_mmap.py      # Mmap-specific
    test_slot_retriever_multifaiss.py
    test_slot_retriever_scann.py
    test_slot_retriever_sqlite.py
    conftest.py                      # Shared fixtures
```

## Acceptance Criteria

- [ ] All 8 retriever files at 80%+ coverage
- [ ] Unit tests for core matching logic
- [ ] Integration tests with small test index
- [ ] Tests marked with @pytest.mark.slow for real index
- [ ] Shared fixtures in conftest.py

## Estimated Effort

~12-15 hours (8 files × ~1.5 hours each)
