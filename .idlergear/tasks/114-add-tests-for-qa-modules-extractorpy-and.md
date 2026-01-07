---
id: 114
title: "Add tests for QA modules: extractor.py and reranker.py (0% \u2192 85%)"
state: open
created: '2026-01-07T00:12:02.400827Z'
labels:
- testing
- coverage
priority: medium
---
## Goal

Add tests for the Q&A extraction and reranking modules.

## Files to Cover

| File | Current | Target | Lines |
|------|---------|--------|-------|
| `qa/extractor.py` | 0% | 85% | 109 |
| `qa/reranker.py` | 0% | 85% | 152 |

## Test Categories

### Extractor Tests
- `test_extract_answer_from_sentence()`
- `test_extract_entity_answer()`
- `test_extract_numeric_answer()`
- `test_extract_from_slot()`
- `test_no_answer_found()`

### Reranker Tests
- `test_rerank_by_relevance()`
- `test_rerank_by_entity_match()`
- `test_rerank_boost_exact_match()`
- `test_rerank_penalize_mismatch()`

### Integration Tests
- `test_extract_and_rerank_pipeline()`
- `test_qa_end_to_end()`

## Acceptance Criteria

- [ ] extractor.py at 85%+ coverage
- [ ] reranker.py at 85%+ coverage
- [ ] Edge cases tested (no answer, multiple answers)
- [ ] Integration with retriever tested

## Estimated Effort

~4-5 hours
