---
id: 115
title: "Add tests for embeddings modules (43-50% \u2192 90%)"
state: open
created: '2026-01-07T00:12:03.133186Z'
labels:
- testing
- coverage
priority: medium
---
## Goal

Improve test coverage for embedding modules.

## Files to Cover

| File | Current | Target | Lines Missing |
|------|---------|--------|---------------|
| `embeddings/compositional.py` | 43% | 90% | 167 |
| `embeddings/linguistic_embeddings.py` | 48% | 90% | 33 |
| `embeddings/topical_embeddings.py` | 50% | 90% | 30 |
| `embeddings/unknown_tracker.py` | 0% | 85% | 102 |

## Test Categories

### Compositional Embeddings
- `test_get_root_embedding()`
- `test_get_affix_transform()`
- `test_compose_word_embedding()`
- `test_unknown_root_fallback()`
- `test_prefix_suffix_combination()`

### Linguistic Embeddings
- `test_load_checkpoint()`
- `test_embed_roots()`
- `test_vocabulary_lookup()`

### Topical Embeddings
- `test_load_topical_model()`
- `test_topical_similarity()`

### Unknown Tracker
- `test_log_unknown_root()`
- `test_get_candidates()`
- `test_mark_added()`
- `test_mark_rejected()`
- `test_persistence()`

## Acceptance Criteria

- [ ] compositional.py at 90%+ coverage
- [ ] linguistic_embeddings.py at 90%+ coverage
- [ ] topical_embeddings.py at 90%+ coverage
- [ ] unknown_tracker.py at 85%+ coverage

## Estimated Effort

~5-6 hours
