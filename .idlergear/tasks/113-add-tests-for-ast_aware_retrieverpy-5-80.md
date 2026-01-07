---
id: 113
title: "Add tests for ast_aware_retriever.py (5% \u2192 80%)"
state: open
created: '2026-01-07T00:12:01.699015Z'
labels:
- testing
- coverage
priority: medium
---
## Goal

Add comprehensive tests for the AST-aware retriever module.

## Current State

- Coverage: 5% (641 lines missing)
- This is a critical module for Q&A accuracy

## Test Categories

### Question Classification Tests
- `test_classify_who_question()`
- `test_classify_what_question()`
- `test_classify_where_question()`
- `test_classify_when_question()`
- `test_classify_how_many_question()`

### Entity Recognition Tests
- `test_recognize_person_entity()`
- `test_recognize_place_entity()`
- `test_recognize_proper_noun()`

### Pattern Matching Tests
- `test_pattern_match_verb_object()`
- `test_pattern_match_with_synonyms()`
- `test_pattern_match_negation()`

### Search Strategy Tests
- `test_search_entity_focused()`
- `test_search_pattern_matching()`
- `test_search_hybrid()`
- `test_strategy_selection()`

### Prefilter Tests
- `test_hnsw_prefilter()`
- `test_keyword_prefilter()`
- `test_slot_reranking()`

### Integration Tests
- `test_search_kiu_fondis_esperanton()` - Classic test case
- `test_search_factual_question()`
- `test_search_grammar_question()`

## Mock Strategy

- Mock HNSW index for unit tests
- Use small test corpus (100 sentences)
- Mock embedding models with fixed vectors

## Acceptance Criteria

- [ ] Coverage: 5% → 80%
- [ ] All search strategies tested
- [ ] Prefilter logic tested
- [ ] Question classification tested
- [ ] Integration tests with real parsing

## Estimated Effort

~6-8 hours
