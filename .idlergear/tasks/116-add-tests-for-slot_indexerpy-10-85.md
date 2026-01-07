---
id: 116
title: "Add tests for slot_indexer.py (10% \u2192 85%)"
state: open
created: '2026-01-07T00:12:03.896845Z'
labels:
- testing
- coverage
priority: medium
---
## Goal

Add comprehensive tests for the slot-based indexer.

## Current State

- Coverage: 10% (184 lines missing)
- Critical for index quality

## Test Categories

### Slot Extraction Tests
- `test_extract_subject_slot()`
- `test_extract_verb_slot()`
- `test_extract_object_slot()`
- `test_extract_from_compound_word()`
- `test_extract_head_vs_modifier()` (for #105)

### Feature Extraction Tests
- `test_extract_negation_feature()`
- `test_extract_tense_feature()`
- `test_extract_mood_feature()`

### Embedding Tests
- `test_embed_slot()`
- `test_embed_missing_slot()`
- `test_average_multiple_roots()`

### Indexing Tests
- `test_index_sentence()`
- `test_index_batch()`
- `test_checkpoint_resume()`
- `test_failed_parse_handling()`

### Output Format Tests
- `test_output_jsonl_format()`
- `test_slot_embedding_shape()`
- `test_feature_values()`

## Acceptance Criteria

- [ ] Coverage: 10% → 85%
- [ ] All slot types tested
- [ ] Feature extraction tested
- [ ] Error handling tested
- [ ] Output format validated

## Estimated Effort

~4-5 hours
