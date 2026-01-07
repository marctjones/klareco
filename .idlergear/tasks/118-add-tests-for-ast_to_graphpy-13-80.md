---
id: 118
title: "Add tests for ast_to_graph.py (13% \u2192 80%)"
state: open
created: '2026-01-07T00:12:05.387060Z'
labels:
- testing
- coverage
priority: low
---
## Goal

Add tests for AST to PyTorch Geometric graph conversion.

## Current State

- Coverage: 13% (221 lines missing)
- Used for GNN-based reasoning

## Test Categories

### Conversion Tests
- `test_convert_simple_sentence()`
- `test_convert_complex_sentence()`
- `test_convert_question()`
- `test_node_features()`
- `test_edge_types()`

### Node Type Tests
- `test_word_node()`
- `test_phrase_node()`
- `test_sentence_node()`

### Edge Type Tests
- `test_subject_edge()`
- `test_object_edge()`
- `test_modifier_edge()`

### Batch Tests
- `test_batch_conversion()`
- `test_variable_length_graphs()`

## Acceptance Criteria

- [ ] Coverage: 13% → 80%
- [ ] All node types tested
- [ ] All edge types tested
- [ ] Batch processing tested

## Estimated Effort

~4-5 hours
