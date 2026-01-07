---
id: 119
title: "Add tests for enriched_ast.py, semantic_pipeline.py, thought_decoder.py (18-34%\
  \ \u2192 80%)"
state: open
created: '2026-01-07T00:12:06.320016Z'
labels:
- testing
- coverage
priority: low
---
## Goal

Add tests for AST enrichment and semantic processing modules.

## Files to Cover

| File | Current | Target | Lines Missing |
|------|---------|--------|---------------|
| `enriched_ast.py` | 34% | 85% | 111 |
| `semantic_pipeline.py` | 23% | 80% | 145 |
| `thought_decoder.py` | 18% | 80% | 182 |

## Test Categories

### EnrichedAST Tests
- `test_enrich_with_embeddings()`
- `test_enrich_with_features()`
- `test_enrich_sentence()`
- `test_batch_enrichment()`

### Semantic Pipeline Tests
- `test_pipeline_stages()`
- `test_stage_composition()`
- `test_error_handling()`

### Thought Decoder Tests
- `test_decode_intent()`
- `test_decode_entities()`
- `test_decode_relations()`

## Acceptance Criteria

- [ ] enriched_ast.py at 85%+ coverage
- [ ] semantic_pipeline.py at 80%+ coverage
- [ ] thought_decoder.py at 80%+ coverage

## Estimated Effort

~6-8 hours
