---
id: 111
title: 'Achieve 90% code coverage: Parent tracking issue'
state: open
created: '2026-01-07T00:12:00.346462Z'
labels:
- testing
- coverage
priority: medium
---
## Current State

**Overall coverage: 36%** (target: 90%)

## Coverage by Module (sorted by priority)

### Critical Path - 0% Coverage (must fix first)
| Module | Current | Target | Lines Missing |
|--------|---------|--------|---------------|
| `cli.py` | 0% | 80% | 176 lines |
| `__main__.py` | 0% | 80% | 3 lines |
| `qa/extractor.py` | 0% | 85% | 109 lines |
| `qa/reranker.py` | 0% | 85% | 152 lines |
| `embeddings/unknown_tracker.py` | 0% | 85% | 102 lines |
| `models/__init__.py` | 0% | 100% | 2 lines |

### Low Coverage - Under 20%
| Module | Current | Target | Lines Missing |
|--------|---------|--------|---------------|
| `rag/ast_aware_retriever.py` | 5% | 80% | 641 lines |
| `rag/slot_indexer.py` | 10% | 85% | 184 lines |
| `rag/slot_retriever_mmap.py` | 11% | 80% | 166 lines |
| `rag/slot_retriever_multifaiss.py` | 12% | 80% | 175 lines |
| `rag/slot_retriever_sqlite.py` | 12% | 80% | 157 lines |
| `ast_to_graph.py` | 13% | 80% | 221 lines |
| `rag/slot_retriever_faiss.py` | 14% | 80% | 148 lines |
| `rag/slot_retriever_hnsw.py` | 14% | 80% | 162 lines |
| `rag/slot_retriever_scann.py` | 14% | 80% | 150 lines |
| `rag/slot_retriever_hybrid.py` | 15% | 80% | 150 lines |
| `thought_decoder.py` | 18% | 80% | 182 lines |
| `semantic_pipeline.py` | 23% | 80% | 145 lines |

### Medium Coverage - 20-50%
| Module | Current | Target | Lines Missing |
|--------|---------|--------|---------------|
| `enriched_ast.py` | 34% | 85% | 111 lines |
| `embeddings/compositional.py` | 43% | 90% | 167 lines |
| `embeddings/linguistic_embeddings.py` | 48% | 90% | 33 lines |
| `embeddings/topical_embeddings.py` | 50% | 90% | 30 lines |

### Good Coverage - 50-80%
| Module | Current | Target | Lines Missing |
|--------|---------|--------|---------------|
| `proper_nouns.py` | 64% | 90% | 20 lines |
| `trace.py` | 67% | 90% | 13 lines |
| `deparser.py` | 69% | 90% | 23 lines |

### Already Good - 80%+
| Module | Current | Target |
|--------|---------|--------|
| `parser.py` | 86% | 90% |
| `rag/question_classifier.py` | 86% | 90% |
| `rag/entity_recognizer.py` | 86% | 90% |
| `canonicalizer.py` | 87% | 90% |
| `embeddings/hybrid_embeddings.py` | 87% | 90% |
| `embeddings/dual_root_embeddings.py` | 95% | 95% |
| `rag/semantic_db.py` | 100% | 100% |
| `logging_config.py` | 100% | 100% |

## Effort Estimate

To reach 90% overall coverage:
- ~2,500 lines of test code needed
- ~40-50 hours of work
- Priority: Focus on critical path and RAG modules first

## Sub-Issues

Coverage work will be tracked in separate issues per module group.
