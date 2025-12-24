# Klareco Task Board (TDD-first)

## ✅ Completed (Nov 2025)
- ✅ Built canonical slot signatures and grammar-driven tokenizer with unit tests (`canonicalizer.py` + `test_canonicalizer.py`)
- ✅ Rewrote retriever/indexer with two-stage hybrid retrieval: structural filtering + neural reranking (`structural_index.py`, updated `retriever.py`)
- ✅ Replaced placeholder orchestrator with AST-first routing + extractive experts (`orchestrator.py`, `experts/extractive.py`, `experts/summarizer.py`)
- ✅ Full test coverage for all new components (11 tests passing)

## Near-term stability
- ✅ Indexing scripts now resumable with checkpoints and line-buffered logs (`scripts/index_corpus.py`, `scripts/build_corpus_with_sources.py`)
- ⚠️ Add tests for resume behavior on small fixtures (partial: `test_index_corpus_resume.py` exists)
- Guard translation/index/model availability with clear errors; keep parser/corpus flows usable offline
- Update requirements/docs to reflect optional deps (`torch-geometric`, `faiss-cpu`) and skip tests when absent

## Retrieval & generation
- ✅ Implemented extractive/template answerer using retrieved ASTs (`experts/extractive.py`, `experts/summarizer.py`)
- 🔜 Optional small AST-aware seq2seq hook with dataset builder (scripts exist: `scripts/train_graph2seq.py`, `scripts/create_synthesis_dataset.py`)
- ⚠️ Structural metadata now stored in index (in-memory filtering, not yet SQLite/JSONL cache)
- ✅ Demo index built: `data/corpus_index_v2/` (49K sentences with structural metadata)

## Testing & coverage
- ✅ Added comprehensive tests for new modules: canonicalizer, structural_index, structural_retrieval, extractive responder, orchestrator
- 🔜 Expand integration tests for full pipeline (non-neural path)
- 🔜 Add tests for corpus CLI, resumable scripts
- Ensure logging/tracing are exercised in tests for debuggability

## Multi-step & learning loop
- Reconnect orchestrator outputs to `execution_loop`/`blueprint` once the new responder is in place.
- Keep trace analysis working with the new pipeline structure.
