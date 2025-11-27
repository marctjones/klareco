# Klareco: Esperanto-First AST Pipeline

Klareco leans on Esperanto’s regular grammar to minimize what must be learned. The AST is the contract between components: parsing, retrieval, routing, and generation all operate on structured, role-annotated trees.

## Why Esperanto shrinks the model
- **Deterministic roles**: Endings encode case/tense/mood; subject/object/verb detection is programmatic, so attention layers do not need to infer roles.
- **Small lexicon, compositional roots**: Morpheme-level embeddings are shared widely; embedding tables and feedforward projections can be smaller.
- **Grammar-driven tokenization**: Tokens are prefix + root + suffix + ending. This removes tokenizer drift, improves semantic locality, and reduces sequence length.
- **Structured signatures**: Slot-based canonical forms (SUBJ/VERB/OBJ + modifiers) allow lexical/structural filtering before any neural step, shrinking the search space and reducing the need for large rerankers.
- **Tensor simplification**: With fewer unique tokens and shorter sequences, we can use smaller embedding dims, fewer transformer layers (or none for retrieval), and tiny decoders for summarization. Many tasks stay symbolic.

## Target architecture
- **Front door**: `front_door.py` performs language ID and optional MarianMT translation; always falls back to the original text to stay deterministic.
- **Parser/deparser**: `parser.py` implements the 16 rules, producing morpheme-aware ASTs with roles/case/tense; `deparser.py` reconstructs text for echo/extractive answers.
- **Canonicalizer/tokenizer**: New module to emit slot signatures and grammar-driven tokens (prefix/root/suffix/ending) for indexing and generation.
- **Retrieval**: Two-stage search. Stage 1 filters by structural signatures (roots/roles/modifiers) backed by SQLite/JSONL metadata. Stage 2 optionally reranks a small candidate set with a lightweight encoder (Tree-LSTM or shallow transformer) and FAISS.
- **Generation**: Default extractive/template answerer from AST contexts; optional small AST-aware seq2seq for abstraction.
- **Routing/traces**: `gating_network.py` routes intents; `trace.py`/`logging_config.py` capture every step.
- **Corpus tooling**: `corpus_manager.py`, `klareco/cli/corpus.py`, `scripts/build_corpus_with_sources.py`, `scripts/index_corpus.py` manage texts, cleaning, and indexing with resumable checkpoints.

## LLM components we shrink or avoid
- **Tokenizer/embedding tables**: grammar tokens (prefix/root/suffix/ending) reduce vocab size; embeddings can be small (e.g., 128d) without loss of coverage.
- **Positional encoding + sequence length**: slot signatures shorten sequences; fewer positions → smaller positional tensors and lighter attention if used.
- **Attention layers**: role/case are explicit; we can use shallow attention or none for retrieval, relying on structure instead of learned dependency detection.
- **Output projection**: extractive/template responses avoid large vocab softmax; optional seq2seq can share the small morpheme vocab.
- **Rerankers**: Stage-1 structural filtering shrinks candidate sets so rerankers can be tiny (one or two layers) or skipped entirely.

## TDD and resilience
- All new modules land with unit + integration tests; coverage target >90% for tokenizer/indexer/retriever/pipeline paths.
- Long-running scripts (cleaning, indexing, training) must be interruptible and resume from checkpoints; log progress to stdout and to rotating files for live inspection.
- Dev environment uses Python 3.13, `python -m venv .venv`, `pip install -r requirements.txt` (no Conda).

## Implementation plan
1) **Canonical signatures and tokenizer**
   - Add slot-based canonicalization (SUBJ/VERB/OBJ + modifiers, tense, case) and deterministic morpheme tokens.
   - Tests: token determinism, signature stability, deparse roundtrips.
2) **Indexer + retriever rewrite**
   - Store structural keys and ASTs; filter by signatures; small reranker optional. Ensure resume support and progress checkpoints.
   - Tests: tiny corpus index build, structural filter correctness, semantic fallback.
3) **Extractive responder**
   - Fill templates from retrieved ASTs; deparse answers; optional small seq2seq hook.
   - Tests: answer selection, formatting, routing via gating network.
4) **Pipeline + scripts hardening**
   - Replace placeholder orchestrator; ensure logging/tracing; add integration tests over the non-neural path.
5) **Training/data**
   - Dataset builders for AST→text summarization/QA; checkpoints saved periodically; resume on restart.

## Status (high level)
- Solid: parser/deparser, AST→graph converter, corpus manager/CLI, symbolic gating, tracing/logging.
- Partial: retriever/indexer (pre-rewrite), orchestrator (placeholder), graph-based generator (untrained), docs for corpus/RAG (being updated).
- Missing: canonicalizer/tokenizer, structural retriever, extractive responder, resumable scripts, updated tests.
