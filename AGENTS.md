# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `klareco/` (parser/deparser, front door, trace/logging, CLI entry). RAG is in `klareco/rag/`, models in `klareco/models/`, corpus helpers in `klareco/corpus_manager.py` and `klareco/cli/corpus.py`, and multi-step scaffolding in `blueprint.py`/`execution_loop.py`. Scripts for cleaning, indexing, benchmarking, and training live in `scripts/`; tests are in `tests/`; docs are in `docs/`. `data/` holds local corpora/indexes and is not tracked; `models/` may contain checkpoints (e.g., Tree-LSTM).

## Setup & Environment
Use Python 3.13. Create a venv (`python -m venv .venv && source .venv/bin/activate`), then `pip install -r requirements.txt`. RAG/graph features also need `torch-geometric` and `faiss-cpu` installed manually plus a Tree-LSTM checkpoint under `models/`. MarianMT translation pulls models from Hugging Face on first use; if unavailable, code should fall back gracefully.

## Build, Test, and Development Commands
- **Parsing/translation:** `python -m klareco parse "mi amas la hundon"`; `python -m klareco translate "The dog sees the cat." --to eo` (requires Marian models).
- **Corpus:** `python -m klareco corpus validate|add|list|remove|stats` and scripts `scripts/build_corpus_with_sources.py` / `scripts/index_corpus.py` to build `data/corpus_index`.
- **Retrieval demo:** `python scripts/run_pipeline.py "Kio estas Esperanto?"` (uses retriever + placeholder generation; requires indexes/models).
- **Pipeline note:** `python -m klareco run ...` now uses a minimal orchestrator that returns placeholder answers; add richer experts before relying on it for real outputs.
- **Tests:** Use targeted runs first (`python -m pytest tests/test_parser.py -k basic`, `tests/test_gating_network.py -k classify`). RAG/generator tests expect `torch-geometric`, `faiss`, and local indexes/models.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants. Add type hints and concise docstrings to public interfaces; keep inline comments minimal and explanatory. Use `klareco.logging_config.setup_logging` instead of print in library code. Tests follow `test_<module>.py` naming and should mirror module boundaries.

## Testing Guidelines
Add pytest coverage for new behavior and keep tests deterministic (no network). Prefer small sentence fixtures; for RAG/corpus flows use temporary paths under `data/` and skip when indexes/models are absent. Use `-k` to scope expensive tests before running the full suite; aim for ~80% coverage on new modules.

## Commit & Pull Request Guidelines
Use conventional commit prefixes seen in history (`feat`, `fix`, `chore`, optional scope like `chore(debug): ...`) and keep changes atomic. PRs should include a short summary, linked issue/task, notes on data/model updates, and a “Tests” section listing commands run (e.g., `python -m pytest tests/test_parser.py -k basic`). Include relevant log snippets or traces when touching pipeline or corpus flows.

## Security & Data Handling
Do not commit corpora, checkpoints, or generated logs; `data/` and `logs/` are local working areas and may contain copyrighted or large assets. Use scripts to download/clean data locally and keep secrets out of code or config.
