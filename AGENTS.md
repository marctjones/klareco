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

## IdlerGear

This project uses [IdlerGear](https://github.com/marctjones/idlergear) for knowledge management.

### CRITICAL: Session Start

**ALWAYS run this command at the start of EVERY session:**

```bash
idlergear context
```

This shows the project vision, current plan, open tasks, and recent notes. Do NOT skip this step.

### FORBIDDEN: File-Based Knowledge

**DO NOT create any of these files:**
- `TODO.md`, `TODO.txt`, `TASKS.md`
- `NOTES.md`, `SESSION_*.md`, `SCRATCH.md`
- `FEATURE_IDEAS.md`, `RESEARCH.md`, `BACKLOG.md`
- Any markdown file for tracking work or capturing thoughts

**ALWAYS use IdlerGear commands instead.**

### FORBIDDEN: Inline TODOs

**DO NOT write inline TODO comments:**
- `// TODO: ...`
- `# FIXME: ...`
- `/* HACK: ... */`

**INSTEAD:** Create a task with `idlergear task create "..." --label technical-debt`

### REQUIRED: Use IdlerGear Commands

| Instead of... | Use this command |
|---------------|------------------|
| Creating TODO.md | `idlergear task create "description"` |
| Writing notes to files | `idlergear note create "content"` |
| Adding TODO comments | `idlergear task create "..." --label technical-debt` |
| Creating VISION.md | `idlergear vision edit` |
| Documenting findings | `idlergear reference add "title" --body "..."` |

### During Development

| Action | Command |
|--------|---------|
| Found a bug | `idlergear task create "..." --label bug` |
| Had an idea | `idlergear note create "..."` |
| Research question | `idlergear explore create "..."` |
| Completed work | `idlergear task close <id>` |
| Check project goals | `idlergear vision show` |
| View open tasks | `idlergear task list` |

### Knowledge Promotion Flow

```
note → explore → task
```
- Quick thoughts: `idlergear note create "..."` (capture now, review later)
- Research threads: `idlergear explore create "..."` (open questions)
- Actionable work: `idlergear task create "..."` (clear completion criteria)
- Promote notes: `idlergear note promote <id>` (convert to task/explore)

### Reference Documentation

- `idlergear reference list` - View reference documents
- `idlergear reference show "title"` - Read a specific reference
- `idlergear reference add "title" --body "..."` - Add documentation
- `idlergear search "query"` - Search across all knowledge types

### Protected Files

**DO NOT modify directly:**
- `.idlergear/` - Data files (use CLI commands)
- `.mcp.json` - MCP configuration

The IdlerGear MCP server is configured in `.mcp.json` and provides these tools directly.
