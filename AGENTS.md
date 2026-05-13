# Repository Guidelines

## Project Structure & Module Organization

Active code lives in `klareco/`:

- `parser.py` / `deparser.py` — 16-rule deterministic parser and the
  Esperanto-text reconstructor
- `proper_nouns.py` — v3 cleaned + Wikipedia-category proper-noun dictionary
- `canonicalizer.py` — slot signatures for indexing
- `cli.py`, `cli/`, `__main__.py` — CLI entry points (`python -m klareco …`)
- `orchestrator/` — immutable `QueryContext` pipeline. Stages live in
  `orchestrator/stages/` (parse_question, retrieve, deterministic_rerank,
  rerank, extract_generate, format_output)
- `rag/` — `whoosh_retriever`, `unified_extractor`, `extractive_answering`,
  `ast_semantic_ranker`, `kuzu_ast_reconstructor`, query expanders
- `knowledge/` — Kuzu-backed synonyms, gazetteers, temporal/spatial,
  morphology, semantic bridge. **No hardcoded lists** — see `CLAUDE.md`
- `eval/qa_metrics.py` — shared evaluator used by local + Modal runners
- `utils/kuzu_open.py` — single Kuzu opener honoring env-var memory caps

Pipeline-running scripts, the data pipeline (corpus → Whoosh + Kuzu), and
the proper-noun-cleaning pipeline are in `scripts/`. Tests are in `tests/`.
Docs in `docs/`. `data/` and `models/` are local-only and not tracked
(indexes, checkpoints, dictionaries, test sets).

For the active design see `DESIGN.md`. For the long-term thesis see
`VISION.md`. For schema-first conventions that govern new code see
`CLAUDE.md`.

## Setup & Environment

Use Python 3.13. Create a venv and install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A Kuzu v2.1 graph + Whoosh index at `data/indexes/` is required for
retrieval. `klareco.utils.kuzu_open` honors `KLARECO_KUZU_DB_PATH`,
`KLARECO_KUZU_BUFFER_MB`, `KLARECO_KUZU_MAX_THREADS` so out-of-tree
runners (Modal) and parallel workers can configure memory without code
changes.

## Build, Test, and Development Commands

- **Parse**: `python -m klareco parse "mi amas la hundon"`
- **Answer a question end-to-end**: `python -m klareco run "Kiu fondis Esperanton?"`
- **Eval (local)**: `python scripts/evaluate_extractive_qa.py --test-set data/test_sets/qa_test_diverse_30.jsonl`
- **Eval (Modal)**: `python scripts/modal_eval.py …` (after `scripts/modal_upload_indexes.sh` syncs the volume)
- **Parallel local bench**: `scripts/local_parallel_bench.sh` (sizes Kuzu memory per worker)
- **Compare runs**: `python scripts/compare_eval_results.py before.json after.json`
- **Tests**: `python -m pytest` or scope with `-k`. Most tests are
  deterministic; tests that need the production Kuzu/Whoosh indexes will
  skip when those aren't present.

## Coding Style & Naming Conventions

PEP 8 + type hints. snake_case for functions/variables, PascalCase for
classes, UPPER_SNAKE for constants. Use `klareco.logging_config.setup_logging`
instead of print in library code. Tests follow `test_<module>.py` and mirror
module boundaries. Inline comments explain *why*, not *what*; if a comment
documents a hidden constraint or a non-obvious tradeoff, write it — otherwise
let the code speak.

Schema-first rule (`CLAUDE.md` for the full list): query the Kuzu ontology
for entity types, verb classes, synonyms, importance weights, thematic
roles. Do not introduce gazetteers, regex tables, or hand-maintained
synonym lists into `klareco/`.

## Testing Guidelines

Keep tests deterministic and offline. Use small sentence fixtures. Tests
that require the production indexes should detect their absence and skip
rather than fail. Aim for ~80% coverage on new modules; parser and
orchestrator paths should stay near 90%. Use `-k` to scope expensive
tests before running the full suite.

## Commit & Pull Request Guidelines

Conventional-style messages (`feat:`, `fix:`, `chore:`) with the *why* in
the body. Keep commits atomic — one logical change each. PRs include a
short summary, linked issue/task, notes on any data/index updates, and a
"Tests" section listing the commands run. Include log snippets or eval
result diffs when touching the pipeline.

## Security & Data Handling

`data/` and `logs/` are local. Do not commit corpora, indexes, or
checkpoints — they're large and may include copyrighted material. Keep
secrets out of code and config.

## IdlerGear

This project uses [IdlerGear](https://github.com/marctjones/idlergear) for
knowledge management. The rules in `/home/marc/.claude/rules/idlergear.md`
and `.claude/rules/idlergear.md` are authoritative.

### Session start (required)

```bash
idlergear context
```

### Forbidden

- File-based knowledge: `TODO.md`, `NOTES.md`, `SESSION_*.md`, `SCRATCH.md`,
  `BACKLOG.md`, or any markdown file used to track work or capture thoughts
- Inline TODO/FIXME comments

### Use these commands instead

| Situation | Command |
|-----------|---------|
| Found a bug | `idlergear task create "..." --label bug` |
| Had an idea | `idlergear note create "..."` |
| Research question | `idlergear explore create "..."` |
| Completed work | `idlergear task close <id>` |
| Project goals | `idlergear vision show` |

### Protected paths

Do not modify `.idlergear/` or `.mcp.json` directly — use the CLI / MCP
tools.
