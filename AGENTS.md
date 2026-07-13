# Repository Guidelines

## ⚠️ Read this first

**Klareco is a boundary-discovery project, not a feature project.** The goal is
to push classical deterministic computation as far as it honestly goes, find
where it breaks, characterize *why*, and spend machine learning only on that
characterized residue. See `VISION.md`.

Two consequences that govern everything below:

1. **No capability merges without a number that moved.** See *The merge gate*
   under Commit & PR Guidelines. This is not style advice — it is the rule the
   project was failing at for months.
2. **Before trusting any number in this repo, read the "Current state" section
   of `DESIGN.md`.** Several data artifacts were lost in a June 2026 migration
   and the pipeline degrades *silently* without them.

## Project Structure & Module Organization

Active code lives in `klareco/`:

- `parser.py` / `deparser.py` — 16-rule deterministic parser and the
  Esperanto-text reconstructor
- `proper_nouns.py` — proper-noun dictionary (⚠️ the v3 JSON is currently
  **missing**; the parser silently falls back to capitalization heuristics)
- `cli.py`, `__main__.py` — CLI entry points. Subcommands: `parse`, `query`,
  `translate`, `corpus`, `info`. **There is no `run` subcommand** — to answer a
  question end-to-end, use the factory (see Commands below).
- `orchestrator/` — immutable `QueryContext` pipeline. Stages in
  `orchestrator/stages/`: parse_question → [dialog, math_tool, planner] →
  retrieve → deterministic_rerank → ast_aware_rerank → rerank (**stub**) →
  extract_generate → biography_format → format_output
- `rag/` — `duckdb_retriever` (**the** retriever), `ast_aware_reranker`,
  `unified_extractor`, `extractive_answering`, `entity_fact_retriever`,
  `question_classifier`
  - ⚠️ `whoosh_retriever.py` is **dead** — its `__init__` raises
    `NotImplementedError`, yet it is still exported from `rag/__init__`.
    `scripts/eval/debug_retrieval.py` constructs it and therefore crashes.
- `reasoning/`, `planning/`, `generation/`, `dialog/`, `tools/` — the symbolic
  layer. **Wired into the default pipeline and never benchmarked** (see #785).
- `knowledge/` — vocabularies, ontology-backed by design. ⚠️ Some hardcoded
  fallbacks are currently live because the ontology is unpopulated. This is
  acknowledged debt, **not** a licence to add more.
- `ontology/` — ⚠️ **dead**; `semantic_query.py` still takes a `kuzu_conn`.
- `eval/qa_metrics.py` — shared evaluator used by local + Modal runners

The store is **DuckDB**, not Kuzu. Kuzu was removed in the May–June 2026
migration: there are no `kuzu` imports left in `klareco/`, no Kuzu database on
disk, and `kuzu` is not even installed. Any doc, script, or comment telling you
to write Cypher is stale — fix it when you find it. The one exception is
`scripts/index/extend_kuzu_schema_semantic_ontology.py`, which looks like dead
Kuzu code but is **the only surviving source of the ontology's class
definitions** — do not delete it.

Data pipeline (corpus → DuckDB store + Whoosh index) and evaluators are in
`scripts/`. Tests in `tests/`. Docs in `docs/`. `data/` and `models/` are
local-only and untracked.

For the active design see `DESIGN.md`. For the thesis see `VISION.md`. For
schema-first conventions that govern new code see `CLAUDE.md`. For test-set
construction see `docs/QA_TEST_SET_QUALITY_STANDARD.md` — it is binding.

## Setup & Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Retrieval requires `data/indexes/duckdb_store.db` (~32 GB, 5.39M sentences,
AST carried as an `ast_json` blob) and `data/indexes/whoosh_v2` (BM25).

## Build, Test, and Development Commands

- **Parse**: `python -m klareco parse "mi amas la hundon"`
- **Answer a question end-to-end** (no `run` subcommand — use the factory):
  ```python
  from klareco.orchestrator.factory import build_default_pipeline
  pipeline = build_default_pipeline(whoosh_index_dir='data/indexes/whoosh_v2')
  print(pipeline.answer("Kiu fondis Esperanton?").text)
  ```
- **Eval (local)**: `python scripts/eval/evaluate_extractive_qa.py --test-set <set>.jsonl`
- **Reranker A/B**: `python scripts/eval/multi_reranker_bench.py --test-set <set>.jsonl`
- **Parser ruler**: `python scripts/eval/eval_ud_prago.py` — UD_Esperanto-Prago,
  131 gold sentences. The only benchmark in the repo that is independent of the
  Q&A stack, and the only one currently trustworthy.
- **Tests**: `python -m pytest`, or scope with `-k`. Tests needing the
  production indexes skip when absent.

## Coding Style & Naming Conventions

PEP 8 + type hints. snake_case functions/variables, PascalCase classes,
UPPER_SNAKE constants. Use `klareco.logging_config.setup_logging`, not `print`,
in library code. Tests follow `test_<module>.py`.

Comments explain *why*, not *what*. Write one only to state a constraint the
code cannot show.

**Schema-first rule** (`CLAUDE.md` has the full list): query the ontology for
entity types, verb classes, synonyms, importance weights, thematic roles. Do not
introduce gazetteers, regex tables, or hand-maintained synonym lists into
`klareco/`.

**Fail loudly.** Every artifact the pipeline loads must raise if it is absent.
A missing file that logs a warning and carries on is a bug — that pattern cost
this project a month of invisible quality loss. Never write
`if path.exists(): load()` with a silent `else`.

## Testing Guidelines

Keep tests deterministic and offline. Small sentence fixtures. Tests that need
production indexes should detect their absence and skip, not fail. Aim for ~80%
coverage on new modules; parser and orchestrator paths near 90%.

## Commit & Pull Request Guidelines

Conventional-style messages (`feat:`, `fix:`, `chore:`) with the *why* in the
body. Atomic commits.

### The merge gate

**No capability merges without a number that moved.**

A PR that adds or changes a capability must state in its description:

1. **Which number it moves** — a specific metric from the frozen benchmark
   (`retrieval_recall@k`, `extraction_exact_match | retrieved`,
   passage-selection accuracy, UD-Prago POS, latency).
2. **Before / after**, produced by the frozen benchmark and appended to
   `data/perf/bench_history.jsonl`.
3. **If the number did not move, the PR does not merge.** It becomes a
   research-track finding: *what did we learn about why the deterministic
   approach didn't help?* Under the boundary-discovery thesis that is a genuine
   result, not a failure.

Elegance, linguistic correctness, and "it obviously should help" are **not**
admissible evidence. The thesis is that we *discover* the boundary by
measurement instead of assuming it — so an unmeasured capability contributes
nothing to the thesis even when it works.

**Exempt** (narrow, and enumerated on purpose): infrastructure and cleanup with
no runtime surface, documentation, test-set construction itself, and research
spikes (which produce a decision, not a merged capability).

### Before you open a capability issue

A well-formed build issue names **(a)** the number it would move and **(b)** the
test that would show it. If you cannot answer both, it is not a build task — it
is a **research spike** (we don't yet know whether it's deterministic or whether
it helps) or it is **deferred** (blocked behind the benchmark). Both are fine.
Filing an open research question as an implementation task is how eight
capabilities got built and none got measured.

## Security & Data Handling

`data/` and `logs/` are local. Do not commit corpora, indexes, or checkpoints —
they are large and may include copyrighted material. Keep secrets out of code
and config.
