# Klareco - Pure Esperanto AI

**Answer Esperanto questions with maximally deterministic processing.**

Klareco uses Esperanto's regular grammar to replace much of what a traditional
LLM has to learn:

- **Deterministic**: 16-rule parser → role-annotated AST, deparser, morphology,
  proper-noun classification, schema-first ontology over a DuckDB v2.2 store
- **Pipeline**: immutable AST flows through orchestrator stages; retrieval and
  extraction match on AST structure, not surface text
- **Learned models are deferred** until the deterministic floor is stable and
  we can measure where a small learned component actually moves a number

For the long-term thesis see `VISION.md`. For the active architecture see
`DESIGN.md`. For development conventions see `CLAUDE.md`.

## Current state

**Working today**: end-to-end extractive QA over a 5.4M-sentence Esperanto
corpus via the orchestrator pipeline.

```
Question → ParseQuestion → Retrieve (Whoosh BM25 + AST roles from DuckDB)
        → DeterministicRerank (question-type AST boost)
        → Rerank (stub) → ExtractAndGenerate → FormatOutput
```

Active work is tracked under
[EPIC #713 — Improve QA accuracy through iterative AST-first improvements](https://github.com/marctjones/klareco/issues/713).
Measurement target is retrieval-rank metrics (top-1 / top-5 / top-20 / MRR)
plus extraction accuracy conditional on retrieval — not final-answer accuracy
alone.

> ⚠️ **Read `DESIGN.md` → "Current state" before trusting any number here.**
> Several data artifacts were lost in a June 2026 laptop migration and the
> pipeline degrades *silently* without them. Recovery is tracked in
> [milestone #14](https://github.com/marctjones/klareco/milestone/14).

| Component | Status |
|-----------|--------|
| 16-rule parser + deparser | ✅ UD-Prago: 80.3% POS strict, 93.3% scheme-adjusted |
| DuckDB store + shredded AST columns | ✅ 5.39M sentences, `ast_json` blob (Kuzu retired 2026-05) |
| Whoosh BM25 index | ✅ live |
| Orchestrator pipeline | ✅ active spine; immutable context, phase-level timing |
| DuckDBRetriever + AST-role matching | ✅ **the** retriever — what `factory.py` builds |
| ExtractiveAnswerGenerator | ✅ slot-keyed extraction |
| Eval (local + Modal cloud) | ✅ `klareco.eval` shared by both runners |
| Proper-noun dictionary | ⚠️ **JSON missing** — parser falls back to capitalization heuristics |
| 4-layer semantic ontology | ⚠️ **not loaded** — `ontology_nodes`/`edges` empty, `verb_klaso` 0% populated |
| `entity_facts` table | ⚠️ **missing** — crashes `BiographyFormatStage` |
| AST rerankers (9 variants) | ⚠️ **all tied** — they differ only in components that read dead columns |
| `WhooshRetriever` | ❌ **dead** — `__init__` raises `NotImplementedError` |
| Symbolic layer (inference, planner, math, dialog, generation) | ❓ **never benchmarked** — in the pipeline, effect unknown |
| Q&A quality framework (R1-R15 + 8 gates) | ✅ `docs/QA_TEST_SET_QUALITY_STANDARD.md` |
| Neural reranker | 🔲 deferred — stage is a stub |
| Learned root embeddings / M1 / M2 / M3 | 🔲 deferred until deterministic floor measured |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Retrieval requires `data/indexes/duckdb_store.db` (DuckDB v2.2 with shredded
AST columns + `ast_json` blob) and `data/indexes/whoosh_v2/` (BM25). Both
are multi-GB and not in git. Build locally by running the per-stage scripts
under `scripts/acquire/`, `scripts/clean/`, `scripts/extract/`, `scripts/parse/`,
`scripts/index/` in order, or sync from a known good snapshot.

If only the Whoosh index is stale or corrupted (e.g. after an aborted
optimize, contaminated `--resume`, or schema change), rebuild it alone
from the trusted DuckDB store without re-parsing the corpus:

```bash
python scripts/index/rebuild_whoosh_from_duckdb.py  # ~15-25 min
```

The script wipes `data/indexes/whoosh_v2/`, streams `(sid, text)` from
DuckDB, skips the segment-merge optimize step, and runs hard correctness
gates (`doc_count`, segment count, round-trip sample) before exiting.

## Usage

```bash
# Parse a sentence
python -m klareco parse "Mi amas la hundon."
```

```python
# Answer a question end-to-end (no `klareco run` subcommand — use the factory)
from klareco.orchestrator.factory import build_default_pipeline
pipeline = build_default_pipeline(whoosh_index_dir='data/indexes/whoosh_v2')
print(pipeline.answer("Kiu fondis Esperanton?").text)
```

```bash
# Run extractive-QA evaluation on a test set
python scripts/eval/evaluate_extractive_qa.py \
    --test-set data/test_sets/qa_test_diverse_30.jsonl

# Compare two eval result files (regression check)
python scripts/eval/compare_eval_results.py before.json after.json
```

Modal cloud evaluation (parallel workers) lives in `scripts/eval/modal_eval.py`;
push the index volume with `scripts/eval/modal_upload_indexes.sh` first.

## Tests

```bash
python -m pytest
python -m pytest tests/test_parser.py -v        # parser only
python -m pytest tests/test_orchestrator.py -v  # pipeline contract
python -m pytest --cov=klareco                  # with coverage
```

## Documentation

| File | Purpose |
|------|---------|
| `VISION.md` | The long-term thesis: decomposable contributions, attribution |
| `DESIGN.md` | The active architecture — orchestrator stages, schema-first foundation |
| `CLAUDE.md` | Development conventions; schema-first rules that prevent hardcoded lists |
| `AGENTS.md` | Repository guidelines |
| `16RULES.MD` | Esperanto grammar specification (reference) |
| `docs/VERSION_COMPATIBILITY.md` | Deferred v3.0 model-retraining plan (for when training resumes) |
| `docs/QA_TEST_SET_QUALITY_STANDARD.md` | 15 rules (R1-R15) + 8 gate stages a Q&A test set must pass before it is fit to evaluate the system |
| `docs/QA_TEST_SET_BUILD_PLAN.md` | Phase A harvest → Phase B generate → Phase C freeze for `capability_100` and `trivia_real_50` |

GitHub issues are the source of truth for in-flight work — see EPIC #713 and
the project board.

## License

Source code only. Corpora, indexes, and trained checkpoints live under `data/`
and `models/` and are not tracked.
