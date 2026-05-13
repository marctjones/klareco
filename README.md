# Klareco - Pure Esperanto AI

**Answer Esperanto questions with maximally deterministic processing.**

Klareco uses Esperanto's regular grammar to replace much of what a traditional
LLM has to learn:

- **Deterministic**: 16-rule parser → role-annotated AST, deparser, morphology,
  proper-noun classification, schema-first ontology over a Kuzu graph
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
Question → ParseQuestion → Retrieve (Whoosh + Kuzu AST roles)
        → DeterministicRerank (question-type AST boost)
        → Rerank (stub) → ExtractAndGenerate → FormatOutput
```

Active work is tracked under
[EPIC #713 — Improve QA accuracy through iterative AST-first improvements](https://github.com/marctjones/klareco/issues/713).
Measurement target is retrieval-rank metrics (top-1 / top-5 / top-20 / MRR)
plus extraction accuracy conditional on retrieval — not final-answer accuracy
alone.

| Component | Status |
|-----------|--------|
| 16-rule parser + deparser | ✅ 91.8% parse rate on 4.2M sentences |
| Proper-noun dictionary v3 | ✅ cleaned + Wikipedia-category enriched (~628K entries) |
| Kuzu v2.1 graph + 4-layer ontology | ✅ production index |
| Orchestrator pipeline | ✅ active spine; immutable context, phase-level timing |
| WhooshRetriever + AST-role matching | ✅ deterministic two-stage retrieval |
| DeterministicRerankStage | ✅ question-type AST boost (WHO/WHERE/WHEN/HOW_MANY) |
| ExtractiveAnswerGenerator | ✅ slot-keyed extraction; top-20 cap |
| Eval (local + Modal cloud) | ✅ `klareco.eval` shared by both runners |
| Neural reranker | 🔲 deferred — stage is a stub |
| Learned root embeddings / M1 / M2 / M3 | 🔲 deferred until deterministic floor measured |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A pre-built Kuzu v2.1 graph and Whoosh index in `data/indexes/` is required
for retrieval. These are not in git (each is multi-GB). Build locally with
the pipeline scripts (see `scripts/pipeline.sh`) or sync from a known good
snapshot.

## Usage

```bash
# Parse a sentence
python -m klareco parse "Mi amas la hundon."

# Answer a question end-to-end
python -m klareco run "Kiu fondis Esperanton?"

# Run extractive-QA evaluation on a test set
python scripts/evaluate_extractive_qa.py \
    --test-set data/test_sets/qa_test_diverse_30.jsonl

# Compare two eval result files (regression check)
python scripts/compare_eval_results.py before.json after.json
```

Modal cloud evaluation (parallel workers) lives in `scripts/modal_eval.py`;
push the index volume with `scripts/modal_upload_indexes.sh` first.

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
| `AGENTS.md` | Repository guidelines, IdlerGear usage |
| `16RULES.MD` | Esperanto grammar specification (reference) |
| `docs/VERSION_COMPATIBILITY.md` | Deferred v3.0 model-retraining plan (for when training resumes) |

GitHub issues are the source of truth for in-flight work — see EPIC #713 and
the project board.

## License

Source code only. Corpora, indexes, and trained checkpoints live under `data/`
and `models/` and are not tracked.
