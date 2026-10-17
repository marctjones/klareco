# The Test Taxonomy

> Six tiers. Each answers a different question and fails for a different reason.
> Tiers are applied **automatically** in `tests/conftest.py`, by filename — not by
> hand. The previous eight markers were declared in `pytest.ini` and used by
> **exactly zero tests**; a taxonomy nobody applies is a comment, not a taxonomy.

## The tiers

| Tier | Question it answers | Needs data? | Fails when |
|---|---|---|---|
| `unit` | Is the **code** correct? | ❌ | A rule, a parse, a score is wrong |
| `environment` | Is the **runtime** set up? | ✅ | An artifact is missing, empty, or incoherent |
| `data` | Did the data **load** correctly? | ✅ | Garbage, gaps, or lost provenance |
| `pipeline` | Do the **pieces** work end to end? | ✅ | A stage or the orchestrator is broken |
| `perf` | Is it **fast enough**? | partial | Latency regressed against the baseline |
| `accuracy` | Is it **good enough**? | ❌ (reads the record) | Quality regressed against the baseline |

```bash
pytest -m unit                  # the inner loop — fast, offline, no data
pytest -m environment           # "can I trust anything on this machine?"
pytest -m "data or pipeline"    # "is the system actually working?"
pytest -m "perf or accuracy"    # "did I make it worse?"   <- the merge gate
pytest                          # everything
```

Current sizes: **unit 353 · data 122 · pipeline 79 · environment 24 · accuracy 7 · perf 3**.

---

## Why each tier exists — and what it caught

### `unit` — is the code correct?
Fast, offline, no data. The inner loop. 210 parser tests alone, covering prefixes,
suffixes, correlatives, participles, compounds, elision, case/number, official
prefixes, non-Esperanto words.

### `environment` — is the runtime set up?
**This is the tier that did not exist**, and its absence is why the June migration
silently degraded output for a month.

`test_preflight.py` + `test_environment_contract.py` assert the **contract between
artifacts**: the store is coherent, its columns carry *information* (not merely
non-nulls), the Whoosh index **agrees with the store and is complete**, a real
sentence **retrieves itself by its own words**, and cross-table references resolve.

What it caught on first run: `success_rate` constant at 0.0 across all 5,391,442
rows (#805), `verb_klaso` 0% populated (#777), 121,939 redirect stubs indexed
(#802). None of it had ever been noticed.

> **Population is not the contract. Variance is.** A column can be 100% non-null
> and completely dead.

### `data` — did the data load correctly?
Per-source row counts, no garbage, provenance intact. At the time this ran, the
store's 5,391,442 rows accounted for **every extracted sentence** exactly — the
ETL was sound. The store has since been cleaned (2026-07-19, #16): 767,332
redirect-stub/markup/English rows removed, leaving 4,624,110 — a documented
cleanup, not a re-appearance of the original problem. What is missing was never
*extracted*: PAG, Proverbaro, Fundamento, ReVo (#810, #806).

⚠️ Several tests in this tier **skip** rather than fail (missing ReVo, dead
FAISS-era assertions). A skipping test masquerades as coverage. See #808.

### `pipeline` — do the pieces work?
Orchestrator, stages, question classifier, entity recognizer.

### `perf` — is it fast enough?
Two levels. **Micro** (parser latency, offline, runs every commit — the parser is
the hot path: every question, and 5.4M corpus sentences). **Stage** (per-stage
wall time from the recorded bench).

Budgets are **deliberately loose**. A perf test that fails on a noisy laptop gets
deleted within a week, and then you have no perf test at all. These catch
order-of-magnitude regressions — the ones that actually happen (#724: WHEN
questions at 109–164 s, found by a human noticing, not by a test).

### `accuracy` — is it good enough?
**This is the tier the merge gate stands on.**

> *"No capability merges without a number that moved."* (#784)

That requires (a) a recorded number and (b) something that notices when it moves
the **wrong** way. `data/perf/bench_history.jsonl` gave us (a) — it was being
*written* by the bench scripts and **read by nothing**. A baseline nobody asserts
against is a diary, not a gate.

This tier does **not** re-run the pipeline (minutes, needs the 32 GB store). It
asserts on the **record**:

```
1. make a change
2. run the bench      -> appends to bench_history.jsonl
3. pytest -m accuracy -> fails if you regressed
```

**It compares only runs on the SAME test set.** `recall_at_5 = 17` on a
17-question set and `= 41` on a 50-question set are numbers from two different
instruments; a "regression" between them is arithmetic, not a fact about the
system. (The first version of this test made exactly that mistake and reported a
48-point collapse that never happened — which is a decent argument for why the
tier needed to exist.)

---

## The honest caveat, stated loudly

`pytest -m accuracy` currently **skips its trust check** with:

> *BASELINE IS NOT TRUSTWORTHY: the latest bench ran on
> `synthetic_who_rebuild_17_cleanish`, which is SATURATED (BM25 already wins).*

58.8% of that set's pairs have the gold passage **already at BM25 rank 1** — it is
the set on which all nine rerankers tied. Any baseline drawn from it is not a
quality signal, and the tier says so rather than lending it false authority.

When the discriminating set lands (#778 + #783), re-baseline and these assertions
become real. Until then they still catch a **catastrophic** regression, which is
worth having.

---

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `dev` and
`master`, and on manual dispatch. Two jobs:

| Job | Command | Needs |
|---|---|---|
| `data-free-tiers` (Python 3.11 and 3.12) | `python -m pytest -m "contract or unit or accuracy" -q -p no:cacheprovider` | `requirements-ci.txt` plus the lexical artifacts below. No store. |
| `script-versions` | `scripts/util/validate_script_versions.py` over the `scripts/**/*.py` files **added or modified** in the push or PR | stdlib only |

What stays local: `environment`, `data`, `pipeline` and `perf` need
`data/indexes/` (44 GB) and never run in CI.

**Dependencies.** `requirements-ci.txt` is the minimal set found by running the
three tiers in a fresh virtualenv and adding a package only when a collected
test demanded it: pytest, duckdb, Whoosh, numpy, sympy (`klareco/__init__`
imports the factory, which imports the math-tool stage, which imports sympy at
module level), pandas (`scripts/index/reparse_store.py`, run as a subprocess by
the contract suite), and torch (CPU build: `klareco/rag/extractive_answering.py`
lazily imports `ast_semantic_ranker.py` for a pure-Python helper, and that
module imports torch at the top, so the golden traces need it even though no
model runs). Two small code changes would shrink this: a lazy torch import in
`ast_semantic_ranker.py`, and a lazy log handler in `build_duckdb_store.py`,
which opens `logs/build_duckdb_store.log` at import time. The workflow does
`mkdir -p logs` for the second one.

**Lexical artifacts, the real prerequisite.** The parser, morphology and
ontology modules refuse to import without seven files that `.gitignore`
excludes under `data/*`. In a clean checkout every test module that imports
the parser fails at collection with `FileNotFoundError: The parser has NO ROOT
LEXICON`. The workflow checks for them before anything else and fails with the
list:

| File | Size | Why it is needed |
|---|---:|---|
| `data/vocabularies/root_vocab.json` | 441 KB | `parser.py` refuses to import without it (#821) |
| `data/vocabularies/protected_roots.json` | 585 B | `morphology.py` reads it unconditionally |
| `data/vocabularies/capitalization_ratio.json` | 7.0 MB | optional in code; the proper-noun tests assume it |
| `data/raw/eo/dictionaries/revo_typed_roots.json` | 573 KB | typed lexicon; `morphology.py` raises without it |
| `data/raw/eo/dictionaries/revo_name_roots.json` | 96 KB | name roots; `morphology.py` |
| `data/raw/eo/dictionaries/affix_table.json` | 6 KB | selectional table; `morphology.py` raises without it |
| `data/raw/eo/dictionaries/revo_ontology.json` | 10.4 MB | `ontology.py` raises without it; the parser calls it per sentence |

`fundamento_roots.json` is already tracked. There are two ways to make CI
green, and the choice is a repository decision because the files derive from
GPL sources (ReVo is GPL-2.0, voko-akrido GPL-3.0) and from the corpus:

1. Un-ignore and commit the seven files (about 18.5 MB in total). This is the
   same reasoning that committed the annotation queues in #932: small enough,
   and exactly the artifact class the June 2026 migration lost.
2. Publish them as a `.tar.gz` with repository-relative paths and set the
   repository variable `KLARECO_LEXICON_URL` to its URL. The workflow downloads
   and extracts it before the preflight. From the repo root,
   `tar -czf klareco-lexicon.tar.gz <the seven paths>` produces a 5.5 MB file.

**Verified 2026-10-17** on a clean `git archive` checkout plus the seven
artifacts, `mkdir -p logs`, and a fresh Python 3.12 virtualenv holding exactly
`requirements-ci.txt`: **858 passed, 1 failed, 7 skipped, 1 xfailed in 12 s.**
The one failure is pre-existing on HEAD in every environment:
`tests/test_clause_tree.py::TestASTCompaction::test_the_blob_is_actually_SMALLER`
asserts the compact blob is under 45% of the expanded JSON and measures 47.9%
(30,225 B vs 63,068 B), which dates from the lossless storage v2 writer
(`docs/PARSER_SEVEN_STEP_RESULTS.md` records blobs growing 47.6%). The
`data-free-tiers` job is red until that threshold is re-baselined or compaction
is improved; that is a storage decision, not a CI one. The script-version
linter reports 32 invalid legacy scripts over the whole tree, which is why the
CI job lints only changed files, the same policy as `.githooks/pre-commit`.

## Adding a test

Put it in a file, and add the file to `_TIERS` in `tests/conftest.py`. Unlisted
files default to `unit` — the safe default (no data, asserts on code).

**A test that skips when its data is missing is fine. A test that skips
*forever* is not** — that is the silent-degradation pattern this project has been
climbing out of. If a test can never run, delete it (`test_kuzu_open.py`,
`test_index_integrity.py`) or fix the artifact.

## See also

- `DESIGN.md` → *The merge gate*, *The benchmark contract*
- `AGENTS.md` → *The merge gate*, *Fail loudly*
- `docs/QA_TEST_SET_QUALITY_STANDARD.md` → R1–R17, the construction rules
