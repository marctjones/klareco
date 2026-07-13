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
Per-source row counts, no garbage, provenance intact. Today the store's 5,391,442
rows account for **every extracted sentence** exactly — the ETL is sound. What is
missing was never *extracted*: PAG, Proverbaro, Fundamento, ReVo (#810, #806).

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
