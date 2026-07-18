# Klareco CLI — command surface & release target

This is the **well-defined target** for Klareco's command-line interface: every
major command a third party needs to set up the environment, prepare the data,
run the AI system (the orchestration engine), and decode what it produced.

It is written as a *contract*, so status is marked honestly:

- ✅ **exists** — implemented today, in `klareco/cli.py` (run via `python -m klareco …`)
- 🔶 **partial** — exists but wrong defaults / thin / not yet the target shape
- 🎯 **target** — the interface we intend to expose; today it is a `scripts/*.sh`
  or a Python entry point, not a first-class subcommand

The design mirrors the architecture (see `VISION.md`, `DESIGN.md` →
"The orchestration contract"): the **orchestrator** is the core, the **AST-thought**
is the universal object, the **decoder** renders it, and data flows through a
fixed **acquire → clean → extract → parse → index** pipeline.

---

## 0. Install & environment

| Status | Command | Purpose |
|---|---|---|
| 🎯 | `pip install klareco` → `klareco …` | Installed console entry point. **Gap today: no `console_scripts`; you must run `python -m klareco`.** |
| ✅ | `python -m klareco <cmd>` | Current invocation for every command below. |
| ✅ | `python -m klareco.preflight [--allow-degraded]` | Verify the runtime: required artifacts exist, cohere, and are usable; **raises loudly** if not (a silently-degrading dependency is a bug). |
| 🎯 | `klareco doctor` | First-class alias for preflight + environment report (Python/venv, store row counts, index freshness, ontology status). Today: `python -m klareco.preflight` + `python -m klareco info`. |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m klareco.preflight          # is this machine set up to run anything?
```

Environment variables (contract):
- `KLARECO_ALLOW_DEGRADED=1` — run despite missing DEGRADING artifacts (loud, itemized, deliberate).
- `KLARECO_QE_WEIGHT` — query-variant expansion weight (default 0.3; 0 disables).
- `AI_MODELS_DIR` — shared HuggingFace model pool (only if/when learned modules load a public base model).

---

## 1. Data lifecycle — acquire → clean → extract → parse → index → validate

The pipeline that turns raw sources into the store + index the orchestrator
reads. **Today these are `scripts/<stage>/*.sh`; the target is a single
`klareco data` command group.** All stages are restartable and log to `logs/`.

| Status | Target command | Today | Purpose |
|---|---|---|---|
| 🎯🔶 | `klareco data acquire [--source tier0\|wikipedia\|gutenberg]` | `./scripts/acquire/acquire_all_tier0.sh` | Download authoritative Esperanto sources. |
| 🎯🔶 | `klareco data clean` | `./scripts/clean/clean_all.sh` | Normalize raw → cleaned text. |
| 🎯🔶 | `klareco data extract` | `./scripts/extract/extract_all.sh` | Cleaned text → sentence JSONL with provenance. |
| 🎯🔶 | `klareco data parse` | `./scripts/parse/parse_corpus.sh` | Parse to ASTs → unified corpus (~15 min for 5.4M; wall-clock is the 20 GB JSONL write, not the parse). |
| 🎯🔶 | `klareco data index build-store` | `python scripts/index/build_duckdb_store.py` | Corpus → DuckDB `sentences` (ast_json blob + shredded columns + provenance). |
| 🎯🔶 | `klareco data index build-search` | `python scripts/index/rebuild_whoosh_from_duckdb.py` | Build the Whoosh BM25 index FROM the store. |
| 🎯🔶 | `klareco data index load-ontology` | (from the literals in `extend_kuzu_schema_semantic_ontology.py`) | Populate `ontology_nodes`/`edges`. ⚠️ Do NOT re-run the old `load_ontology.py` (destroys the hand-fixed schema — see `28ce022`). |
| 🎯🔶 | `klareco data validate` | `python scripts/index/validate_duckdb_store.py` · `./scripts/validate/validate_all.sh` | Integrity checks: row counts, no garbage rows, columns carry information (not constant/NULL). |
| 🎯🔶 | `klareco data rebuild` | `./scripts/pipeline/rebuild_all.sh` | Orchestrate parse → store → index → validate in one pass. |

> ⚠️ Long-running stages are run as shell wrappers in a separate terminal, not
> by the assistant, and never against the default branch without a preflight
> disk check (`scripts/util/preflight_disk.sh`).

---

## 2. Run the AI system — the orchestration engine

The core. Every command here builds the default pipeline
(`klareco.orchestrator.factory.build_default_pipeline`) and threads an immutable
AST-thought through the mandatory stages (parse → retrieve → rerank → extract →
format), plus any enabled optional modules.

| Status | Command | Purpose |
|---|---|---|
| 🔶 | `python -m klareco query "<demando>"` | Answer an Esperanto question with citations. **Gap: defaults still point at retired `data/indexes/whoosh` + a Kuzu path; pass `--whoosh-dir data/indexes/whoosh_v2` (fix tracked below).** |
| ✅ | `python -m klareco explain "<demando>"` | Answer **and print the decoded thought at every stage** — the universal decoder (#882). The primary tool for seeing what the system "thought". |
| ✅ | `python -m klareco parse "<frazo>" [--format json]` | Parse one sentence to a role-annotated AST (no retrieval). |
| ✅ | `python -m klareco translate "<text>" [--to eo]` | Deterministic translation to/from Esperanto. |
| 🎯 | `klareco serve [--port N]` | Long-lived server: build the pipeline once, answer many questions (avoids per-call startup). |

```bash
python -m klareco explain "Kiu fondis Esperanton?" --whoosh-dir data/indexes/whoosh_v2
```

**Optional modules** are opt-in factory flags and run **default-OFF** until they
pass the contract suite and carry a number (`DESIGN.md` → orchestration
contract). Target: expose as `--enable-<module>`:

| Module | Flag | Status |
|---|---|---|
| Math tool | `enable_math_tool` | ✅ on — live, 5/5 smoke |
| Dialog (multi-turn) | `enable_dialog` | off — MVP-2 (#890/#891) |
| Planner | `enable_planner` | off — silently broken vs live schema (#881) |
| Biography/definition generators | `enable_biography` | off — same (#881) |

---

## 3. Inspect & decode the thought

| Status | Command | Purpose |
|---|---|---|
| ✅ | `python -m klareco explain "<demando>"` | Decode the full per-stage thought evolution + final thought, tagged `[regulo]`/`[modelo]`. |
| ✅ | `python -m klareco explain … --final-only` | Just the final decoded thought. |
| ✅ | `python -m klareco info` | System info: versions, store/index presence, capability status. |
| 🎯 | `klareco inspect ast --sid <N>` | Print a stored sentence's AST (read `ast_json` from the store; never re-parse). |
| 🎯 | `klareco inspect store` | Row counts, schema, ontology edge counts by relation, index freshness (the "executable status", #887). |

The decoder is also the **test oracle**: if a thought cannot be decoded, it does
not merge (contract rule 4).

---

## 4. Evaluate & develop

| Status | Command | Purpose |
|---|---|---|
| ✅ | `pytest -m contract` | **Primary suite** — the orchestrator holds every stage to the contract, on a tiny in-memory store (no production indexes, <1s). |
| ✅ | `python scripts/eval/evaluate_extractive_qa.py --test-set <jsonl>` | Answer-accuracy / retrieval metrics on a gold set. |
| ✅ | `python scripts/eval/multi_reranker_bench.py --test-set <jsonl>` | A/B rerankers with paired-bootstrap CIs. |
| ✅ | `python scripts/eval/debug_retrieval.py "<demando>"` | Interactive retrieval debugger. |
| 🎯 | `klareco eval --test-set <jsonl>` | First-class wrapper over the evaluators, writing to `data/perf/bench_history.jsonl` (the merge-gate ledger). |

---

## Release checklist — what "third-party ready" requires

Concrete gaps between today and a shippable CLI (each should be an issue):

1. **Console entry point** — add `console_scripts` so `pip install` yields a
   `klareco` command; today only `python -m klareco` works.
2. **Fix `query` defaults** — it still defaults to `data/indexes/whoosh` and a
   retired Kuzu path; should be `whoosh_v2` with no Kuzu. (`explain` is already
   correct.)
3. **`klareco data` command group** — promote the `scripts/*.sh` data pipeline
   to first-class subcommands (§1) so setup does not require reading shell.
4. **`klareco doctor` / `inspect store`** — surface preflight + live status as
   commands (§0, §3), backed by the executable-status work (#887).
5. **Prune dead `scripts/index/*_kuzu_*.sh`** — Kuzu is retired; these mislead a
   new user (companion to the code removal in `chore/remove-dead-kuzu-code`).
6. **Stable, documented exit codes + `--json` output** on every command, so the
   CLI is scriptable by third parties.

This file is the target. Keep it honest: when a 🎯 or 🔶 ships as ✅, update the
row in the same PR.

See also: `README.md` (quickstart), `DESIGN.md` (architecture + contract),
`CLAUDE.md` (conventions), `docs/VERSION_COMPATIBILITY.md` (script versioning).
