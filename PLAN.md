# Klareco — build & implementation plan

The single authoritative plan for building this system: what it is, how it is
architected, and **in what order we build it, with the number that closes each
step.** This file owns *the sequence and status*. For depth, it points elsewhere:

- **Why** (the thesis) → `VISION.md`
- **Architecture as it actually is** (incl. the normative contract) → `DESIGN.md`
- **Working conventions** → `CLAUDE.md`
- **Command surface / release target** → `docs/CLI_ARCHITECTURE.md`
- **Live tracker** → [EPIC #897](https://github.com/marctjones/klareco/issues/897)

Status legend: ✅ done · 🔶 in progress · 🎯 planned. Keep it honest — flip a row
in the same PR that ships it. **Verify status claims against the live store; the
docs have been wrong before** (four claims falsified in one 2026-07-18 audit).

---

## 1. What we are building

An Esperanto question-answering system whose organizing bet is **boundary
discovery**: push classical, deterministic computation (rules, tables, graph
queries, search) as far as it honestly goes, *measure* the irreducible residue,
and add a learned component only there. The deliverable is a **map of where
deterministic methods stop** for a language with a perfectly regular grammar —
not a leaderboard score. Esperanto is the testbed because its regularity means a
failure is real evidence the problem is not rule-shaped.

Two consequences shape everything below:

1. **The orchestrator is the product.** Capabilities are optional modules that
   plug into a stable, enforced core — not features bolted onto a script.
2. **Nothing counts until a number moves.** A built capability with no benchmark
   contributes nothing to the thesis (the merge gate).

---

## 2. Architecture at a glance

```
text ──▶ Parser (16 rules) ──▶ AST ══▶  ORCHESTRATOR  ══▶ AST ──▶ Decoder ──▶ text
         └ deterministic            threads an immutable         └ deterministic
                                    dual-layer "thought"          (any stage)
```

- **The AST-thought is the universal contract.** The orchestrator passes an
  immutable `QueryContext` between stages: a **SymbolicLayer** (AST-expressible —
  question AST, passages, fact triples, segments, citations) and a **LatentLayer**
  (dense vectors with no clean AST encoding). Everything a module contributes
  lands in one of these; side channels are contract violations.
- **Mandatory spine (single-turn QA):** ParseQuestion → Retrieve (BM25 + AST
  roles) → DeterministicRerank → Extract → Format (cited answer).
- **Optional modules** (math, dialog, planner, generators, symbolic reasoning)
  are opt-in and run **default-OFF until they pass the contract suite and carry a
  number.** Math is the one currently live.
- **Every capability is a dual-track slot:** a required *deterministic*
  implementation plus an optional *learned* one, composed as `shadow` (measured,
  not shipped), `enrich` (fills only what's underdetermined), or `replace`
  (earned via the gate).
- **The universal thought decoder** renders any thought at any stage back to
  readable Esperanto, tagged `[regulo]`/`[modelo]`. Possible *because* the grammar
  is regular and the root base small; it is the observability tool **and** the
  test oracle (if it can't be decoded, it doesn't merge).
- **Data substrate:** DuckDB store (4,624,110 sentences: `ast_json` blob +
  shredded columns + provenance, plus `clauses` 6.95M and `dependency_arcs`
  68.9M) → Whoosh BM25 index → hand-seeded semantic ontology (237,739 edges).
  Built by a fixed pipeline: acquire → clean → extract → parse → index.
  ⚠️ **The parser emits three parallel views** — `vortoj` (the UD dependency
  tree, its own declared source of truth), `propozicioj` (clause list), and a
  legacy flat frame. The store shreds the *legacy* one, and production reads a
  lossy expansion of it. Correcting that is Phase 0.5 (#901).
- **Models are deferred by design.** The learned stack is pruned from HEAD; it
  re-enters only through the shadow harness, targeted at a *characterized*
  residue.

The normative contract (six rules: enrichments-land-in-the-thought, injected
resources, loud failure, decodability, attribution, dual-track slots) lives in
`DESIGN.md` → "The orchestration contract" and is **enforced by tests**, not
documented and hoped for.

---

## 3. Operating principles (how we build, always on)

- **The merge gate.** No capability merges without a moved benchmark number,
  appended to `data/perf/bench_history.jsonl`. If the number didn't move, it is a
  research-track finding — a real result under boundary-discovery, not a failure.
- **Contract enforcement.** A new capability *is* a stage that passes
  `pytest -m contract`. Capability code exists nowhere else. A contract that
  isn't tested is a naming convention.
- **Deterministic-first, then measure the residue.** Attempt every capability as
  rules/tables/search; characterize the break as a property of the problem;
  only then reach for a model, in shadow mode first.
- **Loud failure.** A silently-degrading dependency is a bug — declared deps,
  preflight raises, no swallowed exceptions. (This is how the whole symbolic
  layer was found silently dead: #881.)
- **Doc honesty.** Status is single-owned and, where possible, *executable*
  (generated from the live store, #887). Claims carry `(measured DATE, source)`.

---

## 4. The roadmap

Sequenced so the **stable core comes first** and optional modules are admitted
one at a time. Each phase closes on a number.

### Phase 0 — Core: enforce the thought  ·  milestone #28  ·  🔶 ~70%
*Goal: make the orchestrator the enforced core so nothing new can plug in silently.*
- ✅ Universal thought decoder + `klareco explain` (#882)
- ✅ Loud-failure preflight + no-swallow lint + failure stamping (#884)
- ✅ Optional modules default-OFF (#888)
- ✅ Golden traces (#886) · ✅ primary contract suite `pytest -m contract` (#883)
- ✅ CLI v0 — registry package + entry point (#898)
- 🔶 Resource injection — stages get a `StoreView`, no private connections (#885)
- 🎯 Executable status — generate the "Current state" table from the store (#887)
- **Exit:** every stage passes the contract suite (or a filed waiver); the
  decoder renders every stage on the golden traces. *(both true today for the
  default spine; extends to all stages once #895 lands)*

### Phase 0.5 — Substrate: the rich AST must reach retrieval  ·  milestone #33  ·  🎯 **NEW, now the critical path**
*Goal: connect three sound components that are not connected. Added 2026-07-20
after a full trace of the production path; epic **#901**.*

The 2026-07-14 rebuild succeeded — and a trace found its richest outputs are read
by nothing. `clauses` (6.95M rows) and `dependency_arcs` (68.9M) have **zero
production readers**; the whole default pipeline touches five columns
(`sentences(sid, subj_vortspeco, obj_radiko, text, ast_json)`) plus
`ontology_edges(ALIASO)`. The store is not under-informative — it is
**mis-projected**.

- 🎯 **#905 (P0)** — the retriever swallows schema errors on the hot path
  (`BinderException` → `_NO_ANSWER`, no `stage_failed` flag) and preflight
  validates columns the hot path never reads. Same class as #881, on the path
  that always runs.
- 🎯 **#902 (P0)** — `compact_ast`/`expand_ast` is lossy despite a docstring and
  a test named for an exact round-trip: `vortoj` survives byte-exact, every
  modifier list and nested clause is destroyed. **Decides what `ast_json` should
  hold**, so it precedes any rebuild.
- 🎯 **#903 → #904 (P0)** — the legacy flat frame *mixes clauses* and is what the
  store shreds. One design question: is the queryable unit the sentence or the
  clause? Measured cost of today's answer: **261,372 sentences** whose subject is
  invisible to the column production filters on; 2,263,488 clause subjects
  (40.9%) reachable only via `clauses`.
- 🎯 **#906 / #907 (P1)** — vocabulary reconciliation: the entity-type gate tests
  for `persono` against a column that only ever holds `propranomo` (**disjoint
  sets** — the gate has never fired), and `clauses.verb_klaso` holds POS tags,
  not semantic classes.
- 🎯 **#807** — the reparse, **folded into this phase rather than run ahead of
  it.** Built and rehearsed, but #871 moved *zero* UD accuracy metrics and
  touches ~4% of the corpus; running it into the current schema banks a small win
  and reproduces every misalignment above. One rebuild, carrying both.
- **Exit:** a stage reads clause-level structure through the injected
  `StoreView`, declaring `REQUIRES`; hot-path schema drift **raises**; and the
  change carries a paired-CI before/after. ⚠️ **Gated by milestone #23** — the
  discriminating stratum is what makes any of this provable (#736/#778).

### Phase 1 — MVP-1: single-turn QA, honest/loud/measured  ·  milestone #29  ·  🎯
*Goal: the smallest QA system that actually works, on the enforced contract.*
- 🎯 **#909 (P0)** — `entity_facts` has **two competing schemas** (`load_ontology.py`
  writes a triple form; `extract_entity_facts.py` targets a slot form) and two
  production readers were never converted — one swallows the resulting
  `BinderException` and returns `[]`. Supersedes the mechanism in #881; pick one
  schema, one writer, then rebuild and bench the revived path on its own number.
- ✅ **#895** — `ast_aware_rerank` demoted out of the pipeline (MRR 0.3619 →
  0.3446). Reviving it means fixing the substrate first (#906/#907/#904), not
  re-tuning the reranker.
- 🎯 **#869** — span extraction: `token_f1` is 0.014 (the worst number in the
  system) — return the span, not the passage.
- 🎯 **#896** — remove `random.choice` from the discourse planner (determinism).
- **Exit (#889):** `answer_accuracy` + `token_f1` re-baselined on
  `rebaseline_210`; **zero silent no-ops** (contract suite green over the full
  default pipeline).

### Phase 2 — MVP-2: multi-turn dialog on the thought  ·  milestone #30  ·  🎯
*Goal: admit the first optional module to the stable core.*
- 🎯 Nested-clause pronoun resolution (#890); type-hints from flowing facts,
  dialog state into the `QueryContext` (#891); first multi-turn gold set (#892).
- **Exit:** multi-turn resolution rate measured; `DialogStage` default-on **only
  if the number clears the gate.**

### Phase 3 — Dual-track slots: deterministic + learned shadow  ·  milestone #31  ·  🎯
*Goal: the mechanism by which learning enters — measured before it ships.*
- 🎯 `StageSlot(deterministic, learned=None, mode=shadow|enrich|replace)`; port
  rerank as the exemplar (#893); shadow harness → det-vs-learned report (#894).
- 🎯 First learned candidate nominated by the synonymy result (#873) or the
  learned-ranker research (#834).
- **Exit:** a det-vs-learned comparison generated from a live eval; the first
  learned component either clears the gate in `replace`/`enrich` or is parked
  research-track with its residue characterized.

### Phase 4+ — admit the rest, one at a time  ·  milestone #32  ·  🎯
Planner (#771), generation (#766/#775), symbolic reasoning (#747/#749/#761),
richer math — each **migrated onto the contract → gold set → merge gate**, and
each currently blocked by the `entity_facts` schema drift (#881, unblocked in
Phase 1). CLI hardening for third-party release (#898) lands here (the
deterministic core stabilizes first).

### Feeder tracks (support the phases, run in parallel)
- **Measurement ruler** — Gold Q&A v1/v2/v3 (#20–#23; epic #840) + Deep Band
  (#25) + Reranker v2 (#26). Without the ruler, no phase can close.
- **Data honesty** — Corpus & Index Integrity (#16; the one-pass rebuild #807,
  entity-fact extraction #745) · Test Coverage for non-orchestration code (#17).
- **Ontology** — restore-and-thin: wire `SINONIMO`, measure thinness, the
  `ALIASO` fold-in (#27, #872, #837).

---

## 5. Where we are (2026-07-20)

- **Foundation, not features.** Phase 0 is most of the way done; almost nothing
  "new" shipped recently by design — the effort proved the old capabilities were
  mostly dead/unmeasured and built the machinery so that can't recur.
- **The data problem is solved; a wiring problem replaced it.** The rebuild
  milestone closed 2026-07-20 (#835/#836/#837/#838), along with #802/#803/#818/
  #821/#823/#777. The corpus is honest: 0 redirect stubs, 0 markup, 0 English,
  provenance 100%, ontology loaded *and consumed* (the `ALIASO` bridge shipped
  the first net-positive deterministic live-path win, #872). **But a trace of
  the production path found the rebuild's richest outputs are read by nothing**
  — 76M rows across `clauses` and `dependency_arcs` with zero production
  readers. That is Phase 0.5 (#901).
- **Working & measured:** 16-rule parser, now regression-guarded against the UD
  gold treebanks (`pytest -m accuracy`, #900); DuckDB store + Whoosh; retrieval
  (recall@200 = 100% trivial/rerankable, 36% deep); the `rebaseline_500` ruler
  with paired-bootstrap CIs, which already converted a borderline result into a
  gate pass (#877); math tool live; contract suite + decoder + CLI v0.
- **The honest weak points:** end-to-end `token_f1 = 0.014` (returns passages,
  not spans, #869); the fact-consuming symbolic layer is dead against **two**
  competing `entity_facts` schemas (#909); the reranker's discriminating inputs
  are absent or vocabulary-mismatched, and the reranker built on them was
  measured worse than BM25 and demoted (#895); the "lexical synonymy residue" is
  *claimed but untested* (#873).
- **Immediate next step:** Phase 0.5 — **#905** (loud failure on the live path)
  and **#902** (decide what `ast_json` holds), then the shape decision
  **#903 → #904**. Run milestone **#23** (reranker-discrimination stratum) in
  parallel: without it, none of Phase 0.5 can clear the merge gate, and it would
  land unmeasured exactly as the 25 capabilities of 2026-05-26 did.

Known-degraded specifics and the "read this before trusting a number" caveats
live in `DESIGN.md` → "Current state".

---

## 6. How this plan stays honest

- Every phase closes on a number recorded in `bench_history.jsonl`, not on
  "it obviously works".
- The contract suite fails if a stage regresses, drifts, or goes silently dead.
- Status here is single-owned and cross-checked against the live store; when a
  🎯/🔶 ships, its row flips in the same PR.
- Scope discipline: a capability that can't name (a) the metric it moves and
  (b) the test that shows it is a **research spike or deferred**, not a build
  task (`CLAUDE.md` → merge gate).
