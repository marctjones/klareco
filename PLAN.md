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
- **Data substrate:** DuckDB store (~4.6M sentences: `ast_json` blob + shredded
  columns + provenance) → Whoosh BM25 index → hand-seeded semantic ontology.
  Built by a fixed pipeline: acquire → clean → extract → parse → index.
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

### Phase 1 — MVP-1: single-turn QA, honest/loud/measured  ·  milestone #29  ·  🎯 next
*Goal: the smallest QA system that actually works, on the enforced contract.*
- 🎯 **#881 (P0)** — reconnect facts as `FactFragment`s over the drifted
  `entity_facts` schema; unblocks reasoning/planner/generation in one move.
- 🎯 **#895** — trim the dead `verb_klaso` SELECT so `ast_aware_rerank` stops
  silently failing (then fold it into the contract suite).
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

## 5. Where we are (2026-07-18)

- **Foundation, not features.** Phase 0 is most of the way done; almost nothing
  "new" shipped recently by design — the effort proved the old capabilities were
  mostly dead/unmeasured and built the machinery so that can't recur.
- **Working & measured:** 16-rule parser (UD-Prago 80.3% POS), DuckDB store +
  Whoosh, retrieval (recall@200 = 100% trivial/rerankable, 36% deep), rerankers
  differentiated on honest sets, math tool live, the contract suite + decoder +
  CLI v0.
- **The honest weak points:** end-to-end `token_f1 = 0.014` (returns passages,
  not spans); the fact-consuming symbolic layer is silently dead against a
  drifted schema (#881); the "lexical synonymy residue" is *claimed but untested*
  (#873).
- **Immediate next step:** Phase 1 — **#881** then **#895/#869**. That converts
  four "0"s into measured numbers and gives MVP-1 its first honest baseline.

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
