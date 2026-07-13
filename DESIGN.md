# Klareco: AST-Native Orchestration

Klareco answers Esperanto questions by passing immutable, role-annotated ASTs
through a pipeline of small stages. Everything that can be done deterministically
is done deterministically; where that breaks, we characterize the break before
reaching for a model (see `VISION.md`).

This document describes the system **as it actually is**, including the parts that
are currently broken. Read the "Current state" section before you trust any number
in this repository.

---

## Current state (2026-07-12) — READ THIS FIRST

The code survived a laptop migration in June 2026; several **data artifacts did
not**. The system runs end-to-end and returns cited answers, and 406 tests pass,
but four artifacts the code assumes exist are missing — and **they fail silently**.
No crash, just quietly degraded output.

| Missing artifact | Effect |
|---|---|
| `data/vocabularies/protected_roots.json` | Parser over-decomposes: `Esperanton` → root `esper` + participle suffix `ant`. Causes ~10 test failures. |
| `data/proper_nouns_dynamic_v3.json` (+ v2, legacy, static) | Proper-noun dictionary loads as `None`; parser falls back to capitalization heuristics only. |
| `data/ontology_export/kuzu_ontology_snapshot.json` | `ontology_nodes` / `ontology_edges` are **empty**; `verb_klaso` is **0% populated**. The semantic ontology is absent at runtime. |
| `entity_facts` table | **Crashes** `BiographyFormatStage`. Rebuildable via `scripts/index/extract_entity_facts.py`. |

Two consequences that are easy to get wrong:

**1. The store was built *without* `protected_roots.json` too.** It contains
`esper`, not `esperant`. Today questions and index are *consistently wrong
together*, which is why retrieval works at all. Restoring the JSON without a
corpus reparse would make questions parse to `esperant` while 5.4M rows still say
`esper`, **breaking retrieval**. Restoring parser data therefore *requires* a
reparse + store rebuild (~5–6 h). It is not a drop-in fix.

**2. The ontology is not lost, only unported.** The verb classes, entity types,
thematic roles, and schema slots are Python literals in
`scripts/index/extend_kuzu_schema_semantic_ontology.py`. They were only ever
loaded into Kuzu. The DuckDB path expects a snapshot JSON exported *from Kuzu* —
and Kuzu is gone. The fix is to emit the snapshot directly from the class
definitions in code.

The Kuzu → DuckDB migration is **complete in code** (no `kuzu` imports remain in
`klareco/`) but **incomplete in data** (the ontology population step was never
ported). Most of what looks like a modeling problem below traces back to that one
fact.

---

## Active architecture

```
Question (Esperanto)
  ↓
ParseQuestionStage        (deterministic) — 16-rule parser, role-annotated AST
  ↓
DialogStage               optional (off by default) — multi-turn pronoun resolution
MathToolStage             short-circuits arithmetic questions (SymPy)
PlannerStage              decomposes nested questions; no-op on simple ones
  ↓
RetrieveStage             DuckDBRetriever — BM25 ∩ AST role compatibility
                          (NB: klareco/rag/whoosh_retriever.py is DEAD — its
                          __init__ raises NotImplementedError. DuckDBRetriever
                          is what factory.py actually builds.)
  ↓
DeterministicRerankStage  AST-feature boost keyed by question type
ASTAwareRerankStage       structural reranker over shredded AST columns (#741)
RerankStage               STUB — no model loaded
  ↓
ExtractAndGenerateStage   ExtractiveAnswerGenerator over top-k passages
BiographyFormatStage      multi-sentence biography / definition output (#775)
  ↓
FormatOutputStage         final text + citations
```

Stages exchange an immutable `QueryContext` and return a `ContextDelta`
(copy-on-write via `dataclasses.replace`). Per-stage timing is captured in
`metrics.py`; stages with non-trivial internal phases publish sub-phase timings
via `PhaseTimer`.

Entry point: `build_default_pipeline(whoosh_index_dir=...)` →
`Orchestrator.answer(question)`. There is **no** `python -m klareco run`
subcommand; the CLI exposes `parse`, `query`, `translate`, `corpus`, `info`.

Files: `klareco/orchestrator/{pipeline,context,stage,factory,metrics,phase_timer}.py`,
`klareco/orchestrator/stages/*.py`.

## The store

`data/indexes/duckdb_store.db` (~32 GB) — one flat table, `sentences`, carrying the
parsed AST as a JSON blob plus shredded columns for fast filtering. Whoosh
(`data/indexes/whoosh_v2`, ~2.8 GB) provides BM25.

Actual column population, measured 2026-07-12 over 5,391,442 rows:

| Column | Populated | Note |
|---|---|---|
| `ast_json` | 100% | the AST blob; `json.loads` ≈ 0.9 ms |
| `aliaj_json` | 100% | modifiers (loko / tempo / numeral bearing) |
| `subj_radiko` / `subj_vortspeco` | 94.7% | |
| `verb_radiko` | 65.0% | |
| `verb_tempo` | 60.0% | |
| `subj_propranoma_kat` | 55.0% | **structural** categories only (see below) |
| `obj_radiko` | 23.4% | |
| `verb_negated` | 1.1% | |
| `verb_klaso` | **0.0%** | column exists, never populated |

Two data-quality facts worth internalizing:

- **`propra_nomo` is massively over-triggered.** 2.25M rows (42% of the corpus)
  have a proper-noun subject, including Wikipedia artifacts like `REDIRECT` (42K),
  `The` (29K), `Ĝia` (29K). A feature that fires on 42% of the corpus is not a
  discriminating feature.
- **`subj_propranoma_kat` speaks a different vocabulary than the reranker.** It
  carries the *parser's structural* categories (`propranomo`, `neologismo`,
  `propranomo_esperantigita`) because it was backfilled from `ast_json`. The
  reranker was written against the *ontology's semantic* categories (`persono`,
  `loko`). They never match.

## Why every reranker is tied (the central open problem)

The last benchmark (`results/bench_cleanish17_rerankers.json`, 17 questions) has
all nine rerankers on identical numbers: recall@1 = 11, recall@5 = 17, answer
accuracy 70–76%. This is not because reranking is hopeless. There are two
independent, compounding causes:

**Cause 1 — every reranker's *distinguishing* component is dead; only their
*shared* components are alive.** This is the precise mechanism, and it is sharper
than "it's all just BM25":

| Component | Status |
|---|---|
| BM25 score | **alive** — shared by all |
| phrase / entity-in-text boost | **alive** — shared by all |
| exact `obj_radiko` match | **alive** (23% populated) — shared |
| tense compatibility | **alive** (60%) — shared |
| exact `verb_radiko` match | **alive** (65%) — shared |
| *same `verb_klaso`* generalization (0.6 weight) | **DEAD** — `verb_klaso` is 0% populated |
| negation agreement | **DEAD in practice** — `verb_negated` 1.1% |
| answer-type gating on `expected_kats` = `persono` | **DEAD** — store holds `propranomo`, never `persono` |

Everything the rerankers *share* is alive; everything that makes any one of them
*different* is dead. Identical live inputs produce identical rankings. So
`G_ast_aware` cannot beat the dumb baseline — **its cleverness lives entirely in
the empty columns.** The smart reranker is already written. It is running on
nothing.

**Cause 2 — the test set has no headroom.** recall@5 = 17/17 means the answer is
*already* in the top 5 for every question before any reranker runs. The questions
("Kiu inventis «Nonograms»?") contain quoted titles that BM25 nails outright. A
perfect reranker could not move recall@5 — and none moved recall@1 either.

These must be fixed **together**. Populating the ontology without a discriminating
test set shows nothing; building a hard test set without the ontology shows only
that BM25 fails.

### But restoring the ontology will NOT be enough (measured 2026-07-12)

The ontology's verb classes are seeded with **32 example roots** total
(`kreado-26` = `["fond","kre","produk","far"]`, and seven more classes like it).
The corpus contains **39,718 distinct verb roots**. Coverage:

| | |
|---|---|
| Sentences whose verb is in an ontology class | **304,057 — 5.6% of corpus** (8.7% of sentences that have a verb) |
| Top verb roots by frequency | `est` (1.24M), `hav` (146K), `situ` (71K), `trov` (65K), `aparten` (55K), `pov` (54K) — **none are in any class** |

So even a perfect ontology restore leaves `verb_klaso` NULL on ~91% of verbed
sentences, and the reranker's verb-class generalization would still almost never
fire. **The ontology is not a database that needs reconnecting; it is a stub that
was never filled in.** Hand-enumerating 39,718 roots into semantic classes is not
a plausible deterministic project — which is precisely the *lexical synonymy
residue* named in `VISION.md`. This is where the boundary actually is, and we
found it by measurement.

### The highest-leverage reranker features need no ontology at all

This reorders the work. The following are computable **today**, from `ast_json`
plus corpus statistics, with zero restore:

- **Answer-slot constraint from the interrogative's case.** `Kiu` (nominative) →
  the answer fills the SUBJECT slot; `Kiun` (accusative) → the OBJECT slot.
  Esperanto states the answer's grammatical role *morphologically*; English can
  only infer it from word order. This is a free, hard constraint and we do not
  currently use it.
- **Pronoun-subject exclusion.** A sentence whose subject is `li`/`ŝi`/`tiu`
  cannot answer a KIU question — the entity is not in the sentence. That is
  **642,063 sentences (11.9% of the corpus)** deterministically excluded from
  KIU candidacy, using nothing but `subj_vortspeco`.
- **Anchor weighting by specificity.** Among the presupposed (non-gap) terms,
  weight by corpus rarity: proper names, quoted titles, and years narrow the
  space; `est`/`hav`/`fari` narrow nothing. The `gold_anchor_50` autopsy already
  proved this empirically — the impossible questions were exactly the ones with
  no rare anchor.
- **Voice / participle normalization.** `-int-` / `-it-` + `de`-phrase means
  "Kiu fondis X", "la fondinto de X", and "X estis fondita de Y" are one
  `(agent, predicate, patient)` triple, derivable **by affix rule**. This
  generalizes across surface forms without knowing any lexical semantics.
- **Negation penalty.** A sentence asserting the opposite of the presupposition
  is lexically maximal and semantically wrong. BM25 ranks it up.

None of these require a verb class, an entity type, or a schema slot. They are
pure structure plus frequency — the deterministic case, unexploited.

## How we measure progress

The measurement target is **retrieval-rank metrics on extractive QA**:

- top-1 / top-5 / top-20 hit rate, MRR (does the right passage rank well?)
- extraction accuracy *conditional on* retrieval (given the right passage, do we
  pull the right span?)
- per-stage and per-phase wall time

Eval entry points: `scripts/eval/evaluate_extractive_qa.py` (local),
`scripts/eval/modal_eval.py` (Modal, parallel),
`scripts/eval/multi_reranker_bench.py` (reranker A/B),
`scripts/eval/test_new_trivia.py` (fresh OpenTDB trivia, head-to-head against a
local Ollama LLM). All share `klareco/eval/qa_metrics.py`.

### Measurement integrity: Q&A pairs must be discriminating

Autopsy of the `gold_anchor_50` recall ceiling: of 16/50 misses, **0/16** were
retrievable from the question terms, and **0/16** even with the labeled answer
entity added. Cause: templated questions ("Kiu kreis verkojn?") plus truncated
answers ("Samuel" not "Samuel Twardowski") contain only high-frequency generic
terms — no term narrows 5.4M sentences to the gold one. That ~68% ceiling was the
**test set's** pathology, not the pipeline's.

**A valid retrieval pair must retain ≥1 discriminating term** — a full proper
name, a quoted title, a year, a rare entity. This is enforced by an empirical gate
in the generator: a pair is kept only if a raw BM25 query on its question terms
surfaces the source sentence within a generous top-K.

Note the tension with the section above: the 17-question `cleanish` set is
*over*-discriminating — BM25 alone puts everything in the top 5. The target is
questions that are **hard but possible**: answerable, but not by lexical overlap
alone. That band is where reranking can be measured, and we do not currently have
a test set in it. This is the blocker (#736, #737).

## Parser quality: deterministic ceiling vs model territory

UD_Esperanto-Prago (131 gold sentences, `scripts/eval/eval_ud_prago.py`) is the
only trustworthy parser ruler — independent of the Q&A stack. Current: POS strict
**80.3%**, scheme-adjusted **93.3%** (the delta is UD-vs-Esperanto scheme choices,
not errors).

**Not errors — UD-vs-Esperanto scheme differences** (do not "fix"; already
credited by the scheme-adjusted score): `PRON→adjektivo` (66; Esperanto
possessives take adjectival agreement), `DET/PRON/ADV→korelativo` (52; the
correlative table), `ADV→partiklo` (29), `VERB→adjektivo/adverbo` (27; participles
are verbal adjectives/adverbs).

**Deterministically fixable — real wins:**
1. Closed-class primitives missing from the inventory: `-aŭ` adverbs/preps
   (almenaŭ, anstataŭ, …) and `ol` → currently `*→nekonata` (~16 tokens).
2. Roman-numeral recognition (II., III., IV.) → currently `NUM→verbo` (~6 tokens).
3. Single-letter initials ("L." in "D-ro L. L. Zamenhof") → kill the bogus
   `ekzemplo` fallback (~11 tokens).
4. Title-Case common nouns governed by `la` or inside «» wrongly promoted to
   `propra_nomo` — gate the promotion on those positions.

**Cannot be done deterministically — needs a learned model.** The irreducible
residue of `NOUN↔propra_nomo`: a capitalized token where morphology, position, and
function-word context give *no* signal. Esperanto has no morphological proper-noun
marker, so disambiguation requires distributional or world knowledge. This is the
principled boundary where deterministic processing stops — and, per `VISION.md`,
exactly the kind of finding this project exists to produce.

## The semantic ontology (intended source of truth, currently absent)

Retrieval and extraction are *supposed* to query a four-layer ontology instead of
pattern-matching on word forms:

- **Lexical** — verb classes (kreado, movo, pensado, …), entity types (persono,
  loko, tempo, organizaĵo, evento, profesio), thematic roles (aganto, paciento,
  instrumento, …)
- **Frame** — FrameNet-style frames with participant slots
- **Discourse** — RST relations (detalaĵo, kaŭzo, celo, kontrasto, …)
- **Schema** — biographical / definitional / event slot weights for importance
  ranking

**Status: defined in code, not loaded at runtime.** See "Current state." Until the
snapshot is regenerated and loaded, every consumer either no-ops (`verb_klaso`) or
falls back to a hardcoded list (`klareco/knowledge/synonyms.py` has an explicit
hardcoded fallback). CLAUDE.md's "always query the ontology" rule is currently
*unfollowable*; the fallbacks are acknowledged debt, not a licence to add more.

**An honest caveat about the ontology itself.** Even when loaded, it is
hand-seeded and thin: `kreado-26` is backed by the root list `["fond", "kre",
"produk", "far"]`; `persono` by `["homo", "vir", "infan", "kuracist"]`. This is a
lookup table, and it cannot generalize to a root nobody enumerated. Querying it
instead of hardcoding a list in Python is better engineering — one source of
truth, one place to extend — but it is not a different *kind* of knowledge.
Lexical synonymy is a genuine learned residue that we are currently approximating
with a list, and the docs should not pretend otherwise.

## What is deferred

The learned stack has been **pruned from the working tree** (commits `b68320e`,
`822a3eb`, `313ec3e`). `klareco/embeddings/`, `klareco/models/`, and
`klareco/summarization/` no longer exist. What remains:

| Component | Status |
|---|---|
| Neural cross-encoder reranker | `RerankStage` is a **stub** — it is in the pipeline and does nothing |
| Root embeddings, M1 selectional preference, entity classifier, summarization | **deleted from the repo**; recoverable from git history, and the training data under `data/` (not in git) is the expensive artifact worth keeping |

This is deliberate: no learned component re-enters the pipeline until the
deterministic floor is stable and we can attribute a measurable improvement to it.

**Genuinely dead code that should be removed** (not deferred — broken):
`klareco/ontology/semantic_query.py` (still takes a `kuzu_conn`),
`klareco/schema/kuzu_ast_schema_v2_1.py`, `tests/test_kuzu_open.py` (imports the
deleted `klareco.utils`), and the `*_kuzu_*` scripts under `scripts/index/` —
**except** `extend_kuzu_schema_semantic_ontology.py`, which is the only surviving
source of the ontology's class definitions and must be preserved (or its literals
migrated) before anything named `kuzu` is deleted wholesale.

## Also unmeasured

The symbolic reasoning stack landed immediately before the migration and has
**never been benchmarked**: forward-chaining inference and transitive closure
(#749), path-finding (#761), the STRIPS-style planner (#771), the SymPy math tool
(#772), dialog state (#767), and the biography / definition / comparison
generators (#766, #775 — currently crashing on the missing `entity_facts` table).
It is wired into the factory and it runs. Whether it *helps* is unknown, and
cannot be known until a discriminating test set exists.

## Decision principles

1. **Deterministic before learned.** Every capability starts as a rule or a query.
   A learned replacement must show it moves a number the rule version couldn't —
   and the failure it fixes must be *characterized*, not merely observed.
2. **Reorder, don't expand.** Adding retrieval paths that produce extra candidates
   almost always dilutes top-k. Prefer stages that *reorder* the existing
   candidate set.
3. **Schema, not hardcoded lists.** One source of truth, extended in one place.
   (See the caveat above about what this does and does not buy you.)
4. **Immutable context.** Stages return `ContextDelta`, never mutate in place.
5. **Make the failure visible.** Phase timings, ranks, and per-stage confidences
   are part of evaluation output, not just final accuracy.
6. **A silently-degrading dependency is a bug.** Every artifact the pipeline loads
   must fail loudly if absent. The June migration cost weeks of invisible quality
   loss precisely because missing files logged a warning and carried on.

## Roadmap

Tracked in GitHub issues. Current EPICs:
[#713](https://github.com/marctjones/klareco/issues/713) (QA accuracy),
[#745](https://github.com/marctjones/klareco/issues/745) (entity-fact extraction),
[#747](https://github.com/marctjones/klareco/issues/747) (symbolic reasoning
layer). Immediate blockers: the restore work in "Current state", plus the
discriminating test sets (#736, #737).

## See also

- `VISION.md` — the thesis: map the deterministic/learned boundary
- `CLAUDE.md` — working conventions
- `README.md` — setup and quickstart
- `16RULES.MD` — Esperanto grammar specification
