# Klareco: AST-Native Orchestration

Klareco answers Esperanto questions by passing immutable, role-annotated ASTs
through a pipeline of small stages. The current effort is to **prove how far
deterministic processing alone can take us** before re-introducing learned
models. Once the deterministic floor is stable and measurable, we'll add
learned components where the data demands them.

This document describes the system as it actually is. For the long-term
thesis (decomposable contributions, explainability via attribution), see
`VISION.md`. For working conventions, see `CLAUDE.md`.

## Active architecture

```
Question (Esperanto)
  ↓
ParseQuestionStage          (deterministic) — 16-rule parser, role-annotated AST
  ↓
RetrieveStage               WhooshRetriever.retrieve_with_ast_roles
                            BM25 ∩ AST role-compatibility, top-k from Kuzu v2.1
  ↓
DeterministicRerankStage    AST-feature boost keyed by question type
                            (propra_nomo for WHO/WHERE, numero for WHEN/HOW_MANY,
                            no-op for WHAT/HOW/WHY)
  ↓
RerankStage                 stub today; will host a learned cross-encoder when
                            the deterministic floor is stable enough to measure
                            an improvement
  ↓
ExtractAndGenerateStage     ExtractiveAnswerGenerator over the top-20
                            passages; fact extraction is keyed by the
                            question's answer slot
  ↓
FormatOutputStage           assemble final_text with citation list
```

Stages exchange an immutable `QueryContext` and return a `ContextDelta` —
copy-on-write via `dataclasses.replace`. Per-stage timing is captured in
`metrics.py`; stages with non-trivial internal phases (Retrieve in
particular) publish sub-phase timings via `PhaseTimer` so evaluation can
see where the wall time actually goes inside one stage.

Files:
- `klareco/orchestrator/{pipeline,context,stage,factory,metrics,phase_timer}.py`
- `klareco/orchestrator/stages/{parse_question,retrieve,deterministic_rerank,rerank,extract_generate,format_output}.py`

## The schema-first foundation

Retrieval and extraction query a v2.1 Kuzu graph instead of pattern-matching
on word forms. The graph carries a four-layer semantic ontology:

- **Lexical** — verb classes (kreado, movo, pensado, ...), entity types
  (persono, loko, tempo, organizaĵo, evento, profesio), thematic roles
  (aganto, paciento, instrumento, ...)
- **Frame** — FrameNet-style frames with participant slots
- **Discourse** — RST relations (detalaĵo, kaŭzo, celo, kontrasto, ...)
- **Schema** — biographical / definitional / event slot weights for
  importance ranking

This means: no hardcoded gazetteers, no manual verb-synonym lists, no
question-type → importance-weight tables in code. The graph is the source
of truth; the pipeline queries it.

Helpers: `klareco/rag/kuzu_ast_reconstructor.py` reconstructs an AST from
the graph in <5ms (vs. ~50ms to re-parse the sentence). The
`klareco/knowledge/` package wraps the synonym / gazetteer / temporal /
spatial queries used by retrieval and extraction.

The graph itself is built offline by the `scripts/load_csv_to_kuzu_v2_1*`
pipeline; runtime opens it via `klareco/utils/kuzu_open.py`, which
honors `KLARECO_KUZU_BUFFER_MB` / `KLARECO_KUZU_MAX_THREADS` /
`KLARECO_KUZU_DB_PATH` so parallel workers and out-of-tree runners (Modal)
can share one knob.

## How we measure progress

The current measurement target is **retrieval-rank metrics on extractive
QA**, not final-answer accuracy. We care about:

- top-1 hit rate, top-5, top-20, MRR (does the right passage rank well?)
- extraction accuracy conditional on retrieval (given the right passage,
  do we pull out the right span?)
- per-stage and per-phase wall time (where does the time actually go?)

Eval entry points:
- `scripts/eval/evaluate_extractive_qa.py` — local, single-process
- `scripts/eval/modal_eval.py` — Modal cloud, parallel workers
- `scripts/eval/local_parallel_bench.sh` — local fanout (DuckDB store)

All three go through `klareco/eval/qa_metrics.py` so the same evaluator
runs everywhere.

Test sets live in `data/test_sets/` (not in git). The hand-curated set is
~30 questions, which is below the noise floor for the effect sizes we
expect from individual changes (#726). `scripts/eval/build_synthetic_who_test_set.py`
generates a larger WHO set from `propra_nomo` subject patterns to give
us measurable headroom.

### Measurement integrity: Q&A pairs must be discriminating

Autopsy (2026-05-19, full 5.4M DuckDB store) of the `gold_anchor_50`
recall ceiling: of the 16/50 misses, **0/16** were retrievable from the
question terms, and **0/16** even with the labeled answer entity added.
Cause: templated questions ("Kiu kreis verkojn?" = "Who created works?")
+ truncated answers ("Samuel" not "Samuel Twardowski") contain only
high-frequency generic terms — no term narrows 5.4M sentences to the
gold one. The ~68% recall ceiling is the **test set's** pathology, not
the pipeline's. **A valid retrieval pair must retain ≥1 discriminating
term** (full proper-name answer + a discriminating object/entity from
the source). Commit `4e9c373` added agent-coherence verification but not
discriminability — that is the missing constraint, now enforced by an
empirical gate in the generator (a pair is kept only if a raw BM25
query on its question terms surfaces the source sentence within a
generous top-K, so impossible pairs are excluded while hard-but-possible
ones remain).

## Parser quality: deterministic ceiling vs model territory

UD_Esperanto-Prago (131 gold sentences, `scripts/eval/eval_ud_prago.py`)
is the only trustworthy parser ruler — independent of the Q&A stack.
Current: POS strict **80.3%**, scheme-adjusted **93.3%** (the +173
delta is UD-vs-Esperanto scheme choices, not errors). Mismatch taxonomy:

**Not errors — UD-vs-Esperanto scheme differences (do NOT "fix"; already
credited by the scheme-adjusted score):**
- `PRON→adjektivo` (66): Esperanto possessives/determiners (nia, ĝiaj,
  siajn) take adjectival agreement — Klareco is linguistically correct.
- `DET/PRON/ADV→korelativo` (52): the Esperanto correlative table-word
  system; UD splits it across DET/PRON/ADV.
- `ADV→partiklo` (29): Esperanto particles (eĉ, ankaŭ, nur).
- `VERB→adjektivo/adverbo` (27): participles are verbal adjectives /
  adverbs in Esperanto.

**Deterministically fixable (concrete, real wins):**
1. Closed-class primitives missing from the inventory: `-aŭ` adverbs/
   preps (almenaŭ, anstataŭ, …) and `ol` → `*→nekonata` (~16 tokens).
2. Roman-numeral recognition (II., III., IV.) → currently `NUM→verbo`
   (~6 tokens).
3. Single-letter initials ("L." in "D-ro L. L. Zamenhof") → kill the
   bogus `ekzemplo` fallback (~11 tokens).
4. Title-Case common nouns governed by "la" or inside «» quotation
   ("la Lingvo «Esperanto»") wrongly promoted to `propra_nomo` — gate
   the promotion on those positions (subset of `NOUN→propra_nomo`).

**Cannot be done deterministically — needs a learned model:**
The irreducible residue of `NOUN↔propra_nomo`: a capitalized token
where morphology, position, and function-word context give *no* signal
(novel surnames that are also common words; foreign text; names whose
only cue is capitalization, neutralized at sentence-initial / all-caps
title / quotation position). Esperanto has no morphological proper-noun
marker, so disambiguation here requires distributional/world knowledge
— this is the learned proper-noun tie-breaker, and it is the principled
boundary where deterministic processing stops.

## What is deferred

These have working code on disk but are not in the active loop:

| Component | Status | Where |
|-----------|--------|-------|
| Stage 1 root embeddings | trained, not consumed by orchestrator | `klareco/embeddings/` |
| M1 selectional preference | superseded by direct AST-role checks for now | `klareco/models/` |
| Neural cross-encoder reranker | RerankStage stub; no model loaded | `klareco/orchestrator/stages/rerank.py` |
| Entity classifier (tier 3) | not in the active pipeline | `klareco/models/entity_classifier.py` |
| Summarization stack | unused | `klareco/summarization/` |

These will return — but only when the deterministic floor is stable and
we can attribute a measurable improvement to a specific learned
component. Adding learned layers before that point only makes the system
harder to diagnose.

When `klareco/embeddings/`, `klareco/models/`, and `klareco/summarization/`
get pruned from working code, the v2.1 schema and the training data
under `data/` (not in git) remain — those are the expensive artifacts
worth keeping. Retraining can be replayed from there.

## Decision principles

1. **Deterministic before learned.** Every capability starts as a rule or
   a graph query. Only after the rule is in place and measured do we
   consider a learned replacement, and only if we can show the learned
   version moves a number the rule version couldn't.
2. **Reorder, don't expand.** Adding retrieval paths that produce extra
   candidates almost always dilutes top-k. Prefer stages that *reorder*
   the existing candidate set (DeterministicRerankStage is the model).
3. **Schema, not hardcoded lists.** If a feature needs a list of place
   names, verb synonyms, importance weights, or question-type tables —
   query the Kuzu ontology, don't hardcode.
4. **Immutable context.** Stages return `ContextDelta`, never mutate in
   place. Side-channel state lives in the model registry passed at
   pipeline-build time, not in the context.
5. **Make the failure visible.** Phase timings, ranks, and per-stage
   confidences are part of evaluation output, not just final accuracy.

## Repository map (current)

```
klareco/
  parser.py              16-rule deterministic parser → AST
  deparser.py            AST → Esperanto text
  proper_nouns.py        v3 cleaned + Wikipedia-category dictionary
  cli.py                 CLI entry
  __main__.py            `python -m klareco …`
  orchestrator/          immutable QueryContext pipeline (active spine)
  rag/                   WhooshRetriever, UnifiedASTExtractor,
                         extractive_answering, query expanders,
                         ast_semantic_ranker, kuzu_ast_reconstructor
  knowledge/             synonyms, gazetteers, temporal/spatial,
                         morphology, semantic_bridge — all backed by
                         Kuzu queries, no hardcoded lists
  eval/                  qa_metrics, used by local + Modal evaluators
  utils/kuzu_open.py     single Kuzu opener honoring env-var memory caps

scripts/
  evaluate_extractive_qa.py / modal_eval.py / compare_eval_results.py
  build_synthetic_who_test_set.py
  pipeline.sh + the per-stage corpus/index scripts it wraps
  *_propra_nomo* / *_propranoma_kategorio*  — proper-noun pipeline
  modal_upload_indexes.sh / local_parallel_bench.sh

data/   (not in git)
  indexes/v2.1_kuzu_index_full/         — production Kuzu graph
  indexes/whoosh_*                       — Whoosh BM25 index
  test_sets/                             — eval question sets
  proper_nouns_dynamic_v{1,2,3}.json     — proper-noun dictionary versions

models/ (not in git)
  reserved for when training resumes
```

## Roadmap (short horizon)

Tracked in GitHub issues, not in this document. The current EPIC is
[#713 — Improve QA accuracy through iterative AST-first improvements](https://github.com/marctjones/klareco/issues/713).
Live priorities are visible via `gh issue list --label "epic:713"` or the
GitHub project board.

## See also

- `VISION.md` — the long-term thesis (decomposable contributions, AST as
  the universal contract)
- `CLAUDE.md` — schema-first development conventions, the rules that
  prevent hardcoded gazetteers/lists from creeping back in
- `README.md` — setup and quickstart commands
- `docs/VERSION_COMPATIBILITY.md` — the deferred v3.0 model-retraining
  plan (kept for when training resumes; not driving current work)
