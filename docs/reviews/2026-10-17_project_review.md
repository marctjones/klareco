# Klareco: project review and recommendations

**Date:** 2026-10-17
**Reviewed:** repository at commit `ac58ea4` (branch `dev`, last commit 2026-09-09), the GitHub tracker (942 issues filed, 122 open, 42 milestones), the benchmark ledger `data/perf/bench_history.jsonl` (127 rows), the project documents (`VISION.md`, `DESIGN.md`, `PLAN.md`, `CLAUDE.md`, `AGENTS.md`, `README.md`, `docs/`), and the code and tests.
**Method:** every number below carries a date and a pointer. Items marked **verified live** were re-run during this review. Everything else is taken from the project's own records, with the date the project recorded it. Under the project's own content tiers this review could also be posted as a GitHub Discussion.

---

## 1. Verdict in one page

**The idea is sound, and it has precedent.** Using a planned, regular language to measure how much of language understanding is classical computation is a legitimate research question. It was asked once before, at scale, by the DLT machine-translation project (1985 to 1990), which used Esperanto as its interlingua for exactly the reasons `VISION.md` gives, and which found that syntax was tractable while meaning was not. Klareco has reproduced that finding with a modern ruler: its best-characterized residue is inferential paraphrase and world knowledge, and the ReVo synonym table could not touch it.

**The most valuable things the project has produced, in order:**

1. **The method.** The merge gate, the append-only ledger, paired-bootstrap confidence intervals, band-stratified gold sets built from external trivia so they cannot be circular, the six-tier test taxonomy, and a status block generated from the live store. This is the part that would transfer to any project.
2. **The parser.** A deterministic Esperanto parser that emits UD CoNLL-U, scores Prago LAS 70.0% and Cairo LAS 81.9% (2026-09-07, 151 development-informed sentences), carries morphological alternatives instead of guessing, and round-trips losslessly to storage. Oya's 2025 UD paper notes that no Esperanto parser produced CoNLL-U; this one does.
3. **The residue map,** even though it does not yet exist as one document. It is spread across 127 ledger rows, issue comments, and `VISION.md`.
4. **The orchestration contract.** Immutable context, decodable thought, loud failure, injected resources, and a contract suite that runs in ten seconds with no data (verified live: 85 passed, 3 skipped).

**The honest state of the system:**

| Area | State (date, pointer) |
|---|---|
| Parser | Prago LAS 70.0 / UAS 76.5, Cairo LAS 81.9 / UAS 85.9 (2026-09-07, `DESIGN.md`); up from 28.8 / 34.9 on 2026-07-14 |
| Question answering | 26.6% answer accuracy on `rebaseline_500`, exact match 0.0, token F1 0.012 (2026-07-19 scorecard, ledger) |
| Production store | Built 2026-07-14 with a July parser in the lossy legacy format; none of the September parser gains reach production; 76M rows in two tables have zero production readers (2026-07-20, #901, #904) |
| Symbolic modules | 1 of 5 live (math); planner, reasoning, generation dead against a drifted schema; dialog off (2026-07-18 probe) |
| Learned components | none; the learned stack was deleted from the tree |
| Gold data | QA: 1,345 pairs, sound. Parser: 151 UD sentences, both development-informed; the independent 200-sentence queue is 0/200 reviewed (#920) |
| Continuous integration | none; no `.github/` directory; only a pre-commit docstring linter (verified live) |
| Human review capacity | one person, who does not read Esperanto (`scripts/eval/build_annotation_review_packet.py`, 2026-09-07) |
| Activity | dormant since 2026-09-09 |

**Three structural problems, not bugs:**

1. **The boundary is a function of effort, and the project has no effort budget.** Where "deterministic computation stops" depends on how many linguist-years you spend. EspGram reached about 92% syntactic function with a declarative Constraint Grammar written by a professional over years. Klareco's `parser.py` is 5,636 lines of procedural Python whose largest function is about 930 lines. The project has the right instruments for this (oracle decomposition, candidate ceilings) but no stated point at which it will stop pushing a rule set and call the remainder residue.
2. **The task mostly measures what the grammar cannot help with.** Open-domain trivia over Wikipedia is decided by lexical retrieval and world knowledge. The deep band confirms it: every reranker measured on the honest set sits within 0.02 MRR of BM25, and the structural lever the project advertises most, reading the answer slot off the interrogative's case, is still unused in retrieval (`DESIGN.md` says so, and a search of the retriever, rerank stages, and extractor for any read of the interrogative's `kazo` found none on 2026-10-17).
3. **Throughput exceeds verification capacity.** The same pattern has occurred three times: November 2025 ("Project 100% complete!" in two days, then a full purge), May 2026 (25 capability issues in one day, eight built, none measured), and September 2026 (a parser cycle shipped a crash while the merge gate itself could not collect). The merge gate did not stop the third one, because a gate that nothing runs automatically is a convention.

**Keep / Redo / Stop / Start, in brief** (details in section 10):

- **Keep:** the merge gate and ledger; the UD parser and its lossless storage; the QA gold sets and band stratification; the orchestration contract and contract suite; the executable status block; the ReVo-derived lexicon and ontology.
- **Redo:** the production store, once, with v2 blobs and a clause-level projection; the parser gold set, with Esperanto-speaking reviewers; the QA benchmark, with gold answer spans and structure-sensitive strata; the documentation, cut to a tenth of its size; the "16 rules" framing.
- **Stop:** building new capability modules (20 open issues in the "Phase 4+" milestone that cannot name a number); parser-candidate and decoder research until gold exists; writing process prose; carrying dead code and hardcoded lists; running 16 open milestones.
- **Start:** a CI workflow for the data-free tiers; an effort budget and stopping rule per capability; the unexploited deterministic retrieval levers; a Hindle-and-Rooth style attachment table from the corpus; one learned component in shadow mode, dense retrieval for the deep band; an LLM reference row on the same gold set; a single residue-map document.

**Strategic recommendation:** treat the parser and the residue map as the deliverables, and the QA system as the instrument that produces the map. Section 10.6 lays out three options; the recommended one is to publish the map and the parser first, and continue the QA line only through the store rebuild and the one learned experiment the thesis has already justified.

---

## 2. What Klareco set out to do, and how the goal moved

**The original framing (November 2025 to mid 2026)** was a small-model project: "Pure Esperanto AI", get grammar for free from the 16 rules so that learned parameters can be spent on reasoning, with milestones literally named "M1: Single-Turn QA (Match OLMo-1B)" and "M2: Multi-Turn Chat (Match Llama-3.1-8B)" (closed milestones 5 and 6). Root embeddings, compositional morpheme embeddings, a selectional-preference model, a plausibility scorer, and a neural reranker were all built in this period.

**The current framing (rewritten 2026-07)** is boundary discovery. `VISION.md` states it as a question: given a language with a genuinely regular grammar, how much of understanding is ordinary deterministic programming, and what is the irreducible residue? The method has five steps: attempt it deterministically, find where it breaks and why, characterize the residue as a property of the problem, only then add a learned component targeted at that residue, and keep the contribution decomposable so the learned component's share can always be measured. The stated deliverable is a map, not a leaderboard score. The working artifact is an Esperanto QA system where every answer is grounded, cited, and decodable stage by stage.

The rewrite is a genuine improvement. It stopped pre-assigning grammar to rules and reasoning to models, and it made negative results first-class. It also inherited a large codebase built under the old framing, which is the source of much of what follows.

---

## 3. Timeline in six eras

From `git log`: 717 commits between 2025-11-11 and 2026-09-09, 707 by Marc Jones and 10 attributed to Claude. Commits per month: Nov 105, Dec 59, Jan 117, Feb 27, Mar 82, Apr 13, May 83, Jun 1, Jul 166, Aug 0, Sep 64.

| Era | What was built | What survived |
|---|---|---|
| **1. The agentic sprint** (2025-11-11 to 11-13) | Expert system with gating network, Tree-LSTM RAG, blueprint and execution loop, memory system, goals and values, external tools, learning loop. Commit 2025-11-13: "Project 100% complete!" | Nothing. 2025-11-25: "Purge obsolete components for rewrite". The pure-Python parser idea survived. |
| **2. The learned stack** (2025-11-25 to 2026-03) | Graph2Seq generator, root embeddings ("93% synonym accuracy", 2025-12-26), compositional embeddings, M1 selectional preference, Kuzu graph store, FAISS, hybrid embeddings, plausibility scorer ("68.1% F1", 2026-03-23), four-layer semantic ontology seeded by hand (2026-03-28), Whoosh BM25, unified extractor. | Whoosh, the ontology layer names, the function-word exclusion principle, the unified extractor (3,084 lines, still on the live path). Embeddings, models, and summarization were deleted from the tree. No number from this era survives in the ledger, which starts 2026-06-15. |
| **3. The AST-native pivot** (2026-04 to 05) | Orchestrator with immutable context (2026-04-27); two pruning passes (scripts 152 to 39); Kuzu retired and DuckDB adopted (2026-05-17); QA test-set quality standard after the legacy sets scored 0/140 on a discriminability audit (2026-05-21); AST-aware reranker; 25 capability issues filed on 2026-05-26; the symbolic stack (reasoning, planner, dialog, math, generators) built 2026-05-26 to 27. | The orchestrator, DuckDB, the QA standard, the math tool. The symbolic stack ran unmeasured and, as later found, mostly dead. |
| **4. The migration** (2026-06) | One commit. A laptop migration lost `merged_roots.json`, the proper-noun dictionary, and the ontology snapshot. The parser fell back to an empty lexicon and tagged 41.8% of store sentences with a proper-noun subject (#821). Nothing failed loudly. | The lesson "a silently degrading dependency is a bug", now enforced in `preflight.py` and in the parser's first lines. |
| **5. The reckoning** (2026-07-13 to 07-20, 166 commits) | Merge gate; six-tier test taxonomy; environment contract tests; UD Prago and Cairo as rulers; the parse forest; LAS from 28.8% to 62.3% on Prago in a week; ReVo-derived lexicon and ontology (12,663 roots, 8,709 hypernym edges); `qa_gold_v2` (1,345 pairs from OpenTDB trivia); band stratification; paired bootstrap; alias bridge and proper-noun reranker shipped; orchestration contract (decoder, loud failure, default-off, contract suite); the store mis-projection finding (#901). | Almost all of it. This is the project's best work. |
| **6. The parser-first cycle** (2026-09-05 to 09-09, 64 commits) | Prago LAS 62.3 to 70.0, Cairo 66.4 to 81.9; lossless storage v2 (151/151 round-trips, up from 45/151); stand-off annotation layers; bounded candidate generation and a global decoder spike (parked); crash fixes (2.47% of store sentences crashed `parse()` before #927); discourse residue spikes; annotation queues built from literary and news corpora; a hand-glossed review packet for a reviewer who does not read Esperanto. Then silence. | Parser gains in code, not yet in the store. |

Two observations about the shape of this history. First, every era that built capabilities in bulk was followed by an era that discovered most of them were dead, unmeasured, or circular. Second, the productive eras were the ones that built rulers before features: July 2026 is the model.

---

## 4. The design as it stands

### 4.1 The pipeline that actually runs

`build_default_pipeline()` in `klareco/orchestrator/factory.py` constructs, in order:

```
ParseQuestionStage          16-rule parser, role-annotated AST
MathToolStage               SymPy; short-circuits arithmetic questions (default on)
RetrieveStage               DuckDBRetriever: Whoosh BM25 over content terms, ALIASO alias bridge,
                            AST role compatibility; reads five store columns
DeterministicRerankStage    answer-shape boost keyed by question type
ProperNounRerankStage       additive proper-noun overlap boost (#877, shipped 2026-07-19)
RerankStage                 stub; no model registered
ExtractAndGenerateStage     ExtractiveAnswerGenerator over the top-k passages
FormatOutputStage           final text plus citations
```

Off by default: `DialogStage`, `PlannerStage`, `BiographyFormatStage`. Demoted: `ASTAwareRerankStage` (MRR 0.3619 to 0.3446 on `rebaseline_210`, #895). The retriever is `DuckDBRetriever`; `klareco/rag/whoosh_retriever.py` (1,901 lines) is dead, raises on construction, and is still exported from `klareco/rag/__init__.py` and constructed by `scripts/eval/debug_retrieval.py`.

### 4.2 The thought

`QueryContext` is a frozen dataclass holding a `SymbolicLayer` (question AST, passages, fact fragments, segments, citations, final text) and a `LatentLayer` (embeddings, relevance matrix). Stages return a `ContextDelta`; the orchestrator applies it copy-on-write and records a trace. A universal decoder renders any context as Esperanto with `[regulo]` or `[modelo]` provenance tags. A recovered stage failure stamps a `stage_failed:<name>` flag so it cannot be invisible. Stages declare `REQUIRES` and construction preflights them. The six normative rules are in `DESIGN.md`; `tests/contract/` enforces immutability, delta discipline, decodability, attribution, no private connections, loud failure, and golden traces.

This is good engineering and it is small: `klareco/orchestrator/` is 2,490 lines including the stages (verified live). It earned its keep on 2026-07-18, when enforcing it found four dead subsystems.

### 4.3 The parser

`klareco/parser.py` (5,636 lines) plus `morphology.py`, `syntax_graph.py`, `syntax_rules.py`, `forest.py`, `ast_storage.py`, `ast_annotations.py`, `conllu.py`. `parse()` returns three views of each sentence sharing token dicts: `vortoj` (a UD dependency tree, declared source of truth), `propozicioj` (clause frames), and a legacy flat subject-verb-object frame. Content words carry ranked morphological alternatives; where the grammar licenses more than one reading, an OR-node records who collapsed it. Storage v2 round-trips the full structure (151/151 fixture sentences, 2026-09-05). Attachment is done by `attach_all()`, a procedural pass of roughly 930 lines with rule IDs and partial before/after traces.

### 4.4 The store

`data/indexes/duckdb_store.db`: six tables, 47.3 GB as measured 2026-07-20, 44 GB on disk today. `sentences` 4,624,110 rows (down from 5,391,442 after the redirect and English cleanup, #823); `clauses` 6,954,531; `dependency_arcs` 68,887,052; `entity_facts` 1,006,992; `ontology_edges` 237,739; `ontology_nodes` 12,798. Whoosh index 2.6 GB, 4,624,110 docs. Production reads `sentences(sid, subj_vortspeco, obj_radiko, text, ast_json)` and the ALIASO edges. The stored `ast_json` is the legacy lossy compact form; `clauses` and `dependency_arcs` have no production readers; 261,372 sentences have a clause subject that the flat column does not show, and 2,263,488 clause subjects (40.9%) differ from the flat column (all 2026-07-20, #901 to #908).

### 4.5 Measurement

- Parser: `tests/fixtures/ud/` (Prago 131 sentences, 2,712 scored tokens; Cairo 20 sentences, 149 tokens); `scripts/eval/eval_conllu.py`; regression floors in `tests/test_parser_ud_accuracy.py`; candidate ceilings via `parser_oracle_decomposition.py`.
- QA: `qa_gold_v2.jsonl` (1,345 pairs: deep 506, rerankable 629, trivial 210), built from OpenTDB English trivia translated by Claude, gated by the parser and a corpus answerability judge; `rebaseline_500` and `rebaseline_210` stratified subsets; band probes (`alias_variant`, `common_terms`, `low_overlap`); `smoke.jsonl`. Paired-bootstrap MRR CIs. Scoring is deterministic (no LLM judge at scoring time).
- Ledger: `data/perf/bench_history.jsonl`, 127 rows, tracked in git (the July memory note saying it was gitignored is no longer true).
- Tests: six tiers applied by filename in `tests/conftest.py`. `pytest -m "contract or accuracy"`: 85 passed, 3 skipped, 10.4 s (verified live 2026-10-17). The accuracy tier still skips its trust check because the latest QA bench row lacks recall fields.
- Status: `scripts/validate/status_report.py --write DESIGN.md` regenerates the "Current state" table from the live store (last generated 2026-09-07).

### 4.6 Process rules

The merge gate (no capability merges without a number that moved), two tracks (build versus research spike), schema-first (query the ontology, no hardcoded lists), script versioning docstrings enforced by a pre-commit hook, and a policy that long-running scripts are wrapped and run by the user. These rules are written in `CLAUDE.md` (1,190 lines), `AGENTS.md`, `DESIGN.md`, and `PLAN.md`, with substantial overlap.

---

## 5. How far along the implementation is

### 5.1 Component status

| Component | Project's claim | Evidence | Assessment |
|---|---|---|---|
| Morphology and POS | deterministic, mature | scheme-adjusted POS 95.1% Prago / 96.6% Cairo; 32% of running tokens have more than one rule-licensed segmentation, collapsed to 0.099% with the typed lexicon and forest (2026-07-13/14 ledger); coverage 95.5% of the 15,000 most frequent types versus Apertium's 47.9% (2026-07-14) | Strong. The collapse required a curated 12,663-root lexicon with semantic root classes from ReVo, which is lexical knowledge, not grammar. |
| Subject and object roles | the accusative gives them for free | subject F1 86.1 / 97.8, object F1 80.5 / 92.3 (2026-09-07) | Holds. The 2026-07-13 move from one frame per sentence to one frame per clause broke a hard 42.5% recall ceiling. |
| Attachment (LAS) | in progress | Prago LAS 70.1 / UAS 76.6; Cairo 81.9 / 85.9; residual Prago errors nmod 145, advmod 105, conj 96, nsubj 83, obj 63 (`docs/PARSER_SEVEN_STEP_RESULTS.md`); candidate-head recall 76.5 / 75.8 against an 80 / 85 target; the global decoder moved 5 of 2,709 tokens (2026-09-07) | Real progress on a ruler too small to see increments. Both fixtures are development-informed. |
| Proper nouns | residue under test (#819) | 27.6% to 58.0% strict F1 without a gazetteer, on 41 gold PROPN tokens; Cairo precision 1.0, recall 0.357 (2026-07-13); document-level consistency settles 36.8% of weak-evidence cases (2026-09-08, no gold) | Promising; unmeasured at any meaningful scale (#820 open). |
| Deparser | "grammatically correct by construction" | round-trip 59.9% to 80.5% (2026-07-13); #801 and #813 open: "the deparser is broken" | The VISION claim is false today, and the project says so. |
| Storage v2 and annotation layers | lossless, extensible | 151/151 exact round-trips; 10,000-row side store round-trips; +47.6% blob size, +20% parse time (2026-09-05) | Done in code; not deployed to production. |
| Production store | honest data | built 2026-07-14; 21 oversized rows block a full rebuild; blobs lossy; rich tables unread | The single biggest gap between what the code can do and what the system does. |
| Retrieval | recall healthy | recall@200: 0.86 gold trivia, 1.0 synthetic, 0.98 clause set (2026-07-17); deep band ~0.07 MRR for every arm | First-stage lexical recall is the binding constraint on the deep band, as designed. |
| Reranking | measured, small wins | on `qa_gold_v2`: BM25 0.2239, B_phrase 0.2407, H 0.2305, J_tree 0.2297, I_clause 0.2176 (2026-07-17); alias bridge +0.0226 MRR on its band (CI excludes 0), proper-noun boost +0.0135 on `rebaseline_500` (CI excludes 0), both shipped | Structure buys a little, locally. Neither win moved answer accuracy. |
| Extraction | returns passages | exact match 0.0, token F1 0.012; three span-extraction gates all regress "contains" because recall@1 is ~28% (2026-07-19) | Blocked on retrieval@1, cleanly characterized. |
| Symbolic modules | built | 2026-07-18 probe: math 5/5 smoke and live; reasoning 5/5 isolated, 0 live; planner 1/1 toy, silent no-op live; generation 0/3; dialog fails its own docstring example; zero tests on all five at build time | One of five live. Built under the old framing, never measured. |
| Ontology | loaded and consumed, thin | 237,739 edges of which 224,527 are Wikipedia-redirect aliases; 128 roots in 8 verb classes; verb classes cover 5.6% of sentences; `est` (1.24M sentences) in no class (2026-07-12); ReVo SINONIMO expansion moved recall@200 by +1 on 147 questions (2026-07-18) | The alias bridge is the one ontology payoff. Verb classes contributed exactly 0.0000 MRR. |
| Learned components | deferred by design | none in tree; `RerankStage` is a stub; the shadow-slot mechanism (#893, #894) is unbuilt | Phase 3 has not started. |
| Orchestration contract | Phase 0 about 70% | decoder, loud failure, default-off, golden traces, contract suite, StoreView, executable status all landed 2026-07-18/19 | Effectively complete: the only open issue in milestone 28 is the tracking epic #897 itself (#885 and #887 closed 2026-07-19). One hygiene item remains: the contract suite hardcodes the store column list (#911). |
| CI | not stated | no `.github/`; pre-commit lints script docstrings only | None. The September gate outage was invisible for this reason. |
| Gold data | ruler exists for QA, not for parser | QA 1,345 pairs, audited clean (~0.2% wrong gold); parser 151 dev-informed sentences; independent queue 0/200 reviewed; proper-noun eval 41 tokens | The parser track is gated on review nobody can do. |
| Documentation | single-owned status | `README.md` and `AGENTS.md` still say 5.39M sentences; `AGENTS.md` and `DESIGN.md` call `klareco/ontology/` dead, but `klareco/ontology.py` is live and imported by the parser; `DESIGN.md` carries three generations of "start here" | Stale in places, and too long to keep current by hand. |

### 5.2 Progress against the plan in `PLAN.md`

| Phase | Plan status | This review |
|---|---|---|
| 0. Core: enforce the thought (ms 28) | ~70% | Done for the default spine; only the epic #897 remains open in milestone 28. |
| 0.5. Substrate: rich AST reaches retrieval (ms 40) | parked 2026-09-08 | Untouched since 2026-07-20. This is where the QA numbers are. |
| 0.5-parser (ms 36 to 39) | current focus | Crash fixes done; LAS up; gold gate unmet; candidate research parked. |
| 1. MVP-1 single-turn QA (ms 29) | planned | Baseline recorded (26.6%). Exit requires span extraction, which requires retrieval@1. |
| 2. MVP-2 dialog (ms 30) | planned | Not started; `DialogStage` fails its own example. |
| 3. Dual-track slots (ms 31) | planned | Not started. |
| 4+. Admit the rest (ms 32) | planned | 20 open issues, none able to name a number. |

The honest summary: the foundation is most of the way built, the parser is good and improving, and the product has been at 26 to 29 percent answer accuracy since July because the two levers that move it (store rebuild and deep-band recall) are parked.

---

## 6. What is of value, and should be kept

1. **The merge gate and the ledger.** 127 rows, each with a method, a before, an after, and a decision. Most rows end in "do not ship" and the project treats that as a result. This is rare discipline, and it is the project's real intellectual product.
2. **The QA gold sets.** `qa_gold_v2` was built from externally authored trivia, so it cannot be circular with the parser's own frames (the F13 failure that invalidated every earlier set). It was audited at ~0.2% wrong gold. The band stratification (trivial, rerankable, deep) and the paired bootstrap turned a borderline reranker result into a confirmed one when the set grew from 210 to 500 (#847, #877).
3. **The parser as an artifact.** First published-quality numbers on the UD Esperanto treebanks; CoNLL-U export; morphological alternatives preserved; lossless storage; stand-off annotation layers bound to source hashes. It is a credible open-source Esperanto parser and morphological analyzer, and the UD community has no other.
4. **The ReVo-derived lexicon and ontology.** 12,663 curated roots, 8,709 hypernym edges, 40,230 senses, typed entity lists, all from a GPL source independent of the parser. This replaced hand lists and broke the "laundering loop" where the lexicon was harvested from the parser's own output.
5. **The orchestration contract.** Small, tested, and it found real bugs.
6. **The characterized residues.** Four stand up: inferential paraphrase (147 of 506 deep-band questions, synonym expansion null); the deep-band recall floor (8.4% answer accuracy, golds not in the pool); the PP-attachment coin flips; and cross-sentence coreference (17% deterministically resolvable at one sentence back, 80% ambiguous). One is under test (proper nouns). One was falsified in the project's favour (morphological ambiguity collapses almost entirely with a typed lexicon).
7. **The infrastructure lessons, codified.** Fail loudly; variance not population; the test set must not be downstream of the component it evaluates; a baseline nobody asserts against is a diary.

---

## 7. The key ideas, and what the evidence says about each

| Idea | What was claimed | What was measured | Verdict |
|---|---|---|---|
| Regular grammar makes morphology and POS deterministic | the 16 rules suffice | 32.0% of tokens ambiguous under rules plus lexicon alone (300k sentences, 2026-07-13); 1.4% after typed root classes; 0.099% after the forest; POS 95 to 97% scheme-adjusted | Holds, with the caveat that the collapse comes from a semantic root lexicon, not from the grammar. Guinard (2016) reached 98.9% with a Markov model over morpheme semantic classes against 94.4% for longest match; the same information, learned. |
| The accusative gives grammatical roles without word order | free subject and object | subject F1 86 / 98, object F1 81 / 92 (2026-09-07) | Holds. |
| The interrogative's case tells you the answer slot | "free, deterministic, and English cannot do it" | not implemented in retrieval or extraction (`DESIGN.md`; no read of the interrogative's `kazo` in the retriever, the rerank stages, or the extractor, verified 2026-10-17) | Untested. The project's signature lever is unexploited. |
| Attachment can be done by rule | in progress | LAS 70 / 82; nmod the largest error class; `al` splits 21 nmod to 18 obl in gold; 99.1% of recurring verb-plus-preposition pairs have the same resolved-or-ambiguous status in every document they appear in (2026-09-08), so the ambiguity is a property of the pair, not of the context | Partly. The last finding says the residue is a lexical-pair statistic, which is Hindle and Rooth's 1993 result. A corpus association table would be the deterministic-adjacent fix. |
| Function words must be excluded from the learned and query layers | embedding collapse; retrieval noise | +0.47 MRR when function words are stripped from the query (2026-07-17); the embedding-collapse number from 2025 did not survive | Holds for retrieval; it is also standard stopword practice. The collapse claim is plausible but unrecorded. |
| Index roots, not surface forms | morphology is the retrieval lever | surface versus roots, function words stripped in both: noise on two sets, worse on the third (2026-07-17) | Falsified for retrieval. |
| AST structure improves ranking | the AST is the universal contract | structure-anchored role match: oracle lifts 0 of 36 (2026-07-18); clause-frame reranker +0.0528 on a circular set, then significantly worse than BM25 on the honest set; arc-level tree reranker +0.0057 (CI excludes 0); ast_aware demoted | Structure buys small, band-local gains. Matches the TREC-era conclusion that linguistically motivated indexing is not needed for effective retrieval. |
| Lexical synonymy is the learned residue | "faking it with a list" | hand-seeded verb classes: 0.0000 MRR; ReVo SINONIMO on the 147 synonym questions: +1 recall@200, 0 at 5 and 20 | The residue is real but misnamed: it is converse relations and world knowledge ("owned" versus "sells"). Best-characterized residue in the project. |
| Proper nouns are irreducibly ambiguous | VISION's flagship example | 58% F1 ceiling so far, on 41 tokens; Cairo precision 100% | Unresolved for lack of data. The project correctly demoted the claim to "under test". |
| Symbolic reasoning is classical CS and should be attempted first | planner, inference, generators | built in two days in May 2026; never benchmarked; 4 of 5 dead or off | Unknown. Nothing in the evidence supports or refutes it. |
| Grounded, cited answers by construction | the system cannot assert what it did not retrieve | holds by design; citations render with an empty title because the retriever never selects `article_title` (#912) | Holds, with a cosmetic bug. |
| Grammatically correct output by construction | the deparser | broken (#801, #813) | False today. |
| Every thought decodable, every contribution attributed | contract rule 4 and 5 | decoder exists; attribution defaults to `[regulo]` because nothing learned runs | Holds trivially until a model enters. |
| Compositional embeddings (learned root plus deterministic grammar features) | ~500K parameters, generalizes to unseen combinations | built 2025-12 to 2026-03, pruned; no surviving number | Unknown. The idea has precedent (fastText subword vectors; morphological recursive networks) and would be the natural first latent-layer citizen. |

---

## 8. Is the idea sound? Precedent and research

### 8.1 Esperanto as a computational pivot: it has been tried

The Distributed Language Translation project (BSO, Utrecht, 1985 to 1990; Witkam 1987) used a disambiguated Esperanto as the interlingua of a multilingual MT system, arguing that Esperanto's regularity made it "the least remote from formalized systems" among natural languages. The project did not reach its goals. Its syntax component (Schubert's "metataxis") was workable; the hard problem was semantic disambiguation, which it attacked with a lexical knowledge bank and a word-expert system (Sadler 1989, DLT volume 5). That is, the first serious attempt at Klareco's premise ended at the same boundary Klareco has now measured: the grammar is cooperative, the lexicon and the world are not. This precedent belongs in `VISION.md`.

### 8.2 Deterministic Esperanto parsing has a measured ceiling

EspGram (Bick 2007) is a Constraint Grammar parser for Esperanto that annotated an 18.5-million-word corpus at 99.5% F-score for POS and 92.1% for syntactic function; the 2009 dependency layer reports a 4% error rate on its test corpus. The repository cites "approximately 96.5%" without a source; use Bick's own figures and note the different corpus and scheme. Three things follow. The next 15 to 20 LAS points are known to be reachable without learning. They were reached with a declarative rule formalism designed for exactly this job, by a professional computational linguist, over years. And Bick's own error analysis names PP attachment and coordination as the top error classes, which is where Klareco's residual errors sit too. Apertium's Esperanto modules (apertium-eo-en, built on cg3) and Bick's 2016 morphological lexicon with morpheme frequencies are the other deterministic reference points, and the project already uses Apertium as a morphology cross-check.

### 8.3 The UD treebanks are new and tiny, and nobody else parses them

Oya (UDW 2025) released UD_Esperanto-Prago (legal text, the Prague Manifesto) and UD_Esperanto-Cairo (the 20 Cairo grammar sentences) in UD v2.16, and noted that automatic parsing was not attempted because no Esperanto parser produced CoNLL-U. Klareco's parser fills that gap. It also means the ruler is 151 sentences: a 25-edge Prago regression is one percentage point, and the project's own ledger records that a real 182,000-sentence corpus fix (#871) was invisible on it. Growing independent gold is not optional for the parser line.

### 8.4 Structure in retrieval: the field's answer is "a little, locally"

Spärck Jones (1999), summarizing the TREC-era experiments, concluded that linguistically motivated indexing "is not needed for effective retrieval", that the 1980s experiments did not demonstrate its value, and that the much larger 1990s full-text tests did not either; a Smeaton study from the same period is reported to have found that matching whole syntactic analyses scored much worse than tf-idf. Modern zero-shot evaluations (BEIR, Thakur et al. 2021) keep BM25 as a strong baseline that dense retrievers beat mainly where lexical overlap fails. Klareco's measurements match this on both ends: rerankers add 0.005 to 0.02 MRR over BM25 on honest questions, and the deep band, where the question's words are not the answer's words, is flat for every deterministic arm. The thesis predicted that learning belongs there. So does the literature.

### 8.5 PP attachment: a known boundary

Hindle and Rooth (1993) resolved PP attachment at about 80% from verb-and-noun association counts over an automatically parsed corpus, against human agreement of about 85% (88% with context). Klareco found that for 99.1% of recurring verb-plus-preposition pairs, whether the attachment is resolvable is the same in every document the pair appears in: the ambiguity belongs to the pair, not to the context. That is the same phenomenon, and it suggests a concrete step that stays inside the project's rules: a lexical association table counted from the 4.6-million-sentence corpus is a table, not a model, and it would target the largest residual error class.

### 8.6 Controlled languages and formal grammars

Attempto Controlled English and the controlled natural language tradition (Kuhn 2014) exist because natural languages are ambiguous above the word, and Esperanto is a natural language in this sense: real text has attachment ambiguity, ellipsis, coordination scope, names, quotation, and code-switching. The project's central empirical finding, that regularity buys morphology and role marking but not syntax or meaning, is consistent with why controlled languages had to be invented. Grammatical Framework (Ranta) shows what a formal grammar for a planned language looks like when it is written declaratively; it is a reasonable reference point if the attachment logic is ever rewritten.

### 8.7 The neurosymbolic framing

The project is a disciplined test of Sutton's "Bitter Lesson" on the most favourable natural language available. So far the lesson holds above the word level and fails below it: morphology and role marking are genuinely free, and everything from attachment upward is either effort-bounded rule work or learned. Kautz's taxonomy of neurosymbolic systems and Marcus (2020) frame hybrid designs; Klareco's "dual-track slot" with shadow, enrich, and replace modes is a sound and unusually honest version of the idea, because it keeps the deterministic floor running for attribution. It is also unbuilt.

### 8.8 Compositional embeddings and function words

Subword and morpheme-aware embeddings (Luong, Socher and Manning 2013; Bojanowski et al. 2017) are established, and Esperanto's segmentable morphology makes the compositional design natural. The claim that function words cause embedding collapse is consistent with known anisotropy and frequency effects in embedding spaces (Gao et al. 2019; Ethayarajh 2019), but the project's own measurement of it did not survive the 2025 to 2026 purges. If embeddings return, record the number.

### 8.9 Large language models and Esperanto

Esperanto Wikipedia has over 380,000 articles; Esperanto is in MADLAD-400 and HPLT, and NLLB handles it well (Open Machine Translation for Esperanto, arXiv 2603.29345, SIGUL 2026). A frontier model will answer most OpenTDB trivia from memory. This does not undermine the thesis, which the documents correctly say is not a competition. It does mean that the end-to-end answer-accuracy number has no external meaning, and that the numbers that carry the thesis are the decomposed ones: recall by band, extraction given retrieval, parser LAS, and residue sizes. The project built an LLM head-to-head harness (`scripts/eval/eval_vs_llm.py`, 2026-05-27) and never recorded a row from it. One reference row would size the residue in absolute terms.

### 8.10 Where the thesis is weak

1. **The boundary is effort-dependent.** "Where deterministic computation stops" is really "where it stopped for this team with this budget". The project knows this (the parser protocol's oracle decomposition and candidate ceilings exist for exactly this reason) but it has no stated effort budget or stopping rule per capability, and it has a known external ceiling (EspGram) it has not matched. Without a budget, every residue claim is open to "you did not try hard enough", which is what happened to the proper-noun claim.
2. **The residue must be defined operationally.** `VISION.md` already warns that "a token where no signal exists" is unfalsifiable. The operational definitions are available: the gap between the normal arm and the gold-candidate oracle arm, and inter-annotator disagreement. Use them.
3. **The task is misaligned with the question.** Wikipedia trivia tests lexical coverage and world knowledge. If the question is what Esperanto's grammar buys, the benchmark needs strata where grammar is the discriminator: case-marked answer slots (Kiu versus Kiun), multi-clause sentences, negation, quantifiers, passives with agents, and coordination. The project built one such set once (the 60-question clause set) and it was circular. Build it from the external trivia instead.
4. **The map is not yet a deliverable.** The thesis says the accumulated residue characterizations are the output. They live in ledger rows and issue comments. Nobody outside the repository could read the map today.
5. **The "16 rules" story misdescribes the system.** `16RULES.MD` is a pasted chatbot answer (its first line is "That is an excellent historical and linguistic question!"). The sixteen rules cover phonology, morphology, and inflection; they say nothing about attachment, coordination, clause structure, or ellipsis, which is where 5,000 of the parser's 5,636 lines live and where all of its errors are. The honest statement is: Esperanto's morphology is closed and regular, its syntax is as open as any language's, and the parser is a hand-written dependency parser whose morphological front end is trivially correct and whose syntactic back end is an engineering project without a formalism.

---

## 9. Problems in how the work is done

1. **Burst, then audit.** Three times. The third happened after the merge gate was adopted, because the gate depends on a human running `pytest -m accuracy`, and for several days nobody could. The fix is throughput control, not another rule: no more than one capability change in flight, and a machine that runs the gate on every push.
2. **No continuous integration.** `pytest -m "contract or unit or accuracy"` needs no data and runs in well under a minute. A single GitHub Actions workflow would have caught the September collection failure on the day it happened. There is no `.github/` directory.
3. **One reviewer who does not read Esperanto.** Every linguistic judgment in the repository (gold labels, parser rules, the glosses in the review packet) is machine-made and machine-checked. The project's own protocol requires two independent annotators and adjudication for 500 to 1,000 sentences. That is not a tooling problem. It is a staffing problem, and the Esperanto community is unusually reachable (Akademio de Esperanto, UEA, the Esperanto Wikipedia editors, the Tekstaro maintainers, Bick's group at VISL).
4. **Documentation as a diary.** `CLAUDE.md` is 1,190 lines; together with `DESIGN.md`, `PLAN.md`, `AGENTS.md`, and `README.md` the operating documents run to about 2,800 lines, with the same rules stated in four places and status claims that have gone stale repeatedly (`README.md` and `AGENTS.md` still say 5.39M sentences; `AGENTS.md` and `DESIGN.md` call `klareco/ontology/` dead when `klareco/ontology.py` is live and imported by `parser.py`). The executable status block was the right answer; extend it and delete the prose it replaces.
5. **Tracker sprawl.** 16 open milestones, 122 open issues, 20 of them in a "Phase 4+" milestone whose issues by the project's own rule cannot be build tasks. The issue list no longer tells a newcomer what is next.
6. **Code that contradicts the rules.** `klareco/knowledge/gazetteers.py` and `synonyms.py` carry hand lists that `CLAUDE.md` says must not exist; they are imported by `unified_extractor.py`, which is on the live path (verified: it imports `place_names` and `person_indicators` through `klareco/knowledge/__init__.py` and uses them in its person and place checks). `whoosh_retriever.py` is 1,901 dead lines still exported. `unified_extractor.py` is 3,084 lines of pre-orchestrator code doing the extraction that the thesis says should be a contract stage. `parser.py` is a monolith with 69 functions.
7. **The store is a year behind the parser.** Every parser improvement since 2026-07-14 is invisible to retrieval. The plan to rebuild once, with the v2 format and a clause-level projection, exists (#807, #901 to #904) and is parked behind 21 oversized source rows and a decision about the queryable unit.
8. **The ruler for the parser cannot see progress.** 151 sentences, both development-informed, and the project itself measured that a real fix did not register. Parser work since September is regression evidence, not generalization evidence, and the project says so on every page. That is honest, and it is also a sign that the parser line has outrun its instrument.

---

## 10. Recommendations

### 10.1 Keep

- The merge gate, the ledger, the paired bootstrap, and the band-stratified gold sets. Do not relax them.
- The parser, its three views, the forest, storage v2, CoNLL-U export, and the annotation-layer contract.
- The orchestration contract and `tests/contract/` as the primary suite.
- The executable status block, and the discipline that a doc claim which can be a query is a query.
- The ReVo lexicon, the typed roots, and the ALIASO bridge.
- The research-spike habit of ending in a decision.

### 10.2 Redo

- **The production store, once.** Rebuild with v2 lossless blobs and a clause-level projection (the queryable unit should be the clause, given 31.8% multi-clause sentences and 40.9% of clause subjects invisible to the flat column). Segment or explicitly exclude the 21 oversized rows with provenance. Measure before and after on `rebaseline_500` with the arms that actually read the new structure (the memory notes are right that `A_bm25_baseline` would report nothing). Number to move: deep-band and rerankable-band MRR, then answer accuracy.
- **The parser gold set, with Esperanto speakers.** Use the existing 200-sentence queue plus the 1,000-sentence literary and news candidates. Pre-annotate with the parser and one or two LLMs, have two Esperanto-competent reviewers adjudicate, record agreement. The protocol in `docs/PARSER_EVALUATION_PROTOCOL.md` is good; it needs people.
- **The QA benchmark.** Backfill gold answer spans (#934 notes about 355 of 500 are already verbatim substrings, so this is cheap) so extraction becomes attributable. Add structure-sensitive strata drawn from the external trivia: accusative-marked answer slots, multi-clause, negation, passives with agents, quantifiers. Fill the under-powered question types (KIE 11, KIES 6, KIAM 36, #862).
- **The residue map as one document.** One table per capability: deterministic method, ceiling reached, oracle gap, residue size, evidence pointer, date, status (held, falsified, under test). This is the thesis deliverable and it should be the first thing a reader opens.
- **The documents.** Cut `CLAUDE.md` to the rules an agent needs (about 150 lines), keep `DESIGN.md` as the generated status plus the normative contract, keep `PLAN.md` as the sequence, delete the duplicated rule text from `AGENTS.md`, fix the stale claims, and replace `16RULES.MD` with references to PMEG and PAG.
- **The framing of the parser.** Describe it as a deterministic dependency parser with a closed morphological front end, state EspGram's figures as the known deterministic ceiling, and set an explicit budget: for example, continue rule work while an unseen stratum gains at least 0.5 LAS per bounded change (the protocol's own threshold), and stop at a stated date or effort, recording the remainder as residue with its oracle gap.

### 10.3 Stop

- **Building new capability modules.** Close or move to Discussions the 20 "Phase 4+" issues (truth maintenance, spatial reasoning, abductive reasoning, modality, set operations, STRIPS planning, and so on). None can name a number today. The May 2026 batch is the project's own cautionary tale.
- **Parser-candidate and global-decoder research** until independent gold exists. Already parked; keep it parked. The bounded candidate generator should default off (#933: four times the parse latency for zero LAS).
- **Writing process prose.** Every rule that matters is already written. New rules should be tests or CI jobs.
- **Carrying dead and contradictory code.** Delete `whoosh_retriever.py` and its exports, `debug_retrieval.py` (it constructs the dead retriever), and the `knowledge/` hand lists once `unified_extractor.py` reads the ontology for them. Retire or decompose `unified_extractor.py` into contract stages with tests.
- **Running 16 open milestones.** Four would do: Parser (accuracy and gold), Substrate (store rebuild), MVP-1 (single-turn QA), Research (spikes and the map).
- **Re-deciding the order.** The July plan said store first, the September decision said parser first. Pick the store rebuild next, because it is the only step that converts the last three months of parser work into a QA number, and because it is one shell command that already exists and has been rehearsed.

### 10.4 Start

- **A CI workflow** running `pytest -m "contract or unit or accuracy"` on every push, plus the script-version linter. Zero data needed. This closes the gap that let the September crash ship.
- **The unexploited deterministic levers,** each as a merge-gated change on `rebaseline_500`: the answer-slot constraint from the interrogative's case; a pronoun-subject penalty after voice normalization; participle and agent normalization; a negation penalty; specificity weighting of anchors. `DESIGN.md` lists all five as computable today from `ast_json` plus corpus statistics, and none of them reads the interrogative's `kazo` today (verified 2026-10-17).
- **A Hindle-and-Rooth attachment table** counted from the corpus for verb-preposition-noun triples, used to rank the PP alternatives the parser already records. Number: nmod and obl LAS on the fixtures now, on the independent set later.
- **The first learned component, in shadow mode.** Dense retrieval for the deep band, evaluated against the 147 inferential-paraphrase questions and the 38 truly irreducible ones, with the deterministic floor still running so the delta is attributable. This is the step the thesis exists to justify, and the ledger's 2026-07-19 scorecard already names it as the next lever. A multilingual encoder off the shelf is the cheap first arm; a small bi-encoder trained on the corpus is the thesis-consistent second arm.
- **An LLM reference row** on `rebaseline_500` using the existing harness, recorded in the ledger as a ceiling reference, not a competitor.
- **Esperanto-speaking collaborators,** even two, for annotation adjudication. Nothing on the parser line can be generalization evidence without them.
- **A single-change rule.** One capability change in flight at a time, merged only with its ledger row, with the CI gate green.

### 10.5 A sequenced 90-day plan

| Weeks | Step | Number that closes it |
|---|---|---|
| 1 | CI workflow; fix stale doc claims; commit the residue-map skeleton; record one LLM reference row | gate runs on push; map v0 exists |
| 1 to 2 | Recruit two Esperanto reviewers; send the 200-sentence queue with the review packet | reviewed sentences > 0 |
| 2 to 4 | Store rebuild: v2 blobs, clause projection, oversized rows handled, verified swap | rerankable and deep MRR before/after with the clause-reading arms |
| 4 to 6 | Gold spans backfilled; structure-sensitive strata built; answer-slot-from-case and the other four levers measured one at a time | extraction exact match becomes nonzero and attributable; per-lever MRR with CI |
| 6 to 8 | Attachment association table; measure on fixtures and on the first reviewed stratum | nmod/obl LAS |
| 8 to 12 | Dense-retrieval shadow arm on the deep band; det-versus-learned report | deep-band recall@20 delta, attributed |
| 12 | Residue map v1 and parser release notes | the deliverable |

### 10.6 Three strategic options

**A. Publish the map and the parser.** Consolidate the residue map, release the parser as an open-source Esperanto UD parser and morphological analyzer with its numbers on Prago, Cairo, and a new independently reviewed set, and write it up (a UD workshop or a Esperanto-studies venue would take it). Two to three months, needs annotators. Highest durable value per hour.

**B. Keep building the QA system toward learned components.** Store rebuild, structure-sensitive strata, the five deterministic levers, then dense retrieval in shadow. Four to six months. This is the line that tests the thesis end to end, and it is the line most exposed to the burst-and-audit pattern.

**C. Narrow to the parser.** Drop the QA system, keep the parser, the lexicon, and the UD tooling. Smallest surface, and it abandons the half of the thesis that is about meaning.

**Recommendation: A, then the first three steps of B.** The parser and the map are what the project has actually produced that nobody else has. The QA system is the instrument that produced the map, and it should be kept running exactly far enough to measure the one learned experiment the thesis has already earned: dense retrieval against a characterized deep-band residue, with the deterministic floor still running underneath it. If that experiment shows a large, attributable gain, the map has its most important entry. If it does not, that is a finding too, and the project's method will have been applied to its own favourite assumption one more time.

---

## 11. Appendix

### 11.1 What was verified live during this review (2026-10-17)

- `git log`: 717 commits, last 2026-09-09 00:02 -0400 (`ac58ea4`); authorship 707 / 10.
- `gh`: 122 open issues, 803 closed, highest issue number 942; 42 milestones, 16 open; open issues per milestone as listed in section 9.
- No read of the interrogative's `kazo` in `question_shape.py`, `question_classifier.py`, `deterministic_rerank.py`, `extractive_answering.py`, `duckdb_retriever.py`, `retrieve.py`, or `extract_generate.py`.
- `klareco/orchestrator/` including `stages/`: 2,490 lines.
- `unified_extractor.py` imports `place_names` and `person_indicators` from `klareco.knowledge`, which re-exports them from `gazetteers.py`.
- `pytest -m "contract or accuracy"`: 85 passed, 3 skipped, 10.35 s.
- `git check-ignore data/perf/bench_history.jsonl`: not ignored; tracked.
- No `.github/` directory; `.githooks/pre-commit` lints script docstrings only.
- Code size: `klareco/` 29,377 lines of Python; `parser.py` 5,636; `unified_extractor.py` 3,084; `whoosh_retriever.py` 1,901; `tests/` 12,546 lines; `scripts/` 112 Python and 37 shell files; `docs/` 2,696 lines plus `CLAUDE.md` 1,190, `DESIGN.md` 894, `PLAN.md` 319, `VISION.md` 258.
- Importers: `whoosh_retriever` exported by `klareco/rag/__init__.py` and constructed by `scripts/eval/debug_retrieval.py`; `klareco/ontology.py` imported by `parser.py` and `tests/test_ontology.py`; `knowledge/` hand lists imported by `unified_extractor.py`, `ast_semantic_ranker.py`, and the dead `whoosh_retriever.py`.
- On-disk artifacts: `duckdb_store.db` 44 GB, `whoosh_v2` 2.6 GB, five `duckdb_parser_candidate*.db` side stores (6 MB to 510 MB), `retrieval_ab.db` 2.0 GB, `alias_table.json` 8.5 MB.

### 11.2 Evidence pointers for the main numbers

| Number | Source |
|---|---|
| Prago LAS 70.06 / UAS 76.55 / Cairo 81.88 / 85.91; subject and object F1; error counts | `DESIGN.md` "Parser quality", `docs/PARSER_SEVEN_STEP_RESULTS.md`, ledger rows 2026-09-05 to 09-07 |
| First UD numbers (Prago LAS 28.8, Cairo 34.9) | ledger `ud_conllu_las_uas`, 2026-07-14 |
| July baseline (Prago 62.3, Cairo 66.4) and #871 invisibility | ledger "PARSER BASELINE on UD", 2026-07-19 |
| Morphological ambiguity 32.0% of tokens; Guinard comparison | ledger `morpheme_ambiguity`, 2026-07-13 |
| Apertium comparison 95.5% vs 47.9% coverage | ledger `reference_comparison_apertium`, 2026-07-14 |
| Proper-noun inference 27.6 to 58.0 F1 | ledger `proper_noun_inference`, 2026-07-13 |
| Candidate recall 76.45 / 75.84 and ceilings | `docs/PARSER_RESOURCE_PLAN.md`, ledger 2026-09-06/07 |
| QA scorecard 26.6% / 0.0 / 0.012 by band | ledger "MVP-1 SCORECARD on rebaseline_500", 2026-07-19 |
| Reranker MRRs on `qa_gold_v2` | ledger "#713/#736 BASELINES", 2026-07-17 |
| Deep-band taxonomy 147 / 136 / 99 / 82 / 29 | ledger "#854 SPIKE RESULT", 2026-07-17 |
| SINONIMO null result | ledger "#873 SINONIMO expansion", 2026-07-18 |
| Structure-anchored oracle negative | ledger "#866", 2026-07-18 |
| Span extraction blocked on retrieval@1 | ledger "#869 DEFINITIVE", 2026-07-19 |
| Symbolic stack 1 of 5 live | ledger "SYMBOLIC STACK", 2026-07-18 |
| Coreference 17 / 80 / 3 and proper-noun consistency 36.8% | ledger rows 2026-09-08 |
| Store shape and dark tables | `DESIGN.md` "The store", #901 to #908, 2026-07-20 |
| Ontology edge counts | `DESIGN.md` generated status, 2026-09-07 |
| Crash rates fixed | ledger rows for #927, #929, #936, #937, 2026-09-07 |
| Reviewer does not read Esperanto; UD treebanks are the entire published set | `scripts/eval/build_annotation_review_packet.py` docstring, 2026-09-07 |

### 11.3 Sources consulted outside the repository

Verified by search during this review:

- Bick, E. (2007). EspGram, Constraint Grammar parsing of Esperanto; 99.5% POS F-score, 92.1% syntactic function. https://edu.visl.dk/pdf/CorpusLinguistics2007_esp.pdf
- Bick, E. (2009). A Dependency Constraint Grammar for Esperanto; 4% error rate. https://edu.visl.dk/pdf/cg-workshop2009_dep.pdf and https://dspace.ut.ee/items/5874e3a2-58b4-418f-a1d9-0ef9cd3f7793
- Bick, E. (2016). A Morphological Lexicon of Esperanto with Morpheme Frequencies. https://aclanthology.org/L16-1171/
- Bick, E. Lingvohelpilo grammar checker. https://edu.visl.dk/lingvohelpilo/Lingvohelpilo_CALICO.pdf
- Guinard, T. (2016). An Algorithm for Morphological Segmentation of Esperanto Words. PBML 105. https://ufal.mff.cuni.cz/pbml/105/pbml105.pdf
- Witkam, T. (1987). DLT overview, MT Summit. https://aclweb.org/anthology/1987.mtsummit-1.9.pdf ; project summary https://en.wikipedia.org/wiki/Distributed_Language_Translation
- Sadler, V. (1989). Working with Analogical Semantics: Disambiguation Techniques in DLT. Foris. Table of contents: https://aclanthology.org/www.mt-archive.info/70/DLT-1989-Sadler-TOC.htm
- Oya, M. (2025). UD Treebanks for Esperanto as a Natural Language. UDW, SyntaxFest 2025. https://aclanthology.org/2025.udw-1.3 ; treebanks https://universaldependencies.org/treebanks/eo_prago and https://universaldependencies.org/treebanks/eo_cairo ; Oya (2026) https://aclanthology.org/2026.udw-1.6/
- Spärck Jones, K. (1999). What is the role of NLP in text retrieval? In Strzalkowski (ed.), Natural Language Information Retrieval. https://www.cl.cam.ac.uk/archive/ksj21/ksjdigipapers/nlirbook99.pdf
- Hindle, D. and Rooth, M. (1993). Structural Ambiguity and Lexical Relations. Computational Linguistics 19(1). https://aclanthology.org/J93-1005/
- Apertium Esperanto-English package (cg3 dependency). https://packages.debian.org/unstable/apertium-eo-en ; GF and Apertium resource sharing (Détrez, Sánchez-Cartagena, Ranta 2014) https://aclanthology.org/L14-1525/
- Open Machine Translation for Esperanto (2026). SIGUL 2026, arXiv 2603.29345. https://arxiv.org/abs/2603.29345
- Colelough and Regli (2025). Neuro-Symbolic AI in 2024: A Systematic Review. https://arxiv.org/abs/2501.05435

Cited from background knowledge and not re-verified in this review: Sutton (2019) The Bitter Lesson; Kuhn (2014) A Survey and Classification of Controlled Natural Languages, Computational Linguistics 40(1); Thakur et al. (2021) BEIR; Bojanowski et al. (2017) Enriching Word Vectors with Subword Information; Luong, Socher and Manning (2013) Better Word Representations with Recursive Neural Networks for Morphology; Gao et al. (2019) Representation Degeneration Problem; Ethayarajh (2019) How Contextual are Contextualized Word Representations; Marcus (2020) The Next Decade in AI; Kautz (2022) The Third AI Summer; Schubert (1988) Metataxis; Hana (1998) Two-level Morphology of Esperanto; Ranta, Grammatical Framework.
