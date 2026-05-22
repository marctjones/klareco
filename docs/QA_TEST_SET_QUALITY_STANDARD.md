# Klareco Q&A Test Set Quality Standard

> Written 2026-05-21 after several sessions of cleanup. The legacy
> 4-set baseline (140 pairs) had **0/140 survive discriminability
> audit**. The `gold_anchor_50` set had **0/16 retrievable even with
> the answer entity handed to the retriever.** This document exists so
> we never spend a week generating noise again.

## 1. Goals — what each test set is for

A Klareco test set is a **measurement instrument**, not a corpus
sample. Every question must let us learn something specific. Each set
has one stated purpose; sets that mix purposes are why we kept getting
useless numbers.

| Set purpose | What it answers | Honest target |
|---|---|---|
| **Pipeline capability** (corpus-derived, properly specified) | "Given that the answer is in the corpus and the question is unambiguous, can the pipeline find and extract it?" | ≥ 70% answer accuracy, ≥ 90% R@10 |
| **Honest ceiling** (real trivia, externally sourced) | "Given a question someone would actually ask, can the pipeline answer it?" | This is the **truth**; whatever we get is the baseline. Expect 40–55% today. |
| **Retrieval discriminability** | "For a given question, does the source sentence rank in BM25 top-K?" | ≥ 95% in top-50; failures expose corpus coverage holes, not pipeline bugs |
| **Regression** | "Did the last change break anything?" | 0 regressions on a frozen subset |
| **Diagnostic / per-shape** | "Where in the pipeline does this question type break?" | Used for analysis, not for headline metrics |

**Do not combine purposes in one file.** The 322-pair eval gave us
63% accuracy that was actually ~71% with ambiguous person-anchor
pairs filtered out — a 8-point measurement error from mixing
properly-specified questions with broken ones.

## 2. Failure-mode bestiary — concrete examples we've paid for

Every rule below traces to a real failure. Keep these in mind when
spot-checking a set.

| # | Failure mode | Concrete example | Root cause |
|---|---|---|---|
| F1 | **First-name-only anchor** | `Kie naskiĝis Béla?` (5 different Bélas in corpus) | Generator extracted `subjekto.kerno.plena_vorto` (head word) instead of the full multi-token name span |
| F2 | **PP-governed WHO answer** | `Kiu publikis «Odbudowa państwa»? → Głos` — but source is `En "Głos" en 1983 publikigis…` — Głos is the venue, not the agent | Generator's `_kiu_answer_constraint` accepted any subject; didn't check for governing preposition |
| F3 | **Multi-PP confusion** | `Kiam naskiĝis Kálmán Balla? → 1990` — but source contains both birth (1949) and death (1990) dates | `_find_first_locative_after_verb` didn't enforce verb-proximity; grabbed the wrong PP |
| F4 | **Function word as proper noun** | `Kie reveni Anstataŭ? → Kursk` — `Anstataŭ` ("instead-of") is a function word the parser capitalized at sentence start | No function-word denylist on `_looks_namelike()` |
| F5 | **REDIRECT stub pollution** | `Kiu verkis «Rosenkavalier»?` returns `ALIDIREKTI 5039 Rosenkavalier` (Wikipedia redirect, content-free) | No filter on `ALIDIREKTI` prefix at index time. 3/3 misses on the diversified-51 set were redirects. |
| F6 | **Tautological "definition" question** | `Kio estas komunumo? → komunumo` | Generator picked answer-type token as both Q-subject and A — circular |
| F7 | **Verb voice mismatch** | Template `publikis` synthesized from intransitive source `naskiĝis` (was-born) | Generator synthesized verb forms instead of using `source.verbo.plena_vorto` |
| F8 | **KIEL accepting non-manner adverbs** | `Kiel verkis Pagnier? → sube` ("below") | `-e` adverb extractor accepted temporal (`daŭre`), discourse (`fine`), and locative (`sube`) particles |
| F9 | **No-signal pairs (legacy)** | `Kiu fondis Esperanton? → (no source_sentence_id, no expected_answer)` | Hand-written pairs with no provenance, no recoverable answer — 140/140 of the legacy sets |
| F10 | **Object truncation** | `«Przystanek Woodstock» → Przystanek` (only first token) | Same root cause as F1, applied to objects |
| F11 | **All-person bias** | 321/322 anchors are humans; no places/events/organizations/languages | Generator's `_looks_namelike` overwhelmingly picks human first names; no `EntecaTipo`-typed dispatch |
| F12 | **Source too thin** | `Kio estas Adelsdorf? → komunumo` with 5-word source `Adelsdorf estas komunumo en Germanio` | Mechanically valid but trivially shallow — generator should prefer richer source sentences when multiple are available |
| F13 | **Parser-bug contamination** | Bugs #1, #2, #3, #4, #5, #6, #11 each produced bad pairs from the same generator (fronted-PP role, missing prepositions, multi-token truncation, ki-correlative misparse, OSV coord, neniam negation, kiam-clause subject theft) | Generator was downstream of parser; every parser bug rippled into the test set. |

## 3. Quality standards — the eleven rules

These are **enforcement rules**, not aspirations. Every rule has a
mechanical check; where it can't, the rule says "human spot-check
required."

### R1. Anchors must be rigid designators
- The named entity in the question must be **globally unique within
  the corpus** (or contextually unique with disambiguating qualifiers).
- Single first names are **never sufficient.** A first-name anchor
  must include surname or other disambiguator.
- Quoted works `«Title»` are preferred — they're naturally unique and
  BM25 handles quoted strings well.
- **Check:** generator must read `multi_token_entities` from the AST,
  not `subjekto.kerno.plena_vorto`. Auditor must verify the anchor
  span has ≥ 2 tokens for person anchors *unless* the source contains
  no other person of the same first name (rare).

### R2. Answer must be grammatically governed correctly
| Question type | Answer must be… | Must NOT be… |
|---|---|---|
| KIU (Who) | Subject NP, not preceded by any preposition in source | Inside an `en` / `de` / `pri` / `anstataŭ` PP |
| KIE (Where) | NP inside a **locative** PP (`en`, `ĉe`, `sur`, `apud`, `tra`, `super`, `sub`) governed by the question's verb | A temporal PP, a non-locative PP, or the subject |
| KIAM (When) | A date/year token, OR an NP inside a **temporal** PP (`en` + year, `je`, `dum`, `antaŭ`, `post`) | A locative PP misread as temporal |
| KIAL (Why) | A `ĉar` / `pro` / `pro tio ke` clause | The clause's object, or any unrelated subordinate clause |
| KIEL (How) | A **manner** adverbial: `per` + NP, OR a manner `-e` adverb | Temporal (`daŭre`, `ĉiujare`), discourse (`fine`, `kadre`), or locative (`sube`, `supre`) `-e` words |
| KIO (What) | Predicate NP after `estas` / `estis` | A subject NP from a different clause |

- **Check:** `audit_qa_pairs.py` runs `_is_pp_governed()` and per-type
  role-alignment checks.

### R3. Use source verb surface form — do not synthesize
- Templates must use `source.verbo.plena_vorto`, not a synthesized
  active form derived from semantic role.
- `naskiĝis` (was-born) stays `naskiĝis`; do not rewrite to `naskis`
  (gave-birth-to).
- **Check:** generator unit test on a `naskiĝis` source must produce a
  question containing the original verb form.

### R4. Verb-proximity for ambiguous PPs
- When source has multiple PPs of the same type (birth-place +
  death-place, founding-year + dissolution-year), require the
  target PP to appear within **10 tokens after the verb** the question
  references.
- **Check:** `_find_first_locative_after_verb` (and temporal
  equivalents) enforce a `max_dist=10` parameter.

### R5. Function-word denylist for namelike heuristics
- Esperanto function words capitalized at sentence start are
  consistently mislabeled as `propra_nomo`.
- Maintain a denylist; reject these from any anchor / answer slot.
- **Minimum members:** `Anstataŭ`, `Kaj`, `Tamen`, `Sed`, `Krom`,
  `Pro`, `Ekde`, `Malgraŭ`, `Aŭ`, `Do`, `Nu`, `Ja`, `Plu`.
- **Check:** `_looks_namelike()` returns False for every token in
  the denylist regardless of case.

### R6. Filter corpus noise at the source
- Index-build pipeline must filter `text NOT LIKE 'ALIDIREKTI%'`
  before Whoosh ingest.
- Disambiguation stubs and category lists should be filtered with
  similar prefix rules.
- This is a retriever-side fix that also cleans the candidate pool
  questions are generated from.
- **Check:** post-index sanity query — `SELECT COUNT(*) FROM
  sentences WHERE text LIKE 'ALIDIREKTI%'` must return 0 *in the
  Whoosh index*, not in the DuckDB store (where they're preserved
  for ontology work).

### R7. Question must be discriminable
- For a corpus-derived set, the source sentence must rank in the
  BM25 top-K (default K=50) for a query of the **question text alone**
  (no answer, no source-text leakage).
- A question that isn't discriminable isn't necessarily a *bad
  question* — it's a *corpus-coverage gap* — but it doesn't belong in
  a pipeline-capability set.
- **Check:** `audit_discriminability.py --top-k 50` must pass for
  every pair in a capability set.

### R8. Provenance — `source_sentence_id` is required
- Every pair must carry `source_sentence_id` and `source_sentence_text`.
- The auditor verifies the ID resolves to the stated text in the
  current DuckDB (drift detector — catches when a re-parse changes
  sentence IDs).
- **Check:** `audit_qa_pairs.py --retrievability-check` reports
  `STALE_SID` for any pair where the resolved text doesn't match.

### R9. Answer must appear verbatim in source
- The expected answer (or one of its `expected_keywords`) must be a
  diacritic-fold-substring of `source_sentence_text`.
- Exception: numeric answers may be normalized (`1987` matches
  `mil naŭcent okdek sep` only after explicit norm).
- **Check:** `audit_qa_pairs.py` runs `_answer_in_source()`.

### R10. Diversify by entity type, not just question type
- Track the entity-type distribution of anchors using `EntecaTipo`:
  `persono`, `loko`, `evento`, `organizaĵo`, `verko`, `lingvo`, `tempo`.
- A capability set should have **no single entity type > 60%** of
  anchors. The legacy sets had ~99% `persono` — that's why we never
  saw the pipeline's geography behavior.
- **Check:** `audit_qa_pairs.py --entity-type-report` prints the
  distribution; fails the set if any type exceeds the ceiling.

### R11. Fail-closed audit gate at write time
- The generator must call the auditor on every candidate pair **before
  appending to the output JSONL**.
- A pair that fails any of R1–R10 is dropped, not flagged. The output
  file is, by construction, audit-clean.
- **Check:** `generate_qa.py` calls `audit_qa_pairs.audit_pair()` and
  only writes pairs with status == PASS. The audit step is not
  optional and not a separate post-processing run.

### R12. Trivia-caliber — would a quiz show ask this?
The user's stated bar: *"would an outside reviewer ask 'is this real
trivia?'"* A pair must clear that bar — mechanical validity is not
enough. Compared to a published English trivia book:

- **Notable subject.** The anchor must be a thing a knowledgeable
  person could plausibly care about. Operationally:
  - Anchor has a Wikidata QID with ≥ 1 sitelink to a major-language
    Wikipedia (en / de / fr / es / ru / zh / ja), **or**
  - Anchor appears in `data/eo_wikipedia_notable_people.json` (we
    built this for exactly this filter), **or**
  - For places: anchor's Wikipedia article ≥ 8 content words *and*
    population ≥ 10,000 (filters out the "1,200-person Italian
    commune" trivia).
- **Non-tautological.** The answer must NOT be the same word class
  the question word demands as a definition.
  - Bad: `Kio estas komunumo? → komunumo` (F6)
  - Bad: `Kio estas urbo? → urbo`
  - Acceptable: `Kio estas Brazilo? → lando` (the answer narrows the
    category)
- **Source-sentence depth.** Source must have ≥ 8 content words and
  contain at least one additional notable token (date, place,
  person, work) beyond the anchor and answer.
  - Bad: `Adelsdorf estas komunumo en Germanio` (5 words — F12)
  - Good: `Lev Tolstoj naskiĝis en 1828 en Jasna Poljana en familio
    de rusa nobelaro` (rich context, multiple anchors usable)
- **English-trivia transferability.** If you mentally replaced the
  Esperanto entities with English equivalents, would the question
  survive in a published trivia book? If the answer is obviously
  "no" (e.g. obscure tiny village), reject.

**Check:** `audit_qa_pairs.py --trivia-caliber` runs:
- Wikidata-QID lookup on the anchor
- Source content-word count
- Definitional-circularity check (`answer in question_text` after
  morphological reduction)
- Population check for `loko`-typed anchors

**Human spot-check:** the 20-sample stratified Stage 3 review applies
the "real trivia?" gut check explicitly per pair. Track accept-rate;
target ≥ 90%.

### R13. Esperanto language quality
Both the question AND the answer must be in clean, idiomatic
Esperanto. This catches generator artifacts and translation drift.

**Mechanical checks:**
- **Parser-clean.** `klareco.parser.parse(question)` returns no word
  with `parse_status` in `{unknown_root, proper_name_unknown, error}`,
  except for the intentional named-entity anchor.
- **Diacritic system consistency.** Question uses **only** ĉĝĥĵŝŭ.
  Reject x-system (`cx`, `gx`, `ux`) or h-system (`ch`, `gh`) — any
  mixing inside one pair indicates a generation or copy-paste bug.
- **Roots in vocabulary.** Every content-word radiko is in ReVo, the
  Fundamento, or attested ≥ 10 times in the corpus. Catches
  fabricated roots from translation pipelines.
- **Well-formed interrogative.** Question ends with `?`; the
  ki-correlative is sentence-initial OR inside a fronted PP (`En kiu
  jaro…`); no two interrogatives in one clause.
- **Accusative agreement.** Direct object of a transitive verb in the
  question is marked `-n` (or is a quoted work which is invariant).
  E.g. `Kiu verkis Faŭston?` ✓ — `Kiu verkis Faŭsto?` ✗.
- **Verb tense matches the claim's time.** Use `estas` (present) for
  stable facts (`Kio estas Brazilo?`); past `-is` for completed
  actions (`Kiu fondis Esperanton?`); future `-os` only for genuinely
  future-referring questions.
- **Preposition correctness for time/place.** `en + year` for year
  reference (`en 1887`), not `je`; `je` for clock-time only. Locative
  `en + place`, not `ĉe + country`.

**Human spot-check (NOT mechanizable):**
- Reads naturally to a fluent Esperanto speaker, not like word-for-word
  translation from English.
- No anglicisms-of-construction (e.g. `Kio estas la fakto pri X?`
  is calqued English "what's the fact about X" — broken).
- No false-friend roots (`librejo` is "bookshop", not "library" —
  library is `biblioteko`).
- Register consistent: encyclopedic / neutral, not chatty, not poetic.

**Check:** `audit_language_quality.py --in <set>.jsonl --strict`
runs all mechanical checks; spot-check covers the rest.

**Common artifacts we've seen and want to catch:**
| Pattern | Bad | Good |
|---|---|---|
| Calqued English | `Kio estas la fakto pri Esperanto?` | `Kio estas Esperanto?` |
| Missing accusative | `Kiu verkis Faust?` | `Kiu verkis «Faust»?` |
| Wrong time prep | `Kiu naskiĝis je 1970?` | `Kiu naskiĝis en 1970?` |
| x-system bleed | `Kio estas Cxehxio?` | `Kio estas Ĉeĥio?` |
| Fabricated root | `Kio estas pizziano?` (not a word) | (reject) |
| False-friend | `Kie estas la granda librejo?` (means "bookshop") | `Kie estas la granda biblioteko?` |

### R14. Corpus-coverage robustness — multiple supporting passages
R8 requires a `source_sentence_id`. R14 adds: a real-trivia pair
should be answerable from **more than one** corpus passage, not just
the one used to generate it.

- For a question, run a BM25 query (question text only) on the
  corpus; fetch top-50; count how many of those passages contain any
  `expected_keywords` (diacritic-fold substring match).
- This is the *corpus-support count*. It's a measure of how
  *findable* the answer is in our index — independent of question
  quality.
- **Capability set target (calibrated):** ≥ 2 supporting passages,
  OR support_count == 1 AND source-rank ≤ 5. Rationale: empirically
  on synthetic_who_trivia_v2 only 9.8% of pairs achieve ≥ 3
  supporting passages — our corpus genuinely doesn't repeat itself
  much for quoted-work entities. ≥ 2 still rejects the single-source
  brittle case while not over-pruning.
- **Aspirational ≥ 3:** record this number too; it indicates pairs
  the pipeline can answer from multiple independent passages — the
  most robust set.
- **Honest-ceiling (real-trivia) set:** no minimum; record the count
  and report it. Pairs with 0 support are corpus-coverage gaps
  worth logging — they tell us what content the corpus is missing.

**What this catches that R7 (discriminability) misses:**
- R7 checks "is the source in the top-K of BM25?" — a yes/no
  retrievability gate.
- R14 checks "how *deeply* does the corpus cover this fact?" —
  the difference between a corpus that knows one thing once vs a
  corpus that knows it many ways.

**Check:** `audit_corpus_coverage.py --in <set>.jsonl --min-support 3`
on the capability set; same script with `--report-only` on the
honest-ceiling set.

### R15. Topic and structural diversity (extends R10)
R10 caps any single entity type at 60%. R15 adds active targets for
question-type, topic, and difficulty distribution.

**Question-type targets** (% of a 100-pair capability set):

| Type | Target | Notes |
|---|---|---|
| KIO (what) | 25% | mostly predicate-NP definitional and entity-classification |
| KIU (who) | 25% | agentive WHO, WHO-of-work (quoted title) |
| KIE (where) | 15% | locative-PP-governed answers |
| KIAM (when) | 15% | date/year answers; mix of `naskiĝis`, `fondiĝis`, `okazis` |
| KIAL (why) | 5% | `ĉar`/`pro` clause answers |
| KIEL (how) | 5% | manner adverbial or `per`-NP answers |
| KIOM (how-many) | 5% | numeric answers |
| ĈU / KIES / KIA | 5% | yes-no, possessive, attributive |

**Topic targets** (% of a 100-pair set), bucketed by anchor's
ontology class (`HAVAS_ENTECAN_TIPON`) or Wikidata top-category:

| Topic | Target | Anchor entity types |
|---|---|---|
| Geography | 15% | `loko` cities/countries/rivers/mountains |
| History | 15% | `evento`, historical `persono` |
| Science / nature | 15% | scientific `persono`, taxonomic anchors |
| Arts / literature / film | 15% | `verko`, artistic `persono` |
| Esperanto language & culture | 10% | Esperanto-movement-specific entities |
| Sports / games | 10% | sport `persono`, sport events |
| Technology | 10% | `organizaĵo` tech-cos, inventors |
| Other (religion, politics, food, language) | 10% | catch-all |

**Difficulty targets** (% of a 100-pair set):

- Easy (high Wikidata sitelink count ≥ 30): 30%
- Medium (sitelink count 5–29): 50%
- Hard (sitelink count 1–4, still notable): 20%

**Structural diversity:**
- Mix of pattern shapes: definite-description (`la ĉefurbo de X`),
  quoted-work (`«Title»`), bare-anchor (`X`), fronted-PP
  (`En kiu lando…`).
- No more than 25% of pairs share the same template-id.
- No more than 5 pairs anchored on the same proper noun.

**Check:** `audit_qa_pairs.py --diversity-report` prints the
distribution and fails the set if any bucket exceeds 2× target or
falls below 0.5× target.

## 4. How to test — the gate stack

Run these in order. Stop at the first failure; fix it; re-run from
the top. **Do not** advance to the next stage when the current one is
red.

### Stage 0 — Generator unit tests (sub-second)
```
python -m pytest tests/test_qa_generators.py -v
```
- Each generator type has at least 5 frozen fixtures (source sentence
  + expected output pair). Tests fail if generation drifts.
- Includes regression fixtures for every failure mode F1–F12 above
  (e.g. a `Béla Kovács … Béla Buzogány` sentence must produce
  full-name anchors, not "Béla").

### Stage 1 — Per-pair mechanical audit (seconds)
```
python scripts/eval/audit_qa_pairs.py --in <set>.jsonl --strict
```
- Runs R1–R10 mechanically per pair.
- Target: 100% PASS. Anything below 100% means a generator regression.
- The auditor is the source of truth — if a question feels wrong but
  passes audit, **add a new check to the auditor** before adding the
  question to the set or before "fixing it by hand."

### Stage 1.5 — Language-quality audit (seconds)
```
python scripts/eval/audit_language_quality.py --in <set>.jsonl --strict
```
- Runs R13 mechanical checks: parser-clean, diacritic system,
  recognized roots, well-formed interrogative, accusative
  agreement, tense/preposition correctness.
- Target: 100% PASS. A pair that fails here is grammatically broken
  Esperanto; no amount of retrieval engineering rescues it.

### Stage 1.7 — Trivia-caliber + corpus-coverage audit (seconds)
```
python scripts/eval/audit_qa_pairs.py --in <set>.jsonl --trivia-caliber
python scripts/eval/audit_corpus_coverage.py --in <set>.jsonl --min-support 3
```
- R12 (notability, non-tautology, source depth) and R14 (≥ 3
  supporting passages for capability set).
- Capability set: 100% PASS on both.
- Honest-ceiling set: skip R14; run R12 and report results.

### Stage 2 — Discriminability audit (seconds)
```
python scripts/eval/audit_discriminability.py --in <set>.jsonl --top-k 50
```
- For a *capability* set: target ≥ 95% in top-50.
- For an *honest-ceiling* set (real trivia): no target — record the
  rate and report it; missed pairs are corpus-coverage gaps to log,
  not pair-quality failures.

### Stage 3 — Spot-check sampling (15 minutes, human)
- Take a stratified sample: 20 pairs balanced across question types
  and entity types.
- For each: ask **"would an outside reviewer accept this as a real
  trivia question?"** That's the user's stated criterion.
- Expect ~5-9 percentage-point gap between mechanical PASS rate and
  human-accept rate. If the gap is bigger, the auditor is too
  permissive — add a check.

### Stage 4 — Extractor-isolation oracle test (one-time per set)
```
python scripts/eval/oracle_isolation_test.py --in <set>.jsonl
```
- For each pair: extract answer from source sentence directly (oracle
  passage), compare to expected.
- Failures here are **extractor bugs**, not retrieval issues. Separates
  the "we found the right passage but didn't pull the right answer"
  problem from the "we didn't even find the right passage" problem.

### Stage 5 — End-to-end pipeline eval (minutes)
```
python scripts/eval/evaluate_extractive_qa.py --test-set <set>.jsonl
```
- This is the headline measurement. Only run it on sets that passed
  stages 0–4.
- Report: answer accuracy, R@1/5/10, MRR, p50/p95 latency.
- Append to `data/perf/bench_history.jsonl` so we can spot regressions.

## 5. Target portfolio — what 100 questions should look like

We've spent days getting to 100 because we kept building sets that
mixed purposes. Concretely, "100 reasonable test questions" =
**three small purpose-specific sets**, not one giant mixed one:

| Set | Size | Purpose | Composition |
|---|---|---|---|
| `capability_100.jsonl` | 100 | Pipeline capability — corpus-derived, properly-specified | Per R15 distribution: KIO 25, KIU 25, KIE 15, KIAM 15, KIAL 5, KIEL 5, KIOM 5, other 5. Entity types: ≤ 50% person, ≥ 30% place, balance event/org/work/language. Topic mix per R15: ≤ 20% in any single bucket. Difficulty mix per R15: 30/50/20 easy/medium/hard. All pairs PASS Stages 1, 1.5, 1.7, 2. |
| `trivia_real_50.jsonl` | 50 | Honest ceiling — externally sourced | Hand-curated from real Esperanto trivia sources (Vikipedio "Ĉu vi sciis?", published Esperanto trivia, translated trivia-book questions). PASSES Stages 1, 1.5 and R12 trivia-caliber. R7/R14 are reporting-only (corpus-gap log, not gate). |
| `regression_frozen_30.jsonl` | 30 | Regression detection | Cherry-picked from `capability_100` — questions the pipeline answers correctly today. Any regression on this set blocks merging. |

Total: 180 pairs, three numbers we can read independently and
together.

## 6. What "done" looks like for the test set effort

The test set is done when:

- [ ] `capability_100.jsonl` exists with 100 pairs, all PASS Stage 0–2.
- [ ] Stage 3 human spot-check returns ≥ 90% accept on the
      stratified sample.
- [ ] Stage 5 pipeline eval produces a stable number that doesn't move
      ± 3% across three identical runs.
- [ ] `trivia_real_50.jsonl` exists with 50 hand-curated pairs and a
      documented pipeline accuracy (whatever the number is).
- [ ] `regression_frozen_30.jsonl` is committed and `evaluate_extractive_qa`
      reports 100% on it (by construction — we chose pairs the
      pipeline gets right today).

When all four are true, we have an honest measurement instrument and
can spend the next month making the pipeline better instead of
arguing about whether the number is real.

## 7. Anti-patterns — things to stop doing

1. **Editing test pairs by hand to "fix" individual failures.** If a
   pair is bad, the generator is wrong. Fix the generator; regenerate.
   Hand-edits drift away from the rules and accumulate over time.
2. **Mixing corpus-derived and real-trivia pairs in one file.** They
   measure different things; their accuracy numbers aren't comparable.
3. **Running pipeline eval on a set that hasn't passed Stages 1-2.**
   The number you get back is uninterpretable; you can't tell
   whether changes hurt the pipeline or just probed a different
   noise pattern.
4. **Adding a "TODO: fix this pair" or comment-marker in the JSONL.**
   Either it's PASS or it's dropped.
5. **Treating audit PASS as semantic correctness.** PASS = no
   *known* mechanical defect. R3 stage spot-checks are the only thing
   that catches semantic drift.
6. **Reusing the legacy `qa_test_set_50` / `gold_anchor_50` /
   `general_knowledge_30_keyed_v2` files.** They're broken (0% audit
   survival). Archive, don't reference.

## See also

- `data/test_sets/discriminability_audit_2026-05-19.md` — the audit
  that killed the legacy sets
- `data/test_sets/suspicion_diversified_2026-05-20.md` — example
  Stage-2-grade output (REVIEW vs CLEAN buckets)
- `scripts/eval/audit_qa_pairs.py` — Stage 1 enforcer
- `scripts/eval/audit_discriminability.py` — Stage 2 enforcer
- `scripts/eval/evaluate_extractive_qa.py` — Stage 5 measurement
