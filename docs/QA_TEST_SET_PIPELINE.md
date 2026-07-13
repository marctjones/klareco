# The Q&A Test-Set Pipeline — Machine-Assisted, Human-Adjudicated

> **Status:** design, adopted 2026-07-13. Implements milestone
> [#15 — Gold Q&A Corpus](https://github.com/marctjones/klareco/milestone/15), EPIC #792.
> Construction rules (R1–R17) live in `QA_TEST_SET_QUALITY_STANDARD.md` and are
> binding. This document says *how the pipeline produces sets that satisfy them*
> — and, crucially, where machine learning is allowed to help.

## Why this document exists

Klareco's benchmark is its ruler, and everything downstream — the merge gate
(#784), the reranker (#780), the retroactive audit of the symbolic layer (#785)
— is calibrated against it. A ruler with garbage on it measures nothing.

We have now failed at this twice, in two *different* ways, and both failures are
instructive:

- **The legacy sets** (`qa_test_set_50`, `gold_anchor_50`, …) were hand-written
  with no provenance: **0/140 survived the discriminability audit**, and 0/16 of
  the `gold_anchor_50` misses were retrievable even with the answer entity handed
  to the retriever. They were unanswerable from our corpus.
- **The synthetic sets** are answerable but were built by *string surgery* —
  grab a span, glue on an accusative `-n`, drop it in a template. That produced
  `Kiu venkis Rorke's Driftn?` and `Kiu reĝisoris Théâtre des Variétésn?`, and
  the hand-written language auditor passed them **8/8** until it was tightened
  (#791). It will have another hole. It always will.

The lesson is not "write a better auditor." It is:

> **You cannot automate perfect Esperanto.**
>
> Automation gives scale, answerability, and discrimination. All three are
> mechanizable. Grammaticality is not. A string-transforming generator has an
> infinite tail of ungrammatical output, and an auditor is a net you keep finding
> holes in. That loop does not converge to gold.

So we stop trying to make the generator produce gold. The generator produces
**silver** — answerable and discriminating. Machines then *triage*. Humans
**adjudicate**. That is what makes a pair gold.

---

## The bright line: construction time vs scoring time

**Machine learning may be used to BUILD a test set. It may never be used to
SCORE one.**

This is not a stylistic preference. It is what keeps the merge gate meaningful.

| | Construction time | Scoring time |
|---|---|---|
| LLM / MT / embeddings | ✅ **Allowed** | ❌ **Never** |
| Output | A **frozen artifact** (a test set, an alias list, a review verdict), committed to git | A number in `bench_history.jsonl` |
| Reproducibility | The artifact is fixed forever; scoring against it is pure deterministic string comparison | Would depend on model version, temperature, sampling |
| Audit | Human spot-check gates the artifact before it is frozen | Nothing to audit — the judge *is* the metric |

If an LLM scored answers at eval time, a number moving in `bench_history` could
mean the reranker improved — **or that the judge drifted**, or the model version
changed, or it was sampled differently. The entire point of the merge gate
(#784) is that a moved number is *evidence*. A non-deterministic scorer makes
the thesis unfalsifiable, which is the exact failure mode this project has spent
months climbing out of.

### Why this does not violate the thesis

It looks like a contradiction. It isn't.

The thesis is a claim about **what is inside the pipeline** — where the
deterministic/learned boundary falls *in the system under test*. **The measuring
apparatus is not the system under test.** Using a telescope to observe a star
does not make the star a telescope; particle physicists use machine learning to
reconstruct detector events without that changing the physics claim.

What *would* violate the thesis: an LLM inside `Orchestrator.answer()`, or an
LLM deciding whether an answer is correct at eval time. Neither is proposed here,
and both remain banned.

---

## The pipeline

```
  A. SOURCE            grounded by construction — the fact is in OUR corpus
        │
  B. CONSTRUCT         minimal transformation — inherit the source's grammar
        │
  C. MACHINE VERIFY    deterministic gates + ML triage   ← frozen, never re-run at scoring
        │
  D. HUMAN ADJUDICATE  the only thing that makes a pair GOLD
        │
  E. FREEZE            commit artifact + provenance; scoring is string comparison forever
```

### A. Source — answerability is guaranteed here, not audited later

Two independent sources, deliberately:

**A1. Corpus-derived (silver).** Mine the DuckDB store for fact-bearing
sentences. The fact is in the corpus *by construction* — this is what the legacy
hand-written sets fatally lacked.

**A2. Externally sourced (gold candidates).** Real trivia — OpenTDB, Vikipedio
"Ĉu vi sciis?", published Esperanto trivia — translated into Esperanto.
`scripts/eval/test_new_trivia.py` already does this. These questions are *not*
derived from our parser, which is what makes them able to test it (see the
circularity note below).

### B. Construct — minimal transformation, never re-inflection

**The rule: copy the source's tokens verbatim; substitute only the answer
constituent with the case-marked interrogative.**

```
  Zamenhof fondis Esperanton en 1887.
→ Kiu      fondis Esperanton en 1887?
```

You **inherit the source sentence's grammaticality for free**. Every
ungrammatical question this project has produced came from *re-inflecting* a span
that could simply have been copied — gluing `-n` onto `Théâtre des Variétés`
instead of quoting it and letting an Esperanto head noun carry the case:

```
  ✗ Kiu reĝisoris Théâtre des Variétésn?          (synthetic generator)
  ✓ Kiu verkis la vortaron "Altdeutsches Wörterbuch"?   (gold review queue)
```

The second is from `build_gold_review_queue.py`, which already does this right.

The interrogative's **case is read off the answer's role**, not guessed:
`Kiu` (nominative) when the answer is the subject, `Kiun` (accusative) when it is
the object. Esperanto marks this explicitly; use it.

> **Deferred, not chosen:** the elegant construction is AST-native generation —
> build the question AST, swap the answer constituent for the interrogative, and
> **deparse** (#774). VISION.md already claims the linearizer makes output
> "grammatically correct by construction." **That claim is currently false:** the
> deparser lowercases proper nouns and scrambles word order
> (`James Dalgety inventis la nomon «Nonograms»` → `James inventis la nomon
> Dalgety «Nonograms»`). Fixing it is a real project (#801), not a prerequisite.
> Minimal transformation gets most of the benefit today.

### C. Machine verify — three layers, all frozen

**C1. Deterministic gates (R1–R17).** Fail-closed, at write time. These are
cheap, auditable, and catch the known failure classes. They are the floor, not
the ceiling.

**C2. Answerability + discrimination.** Fully mechanized and already built:

| Gate | Question it answers |
|---|---|
| **R7 floor** | Is the gold passage findable? (BM25 top-50) |
| **R16 ceiling** | Is it *already won* by BM25? (rank 1 → reject: nothing left to measure) |
| **R14 support** | Does the corpus know this fact more than once? |
| **R8 provenance** | `source_sentence_id`, with drift detection |

**C3. ML triage — the new layer.** Construction-time only.

| Job | Model | Why this model |
|---|---|---|
| **Grammaticality judgment** | **Claude** | The auditor passed 5 broken questions 8/8; a model that reads Esperanto flags them instantly, *without our having to enumerate failure modes we have not thought of yet*. This is the highest-value use. |
| **Semantic answerability** | **NLLB-200** (`epo_Latn`) or **OPUS-MT** | Translate the question AND the source sentence to English; check the English question is answerable from the English sentence. **This path never touches our parser** — see circularity, below. |
| **Answer aliases** | **Claude**, once, frozen | `Ludoviko Zamenhof` / `L. L. Zamenhof` / `Zamenhof` all name the same person. Enumerate acceptable variants into a frozen `gold_answer_aliases[]`. Scoring stays exact-match against a fixed list — semantic tolerance **without** a runtime judge. |
| **Bulk pre-filter** | Local models via **Ollama** | Already wired (`test_new_trivia.py`). Cheap first pass. |

Note the asymmetry: **open-weight MT for translation** (a narrow, solved task)
and **Claude for judgment** (Esperanto grammaticality is exactly where a weaker
model fails quietly and you do not notice).

ML triage **does not decide**. It produces a score and a reason, which **ranks
the human review queue**. Its job is to spend the scarce resource — human
attention — where it will do the most good.

### D. Human adjudicate — the only source of gold

- Review queue is ordered **worst-first** by ML confidence.
- Each pair gets a verdict: `gold` / `silver` / `rejected` + reason, **frozen**.
- **Stage-3 gate (from the standard):** a 20-pair stratified sample must hit
  **≥ 90% human accept**. Below that, the pipeline is producing junk and the
  auditor is too permissive — *add a check* (that is R11's escalation rule, and
  it is how #791 was found).
- Track the **auditor-vs-human gap**. The standard says expect 5–9 points. When
  the auditor said 8/8 PASS on a batch with five broken questions, the gap was
  ~60 points. **That gap is the health metric of the instrument itself.**

### E. Freeze

- Commit the set. Commit the provenance: which model, which version, which date,
  which pairs it touched (`llm_filtered: true`, `llm_model: …`).
- **Attribution is decomposable** — the same discipline the thesis demands of the
  pipeline, applied to the benchmark that measures it.
- After freezing, **no model is ever queried again**. Scoring is exact match plus
  a frozen alias list, forever reproducible from git.

---

## The circularity problem (F13), and how translation breaks it

Failure mode **F13** in the quality standard:

> *"Generator was downstream of parser; every parser bug rippled into the test
> set."*

This is a genuine circularity: **a parser-derived test set cannot independently
validate a parser-based pipeline.** If the parser mis-analyses a sentence, the
generator builds a pair around the mis-analysis, and the pipeline is then scored
against its own error.

Two things break it, and we need both:

1. **The externally-sourced gold set (A2).** Its questions did not come from our
   parser at all.
2. **Translation-based verification (C3).** Translating the question and the
   source to English and checking answerability there is a check that *routes
   around our parser entirely*.

This is the strongest argument for using ML in construction: not that it is
convenient, but that it provides an **independent** signal that no
deterministic component of ours can provide, because all of ours share the same
parser.

---

## What ML may NOT do here

**An LLM is not gold for parser quality.** If we label ASTs with a model and
measure agreement, we are measuring *agreement with the model*, not correctness —
and worse, we would start tuning the 16 rules toward the model's idiolect.

- **Parser gold stays UD_Esperanto-Prago** (131 sentences, human, external).
- The legitimate use is **active learning**: have the model flag *disagreements*
  between its reading and Klareco's parse, and route those to a human. The model
  **samples**; the human **adjudicates**. This grows UD-Prago cheaply without
  ever letting the model define correctness.

---

## The target portfolio (≥ 250 pairs)

Three sets, three purposes. **Never mix purposes in one file** — the 322-pair
eval gave 63% accuracy that was really ~71% once ambiguous pairs were filtered
out: an 8-point measurement error from mixing.

| Set | Size | Grade | Purpose | Built by |
|---|---|---|---|---|
| `gold_trivia_150.jsonl` | 150 | **GOLD** | Honest ceiling; breaks F13 circularity | A2 → C → **D (human)** |
| `capability_100.jsonl` | 100 | silver | Retrieval/extraction A/B; R16 headroom guaranteed | A1 → B → C |
| `regression_frozen_30.jsonl` | 30 | gold | Regression detection | Carved from the above (pairs we answer correctly today) |

**≥ 250 pairs**, plus a reusable pipeline: every stage is parameterized by
question type, topic, and entity type, so *new* subsets ("give me 50 KIAM
questions about science") are a config change, not a new project.

---

## Triage of the existing sets (measured 2026-07-13)

| Set | n | R13 language | R16 headroom | Verdict |
|---|---|---|---|---|
| `gold_trivia_review_queue_v1` | 63 | 82.5% pass | ungated | **KEEP — review it.** All 63 still `needs_review`; the queue was built in June and never adjudicated. This is the seed of the gold set. |
| `synthetic_who_rebuild_50` | 50 | 98% pass | ❌ 46% rank-1 | **SALVAGE.** Keep only the ~26 pairs with gold rank ≥ 2; they are language-clean and non-trivial. Backfill `gold_answer_span`. |
| `synthetic_who_rebuild_17_cleanish` | 17 | 100% pass | ❌ 58.8% rank-1 | **RETIRE as a capability set.** Linguistically perfect but measurement-useless — this is the set on which nine rerankers tied. Its 7 rank-≥2 pairs may seed the regression set. |
| `synthetic_kiu_active` | 8 | 37.5% pass | ✅ has headroom | **SCRATCH.** Generated as a smoke test under the new gates; regenerate under the fixed pipeline. |
| `data/evaluation/rag_test_set` | 30 | — | — | **RETIRE.** **0/30 carry a `source_sentence_id`** and there are no expected answers — only `expected_answer_pattern`. This is failure mode **F9** (no-signal pairs) exactly. Unusable; archive. |

The two axes are **independent**, and the table proves it: the 17-set is
linguistically perfect and useless; the new 8 have headroom and bad Esperanto. A
set must pass **both** to be a measurement instrument.

---

## See also

- `QA_TEST_SET_QUALITY_STANDARD.md` — R1–R17, binding construction rules
- `DESIGN.md` → *The benchmark contract* — the three metrics, reported separately
- `VISION.md` — why the residue matters, and why the measuring apparatus is not
  the system under test
- `AGENTS.md` → *The merge gate* — why a scoring-time judge would break it
