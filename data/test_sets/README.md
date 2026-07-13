# `data/test_sets/` — what is here and what it may be used for

Last triaged **2026-07-13** under
[#793](https://github.com/marctjones/klareco/issues/793). Every number below was
measured against the files in this directory on that date
(`scripts/eval/salvage_test_sets.py`, `scripts/eval/audit_discriminability.py
--rank-histogram`). Re-measure before you trust any of it.

| file | n | may be used as a **capability** set? | why |
|---|---|---|---|
| `salvaged_who_nontrivial.jsonl` | **26** | ✅ yes (with caveats below) | 0% rank-1; every pair in the measurable band 2..50 |
| `gold_trivia_review_queue_v1.jsonl` | 63 | ⚠️ not yet | all 63 are still `review_status: needs_review`; nobody has adjudicated them |
| `synthetic_who_rebuild_50.jsonl` | 50 | ❌ **NO** | 46% rank-1 — saturated (superseded by the salvaged file) |
| `synthetic_who_rebuild_17_cleanish.jsonl` | 17 | ❌ **NO — see below** | 58.8% rank-1 |
| `archive/` | — | ❌ never | retired; see `archive/README.md` |

---

## ⛔ `synthetic_who_rebuild_17_cleanish.jsonl` — DO NOT BENCHMARK AGAINST THIS

**This is the set on which all nine rerankers tied.** It cannot measure ranking,
and no amount of linguistic polish will change that.

Measured 2026-07-13 (BM25 rank of the gold passage, question text alone, K = 50):

```
ranks = [1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 2, 1, 2, 1, 2, 2, 6]
rank-1: 10 / 17 = 58.8%     (R16 ceiling is 20%)
median gold rank: 1
```

**BM25 already puts the gold passage first for 58.8% of the pairs.** A *perfect*
reranker could not move those pairs, because there is nowhere for them to move.
The nine-way tie was not a finding about the rerankers; it was a property of the
ruler. (The set is also *linguistically perfect* — 100% R13 pass. That is exactly
why it is dangerous: it looks like a good set.)

Its questions hand BM25 the answer: `Kiu inventis «Nonograms»?` quotes a rigid,
rare title that occurs in precisely one sentence in the corpus. Lexical overlap
solves it. That is not a *wrong* question — it is an **uninformative** one for
ranking work.

**It is not junk — it is the wrong instrument.** Rank-1 pairs are perfectly good
**regression** pairs (things we already get right and must not break). Its 7
rank-≥2 pairs may seed `regression_frozen_30`. But it must never again appear in a
reranker or retrieval A/B.

Its **7 salvageable (rank ≥ 2) pairs are already inside
`salvaged_who_nontrivial.jsonl`** — the 17-set is an `id`-subset of the 50-set, so
salvaging both and deduplicating by `id` yields one file, not two.

---

## ✅ `salvaged_who_nontrivial.jsonl` — 26 pairs

Produced by `scripts/eval/salvage_test_sets.py` from the union (deduplicated by
`id`) of `synthetic_who_rebuild_50` and `synthetic_who_rebuild_17_cleanish`. The
17 is a subset of the 50, so the union is 50 unique pairs, not 67.

```
in                 50 unique pairs
dropped rank-1     23   (46.0%)  — BM25 already won; R16
dropped not-found   1   ( 2.0%)  — gold sid outside BM25 top-50; R7
dropped bad-span    0
out                26   (52.0%)
  rank-1 share      0.0%      <- R16 satisfied
  band 2..50      100.0%
  median gold rank    5
```

Each pair carries `gold_answer_span` (R17), `bm25_gold_rank` and `bm25_top_k`
(R16 provenance), and `salvaged_from`.

### ⚠️ Only 8 of the 26 have a form-clean answer span — and only ~6 are correct

**18 carry `gold_answer_span_suspect` + `review_status: needs_review`.** Filter on
that field before using the set for **extraction** scoring. For
**retrieval/reranking** scoring all 26 are usable — retrieval is scored against
`source_sentence_id`, which is sound regardless of the span.

**The mechanical check verifies span FORM, not answer CORRECTNESS.** It asks "is
this string shaped like a name drawn from this sentence?", which no automatic test
can extend to "is this the right answer to this question?". A manual read of the 8
form-clean survivors found **2 that are still wrong**, both R4 (verb-proximity)
failures where the question's verb does not govern the quoted anchor:

| pair | gold span | why it is wrong |
|---|---|---|
| `who_gen_041` | `Schröder` | source: *"Schröder verkis lernolibron … kaj **tradukis** la dramon „La verda kakatuo" **de Arthur Schnitzler**"*. Schröder *translated* it. The answer to "Kiu **verkis** «La verda kakatuo»?" is **Schnitzler**. |
| `who_gen_024` | `Trichet` | the anchor «I am not a Frenchman» is governed by ***deklaris***, not *gajnis* — he won a prize *because* he said it. Nobody "won" the quote. |

So the honest count for **extraction** is **~6 solid pairs**
(`who_gen_011, 019, 036, 044, 047, 050`), not 8. Verb-role correctness is left to
the human review pass; automating it is a separate problem.

The spans are bad because of an upstream **parser** bug, not a generator whim.
With the proper-noun artifacts lost in the June migration (see `CLAUDE.md` →
*Broken / degraded*), the parser tags **every sentence-initial capitalised word**
as `propra_nomo`. The generator then lifted `subjekto.kerno` as the answer. So it
produced WHO questions whose gold answer is an **adverb**:

| pair | gold answer | what it actually is |
|---|---|---|
| `who_gen_022` | `Nuntempe` | adverb — "nowadays" |
| `who_gen_046` | `Drame` | adverb |
| `who_gen_048` | `Anstataŭe` | adverb — "instead" |
| `who_gen_039` | `Britaj` | adjective |
| `who_gen_049` | `Teorio` | common noun; the publisher is *Albert Einstein* |
| `who_gen_038` | `Большая Книга` | the **prize**; the winner is *Rubina* |
| `who_gen_002` | `Maksim` | truncated — the source reads *Maksim Gorkij* |
| `who_gen_006` | `ThomasPusch` | a Wikipedia **talk-page signature** (R6 corpus noise); the anchor `«(Nomo)»` is a literal placeholder, not a rigid designator (R1) |

`kerno` is **one word**, so every multiword name is truncated by construction.
**The span defect is systematic and will recur in any set generated before the
proper-noun artifacts are restored.** Fix the parser data first, or the next
generator run reproduces this exactly.

Suspect spans were **flagged, never auto-repaired**: guessing a boundary
(`Maksim` → `Maksim Gorkij`?) would put a silently-wrong gold answer *inside the
measuring instrument*, which is the precise trap R17 exists to close.

---

## Usable pairs we own today

| | pairs | usable for retrieval/reranking | usable for extraction |
|---|---|---|---|
| `salvaged_who_nontrivial` | 26 | 26 | **~6** (8 form-clean, 2 of those R4-wrong; 18 flagged) |
| `gold_trivia_review_queue_v1` | 63 | 63 (unadjudicated) | 0 (none reviewed) |
| **total, ready to measure with** | | **26** | **~6** |

The `gold_trivia_review_queue_v1` bottleneck is **adjudication, not tooling**: the
queue was built in June and never reviewed. It is the seed of `gold_trivia_150`.

---

## See also

- `archive/README.md` — the two retired sets and the measured reasons
- `docs/QA_TEST_SET_QUALITY_STANDARD.md` — R1–R17 (R7 floor, R16 ceiling, R17 span)
- `docs/QA_TEST_SET_PIPELINE.md` → *Triage of the existing sets*
- `scripts/eval/salvage_test_sets.py` — regenerates `salvaged_who_nontrivial.jsonl`
- `scripts/eval/audit_discriminability.py --rank-histogram` — the R16 gate
