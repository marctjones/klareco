# Archived test sets — retired, do not resurrect

Retired 2026-07-13 under [#793](https://github.com/marctjones/klareco/issues/793).
Every claim below was **re-measured against the files in this directory** on that
date, not copied from the issue. The measurement commands are given so you can
repeat them before arguing with the verdict.

These files are kept only so that nobody re-derives them from scratch and repeats
the mistake. **They are not measurement instruments and must not appear in any
benchmark, `bench_history.jsonl` row, or reranker A/B.**

---

## `rag_test_set.jsonl` (was `data/evaluation/rag_test_set.jsonl`) — 30 pairs

**Verdict: RETIRE. Unusable. Do not repair.**

Measured:

| property | count |
|---|---|
| pairs | 30 |
| pairs with a `source_sentence_id` | **0 / 30** |
| pairs with an `expected_answer` | **0 / 30** |
| pairs with a `gold_answer_span` | **0 / 30** |
| pairs with only an `expected_answer_pattern` | 30 / 30 |

Its full key set is:
`category, difficulty, expected_answer_pattern, expected_performance,
expected_sources, id, notes, question, required_capabilities, subcategory`

There is **no answer in it anywhere**. `expected_answer_pattern` holds strings like
`"Esperanto estas [definition]"` and `"L.L. Zamenhof / Doktoro Zamenhof"` — a
human-readable *hint*, not a label. Nothing in the file can be compared to a
pipeline output by a deterministic scorer.

This is failure mode **F9** (no `source_sentence_id`, no `expected_answer`)
exactly — the same defect that killed the legacy sets. It violates:

- **R8** — provenance: `source_sentence_id` is required. Without it, retrieval
  recall cannot be computed at all; there is no gold passage to find.
- **R17** — gold answer span: extraction cannot be scored without one.

**Why not repair it?** Repair means hand-labelling a source sentence and an answer
span for all 30 — i.e. authoring 30 new pairs, with the old questions as a prompt.
That work belongs in the gold pipeline (#796/#799) against the current corpus, not
in a rehabilitation of a file whose questions (`Kio estas Esperanto?`) are also
mostly R12 failures. Archiving is cheaper and more honest than a repair that would
be a rewrite.

Reproduce:

```bash
python - <<'EOF'
import json
rows = [json.loads(l) for l in open('data/test_sets/archive/rag_test_set.jsonl')]
print(len(rows),
      sum(1 for r in rows if r.get('source_sentence_id') is not None),
      sum(1 for r in rows if r.get('expected_answer')))
EOF
# -> 30 0 0
```

---

## `synthetic_kiu_active.jsonl` — 8 pairs

**Verdict: SCRATCH. Smoke-test artifact. Regenerate, do not reuse.**

This file was produced as a **smoke test** while building the R16 gate (#778) — it
exists to prove the generator ran, not to measure anything. n = 8 is far below any
useful power.

Measured:

- **R16 headroom: fine.** BM25 gold ranks are `[2, 9, 5, 21, 2, 7, 2, 6]` — no
  rank-1 pairs at all. It is the *only* legacy set with headroom.
- **R13 language quality: 37.5% pass** (3/8). The questions are bad Esperanto,
  because the generator re-inflects spans it should have copied verbatim
  (`Kiu kreis Fonduso Tonkin por novaj iniciatojn?` — the head noun is left
  nominative while the trailing adjective takes the accusative).
- Its `expected_answer` values include `Aŭgusto`, `Britaj` (an **adjective**) and
  `Fugueuses` — the same broken-span defect documented in
  `../README.md`.

**This set is the proof that the two axes are independent.** It has the headroom
`synthetic_who_rebuild_17_cleanish` lacks, and lacks the language quality the 17
has. A set must pass **both** R13 and R16 to be an instrument. Neither of these two
files does, in opposite directions.

Regenerate under the fixed pipeline rather than salvaging: with n = 8, salvage buys
at most a handful of pairs and inherits a known-broken span extractor.

---

## See also

- `../README.md` — what is still live in `data/test_sets/`, and the standing
  warning about `synthetic_who_rebuild_17_cleanish`
- `docs/QA_TEST_SET_QUALITY_STANDARD.md` — R7, R8, R13, R16, R17
- `docs/QA_TEST_SET_PIPELINE.md` → *Triage of the existing sets*
- `scripts/eval/salvage_test_sets.py` — the triage that produced this archive
