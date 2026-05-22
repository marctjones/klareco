# Q&A Test Set Build Plan: getting to 100+ measurable questions

> Companion to `QA_TEST_SET_QUALITY_STANDARD.md`. Written 2026-05-21
> after building the R12–R14 auditors and running them on every
> existing candidate set. This is the **how**, not the **what**.

## Status of existing sets (measured 2026-05-21)

Running the new R12 trivia-caliber audit + Stage-1 mechanical audit
across every set we have:

| Source set | Total | Mechanical PASS | R12-PASS | ≥ 2 corpus support | Net usable |
|---|---|---|---|---|---|
| `synthetic_who_trivia_v2.jsonl` | 194 | ~190 | 182 (94%) | ~43 (22%) | ~40 (KIU only) |
| `all_active_combined.jsonl` | 322 | ~310 | 185 (58%) | TBD | ~60 (multi-type) |
| `audit_2026-05-20_final.jsonl` | 349 | 244 | 187 (54%) | TBD | overlap w/ above |
| `diversified_combined.jsonl` | 51 | 51 (100%) | 0 (0%) | ~30 (≥ 2 support) | **0** (R12 fail) |
| `trivia_bank.jsonl` | 13 | 0* | 0* | TBD | **13** (different schema) |

\* trivia_bank uses `eo_question`/`eo_answer` field names, not
`question`/`expected_answer`; the auditor reports 0 PASS purely
because of field-name mismatch. Pairs are real.

**Observations:**
- The richest harvest source for the **capability** set is the
  overlap between `synthetic_who_trivia_v2` (after R12) and
  `all_active_combined` (multi-type). Estimated yield: 60-90 pairs
  before deduplication.
- `diversified_combined` cannot enter the capability set — every
  pair is an obscure place name with a 4-word source. It stays as
  a *retrieval-discriminability* set only.
- `trivia_bank` IS the seed of the **honest-ceiling** set. 13 pairs
  exist; we want 50.
- Question-type distribution will tilt toward KIU (WHO) — we need
  to actively pull KIE/KIAM/KIAL/KIEL from `all_active_combined`.

## Goal

Two purpose-specific files, each PASSing the full gate stack:

| File | N | Purpose | Gate stack |
|---|---|---|---|
| `capability_100.jsonl` | 100 | Pipeline capability — corpus-derived, R12-PASS | Stages 1, 1.5, 1.7 (R12 + R14 ≥ 2), 2 |
| `trivia_real_50.jsonl` | 50 | Honest ceiling — externally-sourced real trivia | Stages 1, 1.5, R12 only. R14 + R2 are reporting-only. |

Plus the existing reference sets stay where they are:
- `regression_frozen_30.jsonl` — cherry-picked from `capability_100`
  after pipeline eval, frozen forever
- `diversified_combined.jsonl` — keep as the
  retrieval-discriminability probe; rename for clarity

## Phase A — Harvest from existing sets

**Goal:** 50-80 capability pairs without writing any new questions.
Estimated wall-clock: 2-4 hours.

### A.1 — Schema-normalize trivia_bank (10 min)

The 13 trivia_bank pairs use different field names. Write a one-shot
converter:

```python
# scripts/eval/normalize_trivia_bank_schema.py
# Maps eo_question → question, eo_answer → expected_answer,
# corpus_coverage.sample_sentence_id → source_sentence_id,
# corpus_coverage.sample_sentence_text → source_sentence_text.
# Preserves en_question/en_answer/source/category as metadata.
```

Output: `data/test_sets/trivia_bank_normalized.jsonl` (13 pairs,
schema-clean).

These 13 are the seed of `trivia_real_50.jsonl`, not
`capability_100.jsonl`.

### A.2 — Run the full gate stack on every candidate (15 min)

```bash
# Stage 1 + R12
python scripts/eval/audit_qa_pairs.py \
  --test-sets data/test_sets/synthetic_who_trivia_v2.jsonl \
              data/test_sets/all_active_combined.jsonl \
  --trivia-caliber \
  --output data/staging/harvest_audit_2026-05-21.jsonl

# Stage 1.5
python scripts/eval/audit_language_quality.py \
  --in data/test_sets/synthetic_who_trivia_v2.jsonl \
       data/test_sets/all_active_combined.jsonl \
  --output data/staging/harvest_lang_audit_2026-05-21.jsonl

# Stage 1.7 (R14)
python scripts/eval/audit_corpus_coverage.py \
  --in data/test_sets/synthetic_who_trivia_v2.jsonl \
       data/test_sets/all_active_combined.jsonl \
  --top-k 50 --min-support 2 --report-only \
  --output data/staging/harvest_coverage_audit_2026-05-21.jsonl

# Stage 2
python scripts/eval/audit_discriminability.py \
  --in data/test_sets/synthetic_who_trivia_v2.jsonl \
       data/test_sets/all_active_combined.jsonl \
  --top-k 50 \
  --output data/staging/harvest_discrim_audit_2026-05-21.jsonl
```

### A.3 — Intersect the four audits + dedupe (script, 5 min)

Write `scripts/eval/harvest_qa_pairs.py` that:

1. Reads all four audit JSONLs by `id`.
2. Keeps only pairs that PASS *all four* audits.
3. Dedupes by `(question, expected_answer)` (lowercase + diacritic-fold).
4. Tags each pair with its source-set.

Output: `data/staging/harvest_candidates_2026-05-21.jsonl`.

Expected yield: 60-90 pairs.

### A.4 — Apply diversity caps (R15) to select 80 of them

The harvest may be 80% KIU. Apply per-type caps to enforce R15
distribution targets and pick the final 80:

| Type | Target in capability_100 | Harvest from |
|---|---|---|
| KIU | ≤ 25 | synthetic_who_trivia_v2 (overlap w/ all_active) |
| KIO | ≤ 25 | all_active (1 today — gap) |
| KIE | ≤ 15 | all_active (45 today) |
| KIAM | ≤ 15 | all_active (43 today) |
| KIAL | ≤ 5 | all_active (7-10 today) |
| KIEL | ≤ 5 | all_active (32-50 today) |
| KIOM | ≤ 5 | (gap — phase B) |
| ĈU/KIES/KIA | ≤ 5 | (gap — phase B) |

Within each type, prefer pairs that have:
1. Multi-token anchor (more notable by R12)
2. Higher corpus support count (R14 robustness)
3. Higher Wikidata sitelink count (R15 difficulty)

Output: `data/test_sets/capability_harvested.jsonl` (target: 60-80
pairs).

### A.5 — Stage-3 human spot-check (20 min, manual)

Stratified sample of 20 pairs. For each: "is this real trivia?"
gut check. Record accept/reject. If accept rate < 90%, the auditor
is too permissive — add a check and re-run from A.2.

## Phase B — Generate the remaining 20-40 pairs from English trivia

**Goal:** fill the gap from Phase A's ~60-80 to a clean 100.
Same pipeline produces the `trivia_real_50.jsonl` set.

Estimated wall-clock: 4-8 hours (mostly LLM translation rounds).

### B.1 — Source English trivia (1 hour)

`build_trivia_bank.py` already supports OpenTriviaDB ingestion. Other
sources to pull from:

- **OpenTriviaDB API** (`opentdb.com/api.php`) — categorized,
  multiple-choice; we use the question + correct answer
- **Trivia repositories on GitHub** (`pulipulichen/trivia`,
  `el-cms/Open-Trivia-DB`) — JSONL exports we can deduplicate against
- **Wikidata SPARQL** — `?item wdt:P31 wd:Q5; wdt:P31/wdt:P279* ?notable_class`
  to surface notable people and their facts (born-in, founded-by,
  etc.) — gives us a clean question template + canonical Esperanto
  Wikipedia entity
- **Manual curation from published trivia books** — last resort

Filter at source:
- Categories: science, history, geography, art, sports, technology,
  literature (R15 topic mix)
- Difficulty: medium + hard (filter out "what color is the sky")
- Answer is a named entity, year, or short phrase (not a sentence)
- Avoid pop-culture-only questions (Marvel movies, today's celebrities)
  that won't be in our Wikipedia-derived corpus

Output: `data/staging/english_trivia_pool.jsonl` (target: 200 pairs)

### B.2 — Translate to Esperanto via Claude (2-3 hours)

For each English pair:
- Generate `eo_question` and `eo_answer` via Claude (claude-opus-4-7 per
  the existing trivia_bank schema)
- Prompt template includes:
  - Examples of well-formed Esperanto trivia questions from
    `synthetic_who_trivia_v2`
  - Explicit instructions: use ĉĝĥĵŝŭ (not x-system); use accusative
    for direct objects; preserve quoted titles in «...»; use proper
    Esperanto preposition for time (`en 1887`, not `je 1887`)
  - Reject and retry if translation contains x-system / h-system

A `scripts/eval/translate_trivia_batch.py` script that:
1. Reads `english_trivia_pool.jsonl`
2. Batches 20 questions per Claude call (keep prompt under 4K tokens)
3. Writes a checkpoint after every batch (resume-safe)
4. Stores both `en_*` and `eo_*` fields

Output: `data/staging/translated_trivia_pool.jsonl` (target: 200 pairs)

### B.3 — Run the full gate stack on translated pool

Same as A.2 but on the translated pool. R14 corpus-coverage will be
the highest-failure check — many real trivia answers won't be in our
Esperanto Wikipedia. That's fine and expected.

Two outputs from the same audit data:

- **For `trivia_real_50.jsonl`**: keep all pairs that PASS Stages
  1 + 1.5 + R12. R14 corpus support is *reported* but doesn't gate.
  Aim for 50 pairs.
- **For `capability_100.jsonl` filler**: only the subset of the above
  that ALSO PASSes R14 (≥ 2 corpus support) AND Stage 2
  (discriminability). Aim for 20-40 pairs to fill out
  `capability_100`.

### B.4 — Targeted gap-filling for under-covered types

After B.3, check the type distribution of `capability_100`. If KIO
or KIOM is under target, generate type-specific English trivia:

- KIO definitional: "What is the largest planet?" (`Kio estas la
  plej granda planedo en la sunsistemo?`) — answer is a category
  noun, not a tautology
- KIOM numeric: "How many countries are in Europe?" (`Kiom da
  landoj estas en Eŭropo?`) — answer is a number
- ĈU yes/no: rarely good trivia; deprioritize

## Phase C — Verify, freeze, commit

Estimated wall-clock: 1-2 hours.

### C.1 — Full gate stack on the two final files

```bash
python scripts/eval/audit_qa_pairs.py \
  --test-sets data/test_sets/capability_100.jsonl \
  --trivia-caliber --strict

python scripts/eval/audit_language_quality.py \
  --in data/test_sets/capability_100.jsonl --strict

python scripts/eval/audit_corpus_coverage.py \
  --in data/test_sets/capability_100.jsonl \
  --top-k 50 --min-support 2 --strict

python scripts/eval/audit_discriminability.py \
  --in data/test_sets/capability_100.jsonl --top-k 50 --strict

# For real-trivia set: only Stages 1, 1.5, R12 are strict
python scripts/eval/audit_qa_pairs.py \
  --test-sets data/test_sets/trivia_real_50.jsonl \
  --trivia-caliber --strict
python scripts/eval/audit_language_quality.py \
  --in data/test_sets/trivia_real_50.jsonl --strict
```

All must exit 0. If any pair fails, fix or drop and re-run.

### C.2 — Run end-to-end pipeline eval

```bash
python scripts/eval/evaluate_extractive_qa.py \
  --test-set data/test_sets/capability_100.jsonl \
  --output results/capability_100_baseline.json

python scripts/eval/evaluate_extractive_qa.py \
  --test-set data/test_sets/trivia_real_50.jsonl \
  --output results/trivia_real_50_baseline.json
```

Three identical runs of each, confirm result moves < ± 3%.

### C.3 — Freeze `regression_frozen_30.jsonl`

Pick the 30 pairs from `capability_100` that the pipeline currently
answers correctly. These become the regression set: any merge that
drops the score below 30/30 on this set is blocked.

```python
# scripts/eval/freeze_regression_set.py
# Reads results/capability_100_baseline.json
# Picks 30 pairs where answer_correct == True
# Stratifies by question_type to preserve diversity
# Writes data/test_sets/regression_frozen_30.jsonl
```

### C.4 — Commit + document

```
git add data/test_sets/capability_100.jsonl
git add data/test_sets/trivia_real_50.jsonl
git add data/test_sets/regression_frozen_30.jsonl
git add results/capability_100_baseline.json
git add results/trivia_real_50_baseline.json
git commit -m "Test set v3: capability_100 + trivia_real_50 + regression_frozen_30"
```

After commit: archive the legacy sets to `data/test_sets/archive/`.

## Risk register

| Risk | Mitigation |
|---|---|
| Phase A yields < 60 pairs after dedupe | Phase B grows to fill more of `capability_100`. Translation pipeline is the bottleneck either way. |
| Translation produces non-idiomatic Esperanto | Stage 1.5 auditor catches mechanical issues. Stage 3 spot-check is the last line. If quality is bad, switch translator (Sonnet-4.6 fallback) and re-run. |
| R14 ≥ 2 too aggressive even at the calibrated threshold | Drop to ≥ 1 with source-rank ≤ 3. Document the decision. |
| Real trivia all has zero corpus coverage | Expected for 50-70% of OpenTriviaDB. Those go in `trivia_real_50` as coverage-gap reports, not failures. |
| Wikidata-notable list misses Esperanto-specific entities (Zamenhof, UEA, etc.) | The list is augmented: any entity with ≥ 100 entity_postings rows is auto-added to the notable set. |
| Pipeline-eval results are noisy across runs | Stage C.2 averages 3 runs; outliers (> 3% drift) trigger investigation. |

## Out of scope (now)

- Auto-generation pipelines for KIO place-type questions (R15 mid-priority)
- Vikipedio "Ĉu vi sciis?" scraping (alternative for B.1; future iteration)
- Multilingual back-translation as a quality check on translations
- Synthetic adversarial questions (e.g. forcing a known parser bug)

## Definition of done

- [x] R12/R13/R14 auditors built and smoke-tested
- [ ] Phase A: `capability_harvested.jsonl` ≥ 60 pairs, all gate-stack-PASS
- [ ] Phase A.5: Stage-3 spot-check accept rate ≥ 90%
- [ ] Phase B: `capability_100.jsonl` reaches 100 pairs
- [ ] Phase B: `trivia_real_50.jsonl` reaches 50 pairs
- [ ] Phase C.2: pipeline eval results stable ± 3% across 3 runs
- [ ] Phase C.3: `regression_frozen_30.jsonl` committed
- [ ] Legacy sets archived under `data/test_sets/archive/`

When all eight are checked, we're done with test-set work for v3.
We can spend the next month making the pipeline better against a
fixed, audited, honest measurement instrument — instead of arguing
about whether the number is real.

## See also

- `QA_TEST_SET_QUALITY_STANDARD.md` — the rules (R1–R15) and gate
  stack (Stages 0–5) that this plan executes
- `scripts/eval/audit_qa_pairs.py` — Stages 1 + R12
- `scripts/eval/audit_language_quality.py` — Stage 1.5 (R13)
- `scripts/eval/audit_corpus_coverage.py` — Stage 1.7 (R14)
- `scripts/eval/audit_discriminability.py` — Stage 2 (R7)
- `scripts/eval/build_trivia_bank.py` — Phase B.2 reference impl
