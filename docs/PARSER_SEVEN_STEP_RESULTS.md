# Seven-step parser execution — 2026-09-05

The parser remains deterministic. The engineering work has progressed through
all seven workstreams; independent gold annotation and full-store promotion have
not passed their gates. This is not a claim that all Esperanto syntax is solved.

## Measured parser result

Both revisions use the same evaluator and frozen fixtures. The baseline is
`46e85d1`. Current evidence is in `data/perf/parser_full/punctuation_review/`;
the append-only ledger is `data/perf/bench_history.jsonl`.

| Metric | Before | After |
|---|---:|---:|
| Prago LAS, all 2,712 non-punctuation tokens | 62.5000% | 64.8968% |
| Cairo LAS, all 149 non-punctuation tokens | 69.1275% | 75.1678% |
| Prago subject F1 | 0.6865 | 0.7354 |
| Prago object F1 | 0.7602 | 0.7876 |
| Cairo subject F1 | 0.9302 | 0.9362 |
| Cairo object F1 | 0.8800 | 0.9231 |
| Complete storage round-trips | 151/151 | 151/151 |

Prago gains 65 correct attachments and loses none; Cairo gains 9 and loses none.
Coverage is unchanged: 99.8894% / 100%. POS is unchanged on a common evaluator.
The POS evaluator previously omitted tokens outside the flat sentence frame;
its correction changes the reported baseline, not parser accuracy. Cairo has
been inspected during development and is a regression set, not untouched gold.

## Workstream status

| Step | Executed | Remaining gate or characterized limit |
|---|---|---|
| 1. Error baseline | Reproducible per-token diagnostics, common-evaluator revision comparison, gained/lost attachment inventory, hashes and ledger entries. | Error counts describe symptoms unless a cause has been investigated. |
| 2. Independent benchmark | Deterministic 200-sentence annotation queue; document-disjoint development/heldout split; review and CoNLL-U validation/export tooling. | All 200 remain unreviewed. Independent annotation/adjudication is required before freezing gold or expanding to 1,000. |
| 3. Coherent AST | Token registry, dependency-derived predicate/phrase views, parent and membership IDs, original/normalized source spans, explicit forest/unaligned/alternative status. | Verbless flat views remain heuristic. Alternatives are explicitly not enumerated. |
| 4. Clause repairs | Imperatives and conditionals recognized as finite; nested infinitives and ke-complements; interrupted subjects; coordination inside fronted relatives; punctuation boundaries and quoted-word cycle prevention. | General coordination, shared arguments, gapping, and difficult nesting still have residual errors. Existing support is not a correctness guarantee. |
| 5. Roles and attachments | Possessive/copular nominal heads, copular parent attachment, PP-aware subject boundaries, preservation of postverbal relative subjects. | No lexical exception lists or model were added. PP ambiguity and name/morphology residue remain. |
| 6. Preservation and validation | Lossless v2 storage, graph/projection validation at storage boundaries, owned read-only JSON snapshots and numeric arrays, versioned frame derivation. | Derivation is coarse; there is no complete per-token rule trace. Structural validation cannot certify a linguistic analysis. |
| 7. Integration and measurement | Production reader/writer contract tests; 10,000-row side-store trials; canonical clause extraction and question readers; paired 500-question downstream experiments; full rebuild/preflight trials. | Full rebuild blocked by 21 oversized source records. No production store was replaced. Entity facts are not rebuilt by the new builder. |

## Diagnosed causes and remaining errors

The main finite-verb predicate previously required a tense field. Imperatives
and conditionals carry mood instead, so entire clauses could disappear. The
infinitive test made the inverse error. These now use explicit finite/infinitive
morphology, and nested infinitives attach to their governing verb.

Copular clauses could select an attributive or possessive modifier as predicate
instead of its noun. Parent clauses could attach to the copula rather than the
predicate. The fixes follow grammatical attachment information already present.

A resumed clause could steal a noun from a preceding PP or the postverbal subject
of a relative clause. Punctuation and preposition scope now constrain that
boundary. Coordination before a containing main predicate stays with its local
preceding predicate. Quoted mentions of `kies`, and modifiers whose heads lie
inside a proposed relative clause, cannot become that clause's antecedent.

The remaining Prago attachment errors are largest for `nmod` (145), `advmod`
(105), `conj` (96), `nsubj` (83), and `obj` (63). These counts are not five proven
root causes. New independently reviewed examples are needed to separate rules
that can resolve them from ambiguities requiring unavailable context.

## Corpus gate

The original store contains 4,624,110 rows and was kept unchanged. A full trial
first exposed dependency cycles in grammatical examples discussing quoted or
mentioned words. Those failures became regression tests and were repaired.
A later trial stalled on a 73,062-character PMEG extract stored as one sentence.
Its workers were stopped and the candidate recorded as failed.

There are 21 source records longer than the new 10,000-character resource bound,
including document fragments and Wikipedia code/project pages. Their IDs, sizes,
and provenance are in `data/perf/parser_full/oversized_source_rows.json`. The full
preflight now records `blocked_source_repair` before writing any candidate rows.
These records need source segmentation or reviewed exclusion; silently dropping
them or writing NULL ASTs would repeat the original integrity failure.

The successful sampled stores preserve sentence IDs and source text and verify
complete AST round-trips. Their manifests identify the exact parser revision;
only the latest completed sample should be used to assess the final code.
The final sample is `data/perf/parser_full/release_sample10000.db`: 10,000/10,000
rows, complete AST equality, and zero source-identity mismatches.

## Downstream evidence

The fixed-candidate experiment on `rebaseline_500.jsonl` improved keyword-scored
answers from 127/500 to 135/500, with 37 wins and 29 regressions. The candidate
pool was identical in both arms. This is a modest signal, not broad QA readiness.
The test set has no gold answer spans, so extraction exact match is unscorable.

The final live-candidate comparison scores 127/500 before and 134/500 after
(25.4% to 26.8%), with 31 wins and 24 regressions. Keyword-based candidate
recall rises from 48.6% to 50.4%. The final recheck compared 9,445 candidate
ASTs, reused 380 questions with equal inputs, and reran all 120 affected questions.

The live-candidate experiment and its final input-equality recheck are recorded
in `data/perf/parser_full/qa_live_candidates.json` and `qa_final.json`. They test
question readers and refreshed passage ASTs against the unchanged production
shredded columns. They must not be described as a complete rebuilt-store A/B.

## Verification limits

The combined contract/unit/accuracy run found only the two already reproduced
legacy-ledger failures in `tests/test_accuracy_baseline.py`: older July records
lack the current attribution/metric schema. Those historical records were not
rewritten or given invented attribution. The focused parser/storage/pipeline
checks and final counts are recorded in the local execution artifacts.

## Cost and final checks

Ten alternating timing runs over the 151 gold sentences measured median totals
of 45.3 ms before and 96.1 ms after, with word caches warm and the sentence cache
cleared before each sentence. No QA or rebuild workers were running. This is
about 112% more parser CPU, or 0.34 ms per sentence in this local test. It excludes
storage and pipeline snapshots. Richer structure and copying are not free.

The final combined run reports **702 passed**, 5 skipped, one known expected
failure, and the two pre-existing legacy-ledger failures described above. The
additional clause tests cover PP boundaries, postverbal relative subjects, and
coordination within relative clauses; the stored-AST tests use the actual writer
and retriever against an isolated database. Production promotion remains gated.
