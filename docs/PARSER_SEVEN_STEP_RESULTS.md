# Seven-step parser execution — 2026-09-05

## Latest cycle: grammar, ASTs, and independent annotations

The current plan is [PARSER_DESIGN.md](PARSER_DESIGN.md), including primary
research, data contracts, and acceptance gates for all seven workstreams.
This cycle compares the complete historical package at `1b68033` with the
new implementation using the same frozen evaluator and lexical artifacts.
Reports are in `data/perf/parser_research/release_comparison/`; per-token
gains and regressions are in `attachment_changes.json`. Code, vocabulary,
and fixture hashes identify both arms. Before/after records are appended to
`data/perf/bench_history.jsonl` with committed-code attribution.

| Metric | Before | After |
|---|---:|---:|
| Prago LAS, all 2,712 non-punctuation tokens | 64.8968% (1,760 correct) | 69.0634% (1,873 correct) |
| Cairo LAS, all 149 non-punctuation tokens | 75.1678% (112 correct) | 79.1946% (118 correct) |
| Prago UAS | 72.6401% | 75.8481% |
| Cairo UAS | 83.8926% | 85.2349% |
| Prago subject / object F1 | 0.7354 / 0.7876 | 0.8610 / 0.8053 |
| Cairo subject / object F1 | 0.9362 / 0.9231 | 0.9778 / 0.9231 |
| Complete storage round-trips | 151/151 | 151/151 |

Coverage remains 99.8894% / 100%, with no parser crashes. Prago gains 115
correct attachments and loses 2; Cairo gains 6 and loses none. Strict native
POS scores are unchanged. Both fixtures have informed development; these are
small regression results, not a new independently held-out accuracy claim.
No downstream QA improvement is claimed for this revision.

The grammar changes stop adverbial correlatives from occupying noun slots,
distinguish delimited comparison phrases from finite clauses, preserve the
class of productive `mal-` degree particles, attach `ĉi`/`ajn` to correlatives,
and repair copular adverbs and local adjective coordination. The two lost
attachments expose existing scope weaknesses: Prago sentence 79's coordinated
PP noun `komunumojn` becomes an object, and sentence 103's `lingvon` after an
inserted participial phrase becomes an oblique. These remain recorded errors;
the implementation contains no sentence-specific repairs.

The data contract now preserves ordered morphology candidates, applied-reading
status, final-ID PP alternatives, and before/after traces for the refinement
rules. Syntax v2 validates predicate/argument/phrase views against the selected
dependencies. Versioned annotation layers bind typed targets to source text and,
where appropriate, tokenization and dependencies. Reviewed gold is exported as
original-span annotations and CoNLL-U without reparsing it. Storage rejects
invalid graph references, stale layers, unsupported versions, and values JSON
would lose or coerce. Syntax v1 remains readable. Full syntactic alternatives
and complete traces of the older attachment passes are still future work.

The final contract/unit/accuracy run reports **760 passed**, 5 skipped,
1 expected failure, and 2 pre-existing failures from unattributed July benchmark
records. Those records have not been rewritten. See
`data/perf/parser_research/checks_release.txt`. Checks ran on Python 3.14;
changed files also pass Python 3.10 grammar checks, and the new annotation types
avoid Python 3.11-only imports. A Python 3.10 runtime was not available locally.

The isolated 10,000-row rebuild is
`data/perf/parser_research/release_candidate10000.db`, with its source and code
hash manifest. All 10,000 ASTs round-trip exactly and preserve sentence IDs,
source text, and provenance, with zero source-identity mismatches. The candidate
occupies about 291 MiB. This is a storage check on a deterministic sample, not
gold syntactic evaluation or a complete production database.

The richer records have a measured cost. Across the 151 fixture sentences,
compact JSON grows from 1,955,238 to 2,886,590 bytes (**47.6%**). Ten alternating
runs with isolated packages, warm word caches, and a cleared sentence cache
measure median parsing totals of 187.2 ms before and 225.1 ms after (**20.3%**
more, or 0.25 ms per sentence). Compact/JSON/expand/equality checks grow from
125.2 to 328.2 ms (**162.3%** more). These are local elapsed-time measurements,
not CPU-only measurements or a throughput guarantee; pipeline snapshots are
excluded. The reproducibility script and results are
`data/perf/parser_research/paired_timing.py` and `paired_timing_final.json`.

Production promotion remains gated by the 21 oversized source records,
independent review of the 200-sentence pilot, full derived-table and entity-fact
rebuilds, index consistency, downstream QA, and acceptable resource costs. The
production store is unchanged. The sections below describe the preceding cycle
and retain its original measurements and limits.

## Previous cycle: clause and storage integration

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
