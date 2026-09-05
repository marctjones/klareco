# Parser measurement and AST storage

The parser is deterministic. Judge changes against independently annotated
syntax, preserve the complete result, and report improvements separately from
downstream QA. No model is part of this workflow.

## Reproduce the measurement

Run from the repository root with the project dependencies installed:

```bash
python scripts/eval/parser_quality_report.py --output data/perf/parser_before.json
python -m pytest tests/test_parser_ud_accuracy.py tests/test_parser_predicate_heads.py
python -m pytest tests/test_ast_storage.py tests/test_clause_tree.py tests/contract
python scripts/eval/parser_quality_report.py --output data/perf/parser_after.json
```

Reports include code/fixture/vocabulary hashes, POS strict and scheme-adjusted,
subject/object metrics, LAS/UAS, coverage, crashes, complete storage round-trips,
byte sizes, and parsing time. Token diagnostics use the evaluator's own alignment.
`head_in_gold` is the predicted head translated into gold token numbering.
Missing and crashed tokens remain in the denominator. Error groups describe
symptoms; a human-readable cause still needs investigation.

The attachment headline is `las_all`: all non-punctuation gold tokens, with
unaligned tokens scored wrong. Keep Prago and Cairo separate. Once a Cairo error
informs implementation, that example is regression evidence, not an untouched
generalization test. Do not change gold labels to make a rule pass.

## Annotation pilot

```bash
python scripts/eval/build_parser_pilot.py --output data/test_sets/parser_pilot_v1
```

The script opens the store read-only and refuses to overwrite an existing queue.
It uses source round-robin selection, at most four sentences per document,
document-disjoint development/heldout splits, and normalized-text deduplication.
It excludes exact normalized UD fixture texts. It never invokes the parser or
filters candidates by parser success. Exact-text exclusion does not establish
that a different excerpt from the same UD source document is absent; review
provenance before promoting the queue to gold.

The first local queue has 160 development and 40 heldout sentences. All are
**unreviewed**, not a gold benchmark. Its source counts are Wikipedia 180 and
four each from PMEG, Krestomatio, Lingvaj Respondoj, Alice, and Andersen. Those
smaller sources are each represented as one document in the store, so the
document cap limits diversification. Acquire more independently sourced
documents before claiming broad genre coverage.

Annotation procedure:

1. Review original text and provenance without showing the parser's prediction.
2. Assign token boundaries, morphology, dependency heads/relations, and clause
   membership. Record relative/subordinate clauses, coordination, attachment,
   negation, names, and other relevant constructions in `phenomena`.
3. Have a separate Esperanto-competent reviewer adjudicate uncertain analyses.
   Preserve alternatives and annotation decisions. Do not fabricate certainty.
4. Populate `gold_conllu`, reviewer identity, and review status. Validate IDs,
   head references, token coverage, and graph structure before export.
5. Freeze file hashes and document membership before using a reviewed split for
   evaluation. Keep heldout labels out of rule development; examples used to
   guide a fix become development/regression examples.

The QA test-set standard governs any downstream QA sets derived from this work.
Its parser-clean acceptance filter must not be used to select a parser benchmark:
that would remove precisely the failures this instrument is meant to measure.

## Storage version 2

`klareco.ast_storage` owns serialization; the public functions remain importable
from `klareco.parser` for compatibility. The JSON envelope is:

```json
{"_ast_format": 2, "vortoj": [{"id": 1}], "structure": {"subjekto": {"$token": 1}}}
```

The abbreviated token above illustrates references, not a complete parsed token.
`vortoj` stores complete token dictionaries once. `structure` recursively stores
all other fields, including phrase wrappers, modifiers, nested clauses, and
additional annotations. A complete token identical to its table entry becomes
a `$token` reference. A token override with different annotations stays explicit.
`$token` is reserved in structural dictionaries.

The contract is complete JSON value equality after compact -> JSON -> expand.
It is stronger than surface-text round-trip or matching head roots. Duplicate or
invalid token IDs, dangling references, malformed references, and unsupported
storage versions raise. Expanding an already expanded AST returns a private copy.

Unversioned legacy blobs are decoded with `_legacy_compact_lossy: true`. That
marker survives re-encoding. Their omitted phrase/clause information cannot be
recovered by resolving token IDs. Old software does not understand v2 blobs:
deploy updated readers before producing a v2 corpus. Retain the old store until
the replacement is validated. The current production store has not been rebuilt.

This format preserves the parser's current representations; it does **not** yet
make the flat frame, clause frames, and dependencies consistent. The authoritative
projection decision (#903/#904), full graph validation, and per-rule annotation
remain separate work. Token-table validity is not syntactic correctness.

## Snapshot isolation

`parse()` returns a deep copy of its cached result, preserving shared references
within that returned graph while isolating callers from one another. The cache
controls remain available. Storage encoding/decoding also isolates its inputs.
This costs copying time; reports include a warm-word-cache, cold-sentence-cache
latency measurement. The pipeline's nested context dictionaries are still mutable;
this change does not claim to enforce deep context immutability.

A paired local timing check (ten alternating runs, 151 UD sentences, sentence
cache cold and word caches warm) measured median total parsing time of 104.6 ms
before and 142.3 ms after, about 36% more or 0.25 ms per sentence. These are local
observations, not a throughput guarantee. On the same fixtures the lossless
blobs total 1,335,304 bytes versus 1,205,600 previously, about 10.8% more, while
remaining about 42.6% of expanded JSON size. Full-corpus time and size are not
yet measured.

## September 2026 first cycle

Local evidence is in `data/perf/parser_cycle/` and `data/perf/bench_history.jsonl`.
The predicate-head fix follows an attributive adjective's nominal head in a
copular phrase. It retains adjectival predicates, excludes PP-governed nouns,
and preserves participial predicates. It does not add a lexical exception list.

| Metric | Before | After |
|---|---:|---:|
| Prago LAS, all 2,712 non-punctuation tokens | 62.2788% | 62.5000% |
| Cairo LAS, all 149 non-punctuation tokens | 66.4429% | 69.1275% |
| Prago coverage | 99.8894% | 99.8894% |
| Cairo coverage | 100% | 100% |
| Complete AST storage round-trips, 151 sentences | 45 | 151 |

These are small-treebank results, not a corpus-wide accuracy claim. POS and
subject/object frame metrics are reported separately. Generalization requires
the independent annotation work above; QA benefit requires consumers of the
corrected structure.
