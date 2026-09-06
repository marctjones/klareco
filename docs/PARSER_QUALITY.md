# Parser measurement and AST storage

Current execution status and remaining gates are recorded in
[PARSER_SEVEN_STEP_RESULTS.md](PARSER_SEVEN_STEP_RESULTS.md).

The parser is deterministic. Judge changes against independently annotated
syntax, preserve the complete result, and report improvements separately from
downstream QA. No model is part of this workflow.

The decision gate for independent gold, oracle decomposition, stopping rules,
and learned-component eligibility is [PARSER_EVALUATION_PROTOCOL.md](PARSER_EVALUATION_PROTOCOL.md).

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

Storage v2 now carries syntax graph v2; readers also accept historical syntax v1. For predicate-bearing clauses, clause
and main-sentence frames are derived from dependencies. Storage boundaries check
dependency cycles, roots, clause membership, parents, and argument consistency.
Verbless fragments retain a legacy heuristic flat view; they have no asserted
predicate frame. Structural validity is not linguistic correctness.

## Snapshot isolation

`parse()` returns a deep copy of its cached result, preserving shared references
within that returned graph while isolating callers from one another. The cache
controls remain available. Storage encoding/decoding also isolates its inputs.
This costs copying time; reports include a warm-word-cache, cold-sentence-cache
latency measurement. Pipeline ASTs and JSON flags are now recursively read-only, owned snapshots.
`deepcopy()` explicitly produces an editable copy. Numeric latent arrays have
immutable byte-backed storage. These guarantees cover the documented JSON and
numeric-array fields, not arbitrary custom Python objects.

A paired local timing check (ten alternating runs, 151 UD sentences, sentence
cache cold and word caches warm) measured median total parsing time of 104.6 ms
before and 142.3 ms after, about 36% more or 0.25 ms per sentence. These are local
observations, not a throughput guarantee. On the same fixtures the lossless
blobs total 1,335,304 bytes versus 1,205,600 previously, about 10.8% more, while
remaining about 42.6% of expanded JSON size. Full-corpus time and size are not
yet measured.

## Historical September 2026 first cycle

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

## Canonical syntax and source text

`vortoj` is the token registry. `kapo`/`rolo` are the attachment authority.
`propozicioj` contains dependency-derived predicate frames, including nonfinite
complements. Each frame carries `predikato`, `token_ids`,
`parent_predicate_id`, direct `argumentoj`, and a derivation version.
`phrases` contains dependency-derived groups with head, member, modifier, and
case-marker IDs. Clause frames also link direct complement predicate IDs. Legacy relative-clause wrappers are derived
from the same dependencies; canonical extraction avoids extracting them twice.

The main clause supplies the top-level subject, verb, object, other dependents,
and negation fields. Tokens in embedded clauses do not supply the main clause's
arguments. The compatibility `verbo` may be a copula while `predikato` identifies
the actual dependency head. Missing shared arguments are not invented.

`syntax.version` is 2. Multiple roots are explicitly marked as a forest; tokens
outside predicate frames are listed in `unassigned_token_ids`. Empty alternatives
with `alternatives_status: not_enumerated` do **not** mean there is no ambiguity.
Refinement traces identify rule IDs, before/after edges, and evidence tokens.
Trace coverage is explicitly partial; older attachment passes remain untraced.

`source.original` retains the input verbatim. `source.normalized` retains the
normalized text before tokenizer protection markers. Tokens carry character
spans in both strings and the normalized `surface_form`. Length-changing
normalization and casing are mapped explicitly; offsets are characters, not
bytes. Spans over normalized replacements refer to the corresponding original
range. An unaligned span is null, never a fabricated offset.

The sentence parser rejects inputs over 10,000 characters before expensive
analysis. This is a resource bound, not a linguistic claim about maximum sentence
length. Segment document-sized source records before passing them to this API.

## Reviewed annotation export

```bash
python scripts/eval/validate_parser_annotations.py \
  data/test_sets/parser_pilot_v1/development.jsonl \
  --output data/test_sets/parser_gold_development_v1
```

Export refuses unreviewed rows, matching annotator/reviewer identities, changed
source text, invalid or cyclic dependencies, incomplete token coverage, missing
POS/relations, and mixed splits. It freezes source/gold hashes and document
membership and refuses overwrite. This validates review records and structure;
it does not authenticate reviewer identities or independently judge Esperanto.
The current pilot is unreviewed and is expected to fail this gate.

## Rebuild safely and measure both parser revisions

```bash
python scripts/eval/compare_parser_revision.py --baseline-ref 46e85d1 \
  --output data/perf/parser_comparison
python scripts/index/reparse_store.py \
  --output data/perf/parser_sample.db
python scripts/index/reparse_store.py --limit 0 \
  --output data/indexes/parser_candidate.db
```

The default sample is 10,000 rows selected by a fixed hash of sentence ID without
consulting parser success. The builder preserves IDs, text, and provenance;
rebuilds ASTs, sentence columns, clauses, token edges, and dependency arcs; and
copies the ontology with its actual schema. The existing `verb_klaso` convention
remains POS/transitivity codes from the typed lexicon, not semantic classes.
The manifest records source statistics, code/vocabulary/lexical-artifact hashes, row counts, and
completion status. Every serialized AST is compared with its fresh parse.

Oversized input blocks before any candidate rows are written. Parse failures
stop the build and record offending IDs. Output paths cannot overwrite previous
candidates or manifests. A failed candidate is diagnostic evidence, not a release.
There is no automatic promotion, and source files are opened read-only.

Entity facts are not rebuilt by this tool. `validated_candidate` means the
storage checks passed; it does not mean production promotion or complete pipeline
validation. Sampling also does not establish corpus-wide accuracy. The full
production store remains unchanged until its source and promotion gates pass.

The QA comparison tool supports a fixed-candidate experiment and
`--live-candidates`. The latter selects candidates using each parser's question
AST against the unchanged production shredded columns, then reparses selected
passages. It is still not a full rebuilt-store comparison. Retain that distinction
when citing results. `recheck_parser_qa.py` can prove input AST equality and rerun
affected questions after a narrowly scoped parser repair; reused timings remain
those of the original run.

## September 5 research and syntax v2 cycle

The active architecture and seven-workstream plan are in [PARSER_DESIGN.md](PARSER_DESIGN.md).
The selected dependency graph remains authoritative. Source hashes, offset
conventions, complete morphological candidates, partial attachment traces, and
stand-off annotation layers now have explicit contracts. Predicate and argument
registries are checked against dependencies; storage rejects values JSON cannot
preserve. The complete package comparison loads historical syntax, morphology,
and storage together rather than mixing revisions.

`klareco.ast_annotations.with_annotation_layer(ast, layer)` returns a validated,
owned copy. The layer binds to source text and, when needed, tokenization or
selected dependencies. Targets support sentence, token IDs, edges, and original
character spans. Review status requires review metadata but does not prove
independent linguistic judgment. The gold exporter now writes `annotations.jsonl`
alongside CoNLL-U; these gold targets are original spans and never depend on a
parser's proposed tokens.

`klareco.conllu.ast_to_conllu(ast, strict=True)` exports an existing graph without
reparsing, and rejects forests or unaligned tokens. Diagnostic mode explicitly
labels forests. The serializer no longer repairs roots privately.

The new measurements supersede the previous cycle's figures for current code;
the production store and previously measured QA results still refer to their
recorded revisions. See the latest section in [PARSER_SEVEN_STEP_RESULTS.md](PARSER_SEVEN_STEP_RESULTS.md).
