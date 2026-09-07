# Parser evaluation protocol

This protocol decides whether deterministic Esperanto parser work should
continue and when a learned component is justified. The Prago and Cairo
fixtures remain regression tests; they are not sufficient as an independent
release gate.

## Gold data gate

The primary gate is a frozen, independently annotated set of at least 500
sentences, with a target of 1,000. It must contain unseen documents from news,
edited prose, informal/web text, older Esperanto, technical writing, names,
fragments, quotations, and creative morphology. Development and heldout splits
are document-disjoint. The parser must not have selected, filtered, or labeled
the sentences.

Every heldout sentence receives two independent annotations and adjudication.
Annotators record disagreement and preserve alternatives where the grammar or
UD policy does not determine one analysis. `scripts/eval/validate_parser_annotations.py`
checks review provenance, token coverage, dependency validity, and frozen hashes;
it cannot authenticate linguistic expertise or resolve an annotation dispute.

Do not promote `data/test_sets/parser_pilot_v1` yet. Its manifest explicitly
marks the queue as unreviewed and not gold.

The non-Wikipedia candidate queue can be regenerated with:

```bash
python scripts/eval/build_parser_corpus_pilot.py \
  --output data/test_sets/parser_gold_candidates_v1 --size 1000
```

This reads the extracted Gutenberg, Libera Folio, and Vikifontaro JSONL
corpora. Selection is deterministic and balanced by source and corpus kind,
deduplicates normalized text, caps reuse of a document, and assigns entire
documents to development or heldout. Each row records source title, author,
kind, license, URL, and source-line provenance. The output is explicitly an
**unreviewed annotation queue, not gold data**. Automatic parser output must
not be copied into labels without independent human review.

For large natural-text robustness checks, use the unlabeled coverage report:

```bash
python scripts/eval/parser_coverage_report.py \
  --output data/perf/parser_research/nonwiki_coverage_5000.json --size 5000
```

It samples the same non-Wikipedia sources and reports parse failures, complete
AST storage round-trips, deterministic reparses, latency, and source/kind
coverage. These are engineering and robustness metrics, not UPOS/UAS/LAS
accuracy. A larger sample is useful for finding failures, but it does not
create gold labels.

## Required scores

For a reviewed gold export, generate sentence-level 95% intervals with:

```bash
python scripts/eval/parser_bootstrap_report.py \
  data/test_sets/parser_gold_heldout_v1/gold.conllu \
  --output data/perf/parser_research/heldout_v1_bootstrap.json
```

The report reuses the fixed-denominator evaluator and resamples whole
sentences. It is a confidence diagnostic, not a substitute for independent
annotation or genre coverage.

Report each split and phenomenon stratum separately:

* token and source-span coverage, including crashes;
* morphology and lemma accuracy;
* UPOS and morphological-feature accuracy;
* UAS and LAS with a fixed denominator over every non-punctuation gold token;
* relation-label accuracy with the gold head supplied;
* root, clause-boundary, subject, object, coordination, PP attachment, and
  proper-noun accuracy;
* complete AST/storage round-trip exactness and latency.

`LAS_all` and `UAS_all` are the headline dependency scores. Aligned-only scores
are diagnostics. A missing or crashed token is wrong; it must not disappear from
the denominator. POS and dependency scores are reported independently so a POS
projection change cannot be mistaken for a syntax improvement.

Use bootstrap confidence intervals over sentences for the independent set.
Do not claim a change from a difference smaller than its uncertainty interval.

## Oracle decomposition

For every major error family, run controlled oracle arms:

1. gold tokenization and spans;
2. gold morphology and lemma;
3. gold UPOS/features;
4. gold clause boundaries and predicate spine;
5. gold candidate heads with deterministic relation selection;
6. gold proper-noun decisions and valency classes.

The gap between the normal arm and each oracle identifies the remaining
deterministic ceiling. A large candidate-head oracle gap means the parser needs
better structural candidate generation. A small gap after gold candidates means
the residue is likely semantic, discourse-level, or annotation-policy
ambiguity.

## Continue, stop, or add a model

Continue deterministic work when a bounded rule or resource change improves an
unseen stratum by at least 0.5 LAS points (or a clearly named POS/coverage
metric), with no material regression elsewhere. Changes that only improve the
development fixtures become regression evidence, not generalization claims.

Call deterministic POS work mature near 98--99% on independent gold. Treat
dependency LAS in the high 80s as a serious intermediate target; investigate
the gap to the published Esperanto Constraint Grammar results before declaring
the grammar exhausted. Published results are not directly comparable because
they use different corpora and syntactic metrics.

Introduce a learned component only after:

* deterministic candidate generation has high recall;
* the independent heldout residue is stable across genres;
* annotator agreement shows the residue is genuinely ambiguous rather than
  missing grammar or inconsistent gold;
* an isolated store A/B shows a frozen downstream gain in retrieval or
  extraction.

The model should rank deterministic candidates. It should not replace
tokenization, morphology, clause detection, or basic dependency construction.

## Product evidence

After parser changes pass the independent linguistic gate, rebuild an isolated
candidate store and compare parser versions on a frozen QA set. Record retrieval
recall, extraction exact match, answer accuracy, and latency. Do not infer
downstream benefit from LAS alone, and do not promote a candidate store merely
because its serialization round-trips.
