# Esperanto parser and annotation design

Status: active design, 2026-09-05. The implementation remains deterministic.
Current measurements and corpus gates are in `DESIGN.md` and
`docs/PARSER_SEVEN_STEP_RESULTS.md`; this document distinguishes implemented
contracts from further work. The production store still contains older parses.

## What the review found

The selected dependencies in `vortoj` already form a useful common contract.
Replacing them with one generic recursive node type would obscure differences
between surface tokens, morphological analyses, predicates, and annotations.
Esperanto also permits discontinuous dependencies. A dependency graph with
explicit, derived views is a better fit than assuming every phrase is a
contiguous interval or that every sentence has one subject slot.

The remaining problems were concrete:

- Alternative PP heads were not renumbered when punctuation changed token IDs.
- `kiam`, `kie`, `kial`, and `kiel` could become noun arguments or determiners
  because every correlative inherited nominal defaults.
- Comparisons such as `kiel la knabo, ...` could open a spurious finite clause.
- A serializer could invent a root that the stored AST did not contain.
- Morphological alternatives omitted ordered compounds, linking vowels, and
  endings. Cached analysis objects could be changed by their callers.
- Arbitrary annotations had no source binding, producer contract, review status,
  or validated target references. JSON accepted values it could not preserve.
- The historical comparison tool loaded only an old `parser.py`; after modular
  changes, that mixed an old parser with new projection and storage code.

These are implementation defects and contract gaps, not evidence that a model
is needed. The new regression tests and full-package comparison cover them.

## Research and its implications

EspGram demonstrates that a substantial Esperanto parser can use Constraint
Grammar and lexical resources. Its pipeline separates morphology, syntactic
functions, and dependency attachment; errors in earlier stages can propagate
into later ones. That supports modular constraints with observable decisions.
Bick's approximately 96.5% result is on revised Arbobanko news annotation with
its own conventions and development procedure. It is **not** a deterministic
ceiling, a Klareco result, or a directly comparable UD-Prago LAS target.
[Dependency Constraint Grammar, 2009](https://dspace.ut.ee/bitstreams/19260085-eb99-467c-85e1-4554522e9952/download),
[Arbobanko analysis, 2020](https://aclanthology.org/2020.lrec-1.630/).

PMEG describes more than object marking for accusative `-n`: measure, time, and
direction also occur. Case and agreement constrain analyses but do not alone
settle every role. Productive `mal-` also applies to uninflected particles such
as degree expressions. We implement grammatical composition using the existing
closed-class inventory, rather than adding a list of corpus-specific words.
[PMEG N](https://bertilow.com/pmeg/gramatiko/rolmontriloj/n/bazaj_reguloj.html),
[PMEG MAL](https://bertilow.com/pmeg/vortfarado/afiksoj/prefiksoj/mal.html),
[PMEG ajn](https://bertilow.com/pmeg/gramatiko/e-vortecaj_vortetoj/ceteraj/ajn.html).

UD distinguishes the basic dependency tree from enhanced relations. In
particular, basic coordination does not tell us whether an argument is shared.
We therefore retain basic attachments, expose clause links, and do not copy a
subject to every conjunct as an asserted fact. A later enhanced layer must state
its evidence and be evaluated separately.
[UD coordination](https://universaldependencies.org/u/dep/conj.html),
[CoNLL-U specification](https://universaldependencies.org/format.html).

Stand-off annotation connects labels to immutable source offsets, including
discontinuous spans. This allows human gold to use its own tokenization without
being forced to agree with the parser it evaluates. Our annotation envelope adds
source hashes, optional tokenization/dependency bindings, and producer metadata.
[brat stand-off format](https://brat.nlplab.org/standoff.html).

The small externally annotated Prago corpus is legal/manifesto text. Its manual
UD labels are valuable, but performance there does not establish performance on
fiction, conversations, grammatical examples, or damaged web text. Both Prago
and Cairo have now informed development and are regression fixtures.
[UD Prago documentation](https://universaldependencies.org/treebanks/eo_prago/index.html),
[Oya's Esperanto treebank paper](https://aclanthology.org/2025.udw-1.3/).

## The data contract

`vortoj` stores each surface token once. A positive integer ID is local to one
sentence's tokenization; it is not a globally stable entity ID. `kapo` and `rolo`
are the selected basic attachment. Zero head requires the `root` relation.
Multiple roots are explicitly a forest. A verbless fragment does not acquire a
fabricated predicate or an implied semantic interpretation.

`source` retains original and normalized strings, the original SHA-256, and the
offset convention: Unicode code points, start inclusive/end exclusive. Tokens
carry both span mappings. Normalization can change lengths; a null span records
failure to align. Source span validation checks bounds and normalized surface
agreement, not the linguistic quality of normalization.

`syntax.version = 2` adds validated attachment candidates and explicit trace
coverage. Candidate heads use final IDs. Candidate sets are local and partial;
they are **not** a packed forest of every jointly valid sentence analysis. A
repeatable proximity preference is a heuristic, not grammatical certainty.
`attachment_trace` records before/after edges, rule IDs, and evidence token IDs
for the refinement stage. Earlier attachment passes are not fully traced yet.

`propozicioj` and `phrases` are dependency-derived views. Clause frames contain
all currently represented direct arguments, parent and complement predicate
IDs, and token membership. Phrase views contain their head ID, surface-ordered
member IDs, direct modifier IDs, and case-marker IDs. Membership follows the
non-predicate, non-punctuation dependency subtree and may be discontinuous;
embedded predicates have their own clause views. Legacy sentence slots derive
from the main frame; verbless flat slots remain a compatibility heuristic.

Morphological alternatives preserve ordered morphemes, including compound
linking vowels, the analyzed surface, lexical type, scores, and violations.
`aplikita` identifies what was used in the selected parse even when `elektita` is
null. `selection_status` distinguishes a heuristic ranking from an unresolved
tie. Search is bounded by the current lexicon and recursion limits; this is not
an exhaustive claim about Esperanto morphology. Cached readings are immutable.
Hyphenated forms may retain a head-component analysis; `morfologia_formo` states
exactly which form the morphemes reconstruct.

`annotation_layers` is the extension contract in `klareco.ast_annotations`:

| Field | Contract |
|---|---|
| `version`, `id`, `schema` | Versioned envelope, unique layer ID, qualified payload schema identifier. |
| `producer` | Name, version, rule/human method, hashes of relevant artifacts. |
| `basis` | Original source hash; tokenization hash for token/edge targets; optional dependency hash for syntax-dependent results. |
| `annotations` | Unique record IDs, typed targets, JSON object payloads. |
| target | Whole sentence, ordered token IDs, dependency endpoints, or one/more original character spans. |
| `status`, `review` | Predicted/draft/reviewed; reviewed requires distinct recorded annotator/reviewer and notes. |

Producer-specific payload semantics belong to their identified schema; the
shared validator cannot certify them. Review metadata is a structural record,
not authentication of identity or proof of independent linguistic judgment.
Prediction layers and human gold coexist. Adding a layer returns an owned copy;
it never relabels the parser's tokens. Original-span gold survives changes in
parser tokenization. Token-bound labels must be explicitly reanchored when IDs
or offsets change; dependency-bound labels also invalidate after reattachment.

Storage remains the lossless v2 envelope in `klareco.ast_storage`; its format
version is separate from the syntax version. Both syntax v1 and v2 readers are
supported. Full JSON equality, reference integrity, graph consistency, and
annotation bindings are checked at storage boundaries. Tuples, sets, non-string
object keys, non-finite numbers, and cyclic JSON are rejected. Unknown payloads
are preserved, not silently discarded. The v2 storage marker `$token` remains
reserved in structural dictionaries, including annotation payloads; collisions
raise instead of being reinterpreted. Older unversioned compact blobs remain
marked lossy; absent historical information cannot be recovered from references.

CoNLL-U exports the selected graph without reparsing or inventing roots. The
strict export gate requires one tree and aligned tokens; diagnostic exports
label forests explicitly and must not be presented as valid UD gold trees.
Neither a source JSON blob nor a CoNLL-U row substitutes for enhanced semantics.

## Seven workstreams and acceptance gates

| Workstream | Implemented in this cycle | Next increment and its gate |
|---|---|---|
| 1. Measurement and source integrity | Full historical package isolation, fixed gold coverage, explicit gains/losses, code and lexical hashes. | Independently review the 200-sentence queue, diversify documents, freeze a new unseen test split. Characterize/segment the 21 oversized source rows with provenance before a full rebuild. |
| 2. Token and graph contracts | Syntax v2, source hashes/spans, phrase and complement IDs, strict reference and projection checks. | Move tokenization/normalization to a dedicated module with complete alignment diagnostics. Measure token coverage, storage failures, and latency before replacing it. |
| 3. Morphological analysis | Complete candidate records, applied-reading status, immutable caches, loud missing-artifact failures. | Obtain independent morpheme boundaries and measure candidate recall as well as selected-boundary accuracy. Evaluate search bounds and lexical gaps before changing ranking. |
| 4. Grammatical constraints | Correlative/particle/degree composition, comparison phrases, copular adverbs, local adjective coordination, root consistency. | Extract the older attachment passes into named modules incrementally. Prioritize actual PP, coordination, relative-clause and argument errors. Each capability change must improve frozen LAS with recorded regressions. |
| 5. Ambiguity and explanations | Valid final-ID local candidates; refinement traces and explicit incomplete-enumeration status. | Test a bounded dependency-candidate solver with case, agreement, clause boundaries, and acyclicity constraints. Measure candidate recall, selected LAS, unresolved rates, resource bounds, and ablations. Retain it as research if quality does not improve. |
| 6. Annotation and interchange | Source-bound layers, review validation, independent gold export, pure CoNLL-U serialization. | Add producer-specific payload validators as real consumers need them; evaluate enhanced coordination/control roles against independently annotated edges. Do not equate a schema extension with a capability result. |
| 7. Integration and promotion | Reader compatibility, complete storage tests, isolated rebuild tooling and measured comparisons. | Rebuild and validate sentence columns, clauses, dependencies, entity facts, and index consistency together. Run downstream QA on that candidate; promote only after corpus, annotation, quality, and cost gates pass. |

This is an incremental parser improvement program, not a claim that all seven
future gates have passed. No model, external labeling service, or production
store replacement is needed for the implemented work. Semantic classes,
valency, and synonyms continue to come from existing lexical/ontology resources;
closed grammatical paradigms are the only language-specific constants added.

## How to reproduce

```bash
python scripts/eval/compare_parser_revision.py --baseline-ref 1b68033 \
  --output data/perf/parser_research/reproduction
python -m pytest -q -m 'contract or unit or accuracy'
python scripts/index/reparse_store.py --limit 10000 \
  --output data/perf/parser_research/new_candidate.db
python scripts/eval/validate_parser_annotations.py <reviewed-pilot.jsonl> \
  --output <new-frozen-gold-directory>
```

Local reports belong under `data/perf/` and the append-only benchmark ledger;
corpora and database candidates must not be committed. A sampled rebuild proves
storage behavior on that sample, not corpus-wide syntax accuracy. Downstream
benefit and parse/serialization costs must be reported separately from LAS.
