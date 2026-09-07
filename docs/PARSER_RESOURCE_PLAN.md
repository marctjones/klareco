# Deterministic parser ceiling plan

Status: active after the 2026-09-07 parser issue/milestone consolidation.

The current parser is not yet selecting reliably among a rich deterministic
candidate set. After bounded structural, relation, and nominative-subject
candidate work, exact gold-edge recall is 76.45% on Prago and 75.84% on Cairo.
The candidate ceiling is 81.21% / 88.59% LAS respectively. This is evidence for
candidate-generation and global-selection work before it is evidence for a
semantic-model problem. The first Apertium-backed global decoder spike improves
Prago by two LAS edges with no Cairo regression, but remains offline.

The decomposition is intentionally conservative. It does not claim that every
candidate is linguistically valid; it asks whether the current parser has even
represented the gold head as an option. The measurement is produced by:

```bash
python scripts/eval/parser_oracle_decomposition.py \
  --output data/perf/parser_research/p0_oracle_decomposition.json
```

## Grammar work, in priority order

### P0: complete bounded candidate generation

Generate explicit head candidates for every nominal, modifier, coordinator,
and clause marker using only the current sentence, morphological case/number,
clause spans, punctuation, and already selected hard attachments. Preserve the
current selected tree, but expose all locally licensed alternatives in
`syntax.attachment_candidates`.

Success metrics are candidate-head recall, selected LAS, and cycle-free output.
The first gate is at least 80% gold-head candidate recall on Prago and 85% on
Cairo, with no coverage loss. This is a research measurement until a selector
improves LAS by at least 0.5 points on an unseen split.

### P0: clause spine and subject/object boundaries

Use finite-verb order, explicit subordinators, relative markers, punctuation,
and PP governance to enumerate subject and object candidates per clause. Keep
the existing agreement rules as hard constraints. Do not infer a missing
predicate or silently collapse verbless fragments.

Measure `nsubj`, `obj`, `root`, `acl`, `advcl`, and `ccomp` head recall
separately, plus overall LAS. These are the structural errors that contaminate
every later attachment decision.

### P1: lexical valency and case-role selection

Use an external lexical resource to constrain whether a verb licenses an object,
oblique, indirect object, or clausal complement. Case remains a hard signal;
valency only ranks candidates that morphology permits. Keep unresolved choices as
alternatives.

Measure `obj`, `obl`, `iobj`, `ccomp`, and `xcomp` LAS, and report the gain on
sentences whose verb has a resource-backed valency entry versus those without
one. A resource must improve the frozen score; coverage alone is insufficient.

### P1: coordination and nominal enumeration

Generate conjunct chains across punctuation, coordinators, shared prepositions,
and coordinated adjectives/verbs. Distinguish coordination from apposition,
address fragments, and PP-internal lists using clause membership and agreement.

Measure `conj` and `cc` head/label accuracy and chain completeness. Require no
regression on punctuated enumeration fixtures.

### P1: modifier scope

Enumerate local adjective, adverb, possessive, degree-particle, and PP heads.
Use agreement and clause boundaries as constraints; use proximity only as a
recorded ranking preference. Comparative `ol`/`kiel`, adverbial correlatives,
and substantivized adjectives belong here.

Measure `nmod`, `advmod`, `amod`, `case`, and `xcomp` LAS, including a separate
ambiguity rate for tokens with multiple surviving candidates.

### P2: relative and subordinate clause attachment

Model relative-clause antecedent candidates and finite subordinate heads as
bounded graphs. Relative markers, subordinators, comma boundaries, and clause
spans constrain the graph; antecedent choice remains an explicit candidate
selection problem.

Measure `acl`, `relcl`, `mark`, `advcl`, and `ccomp` by genre and sentence
length. Stop adding rules when candidate recall is high but deterministic
selection cannot improve held-out LAS.

## Lexical resources

### 1. ReVo structured lexicon

Build a versioned, fail-loudly snapshot from `scripts/acquire/acquire_revo.py`
and `scripts/acquire/acquire_revo_ontology.py`. Preserve headwords, POS,
inflection, derived forms, semantic types, synonym relations, and any explicit
valency evidence. ReVo is the primary lexical source because it is curated and
independent of the parser's own output.

Metrics: root coverage on the held-out corpus, decomposition accuracy, proper
noun false-positive rate, and LAS for resource-covered verbs/nouns.

### 2. Voko-Akrido grammatical bridge

Use `scripts/acquire/acquire_voko_akrido.py` for independently extracted POS,
transitivity, semantic classes, and proper-name roots. Keep it as a separate
producer from ReVo so disagreements are measurable rather than overwritten.

Metrics: agreement with ReVo, coverage of finite verbs, and `obj`/`obl`/`nmod`
LAS changes. Conflicting entries remain ambiguous until the sentence resolves
them.

### 3. Apertium Esperanto morphology cross-check

Use `scripts/acquire/acquire_apertium_epo.py` as an independent analyzer, not
as a source of dependency labels. Compare it with Klareco and ReVo on compounds,
affixes, participles, and function words.

Metrics: morphology/lemma accuracy, unaligned-token rate, and downstream LAS on
sentences whose analysis changes. It must not replace the native morphology
contract without a measured gain.

### 4. Corpus usage statistics

Derive frequency and attachment priors only from document-disjoint development
text. Never harvest parser predictions into the gold or root lexicon. Priors may
rank explicit candidates, but cannot delete a grammatically licensed candidate.

Metrics: held-out LAS, calibration of ambiguity choices, and error rates by
genre, sentence length, and construction.

## Decision gate for a semantic model

Do not add a model while gold-head candidate recall is low. First complete the
candidate generator and lexical bridge. A model becomes justified only when:

1. candidate recall is high on a document-disjoint held-out set;
2. the same residue persists across news, literature, dialogue, and web text;
3. independent annotation shows the residue is not missing grammar or annotation
   disagreement; and
4. a model that ranks existing candidates improves held-out LAS or a frozen
   downstream metric.

The model's permitted role is candidate ranking. Tokenization, morphology,
clause detection, and hard case/agreement constraints remain deterministic.
