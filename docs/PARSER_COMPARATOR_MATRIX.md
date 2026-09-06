# Esperanto parser comparator matrix

Klareco should compare against three different kinds of existing systems. They
answer different questions and must not be collapsed into one leaderboard.

| System | Public resource | Comparable output | Role in the benchmark |
|---|---|---|---|
| EspGram / EspCG (Bick) | [EspGram](https://edu.visl.dk/eo/) and the published parser evaluations | Morphology, POS, shallow/dependency syntax; conversion to UD required | Strong deterministic Esperanto reference |
| Apertium Esperanto modules | [eo-en](https://github.com/apertium/apertium-eo-en), [eo-ca](https://github.com/apertium/apertium-eo-ca), and related pairs | Morphological and transfer analyses, not native UD dependencies | Lexical and morphology baseline |
| UDPipe | [ufal/udpipe](https://github.com/ufal/udpipe) | CoNLL-U tokenization, UPOS, features, UAS, LAS | Standard trainable UD baseline |
| Trankit | [language support](https://trankit.readthedocs.io/en/latest/pkgnames.html) | CoNLL-U UD pipeline when trained for Esperanto | Transformer baseline |

The standard spaCy model catalog has no Esperanto pipeline, and no local
installation of UDPipe, Trankit, Stanza, or spaCy is currently present. Do not
call a missing off-the-shelf model a baseline. Install or build comparators in
an isolated evaluation environment and record exact versions and model hashes.

## Fair comparison procedure

1. Freeze the independent reviewed gold split before training any comparator.
2. Keep documents, not just sentences, disjoint between development and heldout
   data.
3. Run all systems on identical raw text and score with the same evaluator.
4. Report tokenization, UPOS, features, UAS, LAS, coverage, and latency
   separately. Use fixed-denominator `UAS_all` and `LAS_all`.
5. Train UDPipe and Trankit only on the development split. A model trained on
   Prago and evaluated on Prago is a memorization check, not a quality result.
6. Preserve native EspGram and Apertium outputs. Convert to UD only through a
   documented mapping, and mark relations that have no one-to-one mapping.
7. Include oracle arms for Klareco morphology, POS, clause spine, and candidate
   generation so a learned comparator does not hide a deterministic resource
   bottleneck.

## Expected interpretation

EspGram is the relevant deterministic ceiling reference. Apertium tells us if
our lexical/morphological front end is weak. UDPipe and Trankit measure what
learned contextual parsing can add, but their scores are only meaningful after
the heldout gold set is independent and large enough. A learned system that
wins on a tiny shared treebank does not establish a product advantage.

The current Prago/Cairo fixtures remain regression tests. They are not a fair
cross-system leaderboard because they are tiny and development-informed.
