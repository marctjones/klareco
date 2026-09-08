"""Discourse-level (cross-sentence) deterministic analysis.

Everything in `klareco/parser.py` operates on ONE sentence at a time — by
design (VISION.md: attempt each capability deterministically, at the
smallest scope that can be measured, before reaching for anything else).
This package is the next scope up: rules that need to see MULTIPLE
sentences from the same document to do their job.

Every ambiguity type already present in the single-sentence AST was
checked for whether document-level context can help. Two genuinely do,
and are implemented here:

- `coreference.py` — a pronoun (li/ŝi/ĝi/ili) or the demonstrative
  tiu/tiuj refers to a NOMINAL mentioned earlier in the document. This is
  the paradigm case: the antecedent is, by definition, somewhere else.
- `proper_noun_consistency.py` — the parser's own weak-evidence
  propra_nomo classification (VISION.md's flagship claimed residue) is a
  property of a WORD'S IDENTITY across the whole document ("Petro" is
  either consistently a name or consistently the common word "rock" within
  one text), so other occurrences of the same word elsewhere in the
  document can confirm or overturn a single-sentence guess.

Two more were checked and found NOT to benefit, for principled reasons
worth recording (VISION.md: a characterized negative finding maps the
boundary as surely as a positive one):

- PP-attachment ambiguity (`alligo_ambigua`/`alligo_opcioj`, e.g.
  en/al/per/kiel): measured on a 300-document sample, the SAME
  (verb, preposition) pair recurring within one document shows the
  identical ambiguous/resolved status 99.1% of the time (4,430 recurring
  pairs, 42 with any variation). That is not evidence of a useful
  cross-sentence signal -- it is the SAME deterministic grammar rule
  producing the SAME classification for the SAME lexical pair every time,
  which is expected regardless of context. Unlike a name's REFERENCE,
  which is a fact about a specific document, PP-attachment ambiguity is a
  near-constant property of the (verb, preposition) TYPE.
- Morphology tie-breaking (`alternativoj` with `selection_status ==
  'unresolved'`): structurally CANNOT benefit, by construction.
  `klareco.parser._apply_morphology` takes only the bare word string, with
  no sentence or document input at all (its own docstring: "no sentence,
  so no position, no agreement, no neighbours"). The candidate set and
  scores for a given word form are identical every time that exact string
  appears, so a tie in one occurrence is a tie in EVERY occurrence -- there
  is no cross-sentence fact that could ever break it. Resolving this would
  require changing what morphology analysis is allowed to look at, not
  adding a discourse-level pass on top of it.
- Word-sense disambiguation (`sencoj`): investigated and found to have no
  bootstrap signal to propagate from. Every occurrence of a polysemous
  root is uniformly `elektita: None` -- no single-sentence mechanism
  anywhere in the pipeline ever resolves a sense, so "propagate the
  resolved sense from one occurrence to another" has nothing to propagate.
  "One sense per discourse" (Gale, Church & Yarowsky 1992) is a real,
  well-established technique, but it needs SOME occurrence to be resolved
  first; that is a prerequisite (a selectional-restriction or
  frame-based single-sentence WSD signal), not something this package can
  add on its own.

Nothing here is wired into the default orchestrator pipeline. Per the
project's contract, a new capability is default-OFF until it passes the
contract suite and carries a measured number.
"""
from __future__ import annotations

from klareco.discourse import coreference, proper_noun_consistency


def resolve_document_ambiguities(
    sentences: list[tuple[int, dict]],
    entity_types: dict[str, str],
    *,
    window: int = 1,
) -> dict:
    """Run every cross-sentence resolver in this package over one document.

    Returns {'coreference': [(mention, finding), ...],
             'proper_noun_classification': [(mention, finding), ...]}.
    """
    return {
        "coreference": coreference.resolve_document(sentences, entity_types, window=window),
        "proper_noun_classification": proper_noun_consistency.resolve_document(sentences),
    }
