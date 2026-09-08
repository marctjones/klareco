"""Cross-sentence consistency checking for the parser's own proper-noun
classification -- VISION.md's flagship claimed residue.

Esperanto has no morphological proper-noun marker (VISION.md: "Esperanto
has no morphological marker distinguishing a proper noun from a common
noun"), so `klareco/parser.py` decides `propra_nomo` per token using a mix
of DEDUCTIVE evidence (follows from grammar: an unlicensed ending, ReVo's
own name-root list, ...) and WEAK, statistically-derived evidence
(capitalization ratio, mid-sentence capitalization, "preceded by la") --
see `klareco.parser._WEAK_EVIDENCE`. A weak-evidence classification is, by
construction, the parser's own admission of uncertainty at the SENTENCE
level. This module asks the next question: does this SPECIFIC DOCUMENT'S
own other uses of the same word settle it, one way or the other?

Two document-local signals, both deterministic (no model, no learned
threshold):
- The SAME word (case-folded, with a trailing accusative -n stripped)
  appearing LOWERCASE anywhere else in the document is decisive evidence
  AGAINST it being a dedicated proper noun: Esperanto's own convention
  keeps a name capitalized everywhere, so a lowercase occurrence anywhere
  means at least one occurrence is definitely an ordinary word, and there
  is no license in the grammar for the SAME root to be a name in one
  clause and not another within one text.
- The SAME word appearing CAPITALIZED elsewhere in a MID-SENTENCE position
  (not sentence-initial, where capitalization is a punctuation artifact
  and carries no signal) is corroborating evidence FOR it -- the same
  signal `mid_sentence_capitalization`/`capitalization_ratio` already use,
  but computed from THIS document's own usage rather than a corpus-wide
  frequency table. The two signals are complementary: the corpus-wide
  ratio is precise for a word globally consistent one way; this is precise
  for a word that is genuinely used BOTH ways across the corpus but
  consistently ONE way within any single document (a common real pattern:
  "Petro" the person's name throughout one biography article, "petro" the
  common word throughout an unrelated geology article).

Characterized limitation, found by inspecting real "ambiguous" output: a
common noun that HEADS an institutional or taxonomic proper name
("Universitato" in "Universitato Carnegie Mellon"; "Luscinia" as a
biological genus) is legitimately used BOTH as an ordinary common noun
AND as part of a specific name within the SAME document -- the "one
classification per document" assumption this module relies on genuinely
does not hold for that word class. This shows up as a real 'ambiguous'
verdict, which is the honest outcome (both classifications ARE attested,
right there in the text), not a bug to silently suppress -- but it means
the ambiguous rate for this specific class of word is not evidence of
model weakness the way it would be for an ordinary personal name. A
future refinement could detect "is this occurrence adjacent to another
propra_nomo token, forming a multi-word name" and treat that occurrence
as a different KIND of evidence than a bare common-noun use; not
attempted here.

This is NOT wired into the default orchestrator pipeline. Per the
project's contract, a new capability stays default-OFF until it passes
the contract suite and carries a measured number.
"""
from __future__ import annotations

from typing import Optional

# The parser's own definition of "statistically derived, therefore
# genuinely uncertain at the single-sentence level" -- imported directly
# rather than re-declared, so the two modules can never silently drift
# apart on what counts as weak evidence.
from klareco.parser import _WEAK_EVIDENCE
from klareco.discourse.annotations import make_annotation_layer as _make_layer
from klareco.discourse.annotations import make_finding

MODULE_NAME = "klareco.discourse.proper_noun_consistency"
MODULE_VERSION = "1"


def _fold(surface: Optional[str]) -> Optional[str]:
    """Case-fold and strip a trailing accusative -n so inflected forms of
    the same word (Petro/Petron) compare equal."""
    if not surface:
        return None
    s = surface.casefold()
    if len(s) > 1 and s.endswith("n") and not s.endswith("nn"):
        s = s[:-1]
    return s


def find_weak_proper_noun_mentions(sid: int, ast: dict) -> list[dict]:
    """Every token in one sentence classified `propra_nomo` on WEAK
    (statistically-derived) evidence -- the parser's own admission of
    single-sentence uncertainty."""
    out = []
    for w in ast.get("vortoj", []):
        if not isinstance(w, dict) or w.get("vortspeco") != "propra_nomo":
            continue
        if w.get("propra_nomo_evidence") not in _WEAK_EVIDENCE:
            continue
        out.append({
            "sid": sid,
            "token_id": w.get("id"),
            "surface": w.get("plena_vorto"),
            "folded": _fold(w.get("plena_vorto")),
            "evidence": w.get("propra_nomo_evidence"),
        })
    return out


def _document_occurrences(sentences: list[tuple[int, dict]], folded: str, exclude_sid: int):
    """Every occurrence of `folded` elsewhere in the document, classified
    as either LOWERCASE-somewhere (against) or CAPITALIZED-mid-sentence
    (for). Sentence-initial capitalized occurrences are skipped entirely --
    position 1 in a sentence carries no capitalization signal at all.

    Uses `surface_form` (the token's TRUE original-case text) rather than
    `plena_vorto` (already case-normalized for morphological analysis) to
    decide casing. This matters concretely: an ALL-CAPS token (a common
    Wikipedia opening-sentence convention -- "Henrik \"Hinke\" BERGEGREN
    (naskita en 1861) ...") is downcased to `plena_vorto='bergegren'` for
    analysis per the project's own rule that all-caps carries no
    capitalization signal (docs/PROPER_NOUNS.md: "ignore ALL-CAPS -- a
    heading carries no capitalization signal"). Reading casing off
    `plena_vorto` would treat that deliberate normalization as if it were
    a genuine lowercase, common-word usage -- a real false-positive class
    this fix removes (found via manual inspection of an early measurement
    run, on exactly this sentence).
    """
    lowercase_elsewhere = []
    capitalized_mid_sentence = []
    for sid, ast in sentences:
        if sid == exclude_sid:
            continue
        for w in ast.get("vortoj", []):
            if not isinstance(w, dict):
                continue
            if _fold(w.get("plena_vorto")) != folded:
                continue
            true_surface = w.get("surface_form") or w.get("plena_vorto")
            if not true_surface:
                continue
            if true_surface.isupper() and len(true_surface) > 1:
                continue   # ALL-CAPS: no capitalization signal either way
            if true_surface[0].islower():
                lowercase_elsewhere.append({"sid": sid, "token_id": w.get("id"),
                                             "surface": true_surface})
            elif true_surface[0].isupper() and w.get("id") != 1:
                capitalized_mid_sentence.append({"sid": sid, "token_id": w.get("id"),
                                                  "surface": true_surface,
                                                  "vortspeco": w.get("vortspeco")})
    return lowercase_elsewhere, capitalized_mid_sentence


def resolve_mention(mention: dict, sentences: list[tuple[int, dict]]) -> dict:
    """Check the rest of the document for evidence that settles one weak
    proper-noun classification. Returns a finding dict (candidates are
    {classification, evidence, sid, token_id, surface} -- there is no
    "distance" here, since ANY occurrence in the document counts equally,
    unlike sentence-local coreference)."""
    lowercase_elsewhere, capitalized_mid_sentence = _document_occurrences(
        sentences, mention["folded"], mention["sid"]
    )
    against = [{"classification": "common_word", "surface": mention["surface"],
                "evidence": "lowercase_elsewhere", "sid": o["sid"],
                "token_id": o["token_id"], "corroborating_surface": o["surface"]}
               for o in lowercase_elsewhere]
    for_ = [{"classification": "propra_nomo", "surface": mention["surface"],
             "evidence": "capitalized_mid_sentence_elsewhere", "sid": o["sid"],
             "token_id": o["token_id"], "corroborating_surface": o["surface"]}
            for o in capitalized_mid_sentence]

    if against and not for_:
        return make_finding("resolved", [against[0]])
    if for_ and not against:
        return make_finding("resolved", [for_[0]])
    if against and for_:
        # Genuinely inconsistent usage within one document -- both stay,
        # honestly, rather than picking a side with no principled tiebreak.
        return make_finding("ambiguous", [against[0], for_[0]])
    return make_finding("unresolved", [])


def resolve_document(sentences: list[tuple[int, dict]]) -> list[tuple[dict, dict]]:
    """Resolve every weak-evidence propra_nomo mention in a document-
    ordered list of (sid, ast) pairs. Returns (mention, finding) pairs."""
    results = []
    for sid, ast in sentences:
        for m in find_weak_proper_noun_mentions(sid, ast):
            results.append((m, resolve_mention(m, sentences)))
    return results


def make_annotation_layer(
    mention_ast: dict,
    mention: dict,
    resolution: dict,
    *,
    artifact_hashes: Optional[dict] = None,
) -> dict:
    """Build one mention's resolution as an annotation_layers entry, via
    the shared klareco.discourse.annotations envelope
    (kind='proper_noun_classification')."""
    return _make_layer(
        kind="proper_noun_classification",
        mention_ast=mention_ast,
        mention_sid=mention["sid"],
        token_id=mention["token_id"],
        finding=resolution,
        value_extra={"surface": mention["surface"],
                     "original_evidence": mention["evidence"]},
        producer_name=MODULE_NAME,
        producer_version=MODULE_VERSION,
        artifact_hashes=artifact_hashes,
    )
