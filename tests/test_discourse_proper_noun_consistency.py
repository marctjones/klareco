"""klareco/discourse/proper_noun_consistency.py -- cross-sentence
consistency checking for the parser's own propra_nomo classification.

VISION.md calls proper-noun disambiguation its flagship claimed residue.
This checks whether a SPECIFIC DOCUMENT'S other uses of a word settle a
single-sentence weak classification, one way or the other.
"""
from __future__ import annotations

from klareco.ast_annotations import validate_layers, with_annotation_layer
from klareco.ast_storage import compact_ast, expand_ast
from klareco.discourse.proper_noun_consistency import (
    find_weak_proper_noun_mentions,
    make_annotation_layer,
    resolve_document,
    resolve_mention,
)
from klareco.parser import parse


class TestWeakMentionIdentification:
    def test_sentence_initial_capitalization_alone_is_not_flagged(self):
        """Position 1 in a sentence carries NO capitalization signal --
        a common word capitalized only by sentence-initial convention must
        not even be a CANDIDATE for this check."""
        ast = parse("Domoj estas grandaj konstruaĵoj.")
        assert find_weak_proper_noun_mentions(1, ast) == []

    def test_a_mid_sentence_capitalized_word_with_no_deductive_evidence_is_flagged(self):
        ast = parse("Poste Petro studis medicinon.")
        mentions = find_weak_proper_noun_mentions(1, ast)
        assert len(mentions) == 1
        assert mentions[0]["surface"] == "Petro"
        assert mentions[0]["evidence"] == "mid_sentence_capitalization"


class TestDocumentConsistency:
    def test_a_second_mid_sentence_occurrence_confirms_the_classification(self):
        sentences = [
            (1, parse("Poste Petro studis medicinon.")),
            (2, parse("Dume Petro laboris en hospitalo.")),
        ]
        results = resolve_document(sentences)
        assert len(results) == 2
        for m, res in results:
            assert res["status"] == "resolved"
            assert res["candidates"][0]["classification"] == "propra_nomo"

    def test_a_lowercase_occurrence_elsewhere_overrides_the_weak_guess(self):
        """'petro' (the common word, e.g. petroleum) appearing lowercase
        anywhere in the document is decisive evidence AGAINST the SAME
        word being a dedicated proper noun elsewhere in that document --
        Esperanto names are conventionally capitalized everywhere."""
        sentences = [
            (1, parse("Poste petro fluis el la tero.")),
            (2, parse("Dume Petro estis nomo de la produkto.")),
        ]
        results = resolve_document(sentences)
        [(m, res)] = results   # only sentence 2's capitalized mention is a candidate
        assert res["status"] == "resolved"
        assert res["candidates"][0]["classification"] == "common_word"

    def test_no_other_occurrence_anywhere_stays_unresolved(self):
        """A single mid-sentence occurrence with nothing else in the
        document to corroborate or contradict it is a genuine residue --
        not silently confirmed, not silently downgraded."""
        sentences = [(1, parse("Poste Petro studis medicinon."))]
        [(m, res)] = resolve_document(sentences)
        assert res["status"] == "unresolved"
        assert res["candidates"] == []

    def test_inconsistent_usage_across_the_document_is_flagged_ambiguous(self):
        """Both a lowercase AND a mid-sentence-capitalized occurrence exist
        elsewhere -- genuinely inconsistent usage within one document, so
        both possibilities are recorded rather than picking a side."""
        sentences = [
            (1, parse("Poste petro fluis el la tero.")),
            (2, parse("Dume Petro estis grava sciencisto.")),
            (3, parse("Fine Petro rehejmiĝis.")),
        ]
        results = resolve_document(sentences)
        # sentence 2's 'Petro' sees: lowercase in s1 (against) AND
        # capitalized mid-sentence in s3 (for) -- inconsistent.
        m2, res2 = next((m, r) for m, r in results if m["sid"] == 2)
        assert res2["status"] == "ambiguous"
        classifications = {c["classification"] for c in res2["candidates"]}
        assert classifications == {"common_word", "propra_nomo"}

    def test_an_all_caps_occurrence_is_not_treated_as_lowercase_evidence(self):
        """A Wikipedia opening-sentence convention -- 'Henrik "Hinke"
        BERGEGREN (naskita en 1861) ...' -- downcases the all-caps surname
        to 'bergegren' for morphological analysis (all-caps carries no
        capitalization signal per docs/PROPER_NOUNS.md). That normalized
        form must not be read as a genuine lowercase, common-word usage --
        found via manual inspection of an early measurement run, where it
        produced a false 'ambiguous' verdict on a name with no real
        inconsistency in the source text at all."""
        sentences = [(1, parse(
            'Henrik "Hinke" BERGEGREN (naskita en 1861) estis politikisto.'
            "Bergegren estis frua ano de la partio."
        ))]
        results = resolve_document(sentences)
        [(m, res)] = [(m, r) for m, r in results if m["surface"] == "Bergegren"]
        assert res["status"] == "unresolved", (
            "the all-caps occurrence must not count as lowercase evidence")

    def test_resolve_mention_matches_inflected_forms(self):
        """'Petron' (accusative) and 'Petro' (nominative) must compare
        equal -- the SAME name inflected, not two different words."""
        sentences = [
            (1, parse("Mi vidis Petron hieraŭ.")),
            (2, parse("Poste Petro foriris.")),
        ]
        results = resolve_document(sentences)
        assert len(results) == 2
        for m, res in results:
            assert res["status"] == "resolved"
            assert res["candidates"][0]["classification"] == "propra_nomo"


class TestAnnotationLayerContract:
    def test_a_resolution_attaches_and_round_trips_through_storage(self):
        s1 = parse("Poste petro fluis el la tero.")
        s2 = parse("Dume Petro laboris en hospitalo.")
        sentences = [(1, s1), (2, s2)]
        [(m, res)] = find_weak_proper_noun_mentions(2, s2) and resolve_document(sentences)[-1:]
        layer = make_annotation_layer(s2, m, res)
        annotated = with_annotation_layer(s2, layer)   # must not raise
        validate_layers(annotated)
        restored = expand_ast(compact_ast(annotated))
        validate_layers(restored)
        [restored_layer] = restored["annotation_layers"]
        assert restored_layer["schema"] == "urn:klareco:cross-sentence-ambiguity:1"
        value = restored_layer["annotations"][0]["value"]
        assert value["kind"] == "proper_noun_classification"
        assert value["resolution_status"] == "resolved"
        assert value["candidates"][0]["classification"] == "common_word"
