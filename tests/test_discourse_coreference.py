"""klareco/discourse/coreference.py — deterministic pronoun-candidate
generation across sentence boundaries.

VISION.md lists cross-sentence coreference as an untested suspected
residue. These tests characterize what the deterministic signals
(sentence order, grammatical number, the thin entity-type ontology,
morphology) can and cannot decide -- not a claim that coreference is
"solved".
"""
from __future__ import annotations

import duckdb
import pytest

from klareco.ast_annotations import validate_layers, with_annotation_layer
from klareco.ast_storage import compact_ast, expand_ast
from klareco.discourse.coreference import (
    build_entity_type_lookup,
    find_pronoun_mentions,
    is_compatible,
    make_annotation_layer,
    pronoun_radiko,
    referring_expression,
    resolve_document,
    resolve_mention,
)
from klareco.parser import parse


@pytest.fixture(scope="module")
def entity_types():
    """A tiny, self-contained entity-type table -- no live store needed."""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE ontology_edges (radiko TEXT, rel TEXT, class_id TEXT)"
    )
    con.executemany(
        "INSERT INTO ontology_edges VALUES (?, 'HAVAS_ENTECAN_TIPON', ?)",
        [("petr", "persono"), ("mari", "persono"), ("urb", "loko"),
         ("hund", "besto"), ("vien", "loko")],
    )
    return build_entity_type_lookup(con)


def _resolve(prev_texts, target_text, entity_types, window=5):
    prev = [(1000 + i, parse(t)) for i, t in enumerate(prev_texts)]
    tast = parse(target_text)
    mentions = find_pronoun_mentions(2000, tast)
    return [
        (m, resolve_mention(m, list(reversed(prev)), entity_types, window=window))
        for m in mentions
    ]


class TestPronounIdentification:
    def test_the_closed_class_is_li_sxi_gxi_ili(self):
        for word, expected in [("li", "li"), ("ŝi", "ŝi"), ("ĝi", "ĝi"),
                                ("ili", "ili")]:
            assert pronoun_radiko(parse(word)["vortoj"][0]) == expected

    def test_the_reflexive_si_is_excluded(self):
        """`si` is bound to its OWN clause's subject by grammar -- it is
        never a cross-sentence reference, so it must not be treated as one."""
        w = next(x for x in parse("Li lavis sin.")["vortoj"]
                 if x.get("plena_vorto") == "sin")
        assert pronoun_radiko(w) is None

    def test_tiu_and_tiuj_are_in_scope_with_their_own_number(self):
        w = next(x for x in parse("Tiu venis.")["vortoj"]
                 if x.get("plena_vorto") == "Tiu")
        assert referring_expression(w) == ("tiu", "singularo")
        w2 = next(x for x in parse("Tiuj venis.")["vortoj"]
                  if x.get("plena_vorto") == "Tiuj")
        assert referring_expression(w2) == ("tiu", "pluralo")

    def test_tio_is_deliberately_out_of_scope(self):
        """'tio' overwhelmingly refers to a preceding CLAUSE/EVENT, not a
        nominal -- searching for a nominal antecedent would answer the
        wrong question, so it is excluded rather than forced."""
        w = next(x for x in parse("Tio okazis.")["vortoj"]
                 if x.get("plena_vorto") == "Tio")
        assert referring_expression(w) is None


class TestResolution:
    def test_a_singular_ntecedent_one_sentence_back_resolves(self, entity_types):
        [(m, res)] = _resolve(
            ["La suno brilis hodiaŭ."], "Ĝi denove aperis morgaŭ.", entity_types)
        assert res["status"] == "resolved"
        assert res["candidates"][0]["surface"] == "suno"

    def test_no_antecedent_at_all_is_unresolved_not_guessed(self, entity_types):
        [(m, res)] = _resolve([], "Ĝi malaperis subite.", entity_types)
        assert res["status"] == "unresolved"
        assert res["candidates"] == []

    def test_two_compatible_candidates_are_flagged_ambiguous_not_picked(self, entity_types):
        """Neither 'hundo' nor 'tapiŝo' is excluded by number or the
        (persono-only) animacy check -- both are legitimate, so the
        annotation must say so instead of silently choosing one."""
        [(m, res)] = _resolve(
            ["La hundo dormis sur la tapiŝo."], "Ĝi dormis tutan tagon.",
            entity_types)
        assert res["status"] == "ambiguous"
        assert {c["surface"] for c in res["candidates"]} == {"hundo", "tapiŝo"}

    def test_a_coordinated_pair_resolves_a_plural_ili(self, entity_types):
        """'Maria kaj Petro' is two SINGULAR nominals, but coordinated they
        are the plural referent 'ili' needs -- read off the dependency
        tree's conj relation, not re-detected some other way."""
        [(m, res)] = _resolve(
            ["Maria kaj Petro venis al la festo."], "Ili dancis tutan nokton.",
            entity_types)
        assert res["status"] == "resolved"
        assert res["candidates"][0]["surface"] == "Maria kaj Petro"

    def test_number_disagreement_excludes_a_candidate(self, entity_types):
        """A plural nominal cannot antecede a singular pronoun."""
        [(m, res)] = _resolve(
            ["La hundoj dormis."], "Ĝi promenis sola.", entity_types)
        assert res["status"] == "unresolved"

    def test_subject_role_outranks_non_subject_at_equal_distance(self, entity_types):
        """Centering Theory's core claim: at equal distance, the SUBJECT
        of the preceding sentence is the preferred antecedent."""
        [(m, res)] = _resolve(
            ["La kato ĉasis la muson."], "Ĝi kuris rapide.", entity_types)
        assert res["candidates"][0]["surface"] == "kato"

    def test_window_bounds_how_far_back_the_search_goes(self, entity_types):
        far = ["Unua frazo pri urbo.", "Dua frazo.", "Tria frazo.",
               "Kvara frazo.", "Kvina frazo.", "Sesa frazo."]
        [(m, res)] = _resolve(far, "Ĝi restis granda.", entity_types, window=2)
        # 'urbo' is 6 sentences back; window=2 must not find it.
        assert not any(c.get("radiko") == "urb" for c in res["candidates"])

    def test_a_predicate_nominal_is_not_ranked_as_the_subject(self, entity_types):
        """'firmao' in a copular sentence is the PREDICATE (rolo='root'),
        not the subject -- it must not outrank the real subject."""
        [(m, res)] = _resolve(
            ["Kongō estis granda firmao."], "Ĝi kreskis rapide.", entity_types)
        assert res["candidates"][0]["surface"] == "Kongō"


class TestDemonstrativeResolution:
    """'tiu'/'tiuj' carry no animacy constraint at all -- grammatically
    fine for both persons and things ("tiu viro" / "tiu tablo") -- so only
    NUMBER agreement applies, unlike li/ŝi/ĝi."""

    def test_tiu_resolves_to_a_singular_antecedent_of_any_animacy(self, entity_types):
        [(m, res)] = _resolve(
            ["Mi legis interesan libron hieraŭ."], "Tiu estis tre bona.",
            entity_types)
        assert res["status"] == "resolved"
        assert res["candidates"][0]["surface"] == "libron"

    def test_tiu_is_compatible_with_a_classified_person_too(self, entity_types):
        cand = {"radiko": "petr", "nombro": "singularo", "sufiksoj": []}
        assert is_compatible("tiu", "singularo", cand, entity_types)

    def test_tiuj_requires_a_plural_antecedent(self, entity_types):
        [(m, res)] = _resolve(
            ["La hundoj bojis."], "Tiuj estis grandaj.", entity_types)
        assert res["status"] == "resolved"
        assert res["candidates"][0]["surface"] == "hundoj"

    def test_tiu_and_gxi_chains_are_tracked_separately(self, entity_types):
        """A document mixing 'tiu' and 'ĝi' references must not let one
        contaminate the other's chain."""
        texts = ["Maria havis hundon.", "Ĝi estis granda.", "Tiu estis feliĉa."]
        sentences = [(5000 + i, parse(t)) for i, t in enumerate(texts)]
        results = resolve_document(sentences, entity_types, window=1)
        # 'Tiu' (3rd sentence) looks back only 1 sentence ("Ĝi estis
        # granda", no nominal there) and has no prior 'tiu' chain -- must
        # be unresolved, not accidentally inherit 'ĝi's chain (hundo).
        tiu_mention, tiu_res = results[-1]
        assert tiu_mention["pronoun"] == "tiu"
        assert tiu_res["status"] == "unresolved"


class TestPronounChaining:
    """A run of same-type pronouns with no re-mentioned noun in between --
    common in biographical/narrative text -- should resolve via chain
    continuation (Centering Theory's CONTINUE transition), not go
    unresolved just because there's no fresh nominal nearby."""

    def test_a_chain_of_pronouns_with_no_intervening_noun_stays_resolved(self, entity_types):
        texts = [
            "Maria naskiĝis en Vieno.",
            "Ŝi studis matematikon.",
            "Ŝi instruis en universitato.",
            "Ŝi mortis en 1990.",
        ]
        sentences = [(3000 + i, parse(t)) for i, t in enumerate(texts)]
        results = resolve_document(sentences, entity_types, window=1)
        assert len(results) == 3
        for m, res in results:
            assert res["status"] == "resolved", (m["surface"], res)
            assert res["candidates"][0]["surface"] == "Maria"
        # The 1st mention resolves from the fresh nominal in sentence 1;
        # the 2nd and 3rd have NO nominal in their own 1-sentence window
        # (their immediate predecessor is itself a pronoun sentence) and
        # can only be resolved via the chain.
        assert results[0][1]["candidates"][0]["source"] == "nominal_mention"
        assert results[1][1]["candidates"][0]["source"] == "pronoun_chain"
        assert results[2][1]["candidates"][0]["source"] == "pronoun_chain"

    def test_a_fresh_unambiguous_mention_still_wins_over_a_stale_chain(self, entity_types):
        """The chain must not blind the resolver to a real, closer answer."""
        texts = [
            "Petro laboris en banko.",
            "Li estis feliĉa.",
            "Poste alvenis Johano.",
            "Li tuj eksidis.",
        ]
        sentences = [(4000 + i, parse(t)) for i, t in enumerate(texts)]
        results = resolve_document(sentences, entity_types, window=1)
        # The last 'Li' has 'Johano' one sentence back (fresh, unambiguous)
        # AND the chain still holding 'Petro' -- both are listed, so status
        # is 'ambiguous', not silently 'Petro'.
        last_mention, last_res = results[-1]
        surfaces = {c["surface"] for c in last_res["candidates"]}
        assert "Johano" in surfaces


class TestAnimacyIsSoftNotHard:
    def test_a_classified_person_is_excluded_from_gxi(self, entity_types):
        cand = {"radiko": "petr", "nombro": "singularo", "sufiksoj": []}
        assert not is_compatible("ĝi", "singularo", cand, entity_types)
        assert is_compatible("li", "singularo", cand, entity_types)

    def test_an_unclassified_proper_noun_is_never_excluded_on_animacy_alone(self, entity_types):
        """The ontology is thin (CLAUDE.md) -- silence must not become a
        false exclusion for a NAME (most capitalized names denote a
        person, place, or organization, so a name defaults to eligible)."""
        cand = {"radiko": "tutecfremda_ne_en_ontologio", "nombro": "singularo",
                "sufiksoj": [], "vortspeco": "propra_nomo"}
        assert is_compatible("li", "singularo", cand, entity_types)
        assert is_compatible("ŝi", "singularo", cand, entity_types)
        assert is_compatible("ĝi", "singularo", cand, entity_types)

    def test_an_unclassified_common_noun_defaults_to_ineligible_for_li_sxi(self, entity_types):
        """The opposite default for ORDINARY vocabulary: an unclassified
        common noun ('matematiko', 'tablo', ...) is overwhelmingly likely
        to be a non-human referent, so it needs a POSITIVE 'persono'
        classification to compete for li/ŝi -- but stays eligible for 'ĝi',
        which is the pronoun expected to cover exactly this case."""
        cand = {"radiko": "tuteca_nekonata_vorto", "nombro": "singularo",
                "sufiksoj": [], "vortspeco": "substantivo"}
        assert not is_compatible("li", "singularo", cand, entity_types)
        assert not is_compatible("ŝi", "singularo", cand, entity_types)
        assert is_compatible("ĝi", "singularo", cand, entity_types)

    def test_the_in_suffix_excludes_li_but_absence_does_not_exclude_sxi(self, entity_types):
        feminine = {"radiko": "instruist", "nombro": "singularo", "sufiksoj": ["in"]}
        assert is_compatible("ŝi", "singularo", feminine, entity_types)
        assert not is_compatible("li", "singularo", feminine, entity_types)
        unmarked = {"radiko": "instruist", "nombro": "singularo", "sufiksoj": []}
        assert is_compatible("li", "singularo", unmarked, entity_types)
        assert is_compatible("ŝi", "singularo", unmarked, entity_types), (
            "absence of -in- is not evidence of male reference -- Esperanto's "
            "unmarked form is not sex-exclusive")


class TestAnnotationLayerContract:
    def test_a_resolution_attaches_and_round_trips_through_storage(self, entity_types):
        prev_ast = parse("La suno brilis hodiaŭ.")
        target_ast = parse("Ĝi denove aperis morgaŭ.")
        [m] = find_pronoun_mentions(2000, target_ast)
        res = resolve_mention(m, [(1000, prev_ast)], entity_types)
        layer = make_annotation_layer(target_ast, m, res)
        annotated = with_annotation_layer(target_ast, layer)   # must not raise
        validate_layers(annotated)
        compact = compact_ast(annotated)
        restored = expand_ast(compact)
        validate_layers(restored)
        [restored_layer] = restored["annotation_layers"]
        assert restored_layer["schema"] == "urn:klareco:cross-sentence-ambiguity:1"
        assert restored_layer["producer"]["method"] == "rule"
        value = restored_layer["annotations"][0]["value"]
        assert value["kind"] == "coreference"
        assert value["resolution_status"] == "resolved"

    def test_the_layer_never_relabels_the_source_parse(self, entity_types):
        """with_annotation_layer must return an owned copy -- the original
        ast is untouched."""
        target_ast = parse("Ĝi denove aperis morgaŭ.")
        before = target_ast.get("annotation_layers")
        [m] = find_pronoun_mentions(2000, target_ast)
        res = resolve_mention(m, [], entity_types)
        layer = make_annotation_layer(target_ast, m, res)
        with_annotation_layer(target_ast, layer)
        assert target_ast.get("annotation_layers") == before
