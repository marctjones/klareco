"""Grammar constraints tested beyond the benchmark sentences that revealed them."""

from dataclasses import FrozenInstanceError
import json
from unittest.mock import patch

import pytest

from klareco.conllu import ast_to_conllu, to_conllu
from klareco.morphology import analyze, best
from klareco.parser import compact_ast, expand_ast, parse, parse_word


@pytest.mark.parametrize("particle", ["malpli", "malplej", "maltro"])
def test_mal_preserves_the_class_of_uninflected_particles(particle):
    word = parse_word(particle)
    assert word["vortspeco"] == "partiklo"
    assert word["prefiksoj"] == ["mal"]
    assert word["radiko"] == particle[3:]


@pytest.mark.parametrize(
    "adverb", ["Kiam", "Kie", "Kial", "Kiel", "Tiam", "Tie", "Nenial"]
)
def test_adverbial_correlatives_do_not_occupy_the_subject_slot(adverb):
    ast = parse(f"{adverb} la knabino legas la libron?")
    assert ast["subjekto"]["kerno"]["plena_vorto"] == "knabino"
    assert ast["objekto"]["kerno"]["radiko"] == "libr"
    assert ast["vortoj"][0]["rolo"] == "advmod"


@pytest.mark.parametrize(
    "text,form",
    [
        ("Mi vidas ĉi tiun hundon.", "ĉi"),
        ("Mi vidas tiun ĉi hundon.", "ĉi"),
        ("Mi vidas iun ajn hundon.", "ajn"),
    ],
)
def test_correlative_particles_attach_to_the_correlative(text, form):
    ast = parse(text)
    word = next(w for w in ast["vortoj"] if w["plena_vorto"] == form)
    head = ast["vortoj"][word["kapo"] - 1]
    assert head["vortspeco"] == "korelativo"
    assert word["rolo"] == "advmod"
    assert any(t["rule"] == "correlative-particle-v1" for t in ast["attachment_trace"])


def test_comparison_phrase_does_not_steal_subject_of_following_clause():
    ast = parse("La knabino, kiel la knabo, estas rapida.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert ast["subjekto"]["kerno"] is words["knabino"]
    assert words["kiel"]["comparison_marker"]
    assert words["knabo"]["rolo"] not in ("nsubj", "obj")
    question = parse("Kiel la knabo kuras?")
    assert question["subjekto"]["kerno"]["radiko"] == "knab"
    assert not question["vortoj"][0].get("comparison_marker")


@pytest.mark.parametrize("text", [
    "Li estas pli juna ol ŝi.",
    "Mia patro estas pli moda ol via.",
    "Mia patro estas pli amika ol la mia.",
])
def test_nominal_comparative_ol_is_a_case_marker(text):
    ast = parse(text)
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["ol"]["vortspeco"] == "prepozicio"
    assert words["ol"]["rolo"] == "case"
    assert words["ol"].get("comparison_marker") is True
    complement = next(
        words[name] for name in ("ŝi", "via", "mia") if name in words
    )
    assert words["ol"]["kapo"] == complement["id"]


def test_finite_comparative_ol_still_opens_a_clause():
    ast = parse("Li estas pli alta ol mi estas.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["ol"]["vortspeco"] == "prepozicio"
    assert not words["ol"].get("comparison_marker")
    assert words["estas"]["rolo"] == "advcl"


def test_adverbs_modify_predicates_and_adjective_coordination_survives_projection():
    ast = parse("La domoj grandaj kaj malgrandaj ne estas novaj.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["malgrandaj"]["rolo"] == "conj"
    assert words["malgrandaj"]["kapo"] == words["grandaj"]["id"]
    assert words["ne"]["kapo"] == words["novaj"]["id"]


def test_punctuated_nominal_enumeration_is_a_coordinated_phrase():
    ast = parse("Mi rekomendas librojn, gazetojn kaj revuojn.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["gazetojn"]["rolo"] == "conj"
    assert words["gazetojn"]["kapo"] == words["librojn"]["id"]
    assert any(
        change["rule"] == "punctuated-nominal-enumeration-v1"
        for change in ast["attachment_trace"]
    )


def test_punctuated_pp_does_not_impersonate_a_nominal_enumeration():
    ast = parse("Mi legis libron, kun bildoj kaj klarigoj.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["bildoj"]["rolo"] != "conj"
    assert not any(
        change["rule"] == "punctuated-nominal-enumeration-v1"
        for change in ast["attachment_trace"]
    )


def test_punctuated_nominal_enumeration_fixes_nsubj_tail_member():
    ast = parse(
        "Ni, anoj de la tutmonda movado por la progresigo de Esperanto, "
        "direktas ĉi tiun manifeston al ĉiuj registaroj, internaciaj "
        "organizaĵoj, kaj homoj de bona volo, deklaras nian intencon."
    )
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["organizaĵoj"]["rolo"] == "conj"
    assert words["organizaĵoj"]["kapo"] == words["registaroj"]["id"]
    assert words["homoj"]["rolo"] == "conj"
    assert words["homoj"]["kapo"] == words["organizaĵoj"]["id"]
    assert any(
        change["rule"] == "punctuated-nominal-enumeration-v1"
        and change["token_id"] == words["homoj"]["id"]
        for change in ast["attachment_trace"]
    )


def test_punctuated_enumeration_does_not_rewrite_a_fragment_root():
    ast = parse(
        "del Hoyo, J., Collar, N.J., Christie, D.A., Elliott, A. kaj "
        "Fishpool, L.D.C. 2014. HBW kaj BirdLife International"
    )
    assert not any(
        change["rule"] == "punctuated-nominal-enumeration-v1"
        for change in ast["attachment_trace"]
    )
    assert expand_ast(json.loads(json.dumps(compact_ast(ast)))) == ast


def test_infinitive_attachment_crosses_intervening_particles():
    ast = parse("Mi deklaras mian intencon firmvole plu labori.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["labori"]["rolo"] == "acl"
    assert words["labori"]["kapo"] == words["intencon"]["id"]


def test_por_ke_is_a_clause_marker():
    ast = parse("Por ke ili venu, mi preparas la ĉambron.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["Por"]["rolo"] == "mark"
    assert words["Por"]["kapo"] == words["venu"]["id"]
    assert any(
        change["rule"] == "por-ke-clause-marker-v1"
        for change in ast["attachment_trace"]
    )


def test_correlative_determiner_can_cross_ajn_particle():
    ast = parse("La edukado per iu ajn etna lingvo estas ligita al perspektivo.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["iu"]["rolo"] == "det"
    assert words["iu"]["kapo"] == words["lingvo"]["id"]
    assert words["per"]["rolo"] == "case"
    assert words["per"]["kapo"] == words["lingvo"]["id"]


def test_preposition_can_govern_substantivized_adjective_before_clause_marker():
    ast = parse("Li postulas de aliaj ke ili venu.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["de"]["kapo"] == words["aliaj"]["id"]
    assert words["aliaj"]["rolo"] == "obl"


def test_clause_marker_skips_intervening_relative_predicate():
    ast = parse(
        "Por ke la personoj, kiuj havas la samajn principojn, sciu la veron."
    )
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["ke"]["kapo"] == words["sciu"]["id"]
    assert words["havas"]["rolo"] == "acl"


def test_correlative_upos_follows_dependency_role():
    subject = ast_to_conllu(parse("Kiu venis."))
    determiner = ast_to_conllu(parse("Ĉiu lingvo venas."))
    assert "\tKiu\tkiu\tPRON\t" in subject
    assert "\tĈiu\tĉiu\tDET\t" in determiner


def test_adverbial_particles_project_as_adv_but_grammatical_particles_stay_part():
    text = ast_to_conllu(parse("Mi ne nur venis, sed ĉu vi ankaŭ venis?"))
    rows = {line.split("\t")[1]: line.split("\t")[3]
            for line in text.splitlines() if line and not line.startswith("#")}
    assert rows["ne"] == "ADV"
    assert rows["nur"] == "ADV"
    assert rows["ankaŭ"] == "ADV"
    assert rows["ĉu"] == "PART"




def test_attachment_alternatives_use_final_surface_ids():
    ast = parse("Hieraŭ, mi vidis la viron en la domo.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    candidate = next(
        c
        for c in ast["syntax"]["attachment_candidates"]
        if c["token_id"] == words["domo"]["id"]
    )
    assert {c["head_id"] for c in candidate["options"]} == {
        words["viron"]["id"],
        words["vidis"]["id"],
    }
    assert {c["kapo"] for c in words["domo"]["alligo_opcioj"]} == {
        words["viron"]["id"],
        words["vidis"]["id"],
    }
    assert expand_ast(json.loads(json.dumps(compact_ast(ast)))) == ast


def test_lost_alternatives_or_false_completeness_are_rejected():
    text = "Hieraŭ, mi vidis la viron en la domo."
    ast = parse(text)
    ast["syntax"]["attachment_candidates"] = []
    with pytest.raises(ValueError, match="omitted token alternatives"):
        compact_ast(ast)
    ast = parse(text)
    ast["syntax"]["trace_coverage"] = "complete"
    with pytest.raises(ValueError, match="coverage"):
        compact_ast(ast)


def test_morphology_candidates_preserve_each_complete_surface_and_cannot_poison_cache():
    for form in ["filino", "hundodomo", "mondmilito", "esperanto"]:
        word = parse_word(form)
        for option in word.get("alternativoj", {}).get("opcioj", []):
            assert (
                "".join(m["form"] for m in option["morfemoj"])
                == option["surface"]
                == form
            )
        reading = best(form)
        with pytest.raises(FrozenInstanceError):
            reading.score = 999
        assert isinstance(reading.morphemes, tuple)
        assert reading is analyze(form)[0]
    alt = parse_word("filino")["alternativoj"]
    assert alt["aplikita"] == 0 and alt["elektita"] is None


def test_missing_morphology_and_ontology_artifacts_are_not_swallowed():
    with patch(
        "klareco.morphology.analyze", side_effect=FileNotFoundError("typed roots")
    ):
        with pytest.raises(FileNotFoundError, match="typed roots"):
            parse_word("hundino")
        parse.cache_clear()
        with pytest.raises(FileNotFoundError, match="typed roots"):
            parse("La hundino venis.")
    with patch("klareco.ontology.ontology", side_effect=FileNotFoundError("ontology")):
        with pytest.raises(FileNotFoundError, match="ontology"):
            parse_word("hundo")


@pytest.mark.parametrize("version", [True, 2, "1"])
def test_unsupported_morphology_versions_cannot_bypass_validation(version):
    ast = parse("La filino venis.")
    ast["vortoj"][1]["alternativoj"]["version"] = version
    with pytest.raises(ValueError, match="Unsupported morphology candidate version"):
        compact_ast(ast)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completeness", "complete"),
        ("selection_status", "certain"),
        ("selection_policy", "unknown"),
    ],
)
def test_morphology_cannot_claim_unrecorded_certainty(field, value):
    ast = parse("La filino venis.")
    ast["vortoj"][1]["alternativoj"][field] = value
    with pytest.raises(ValueError, match="selection metadata"):
        compact_ast(ast)


def test_conllu_is_a_pure_export_and_records_spacing_and_mood():
    ast = parse("Venu, mia amiko!")
    with patch("klareco.conllu.parse", side_effect=AssertionError("must not reparse")):
        text = ast_to_conllu(ast, strict=True)
    rows = [
        line.split("\t")
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]
    assert [(int(r[6]), r[7]) for r in rows] == [
        (w["kapo"], w["rolo"]) for w in ast["vortoj"]
    ]
    assert "Mood=Imp" in rows[0][5] and "VerbForm=Fin" in rows[0][5]
    assert "SpaceAfter=No" in rows[0][9]
    assert text.endswith("\n\n")


def test_fragment_roots_are_canonical_and_forests_are_explicit():
    ast = parse("Mia granda domo")
    root = next(w for w in ast["vortoj"] if w["kapo"] == 0)
    assert root["rolo"] == "root"
    assert ast["propozicioj"] == []
    assert "# parse_status = forest" in to_conllu("Mi, ho.")
    with pytest.raises(ValueError, match="one complete tree"):
        to_conllu("Mi, ho.", strict=True)


@pytest.mark.parametrize(
    "corruption", ["argument", "predicate", "phrase", "trace", "span", "candidate"]
)
def test_corrupt_derived_graph_fields_cannot_be_stored(corruption):
    ast = parse("Hieraŭ, mi ne estas la patro en la domo.")
    if corruption == "argument":
        ast["propozicioj"][0]["argumentoj"]["nsubj"] = []
    elif corruption == "predicate":
        ast["propozicioj"] = []
    elif corruption == "phrase":
        ast["phrases"][0]["token_ids"] = []
    elif corruption == "trace":
        ast["attachment_trace"][-1]["after"]["head_id"] = 99
    elif corruption == "span":
        ast["vortoj"][0]["normalized_span"] = [0, 999]
    else:
        ast = parse("Hieraŭ, mi vidis la viron en la domo.")
        ast["syntax"]["attachment_candidates"][0]["options"][0]["head_id"] = 99
    with pytest.raises(ValueError):
        compact_ast(ast)
