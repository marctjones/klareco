"""Dependencies, projections, and stored/immutable snapshots must agree."""

import json
from copy import deepcopy
import pytest
from klareco.parser import parse, compact_ast, expand_ast
from klareco.syntax_graph import validate_tokens
from klareco.orchestrator.context import QueryContext, ContextDelta, ParsedPassage


@pytest.mark.parametrize(
    "text",
    [
        "La homo kiu venis estas mia amiko.",
        "La viro kiu venis hieraŭ ne estas mia frato, sed li konas ŝin.",
        "Li provis ĉesi fumi.",
        "Mi venus se vi invitus min.",
        "Sam, malfermu la pordon!",
        "Ŝi diris ke li kredas ke mi venos.",
        "Kiu vidis la hundon?",
        "Li vidis la hundon kun teleskopo.",
    ],
)
def test_graph_and_frames_agree(text):
    ast = parse(text)
    registry = {w["id"]: w for w in ast["vortoj"]}
    validate_tokens(ast["vortoj"])
    for clause in ast["propozicioj"]:
        predicate = clause["predikato"]["id"]
        for slot, role in [("subjekto", "nsubj"), ("objekto", "obj")]:
            group = clause[slot]
            if group:
                token = group["kerno"]
                assert token is registry[token["id"]]
                assert token["kapo"] == predicate
                assert token["rolo"].split(":")[0] == role
                assert all(
                    (w["predikato"] if w.get("tipo") == "rilata_subfrazo" else w)[
                        "kapo"
                    ]
                    == token["id"]
                    for w in group["priskriboj"]
                )
    assert expand_ast(json.loads(json.dumps(compact_ast(ast)))) == ast


def test_interrupted_subject_and_possessive_predicate():
    ast = parse("La homo kiu venis estas mia amiko.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["homo"]["kapo"] == words["amiko"]["id"]
    assert words["mia"]["kapo"] == words["amiko"]["id"]
    assert ast["subjekto"]["kerno"] is words["homo"]
    assert words["mia"] not in ast["subjekto"]["priskriboj"]


@pytest.mark.parametrize(
    "tokens",
    [
        [{"id": 1, "kapo": 2, "rolo": "obj"}],
        [{"id": 1, "kapo": 1, "rolo": "root"}],
        [{"id": 1, "kapo": 2, "rolo": "obj"}, {"id": 2, "kapo": 1, "rolo": "obj"}],
        [{"id": 1, "kapo": 0, "rolo": "root"}, {"id": 1, "kapo": 0, "rolo": "root"}],
    ],
)
def test_invalid_dependencies_fail(tokens):
    with pytest.raises(ValueError):
        validate_tokens(tokens)


def test_nested_context_snapshots_are_owned_and_read_only():
    ast = parse("Mi amas vin.")
    context = QueryContext("test").apply(
        ContextDelta(symbolic={"question_ast": ast}, flags={"nested": {"items": [1]}})
    )
    snapshot = context.symbolic.question_ast
    ast["vortoj"][0]["radiko"] = "corrupted"
    assert snapshot["vortoj"][0]["radiko"] == "mi"
    for mutation in [
        lambda: snapshot.update(x=1),
        lambda: snapshot["vortoj"].pop(),
        lambda: snapshot["vortoj"][0].__setitem__("radiko", "x"),
        lambda: context.flags["nested"]["items"].append(2),
    ]:
        with pytest.raises(TypeError, match="immutable"):
            mutation()
    editable = deepcopy(snapshot)
    editable["vortoj"].clear()
    assert snapshot["vortoj"]
    next_context = context.apply(ContextDelta(flags={"other": True}))
    assert next_context.symbolic is context.symbolic
    passage = ParsedPassage("1", "test", ast, 1.0, "test", "test")
    with pytest.raises(TypeError):
        passage.ast.clear()


def test_latent_input_cannot_mutate_a_context():
    import numpy as np
    from klareco.orchestrator.context import LatentLayer

    source = np.array([1.0, 2.0])
    context = QueryContext("test", latent=LatentLayer(question_embedding=source))
    source[0] = 9.0
    assert context.latent.question_embedding.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        context.latent.question_embedding.flags.writeable = True


@pytest.mark.parametrize(
    "text", ["Mi sxatas la domon.", "İstiklal-avenuo en Istanbulo.", "  Mi — venas."]
)
def test_source_spans_survive_length_changing_normalization(text):
    ast = parse(text)
    assert ast["source"]["original"] == text
    for token in ast["vortoj"]:
        start, end = token["normalized_span"]
        assert ast["source"]["normalized"][start:end] == token["surface_form"]
        first, last = token["original_span"]
        assert 0 <= first < last <= len(text)


def test_nested_complement_has_its_own_governor():
    ast = parse("Ŝi diris ke li kredas ke mi venos.")
    words = {w["plena_vorto"]: w for w in ast["vortoj"]}
    assert words["venos"]["kapo"] == words["kredas"]["id"]
    assert words["kredas"]["kapo"] == words["diris"]["id"]


@pytest.mark.parametrize(
    "text",
    [
        "Pri “kies” “Kies” povas signifi de kiu, ĉar ĝi estas vorto.",
        "Objekta kies Rilata kies povas montri la sencan objekton de aga O-vorto.",
    ],
)
def test_mentioned_possessives_do_not_create_relative_dependency_cycles(text):
    validate_tokens(parse(text)["vortoj"])


def test_document_sized_input_fails_before_expensive_parsing():
    from klareco.parser import MAX_SENTENCE_CHARACTERS

    with pytest.raises(ValueError, match="segment document text"):
        parse("x" * (MAX_SENTENCE_CHARACTERS + 1))


def test_storage_rejects_corrupted_dependencies_and_views():
    ast = parse("Mi amas vin.")
    packed = compact_ast(ast)
    packed["vortoj"][0]["kapo"] = packed["vortoj"][0]["id"]
    with pytest.raises(ValueError, match="cycle"):
        expand_ast(packed)
    ast["propozicioj"][0]["subjekto"] = ast["objekto"]
    with pytest.raises(ValueError, match="argument contradicts"):
        compact_ast(ast)


def test_prepositional_noun_does_not_take_the_resumed_subject_slot():
    ast = parse('Por ke ili sciu, la adresoj de la homoj estos publikigataj.')
    words = {w['plena_vorto']: w for w in ast['vortoj']}
    assert ast['subjekto']['kerno'] is words['adresoj']
    assert words['homoj']['kapo'] == words['adresoj']['id']
    assert words['homoj']['rolo'] == 'nmod'


def test_postverbal_relative_subject_stays_inside_the_relative_clause():
    ast = parse('La nomoj, kiujn portas multaj landoj de la mondo, estas problemaj.')
    words = {w['plena_vorto']: w for w in ast['vortoj']}
    assert words['landoj']['kapo'] == words['portas']['id']
    assert ast['subjekto']['kerno'] is words['nomoj']


def test_coordination_before_the_main_predicate_stays_in_its_relative_clause():
    ast = parse('La sistemo kiu favoras iujn sed ekskludas aliajn estas maljusta.')
    words = {w['plena_vorto']: w for w in ast['vortoj']}
    assert words['ekskludas']['kapo'] == words['favoras']['id']
    assert words['ekskludas']['rolo'] == 'conj'


def test_query_text_keeps_embedded_clause_words_and_surface_order():
    from klareco.rag.duckdb_retriever import DuckDBRetriever
    text = 'Kiu verkis la libron kiu nomiĝas La Vojo?'
    assert DuckDBRetriever._question_text(parse(text)) == text


def test_question_type_uses_the_main_clause_token_registry():
    from klareco.orchestrator.stages.parse_question import _classify_from_ast
    assert _classify_from_ast(parse('Ĉu vi scias kiu venis?')) == 'ĉu'
