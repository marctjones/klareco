"""Storage must preserve syntax, not just token heads or surface text."""
import copy
import json

import pytest

from klareco.parser import compact_ast, expand_ast, parse


@pytest.mark.parametrize('text', [
    'Mi scias ke vi venos.',
    'La granda hundo vidas la malgrandan katon.',
    'La homo kiu vidas la hundon estas mia amiko.',
    'Maria gajnis bronzon, Petro arĝenton, kaj Jane oron.',
])
def test_json_roundtrip_preserves_complete_structure(text):
    ast = parse(text)
    stored = json.loads(json.dumps(compact_ast(ast)))
    assert expand_ast(stored) == ast


def test_preserves_extra_annotations_and_token_identity():
    ast = parse('Mi vidas vin.')
    ast['annotation_id'] = 42
    ast['annotation_idj'] = [2]
    ast['vortoj'][0]['provenance'] = {'rule': 'test', 'alternatives': []}
    restored = expand_ast(json.loads(json.dumps(compact_ast(ast))))
    assert restored == ast
    subject = restored['subjekto']
    subject = subject.get('kerno', subject)
    assert subject is restored['vortoj'][0]


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match='version'):
        expand_ast({'_ast_format': 999, 'vortoj': [], 'structure': {}})


def test_dangling_reference_is_rejected():
    stored = compact_ast(parse('Mi vidas vin.'))
    stored['structure']['subjekto'] = {'$token': 999}
    with pytest.raises(ValueError, match='reference'):
        expand_ast(stored)


def test_duplicate_token_ids_are_rejected():
    ast = parse('Mi vidas vin.')
    ast['vortoj'][1]['id'] = ast['vortoj'][0]['id']
    with pytest.raises(ValueError, match='token id'):
        compact_ast(ast)


def test_legacy_compact_is_explicitly_lossy():
    legacy = {'tipo': 'frazo', 'vortoj': [{'id': 1, 'radiko': 'mi'}],
              'subjekto_id': 1, 'verbo_id': None, 'objekto_id': None,
              'aliaj_idj': [], 'propozicioj': []}
    ast = expand_ast(legacy)
    assert ast['subjekto'] is ast['vortoj'][0]
    assert ast['_legacy_compact_lossy'] is True
    assert expand_ast(ast) == ast


def test_expansion_does_not_share_mutable_storage_data():
    stored = compact_ast(parse('Mi vidas vin.'))
    before = copy.deepcopy(stored)
    expand_ast(stored)['vortoj'][0]['radiko'] = 'changed'
    assert stored == before


def test_parse_cache_is_not_mutable_by_callers():
    text = 'La malgranda kato dormas.'
    expected = copy.deepcopy(parse(text))
    parse(text)['vortoj'][0]['radiko'] = 'changed'
    assert parse(text) == expected


def test_extractor_rejects_versioned_compact_input():
    from klareco.rag.unified_extractor import UnifiedASTExtractor
    with pytest.raises(ValueError, match='compact AST'):
        UnifiedASTExtractor().extract(compact_ast(parse('Mi vidas vin.')))


def test_legacy_dangling_reference_is_rejected():
    with pytest.raises(ValueError, match='reference'):
        expand_ast({'vortoj': [], 'subjekto_id': 3})


def test_token_override_with_same_id_is_not_discarded():
    ast = parse('Mi vidas vin.')
    ast['alternative'] = {**ast['vortoj'][0], 'rolo': 'alternative'}
    assert expand_ast(json.loads(json.dumps(compact_ast(ast)))) == ast


def test_root_cannot_be_a_token_reference():
    stored = compact_ast(parse('Mi vidas vin.'))
    stored['structure'] = {'$token': 1}
    with pytest.raises(ValueError, match='structure'):
        expand_ast(stored)


def test_unknown_envelope_fields_are_not_silently_discarded():
    stored = compact_ast(parse('Mi vidas vin.'))
    stored['annotation'] = 'would otherwise disappear'
    with pytest.raises(ValueError, match='envelope'):
        expand_ast(stored)
