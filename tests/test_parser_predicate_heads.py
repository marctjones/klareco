"""Copular noun phrases are headed by their noun, not its adjective."""
import pytest

from klareco.parser import parse


@pytest.mark.parametrize('text, predicate, modifier', [
    ('La domo estas granda konstruaĵo.', 'konstruaĵo', 'granda'),
    ('La virino estas bona instruisto.', 'instruisto', 'bona'),
    ('La urbo estas granda aŭ malgranda loko.', 'loko', 'granda'),
    ('La domo estas granda kaj bela konstruaĵo.', 'konstruaĵo', 'granda'),
])
def test_noun_heads_predicate_phrase(text, predicate, modifier):
    words = {w['plena_vorto']: w for w in parse(text)['vortoj']}
    head = words[predicate]
    assert (head['kapo'], head['rolo']) == (0, 'root')
    assert (words['estas']['kapo'], words['estas']['rolo']) == (head['id'], 'cop')
    assert (words[modifier]['kapo'], words[modifier]['rolo']) == (head['id'], 'amod')


@pytest.mark.parametrize('text', [
    'La domo estas granda.',
    'La domo estas granda kaj bela.',
    'La domo estas granda en la urbo.',
])
def test_adjective_remains_predicate_without_a_nominal_head(text):
    words = {w['plena_vorto']: w for w in parse(text)['vortoj']}
    assert (words['granda']['kapo'], words['granda']['rolo']) == (0, 'root')


def test_participle_remains_head_with_a_following_subject():
    words = {w['plena_vorto']: w for w in parse('Estas aranĝita la kongreso.')['vortoj']}
    assert (words['aranĝita']['kapo'], words['aranĝita']['rolo']) == (0, 'root')
    assert (words['Estas']['kapo'], words['Estas']['rolo']) == (words['aranĝita']['id'], 'aux')
