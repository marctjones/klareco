"""ROUND-TRIP: parse(text) -> deparse -> text. (#833)

    If text' != text, the AST LOST SOMETHING.

This is the cheapest possible check that the AST is complete, and we did not have
it. Everything it caught had been sitting in the code for months, silently:

    la              ABSORBED into the vortgrupo and never emitted (9.1% of LAS)
    enhavas   ->    enohavas      a linking vowel inserted where there is none
    jarcento  ->    jarocento     same
    devenas   ->    dedeovenas    the prefix counted TWICE
    1-a       ->    a             the DIGIT dropped from an ordinal
    Enhavas   ->    PROPER NOUN   a capitalized PREFIXED verb, mis-tagged
    mondmilito->    mond+MIL+it   "world-thousand-ed", not "world war"

The deparser had 21 passing tests while producing

    "Perfekta perfekta Esperanto estas ne ĝi funkcias bone."

for "Kvankam Esperanto ne estas perfekta, ĝi funkcias bone." — duplicated,
reordered, and `Kvankam` deleted. The tests passed because they never checked the
one property that matters.

MEASURED over real corpus sentences: 59.9% -> 80.5% exact round-trip.

⚠️ Note the deparser reconstructs the SURFACE from `vortoj`. For GENERATION from
the tree (VISION.md's "grammatically correct by construction") see
`deparse_structural` — Esperanto's word order is FREE, so that path legitimately
produces a different string and CANNOT be tested by string equality. Testing it
that way is exactly how the old tests stayed green.
"""

import re

import pytest

from klareco.deparser import deparse
from klareco.parser import parse


def _norm(x: str) -> str:
    x = re.sub(r'[.,;:!?«»"„()\[\]]', ' ', x)
    return re.sub(r'\s+', ' ', x).strip().lower()


def _roundtrips(s: str) -> bool:
    return _norm(deparse(parse(s))) == _norm(s)


class TestNothingIsLost:
    @pytest.mark.parametrize('s', [
        'La hundo vidis la katon.',
        'La granda hundo ne vidis la katon en la ĝardeno.',
        'Zamenhof fondis Esperanton kaj li skribis librojn.',
        'Kvankam Esperanto ne estas perfekta, ĝi funkcias bone.',
        'La domo estas granda.',
        'Maria gajnis bronzon, Petro arĝenton, kaj Jane oron.',
    ])
    def test_it_comes_back(self, s):
        assert _roundtrips(s), f'the AST LOST something:\n  in : {s}\n  out: {deparse(parse(s))}'

    def test_the_ARTICLE_comes_back(self):
        """`la` used to be ABSORBED into the vortgrupo as an attribute and simply
        VANISH — 13 tokens in, 10 out. 9.1% of LAS, and no test caught it."""
        assert deparse(parse('La hundo vidis la katon.')).lower().count('la ') == 2


class TestTheLinkingVowelIsRecorded:
    """It is OPTIONAL in Esperanto — `hundodomo` has one, `mondmilito` does not.
    The old AST did not record it, so the deparser GUESSED and always inserted an
    `o`."""

    @pytest.mark.parametrize('word', ['enhavas', 'jarcento', 'mondmilito', 'devenas'])
    def test_compounds_without_a_linking_vowel(self, word):
        assert _roundtrips(f'Mi vidis {word}.') or _roundtrips(word.capitalize() + '.')

    def test_a_compound_WITH_a_linking_vowel_also_survives(self):
        from klareco.morphology import analyze
        a = analyze('hundodomo')[0]
        assert a.surface == 'hundodomo'
        assert a.kunmetitaj_radikoj == ['hund', 'dom']


class TestMorphologyIsInvertible:
    """`Analysis.surface` rebuilds the exact word from its morphemes. If it does
    not, the decomposition is wrong — and that is a free, corpus-wide check."""

    @pytest.mark.parametrize('word', [
        'hundo', 'papero', 'esperanto', 'organo', 'amerikano', 'refari',
        'mondmilito', 'hundodomo', 'jarcento', 'enhavas', 'malbela',
    ])
    def test_the_surface_rebuilds_exactly(self, word):
        from klareco.morphology import analyze
        a = analyze(word)
        assert a and a[0].surface == word, \
            f'{word} -> {a[0].surface if a else None}'


class TestTheBugsItCaught:
    def test_a_capitalized_PREFIXED_verb_is_not_a_proper_noun(self):
        """`Enhavas` = en+hav. The capitalization guard had NO VERB BRANCH, so a
        prefixed verb's stem was not a whole root and it fell through to
        'unknown -> proper noun'. Bare verbs (`Venis`) survived because their stem
        IS a root — which is why this went unnoticed."""
        from klareco.parser import parse_word
        for w in ('Enhavas', 'Elvenis', 'Alvenis', 'Rekonstruis'):
            assert parse_word(w)['vortspeco'] == 'verbo', f'{w} mis-tagged'

    def test_mondmilito_is_a_WAR_not_a_thousand(self):
        """`mondmilito` was analysed as mond + MIL + it — "world-thousand-ed".
        morphology.py had NO COMPOUND SUPPORT, so it returned nothing and the
        parser's wrong analysis stood."""
        from klareco.parser import parse_word
        a = parse_word('mondmilito')
        assert a['radiko'] == 'milit'
        assert a['kunmetitaj_radikoj'] == ['mond', 'milit']

    def test_an_ordinal_keeps_its_DIGIT(self):
        """`1-a` (the ordinal "1st") came back as bare `a`. Digits are surface,
        not morphemes."""
        assert _roundtrips('La 1-a libro estas granda.')
