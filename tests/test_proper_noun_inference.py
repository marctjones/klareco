"""Tests for proper-noun INFERENCE (#804) — no gazetteer.

The claim under test: you can decide proper-nounhood from the language's own
vocabulary and rules, without a list of the world's names. See
`docs/PROPER_NOUNS.md`.
"""

import pytest

from klareco.proper_noun_inference import (
    decomposes_to_root,
    has_foreign_orthography,
    is_abbreviation,
    is_proper_noun,
    load_roots,
)


class TestRootLexicon:
    def test_lexicon_loads_and_is_substantial(self):
        roots = load_roots()
        assert len(roots) > 2000, 'lexicon suspiciously small'

    @pytest.mark.parametrize('root', ['hund', 'urb', 'nord', 'brit', 'pres', 'ofic'])
    def test_esperanto_roots_are_present(self, root):
        assert root in load_roots()

    @pytest.mark.parametrize('name', ['zamenhof', 'shakespear', 'dalgety'])
    def test_names_are_NOT_in_the_lexicon(self, name):
        """This is the whole mechanism: names are absent from the language's
        vocabulary, so their absence IS the signal. We never list them."""
        assert name not in load_roots()


class TestDecomposition:
    """Naive final-ending stripping was not enough. `Homaranismo`, `Presejo`,
    `Oficejo`, `Britio` are ordinary DERIVED nouns whose roots were already in
    the lexicon — the stemmer just never reached them. Full affix decomposition
    took precision from 38.5% to 83.3%."""

    @pytest.mark.parametrize('word', [
        'hundo',          # root + ending
        'hundojn',        # + plural + accusative
        'presejo',        # pres + ej + o        (printing house)
        'oficejo',        # ofic + ej + o        (office)
        'britio',         # brit + i + o         (Britain)
        'homaranismo',    # homar + an + ism + o (Zamenhof's doctrine)
        'malbeleco',      # mal + bel + ec + o
        'geinstruistoj',  # ge + instru + ist + oj
    ])
    def test_derived_esperanto_words_decompose(self, word):
        assert decomposes_to_root(word), f'{word} should decompose to a known root'

    @pytest.mark.parametrize('name', ['Zamenhof', 'Varsovio', 'Dalgety', 'Makita'])
    def test_names_do_NOT_decompose(self, name):
        assert not decomposes_to_root(name)


class TestForeignOrthography:
    """Zamenhof, Lingvaj Respondoj 63 (1891): a proper name may keep its native
    orthography. 16RULES Rule 1: the alphabet is CLOSED — 28 letters. So a
    foreign spelling positively LICENSES proper-nounhood."""

    @pytest.mark.parametrize('word', ['Shakespeare', 'New York', 'Washington',
                                      'Lausanne', 'Yxkull'])
    def test_non_esperanto_orthography_is_detected(self, word):
        assert has_foreign_orthography(word)

    @pytest.mark.parametrize('word', ['Zamenhof', 'Varsovio', 'hundo', 'Ĉeĥio'])
    def test_esperanto_orthography_is_clean(self, word):
        assert not has_foreign_orthography(word)


class TestAbbreviations:
    """`D-ro`, `L.` — a separate token class, not a proper-noun question. They
    were 6 of the 13 remaining false positives."""

    @pytest.mark.parametrize('tok', ['D-ro', 'L.', 'L', 'S-ro'])
    def test_abbreviations_recognised(self, tok):
        assert is_abbreviation(tok)

    def test_a_real_name_is_not_an_abbreviation(self):
        assert not is_abbreviation('Zamenhof')


class TestInference:
    def test_a_name_mid_sentence_is_inferred(self):
        assert is_proper_noun('Zamenhof', prev_token='fondis')

    def test_a_common_noun_is_not(self):
        assert not is_proper_noun('hundo', prev_token='la')

    def test_a_capitalised_derived_noun_is_not_a_name(self):
        """`Presejo` is a printing house, not a person — and the lexicon knows
        `pres` even though it has never seen `presejo`."""
        assert not is_proper_noun('Presejo', prev_token='la')

    def test_foreign_orthography_wins_ANYWHERE(self):
        """Position does not veto orthography: a foreign spelling is evidence
        even sentence-initially (Zamenhof LR63)."""
        assert is_proper_noun('Shakespeare', is_sentence_initial=True)

    def test_all_caps_carries_no_capitalisation_signal(self):
        assert not is_proper_noun('EDUKADO', prev_token='.')

    def test_sentence_initial_capitalisation_is_uninformative(self):
        """EVERY sentence starts with a capital, so it proves nothing.

        This costs recall (`Varsovio` at position 1 is missed) and buys a lot of
        precision — dropping the rule took precision 83% -> 46%. The honest fix
        is a bigger eval set (#820), not a guess.
        """
        assert not is_proper_noun('Varsovio', is_sentence_initial=True)

    def test_abbreviations_are_excluded(self):
        assert not is_proper_noun('D-ro', prev_token='la')

    def test_lowercase_is_never_a_proper_noun(self):
        assert not is_proper_noun('zamenhof', prev_token='fondis')
