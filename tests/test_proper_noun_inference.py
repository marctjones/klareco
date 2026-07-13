"""Tests for proper-noun INFERENCE (#804) — no gazetteer.

The claim under test: you can decide proper-nounhood from the language's own
vocabulary and rules, without a list of the world's names. See
`docs/PROPER_NOUNS.md`.
"""

import pytest

from klareco.proper_noun_inference import (
    adjective_reading_is_licensed,
    decomposes_to_root,
    has_foreign_orthography,
    implied_pos,
    is_abbreviation,
    is_proper_noun,
    is_valid_esperanto_word,
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

    @pytest.mark.parametrize('name', ['Zamenhof', 'Varsovio', 'Dalgety'])
    def test_names_do_NOT_decompose(self, name):
        assert not decomposes_to_root(name)

    def test_lexicon_CONTAMINATION_makes_a_name_decompose(self):
        """`Makita` decomposes to `mak-it-a` — but only because `mak` leaked into
        the corpus-harvested lexicon. `mak` is NOT in the Fundamento.

        This is not a rule error, it is the PURITY limit: the lexicon is
        harvested from a degraded parser that over-tags `propra_nomo`, so names
        contaminate it, and a contaminated root then "explains" the very name it
        came from. It is the case for a curated lexicon (#806) — quality, not
        size. Pinned as a known limitation so a future ReVo import flips it.
        """
        assert decomposes_to_root('Makita')       # ← wrong, and we know why

    @pytest.mark.parametrize('word', [
        'konsciante',    # konsci + ant + e   — Rule 6 participle
        'planita',       # plan  + it  + a
        'hispanlando',   # hispan + land + o  — compound
        'plurlingveco',  # plur + lingv + ec + o
        'subskribo',     # sub + skrib + o    — prepositional prefix
        'antaŭparolo',   # antaŭ + parol + o
    ])
    def test_participles_compounds_and_prepositional_prefixes(self, word):
        """These are ordinary Esperanto word-formation. Every one of them was
        undecomposable — and therefore a FALSE NAME — until participles (Rule 6),
        root+root compounding, and prepositional prefixes were added."""
        assert decomposes_to_root(word)


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


class TestEndingValidity:
    """Rules 2-7: every CONTENT word carries a grammatical ending. `sam` is a
    ROOT; `sama` is a WORD. A bare root is not a word form at all.

    This was a real bug: `decomposes_to_root` matched a bare root, so `Sam` and
    `Peter` (`pet`+`er` — a suffix, not an ending) were accepted as ordinary
    Esperanto words and could never be recognised as names.
    """

    @pytest.mark.parametrize('word', ['hundo', 'sama', 'peti', 'Petro', 'presejo'])
    def test_well_formed_words(self, word):
        assert is_valid_esperanto_word(word)

    @pytest.mark.parametrize('name', ['Sam', 'Peter', 'Zamenhof', 'Varsovio'])
    def test_bare_roots_and_names_are_NOT_word_forms(self, name):
        assert not is_valid_esperanto_word(name)

    @pytest.mark.parametrize('fw', ['la', 'kaj', 'mi', 'tiu', 'de', 'ankaŭ'])
    def test_the_closed_ending_less_class_is_still_valid(self, fw):
        """The grammar names its own exceptions: particles, prepositions,
        correlatives, numerals. Without this, `La` at sentence-start has no
        ending and would be declared a name."""
        assert is_valid_esperanto_word(fw)

    @pytest.mark.parametrize('fw', ['La', 'Kaj', 'Tiu', 'Ĉiu', 'Mi'])
    def test_function_words_are_never_names(self, fw):
        """`La` and `Kaj` merely END in `-a`. They are not adjectives, so the
        agreement rule must not fire on them."""
        assert not is_proper_noun(fw, next_token='estas', is_sentence_initial=True)

    @pytest.mark.parametrize('word,pos', [
        ('hundo', 'substantivo'), ('Maria', 'adjektivo'),
        ('Jane', 'adverbo'), ('legas', 'verbo'), ('Zamenhof', None),
    ])
    def test_the_ending_declares_the_part_of_speech(self, word, pos):
        assert implied_pos(word) == pos


class TestAdjectiveAgreement:
    """Rule 3: an adjective agrees with its head noun in NUMBER and CASE.

    This is the signal token-internal morphology provably CANNOT see — it lives
    in the rest of the sentence. `Maria` really does decompose to `mar-i-a`
    ("of the sea"), and in isolation that reading is legitimate. A following verb
    kills it: an adjective cannot be a subject.
    """

    def test_an_adjective_with_a_head_noun_is_an_ordinary_adjective(self):
        assert adjective_reading_is_licensed('Centra', 'la', 'Oficejo')
        assert not is_proper_noun('Centra', prev_token='la', next_token='Oficejo')

    def test_an_adjective_with_NOTHING_to_agree_with_is_a_name(self):
        """`Maria gajnis bronzon` — the SENTENCE rules out the common reading
        that morphology alone accepts."""
        assert not adjective_reading_is_licensed('Maria', None, 'gajnis')
        assert is_proper_noun('Maria', next_token='gajnis')

    def test_agreement_is_checked_on_number_and_case(self):
        assert adjective_reading_is_licensed('belajn', None, 'hundojn')
        assert not adjective_reading_is_licensed('bela', None, 'hundojn')

    def test_the_head_noun_may_PRECEDE(self):
        """Esperanto word order is free; `domo granda` is as good as
        `granda domo`."""
        assert adjective_reading_is_licensed('Granda', 'domo', 'estas')


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
        """EVERY sentence starts with a capital, so capitalisation proves nothing
        there — and a token that IS an ordinary Esperanto word gets no help from
        being first."""
        assert not is_proper_noun('Presejo', is_sentence_initial=True)
        assert not is_proper_noun('Deklaracio', is_sentence_initial=True)

    def test_position_does_NOT_veto_grammar(self):
        """But position only vetoes the CAPITALISATION rule, which is evidential.

        The grammar rules are deductive and hold everywhere. `Varsovio` is not a
        well-formed Esperanto word (Rules 2-7), and that is true at position 1
        exactly as it is mid-sentence. This used to be a forced miss."""
        assert is_proper_noun('Varsovio', is_sentence_initial=True)
        assert is_proper_noun('Zamenhof', is_sentence_initial=True)

    def test_abbreviations_are_excluded(self):
        assert not is_proper_noun('D-ro', prev_token='la')

    def test_lowercase_is_never_a_proper_noun(self):
        assert not is_proper_noun('zamenhof', prev_token='fondis')
