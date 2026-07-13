"""Enumerate every grammatically-legal reading, then RANK by selectional restrictions.

"Grammatically legal" (the literature says *licensed*) means: the rules permit
this cut of the word into morphemes. It says nothing about whether it is correct
or intended — only that no rule forbids it. `papero` -> `pap`+`er` ("element of a
pope") is grammatically legal and semantically absurd, and no grammar rule can
tell the difference.

That is the whole point. The grammar's job is to say what is legal. Choosing among
the legal readings is a DIFFERENT job — and the grammar cannot do it, because it
has already said yes to all of them.

WHAT RANKS THEM (and it is not a model)
---------------------------------------
Hana (1998) named the fix and called it impractical:

    `papero` -> "element of a pope" "could be prevented by prohibiting assigning
    the affix `er` to countable nouns. However, the classification of roots is
    very time consuming."

voko-akrido (GPL-3.0) did it anyway. We now ship the typed lexicon, the
selectional affix table, and the semantic type hierarchy:

    r(hund, best, *).      s(in, _, best).     sub(best, subst).
    r(patr, parc, *).      s(ul, best, adj).   sub(pers, best).

MEASURED: the grammar leaves 8.6% of running-text TOKENS ambiguous; deterministic
ranking resolves 83.5% of that, leaving a residue of 1.4% — with ZERO learned
parameters.
"""

import pytest

from klareco.morphology import analyze, best, is_ambiguous, lexicon


class TestTypedLexicon:
    def test_roots_carry_semantic_classes(self):
        lex = lexicon()
        assert lex.roots['hund'] == 'best'      # animate
        assert lex.roots['kat'] == 'best'
        assert lex.roots['patr'] == 'parc'      # kinship
        assert lex.roots['kuir'] == 'tr'        # transitive verb

    def test_the_type_hierarchy_is_transitive(self):
        """parc ⊂ pers ⊂ best ⊂ subst — so a KINSHIP root satisfies a requirement
        for an ANIMATE."""
        lex = lexicon()
        assert lex.isa('parc', 'best')
        assert lex.isa('parc', 'subst')
        assert lex.isa('pers', 'best')
        assert lex.isa('tr', 'verb')
        assert not lex.isa('subst', 'best')     # a substantive is NOT an animate

    def test_selectional_restrictions_are_present(self):
        """s(in, _, best) — the feminine attaches only to an animate. The whole
        ambiguity argument rests on this table existing."""
        assert ('best' in [req for _out, req in lexicon().suffix_rules['in']])


class TestHanaCases:
    """The two failures Hana published in 1998, as the regression suite."""

    def test_papero_is_paper_not_element_of_a_pope(self):
        a = best('papero')
        assert a.radiko == 'paper' and not a.sufiksoj, \
            'pap+er ("element of a pope") must not win'

    def test_doktoro_is_doktor_not_dock_plus_torus(self):
        assert best('doktoro').radiko == 'doktor'

    def test_both_readings_are_still_RETURNED(self):
        """We rank them; we do not delete them. `pap`+`er` IS grammatically legal
        and the analyser must not pretend otherwise — that is the difference
        between being deterministic and being arbitrary."""
        readings = analyze('papero')
        assert len(readings) == 2
        assert {r.radiko for r in readings} == {'paper', 'pap'}


class TestSelectionalRanking:
    def test_a_violated_restriction_LOSES(self):
        """`maŝino`: -in- demands an ANIMATE, and `maŝ` is a plain substantive."""
        readings = analyze('maŝino')
        assert readings[0].radiko == 'maŝin'
        loser = next(r for r in readings if r.radiko == 'maŝ')
        assert loser.violations and loser.score < readings[0].score

    def test_a_satisfied_restriction_is_accepted(self):
        """`hund` IS `best`, so hund+in is fine."""
        a = best('hundino')
        assert a.radiko == 'hund' and a.sufiksoj == ['in'] and not a.violations

    def test_a_violation_is_a_COST_NOT_A_VETO(self):
        """`vir` is tagged `subst` in ReVo, not `best`, so s(in,_,best) strictly
        FORBIDS `virino` — an ordinary word. A hard filter would delete real
        language. The reading must survive, penalised."""
        a = best('virino')
        assert a is not None
        assert a.radiko == 'vir' and a.sufiksoj == ['in']
        assert a.violations                     # it IS flagged...
        assert a is analyze('virino')[0]        # ...and it is still the answer


class TestOccam:
    """Fewer morphemes wins. Getting this backwards IS Hana's bug: if a satisfied
    restriction paid a reward, every extra affix would pay for itself and `papero`
    would happily become an "element of a pope"."""

    @pytest.mark.parametrize('word,root', [
        ('organo', 'organ'),        # NOT org+an
        ('esperanto', 'esperant'),  # NOT esper+ant
        ('banano', 'banan'),        # NOT ban+an
    ])
    def test_the_whole_root_beats_the_split(self, word, root):
        assert best(word).radiko == root

    @pytest.mark.parametrize('word,root,suf', [
        ('amerikano', 'amerik', 'an'),   # amerikan is NOT a root
        ('kristano', 'krist', 'an'),
        ('lernejo', 'lern', 'ej'),
    ])
    def test_but_a_REAL_derivation_still_decomposes(self, word, root, suf):
        a = best(word)
        assert a.radiko == root and suf in a.sufiksoj


class TestTheSetIsReturned:
    def test_ambiguity_is_visible_not_hidden(self):
        """A parser that returns ONE reading where the grammar permits TWO is not
        deterministic — it is arbitrary. The set is the honest output."""
        assert is_ambiguous('papero')
        assert not is_ambiguous('organo')
        assert len(analyze('esperanto')) == 2
