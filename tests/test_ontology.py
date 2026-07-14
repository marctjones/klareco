"""The ontology — CURATED, not hand-seeded, and finally not empty. (#830)

CLAUDE.md, on this project's own ontology:

    "`ontology_nodes` and `ontology_edges` are EMPTY and `verb_klaso` is 0%
     populated ... the 'always query the ontology' rule is UNFOLLOWABLE, and a
     couple of paths fall back to hardcoded lists. That is ACKNOWLEDGED DEBT."

and, honestly:

    "even when loaded, the ontology is hand-seeded and THIN (`persono` =
     ["homo","vir","infan","kuracist"]) ... Lexical synonymy is a genuine learned
     residue we are currently FAKING WITH A LIST."

It is not a learned residue. ReVo ships it, curated by lexicographers, GPL-2.0:

    8,709 hypernym edges · 2,984 synonyms · 22,770 domain labels (78 distinct)
    133 typed entity lists · 40,230 senses
"""

import pytest

from klareco.ontology import ontology
from klareco.parser import parse_word


class TestTheClassesAreRealNow:
    """`persono` was FOUR hand-picked roots. ReVo has 377, attested."""

    def test_persono_is_curated_not_hand_seeded(self):
        m = ontology().members('persono')
        assert len(m) > 300, f'only {len(m)} — the hand-seeded version had 4'
        assert 'Abraham' in m

    def test_loko_replaces_the_gazetteer(self):
        """CLAUDE.md: 'Files That Should NOT Exist: ❌ *_gazetteer.py'."""
        m = ontology().members('loko')
        assert len(m) > 600
        assert 'Abudabi' in m

    def test_is_a_answers_the_question_the_gazetteer_was_for(self):
        o = ontology()
        assert o.is_a('Abraham', 'persono')
        assert not o.is_a('Abraham', 'loko')


class TestTheTaxonomy:
    def test_hypernyms_exist(self):
        """8,709 <ref tip="super"> edges — a real taxonomy, not four roots."""
        o = ontology()
        assert any(o.hypernyms(r) for r in ('hund', 'kat', 'ĉeval'))

    def test_domains(self):
        assert 'ZOO' in ontology().domains('hund')

    def test_synonyms_are_CURATED(self):
        """Not a hand-written dict. From ReVo's <ref tip="sin">."""
        o = ontology()
        assert any(o.synonyms(r) for r in ('hund', 'dom', 'grand'))


class TestSenseLevelORNodes:
    """The THIRD level of the forest. Same node, same fonto/kialo machinery —
    only `nivelo` changes:

        morfemo   papero = paper|o  OR  pap|er|o
        alligo    "kun teleskopo" attaches to VIDIS  OR  to VIRON
        senco     `hundo` = the ANIMAL  OR  an INSULT for an aggressive man
    """

    def test_hundo_is_polysemous_and_we_ADMIT_it(self):
        """ReVo gives `hund` three senses: the genus, the domestic animal, and
        an insult for an aggressive man.

        WHICH ONE IS MEANT IS NOT A GRAMMATICAL QUESTION. Nothing in this parser
        answers it, so we do not pretend to — fonto=None, and the OR-node stands.
        """
        sc = parse_word('hundo')['sencoj']
        assert len(sc['opcioj']) == 3
        assert sc['fonto'] is None, 'no rule can choose — say so'
        assert sc['elektita'] is None
        assert sc['nivelo'] == 'senco'

    def test_a_MONOSEMOUS_word_gets_no_OR_node(self):
        """Do not pay for ambiguity you do not have."""
        a = parse_word('tablo')
        assert 'sencoj' not in a
        assert a.get('senco')                      # but the sense IS recorded

    def test_the_definitions_are_readable(self):
        """`<tld/>` is ReVo's placeholder for the root. Stripping it naively gave
        'Ago i aŭ ties rezulto' — gibberish. It must be SUBSTITUTED, not deleted."""
        d = ontology().senses('hund')[0]
        assert 'hundedoj' in d, 'the <tld/> root substitution is broken'
        assert '&' not in d, 'HTML entities are not decoded'
