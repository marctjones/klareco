"""Every token gets a HEAD and a ROLE. `aliaj` is no longer a bucket. (#825)

The AST modelled `subjekto`, `verbo`, `objekto` and dumped everything else into
`aliaj` — a flat list, no heads, no roles. Worse, some tokens were ABSORBED and
never emitted at all: `la` was folded into the `vortgrupo` as an attribute and
simply vanished. Thirteen tokens in, ten in the AST.

MEASURED COST on UD gold (share of LAS -> our recall):
    det   (`la` -> its noun)      9.1%  ->  0.0%
    cc    (coordinator)           5.3%  ->  0.0%
    mark  (subordinator)          3.1%  ->  0.0%
    aux+cop (`estas`)             2.7%  ->  0.0%
                                 ~20%   of LAS scored ZERO

We were not attaching these wrongly. We were not emitting them.
"""

import pytest

from klareco.conllu import to_conllu
from klareco.parser import parse


def _rows(sentence):
    out = []
    for line in to_conllu(sentence).split('\n'):
        if line and not line.startswith('#'):
            x = line.split('\t')
            if len(x) >= 8 and x[0].isdigit():
                out.append({'id': int(x[0]), 'form': x[1], 'upos': x[3],
                            'head': int(x[6]), 'dep': x[7]})
    return out


def _dep(rows, form):
    return next(r for r in rows if r['form'].lower() == form.lower())


class TestNoTokenIsLost:
    def test_every_input_token_appears_in_the_AST(self):
        """`La granda hundo ne vidis la katon en la ĝardeno, ĉar ĝi dormis` used
        to lose all three `la`."""
        s = 'La granda hundo ne vidis la katon en la ĝardeno, ĉar ĝi dormis.'
        rows = _rows(s)
        forms = [r['form'].lower() for r in rows]
        assert forms.count('la') == 3, 'the articles are being absorbed and lost'
        assert len(rows) == 13

    def test_every_token_has_a_head_and_a_role(self):
        for w in parse('La hundo ne vidis la katon en la ĝardeno.')['vortoj']:
            assert w.get('kapo') is not None, f"{w.get('plena_vorto')} has no head"
            assert w.get('rolo'), f"{w.get('plena_vorto')} has no role"


class TestTheRelationsWeScoredZeroOn:
    def test_det(self):
        """9.1% of LAS, and we scored 0.0% — `la` was not emitted at all."""
        rows = _rows('La hundo vidis la katon.')
        assert _dep(rows, 'La')['dep'] == 'det'
        assert _dep(rows, 'La')['head'] == _dep(rows, 'hundo')['id']

    def test_mark(self):
        """Subordinators open a clause and attach to ITS verb."""
        rows = _rows('Li venis, ĉar li amis ŝin.')
        assert _dep(rows, 'ĉar')['dep'] == 'mark'
        assert _dep(rows, 'ĉar')['head'] == _dep(rows, 'amis')['id']

    def test_cc(self):
        rows = _rows('Li venis kaj ŝi foriris.')
        assert _dep(rows, 'kaj')['dep'] == 'cc'

    def test_advmod(self):
        rows = _rows('La hundo ne vidis la katon.')
        assert _dep(rows, 'ne')['dep'] == 'advmod'
        assert _dep(rows, 'ne')['head'] == _dep(rows, 'vidis')['id']

    def test_case_attaches_the_ADPOSITION_to_its_noun(self):
        """UD's convention, not ours: the preposition depends on the noun."""
        rows = _rows('La hundo estas en la ĝardeno.')
        assert _dep(rows, 'en')['dep'] == 'case'
        assert _dep(rows, 'en')['head'] == _dep(rows, 'ĝardeno')['id']


class TestCopula:
    """`Esperanto estas lingvo` — UD makes the PREDICATE the root and `estas` a
    `cop` child of it. We were making `estas` the root, which is not merely a
    label difference: it dragged `root` accuracy down, mis-attached the subject,
    and scored 0.0% on aux+cop.

    Esperanto has no auxiliary class — `esti` is an ordinary verb — so this is a
    genuine SCHEME difference. We adopt UD's view for comparability and keep our
    native analysis in MISC.
    """

    def test_the_PREDICATE_is_the_root_and_estas_is_the_cop(self):
        rows = _rows('Esperanto estas lingvo internacia.')
        assert _dep(rows, 'lingvo')['dep'] == 'root'
        assert _dep(rows, 'lingvo')['head'] == 0
        assert _dep(rows, 'estas')['dep'] == 'cop'
        assert _dep(rows, 'estas')['head'] == _dep(rows, 'lingvo')['id']

    def test_the_subject_attaches_to_the_PREDICATE_not_the_copula(self):
        rows = _rows('Esperanto estas lingvo.')
        assert _dep(rows, 'Esperanto')['dep'] == 'nsubj'
        assert _dep(rows, 'Esperanto')['head'] == _dep(rows, 'lingvo')['id']


class TestPPAttachmentIsMARKED_not_guessed:
    """WHERE a prepositional phrase attaches — to the verb, or to a preceding
    noun — is ambiguous BY GRAMMAR. `Mi vidis la viron kun teleskopo` licenses
    both, and Bick measured it as 1/4-1/3 of all Esperanto attachment errors.

    We take the verb (the majority baseline) and MARK the decision, so #826 can
    turn it into an OR-node rather than a silent guess."""

    def test_the_ambiguous_choice_is_flagged(self):
        w = next(x for x in parse('Mi vidis la viron kun teleskopo.')['vortoj']
                 if x.get('plena_vorto') == 'teleskopo')
        assert w.get('alligo_ambigua') is True
        assert w['rolo'] == 'obl'
