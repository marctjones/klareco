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


class TestPPGovernedIsNotTheObject:
    """A REAL BUG, and a serious one. `is_pp_governed` checked only the
    IMMEDIATELY preceding token, so `en la domon` — preposition, ARTICLE,
    accusative noun — looked ungoverned and `domon` became the DIRECT OBJECT.

        "La hundo kuris en la domon"  ->  parsed as "the dog ran THE HOUSE"

    That corrupted `obj_radiko`, which DuckDBRetriever and every reranker read.
    In Esperanto, most prepositional phrases contain an article, so most of them
    were affected."""

    def test_an_accusative_inside_a_PP_is_not_the_direct_object(self):
        rows = _rows('La hundo kuris en la domon.')
        assert _dep(rows, 'domon')['dep'] == 'obl', \
            '`en la domon` is a destination, not the thing the dog ran'

    def test_a_REAL_direct_object_still_works(self):
        rows = _rows('La hundo vidis la katon.')
        assert _dep(rows, 'katon')['dep'] == 'obj'


class TestPPAttachment:
    """#826 — Bick's #1 error class, and the one place the grammar runs out.

    `Mi vidis la viron kun teleskopo` — "with the telescope" can attach to SEEING
    or to THE MAN. No case, no agreement, no rule in the 16 disambiguates it. The
    grammar licenses BOTH and is CORRECT to.

    Measured on gold:
        `de`      -> nmod 105 : obl   2      the genitive. overwhelming.
        (no prep) -> nmod 115 : obl   5      bare nominal after a noun
        en/al/per -> ~2:1                    GENUINELY AMBIGUOUS

    We take the deterministic half and MARK the rest. Result: 92.5% of PP
    attachments are decided by rule; 7.5% become OR-nodes. That 7.5% IS the PP
    residue — counted, not argued.
    """

    def test_de_is_the_genitive_and_attaches_to_the_NOUN(self):
        """105:2 in gold — the strongest attachment signal in the language."""
        rows = _rows('Mi legis la libron de la instruisto.')
        assert _dep(rows, 'instruisto')['dep'] == 'nmod'
        assert _dep(rows, 'instruisto')['head'] == _dep(rows, 'libron')['id']

    def test_the_ACCUSATIVE_OF_DIRECTION_attaches_to_the_VERB(self):
        """`en la domoN` = INTO the house — motion, therefore the verb. A hard
        morphological signal English simply does not have, and it is free."""
        rows = _rows('La hundo kuris en la domon.')
        assert _dep(rows, 'domon')['dep'] == 'obl'
        assert _dep(rows, 'domon')['head'] == _dep(rows, 'kuris')['id']

    def test_no_candidate_noun_means_the_verb(self):
        rows = _rows('Li venis kun sia amiko.')
        assert _dep(rows, 'amiko')['head'] == _dep(rows, 'venis')['id']

    def test_a_GENUINELY_AMBIGUOUS_pp_becomes_an_OR_NODE(self):
        """We do not guess silently. Both readings are recorded, with fonto=None
        — nothing deterministic could choose, and we say so."""
        w = next(x for x in parse('Mi vidis la viron kun teleskopo.')['vortoj']
                 if x.get('plena_vorto') == 'teleskopo')
        # `kun` is in the NMOD set, so this one IS decided — the AMBIGUOUS set is
        # en/al/per/kiel. Use one of those:
        w = next(x for x in parse('Mi vidis la viron en la parko.')['vortoj']
                 if x.get('plena_vorto') == 'parko')
        assert w.get('alligo_ambigua') is True
        opts = w['alligo_opcioj']
        assert {o['rolo'] for o in opts} == {'nmod', 'obl'}
        assert all(o['fonto'] is None for o in opts), \
            'nothing deterministic chose — the AST must say so'


class TestAdverbScope:
    def test_an_adverb_modifying_an_ADJECTIVE_attaches_to_IT(self):
        """`tre granda` — not to the clause verb. We were sending every adverb to
        the verb, which is why advmod sat at 18%."""
        rows = _rows('La domo estas tre granda.')
        assert _dep(rows, 'tre')['dep'] == 'advmod'
        assert _dep(rows, 'tre')['head'] == _dep(rows, 'granda')['id']
