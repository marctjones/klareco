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


class TestCoordination:
    """#827 — 11.5% of LAS. Bick found coordination 4x over-represented among
    Esperanto attachment errors, second only to PP attachment.

    The STRUCTURE is fully deterministic, and the gold data says so:
        conj:  NOUN<-NOUN 92 · VERB<-VERB 45 · ADJ<-ADJ 18   LIKE coordinates LIKE
        direction: head BEFORE conj   177/177 = 100%
        cc:        head AFTER cc      150/153 =  98%
    """

    def test_coordinated_NOUNS(self):
        """`kaj` attaches to the SECOND conjunct; the second attaches to the FIRST."""
        rows = _rows('Zamenhof kaj Ludoviko venis.')
        assert _dep(rows, 'Ludoviko')['dep'] == 'conj'
        assert _dep(rows, 'Ludoviko')['head'] == _dep(rows, 'Zamenhof')['id']
        assert _dep(rows, 'kaj')['dep'] == 'cc'
        assert _dep(rows, 'kaj')['head'] == _dep(rows, 'Ludoviko')['id']

    def test_coordinated_CLAUSES_coordinate_the_VERBS(self):
        """UD coordinates the HIGHEST elements. `Li venis kaj ŝi foriris` joins
        the VERBS — not the pronouns, even though `ŝi` is the first content word
        after `kaj`. We used to send every coordinator to the next finite verb,
        which got this right by accident and got nominal coordination wrong."""
        rows = _rows('Li venis kaj ŝi foriris.')
        assert _dep(rows, 'foriris')['dep'] == 'conj'
        assert _dep(rows, 'foriris')['head'] == _dep(rows, 'venis')['id']
        assert _dep(rows, 'ŝi')['dep'] == 'nsubj'
        assert _dep(rows, 'ŝi')['head'] == _dep(rows, 'foriris')['id']

    def test_coordinated_ADJECTIVES(self):
        rows = _rows('La domo estas granda kaj bela.')
        assert _dep(rows, 'bela')['dep'] == 'conj'
        assert _dep(rows, 'bela')['head'] == _dep(rows, 'granda')['id']


class TestAgreementDecidesTheHeadNoun:
    """Esperanto's adjective agreement does work English cannot.

    An adjective agrees with its head in NUMBER and CASE, so it can only attach
    to a noun it agrees with. In a coordination that is a hard disambiguation:

        maljuna  viro kaj virinoj   `maljuna` is SINGULAR -> cannot head `virinoj`
        maljunaj viroj kaj virinoj  `maljunaj` is PLURAL

    (Note: UD does not encode adjective SCOPE structurally — both attach to the
    first conjunct. What agreement buys us is the correct HEAD NOUN, which is the
    part that can actually go wrong.)
    """

    def test_a_singular_adjective_cannot_head_a_plural_noun(self):
        rows = _rows('La maljuna viro kaj virinoj venis.')
        adj = _dep(rows, 'maljuna')
        assert adj['head'] == _dep(rows, 'viro')['id'], \
            'a SINGULAR adjective must not attach to a PLURAL noun'

    def test_plural_agreement(self):
        rows = _rows('La maljunaj viroj kaj virinoj venis.')
        assert _dep(rows, 'maljunaj')['head'] == _dep(rows, 'viroj')['id']


class TestPredicativeVsAttributive:
    """`La domo estas granda` — `granda` AGREES with `domo` (both nominative
    singular), so the agreement pass filed it as an attributive adjective. But it
    is PREDICATIVE: it comes after the copula, and UD makes it the ROOT.

    POSITION is what separates them, and nothing else can:
        la GRANDA domo estas bela    precedes the noun -> attributive (amod)
        la domo estas GRANDA         follows the verb  -> predicative (root)
    """

    def test_a_predicative_adjective_is_the_ROOT(self):
        rows = _rows('La domo estas granda.')
        assert _dep(rows, 'granda')['dep'] == 'root'
        assert _dep(rows, 'estas')['dep'] == 'cop'
        assert _dep(rows, 'domo')['dep'] == 'nsubj'

    def test_an_attributive_adjective_is_still_an_amod(self):
        rows = _rows('La granda domo estas bela.')
        assert _dep(rows, 'granda')['dep'] == 'amod'
        assert _dep(rows, 'granda')['head'] == _dep(rows, 'domo')['id']
        assert _dep(rows, 'bela')['dep'] == 'root'


class TestEllipsisGapping:
    """#829 — `Maria gajnis bronzon, Petro arĝenton, kaj Jane oron.`
       (Mary won bronze, Peter [won] silver, and Jane [won] gold.)

    TWO of those clauses have NO VERB. `segment_clauses` found ONE, and `Petro`,
    `arĝenton`, `Jane`, `oron` floated as `nmod` — two whole clauses, lost.

    ESPERANTO TELLS US THE GAP IS THERE, MORPHOLOGICALLY: an ACCUSATIVE needs a
    verb to govern it. So a NOMINATIVE nominal immediately followed by an
    ACCUSATIVE one, with no verb between, is a clause whose predicate has been
    elided. English has no such signal — it has to guess.

    Schuster, Nivre & Manning (2018): reconstruction works well "when the parser
    correctly predicts the EXISTENCE of a gap" — DETECTION is the bottleneck.
    Here the accusative detects it for free.
    """

    def test_the_gapped_clauses_are_recovered(self):
        rows = _rows('Maria gajnis bronzon, Petro arĝenton, kaj Jane oron.')
        v = _dep(rows, 'gajnis')['id']
        # the promoted head of each gapped clause attaches to the VERB
        assert _dep(rows, 'Petro')['dep'] == 'conj'
        assert _dep(rows, 'Petro')['head'] == v
        assert _dep(rows, 'Jane')['dep'] == 'conj'
        assert _dep(rows, 'Jane')['head'] == v
        # and the stranded argument attaches to it as an ORPHAN
        assert _dep(rows, 'arĝenton')['dep'] == 'orphan'
        assert _dep(rows, 'arĝenton')['head'] == _dep(rows, 'Petro')['id']
        assert _dep(rows, 'oron')['dep'] == 'orphan'
        assert _dep(rows, 'oron')['head'] == _dep(rows, 'Jane')['id']

    def test_the_elision_is_FLAGGED_not_silently_reconstructed(self):
        w = next(x for x in parse('Maria gajnis bronzon, Petro arĝenton.')['vortoj']
                 if x.get('plena_vorto') == 'Petro')
        assert w.get('elipsa') is True, \
            'a reconstructed predicate must never be presented as if it were surface text'

    def test_coordination_does_not_CLOBBER_the_gapped_head(self):
        """`kaj` runs through the coordination pass, which used to overwrite the
        head that gapping had already set — re-breaking the ellipsis. `Jane`
        belongs to the VERB, not to the nearest preceding noun."""
        rows = _rows('Maria gajnis bronzon, Petro arĝenton, kaj Jane oron.')
        assert _dep(rows, 'Jane')['head'] == _dep(rows, 'gajnis')['id']

    def test_an_ordinary_subject_object_clause_is_NOT_treated_as_a_gap(self):
        rows = _rows('La hundo vidis la katon.')
        assert _dep(rows, 'katon')['dep'] == 'obj'
        assert 'orphan' not in {r['dep'] for r in rows}
