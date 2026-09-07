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
        # 13 words + the comma + the period. PUNCTUATION IS IN THE AST NOW (#836) —
        # which is precisely what this test's name asks for and what it did not
        # previously check. UD gold has 454 PUNCT tokens and every one has a head;
        # we used to delete them all at tokenization.
        assert len(rows) == 15, 'punctuation must be in the AST too'
        assert ',' in forms and '.' in forms

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


class TestAmbiguousPPWithNoVerb:
    """A verbless fragment ("Ludgerus en Rheine-Elte", a Wikipedia-style
    title) still goes through the genuinely-ambiguous-PP residue path
    (TestPPAttachment, case 5), which used to unconditionally record an
    `obl`-to-the-verb alternative. With no verb, that alternative's head_id
    was 0 (this codebase's "no such token" sentinel) paired with relation
    'obl' — an edge that violates syntax_graph's own invariant that head_id
    0 always means relation 'root'. compact_ast() validates every
    attachment alternative and raised on it: 212/5,000 sampled store
    sentences (4.24%) failed to serialize at all.
    """

    def test_a_verbless_fragment_does_not_crash_compact_ast(self):
        from klareco.parser import compact_ast
        ast = parse('Ludgerus en Rheine-Elte')
        compact_ast(ast)   # must not raise

    def test_the_bogus_root_obl_alternative_is_not_recorded(self):
        ast = parse('Ludgerus en Rheine-Elte')
        w = next(x for x in ast['vortoj'] if x.get('radiko') == 'Elte')
        assert w.get('alligo_opcioj') == [
            {'kapo': w['kapo'], 'rolo': 'nmod', 'fonto': None}
        ], 'with no verb, the ambiguity is moot -- only the noun option remains'
        assert not w.get('alligo_ambigua'), \
            'a single-option "choice" must not be flagged ambiguous'

    def test_a_genuinely_ambiguous_pp_with_a_real_verb_is_unaffected(self):
        """The fix must not remove the alternative when a verb DOES exist."""
        w = next(x for x in parse('Mi vidis la viron en la parko.')['vortoj']
                 if x.get('plena_vorto') == 'parko')
        assert w.get('alligo_ambigua') is True
        assert {o['rolo'] for o in w['alligo_opcioj']} == {'nmod', 'obl'}


class TestAttachmentTraceToleratesAPreRootSweepSnapshot:
    """A third, distinct compact_ast() crash class (4/50,000 sampled store
    sentences): `attach_all` can leave a token with head_id 0 (unattached)
    and whatever relation label it happened to carry at that point.
    `syntax_rules.refine_dependencies`'s root-relation-v1 sweep is what
    visits every token afterward and normalizes this to relation 'root' —
    but it is not necessarily the FIRST refinement rule to touch such a
    token. `punctuated-nominal-enumeration-v1` runs earlier and can call
    `attach()` on one first, capturing {head_id: 0, relation: <not 'root'>}
    as ITS "before" snapshot in the attachment trace. The validator already
    special-cased exactly this shape for root-relation-v1's OWN before-edge
    (root-relation-v1 uses attach(word, 0, 'root', ...), so its own before
    edge is naturally this shape) but rejected it from any other rule.
    """

    def test_a_biographical_listing_sentence_does_not_crash_compact_ast(self):
        """'Name (dates) profession, profession, profession kaj kaj mayor of
        City' -- the duplicated 'kaj' is present in the reproducing corpus
        sentence and is incidental; the enumeration/coordination structure
        is what exercises the pre-root-sweep state."""
        from klareco.parser import compact_ast
        s = ('Johann Sebastian Bruch (1759-1828) komercisto, politikisto, '
             'juĝisto kaj kaj urbestro de Saarbrücken')
        compact_ast(parse(s))   # must not raise

    def test_the_final_selected_tree_has_no_zero_head_non_root_relation(self):
        """Whatever the trace's intermediate 'before' snapshots look like,
        the FINAL selected kapo/rolo on every token must still satisfy the
        root invariant -- this fix only widens what the TRACE HISTORY may
        show, never the answer the parser actually commits to."""
        s = ('Johann Sebastian Bruch (1759-1828) komercisto, politikisto, '
             'juĝisto kaj kaj urbestro de Saarbrücken')
        for w in parse(s)['vortoj']:
            if not isinstance(w, dict):
                continue
            assert (w.get('kapo') == 0) == (w.get('rolo') == 'root'), \
                f"token {w.get('id')} ({w.get('radiko')!r}) has kapo=0 with rolo={w.get('rolo')!r}"


class TestAdverbScope:
    def test_an_adverb_modifying_an_ADJECTIVE_attaches_to_IT(self):
        """`tre granda` — not to the clause verb. We were sending every adverb to
        the verb, which is why advmod sat at 18%."""
        rows = _rows('La domo estas tre granda.')
        assert _dep(rows, 'tre')['dep'] == 'advmod'
        assert _dep(rows, 'tre')['head'] == _dep(rows, 'granda')['id']

    def test_an_adverb_between_finite_and_infinitive_attaches_to_infinitive(self):
        """`Ni volas rapide labori` — the adverb belongs to the infinitive."""
        rows = _rows('Ni volas rapide labori.')
        assert _dep(rows, 'rapide')['dep'] == 'advmod'
        assert _dep(rows, 'rapide')['head'] == _dep(rows, 'labori')['id']


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


class TestApposition:
    """Rename relationships are appositions, not nominal modifiers."""

    def test_comma_introduced_apposition(self):
        rows = _rows('Ni, anoj de la movado, laboras.')
        assert _dep(rows, 'anoj')['dep'] == 'appos'
        assert _dep(rows, 'anoj')['head'] == _dep(rows, 'Ni')['id']

    def test_proper_noun_after_head_is_apposition(self):
        rows = _rows('La lingvo Esperanto estas internacia.')
        assert _dep(rows, 'Esperanto')['dep'] == 'appos'
        assert _dep(rows, 'Esperanto')['head'] == _dep(rows, 'lingvo')['id']

    def test_article_phrase_after_comma_is_apposition(self):
        rows = _rows('Ŝi kreskis en Parizo, la ĉefurbo de Francio.')
        assert _dep(rows, 'ĉefurbo')['dep'] == 'appos'
        assert _dep(rows, 'ĉefurbo')['head'] == _dep(rows, 'Parizo')['id']


class TestNoDependencyCycleOnUnseenText:
    """#927 / #929 — the parser must never raise on text it has not seen
    before, no matter how a governor-search rule chose a candidate.

    #927: an apposition rule assigned a head directly, without the
    ancestor-walk guard `syntax_rules.attach()` already used elsewhere,
    letting `w -> prev_nominal` close a loop back through `w` itself.

    #929: a RELATIVE-clause antecedent search (`_governor`, for clauses
    whose opener is a ki-korelativo) could pick a KI-PREFIX adverbial
    correlative ('kiel'/'kiam'/'kie'/'kiom'/'kial') as the antecedent. That
    candidate's own head is often not yet resolved at check time (its
    advmod attachment happens in a later pass), so the existing ancestor
    walk can't see that it will later chain back into the very clause being
    attached. Fixed two ways: (1) a ki-prefix adverbial correlative is
    never accepted as a relative-clause antecedent — but a TI-prefix one
    (`tiel`/`tiam`/...) still is, because "tiel X, kiel Y" is a genuine
    correlative pairing where `tiel` IS the correct antecedent (see
    TestGeneral below); (2) a safety-net repair pass at the end of
    `attach_all` runs the same cycle check `syntax_graph.validate_tokens`
    uses and, if a THIRD unknown rule ever reproduces this class, breaks
    the cycle deterministically (reattach to the main verb as `parataxis`)
    instead of raising.
    """

    def test_kiam_after_a_kiel_comparison_does_not_crash(self):
        """klareco#929's first reproduction: 'kiel parafiletika' (a
        comparison) precedes a 'kiam' clause; the old code picked 'kiel' as
        the kiam-clause's antecedent even though 'kiel' itself attaches
        inside that same clause's structure."""
        s = ('Grupo konstituiĝas kiel parafiletika kiam al klado '
             '(evolua branĉo) oni forprenas unu aŭ pliajn grupojn '
             'holofiletikajn.')
        parse(s)   # must not raise

    def test_tiam_before_a_kiel_predicate_does_not_crash(self):
        """klareco#929's second reproduction: a different cause (not the
        ki/ti-prefix rule above) — the safety-net repair pass is what
        catches this one."""
        s = ('Tio estas konata kiel malsano tiam konata kiel '
             '"malsano de la francoj" aŭ "morbus gallico".')
        parse(s)   # must not raise

    def test_tiel_kiel_pairing_still_attaches_to_tiel(self):
        """The fix must NOT bar every adverbial correlative — only the
        ki-prefix half of a tiel...kiel pair. Prago sentence 68 is gold for
        exactly this attachment (advcl:relcl -> tiel)."""
        rows = _rows('Li komentarios tiel, kiel li volas.')
        assert _dep(rows, 'volas')['head'] == _dep(rows, 'tiel')['id']

    def test_safety_net_repairs_a_cycle_instead_of_raising(self, caplog):
        """Unit-level: force attach_all's final repair pass to fire on a
        synthetic cycle, independent of which upstream rule caused it."""
        import logging
        from klareco.parser import attach_all

        # A 3-token cycle: 1 -> 2 -> 3 -> 1. No real rule should ever
        # produce this; the test manufactures it directly.
        word_asts = [
            {'id': 1, 'kapo': 2, 'rolo': 'dep', 'vortspeco': 'substantivo',
             'radiko': 'a'},
            {'id': 2, 'kapo': 3, 'rolo': 'dep', 'vortspeco': 'substantivo',
             'radiko': 'b'},
            {'id': 3, 'kapo': 1, 'rolo': 'dep', 'vortspeco': 'substantivo',
             'radiko': 'c'},
        ]
        with caplog.at_level(logging.WARNING, logger='klareco.parser'):
            attach_all(word_asts, clauses=[])
        from klareco.syntax_graph import validate_tokens
        validate_tokens(word_asts)   # must not raise: the cycle is gone
        assert any('dependency cycle' in r.message.lower()
                  for r in caplog.records), \
            'a repaired cycle must be logged loudly, not silently fixed'
