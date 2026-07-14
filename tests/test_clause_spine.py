"""The CLAUSE SPINE — which token heads a clause, and what it hangs from.

These are the four bugs that held LAS_all at 41.6% on Prago. Each one was a
ROOT CAUSE: it did not merely mislabel its own tokens, it misattached everything
underneath, which is why fixing them moved `nsubj` (+11) and `root` (+7) without
either being touched.

    copula          `estas` was the head; UD makes the PREDICATE the head
    relative clause never opened, so its verb became a SECOND SENTENCE ROOT
    infinitive      could never head a clause (no TENSE), so it fell through
    `ke`            was `advcl`; it is `ccomp` — a clausal OBJECT
"""

import pytest

from klareco.conllu import to_conllu


def deps(text):
    """-> {form: (head_form_or_ROOT, relation)}"""
    rows = []
    for line in to_conllu(text, sent_id='1').split('\n'):
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split('\t')
        if not f[0].isdigit():
            continue
        rows.append({'id': int(f[0]), 'form': f[1], 'head': int(f[6]),
                     'rel': f[7].split(':')[0]})
    byid = {r['id']: r for r in rows}
    return {r['form'].lower():
            ('ROOT' if r['head'] == 0 else byid[r['head']]['form'].lower(),
             r['rel'])
            for r in rows}


class TestExactlyOneRoot:
    """A dependency tree has EXACTLY ONE root. We were emitting forests."""

    @pytest.mark.parametrize('text', [
        'La homo kiu venis estas mia amiko.',
        'Mi scias ke li venis.',
        'Nur kelkaj el tiuj kiuj studas lingvon ekmastras ĝin.',
        'Mi volas lerni kaj instrui.',
    ])
    def test_one_root(self, text):
        n = sum(1 for h, _ in deps(text).values() if h == 'ROOT')
        assert n == 1, f'{n} roots in {text!r} — that is a forest, not a tree'


class TestCopula:
    """UD: the PREDICATE heads a copular clause; `esti` is its `cop`/`aux`."""

    def test_nominal_predicate_is_head(self):
        d = deps('Esperanto estas lingvo.')
        assert d['lingvo'][0] == 'ROOT'
        assert d['estas'] == ('lingvo', 'cop')
        assert d['esperanto'] == ('lingvo', 'nsubj')

    def test_adjectival_predicate_is_head(self):
        d = deps('La domo estas granda.')
        assert d['granda'][0] == 'ROOT'
        assert d['estas'] == ('granda', 'cop')

    def test_participle_predicate_takes_aux_not_cop(self):
        # `estas ligita` is a PERIPHRASTIC VERB FORM: the participle carries the
        # predication. Esperanto marks it morphologically (-it-), so it is free.
        d = deps('La lingvo estas ligita al kulturo.')
        assert d['ligita'][0] == 'ROOT'
        assert d['estas'] == ('ligita', 'aux')

    def test_predicate_is_never_inside_a_prepositional_phrase(self):
        # THE BUG. `kulturo` is the object of `al` — it is already governed and
        # cannot be the predicate. We used to pick it, which put the head of the
        # clause inside a PP and scored 0% on `aux`.
        d = deps('La lingvo estas ligita al kulturo.')
        assert d['estas'][0] != 'kulturo'


class TestRelativeClause:
    """A relative clause modifies its ANTECEDENT, not the main verb."""

    def test_relative_verb_attaches_to_antecedent(self):
        d = deps('La homo kiu venis estas mia amiko.')
        assert d['venis'] == ('homo', 'acl')

    def test_relative_clause_opens_without_a_preceding_verb(self):
        # The guard `has_verb and opens_a_clause` exists so that a coordinated
        # SUBJECT (`Zamenhof kaj Ludoviko venis`) is not cut in two. But `kiu`
        # cannot coordinate noun phrases — it can ONLY open a clause. Making it
        # wait for a verb meant a relative clause whose antecedent had none never
        # opened at all, and its verb became a second root.
        d = deps('Tiuj kiuj studas lingvon sukcesas.')
        assert d['studas'] == ('tiuj', 'acl')
        assert d['sukcesas'][0] == 'ROOT'


class TestCoordinatorStillGuarded:
    """…and the guard must SURVIVE for coordinators, or we break what worked."""

    def test_coordinated_subject_is_one_clause(self):
        d = deps('Zamenhof kaj Ludoviko venis.')
        assert d['venis'][0] == 'ROOT'
        assert d['ludoviko'][1] == 'conj'    # a conjunct, NOT a new clause


class TestInfinitive:
    """An infinitive has no TENSE, so it could never head a clause. It is one."""

    def test_infinitive_complement_of_verb_is_xcomp(self):
        d = deps('Mi volas lerni Esperanton.')
        assert d['lerni'] == ('volas', 'xcomp')

    def test_infinitive_modifying_a_noun_is_acl(self):
        d = deps('Li akceptis la taskon lerni la lingvon.')
        assert d['lerni'] == ('taskon', 'acl')

    def test_purpose_infinitive_is_advcl_and_por_is_its_mark(self):
        d = deps('Li venis por vidi min.')
        assert d['vidi'][1] == 'advcl'
        assert d['por'] == ('vidi', 'mark')


class TestComplementiser:
    """`ke` is the ONE complementiser — a clausal OBJECT, not an adverbial."""

    def test_ke_clause_is_ccomp(self):
        d = deps('Mi scias ke li venis.')
        assert d['venis'][1] == 'ccomp'

    def test_car_clause_is_advcl(self):
        d = deps('Mi foriris ĉar li venis.')
        assert d['venis'][1] == 'advcl'
