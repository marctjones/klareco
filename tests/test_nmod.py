"""`nmod` — the biggest error bucket, and mostly NOT the residue we assumed.

We had written `nmod` off as "PP-attachment ambiguity, hand it to a model", on the
strength of Church & Patil and Hindle & Rooth. Then we measured the deterministic
ceiling:

    a PERFECT majority-vote rule over the preposition scores 85%.
    We scored 35%.

So 50 points of it were ordinary classical work we simply were not doing, and only
the last ~15% (the `al` coin-flips: 21 nmod vs 18 obl in gold) is genuinely
irreducible. These tests pin the deterministic part.
"""

import pytest

from klareco.conllu import to_conllu


def deps(text):
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


class TestPossessive:
    """`mia`/`lia`/`ĝia` are ADJECTIVES morphologically and `nmod` relationally.

    66 tokens in gold; we scored 0.0% on every one. 41 of them were already
    attached to the CORRECT head and merely called `amod`.
    """

    @pytest.mark.parametrize('text,poss,noun', [
        ('Mia hundo dormas.', 'mia', 'hundo'),
        ('La celoj de ĝiaj parolantoj estas gravaj.', 'ĝiaj', 'parolantoj'),
        ('Ilia domo estas granda.', 'ilia', 'domo'),
        ('Li vidis lian patron.', 'lian', 'patron'),
    ])
    def test_possessive_is_nmod_not_amod(self, text, poss, noun):
        assert deps(text)[poss] == (noun, 'nmod')

    def test_a_real_adjective_is_still_amod(self):
        # The rule keys on the ROOT being a personal pronoun. A plain adjective
        # must be untouched, or we have traded one error for another.
        assert deps('La granda hundo dormas.')['granda'] == ('hundo', 'amod')


class TestNumeral:
    """A numeral PRECEDES its noun — so a backward search never finds it."""

    def test_numeral_attaches_forward_as_nummod(self):
        # `_nearest_noun_head` searches BACKWARD, because a PP follows what it
        # modifies. For `la TRI hundoj` it found nothing and fell through to
        # "attach to the verb", giving `tri --nmod--> kuras`. 0/15 in gold.
        assert deps('La tri hundoj kuras.')['tri'] == ('hundoj', 'nummod')


class TestNmodVsObl:
    """These labels are DEFINITIONS, not choices: nmod hangs off a noun, obl off
    a verb. So the label is a FUNCTION of the head and can just be recomputed."""

    def test_pp_on_a_verb_is_obl(self):
        assert deps('Li laboris en la urbo.')['urbo'][1] == 'obl'

    def test_pp_on_a_noun_is_nmod(self):
        assert deps('La libro de Petro estas nova.')['petro'] == ('libro', 'nmod')

    def test_pp_under_a_copular_predicate_is_nmod_not_obl(self):
        # THE BUG. In a copular clause the clause head is the PREDICATE — a NOUN.
        # `clause_of` maps the clause's tokens to it, so `_attach_pp` attached
        # `por ĉiuj` to that noun (correctly) and stamped it `obl`, because the
        # variable it came from is named `verb`. Right head, impossible label.
        d = deps('Esperanto estas lingvo por ĉiuj.')
        assert d['ĉiuj'][1] == 'nmod', 'a nominal hanging off a NOUN cannot be obl'


class TestGenitiveDe:
    """`de` is 103:2 genitive in gold — the strongest attachment signal there is."""

    def test_de_attaches_to_the_noun_not_the_verb(self):
        assert deps('Mi legis la libron de Petro.')['petro'][0] == 'libron'
