"""The parser's proper-noun rules, and the attribution that makes them measurable.

Two things are pinned here.

**Ending validity (Rules 2-7).** A known ROOT is not a known WORD. `sam` is a
Fundamento root (`sama` = "same"), but `Sam` carries no grammatical ending and so
is not an Esperanto word at all. The old negative-detection check asked only
"is the ROOT known?", answered yes, and therefore refused to call `Sam` a name —
leaving it `nekonata` forever. See #821.

**Attribution.** Every `propra_nomo` decision records WHICH rule made it. This is
VISION.md's "attribution is built in": it makes per-rule precision measurable on
gold, and it is where a learned tie-breaker for the residue would later have to
declare itself. Without it, the parser's six proper-noun rules are a single
opaque verdict and the deterministic/learned boundary is invisible.
"""

import pytest

from klareco.parser import _has_grammatical_ending, parse


def _nodes(ast):
    out = []

    def rec(n):
        if not isinstance(n, dict):
            return
        if n.get('tipo') == 'vorto':
            out.append(n)
        for k in ('kerno', 'subjekto', 'verbo', 'objekto'):
            rec(n.get(k))
        for k in ('aliaj', 'priskriboj'):
            for x in (n.get(k) or []):
                rec(x)

    rec(ast)
    return out


def _find(sentence, surface):
    for n in _nodes(parse(sentence)):
        if (n.get('plena_vorto') or '').lower() == surface.lower():
            return n
    raise AssertionError(f'{surface!r} not found in parse of {sentence!r}')


class TestEndingValidity:
    """Rules 2-7: a content word must carry a grammatical ending."""

    @pytest.mark.parametrize('word', ['hundo', 'sama', 'peti', 'belaj', 'rapide'])
    def test_words_carry_an_ending(self, word):
        assert _has_grammatical_ending(word)

    @pytest.mark.parametrize('word', ['sam', 'Zamenhof', 'Peter', 'hund'])
    def test_bare_roots_and_names_do_not(self, word):
        assert not _has_grammatical_ending(word)

    def test_a_known_ROOT_with_no_ending_is_a_NAME_not_an_unknown(self):
        """The bug this fixes: `sam` IS in the Fundamento, so the old check said
        'known root -> common word' and left `Sam` as `nekonata`."""
        assert _find('Sam malfermis la fenestron.', 'Sam')['vortspeco'] == 'propra_nomo'

    def test_function_words_are_exempt(self):
        """The closed ending-less class (la, kaj, mi, tiu) is the grammar's own
        named exception — it carries no ending, so without the exemption the
        ending rule would declare every sentence-initial `La` a name.

        (`la` itself is absorbed into the vortgrupo rather than emitted as a word
        node, so we assert on the outcome: nothing here is a name.)
        """
        for s in ['La hundo kuras.', 'Kaj poste li iris.', 'Mi vidis lin.',
                  'Tiu domo estas granda.', 'Ĉiu homo estas egala.']:
            names = [n.get('plena_vorto') for n in _nodes(parse(s))
                     if n.get('vortspeco') == 'propra_nomo']
            assert not names, f'function word(s) called a name in {s!r}: {names}'


class TestAttribution:
    """Each proper-noun verdict must say which rule produced it."""

    @pytest.mark.parametrize('sentence,surface,evidence', [
        ('Sam malfermis la fenestron.', 'Sam', 'no_valid_ending'),
        ('Zamenhof fondis Esperanton.', 'Zamenhof', 'morphology_no_decomposition'),
        ('Mi vidis Shakespeare hieraŭ.', 'Shakespeare', 'mid_sentence_capitalization'),
    ])
    def test_the_deciding_rule_is_recorded(self, sentence, surface, evidence):
        assert _find(sentence, surface)['propra_nomo_evidence'] == evidence

    def test_every_propra_nomo_is_attributed(self):
        """No unattributed verdicts — an untagged one is a rule we cannot measure,
        and therefore a rule we cannot defend under the merge gate."""
        for s in ['Zamenhof fondis Esperanton en Varsovio.',
                  'Maria kaj Petro vizitis Parizon.',
                  'La Centra Oficejo estas granda.']:
            for n in _nodes(parse(s)):
                if n.get('vortspeco') == 'propra_nomo':
                    assert n.get('propra_nomo_evidence'), \
                        f"unattributed propra_nomo: {n.get('plena_vorto')!r} in {s!r}"


class TestAgreementPassStillWorks:
    """Rule 3 — an adjective must agree with a head noun. This pass was NOT
    broken; with an empty lexicon it was simply never REACHED, because `Centra`
    was tagged propra_nomo (undecomposed) before agreement could run. Pinned so a
    future lexicon change cannot silently break it again."""

    @pytest.mark.parametrize('sentence,surface', [
        ('Bela hundo kuras.', 'Bela'),
        ('Polaj studentoj venis.', 'Polaj'),
        ('Centra Oficejo estas granda.', 'Centra'),
        ('Usona Senato ratifis traktaton.', 'Usona'),
    ])
    def test_an_adjective_with_a_head_noun_is_NOT_a_name(self, sentence, surface):
        assert _find(sentence, surface)['vortspeco'] == 'adjektivo'

    def test_an_adjective_with_NOTHING_to_agree_with_IS_a_name(self):
        """`Maria gajnis bronzon` — mar-i-a is an adjective form, the next token
        is a verb, and there is no noun to agree with."""
        assert _find('Maria gajnis bronzon.', 'Maria')['vortspeco'] == 'propra_nomo'

    def test_the_head_noun_keeps_the_subject_slot(self):
        """#821: the mis-tagged adjective used to CAPTURE `subjekto`, demoting the
        real head noun to `aliaj`. That is what corrupted 700,925 sentences."""
        subj = parse('La Centra Oficejo estas granda.')['subjekto']
        kerno = subj.get('kerno', subj)
        assert kerno['radiko'] == 'ofic'
        assert kerno['vortspeco'] == 'substantivo'


class TestLexicalization:
    """`protected_roots` — a USAGE fact the grammar cannot recover.

    The parser splits `Esperanton` -> esper+ant, and ETYMOLOGICALLY IT IS RIGHT:
    Zamenhof was *Doktoro Esperanto*, "Doctor One-Who-Hopes". The word genuinely
    IS esper-ant-o. What it no longer is, is COMPOSITIONAL — it has frozen into a
    lexeme. Derived from derivational productivity over RAW SURFACE TEXT
    (scripts/index/build_surface_lexical_facts.py), never from parser output.
    """

    def test_esperanton_is_not_split(self):
        """The flagship bug from the June migration, named in CLAUDE.md."""
        assert _find('Zamenhof fondis Esperanton.', 'Esperanton')['radiko'] == 'esperant'

    @pytest.mark.parametrize('sentence,surface,root', [
        ('Milito estas malbona.', 'Milito', 'milit'),       # NOT mil+it
        ('La reguloj estas klaraj.', 'reguloj', 'regul'),   # NOT reg+ul
    ])
    def test_accidental_homographs_are_protected(self, sentence, surface, root):
        assert _find(sentence, surface)['radiko'] == root

    def test_TRANSPARENT_suffixes_still_decompose(self):
        """`kristano` really IS "a Christian" (krist+an) — -an/-ist/-ism compose
        reliably and must NOT be protected, or we destroy the useful root."""
        assert _find('La kristano preĝis.', 'kristano')['radiko'] == 'krist'


class TestCapitalizationRatio:
    """The residue rule (#819). `Petro` = petr-o = "rock": morphology says
    ordinary word, syntax says ordinary word, and BOTH ARE RIGHT — as a *word*
    that is what it is. Only USAGE says name, and usage is countable.

    ⚠️ Memoization of usage, NOT world knowledge: silent on unseen types, where
    the morphological rules still carry the decision.
    """

    def test_usage_overrides_a_valid_decomposition(self):
        n = _find('Petro kaj Maria venis.', 'Petro')
        assert n['vortspeco'] == 'propra_nomo'
        assert n['propra_nomo_evidence'] == 'capitalization_ratio'

    @pytest.mark.parametrize('sentence,surface', [
        ('La hundo vidis la urbon.', 'hundo'),
        ('La hundo vidis la urbon.', 'urbon'),
        ('Mi legis la libron.', 'libron'),
    ])
    def test_common_nouns_are_NOT_promoted(self, sentence, surface):
        assert _find(sentence, surface)['vortspeco'] == 'substantivo'


class TestUsageVeto:
    """Usage says NO as well as YES — and that is where the precision was hiding.

    The EVIDENTIAL rules (mid_sentence_capitalization, preceded_by_la,
    morphology_no_decomposition) fire on capitalisation and absence-of-evidence.
    A strong corpus opinion that a type is a COMMON word overrules them.
    Measured: Prago precision 44.3% -> 49.1%, recall UNCHANGED at 100%.

    It must NOT veto the DEDUCTIVE rules — a frequency count does not overrule
    grammar.
    """

    def test_a_common_word_capitalised_mid_sentence_is_not_promoted(self):
        n = _find('Mi vidis la Urbon hieraŭ.', 'Urbon')
        assert n['vortspeco'] != 'propra_nomo', 'usage says `urbo` is a common word'

    def test_usage_does_NOT_overrule_grammar(self):
        """`Sam` is decided by ending-validity (Rules 2-7). No frequency count
        may override a grammatical impossibility."""
        n = _find('Sam malfermis la fenestron.', 'Sam')
        assert n['vortspeco'] == 'propra_nomo'
        assert n['propra_nomo_evidence'] == 'no_valid_ending'

    def test_real_names_survive_the_veto(self):
        for s, w in [('Mi vidis Zamenhof hieraŭ.', 'Zamenhof'),
                     ('Petro kaj Maria venis.', 'Petro')]:
            assert _find(s, w)['vortspeco'] == 'propra_nomo'
