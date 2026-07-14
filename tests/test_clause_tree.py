"""The AST is a TREE — one predicate-argument frame per CLAUSE, not per sentence.

It was not a tree. It was a fixed-arity RECORD: one `subjekto`, one `verbo`, one
`objekto` per SENTENCE, and nothing recursed. That is a hard ceiling, measured on
gold:

    1.64 gold subjects per sentence
    35.8% of sentences have 2+ subjects (more than one clause)
    -> a single-slot AST can never recover more than 42.5% of subjects

We measured 33.7% — already 79% of what the SHAPE allowed. No rule improvement
could break 42.5%. This does.

    Prago subject recall  32.3% -> 60.2%   (past the 46.2% single-slot ceiling)
    Prago object F1       23.8% -> 61.8%
    Cairo subject F1      87.2% -> 93.0%
"""

import pytest

from klareco.parser import parse, segment_clauses


def _clauses(sentence):
    return parse(sentence).get('propozicioj', [])


def _kern(node):
    if not node:
        return None
    k = node.get('kerno', node)
    return k.get('plena_vorto')


class TestClauseSegmentation:
    def test_a_compound_sentence_yields_one_frame_per_finite_verb(self):
        c = _clauses('Zamenhof fondis Esperanton kaj li skribis librojn, '
                     'ĉar li amis la lingvon.')
        assert len(c) == 3
        assert [_kern(x.get('verbo')) for x in c] == ['fondis', 'skribis', 'amis']
        assert [_kern(x.get('subjekto')) for x in c] == ['Zamenhof', 'li', 'li']
        assert [_kern(x.get('objekto')) for x in c] == ['Esperanton', 'librojn', 'lingvon']

    def test_a_COORDINATED_SUBJECT_is_still_ONE_clause(self):
        """`Zamenhof kaj Ludoviko venis` has one verb, so it is one clause. A
        naive split on `kaj` would cut it in two — hence the has-verb guard."""
        c = _clauses('Zamenhof kaj Ludoviko venis.')
        assert len(c) == 1

    def test_a_comma_boundary_with_no_conjunction(self):
        """Esperanto writes this with a comma, and the comma is not in our token
        stream — so the SECOND FINITE VERB is the boundary."""
        c = _clauses('Kvankam Esperanto ne estas perfekta, ĝi funkcias bone.')
        assert len(c) == 2
        assert [_kern(x.get('subjekto')) for x in c] == ['Esperanto', 'ĝi']

    def test_clauses_are_typed_by_their_role(self):
        c = _clauses('Zamenhof fondis Esperanton kaj li skribis librojn, '
                     'ĉar li amis la lingvon.')
        assert [x['rolo'] for x in c] == ['ĉefa', 'kunordigita', 'subordigita']

    def test_a_verbless_fragment_yields_NO_frame(self):
        """`Manifesto de Prago` is a HEADING. It has no finite verb, so it has no
        subject, and we must not invent one."""
        assert _clauses('Manifesto de Prago de la movado.') == []


class TestBackwardCompatibility:
    """The legacy top-level slots still describe the MAIN clause, so every
    existing consumer — DuckDBRetriever, the rerankers, the shredded columns —
    keeps working untouched while new consumers walk the tree."""

    def test_top_level_slots_still_hold_the_main_clause(self):
        a = parse('La hundo vidis la katon.')
        assert _kern(a['subjekto']) == 'hundo'
        assert _kern(a['objekto']) == 'katon'
        assert len(a['propozicioj']) == 1

    def test_every_clause_carries_attribution(self):
        """VISION.md: attribution is built in. A learned ranker would have to
        declare itself here."""
        for c in _clauses('Li venis kaj ŝi foriris.'):
            assert c['fonto'] == 'regulo'
            assert c['tipo'] == 'propozicio'


# ── AST COMPACTION (#835) ────────────────────────────────────────────────────
# The same token dict is reachable from `vortoj`, from every clause frame in
# `propozicioj`, and from the legacy top-level slots. In memory those are SHARED
# REFERENCES and cost nothing; `json.dumps` writes each one out in full.
#
# That triplication — plus the ReVo definition text formerly embedded in every
# token — took the corpus from 20 GB to 101 GB and FILLED THE DISK mid-rebuild.
# `compact_ast` stores `vortoj` once and everything else as ID references.

class TestASTCompaction:
    SENT = ('La anoj de ĉiuj lingvoj parolas Esperanton en la mondo, '
            'kaj ili volas lerni ĝin bone.')

    def _kern(self, x):
        if not isinstance(x, dict):
            return None
        k = x.get('kerno', x)
        return k.get('radiko') if isinstance(k, dict) else None

    def test_compaction_round_trips_EXACTLY(self):
        import json

        from klareco.parser import compact_ast, expand_ast, parse
        a = parse(self.SENT)
        # through JSON, as it actually goes to disk
        e = expand_ast(json.loads(json.dumps(compact_ast(a), ensure_ascii=False)))

        for slot in ('subjekto', 'verbo', 'objekto'):
            assert self._kern(e.get(slot)) == self._kern(a.get(slot)), slot
        assert len(e['aliaj']) == len(a['aliaj'])
        assert len(e['propozicioj']) == len(a['propozicioj'])
        for x, y in zip(a['propozicioj'], e['propozicioj']):
            assert self._kern(y.get('verbo')) == self._kern(x.get('verbo'))
            assert self._kern(y.get('subjekto')) == self._kern(x.get('subjekto'))

    def test_the_blob_is_actually_SMALLER(self):
        import json

        from klareco.parser import compact_ast, parse
        a = parse(self.SENT)
        full = len(json.dumps(a, ensure_ascii=False))
        small = len(json.dumps(compact_ast(a), ensure_ascii=False))
        assert small < full * 0.45, (
            f'{small}B vs {full}B — compaction is not working, and a 5.4M-sentence '
            f'corpus will fill the disk again')
