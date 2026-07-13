"""The corpus quality gate (#823).

The bug this exists to prevent: the redirect filter was written once, in
`rebuild_whoosh_from_duckdb.py`, whose own comment says *"this filter must land
FIRST or the pollution gets baked into the dictionary"* — and it never landed
anywhere else. The Whoosh index is clean; the STORE still carries 123,654
redirect stubs, and `REDIRECT` was the single most common proper-noun SUBJECT in
the whole corpus.

**A filter that lives in one consumer is a filter the next consumer forgets.**

The hardest test here is `test_esperanto_quoting_foreign_names_is_KEPT`. My first
gate scored token purity and would have deleted ~569,000 sentences like

    "La franclingva libro aperis en novembro de 1997 sub titolo Le Livre noir"

— perfectly good Esperanto that happens to QUOTE a foreign title, and therefore
exactly the sentences richest in proper nouns. Deleting them would have biased
the corpus away from the very thing we are trying to learn.
"""

import pytest

from klareco.corpus_quality import (
    assess,
    esperanto_score,
    has_esperanto_grammar,
    strip_markup,
)


class TestLanguageGate:
    @pytest.mark.parametrize('text', [
        'La hundo vidis la katon en la ĝardeno.',
        'Zamenhof fondis Esperanton en la jaro 1887.',
        'Ĉiu etna lingvo estas valora heredaĵo de la homaro.',
    ])
    def test_plain_esperanto_is_kept(self, text):
        assert assess(text).keep

    @pytest.mark.parametrize('text', [
        'The dog saw the cat in the garden.',
        'Der Hund sah die Katze im Garten.',
        'Les chiens sont grands et forts.',
        'Reinhold Verlag, Altenburg 2005, ISBN 3-937940-09-X, p. 286',
        'Sullivan DH, Sun S, Walls RC (1999) Protein-energy malnutrition',
        '1516 Brewing Company',
        '1753: Oliver Cromwell',
        'Just William (1940)',
    ])
    def test_foreign_text_and_fragments_are_dropped(self, text):
        v = assess(text)
        assert not v.keep and v.reason == 'not_esperanto'

    @pytest.mark.parametrize('text', [
        'La franclingva libro aperis en novembro de 1997 sub titolo Le Livre noir',
        'Krom la kompanio mem, ankaŭ ĝiaj aŭtoj ofte nomatas «Land Rover».',
        'Mi legis pri Shakespeare kaj New York.',
    ])
    def test_esperanto_quoting_foreign_names_is_KEPT(self, text):
        """THE load-bearing test. A token-purity gate scores these ~0.59 and
        deletes them — ~569,000 sentences, and the proper-noun-richest ones at
        that. Esperanto GRAMMAR is not diluted by quotation, so that is what we
        test on."""
        assert assess(text).keep

    @pytest.mark.parametrize('text', [
        'katolika Preĝejo Nomo de Sankta Maria (Taliándörögd)',
        'Erwin Weiss (kemiisto) (* 1926), germana kemiisto',
    ])
    def test_VERBLESS_esperanto_is_kept(self, text):
        """Titles, captions and biographical stubs have no verb and sometimes no
        function word — and they are ENTITY-RICH, so they are the last thing we
        should throw away. Kept via 'mostly-Esperanto AND nothing foreign'."""
        assert assess(text).keep

    def test_a_finite_verb_is_decisive(self):
        """No other language puts -as/-is/-os/-us on a content stem."""
        assert has_esperanto_grammar('Li aperis.')
        assert not has_esperanto_grammar('The Grand Budapest Hotel')


class TestRedirects:
    @pytest.mark.parametrize('text', [
        'REDIRECT Isaac Asimov',
        '#REDIRECT Libin (Belgio)',
        'ALIDIREKTI Katedralo de Narbono',
        'ALIDIREKTU Lucerno (distrikto)',
    ])
    def test_both_spellings_and_the_imperative(self, text):
        """`ALIDIREKTI` is the Esperanto redirect and `ALIDIREKTU` (imperative)
        also occurs. The store holds 123,654 of these."""
        v = assess(text)
        assert not v.keep and v.reason == 'redirect_stub'


class TestMarkupIsStrippedNotDropped:
    """`{{DISPLAYTITLE: …}} (19215) 1993 FS29 estas asteroido` carries a REAL
    Esperanto sentence. Dropping the row to be rid of the template would throw
    away the article with it. ~27,000 sentences are rescued this way."""

    def test_a_template_does_not_kill_the_sentence(self):
        v = assess('{{DISPLAYTITLE: (19215) 1993 FS29}} (19215) 1993 FS29 estas asteroido.')
        assert v.keep
        assert '{{' not in v.text and 'estas asteroido' in v.text

    def test_wikilinks_resolve_to_their_display_text(self):
        v = assess('Mi legis pri [[Shakespeare]] kaj [[New York|Novjorko]].')
        assert v.keep
        assert v.text == 'Mi legis pri Shakespeare kaj Novjorko.'

    def test_pure_table_rows_are_still_dropped(self):
        assert not assess('| kat = ne').keep
        assert not assess('}}||280}}px; background: transparent;').keep

    def test_strip_markup_keeps_prose(self):
        assert strip_markup('<ref>x</ref>La [[hundo]] kuras.') == 'La hundo kuras.'


class TestScore:
    def test_esperanto_scores_high_and_foreign_low(self):
        assert esperanto_score('La hundo vidis la katon.') >= 0.9
        assert esperanto_score('Les chiens sont grands.') <= 0.2
