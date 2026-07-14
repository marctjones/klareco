"""ONE corpus quality gate — used by extract, by the store build, and by Whoosh.

Why this module exists at all
-----------------------------
The redirect filter was written once, in `rebuild_whoosh_from_duckdb.py`, whose
own comment says:

    "this filter must land FIRST or the pollution gets baked into the dictionary"

It never landed anywhere else. So the Whoosh index is clean and the **store** —
the ASTs, the shredded columns, and every statistic computed from them — still
carries **123,654 redirect stubs**. `REDIRECT` was the single most common
proper-noun SUBJECT in the entire corpus.

A filter that lives in one consumer is a filter that will be forgotten by the
next one. This module is the single source of truth, and every stage imports it.
See #823.

What it rejects, and why
------------------------
**Markup** — redirect stubs (`REDIRECT` and the Esperanto `ALIDIREKTI`), wiki
link/template syntax, table rows. Content-free, and they poison anything that
mines the corpus for entities.

**Non-Esperanto text** — 24,146 English sentences sit in the store. The parser
handles them "successfully" (#818) and dutifully tags `The`, `Der`, `Les` as
proper nouns, because they genuinely are not Esperanto words. The parser is
right; the corpus is wrong.

The language gate is Esperanto's own grammar, not a language-ID model:

  * **Rules 2-7** give every content word a grammatical ending (`-o -a -e -i -as
    -is -os -us -u`, plus `-j` / `-n`). English and German words do not carry
    them.
  * The **closed function-word class** (`la`, `de`, `kaj`, `en`, `estas`, …) is
    small, fixed, and appears in essentially every real Esperanto sentence.

A sentence that has neither is not Esperanto. This is deterministic, needs no
model, and needs no lexicon — so it cannot degrade silently the way a missing
data file can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Grammatical endings (Rules 2-7), longest first.
_ENDINGS = ('ojn', 'oj', 'on', 'ajn', 'aj', 'an', 'en', 'as', 'is', 'os', 'us',
            'o', 'a', 'e', 'i', 'u')

# The closed function-word class. Small, fixed, and present in essentially every
# real Esperanto sentence.
_FUNCTION_WORDS = frozenset("""
la kaj aŭ sed nek do ĉar se ke ol ĉu ne jes ja kvankam kvazaŭ tamen
de da al el en sur sub super apud antaŭ post inter tra trans ĉe kun sen por pri
per pro laŭ kontraŭ krom malgraŭ anstataŭ ekster preter ĝis dum po je ĉirkaŭ
mi vi li ŝi ĝi ni ili oni si sia lia ŝia ĝia nia ilia mia via
kiu kio kia kie kiam kial kiel kiom kies tiu tio tia tie tiam tial tiel tiom
iu io ia ie iam ĉiu ĉio ĉia ĉie ĉiam neniu nenio nenie neniam
estas estis estos estus esti ne nur ankaŭ ankoraŭ jam nun tre tro plej pli
""".split())

_TOKEN_RE = re.compile(r"[A-Za-zĈĉĜĝĤĥĴĵŜŝŬŭ]+")

# NEGATIVE evidence. Esperanto function words are a POSITIVE signal; these are the
# other side of the coin, and they are what let us keep verbless Esperanto stubs
# without also keeping English prose.
#
#     "Erwin Weiss (kemiisto) (* 1926), germana kemiisto"   <- Esperanto, no verb,
#                                                              no EO function word
#     "The dog saw the cat in the garden."                  <- must still go
#
# Both score ~0.6 on token purity. What separates them is that one contains
# `the`/`in` and the other contains nothing foreign at all.
#
# `la` is deliberately ABSENT (it is Esperanto's own article, and collides with
# French/Spanish). `le`, `der`, `die` etc. are safe because Esperanto has no such
# words — but a QUOTED foreign title can contain them, which is why a finite
# Esperanto verb always wins over this signal.
_FOREIGN_FUNCTION_WORDS = frozenset("""
the of and or to in on at from with by is was were are be been has have had
der die das den dem des und ist sind war ein eine einer im zum zur von auf
le les du des un une est sont et dans pour avec sur
el los las un una es son y con por para
il lo gli un una di che sono per con
""".split())

# `ALIDIREKTI` is the Esperanto redirect; `ALIDIREKTU` (imperative) also occurs.
# ⚠️ NO `\b` HERE, AND THAT IS THE POINT.
#
# This used to end in `\b` (a word boundary), which is the obvious way to write it
# and is WRONG for this data. The Wikipedia extractor GLUES the keyword to the
# title when it strips the `[[...]]`:
#
#     RedirectKantono Apencelo Ekstera
#     ALIDIREKTUPlena Manlibro de Esperanta Gramatiko
#
# `Redirect` is followed by `K` — both word characters, so there is NO boundary,
# so the regex never fired and **5,247 redirect stubs walked straight through the
# quality gate** into the store. The one place a word boundary looks obviously
# correct is the one place the data does not have one.
_REDIRECT_RE = re.compile(r'^\s*#?\s*(REDIRECT|ALIDIREKTI|ALIDIREKTU)', re.I)

# Markup we could not STRIP is not prose. `strip_markup` uses matched-bracket
# regexes, so it cannot touch UNCLOSED markup:
#
#     la [[Karikaturmuseum|Karikaturmusuem (karikaturmuzeo) en Krems,
#     {{Taksonomio |nomo = |koloro = |dosiero = bristol.zoo...
#
# 856 rows kept their brackets and reached the store. If a bracket survives the
# stripper, the row is a broken template, not a sentence.
_RESIDUAL_MARKUP_RE = re.compile(r'\[\[|\]\]|\{\{|\}\}')

# Markup is STRIPPED, not used to delete the row. `{{DISPLAYTITLE: …}} (19215)
# 1993 FS29 estas asteroido …` carries a real Esperanto sentence — dropping it
# would throw away the whole article to get rid of a template.
_TEMPLATE_RE = re.compile(r'\{\{[^{}]*\}\}')
_REF_RE = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^>]*/?>|</?\w+[^>]*>', re.S)
_WIKILINK_PIPED_RE = re.compile(r'\[\[[^\[\]|]*\|([^\[\]]*)\]\]')   # [[target|text]] -> text
_WIKILINK_RE = re.compile(r'\[\[([^\[\]]*)\]\]')                    # [[text]]        -> text
_TABLE_ROW_RE = re.compile(r'^\s*[|!].*$|^\s*\}\}')


def strip_markup(text: str) -> str:
    """Remove wiki/HTML syntax but KEEP the prose it wraps."""
    t = _REF_RE.sub(' ', text)
    for _ in range(3):                       # templates can nest
        t = _TEMPLATE_RE.sub(' ', t)
    t = _WIKILINK_PIPED_RE.sub(r'\1', t)
    t = _WIKILINK_RE.sub(r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()


@dataclass(frozen=True)
class Verdict:
    keep: bool
    reason: str          # '' when kept
    eo_score: float      # fraction of tokens that look Esperanto
    text: str = ''       # markup-STRIPPED text — this is what should be indexed


def esperanto_score(text: str) -> float:
    """Fraction of tokens that are Esperanto function words or carry a
    grammatical ending. Pure grammar — no lexicon, no model, nothing to lose."""
    toks = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]
    if not toks:
        return 0.0
    ok = 0
    for t in toks:
        if t in _FUNCTION_WORDS:
            ok += 1
            continue
        for e in _ENDINGS:
            # the stem must survive: `is` in English is not the verb ending -is
            if t.endswith(e) and len(t) - len(e) >= 2:
                ok += 1
                break
    return ok / len(toks)


_FINITE_VERB_RE = re.compile(
    r'\b[a-zĉĝĥĵŝŭ]{2,}(as|is|os|us)\b', re.I)


def has_esperanto_grammar(text: str) -> bool:
    """Does the sentence have Esperanto STRUCTURE — function words and a verb?

    This, not token-purity, is the right test. A ratio-of-Esperanto-looking-tokens
    gate DELETES exactly the sentences we most want:

        "La franclingva libro aperis en novembro de 1997 sub titolo Le Livre noir"
        "Krom la kompanio mem, ankaŭ ĝiaj aŭtoj ofte nomatas «Land Rover»"

    Both score ~0.59 on token purity, because they QUOTE foreign titles — and
    they are perfectly good Esperanto, and they are the sentences RICHEST in
    proper nouns. Dropping them would bias the corpus away from the very thing
    we are trying to learn.

    Esperanto grammar, by contrast, is not diluted by quotation. `la`, `en`, `de`,
    `sub`, `aperis` are all still there. So: does the sentence have Esperanto
    function words and a finite Esperanto verb? Foreign text has neither.
    """
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not toks:
        return False
    n_fw = sum(1 for t in toks if t in _FUNCTION_WORDS)
    has_verb = bool(_FINITE_VERB_RE.search(text))
    # A finite Esperanto verb is decisive on its own — no other language produces
    # -as/-is/-os/-us on a content stem. Otherwise require function-word density:
    # a real Esperanto sentence is dense in `la`/`de`/`en`/`kaj`.
    if has_verb:
        return True          # no other language puts -as/-is/-os/-us on a stem
    if n_fw >= 2:
        return True

    # A VERBLESS Esperanto phrase is still Esperanto — titles, captions, list
    # items and biographical stubs are legitimate, and they are ENTITY-RICH:
    #     "katolika Preĝejo Nomo de Sankta Maria (Taliándörögd)"
    #     "Erwin Weiss (kemiisto) (* 1926), germana kemiisto"
    # Requiring a function word, or a minimum length, deleted these. So instead:
    # mostly-Esperanto tokens AND nothing foreign.
    n_foreign = sum(1 for t in toks if t in _FOREIGN_FUNCTION_WORDS)
    return n_foreign == 0 and esperanto_score(text) >= 0.6


def assess(text: str, *, min_eo_score: float = 0.35) -> Verdict:
    """Should this sentence be in the corpus?

    Order matters: STRUCTURE first, purity second. Structure keeps the Esperanto
    sentences that quote foreign names; purity alone would throw them away.
    """
    if not text or not text.strip():
        return Verdict(False, 'empty', 0.0, '')
    if _REDIRECT_RE.search(text):
        return Verdict(False, 'redirect_stub', 0.0, '')

    # STRIP markup, then judge the prose underneath. Dropping the row instead
    # would discard the article with the template.
    text = strip_markup(text)
    if not text or _TABLE_ROW_RE.match(text):
        return Verdict(False, 'wiki_markup', 0.0, '')
    # …and if a bracket SURVIVED the stripper, the markup was unclosed. That is a
    # broken template, not a sentence. Stripping is for markup we can remove
    # cleanly; markup we cannot remove is a reason to drop the row.
    if _RESIDUAL_MARKUP_RE.search(text):
        return Verdict(False, 'wiki_markup', 0.0, '')

    score = esperanto_score(text)

    # No Esperanto grammar at all -> not an Esperanto sentence, at any length.
    # This also catches the foreign FRAGMENTS ("1516 Brewing Company",
    # "17 Kleiner Morgenwanderer") that a minimum-length bypass would wave through.
    if not has_esperanto_grammar(text):
        return Verdict(False, 'not_esperanto', score, text)

    # Structure present but the token stream is still overwhelmingly foreign —
    # a citation or a catalogue row with one stray Esperanto word.
    if score < min_eo_score:
        return Verdict(False, 'not_esperanto', score, text)

    return Verdict(True, '', score, text)
