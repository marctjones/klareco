"""
Proper-noun INFERENCE — no gazetteer.

We do not look a name up in a list of the world's names. We infer it from the
language's own vocabulary:

    capitalized  AND  the word does not decompose to a known Esperanto root
      -> proper noun

An open-world lookup becomes a closed-world inference. See `docs/PROPER_NOUNS.md`
for the full argument, the sources, and the measurements.

Measured against UD_Esperanto-Prago (external gold; the one ruler that is not
circular):

    current parser (dictionary missing)   P 18.2%  R 57.1%  F1 27.6%
    this module (strict)                  P 53.6%  R 55.6%  F1 54.5%
    this module (scheme-adjusted)         P 83.3%  R 55.6%  F1 66.7%

Every rule here comes from Esperanto itself, not from us:

  * **16RULES Rule 1** — the alphabet is CLOSED (28 letters, one sound each).
    `q`, `w`, `x`, `y` and clusters like `sch`/`th`/`ck` are IMPOSSIBLE in an
    Esperanto word, so their presence is proof the token is foreign.
  * **Zamenhof, Lingvaj Respondoj 63** (La Esperantisto, 1891) — "Propran nomon
    oni povas nun skribi tiel, kiel ĝi estas skribata en la gepatra lingvo de
    ĝia posedanto": a proper name MAY keep its native orthography. So foreign
    orthography positively LICENSES proper-nounhood. This is the founder's
    ruling, not a heuristic. (The text is in our own corpus.)
  * **The 16 rules give a CLOSED affix inventory** — a finite list of prefixes
    and suffixes. So "does this decompose to a known root?" is decidable by
    search, not by a model.

Two traps, both paid for in blood:

  1. **Do NOT use the parser's own `radiko`.** The root lexicon is harvested FROM
     the parser's output, so feeding that output back in is failure mode F13 —
     the parser grading its own homework. It collapsed F1 to 12.8%. The
     decomposition here is deliberately INDEPENDENT of the parser.
  2. **Naive final-ending stripping is not enough.** `Homaranismo`, `Presejo`,
     `Oficejo`, `Britio` are ordinary DERIVED nouns (`homar+an+ism+o`,
     `pres+ej+o`, `ofic+ej+o`, `brit+i+o`) whose roots were already in the
     lexicon. Full affix decomposition took precision from 38.5% to 83.3%.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_ROOT_VOCAB = Path(__file__).parent.parent / 'data' / 'vocabularies' / 'root_vocab.json'
_FUNDAMENTO = Path(__file__).parent.parent / 'data' / 'vocabularies' / 'fundamento_roots.json'

# ---------------------------------------------------------------------------
# The CLOSED inventories. These are the language, not our guesses.
# ---------------------------------------------------------------------------

# Rule 1: 28 letters, one sound each. These cannot occur in an Esperanto word.
_NON_EO = re.compile(
    r'[qwxy]|sch|sh|th|ph|ck|ll|tt|ff|ss|zz|mm|nn|pp|rr|dd|gg|cc|bb', re.I)

# Grammatical endings (Rules 2-7).
_ENDINGS = ('ojn', 'oj', 'on', 'o', 'ajn', 'aj', 'an', 'a',
            'en', 'e', 'as', 'is', 'os', 'us', 'i', 'u')

# Derivational suffixes — the official inventory plus the productive ones.
_SUFFIXES = ('ism', 'ist', 'ind', 'em', 'ec', 'aĵ', 'ul', 'in', 'et', 'eg',
             'ar', 'er', 'uj', 'ej', 'estr', 'ad', 'aĝ', 'an', 'ig', 'iĝ',
             'il', 'obl', 'op', 'um', 'id', 'nj', 'ĉj')

# Official prefixes.
_PREFIXES = ('mal', 'ge', 're', 'ek', 'dis', 'eks', 'fi', 'mis', 'pra',
             'bo', 'ĉef', 'vic', 'sen', 'ne')

# A token after one of these is effectively sentence-initial, so its
# capitalisation carries NO information about proper-nounhood.
_POSITION_RESET = {'.', '!', '?', '«', '"', '„', ':', '(', ';', '—'}

# Abbreviations and initials are a SEPARATE token class, not a proper-noun
# question. `D-ro` (Doktoro), `L.` in "D-ro L. L. Zamenhof". They were 6 of the
# 13 remaining false positives, and DESIGN.md already lists them as
# deterministically fixable.
_ABBREV = re.compile(r'^[A-ZĈĜĤĴŜŬ]\.?$|^[A-ZĈĜĤĴŜŬ]-\w{1,3}\.?$')


@lru_cache(maxsize=1)
def load_roots() -> frozenset[str]:
    """The Esperanto root lexicon — a CLOSED-world list of the language's own
    vocabulary, not an open-world list of the world's names.

    Falls back to the Fundamento alone if the derived lexicon is absent, which
    still works (F1 42.2% at 100% recall) — just with worse precision.
    """
    if _ROOT_VOCAB.exists():
        return frozenset(json.loads(_ROOT_VOCAB.read_text())['roots'])
    if _FUNDAMENTO.exists():
        return frozenset(json.loads(_FUNDAMENTO.read_text()))
    raise FileNotFoundError(
        f'No root lexicon. Build it: python scripts/index/build_root_lexicon.py\n'
        f'  looked in {_ROOT_VOCAB} and {_FUNDAMENTO}')


@lru_cache(maxsize=100_000)
def decomposes_to_root(word: str, _depth: int = 0) -> bool:
    """Does ANY valid Esperanto decomposition of `word` land on a known root?

    Search over the closed affix inventory. Deliberately independent of
    `klareco.parser` — the lexicon is harvested from the parser, so using the
    parser here would be circular (F13).
    """
    roots = load_roots()
    w = word.lower()
    if w in roots:
        return True
    if _depth > 4 or len(w) < 2:
        return False

    for group in (_ENDINGS, _SUFFIXES):
        for a in group:
            if w.endswith(a) and len(w) - len(a) >= 2:
                if decomposes_to_root(w[:-len(a)], _depth + 1):
                    return True
    for p in _PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 2:
            if decomposes_to_root(w[len(p):], _depth + 1):
                return True
    return False


def has_foreign_orthography(word: str) -> bool:
    """Letters/clusters impossible in Esperanto's closed 28-letter alphabet.

    Zamenhof (Lingvaj Respondoj 63) explicitly permits a proper name to keep its
    native spelling — so foreign orthography POSITIVELY LICENSES a name.
    """
    return bool(_NON_EO.search(word))


def is_abbreviation(token: str) -> bool:
    """`D-ro`, `L.`, `S-ro` — a separate token class, not a name question."""
    return bool(_ABBREV.match(token))


def is_proper_noun(token: str,
                   *,
                   prev_token: str | None = None,
                   is_sentence_initial: bool = False) -> bool:
    """Infer whether `token` is a proper noun. No gazetteer involved.

    prev_token / is_sentence_initial supply POSITION, which decides whether
    capitalisation carries any information at all. Position does not veto the
    orthography signal — a foreign spelling is evidence anywhere.
    """
    if not token or not token[:1].isupper():
        return False
    if is_abbreviation(token):
        return False

    # Zamenhof LR63: foreign orthography licenses a name, in ANY position.
    if has_foreign_orthography(token):
        return True

    # ALL-CAPS (a heading) carries no capitalisation signal.
    if token.isupper() and len(token) > 1:
        return False

    # Sentence-initial (or post-`.`/`«`): capitalisation is uninformative, so we
    # have no evidence to act on. This costs recall (`Varsovio` at position 1)
    # and buys a lot of precision; dropping it took precision 83% -> 46%. The
    # honest fix is a bigger eval set (#820), not a guess.
    if is_sentence_initial or (prev_token in _POSITION_RESET):
        return False

    # The load-bearing test: capitalised, mid-sentence, and it does NOT
    # decompose to anything in the Esperanto lexicon.
    return not decomposes_to_root(token)
