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
#
# The six PARTICIPLES (`ant`/`int`/`ont` active, `at`/`it`/`ot` passive) are
# Rule 6 — as core as the endings themselves. Omitting them made `Konsciante`
# (konsci-ant-e), `Planita` (plan-it-a) and `Lanĉita` undecomposable, and so
# falsely NAMES. They also make `Esperanto` decompose (esper-ant-o), which is
# etymologically exactly right — see the lexicalization note below.
_SUFFIXES = ('ant', 'int', 'ont', 'at', 'it', 'ot',            # Rule 6
             'ism', 'ist', 'ind', 'em', 'ec', 'aĵ', 'ul', 'in', 'et', 'eg',
             'ar', 'er', 'uj', 'ej', 'estr', 'ad', 'aĝ', 'an', 'ig', 'iĝ',
             'il', 'obl', 'op', 'um', 'id', 'nj', 'ĉj')

# Prefixes. The official ones, PLUS the prepositions — Esperanto turns any
# preposition into a prefix freely (`sub-skribo`, `trans-nacia`, `antaŭ-parolo`,
# `inter-tempe`), and leaving them out made ordinary words look like names.
_PREFIXES = ('mal', 'ge', 're', 'ek', 'dis', 'eks', 'fi', 'mis', 'pra',
             'bo', 'ĉef', 'vic', 'sen', 'ne',
             'sub', 'super', 'trans', 'tra', 'en', 'el', 'al', 'kun',
             'antaŭ', 'post', 'inter', 'ĉirkaŭ', 'kontraŭ', 'pri', 'per',
             'pro', 'ekster', 'sur', 'apud', 'kun', 'for', 'plur', 'mult')

# A token after one of these is effectively sentence-initial, so its
# capitalisation carries NO information about proper-nounhood.
_POSITION_RESET = {'.', '!', '?', '«', '"', '„', ':', '(', ';', '—'}

# Abbreviations and initials are a SEPARATE token class, not a proper-noun
# question. `D-ro` (Doktoro), `L.` in "D-ro L. L. Zamenhof". They were 6 of the
# 13 remaining false positives, and DESIGN.md already lists them as
# deterministically fixable.
_ABBREV = re.compile(r'^[A-ZĈĜĤĴŜŬ]\.?$|^[A-ZĈĜĤĴŜŬ]-\w{1,3}\.?$'
                     r'|^[IVXLCDM]{1,7}\.?$'          # Roman numerals: II, IV
                     r'|^[A-ZĈĜĤĴŜŬ]\.\w\.?$')        # I.a

# The CLOSED class of ending-less words. Rules 2-7 require every CONTENT word to
# carry a grammatical ending; these are the exceptions the grammar itself names
# (particles, prepositions, conjunctions, correlatives, numerals, primitive
# adverbs). It is finite and it is part of the grammar — not world knowledge, and
# not a gazetteer. Without it, `La` at sentence-start has no ending and would be
# declared a name.
_FUNCTION_WORDS = frozenset("""
la kaj aŭ sed nek do ĉar se ke ol ju des ĉu ne jes ja nu kvankam kvazaŭ
de da al el en sur sub super apud antaŭ post inter tra trans ĉe kun sen por pri
per pro laŭ kontraŭ krom malgraŭ anstataŭ ekster preter ĝis dum po je
mi vi li ŝi ĝi ni ili oni si ci
kiu kio kia kie kiam kial kiel kiom kies
tiu tio tia tie tiam tial tiel tiom ties
iu io ia ie iam ial iel iom ies
ĉiu ĉio ĉia ĉie ĉiam ĉial ĉiel ĉiom ĉies
neniu nenio nenia nenie neniam nenial neniel neniom nenies
unu du tri kvar kvin ses sep ok naŭ dek cent mil nul
nun jam ankoraŭ ankaŭ nur eĉ tre tro plu tuj for tie ĉi
hodiaŭ hieraŭ morgaŭ baldaŭ ambaŭ almenaŭ kvazaŭ apenaŭ preskaŭ adiaŭ
ajn mem plej pli malpli tamen tial ktp
""".split())

# What grammatical ending implies what part of speech (Rules 2-7). This is the
# hinge of the syntactic rule: `Maria` -> `mar-i-a` is an ADJECTIVE form, and an
# adjective cannot be a subject.
_ENDING_POS = (
    (('ojn', 'oj', 'on', 'o'), 'substantivo'),
    (('ajn', 'aj', 'an', 'a'), 'adjektivo'),
    (('en', 'e'), 'adverbo'),
    (('as', 'is', 'os', 'us', 'i', 'u'), 'verbo'),
)

_PLURAL = ('ojn', 'oj', 'ajn', 'aj')
_ACCUSATIVE = ('ojn', 'on', 'ajn', 'an')


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

    # COMPOUNDING — root+root, the most productive word-formation process in the
    # language (`hispan-land-o`, `plur-lingv-ec-o`, `patr-o-lando` with the
    # optional linking vowel). Without it, ordinary compounds are undecomposable
    # and therefore falsely NAMES.
    #
    # Both halves must be >= 3 chars and the LEFT half must be a bare root, not
    # another recursive decomposition — otherwise the search over-generates and
    # starts "explaining" real names as compounds, which costs recall.
    if _depth < 2:
        for i in range(3, len(w) - 2):
            left, right = w[:i], w[i:]
            if left not in roots:
                continue
            if decomposes_to_root(right, _depth + 1):
                return True
            if right[:1] == 'o' and decomposes_to_root(right[1:], _depth + 1):
                return True     # linking vowel: patr-O-lando
    return False


def implied_pos(word: str) -> str | None:
    """The part of speech the word's grammatical ENDING declares (Rules 2-7).

    Esperanto marks POS on the surface, so this is free and exact. `Maria` ends
    in `-a` and is therefore an ADJECTIVE form, whatever else it may be.
    """
    w = word.lower()
    for endings, pos in _ENDING_POS:
        if any(w.endswith(e) for e in endings):
            return pos
    return None


def is_valid_esperanto_word(word: str) -> bool:
    """Is `word` a well-formed Esperanto WORD — not merely a known root?

    **Rules 2-7: every content word must carry a grammatical ending.** `sam` is a
    root; `sama` is a word. `pet` is a root; `peti` is a word. A bare root, or a
    root plus a derivational suffix with no ending (`pet`+`er`), is not a word
    form at all — so a capitalised token that cannot be one is a name, and we
    know that WITHOUT knowing anything about the world.

    This is the fix for a real bug: `decomposes_to_root` matched a bare root and
    so accepted `Sam` and `Peter` as ordinary Esperanto words.
    """
    w = word.lower()
    if w in _FUNCTION_WORDS:
        return True          # the closed ending-less class (la, kaj, mi, tiu, …)

    # Correlatives and pronouns INFLECT (`kiu`->`kiun`/`kiuj`/`kiujn`,
    # `ĉia`->`ĉian`, `mi`->`min`). Strip the inflection and re-check the closed
    # class — otherwise `Kion` and `Ĉian` at sentence-start look like names.
    for infl in ('jn', 'n', 'j'):
        if w.endswith(infl) and w[: -len(infl)] in _FUNCTION_WORDS:
            return True

    for e in _ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 2:
            if decomposes_to_root(w[: -len(e)]):
                return True
    return False


def _agrees(adj: str, noun: str) -> bool:
    """Rule 3: an adjective agrees with its head noun in NUMBER and CASE."""
    a, n = adj.lower(), noun.lower()
    a_pl = any(a.endswith(e) for e in _PLURAL)
    n_pl = any(n.endswith(e) for e in _PLURAL)
    a_acc = any(a.endswith(e) for e in _ACCUSATIVE)
    n_acc = any(n.endswith(e) for e in _ACCUSATIVE)
    return a_pl == n_pl and a_acc == n_acc


def adjective_reading_is_licensed(token: str,
                                  prev_token: str | None,
                                  next_token: str | None) -> bool:
    """Could `token`, read as an ADJECTIVE, actually be one here?

    Rule 3 makes this decidable: an Esperanto adjective must agree with a head
    noun in number and case. If neither neighbour is a noun it agrees with, the
    adjective reading is **ungrammatical** — so the token is not the Esperanto
    word it looks like.

        Centra Oficejo    -> `Centra` agrees with `Oficejo`   -> licensed
        Maria gajnis      -> next is a VERB, no head noun     -> UNLICENSED -> name

    This is the signal token-internal morphology provably cannot see: it lives in
    the rest of the sentence. It is also why `Maria` = `mar-i-a` ("of the sea")
    stops being a genuine ambiguity — the sentence rules the common reading out.
    """
    for nb in (next_token, prev_token):
        if nb and implied_pos(nb) == 'substantivo' and _agrees(token, nb):
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
                   next_token: str | None = None,
                   is_sentence_initial: bool = False) -> bool:
    """Infer whether `token` is a proper noun. No gazetteer involved.

    The rules fire in order of how much they DEDUCE versus how much they GUESS:

      1. foreign orthography      — deductive (Rule 1 + Zamenhof LR63)
      2. not a valid word form    — deductive (Rules 2-7: words need endings)
      3. adjective can't agree    — deductive (Rule 3: agreement is obligatory)
      4. capitalised + no root    — evidential, and needs a signal-bearing position

    Rules 1-3 are grammar, so **position cannot veto them**: they hold at
    sentence-start, where capitalisation says nothing, exactly as they hold
    mid-sentence. Only rule 4 leans on capitalisation, and only rule 4 is vetoed
    by position.

    `prev_token` / `next_token` are the syntactic context. They are what lets us
    reject a reading the token alone cannot: `Maria` IS `mar-i-a` in isolation,
    and stops being so the moment a verb follows it.
    """
    if not token or not token[:1].isupper():
        return False
    if is_abbreviation(token):
        return False

    # (1) Zamenhof LR63: foreign orthography licenses a name, in ANY position.
    if has_foreign_orthography(token):
        return True

    # (2) Rules 2-7: a content word MUST carry a grammatical ending. If the token
    # is not a well-formed Esperanto word, it is not an Esperanto word — and that
    # is true regardless of where it sits. `Sam`, `Peter`, `Zamenhof`, `Varsovio`.
    if not is_valid_esperanto_word(token):
        return True

    # (3) Rule 3: an adjective must agree with a head noun. A capitalised token
    # whose ending makes it an adjective, with no noun to agree with, cannot be
    # the adjective it looks like. `Maria gajnis bronzon` — the SENTENCE rules
    # out the common-word reading that morphology alone accepts.
    #
    # Function words are exempt: `La` and `Kaj` merely END in `-a`, they are not
    # adjectives, and the ending-POS map does not apply to the closed class.
    if (token.lower() not in _FUNCTION_WORDS
            and implied_pos(token) == 'adjektivo'
            and not adjective_reading_is_licensed(token, prev_token, next_token)):
        return True

    # (4) Everything below rests on CAPITALISATION, which is evidential rather
    # than deductive — so, and ONLY here, it can be vetoed by a context in which
    # capitalisation carries no information:
    #
    #   * ALL-CAPS  — a heading capitalises everything, names and nouns alike
    #   * sentence-initial (or post-`.` / `«`) — EVERY sentence starts with a
    #     capital, so being capitalised there says nothing
    #
    # Note these veto rule 4 only. They must NOT veto rules 1-3: `ZAMENHOF` in a
    # heading is still not a well-formed Esperanto word, and that fact does not
    # care about typography.
    if token.isupper() and len(token) > 1:
        return False
    if is_sentence_initial or (prev_token in _POSITION_RESET):
        return False

    return not decomposes_to_root(token)
