"""Enumerate every licensed segmentation — then RANK them by selectional restrictions.

This is the architecture the literature converged on and we did not have:

    **The symbolic analyser enumerates the licensed readings.
     A separate stage ranks them.**

A finite-state analyser emits EVERY accepting path, unranked — there is no
preference operator in lexc/twolc/lttoolbox, and longest-match is a heuristic
(94.4%), not the mechanism. Our parser instead returned ONE reading and discarded
the rest *silently*, which is how `Esperanton` -> `esper+ant` happened. It did not
fail; it committed.

Measured on our own corpus: **32.0% of running-text tokens have 2+ licensed
segmentations** (independently replicating Guinard 2016). Every one of them is
grammatical.

WHAT RANKS THEM
---------------
Hana (1998) diagnosed the problem and named the fix:

    `papero` -> `pap`+`er` ("element of a pope") *"could be prevented by
    prohibiting assigning the affix `er` to countable nouns. However, the
    classification of roots is very time consuming."*

That is a SEMANTIC SUBCATEGORIZATION fact — the class of the root. It is not in
the grammar and cannot be. **voko-akrido (GPL-3.0) has it**, and we now ship it:

    r(hund, best, *).       hund is ANIMATE
    r(patr, parc, *).       patr is KINSHIP
    r(kuir, tr,   *).       kuir is a TRANSITIVE VERB

    s(in,  _,     best).    -in-  attaches ONLY to an ANIMATE
    s(ul,  best,  adj).     -ul-  makes an animate FROM an adjective
    s(ej,  subst, verb).    -ej-  makes a place FROM a verb
    s(ig,  tr,    adj).     -ig-  makes a transitive verb FROM an adjective

    sub(best, subst).  sub(pers, best).  sub(parc, pers).  sub(tr, verb).
                            ^ the SEMANTIC TYPE HIERARCHY

WHY IT RANKS RATHER THAN FILTERS
--------------------------------
Because the table is imperfect and the lexicon is imperfect. `vir` is tagged
`subst` in ReVo, not `best` — so `s(in, _, best)` would *strictly forbid*
`virino`, which is an ordinary word. A hard filter would delete real language.

So a violation COSTS points; it does not kill the reading. That keeps the
analyser's recall intact while its precision improves, and it means the ranking
degrades gracefully as the lexicon improves rather than breaking.

**And the ranker is still fully deterministic.** No learned parameters. The
residue that survives THIS is the residue that genuinely needs a model — and that
is exactly the boundary this project exists to find.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DICT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'eo' / 'dictionaries'

# Grammatical endings (Rules 2-7) -> the POS they declare.
ENDINGS: dict[str, str] = {
    'o': 'subst', 'oj': 'subst', 'on': 'subst', 'ojn': 'subst',
    'a': 'adj', 'aj': 'adj', 'an': 'adj', 'ajn': 'adj',
    'e': 'adv', 'en': 'adv',
    'i': 'verb', 'as': 'verb', 'is': 'verb', 'os': 'verb', 'us': 'verb',
    'u': 'verb',
}
_ENDINGS_LONGEST = sorted(ENDINGS, key=len, reverse=True)


@dataclass(frozen=True)
class Morpheme:
    form: str
    kind: str          # 'radiko' | 'sufikso' | 'prefikso' | 'finaĵo'
    pos: str | None = None


@dataclass
class Analysis:
    morphemes: list[Morpheme]
    pos: str | None = None            # POS of the whole word
    score: float = 0.0
    violations: list[str] = field(default_factory=list)

    @property
    def radiko(self) -> str:
        for m in self.morphemes:
            if m.kind == 'radiko':
                return m.form
        return ''

    @property
    def sufiksoj(self) -> list[str]:
        return [m.form for m in self.morphemes if m.kind == 'sufikso']

    @property
    def prefiksoj(self) -> list[str]:
        return [m.form for m in self.morphemes if m.kind == 'prefikso']

    def __repr__(self) -> str:
        parts = []
        for m in self.morphemes:
            parts.append({'prefikso': f'{m.form}-', 'sufikso': f'-{m.form}',
                          'finaĵo': f'|{m.form}'}.get(m.kind, m.form))
        v = f'  ✗{len(self.violations)}' if self.violations else ''
        return f'{" ".join(parts)}  [{self.pos}] {self.score:+.1f}{v}'


class Lexicon:
    """The typed lexicon + selectional table + type hierarchy (voko-akrido)."""

    def __init__(self) -> None:
        self.roots: dict[str, str] = {}        # root -> POS/semantic class
        self.names: dict[str, str] = {}        # NAME root -> pers | subst
        self.suffix_rules: dict[str, list[tuple[str | None, str | None]]] = {}
        self.prefix_rules: dict[str, list[str | None]] = {}
        self.prefixes: set[str] = set()
        self._super: dict[str, set[str]] = {}

        tr = _DICT_DIR / 'revo_typed_roots.json'
        nr = _DICT_DIR / 'revo_name_roots.json'
        af = _DICT_DIR / 'affix_table.json'
        if not (tr.exists() and af.exists()):
            raise FileNotFoundError(
                f'Typed lexicon missing ({tr} / {af}).\n'
                '  Acquire it:  python scripts/acquire/acquire_voko_akrido.py\n'
                'Without the SELECTIONAL TABLE there is nothing to rank with, and '
                '32% of tokens stay arbitrarily disambiguated.')

        self.roots = {k: v['pos']
                      for k, v in json.loads(tr.read_text(encoding='utf-8'))['roots'].items()}
        if nr.exists():
            self.names = {k.lower(): v['pos']
                          for k, v in json.loads(nr.read_text(encoding='utf-8'))['roots'].items()}
            # A name root is a ROOT for morphology — `amerik` must be present or
            # `amerikano` cannot decompose.
            for k, pos in self.names.items():
                self.roots.setdefault(k, pos)

        a = json.loads(af.read_text(encoding='utf-8'))
        for s in a['suffixes']:
            self.suffix_rules.setdefault(s['affix'], []).append((s['out'], s['in']))
        self.prefixes = {p['affix'] for p in a['prefixes']}
        for p in a['prefixes']:
            self.prefix_rules.setdefault(p['affix'], []).append(p['in'])
        # A prefix with an unrestricted rule (mal-, pseŭdo-) selects for nothing.
        for k, v in list(self.prefix_rules.items()):
            if None in v:
                self.prefix_rules[k] = []

        # Transitive closure of sub(Subtype, Supertype).
        direct: dict[str, set[str]] = {}
        for sub, sup in a.get('hierarchy', []):
            direct.setdefault(sub, set()).add(sup)
        for x in list(direct):
            seen, stack = set(), [x]
            while stack:
                y = stack.pop()
                for z in direct.get(y, ()):
                    if z not in seen:
                        seen.add(z)
                        stack.append(z)
            self._super[x] = seen

    def isa(self, pos: str | None, target: str | None) -> bool:
        """Does `pos` satisfy a requirement for `target`? (parc IS-A pers IS-A best…)"""
        if target is None or pos is None:
            return True
        return pos == target or target in self._super.get(pos, ())


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    return Lexicon()


# Scoring. Two principles, and the ORDER of magnitude between them is what does
# the work — not the exact numbers.
#
# 1. OCCAM / LEXICON AUTHORITY. Fewer morphemes wins. If the dictionary lists
#    `paper` as a root, that reading beats `pap`+`er`, because positing an extra
#    morpheme needs a reason. Getting this backwards is EXACTLY Hana's 1998 bug:
#    a satisfied restriction must be worth *nothing* (it is merely "not wrong"),
#    or every extra affix would pay for itself and the analyser would happily
#    decide that `papero` is an "element of a pope".
#
# 2. A SELECTIONAL VIOLATION IS EXPENSIVE — but it is a COST, NOT A VETO.
#    `vir` is tagged `subst` in ReVo, not `best`, so `s(in, _, best)` strictly
#    forbids `virino` — an ordinary word. A hard filter would delete real
#    language. So a violating reading survives; it just loses.
_SCORE_ROOT_KNOWN = 3.0        # the root is in the curated lexicon
_SCORE_SELECTION_OK = 0.0      # restriction satisfied: NOT WRONG is not a reward
_PENALTY_SELECTION_BAD = -3.0  # restriction violated: expensive, never fatal
_PENALTY_PER_MORPHEME = -1.0   # Occam: `organ` beats `org`+`an`; `paper` beats `pap`+`er`


def _analyses_of_stem(stem: str, depth: int = 0) -> list[list[Morpheme]]:
    """EVERY licensed morpheme decomposition of `stem`. Unranked, like an FST."""
    lex = lexicon()
    if depth > 4 or len(stem) < 2:
        return []
    out: list[list[Morpheme]] = []
    if stem in lex.roots:
        out.append([Morpheme(stem, 'radiko', lex.roots[stem])])
    for suf, _rules in lex.suffix_rules.items():
        if stem.endswith(suf) and len(stem) - len(suf) >= 2:
            for inner in _analyses_of_stem(stem[: -len(suf)], depth + 1):
                out.append(inner + [Morpheme(suf, 'sufikso')])
    for pre in lex.prefixes:
        if stem.startswith(pre) and len(stem) - len(pre) >= 2:
            for inner in _analyses_of_stem(stem[len(pre):], depth + 1):
                out.append([Morpheme(pre, 'prefikso')] + inner)

    # DEDUPE. `malkovrit` is reachable by two paths — strip the suffix first
    # (mal-kovr, then -it) or strip the prefix first (mal-, then kovr-it) — and
    # both yield the IDENTICAL morpheme sequence. Without this, the same reading
    # appears twice, ties with itself, and gets counted as ambiguity that does
    # not exist. It was inflating the measured residue.
    seen: set[tuple] = set()
    uniq: list[list[Morpheme]] = []
    for ms in out:
        key = tuple((m.form, m.kind) for m in ms)
        if key not in seen:
            seen.add(key)
            uniq.append(ms)
    return uniq


def _score(morphemes: list[Morpheme]) -> Analysis:
    """Rank a reading by whether its affixes' SELECTIONAL RESTRICTIONS hold.

    `maŝino` = maŝ + in:  `-in-` demands an ANIMATE; `maŝ` is `subst`. Violation.
    `hundino` = hund + in: `hund` is `best`. Satisfied.
    """
    lex = lexicon()
    a = Analysis(morphemes=list(morphemes))
    cur: str | None = None

    # PREFIXES select too, and they were the ENTIRE remaining residue.
    #     p(re,  verb).   re- means "again" and demands a VERB
    #     p(ge,  best).   ge- demands an ANIMATE
    #     p(bo,  parc).   bo- (in-law) demands a KINSHIP term
    # A prefix attaches to the ROOT that FOLLOWS it, so this needs a look-ahead
    # rather than the left-to-right accumulation the suffixes use:
    #     registaro = reg(VERB) + ist + ar    vs    re- + gist(NOUN = "yeast") + ar
    #     `re-` demands a verb, `gist` is a noun -> the re- reading is penalised
    #     and the government wins over the collection of re-yeasted things.
    root_pos = next((m.pos for m in morphemes if m.kind == 'radiko'), None)
    for m in morphemes:
        if m.kind != 'prefikso':
            continue
        reqs = [p for p in lex.prefix_rules.get(m.form, [])]
        if reqs and not any(lex.isa(root_pos, r) for r in reqs):
            a.score += _PENALTY_SELECTION_BAD
            a.violations.append(
                f'{m.form}- demands {{{", ".join(str(r) for r in reqs)}}}, '
                f'got {root_pos}')

    for m in morphemes:
        if m.kind == 'radiko':
            cur = m.pos
            a.score += _SCORE_ROOT_KNOWN
        elif m.kind == 'sufikso':
            rules = lex.suffix_rules.get(m.form, [])
            ok = [(out, req) for out, req in rules if lex.isa(cur, req)]
            if ok:
                a.score += _SCORE_SELECTION_OK
                cur = ok[0][0] or cur
            else:
                a.score += _PENALTY_SELECTION_BAD
                a.violations.append(
                    f'-{m.form}- demands '
                    f'{{{", ".join(str(r) for _, r in rules)}}}, got {cur}')
                cur = rules[0][0] if rules else cur
        a.score += _PENALTY_PER_MORPHEME

    a.pos = cur
    return a


@lru_cache(maxsize=100_000)
def analyze(word: str) -> tuple[Analysis, ...]:
    """All licensed analyses of a word form, BEST FIRST.

    Returns the SET. The caller may take `[0]`, but the rest are still there —
    which is the whole point: a parser that returns one reading where the grammar
    licenses two is not deterministic, it is arbitrary.
    """
    w = word.lower()
    out: list[Analysis] = []
    for end in _ENDINGS_LONGEST:
        if w.endswith(end) and len(w) - len(end) >= 2:
            for ms in _analyses_of_stem(w[: -len(end)]):
                a = _score(ms + [Morpheme(end, 'finaĵo', ENDINGS[end])])
                # The ending declares the surface POS (Rules 2-7) and always wins:
                # `hundo` is a noun even though `hund` is `best`.
                a.pos = ENDINGS[end]
                out.append(a)
    if w in lexicon().roots:                       # bare root (rare; particles)
        out.append(_score([Morpheme(w, 'radiko', lexicon().roots[w])]))
    out.sort(key=lambda a: -a.score)
    return tuple(out)


def best(word: str) -> Analysis | None:
    a = analyze(word)
    return a[0] if a else None


def is_ambiguous(word: str) -> bool:
    """More than one licensed reading — i.e. the grammar did not decide."""
    return len(analyze(word)) > 1
