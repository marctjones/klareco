#!/usr/bin/env python3
"""
Count the LICENSED PARSES the parser is silently throwing away.

VERSION: v1.0
COMPATIBLE WITH: root_vocab.json v2 (ReVo-first)
DEPENDENCIES: data/vocabularies/root_vocab.json; duckdb (for corpus frequencies)
STAGE: Evaluation

Description:
    "A parser that returns ONE parse where the grammar licenses TWO is not
    deterministic — it is arbitrary."

    Ours returns one. That is how `Esperanton` -> esper+ant happened: the parser
    did not FAIL, it silently COMMITTED. This script makes the discarded readings
    visible and counts them, with zero learned parameters.

    WHY THIS IS THE RIGHT MEASUREMENT
    ---------------------------------
    A finite-state analyzer emits EVERY accepting path, unranked — there is no
    preference operator in lexc/twolc/lttoolbox. Listing `organ` as a root does
    NOT suppress `org+an`, because `org` and `an` are themselves legitimate
    entries. Published Esperanto measurements:

        Hana (1998, PC-KIMMO):   13.6% lexical homonymy over 460k words
          doktoro -> dok|tor|o ("dock+torus") · dokt|or|o ("erudite+gold") · doktor|o
          papero  -> pap|er|o  ("element of a pope")   · paper|o
        Guinard (2016, PBML 105): mean 2.15 segmentations/word; 53.5% of words
          have >= 2; MAXIMUM 112.  katokulo = kat'okul'o (cat eye)
                                            = kat'o'kul'o (cat-like gnat)

    So the ambiguity is INTRINSIC to productive derivation, not a symptom of a
    broken lexicon. Apertium sidesteps it by refusing to derive at all — 93,100
    hand-listed stems, no productive affix rules — and pays for it: Bick reports
    25.1% of noun lemmas in real text are NOT in the lexicon and must be guessed.
    You can have generativity or you can have unambiguity. Not both.

    WHAT THE NUMBER MEANS
    ---------------------
    The ambiguity rate here IS the residue, quantified, before a single learned
    parameter is spent. It answers, empirically, whether a learned disambiguator
    is worth having — and it is exempt from the merge gate, because it is
    measurement, not capability.

    AND IT LOCATES WHERE THE INFORMATION IS MISSING
    ----------------------------------------------
    Hana diagnosed his own failure precisely, in 1998:

      `papero` -> "element of a pope" "could be prevented by prohibiting assigning
      the affix `er` to countable nouns. However, the classification of roots is
      very time consuming."

    That is a SEMANTIC SUBCATEGORIZATION fact — which roots are countable objects,
    which are substances, which are persons. It is not in the grammar and cannot
    be. It is exactly what an ONTOLOGY holds, and klareco's ontology tables are
    empty (see CLAUDE.md). Guinard's Markov model over morpheme SEMANTIC CLASSES
    (98.9% vs 94.4% for longest-match) is the statistical stand-in for that same
    missing information.

    So the boundary is not "morphology is hard". It is:

      **The deterministic analyzer provably lacks the semantic class of the root,
      and therefore cannot know that `paper` cannot take `-er-`. Supply the class
      and the ambiguity collapses. Withhold it and you need a ranker.**

Pipeline Position:
    root_vocab.json + corpus vocabulary --[THIS]--> ambiguity rate = the residue

Usage:
    python scripts/eval/measure_morpheme_ambiguity.py
    python scripts/eval/measure_morpheme_ambiguity.py --limit 200000

Outputs:
    - stdout report: segmentations/word, % ambiguous, worst offenders

Quality Checks:
    - Reports TYPE and TOKEN ambiguity separately (frequent words are usually
      unambiguous; the tail is where the ambiguity lives).
    - Compares against Guinard's published 53.5% / 2.15.

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #819, #804, #806
See Also: https://ufal.mff.cuni.cz/pbml/105/art-guinard.pdf
          https://ufal.mff.cuni.cz/~hana/esr/thesis.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ENDINGS = ('ojn', 'oj', 'on', 'o', 'ajn', 'aj', 'an', 'a',
           'en', 'e', 'as', 'is', 'os', 'us', 'i', 'u')
SUFFIXES = ('ant', 'int', 'ont', 'at', 'it', 'ot',
            'ism', 'ist', 'ind', 'em', 'ec', 'aĵ', 'ul', 'in', 'et', 'eg',
            'ar', 'er', 'uj', 'ej', 'estr', 'ad', 'aĝ', 'an', 'ig', 'iĝ',
            'il', 'obl', 'op', 'um', 'id')
PREFIXES = ('mal', 'ge', 're', 'ek', 'dis', 'eks', 'fi', 'mis', 'pra',
            'bo', 'ĉef', 'vic', 'sen', 'ne', 'sub', 'super', 'trans', 'tra',
            'en', 'el', 'al', 'kun', 'antaŭ', 'post', 'inter', 'pri')
WORD_RE = re.compile(r"[a-zĉĝĥĵŝŭ]+")

ROOTS: frozenset[str] = frozenset()


@lru_cache(maxsize=200_000)
def segmentations(stem: str, depth: int = 0) -> tuple:
    """EVERY licensed morpheme decomposition of `stem` — not just the first.

    This is what an FST does. The parser picks one and discards the rest.
    """
    if depth > 4 or len(stem) < 2:
        return ()
    out = []
    if stem in ROOTS:
        out.append((stem,))
    for suf in SUFFIXES:
        if stem.endswith(suf) and len(stem) - len(suf) >= 2:
            for inner in segmentations(stem[: -len(suf)], depth + 1):
                out.append(inner + ('-' + suf,))
    for pre in PREFIXES:
        if stem.startswith(pre) and len(stem) - len(pre) >= 2:
            for inner in segmentations(stem[len(pre):], depth + 1):
                out.append((pre + '-',) + inner)
    # root + root compounding
    if depth < 2:
        for i in range(3, len(stem) - 2):
            left, right = stem[:i], stem[i:]
            if left in ROOTS:
                for inner in segmentations(right, depth + 1):
                    out.append((left, '+') + inner)
    return tuple(dict.fromkeys(out))


def analyses(word: str) -> tuple:
    """All licensed analyses of a full word form (ending stripped first)."""
    out = []
    for e in ENDINGS:
        if word.endswith(e) and len(word) - len(e) >= 2:
            for seg in segmentations(word[: -len(e)]):
                out.append(seg + ('|' + e,))
    return tuple(dict.fromkeys(out))


def main() -> int:
    global ROOTS
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--limit', type=int, default=400_000,
                    help='sentences to sample for the vocabulary')
    args = ap.parse_args()

    d = json.loads(Path('data/vocabularies/root_vocab.json').read_text(encoding='utf-8'))
    ROOTS = frozenset(d['roots'])
    print(f'  lexicon: {len(ROOTS):,} roots (ReVo + Fundamento + corpus)\n')

    import duckdb
    con = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
    rows = con.execute(
        f'SELECT text FROM sentences USING SAMPLE {args.limit} ROWS (reservoir, 11)'
    ).fetchall()

    freq: Counter = Counter()
    for (t,) in rows:
        if t:
            freq.update(WORD_RE.findall(t.lower()))
    print(f'  sampled {len(rows):,} sentences -> {len(freq):,} word TYPES, '
          f'{sum(freq.values()):,} TOKENS\n')

    n_types = n_amb_types = 0
    n_tokens = n_amb_tokens = 0
    total_seg = 0
    dist: Counter = Counter()
    worst: list = []

    for w, c in freq.items():
        if len(w) < 3:
            continue
        a = analyses(w)
        if not a:
            continue                       # unanalysable — a different problem
        k = len(a)
        n_types += 1
        n_tokens += c
        total_seg += k
        dist[min(k, 6)] += 1
        if k > 1:
            n_amb_types += 1
            n_amb_tokens += c
            worst.append((k, w, c, a))

    print('  ═══ LICENSED ANALYSES PER WORD — the parses we silently discard ═══\n')
    print(f'    mean analyses per word TYPE : {total_seg / n_types:.2f}   '
          f'(Guinard 2016 published: 2.15)')
    print(f'    TYPES with >= 2 analyses    : {n_amb_types:,}/{n_types:,} = '
          f'{n_amb_types / n_types:.1%}   (Guinard: 53.5%)')
    print(f'    TOKENS with >= 2 analyses   : {n_amb_tokens:,}/{n_tokens:,} = '
          f'{n_amb_tokens / n_tokens:.1%}   <- the rate in RUNNING TEXT')
    print()
    print('    distribution (analyses per type):')
    for k in sorted(dist):
        lab = f'{k}' if k < 6 else '6+'
        print(f'      {lab:>3s}  {"#" * (dist[k] * 40 // n_types):40s} '
              f'{dist[k]:6,} ({dist[k] / n_types:5.1%})')

    worst.sort(key=lambda x: (-x[0], -x[2]))
    print('\n    most ambiguous words (all readings are GRAMMATICALLY LEGAL):')
    for k, w, c, a in worst[:8]:
        print(f'      {w:16s} {k:3d} analyses (x{c:,})')
        for seg in a[:3]:
            print(f'          {" ".join(seg)}')
        if k > 3:
            print(f'          … and {k - 3} more')

    print('\n  ═══ WHAT THIS NUMBER IS ═══')
    print('    This is the RESIDUE, quantified, with zero learned parameters.')
    print('    Every one of these readings is licensed by the 16 rules and the')
    print('    lexicon. The grammar has done its job and returned N answers.')
    print('    Choosing among them is a RANKING problem, not a grammar problem —')
    print('    and Hana (1998) named the missing information exactly: the SEMANTIC')
    print('    CLASS of the root (`paper` is a substance, so it cannot take -er-).')
    print('    That is what the ONTOLOGY holds, and ours is empty.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
