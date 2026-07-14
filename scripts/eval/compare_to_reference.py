#!/usr/bin/env python3
"""
Compare klareco against apertium-epo — an INDEPENDENT reference implementation.

VERSION: v1.0
COMPATIBLE WITH: klareco.parser, apertium_lexicon.json
DEPENDENCIES: data/raw/eo/dictionaries/apertium_lexicon.json; duckdb
STAGE: Evaluation

Description:
    There is no Esperanto model for Stanza, UDPipe or Trankit — the free gold data
    is 3,343 tokens, far below their training threshold. And EspGram, the one
    system that beats us, is PROPRIETARY. So the only reference implementation we
    can actually run against is **apertium-epo** (GPL-3.0), and it is a good one
    precisely because it was built INDEPENDENTLY of ReVo and voko-akrido.

    IT MAKES THE OPPOSITE ARCHITECTURAL CHOICE, AND THAT IS THE POINT
    ----------------------------------------------------------------
    Apertium does NOT model productive derivation. `hundejo` is a hand-listed stem
    sitting next to `hundo`; there is no `-ej-` rule in its transducer. So:

        its morpheme ambiguity  ~  ZERO
        its lexicon gap          =  25.1% of noun lemmas in real text (Bick 2020)

        You can have GENERATIVITY or you can have UNAMBIGUITY. Not both.

    We chose generativity. So this is not a target to match — it is a RULER, and
    the DISAGREEMENTS are the interesting part:

      * klareco analyses a word apertium has never seen  -> generativity paying off
      * apertium lists a stem klareco decomposes         -> possible over-splitting
      * they disagree about PROPER-NOUN-hood             -> a real, checkable claim

Pipeline Position:
    corpus vocabulary + klareco.parser + apertium --[THIS]--> agreement + disagreements

Usage:
    python scripts/eval/compare_to_reference.py
    python scripts/eval/compare_to_reference.py --limit 100000

Quality Checks:
    - Reports COVERAGE for both systems separately: a system that analyses fewer
      words can look more accurate, and must not be allowed to.

Last Updated: 2026-07-14
Author: Claude (with Marc Jones)
Related Issues: #806, #820
See Also: docs/PARSER_DESIGN.md §3
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.morphology import analyze
from klareco.parser import parse_word

REF = Path('data/raw/eo/dictionaries/apertium_lexicon.json')
WORD = re.compile(r"[a-zĉĝĥĵŝŭA-ZĈĜĤĴŜŬ]+")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--limit', type=int, default=200_000)
    ap.add_argument('--top', type=int, default=20_000,
                    help='compare the N most frequent word types')
    args = ap.parse_args()

    if not REF.exists():
        raise FileNotFoundError(
            f'{REF} missing.\n'
            '  Acquire it: python scripts/acquire/acquire_apertium_epo.py')
    ref = json.loads(REF.read_text(encoding='utf-8'))['entries']
    ref_lower = {k.lower(): v for k, v in ref.items()}
    print(f'  apertium-epo reference: {len(ref):,} entries '
          f'({sum(1 for v in ref.values() if v["pos"] == "propra_nomo"):,} proper nouns)')

    con = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
    rows = con.execute(
        f'SELECT text FROM sentences USING SAMPLE {args.limit} ROWS (reservoir, 3)'
    ).fetchall()
    freq: collections.Counter = collections.Counter()
    for (t,) in rows:
        if t:
            freq.update(WORD.findall(t))
    vocab = [w for w, _ in freq.most_common(args.top) if len(w) >= 3]
    print(f'  comparing the {len(vocab):,} most frequent word types '
          f'from {len(rows):,} sentences\n')

    both = agree_pos = 0
    only_us = only_them = neither = 0
    pos_conf: collections.Counter = collections.Counter()
    we_split_they_list: list = []
    propn_disagree: list = []

    for w in vocab:
        lw = w.lower()
        r = ref_lower.get(lw)
        try:
            ours = parse_word(w)
        except Exception:
            ours = None
        we_know = bool(ours) and ours.get('vortspeco') not in (
            'nekonata', 'fremda_vorto', None)

        if r and we_know:
            both += 1
            theirs, mine = r['pos'], ours['vortspeco']
            if theirs == mine:
                agree_pos += 1
            else:
                pos_conf[(theirs, mine)] += 1
                if {theirs, mine} == {'propra_nomo', 'substantivo'} \
                        and len(propn_disagree) < 12:
                    propn_disagree.append((w, theirs, mine))
            # they LIST a stem we DECOMPOSE -> we may be over-splitting
            a = analyze(lw)
            if (a and r.get('stem') and len(a[0].morphemes) > 2
                    and r['stem'] == lw[:len(r['stem'])]
                    and a[0].radiko != r['stem']
                    and len(we_split_they_list) < 12):
                we_split_they_list.append((w, r['stem'], repr(a[0])))
        elif we_know:
            only_us += 1
        elif r:
            only_them += 1
        else:
            neither += 1

    n = len(vocab)
    print('  ═══ COVERAGE — who can analyse the word at all? ═══\n')
    print(f'    BOTH                     {both:>7,}  ({both / n:5.1%})')
    print(f'    ONLY klareco             {only_us:>7,}  ({only_us / n:5.1%})   '
          f'<- generativity: apertium has never seen these')
    print(f'    ONLY apertium            {only_them:>7,}  ({only_them / n:5.1%})   '
          f'<- their hand-listed stems we cannot build')
    print(f'    NEITHER                  {neither:>7,}  ({neither / n:5.1%})')
    print()
    print(f'    klareco coverage         {(both + only_us) / n:5.1%}')
    print(f'    apertium coverage        {(both + only_them) / n:5.1%}')
    print()
    print('  ═══ AGREEMENT, on the words BOTH can analyse ═══\n')
    print(f'    POS agreement            {agree_pos:>7,} / {both:,} = '
          f'{agree_pos / both:5.1%}' if both else '    (none)')
    print()
    print('    top disagreements (apertium -> klareco):')
    for (t, m), c in pos_conf.most_common(8):
        print(f'      {t:14s} -> {m:14s} {c:5,}')
    if propn_disagree:
        print('\n    PROPER-NOUN disagreements (a real, checkable claim):')
        for w, t, m in propn_disagree[:8]:
            print(f'      {w:18s} apertium={t:14s} klareco={m}')
    if we_split_they_list:
        print('\n    we DECOMPOSE what apertium LISTS (possible over-splitting):')
        for w, stem, a in we_split_they_list[:8]:
            print(f'      {w:18s} apertium stem={stem:14s} ours={a}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
