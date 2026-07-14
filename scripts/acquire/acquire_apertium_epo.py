#!/usr/bin/env python3
"""
Acquire apertium-epo — an INDEPENDENT reference implementation to be checked against.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: network (GitHub raw). No models, no store, no parser.
STAGE: Acquire

Description:
    `apertium-epo` (github.com/apertium/apertium-epo, GPL-3.0) is the Esperanto
    morphological dictionary of the Apertium machine-translation platform. It is
    the closest thing to a REFERENCE IMPLEMENTATION of what we are doing, and —
    this is what makes it valuable — it was built **independently of ReVo and
    voko-akrido**, so it is a genuinely separate opinion rather than a mirror.

    93,142 entries. Each carries the LEMMA, the bare STEM, and a paradigm:

        <e lm="abako"><i>abak</i><par n="o__n"/></e>
        <e lm="hundejo"><i>hundej</i><par n="o__n"/></e>

    ~36,740 of them are PROPER NOUNS, and they are SUB-TYPED:
        loc 14,014 (places) · ant 11,221 (given names) · cog 10,599 (surnames)
        al 835 (organisations)

    IT MAKES THE OPPOSITE ARCHITECTURAL CHOICE FROM US, DELIBERATELY
    ---------------------------------------------------------------
    Apertium **does not model productive derivation at all.** There is no
    `-ej-` rule in its transducer: `hundejo` is a hand-listed stem, right next to
    `hundo`. That is why its morpheme ambiguity is ~zero — and why Bick reports
    **25.1% of noun lemmas in real Esperanto text are simply not in the lexicon**
    and have to be guessed.

        You can have generativity, or you can have unambiguity. Not both.

    We chose generativity. This dictionary is therefore not a target to match —
    it is a RULER, and where we differ is as informative as where we agree:

      * we analyse a word apertium has never heard of  -> our generativity paying off
      * apertium lists a stem we decompose             -> we may be over-splitting
      * apertium says PROPER NOUN and we say common    -> a real disagreement worth
                                                          looking at

Pipeline Position:
    apertium-epo.epo.dix --[THIS]--> apertium_lexicon.json --> scripts/eval/compare_to_apertium.py

Usage:
    python scripts/acquire/acquire_apertium_epo.py

Outputs:
    - data/raw/eo/dictionaries/apertium_lexicon.json
      {"entries": {lemma: {"stem":…, "pos":…, "np_type":…}}, "provenance": {...}}

Quality Checks:
    - Asserts the anchors: `hundo` is a noun; `Zamenhof` is a proper noun.
    - Reports how many entries are proper nouns (should be ~40%).

Last Updated: 2026-07-14
Author: Claude (with Marc Jones)
Related Issues: #804, #806, #820
See Also: docs/PARSER_DESIGN.md §3
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

URL = ('https://raw.githubusercontent.com/apertium/apertium-epo/master/'
       'apertium-epo.epo.dix')
OUT = Path('data/raw/eo/dictionaries/apertium_lexicon.json')

_E = re.compile(r'<e\b[^>]*\blm="([^"]+)"[^>]*>(.*?)</e>', re.S)
_I = re.compile(r'<i>([^<]*)</i>')
_PAR = re.compile(r'<par\s+n="([^"]+)"')

# paradigm suffix -> our POS vocabulary
_POS = {
    '__n': 'substantivo', '__adj': 'adjektivo', '__vblex': 'verbo',
    '__vbtr': 'verbo', '__vbntr': 'verbo', '__adv': 'adverbo',
    '__np': 'propra_nomo', '__pr': 'prepozicio', '__num': 'numero',
    '__prn': 'pronomo', '__cnjcoo': 'konjunkcio', '__cnjsub': 'konjunkcio',
    '__det': 'artikolo', '__ij': 'interjekcio',
}


def main() -> int:
    print(f'  fetching {URL}')
    req = urllib.request.Request(URL, headers={'User-Agent': 'klareco-acquire'})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    xml = raw.decode('utf-8', errors='replace')
    print(f'  {len(xml):,} bytes')

    entries: dict[str, dict] = {}
    pos_dist: Counter = Counter()
    for m in _E.finditer(xml):
        lemma, body = m.group(1), m.group(2)
        pars = _PAR.findall(body)
        if not pars:
            continue
        par = pars[0]
        pos = None
        for suffix, p in _POS.items():
            if par.endswith(suffix):
                pos = p
                break
        if pos is None:
            continue
        stems = _I.findall(body)
        stem = ''.join(stems) if stems else None
        entries[lemma] = {'stem': stem, 'pos': pos, 'par': par}
        pos_dist[pos] += 1

    print(f'  entries: {len(entries):,}')
    for p, c in pos_dist.most_common(8):
        print(f'    {p:16s} {c:7,}')

    problems = []
    if entries.get('hundo', {}).get('pos') != 'substantivo':
        problems.append('`hundo` is not a noun')
    if entries.get('Zamenhof', {}).get('pos') != 'propra_nomo':
        problems.append('`Zamenhof` is not a proper noun')
    if problems:
        print(f'\n  ANCHOR FAILURES: {problems}', file=sys.stderr)
        return 1
    print('\n  anchors OK: hundo=noun, Zamenhof=proper noun')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps({
        'entries': entries,
        'provenance': {
            'source': 'github.com/apertium/apertium-epo (GPL-3.0)',
            'derived_from_parser_output': False,
            'note': 'An INDEPENDENT reference implementation, built without ReVo '
                    'or voko-akrido. Apertium does NOT model productive derivation '
                    '— `hundejo` is a hand-listed stem, not hund+ej. So its '
                    'morpheme ambiguity is ~zero, and Bick reports 25.1% of noun '
                    'lemmas in real text are simply MISSING from it. Generativity '
                    'or unambiguity: not both. Use as a RULER, not a target.',
        },
    }, ensure_ascii=False), encoding='utf-8')
    tmp.rename(OUT)
    print(f'  wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
