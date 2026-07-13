#!/usr/bin/env python3
"""
Acquire the ReVo root inventory — a lexicon that is NOT downstream of our parser.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: network (GitHub API). No models, no store, no parser.
STAGE: Acquire

Description:
    THE PROBLEM THIS SOLVES
    -----------------------
    `build_root_lexicon.py` harvests roots from `subj_radiko` / `verb_radiko` /
    `obj_radiko` — i.e. FROM PARSER OUTPUT. So when the parser is wrong, the
    harvest writes the error down as evidence and feeds it straight back:

        degraded parser splits   organo -> org + an
        harvest records          `org` as a ROOT
        parser reloads,  splits  organo -> org + an        <-- self-reinforcing

    That is failure mode F13 in the DATA pipeline. **A corpus-harvested lexicon
    can never be more correct than the parser that harvested it**, and no
    `--min-count` threshold escapes it: the parser makes the SAME mis-split on
    every occurrence, so the bad root has a HIGH count.

    The only way out is a root inventory with an INDEPENDENT source of truth.

    WHY ReVo
    --------
    Reta Vortaro is the standard Esperanto dictionary. Its source
    (github.com/revuloj/revo-fonto, GPL-2.0) is one XML article per root, and
    **the filename IS the root** — `hund.xml`, `organ.xml`. So the root inventory
    is the directory listing. No parsing, no heuristics, nothing of ours in it.

    It resolves every contaminated case, exactly:

        root        ReVo   our corpus harvest
        organ        YES   YES
        org           no   YES     <- laundered from organo
        amerik       YES    no     <- the REAL root; our harvest never had it
        amerikan      no   YES     <- laundered from amerikano
        mak           no   YES     <- laundered from the NAME Makita
        banan        YES    no     <- the REAL root (banano, NOT ban+an)

    And `esperant` is correctly ABSENT — it is not a root, it is a LEXICALIZED
    form, which is `protected_roots`' job (build_surface_lexical_facts.py).

    WHAT WE TAKE, AND WHAT WE DO NOT
    --------------------------------
    Only the ROOT INVENTORY (the filenames). We do not take definitions, glosses
    or translations — we need to know WHICH STRINGS ARE ESPERANTO ROOTS, not what
    they mean. That keeps the download to one API call instead of a 495 MB clone,
    and it keeps the artifact small and auditable.

    Filenames use the X-SYSTEM (`abajxur` = abaĵur) and carry homonym numbering
    (`abel1`), both of which are undone here.

Pipeline Position:
    GitHub (revuloj/revo-fonto) --[THIS]--> data/raw/eo/dictionaries/revo_roots.json
                                        --> scripts/index/build_root_lexicon.py
                                        --> klareco/parser.py

Usage:
    python scripts/acquire/acquire_revo.py

Inputs:
    - GitHub git-trees API (one request; the contents API truncates at 1000).

Outputs:
    - data/raw/eo/dictionaries/revo_roots.json
      {"roots": [...], "provenance": {...}}

Quality Checks:
    - Refuses to write a truncated tree.
    - Asserts the anchors: `organ` present / `org` absent, `amerik` present /
      `amerikan` absent, `banan` present, `mak` absent. If ReVo ever stops
      separating these, the whole premise of #806 is wrong and we want to know.
    - Reports overlap with the Fundamento (should be near-total).

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #806, #804, #821
See Also: docs/PROPER_NOUNS.md, scripts/index/build_root_lexicon.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

TREE_URL = ('https://api.github.com/repos/revuloj/revo-fonto/'
            'git/trees/master?recursive=1')
OUT = Path('data/raw/eo/dictionaries/revo_roots.json')
FUNDAMENTO = Path('data/vocabularies/fundamento_roots.json')

# ReVo filenames use the x-system (no Unicode in filenames).
_X = (('cx', 'ĉ'), ('gx', 'ĝ'), ('hx', 'ĥ'), ('jx', 'ĵ'), ('sx', 'ŝ'), ('ux', 'ŭ'))

# If ReVo ever fails these, the premise of #806 is wrong and the build must stop.
_MUST_CONTAIN = ('organ', 'amerik', 'banan', 'hund', 'milit', 'regul')
_MUST_NOT_CONTAIN = ('org', 'amerikan', 'mak')


def unx(s: str) -> str:
    for a, b in _X:
        s = s.replace(a, b)
    return s


def main() -> int:
    print(f'  fetching {TREE_URL}')
    req = urllib.request.Request(
        TREE_URL, headers={'Accept': 'application/vnd.github+json',
                           'User-Agent': 'klareco-acquire-revo'})
    with urllib.request.urlopen(req, timeout=120) as r:
        tree = json.loads(r.read())

    if 'tree' not in tree:
        print(f'  ERROR: {tree.get("message")}', file=sys.stderr)
        return 1
    if tree.get('truncated'):
        print('  ERROR: the tree came back TRUNCATED — the root inventory would '
              'be silently incomplete. Refusing to write.', file=sys.stderr)
        return 1

    names = [t['path'][len('revo/'):-len('.xml')] for t in tree['tree']
             if t['path'].startswith('revo/') and t['path'].endswith('.xml')]
    roots = sorted({unx(re.sub(r'\d+$', '', n)) for n in names if n})
    print(f'  articles: {len(names):,}   distinct roots: {len(roots):,}')

    rs = set(roots)
    missing = [r for r in _MUST_CONTAIN if r not in rs]
    present = [r for r in _MUST_NOT_CONTAIN if r in rs]
    if missing or present:
        print(f'  ERROR: ReVo failed the anchors that #806 rests on.\n'
              f'    expected but ABSENT : {missing}\n'
              f'    unexpected, PRESENT : {present}', file=sys.stderr)
        return 1
    print(f'  anchors OK: {_MUST_CONTAIN} present; {_MUST_NOT_CONTAIN} absent')

    if FUNDAMENTO.exists():
        fund = set(json.loads(FUNDAMENTO.read_text(encoding='utf-8')).keys())
        cov = len(fund & rs) / len(fund)
        print(f'  Fundamento coverage: {len(fund & rs):,}/{len(fund):,} ({cov:.1%})')
        if cov < 0.80:
            print('  WARNING: ReVo covers < 80% of the Fundamento — suspicious.',
                  file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps({
        'roots': roots,
        'provenance': {
            'source': 'github.com/revuloj/revo-fonto (Reta Vortaro), GPL-2.0',
            'method': 'root inventory = the revo/*.xml FILENAMES. One article per '
                      'root; the filename IS the root.',
            'n_articles': len(names), 'n_roots': len(roots),
            'derived_from_parser_output': False,
            'note': 'INDEPENDENT of our parser — that is the entire point. '
                    'build_root_lexicon.py harvests from subj_radiko and so '
                    'launders the parser\'s own mis-splits back in as roots '
                    '(org<-organo, amerikan<-amerikano, mak<-Makita). See #806.',
        },
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.rename(OUT)
    print(f'  wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
