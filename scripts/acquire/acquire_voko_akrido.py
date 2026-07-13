#!/usr/bin/env python3
"""
Acquire voko-akrido — typed roots, typed NAME roots, and the SELECTIONAL AFFIX TABLE.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: network (GitHub raw). No models, no store, no parser.
STAGE: Acquire

Description:
    `voko-akrido` (github.com/revuloj/voko-akrido, GPL-3.0) is a SWI-Prolog
    morphological analyser for Esperanto, and it ships pre-generated, MACHINE-
    READABLE lexicons derived from the ReVo XML. Three artifacts, and each one
    fixes a distinct thing that is currently broken here.

    1. r(radiko, POS, officialness).      — 11,158 COMMON roots
       ------------------------------------------------------------------
       `r(ŝvit, ntr, *).`  `r(hund, best, *).`  `r(kre, tr, '1').`
       POS is partly SEMANTIC, not merely grammatical: `subst` 8,723 ·
       `tr`/`ntr` (transitive/intransitive verb) · `adj` · **`best` 250
       (animate)** · **`pers` 28 (person)** · `parc` (kinship). Those last three
       map straight onto the `persono` ontology class that CLAUDE.md says is
       hardcoded debt.
       Officialness: `*` Fundamento · `1`..`10` the Oficialaj Aldonoj · `+`
       unofficial · `!` EVITINDA (deprecated).

    2. nr(NomRadiko, POS, officialness).  — 1,845 PROPER-NAME roots
       ------------------------------------------------------------------
       `nr('Ŝlejer', pers, +).`   `nr('Varsovi', subst, +).`
       **THIS FIXES A BUG I INTRODUCED.** `acquire_revo.py` harvested ReVo by
       FILENAME, and ReVo has ~1,845 articles whose root is a PROPER NOUN
       (`zamenhof.xml`, `varsovio.xml`). They went straight into the common-word
       lexicon, so `Zamenhof` and `Varsovio` began "decomposing" to known roots
       and stopped being detectable as names. voko-akrido separates them for us —
       and types them **person vs place**.

    3. s(Suffix, OutPOS, InPOS). / p(Prefix, POS). / f(Ending, POS).
       ------------------------------------------------------------------
       **THE SELECTIONAL AFFIX TABLE — and it is the whole ballgame.**

           s(in,  _,     best).   % -in- (feminine) attaches ONLY to an ANIMATE
           s(ul,  best,  adj).    % -ul- makes an animate FROM an adjective
           s(ist, best,  _).
           s(ej,  subst, verb).   % lernejo
           s(er,  subst, subst).
           s(ig,  tr,    adj).    s(iĝ, ntr, tr).

       Hana (1998) diagnosed our exact problem and named exactly this fix:

         `papero` -> "element of a pope" *"could be prevented by prohibiting
         assigning the affix `er` to countable nouns. **However, the
         classification of roots is very time consuming.**"*

       Someone did it anyway. This table is the SEMANTIC SUBCATEGORIZATION that a
       deterministic analyser provably lacks — the reason 32.0% of our tokens have
       2+ licensed segmentations. `maŝino` cannot be `maŝ`+`in` because `maŝ` is
       not `best`. The ambiguity does not need a model. It needs THIS.

Pipeline Position:
    voko-akrido --[THIS]--> revo_typed_roots.json   -> build_root_lexicon.py
                        --> revo_name_roots.json    -> proper-noun lexicon
                        --> affix_table.json        -> parser morphology + ontology

Usage:
    python scripts/acquire/acquire_voko_akrido.py

Outputs:
    - data/raw/eo/dictionaries/revo_typed_roots.json
    - data/raw/eo/dictionaries/revo_name_roots.json
    - data/raw/eo/dictionaries/affix_table.json

Quality Checks:
    - Asserts the anchors: `hund` is a common root; `Zamenhof`/`Varsovi` are NAME
      roots and NOT common roots (this is the bug it exists to fix).
    - Asserts the selectional table contains s(in, _, best) — if that ever
      disappears, the ambiguity argument in docs/PARSER_DESIGN.md changes.

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #806, #804, #777 (ontology), #819
See Also: docs/PARSER_DESIGN.md §4, https://github.com/revuloj/voko-akrido
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

BASE = 'https://raw.githubusercontent.com/revuloj/voko-akrido/master'
FILES = {
    'roots': f'{BASE}/pro/revo/v_revo_radikoj.pl',
    'names': f'{BASE}/pro/revo/v_revo_nomoj.pl',
    'affix': f'{BASE}/pro/vrt/v_elementoj.pl',
    'hier':  f'{BASE}/pro/gra/vorto_chr.pl',
}
OUT = Path('data/raw/eo/dictionaries')

# r(radiko, POS, ofc).  — quoted or bare, any of the three args
_R = re.compile(r"^r\(\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*\)\.")
_NR = re.compile(r"^nr\(\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*\)\.")
# s(suffix, OutPOS, InPOS).  `_` means "unrestricted"
_S = re.compile(r"^s\(\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*\)\.")
_P = re.compile(r"^p\(\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*\)\.")
_F = re.compile(r"^f\(\s*'?([^',()]+)'?\s*,\s*'?([^',()]+)'?\s*\)\.")
# sub(Subtype, Supertype).  THE SEMANTIC TYPE HIERARCHY — this is `ontology_edges`.
_SUB = re.compile(r"^sub\(\s*'?([a-zĉĝĥĵŝŭ]+)'?\s*,\s*'?([a-zĉĝĥĵŝŭ]+)'?\s*\)\.")


def fetch(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={'User-Agent': 'klareco-acquire'})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode('utf-8').splitlines()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. common roots -------------------------------------------------
    roots: dict[str, dict] = {}
    for line in fetch(FILES['roots']):
        m = _R.match(line.strip())
        if m:
            r, pos, ofc = m.groups()
            roots[r] = {'pos': pos, 'ofc': ofc}
    pos_dist = Counter(v['pos'] for v in roots.values())
    print(f'  COMMON roots            : {len(roots):,}')
    print(f'    POS/semantic classes  : {dict(pos_dist.most_common(8))}')

    # ---- 2. NAME roots ---------------------------------------------------
    names: dict[str, dict] = {}
    for line in fetch(FILES['names']):
        m = _NR.match(line.strip())
        if m:
            n, pos, ofc = m.groups()
            names[n] = {'pos': pos, 'ofc': ofc}
    name_dist = Counter(v['pos'] for v in names.values())
    print(f'  NAME roots              : {len(names):,}   {dict(name_dist)}')

    # ---- 3. the selectional affix table ----------------------------------
    suffixes: list[dict] = []
    prefixes: list[dict] = []
    endings: list[dict] = []
    for line in fetch(FILES['affix']):
        t = line.strip()
        if (m := _S.match(t)):
            suf, out_pos, in_pos = m.groups()
            suffixes.append({'affix': suf,
                             'out': None if out_pos == '_' else out_pos,
                             'in': None if in_pos == '_' else in_pos})
        elif (m := _P.match(t)):
            pre, pos = m.groups()
            prefixes.append({'affix': pre, 'in': None if pos == '_' else pos})
        elif (m := _F.match(t)):
            end, pos = m.groups()
            endings.append({'ending': end, 'pos': pos})
    print(f'  SELECTIONAL affix table : {len(suffixes)} suffix rules, '
          f'{len(prefixes)} prefix rules, {len(endings)} endings')

    # ---- 4. the SEMANTIC TYPE HIERARCHY ----------------------------------
    # sub(best, subst). sub(pers, best). sub(parc, pers). sub(tr, verb).
    # This IS `ontology_edges` — the table CLAUDE.md reports as 0 rows.
    hierarchy: list[list[str]] = []
    for line in fetch(FILES['hier']):
        m = _SUB.match(line.strip())
        if m and m.group(1) != 'X':
            hierarchy.append([m.group(1), m.group(2)])
    print(f'  SEMANTIC TYPE HIERARCHY : {len(hierarchy)} IS-A edges  '
          f'{[f"{a}<{b}" for a, b in hierarchy[:5]]}')

    # ---- anchors ---------------------------------------------------------
    problems = []
    if 'hund' not in roots:
        problems.append('`hund` missing from COMMON roots')
    for n in ('Zamenhof', 'Varsovi'):
        if n not in names:
            problems.append(f'`{n}` missing from NAME roots')
        if n.lower() in roots:
            problems.append(f'`{n.lower()}` leaked into COMMON roots')
    # the rule the whole ambiguity argument rests on
    if not any(s['affix'] == 'in' and s['in'] == 'best' for s in suffixes):
        problems.append('s(in, _, best) is GONE — the selectional argument in '
                        'docs/PARSER_DESIGN.md rests on it')
    if problems:
        print('\n  ANCHOR FAILURES:')
        for p in problems:
            print(f'    - {p}')
        return 1
    print('\n  anchors OK: `hund` common; `Zamenhof`/`Varsovi` are NAMES, not '
          'common roots; s(in, _, best) present')

    for fn, payload, note in [
        ('revo_typed_roots.json', roots,
         'ReVo COMMON roots via voko-akrido. POS is partly SEMANTIC (best=animate, '
         'pers=person, parc=kinship) — the seed for the empty `persono` ontology class.'),
        ('revo_name_roots.json', names,
         'ReVo PROPER-NAME roots, typed person vs place. acquire_revo.py harvested '
         'ReVo by FILENAME and so mixed these into the common lexicon, which is why '
         '`Zamenhof` and `Varsovio` started decomposing. This separates them.'),
    ]:
        (OUT / fn).write_text(json.dumps({
            'roots': payload,
            'provenance': {'source': 'github.com/revuloj/voko-akrido (GPL-3.0)',
                           'derived_from_parser_output': False, 'note': note},
        }, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'  wrote {OUT / fn}')

    (OUT / 'affix_table.json').write_text(json.dumps({
        'suffixes': suffixes, 'prefixes': prefixes, 'endings': endings,
        'hierarchy': hierarchy,
        'provenance': {
            'source': 'voko-akrido/pro/vrt/v_elementoj.pl (GPL-3.0)',
            'semantics': "s(affix, out, in): the affix takes a stem of POS `in` and "
                         "produces one of POS `out`. null = unrestricted.",
            'why_it_matters':
                "THE SELECTIONAL RESTRICTIONS. s(in, _, best) says the feminine -in- "
                "attaches ONLY to an ANIMATE, so `maŝino` cannot be maŝ+in. This is "
                "the semantic subcategorization a deterministic analyser provably "
                "lacks — the reason 32.0% of our tokens have 2+ licensed "
                "segmentations. Hana (1998) named this exact fix and called building "
                "it 'very time consuming'. Someone built it.",
            'hierarchy':
                "sub(Subtype, Supertype) — the SEMANTIC TYPE HIERARCHY: "
                "parc < pers < best < subst, and tr/ntr < verb. This is literally "
                "`ontology_edges`, which CLAUDE.md reports as 0 rows. hund/kat are "
                "`best` (animate); patr is `parc` (kinship).",
        },
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'  wrote {OUT / "affix_table.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
