#!/usr/bin/env python3
"""
Acquire the ReVo ONTOLOGY — senses, hypernyms, domains, typed entity lists.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: git (clones revuloj/revo-fonto, GPL-2.0). No models, no parser.
STAGE: Acquire

Description:
    CLAUDE.md says, of this project's own ontology:

        "`ontology_nodes` and `ontology_edges` are EMPTY and `verb_klaso` is 0%
         populated ... the 'always query the ontology' rule below is currently
         UNFOLLOWABLE, and a couple of paths fall back to hardcoded lists. That is
         ACKNOWLEDGED DEBT."

    and, honestly:

        "even when loaded, the ontology is hand-seeded and thin (`kreado-26` =
         ["fond","kre","produk","far"]; `persono` = ["homo","vir","infan",
         "kuracist"]) ... Lexical synonymy is a genuine learned residue we are
         currently faking with a list."

    **It does not have to be faked.** ReVo ships all of it, curated, GPL-2.0:

        <ref tip="super">   ~8,680 HYPERNYM edges — a real taxonomy
        <ref tip="sin">     ~2,985 synonyms
        <ref tip="ant">     ~623   antonyms
        <ref tip="prt">     ~4,931 meronyms / holonyms
        <uzo tip="fak">     ~22,769 DOMAIN labels, 78 distinct (ZOO, BOT, MED, GEOG…)
        <ref lst="voko:…">  ~4,528 TYPED ENTITY LISTS, 135 distinct:
                                voko:urboj (311 cities) · voko:personaj_nomoj (294)
                                voko:ŝtatoj · voko:riveroj · voko:historiaj_personoj
        <snc>               NUMBERED SENSES per root — the sense inventory itself

    `voko:urboj` and `voko:personaj_nomoj` ARE the `loko` and `persono` classes —
    attested, curated, and exactly what the Decision Checklist in CLAUDE.md tells
    us to query instead of hardcoding.

    AND THE SENSES ARE WHERE THE NEXT RESIDUE LIVES
    -----------------------------------------------
    `hund.xml` has TWO senses:
        1. "Dombesto apartenanta al tiu genro, devenanta de lupo…"   (the animal)
        2. "Insultvorto por atakema viro"                            (an insult)

    Which one is meant is not a grammatical question. Bick measured it on
    Esperanto (Arbobanko): 3.8% of noun lemmas, 2.4% of adjectives and 2.2% of
    verbs are semantically ambiguous IN THE CORPUS, and the lexicon's UNREALIZED
    ambiguity potential is ~3x higher (10.5% / 8.2% / 7.7%).

    Some of that will fall to the SELECTIONAL RESTRICTIONS we already have — the
    same table that took morpheme ambiguity to 0.285%. The rest is world knowledge,
    and it must become an OR-node, not a guess.

Pipeline Position:
    revo-fonto XML --[THIS]--> revo_ontology.json --> ontology_nodes/edges (#837)
                                                  --> sense-level OR-nodes

Usage:
    python scripts/acquire/acquire_revo_ontology.py
    python scripts/acquire/acquire_revo_ontology.py --repo /path/to/revo-fonto

Outputs:
    - data/raw/eo/dictionaries/revo_ontology.json
      {"roots": {root: {senses, hypernyms, synonyms, domains, lists}}, ...}

Quality Checks:
    - Anchors: `hund` must have >=2 senses and a ZOO domain label.
    - Reports the class sizes for `voko:urboj` / `voko:personaj_nomoj` — these
      replace the hardcoded `loko` / `persono` lists CLAUDE.md calls debt.

Last Updated: 2026-07-14
Author: Claude (with Marc Jones)
Related Issues: #830, #777, #837, EPIC #713
See Also: CLAUDE.md ("Ontology status: defined in code, NOT loaded at runtime")
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = 'https://github.com/revuloj/revo-fonto.git'
OUT = Path('data/raw/eo/dictionaries/revo_ontology.json')

_RAD = re.compile(r'<rad>([^<]+)</rad>')
# TOP-LEVEL senses only. <snc> NESTS (subsenses), and a naive non-greedy match
# counted `hund` as having 21 senses. A root has a handful of senses, not 21.
_SNC = re.compile(r'<snc\b[^>]*>((?:(?!<snc\b).)*?)</snc>', re.S)
_DIF = re.compile(r'<dif>(.*?)</dif>', re.S)
# A ReVo article is a ROOT plus its DERIVED FORMS, each in a <drv> with its own
# <kap> headword and its own senses. `hund.xml` holds hundo (3 senses), hunda (2),
# hundejo, hundido, hundino, hundĉaso… Lumping all 21 onto the root `hund` is
# wrong: sense disambiguation needs senses keyed to the WORD, not the morpheme.
# `<tld/>` is ReVo's placeholder for the root itself.
_DRV = re.compile(r'<drv\b[^>]*>(.*?)</drv>', re.S)
_KAP = re.compile(r'<kap>(.*?)</kap>', re.S)
_TLD = re.compile(r'<tld[^>]*/?>')
_REF = re.compile(r'<ref\b([^>]*)>(.*?)</ref>', re.S)
_UZO = re.compile(r'<uzo\s+tip="fak"[^>]*>([^<]+)</uzo>')
_TAGS = re.compile(r'<[^>]+>')
_WS = re.compile(r'\s+')

# ReVo's XML predates Unicode-in-XML and uses SGML entities for the six accented
# letters. Without decoding, class names come out as `voko:&ccirc;efurboj` and
# definitions are unreadable.
_ENTITIES = {
    '&ccirc;': 'ĉ', '&gcirc;': 'ĝ', '&hcirc;': 'ĥ', '&jcirc;': 'ĵ',
    '&scirc;': 'ŝ', '&ubreve;': 'ŭ',
    '&Ccirc;': 'Ĉ', '&Gcirc;': 'Ĝ', '&Hcirc;': 'Ĥ', '&Jcirc;': 'Ĵ',
    '&Scirc;': 'Ŝ', '&Ubreve;': 'Ŭ',
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'",
}


def _entities(s: str) -> str:
    for a, b in _ENTITIES.items():
        s = s.replace(a, b)
    return s


def _clean(s: str, root: str = '') -> str:
    """Strip tags — but first put the ROOT back where <tld/> stands for it.

    ReVo writes definitions as "Ago <tld/>i aŭ ties rezulto". Stripping tags
    naively yields "Ago i aŭ ties rezulto", which is gibberish. `<tld/>` IS the
    root, and it must be substituted, not deleted.
    """
    if root:
        s = _TLD.sub(root, s)
    return _entities(_WS.sub(' ', _TAGS.sub('', s)).strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--repo', help='an existing revo-fonto checkout')
    args = ap.parse_args()

    tmp = None
    if args.repo:
        root_dir = Path(args.repo)
    else:
        tmp = tempfile.mkdtemp()
        print(f'  cloning {REPO} (shallow) …')
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none',
                        REPO, tmp], check=True, capture_output=True)
        root_dir = Path(tmp)

    xml_dir = root_dir / 'revo'
    files = sorted(xml_dir.glob('*.xml'))
    if not files:
        print(f'  ERROR: no XML under {xml_dir}', file=sys.stderr)
        return 1
    print(f'  {len(files):,} ReVo articles')

    roots: dict[str, dict] = {}
    dom_count: Counter = Counter()
    list_members: dict[str, list[str]] = defaultdict(list)
    n_hyper = n_syn = n_sense = 0

    for f in files:
        try:
            x = f.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        m = _RAD.search(x)
        root = _entities((m.group(1) if m else f.stem).strip())

        # SENSES, keyed to the WORD FORM (not the root).
        forms: dict[str, list[str]] = {}
        for body in _DRV.findall(x):
            kap = _KAP.search(body)
            if not kap:
                continue
            head = _clean(kap.group(1), root).lstrip('*').strip()
            head = head.split()[0].strip('.,;:') if head else ''
            if not head:
                continue
            defs = []
            for snc in _SNC.findall(body):
                d = _DIF.search(snc)
                if d:
                    defs.append(_clean(d.group(1), root)[:160])
            if not defs:
                d = _DIF.search(body)
                if d:
                    defs.append(_clean(d.group(1), root)[:160])
            if defs:
                forms[head] = defs
                n_sense += len(defs)
        senses = forms.get(root + 'o') or forms.get(root + 'i') \
            or forms.get(root + 'a') or []

        hyper, syn, lists = [], [], []
        for attrs, body in _REF.findall(x):
            tip = re.search(r'tip="([^"]+)"', attrs)
            lst = re.search(r'lst="([^"]+)"', attrs)
            txt = _clean(body)
            if lst:
                name = _entities(lst.group(1))
                lists.append(name)
                list_members[name].append(root)
            if not tip or not txt:
                continue
            if tip.group(1) == 'super':
                hyper.append(txt)
                n_hyper += 1
            elif tip.group(1) == 'sin':
                syn.append(txt)
                n_syn += 1

        domains = _UZO.findall(x)
        dom_count.update(domains)

        if senses or hyper or syn or domains or lists:
            roots[root] = {
                'senses': senses,          # the root's OWN senses (root+o / +i / +a)
                'formoj': forms,           # every derived form -> ITS senses
                'hypernyms': sorted(set(hyper)),
                'synonyms': sorted(set(syn)),
                'domains': sorted(set(domains)),
                'lists': sorted(set(lists)),
            }

    polysemous = sum(1 for v in roots.values() if len(v['senses']) > 1)
    n_forms = sum(len(v.get('formoj', {})) for v in roots.values())
    print(f'\n  roots with ontology data : {len(roots):,}')
    print(f'  WORD FORMS with senses   : {n_forms:,}')
    print(f'  SENSES                   : {n_sense:,}  '
          f'({polysemous:,} ROOTS are POLYSEMOUS in their base form — the sense residue)')
    print(f'  HYPERNYM edges           : {n_hyper:,}   <- a real taxonomy')
    print(f'  synonym edges            : {n_syn:,}')
    print(f'  DOMAIN labels            : {sum(dom_count.values()):,} '
          f'({len(dom_count)} distinct)  {[d for d, _ in dom_count.most_common(6)]}')
    print(f'  TYPED ENTITY LISTS       : {len(list_members)} distinct')

    print('\n  the classes that replace the HARDCODED lists CLAUDE.md calls debt:')
    for k in sorted(list_members, key=lambda x: -len(list_members[x]))[:8]:
        print(f'    {k:32s} {len(list_members[k]):5,} members')

    problems = []
    h = roots.get('hund')
    if not h or len(h['senses']) < 2:
        problems.append('`hund` should have >=2 senses (the animal, and the insult)')
    if not h or 'ZOO' not in h['domains']:
        problems.append('`hund` should carry the ZOO domain label')
    if problems:
        print(f'\n  ANCHOR FAILURES: {problems}', file=sys.stderr)
        return 1
    print(f'\n  anchors OK: `hund` has {len(h["senses"])} senses, domains={h["domains"]}')
    for i, s in enumerate(h['senses'][:2], 1):
        print(f'      sense {i}: {s[:72]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmpf = OUT.with_suffix('.tmp')
    tmpf.write_text(json.dumps({
        'roots': roots,
        'lists': {k: sorted(set(v)) for k, v in list_members.items()},
        'provenance': {
            'source': 'github.com/revuloj/revo-fonto (Reta Vortaro), GPL-2.0',
            'derived_from_parser_output': False,
            'note': 'The ontology CLAUDE.md reports as 0 rows. `voko:urboj` and '
                    '`voko:personaj_nomoj` ARE the `loko` and `persono` classes — '
                    'attested and curated, not hand-seeded. The <snc> senses are '
                    'the sense inventory: `hund` means BOTH the animal and an '
                    'insult for an aggressive man, and no grammar rule can choose.',
        },
    }, ensure_ascii=False), encoding='utf-8')
    tmpf.rename(OUT)
    print(f'\n  wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
