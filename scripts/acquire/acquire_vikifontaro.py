#!/usr/bin/env python3
"""
Acquire Vikifontaro (Esperanto Wikisource) — ~8M words, redistributable.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: klareco.corpus_quality
STAGE: Acquire

Description:
    The largest cleanly-redistributable body of Esperanto prose that exists.
    13,514 content pages / ~8.0M words. Two licence layers, both fine:

        the WORKS       hosted because they are PUBLIC DOMAIN
        the TRANSCRIPTION  CC BY-SA 4.0 (siteinfo `rightsinfo`)

    We take the official monthly dump (14.7 MB bz2) rather than scraping — it is
    the sanctioned path and it is one request.

    WHAT WE THROW AWAY, AND WHY
    ---------------------------
    * NON-ARTICLE NAMESPACES. `Paĝo:` (per-scan-page proofreading), `Indekso:`,
      `Kategorio:`, `Vikifontaro:` … These are scaffolding, not prose. Only ns=0.

    * THE FUNDAMENTO. `Fundamento de Esperanto` is a FIVE-LANGUAGE PARALLEL TEXT
      (French, English, German, Russian, Polish alongside the Esperanto). It is the
      single most important document in the language and it would be the single
      worst thing in a monolingual corpus. Excluded by name.

    * ANYTHING THAT DOES NOT READ AS ESPERANTO. Same rule as the Gutenberg shelf,
      and for the same reason: metadata lies, prose does not. Gutenberg tagged
      eleven issues of a bilingual magazine `languages=["eo"]` and we only caught it
      by measuring. So every page is scored by the Esperanto grammar gate and
      dropped if it does not pass. This catches the parallel texts, the
      English-language front matter, and the Latin/Hebrew quotations without our
      needing to enumerate them.

Pipeline Position:
    [THIS] -> data/raw/eo/wikisource/vikifontaro.jsonl -> extract -> treebank

Usage:
    python scripts/acquire/acquire_vikifontaro.py
    python scripts/acquire/acquire_vikifontaro.py --keep-dump   # don't delete the bz2

Outputs:
    - data/raw/eo/wikisource/vikifontaro.jsonl   {title, text, licence, source}

Quality Checks:
    - Reports pages dropped by namespace, by the Fundamento rule, and by the
      Esperanto gate SEPARATELY, so a stripper bug cannot hide as "low yield".

Last Updated: 2026-07-14
Related Issues: #820
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DUMP = ('https://dumps.wikimedia.org/eowikisource/latest/'
        'eowikisource-latest-pages-articles.xml.bz2')
OUT = Path('data/raw/eo/wikisource')
UA = {'User-Agent': 'klareco/1.0 (Esperanto treebank research; '
                    'marc.t.jones@gmail.com)'}

# `Fundamento de Esperanto` is a FIVE-LANGUAGE parallel text. It is the founding
# document of the language and the worst possible thing in a monolingual corpus.
_EXCLUDE_TITLE = re.compile(r'fundamento de esperanto|universala vortaro|'
                            r'ekzercaro.*(franc|angl|german|rus|pol)', re.I)

# ── wikitext stripping ───────────────────────────────────────────────────────
_COMMENT = re.compile(r'<!--.*?-->', re.S)
_REF = re.compile(r'<ref[^>]*?/>|<ref[^>]*?>.*?</ref>', re.S | re.I)
_TAG = re.compile(r'<[^>]+>')
_TABLE = re.compile(r'\{\|.*?\|\}', re.S)
_HEADER = re.compile(r'^\s*=+.*?=+\s*$', re.M)
_LIST = re.compile(r'^[\*#:;]+', re.M)
# [[File:…]] / [[Kategorio:…]] — namespaced links carry no prose.
_NSLINK = re.compile(r'\[\[(?:Dosiero|File|Image|Bildo|Kategorio|Category)'
                     r':[^\]]*?\]\]', re.I)
_LINK = re.compile(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]')
_EXTLINK = re.compile(r'\[https?://\S+\s+([^\]]*)\]')
_BOLD = re.compile(r"'{2,5}")


def _templates(t: str) -> str:
    """Drop {{…}}, honouring nesting. A regex cannot — templates nest."""
    out, depth, i = [], 0, 0
    while i < len(t):
        if t.startswith('{{', i):
            depth += 1
            i += 2
        elif t.startswith('}}', i) and depth:
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(t[i])
            i += 1
    return ''.join(out)


def strip_wikitext(t: str) -> str:
    t = _COMMENT.sub('', t)
    t = _REF.sub('', t)
    t = _TABLE.sub('', t)
    t = _templates(t)
    t = _NSLINK.sub('', t)
    t = _EXTLINK.sub(r'\1', t)
    t = _LINK.sub(r'\1', t)
    t = _TAG.sub('', t)
    t = _HEADER.sub('', t)
    t = _LIST.sub('', t)
    t = _BOLD.sub('', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description='Acquire Esperanto Wikisource')
    ap.add_argument('--keep-dump', action='store_true')
    ap.add_argument('--min-chars', type=int, default=300)
    args = ap.parse_args()

    from klareco.corpus_quality import assess

    OUT.mkdir(parents=True, exist_ok=True)
    dump = OUT / 'eowikisource-latest.xml.bz2'
    if not dump.exists():
        print(f'  downloading {DUMP}')
        req = urllib.request.Request(DUMP, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r, open(dump, 'wb') as f:
            f.write(r.read())
    print(f'  dump: {dump.stat().st_size:,} bytes\n')

    _SENT = re.compile(r'(?<=[.!?])\s+')

    def eo_rate(text: str, sample: int = 60) -> float:
        s = [x.strip().replace('\n', ' ') for x in _SENT.split(text)]
        s = [x for x in s if len(x.split()) >= 4]
        if not s:
            return 0.0
        step = max(1, len(s) // sample)
        probe = s[::step][:sample]
        return sum(1 for x in probe if assess(x).keep) / len(probe)

    kept = []
    n = drop_ns = drop_title = drop_short = drop_eo = 0
    NS = '{http://www.mediawiki.org/xml/export-0.11/}'

    with bz2.open(dump, 'rb') as f:
        for _ev, el in ET.iterparse(f, events=('end',)):
            if not el.tag.endswith('}page'):
                continue
            n += 1
            ns = el.findtext(f'{NS}ns')
            title = el.findtext(f'{NS}title') or ''
            text = el.findtext(f'{NS}revision/{NS}text') or ''
            el.clear()

            # ns=0 is the article namespace. `Paĝo:` (proofreading scans),
            # `Indekso:`, `Kategorio:` … are scaffolding, not prose.
            if ns != '0':
                drop_ns += 1
                continue
            if _EXCLUDE_TITLE.search(title):
                drop_title += 1
                print(f'    EXCLUDED (parallel text): {title}')
                continue
            body = strip_wikitext(text)
            if len(body) < args.min_chars:
                drop_short += 1
                continue
            # MEASURE THE PROSE. Do not trust that a page on the Esperanto
            # Wikisource is in Esperanto — many are parallel or quoted texts.
            r = eo_rate(body)
            if r < 0.90:
                drop_eo += 1
                continue
            kept.append({'title': title, 'text': body, 'esperanto_rate': round(r, 3),
                         'source': 'vikifontaro',
                         'licence': 'PD (work) + CC BY-SA 4.0 (transcription)',
                         'redistributable': True})

    out = OUT / 'vikifontaro.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    chars = sum(len(r['text']) for r in kept)
    print(f'\n  pages in dump          : {n:,}')
    print(f'    dropped: not ns=0    : {drop_ns:,}   (Paĝo:, Indekso:, Kategorio: …)')
    print(f'    dropped: parallel    : {drop_title:,}   (Fundamento — 5-language)')
    print(f'    dropped: too short   : {drop_short:,}')
    print(f'    dropped: NOT ESPERANTO: {drop_eo:,}   <- the gate that caught')
    print('                              The Esperantist on the Gutenberg shelf')
    print(f'\n  KEPT                   : {len(kept):,} pages')
    print(f'  chars                  : {chars:,}  (~{chars // 6:,} words)')
    print(f'\n  wrote {out}')
    if not args.keep_dump:
        dump.unlink()
        print('  (dump deleted; --keep-dump to retain)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
