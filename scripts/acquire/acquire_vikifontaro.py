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

    THE TEXT IS IN `Paĝo:`, NOT IN THE ARTICLE NAMESPACE
    ----------------------------------------------------
    This is counter-intuitive and I got it wrong the first time. Measured from the
    dump:

        ns=104  `Paĝo:`   22,126 pages   37,282,626 chars   80.8% of all text
        ns=0    articles  10,998 pages    6,371,706 chars   13.8%

    Wikisource runs the ProofreadPage extension. A main-namespace page is usually a
    TRANSCLUSION STUB — `<pages index="Foo.djvu" from=1 to=20 />` — and the actual
    proofread prose lives in the `Paĝo:` namespace, one page of the scan per wiki
    page. Dropping ns=104 as "scaffolding" throws away FOUR FIFTHS OF THE CORPUS
    and leaves you wondering why an 8M-word source yielded 152k words.

    So we take BOTH, and we reassemble `Paĝo:` pages into their parent work
    (`Paĝo:<index>/<n>` → group by index, order by n) so that sentences spanning a
    page break are not severed.

    PROOFREADING QUALITY IS FREE, AND IT IS LOAD-BEARING
    ---------------------------------------------------
    Every `Paĝo:` carries `<pagequality level="N">`:

        4  validated — proofread, then checked by a SECOND person
        3  proofread by one person
        2  incomplete
        1  NOT proofread — RAW OCR
        0  no text

    Raw OCR in a gold treebank would be a catastrophe: we would be annotating
    scanner errors. We take level >= 3 only, and we REPORT how many pages each
    level cost us, because "we used the proofread subset" is a claim that has to
    carry a number.

    WHAT ELSE WE THROW AWAY, AND WHY
    --------------------------------
    * THE FUNDAMENTO. `Fundamento de Esperanto` is a FIVE-LANGUAGE PARALLEL TEXT
      (French, English, German, Russian, Polish alongside the Esperanto). It is the
      single most important document in the language and it would be the single
      worst thing in a monolingual corpus. Excluded by name.

    * `<noinclude>` BLOCKS — running heads, folio numbers, catchwords. They are
      marked noinclude precisely because they are NOT the body text.

    * ANYTHING THAT DOES NOT READ AS ESPERANTO. Same rule as the Gutenberg shelf,
      and for the same reason: metadata lies, prose does not. Gutenberg tagged
      eleven issues of a bilingual magazine `languages=["eo"]` and we only caught it
      by measuring. Being ON the Esperanto Wikisource does not make a page
      Esperanto. So every work is scored by the Esperanto grammar gate.

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

# ProofreadPage quality. 4 = validated by a SECOND person, 3 = proofread once,
# 1 = RAW OCR. Annotating raw OCR would mean annotating scanner errors.
_QUALITY = re.compile(r'<pagequality\s+level="(\d)"', re.I)
_MIN_QUALITY = 3

# `Paĝo:Zamenhof - Foo.djvu/190` → work = "Zamenhof - Foo.djvu", page = 190.
_PAGO = re.compile(r'^Paĝo:(.+?)/(\d+)$')

# Running heads, folio numbers, catchwords. Marked noinclude BECAUSE they are not
# the body text.
_NOINCLUDE = re.compile(r'<noinclude>.*?</noinclude>', re.S | re.I)
_NOINC_OPEN = re.compile(r'</?noinclude>', re.I)
_INCLUDEONLY = re.compile(r'<includeonly>|</includeonly>', re.I)
# A page break may sever a word: "esper-\n" + "anto" → "esperanto".
_HYPHEN_BREAK = re.compile(r'(\w)[-­]\s*$')

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

    import collections
    NS = '{http://www.mediawiki.org/xml/export-0.11/}'
    n = drop_title = drop_ns = 0
    qual = collections.Counter()
    works: dict[str, list] = collections.defaultdict(list)   # Paĝo:, by parent work
    articles: list = []                                       # ns=0

    with bz2.open(dump, 'rb') as f:
        for _ev, el in ET.iterparse(f, events=('end',)):
            if not el.tag.endswith('}page'):
                continue
            n += 1
            ns = el.findtext(f'{NS}ns')
            title = el.findtext(f'{NS}title') or ''
            text = el.findtext(f'{NS}revision/{NS}text') or ''
            el.clear()

            if _EXCLUDE_TITLE.search(title):
                drop_title += 1
                continue

            if ns == '104':                      # `Paĝo:` — 80.8% of all the text
                m = _PAGO.match(title)
                if not m:
                    drop_ns += 1
                    continue
                # PROOFREADING QUALITY. level 1 is RAW OCR; annotating it would
                # mean annotating scanner errors.
                q = _QUALITY.search(text)
                lvl = int(q.group(1)) if q else 0
                qual[lvl] += 1
                if lvl < _MIN_QUALITY:
                    continue
                works[m.group(1)].append((int(m.group(2)), text))
            elif ns == '0':
                articles.append((title, text))
            else:
                drop_ns += 1

    kept, drop_short, drop_eo = [], 0, 0

    def _emit(title: str, body: str, kind: str) -> None:
        nonlocal drop_short, drop_eo
        if len(body) < args.min_chars:
            drop_short += 1
            return
        r = eo_rate(body)
        if r < 0.90:
            drop_eo += 1
            return
        kept.append({'title': title, 'text': body, 'esperanto_rate': round(r, 3),
                     'source': 'vikifontaro', 'ns': kind,
                     'licence': 'PD (work) + CC BY-SA 4.0 (transcription)',
                     'redistributable': True})

    # Reassemble each scanned work from its pages, IN ORDER, healing words that a
    # page break severed ("esper-" / "anto").
    for work, pages in works.items():
        buf: list[str] = []
        for _num, raw in sorted(pages):
            t = _NOINCLUDE.sub('', raw)      # running heads, folio numbers
            t = _NOINC_OPEN.sub('', t)
            t = _INCLUDEONLY.sub('', t)
            t = strip_wikitext(t)
            if not t:
                continue
            if buf and _HYPHEN_BREAK.search(buf[-1]):
                buf[-1] = _HYPHEN_BREAK.sub(r'\1', buf[-1]) + t.lstrip()
            else:
                buf.append(t)
        _emit(work, '\n'.join(buf).strip(), 'pago')

    for title, raw in articles:
        _emit(title, strip_wikitext(raw), 'artikolo')

    out = OUT / 'vikifontaro.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    chars = sum(len(r['text']) for r in kept)
    print(f'\n  pages in dump            : {n:,}')
    print(f'    other namespaces       : {drop_ns:,}')
    print(f'    Fundamento (5-language): {drop_title:,}')
    print('\n  PROOFREADING QUALITY of the Paĝo: pages (raw OCR would poison a')
    print('  gold treebank — we are annotating text, not scanner errors):')
    for lvl, lbl in ((4, 'validated (2nd person)'), (3, 'proofread'),
                     (2, 'incomplete'), (1, 'RAW OCR'), (0, 'no text')):
        mark = '  <- taken' if lvl >= _MIN_QUALITY else '  <- DROPPED'
        print(f'    level {lvl}  {qual[lvl]:6,}  {lbl:24s}{mark}')
    print(f'\n  scanned works reassembled: {len(works):,}')
    print(f'    dropped: too short     : {drop_short:,}')
    print(f'    dropped: NOT ESPERANTO : {drop_eo:,}')
    print(f'\n  KEPT                     : {len(kept):,} texts')
    print(f'  chars                    : {chars:,}  (~{chars // 6:,} words)')
    print(f'\n  wrote {out}')
    if not args.keep_dump:
        dump.unlink()
        print('  (dump deleted; --keep-dump to retain)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
