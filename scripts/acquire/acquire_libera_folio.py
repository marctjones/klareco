#!/usr/bin/env python3
"""
Acquire Libera Folio — modern, ORIGINAL, CC-BY Esperanto journalism.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: klareco.corpus_quality
STAGE: Acquire

Description:
    WHY THIS MATTERS MORE THAN ITS SIZE
    -----------------------------------
    Of the 113 public-domain Gutenberg books we hold, only **18 are original
    Esperanto**. Eighty-six are TRANSLATIONS — eleven Ibsen plays, four
    Shakespeares, Poe, Twain, Dickens, Balzac, Turgenev — and a translation carries
    the SOURCE language's syntax. An Ibsen play rendered from Norwegian has
    Norwegian clause structure wearing Esperanto endings.

    A treebank built mostly on that measures Esperanto-as-relexified-European,
    which is precisely the assumption this project exists to NOT make.

    Libera Folio is contemporary journalism WRITTEN in Esperanto by Esperantists.
    Nothing is pulling its syntax toward another language. It is 834 articles — far
    smaller than the Gutenberg shelf — and it is worth more per token than any of
    it, because it is the register we are starving for.

    LICENCE — and the escape hatch we must respect
    ---------------------------------------------
    The site footer states:

        "Se nenio alia estas indikita ĉe la koncerna artikolo, la tekstoj de
         Libera Folio estas disponeblaj laŭ la permesilo Krea Komunaĵo Atribuite
         4.0 Tutmonda"   [= CC BY 4.0]

    Note "se nenio alia estas indikita" — UNLESS OTHERWISE INDICATED. It is a
    default, not a blanket grant. We therefore record the per-article link and date
    in every row, so that if a specific article turns out to carry its own terms it
    can be traced and removed, rather than being silently laundered into a corpus
    we later publish.

    robots.txt disallows only /wp-admin/. The WordPress REST API is the sanctioned,
    documented read path — we use it rather than scraping HTML.

Pipeline Position:
    [THIS] -> data/raw/eo/libera_folio/libera_folio.jsonl -> extract -> treebank

Usage:
    python scripts/acquire/acquire_libera_folio.py

Outputs:
    - data/raw/eo/libera_folio/libera_folio.jsonl
      {title, text, url, date, licence, esperanto_rate}

Quality Checks:
    - Same empirical language gate as the Gutenberg shelf: MEASURE the prose, do
      not trust that a site's articles are all in the language the site is in.

Last Updated: 2026-07-14
Related Issues: #820
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

API = 'https://www.liberafolio.org/wp-json/wp/v2/posts'
OUT = Path('data/raw/eo/libera_folio')
UA = {'User-Agent': 'klareco/1.0 (Esperanto treebank research; '
                    'marc.t.jones@gmail.com)'}

LICENCE = ('CC BY 4.0 — site default ("se nenio alia estas indikita"); '
           'per-article terms may differ, hence the url/date on every row')

_TAG = re.compile(r'<[^>]+>')
_SCRIPT = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.S | re.I)
_SENT = re.compile(r'(?<=[.!?])\s+')


def _text(rendered: str) -> str:
    t = _SCRIPT.sub('', rendered)
    t = re.sub(r'</p>|<br\s*/?>', '\n', t, flags=re.I)
    t = _TAG.sub('', t)
    t = html.unescape(t)
    return re.sub(r'\n{3,}', '\n\n', t).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description='Acquire Libera Folio (CC BY 4.0)')
    ap.add_argument('--delay', type=float, default=0.5)
    args = ap.parse_args()

    from klareco.corpus_quality import assess

    def eo_rate(text: str, sample: int = 60) -> float:
        s = [x.strip().replace('\n', ' ') for x in _SENT.split(text)]
        s = [x for x in s if len(x.split()) >= 4]
        if not s:
            return 0.0
        step = max(1, len(s) // sample)
        probe = s[::step][:sample]
        return sum(1 for x in probe if assess(x).keep) / len(probe)

    OUT.mkdir(parents=True, exist_ok=True)
    kept, page, drop_eo, drop_short = [], 1, 0, 0

    while True:
        url = f'{API}?per_page=100&page={page}&_fields=title,content,link,date'
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                posts = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400:      # past the last page
                break
            raise
        if not posts:
            break

        for p in posts:
            body = _text((p.get('content') or {}).get('rendered', ''))
            if len(body) < 300:
                drop_short += 1
                continue
            # MEASURE. Libera Folio occasionally quotes at length in other
            # languages; the site being Esperanto does not make every article so.
            r = eo_rate(body)
            if r < 0.90:
                drop_eo += 1
                continue
            kept.append({
                'title': html.unescape(
                    _TAG.sub('', (p.get('title') or {}).get('rendered', ''))),
                'text': body,
                'url': p.get('link'),
                'date': p.get('date'),
                'esperanto_rate': round(r, 3),
                'source': 'libera_folio',
                'kind': 'originala',   # written IN Esperanto, not translated
                'licence': LICENCE,
                'redistributable': True,
            })
        print(f'    page {page:>3}  cumulative kept {len(kept):,}')
        page += 1
        time.sleep(args.delay)

    out = OUT / 'libera_folio.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    chars = sum(len(r['text']) for r in kept)
    print(f'\n  KEPT                 : {len(kept):,} articles')
    print(f'    dropped: too short : {drop_short:,}')
    print(f'    dropped: not eo    : {drop_eo:,}')
    print(f'  chars                : {chars:,}  (~{chars // 6:,} words)')
    print('\n  ALL OF IT IS ORIGINAL ESPERANTO — the register the Gutenberg shelf')
    print('  is starving for (18 original books vs 86 translations).')
    print(f'\n  wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
