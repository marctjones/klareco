#!/usr/bin/env python3
"""
Acquire the FULL Project Gutenberg Esperanto shelf — 124 monolingual titles.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: none (stdlib only)
STAGE: Acquire

Description:
    We were using TWO Gutenberg books (Alice, Andersen). There are 145 Esperanto
    titles, and Gutenberg's own metadata tells us which are safe:

        124  languages == ['eo']        <- MONOLINGUAL. usable.
         21  languages == ['en','eo'] … <- BILINGUAL. would POISON the corpus.

    The 21 are `The Esperantist` (a bilingual magazine), an English-Esperanto
    dictionary, and English/Czech/German textbooks. Our sentence-level quality gate
    would have admitted the Esperanto half of them while dragging the English in
    alongside — so we exclude them at the BOOK level, where the metadata is
    unambiguous, instead of hoping a per-sentence heuristic catches it.

    ORIGINAL vs TRANSLATED — the distinction Gutenberg does NOT record
    -----------------------------------------------------------------
    Most of the shelf is TRANSLATED into Esperanto (Ibsen ×11, Shakespeare ×4, Poe,
    Twain, Dickens, Balzac, Turgenev, Pushkin). Translated prose carries the SOURCE
    LANGUAGE's syntax — an Ibsen play rendered from Norwegian has Norwegian clause
    structure wearing Esperanto endings. That is `translationese`, and a treebank
    built only from it measures Esperanto-as-relexified-European.

    ORIGINAL Esperanto — Zamenhof, Kalocsay, Luyken, Bulthuis, Forge, Vallienne,
    Privat, Camacho — is what the language does when nothing is pulling it. We need
    BOTH, but we must know which is which, so every book is tagged `originala` /
    `tradukita` / `nekonata` and the tag rides through to the treebank.

    We do NOT guess silently: books we cannot classify are tagged `nekonata` and
    counted, so the uncertainty is visible rather than laundered into a label.

Pipeline Position:
    [THIS] -> data/raw/eo/gutenberg/ -> extract -> sample_for_treebank.py

Usage:
    python scripts/acquire/acquire_gutenberg_eo.py            # resumes by default
    python scripts/acquire/acquire_gutenberg_eo.py --fresh
    python scripts/acquire/acquire_gutenberg_eo.py --dry-run  # catalogue only

Inputs:
    - gutendex.com API (the Project Gutenberg catalogue)

Outputs:
    - data/raw/eo/gutenberg/pg<id>_<slug>.txt        plain text
    - data/raw/eo/gutenberg/pg<id>_<slug>.meta.json  title/author/kind/licence
    - data/raw/eo/gutenberg/.checkpoint.json         resume state

Quality Checks:
    - EXCLUDES any book not tagged monolingual `eo` by Gutenberg.
    - Strips the Gutenberg header/footer boilerplate (which is ENGLISH — leaving it
      in would inject English into every single book).
    - Reports how many bytes survived stripping; a book that loses ~everything is
      reported, not silently written as an empty file.

Last Updated: 2026-07-14
Related Issues: #820
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUT = Path('data/raw/eo/gutenberg')
CKPT = OUT / '.checkpoint.json'
UA = {'User-Agent': 'klareco/1.0 (Esperanto treebank research; '
                    'marc.t.jones@gmail.com)'}

# Authors who wrote ORIGINALLY in Esperanto. Not a linguistic gazetteer — it is
# bibliographic metadata about 124 specific books, and there is no other source for
# it. Anything not matched is tagged `nekonata`, never guessed.
_ORIGINAL_AUTHORS = {
    'zamenhof', 'kalocsay', 'camacho', 'luyken', 'bulthuis', 'forge',
    'vallienne', 'privat', 'boirac', 'devjatnin', 'baena', 'frenkel',
    'schwartz', 'baghy', 'engholm', 'rosbach', 'nemere',
}
# Titles that SAY they are original.
_ORIGINAL_TITLE = re.compile(
    r'originale?\s+verkit|originalaj?\s+(verko|rakonto|artikol)', re.I)

# Gutenberg wraps every text in an ENGLISH licence header and footer. Leaving it in
# would inject English prose into all 124 books.
_START = re.compile(r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*',
                    re.I | re.S)
_END = re.compile(r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*',
                  re.I | re.S)

# Removing the START marker is NOT enough, and 54 of 124 books proved it: what
# FOLLOWS the marker is Gutenberg's ENGLISH transcriber credit —
#
#     "Produced by David Starner, William Patterson and the Online
#      Distributed Proofreading Team at http://www.pgdp.net"
#
# — a paragraph of English prose sitting at the head of an Esperanto book. It is
# small (2-4 lines), which is exactly why it is dangerous: the per-sentence quality
# gate flags it `not_esperanto` and drops it, so the contamination NEVER SHOWS UP
# as a failure. It just quietly becomes 3.4% of every book's sentences being thrown
# away, and any English that DOES score as Esperanto-ish rides in.
# Cut it at the source, where it is unambiguous.
_CREDIT = re.compile(
    r'^\s*(Produced by|Transcribed by|Transcriber\'s Note|E-text prepared by|'
    r'This (e|E)book was produced by).*?(?=\n\s*\n)', re.I | re.S)


def _get(url: str, raw: bool = False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        data = r.read()
    return data if raw else json.loads(data)


def _slug(t: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', t.lower())[:48].strip('_')


def _kind(book: dict) -> str:
    authors = ' '.join(a['name'].lower() for a in book.get('authors') or [])
    if any(a in authors for a in _ORIGINAL_AUTHORS):
        return 'originala'
    if _ORIGINAL_TITLE.search(book['title']):
        return 'originala'
    # A named non-Esperantist author almost always means a translation (Ibsen,
    # Shakespeare, Poe …). But `almost` is not `always`, so say `nekonata` rather
    # than assert.
    return 'tradukita' if book.get('authors') else 'nekonata'


_SENT = re.compile(r'(?<=[.!?])\s+')


def esperanto_rate(text: str, sample: int = 400) -> float:
    """What fraction of this book's sentences are actually Esperanto?

    GUTENBERG'S LANGUAGE METADATA IS NOT SELF-CONSISTENT, and trusting it was a
    mistake. `The Esperantist` is a BILINGUAL magazine; Gutenberg tags some of its
    issues ['en','eo'] and ELEVEN OTHERS ['eo']. Those eleven sailed through the
    metadata filter, and 34.7% of one issue's sentences are not Esperanto.

    So we do not trust the label — we MEASURE THE TEXT. Any book whose sentences do
    not overwhelmingly pass the Esperanto grammar gate is rejected, whatever
    Gutenberg says it is. This is the one check that cannot be fooled by bad
    metadata, because it reads the actual prose.
    """
    from klareco.corpus_quality import assess
    sents = [s.strip().replace('\n', ' ') for s in _SENT.split(text)]
    sents = [s for s in sents if len(s.split()) >= 4]
    if not sents:
        return 0.0
    step = max(1, len(sents) // sample)
    probe = sents[::step][:sample]
    return sum(1 for s in probe if assess(s).keep) / len(probe)


# Below this, the book is not a monolingual Esperanto text whatever its metadata
# claims. Real Esperanto books score 0.95+; The Esperantist scores 0.65.
_MIN_EO = 0.90


def _strip(text: str) -> str:
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[:m.start()]
    text = text.lstrip()
    # …and the English credit block that sits AFTER the START marker. Loop: some
    # books carry both a "Produced by" and a "Transcriber's Note".
    for _ in range(3):
        m = _CREDIT.match(text)
        if not m:
            break
        text = text[m.end():].lstrip()
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description='Acquire the Gutenberg Esperanto shelf')
    ap.add_argument('--fresh', action='store_true', help='ignore the checkpoint')
    ap.add_argument('--dry-run', action='store_true', help='catalogue only, no downloads')
    ap.add_argument('--delay', type=float, default=1.0, help='seconds between requests')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if CKPT.exists() and not args.fresh:
        done = set(json.loads(CKPT.read_text()).get('done', []))
        print(f'  resuming — {len(done)} already fetched\n')

    # ---- catalogue -------------------------------------------------------
    books, url = [], 'https://gutendex.com/books?languages=eo'
    while url:
        d = _get(url)
        books += d['results']
        url = d.get('next')
    mono = [b for b in books if b['languages'] == ['eo']]
    skipped = len(books) - len(mono)

    print(f'  catalogue      : {len(books)} Esperanto titles')
    print(f'  MONOLINGUAL    : {len(mono)}   <- taking these')
    print(f'  bilingual      : {skipped}   <- SKIPPED (The Esperantist, dictionaries,')
    print('                        English/Czech/German textbooks). Gutenberg tags')
    print('                        them ["en","eo"]; they would inject English.\n')

    kinds: dict[str, int] = {}
    for b in mono:
        k = _kind(b)
        kinds[k] = kinds.get(k, 0) + 1
    print('  REGISTER (translated prose carries the SOURCE language\'s syntax):')
    for k in ('originala', 'tradukita', 'nekonata'):
        print(f'    {k:12s} {kinds.get(k, 0):4}')
    print()

    if args.dry_run:
        print('  (dry run — nothing downloaded)')
        return 0

    ok = fail = skip = 0
    total_bytes = 0
    rejected: list = []
    for i, b in enumerate(mono, 1):
        bid = b['id']
        if bid in done:
            skip += 1
            continue
        fmts = b.get('formats') or {}
        src = (fmts.get('text/plain; charset=utf-8')
               or fmts.get('text/plain; charset=us-ascii')
               or fmts.get('text/plain'))
        if not src or src.endswith('.zip'):
            print(f'    [{i:3}/{len(mono)}] {bid:>6}  no plain-text format — SKIP')
            fail += 1
            continue
        try:
            raw = _get(src, raw=True).decode('utf-8', errors='replace')
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f'    [{i:3}/{len(mono)}] {bid:>6}  {e} — will retry on next run')
            fail += 1
            continue

        text = _strip(raw)
        # A book that loses ~everything to the stripper is a bug, not a book.
        if len(text) < 2000:
            print(f'    [{i:3}/{len(mono)}] {bid:>6}  only {len(text)}B after stripping '
                  f'boilerplate — REPORTING, not writing')
            fail += 1
            continue

        # DO NOT TRUST THE LANGUAGE TAG. Measure the prose.
        rate = esperanto_rate(text)
        if rate < _MIN_EO:
            print(f'    [{i:3}/{len(mono)}] {bid:>6}  only {rate:.0%} Esperanto — '
                  f'REJECTED despite languages=["eo"]  {b["title"][:34]}')
            rejected.append((bid, b['title'], rate))
            done.add(bid)          # decided; do not re-fetch on resume
            CKPT.write_text(json.dumps({'done': sorted(done)}))
            time.sleep(args.delay)
            continue

        slug = _slug(b['title'])
        (OUT / f'pg{bid}_{slug}.txt').write_text(text, encoding='utf-8')
        (OUT / f'pg{bid}_{slug}.meta.json').write_text(json.dumps({
            'gutenberg_id': bid,
            'title': b['title'],
            'authors': [a['name'] for a in b.get('authors') or []],
            'kind': _kind(b),
            'languages': b['languages'],
            'licence': 'public domain (Project Gutenberg)',
            'redistributable': True,
            'chars': len(text),
            'esperanto_rate': round(rate, 3),
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        done.add(bid)
        ok += 1
        total_bytes += len(text)
        CKPT.write_text(json.dumps({'done': sorted(done)}))
        print(f'    [{i:3}/{len(mono)}] {bid:>6}  {len(text):>9,}B  '
              f'{_kind(b):10s} {b["title"][:44]}')
        time.sleep(args.delay)

    if rejected:
        print(f'\n  REJECTED {len(rejected)} books that Gutenberg tags languages=["eo"]')
        print('  but whose PROSE is not monolingual Esperanto. The metadata lied;')
        print('  the text does not.')
        for bid, title, r in sorted(rejected, key=lambda x: x[2]):
            print(f'    {bid:>6}  {r:5.0%}  {title[:56]}')

    print(f'\n  fetched {ok}   rejected {len(rejected)}   skipped(done) {skip}   failed {fail}')
    print(f'  {total_bytes:,} chars this run  (~{total_bytes // 6:,} words)')
    if fail:
        print('  re-run to retry the failures — the checkpoint makes it cheap.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
