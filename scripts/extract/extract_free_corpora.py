#!/usr/bin/env python3
"""
Extract sentences from the redistributable corpora — Gutenberg, Vikifontaro, Libera Folio.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: klareco.corpus_quality
STAGE: Extract

Description:
    Turns the three acquired sources (6.68M words, all redistributable) into
    sentence-level JSONL in the same shape the tier-0 extractors emit, so that
    `sample_for_treebank.py` can draw a source-stratified gold sample from all of
    them instead of the four files it currently sees.

        data/raw/eo/gutenberg/*.txt          113 books      2.73M words   PD
        data/raw/eo/wikisource/*.jsonl       627 texts      3.18M words   PD+CC-BY-SA
        data/raw/eo/libera_folio/*.jsonl     833 articles   0.77M words   CC BY 4.0

    TWO THINGS THAT WILL SILENTLY WRECK THIS IF DONE NAIVELY
    -------------------------------------------------------
    1. GUTENBERG PLAIN TEXT IS HARD-WRAPPED at ~70 columns. A newline is NOT a
       sentence boundary — it is a typesetting artifact. Splitting on lines shreds
       every sentence in every book into fragments, and the fragments still LOOK
       like Esperanto, so the quality gate passes them and the damage is invisible.
       Paragraphs are delimited by BLANK lines; we join within a paragraph first.

    2. `.` IS NOT A SENTENCE BOUNDARY IN ESPERANTO EITHER. `k.t.p.`, `ktp.`, `s-ro`,
       `d-ro`, `t.e.`, `n-ro`, ordinals (`la 9-an`), and initials (`L. L. Zamenhof`)
       all carry periods. Naive splitting turns "L. L. Zamenhof" into three
       "sentences", two of which are a single capital letter. Those are exactly the
       fragments a treebank must not contain, because a human will dutifully
       annotate them.

    We therefore MEASURE the split: the report prints the share of output sentences
    that are suspiciously short or do not begin with a capital. A splitter bug shows
    up there as a number, instead of quietly becoming the corpus.

Pipeline Position:
    acquire_* -> [THIS] -> data/extracted/eo/free/*.jsonl -> sample_for_treebank.py

Usage:
    python scripts/extract/extract_free_corpora.py

Outputs:
    - data/extracted/eo/free/gutenberg_sentences.jsonl
    - data/extracted/eo/free/vikifontaro_sentences.jsonl
    - data/extracted/eo/free/libera_folio_sentences.jsonl
      each row: {sentence, source, source_title, author, kind, licence}

Quality Checks:
    - the Esperanto grammar gate (#823) on every sentence
    - splitter sanity: % of sentences that are <3 words or start lowercase

Last Updated: 2026-07-14
Related Issues: #820
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUT = Path('data/extracted/eo/free')

# ── sentence splitting ───────────────────────────────────────────────────────
# Abbreviations whose period does NOT end a sentence. Esperanto's are a small
# closed set (this is the language's whole point) plus the usual initials.
_ABBREV = {
    'ktp', 'k.t.p', 's-ro', 's-ino', 's-rino', 'd-ro', 'd-rino', 'prof', 'f-ino',
    't.e', 'n-ro', 'ekz', 'k.a', 'i.a', 'p.k', 'a.k', 'sro', 'dro', 'vd', 'kp',
    'inĝ', 'k', 'no', 'vol', 'p', 'pĝ', 'sr', 'jn',
}
_END = re.compile(r'([.!?…]+["»\')\]]?)\s+')
# A single capital letter + period is an INITIAL ("L. L. Zamenhof"), never a
# sentence end.
_INITIAL = re.compile(r'\b[A-ZĈĜĤĴŜŬ]\.$')
_ORDINAL = re.compile(r'\b\d+[-–]?[a-zĉĝĥĵŝŭ]*\.$')   # "la 9-an.", "1905."


def split_sentences(text: str) -> list[str]:
    """Split a PARAGRAPH (already un-wrapped) into sentences."""
    out, buf, pos = [], '', 0
    for m in _END.finditer(text):
        chunk = text[pos:m.end()]
        buf += chunk
        head = buf.rstrip()
        last = head.split()[-1] if head.split() else ''
        bare = last.rstrip('.!?…"»\')]').lower()
        # keep going if the "end" is really an abbreviation, an initial, or an
        # ordinal — none of which terminate a sentence
        if bare in _ABBREV or _INITIAL.search(head) or _ORDINAL.search(head):
            pos = m.end()
            continue
        out.append(head)
        buf = ''
        pos = m.end()
    tail = (buf + text[pos:]).strip()
    if tail:
        out.append(tail)
    return [s for s in (x.strip() for x in out) if s]


def paragraphs(text: str) -> list[str]:
    """Un-wrap hard-wrapped text. A BLANK line ends a paragraph; a single newline
    inside one is a typesetting artifact and must NOT be treated as a break."""
    out = []
    for block in re.split(r'\n\s*\n', text):
        p = re.sub(r'\s*\n\s*', ' ', block).strip()
        p = re.sub(r'\s{2,}', ' ', p)
        if p:
            out.append(p)
    return out


def sentences_of(text: str) -> list[str]:
    return [s for p in paragraphs(text) for s in split_sentences(p)]


def main() -> int:
    ap = argparse.ArgumentParser(description='Extract the redistributable corpora')
    ap.add_argument('--min-words', type=int, default=3)
    ap.add_argument('--max-words', type=int, default=80)
    args = ap.parse_args()

    from klareco.corpus_quality import assess

    OUT.mkdir(parents=True, exist_ok=True)
    report: list[tuple] = []

    def write(name: str, rows: list[dict]) -> None:
        kept = drop_gate = drop_len = 0
        short = lower = 0
        path = OUT / f'{name}_sentences.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                for s in sentences_of(r.pop('_text')):
                    n = len(s.split())
                    if n < args.min_words or n > args.max_words:
                        drop_len += 1
                        continue
                    v = assess(s)
                    if not v.keep:
                        drop_gate += 1
                        continue
                    s = v.text or s
                    kept += 1
                    # splitter sanity — a fragment still LOOKS like Esperanto, so
                    # the gate will not catch it. These two counters will.
                    if len(s.split()) <= 3:
                        short += 1
                    if s[:1].islower():
                        lower += 1
                    f.write(json.dumps({'sentence': s, **r},
                                       ensure_ascii=False) + '\n')
        report.append((name, kept, drop_gate, drop_len, short, lower, path))

    # ── Gutenberg: 113 books, hard-wrapped plain text ────────────────────
    g = []
    for meta in sorted(Path('data/raw/eo/gutenberg').glob('pg*.meta.json')):
        m = json.loads(meta.read_text())
        txt = meta.with_name(meta.name.replace('.meta.json', '.txt'))
        if not txt.exists():
            continue
        g.append({'_text': txt.read_text(encoding='utf-8'),
                  'source': 'gutenberg', 'source_title': m['title'],
                  'author': (m['authors'] or [None])[0], 'kind': m['kind'],
                  'licence': 'public domain'})
    write('gutenberg', g)

    # ── Vikifontaro ───────────────────────────────────────────────────────
    v = []
    p = Path('data/raw/eo/wikisource/vikifontaro.jsonl')
    if p.exists():
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            v.append({'_text': r['text'], 'source': 'vikifontaro',
                      'source_title': r['title'], 'author': None,
                      'kind': 'nekonata',
                      'licence': 'PD + CC BY-SA 4.0'})
    write('vikifontaro', v)

    # ── Libera Folio: all ORIGINAL Esperanto ─────────────────────────────
    lf = []
    p = Path('data/raw/eo/libera_folio/libera_folio.jsonl')
    if p.exists():
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            lf.append({'_text': r['text'], 'source': 'libera_folio',
                       'source_title': r['title'], 'author': None,
                       'kind': 'originala', 'licence': 'CC BY 4.0',
                       'url': r.get('url')})
    write('libera_folio', lf)

    print(f'  {"source":14s} {"sentences":>10s} {"drop:gate":>10s} {"drop:len":>9s}')
    tot = 0
    for name, kept, dg, dl, _s, _l, _p in report:
        tot += kept
        print(f'  {name:14s} {kept:10,} {dg:10,} {dl:9,}')
    print(f'  {"TOTAL":14s} {tot:10,}')

    print('\n  SPLITTER SANITY — a shredded sentence still LOOKS like Esperanto, so')
    print('  the quality gate cannot catch it. These two numbers can:')
    print(f'\n  {"source":14s} {"<=3 words":>10s} {"starts lower":>13s}')
    for name, kept, _dg, _dl, short, lower, _p in report:
        if not kept:
            continue
        print(f'  {name:14s} {short / kept:9.1%} {lower / kept:12.1%}')
    print('\n  Both should be LOW. A hard-wrap bug drives `starts lower` up (fragments')
    print('  begin mid-sentence); a bad abbreviation list drives `<=3 words` up')
    print('  ("L." / "L." / "Zamenhof ...").')

    print()
    for _n, _k, _dg, _dl, _s, _l, path in report:
        print(f'  wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
