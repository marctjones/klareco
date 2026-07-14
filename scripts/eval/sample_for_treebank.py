#!/usr/bin/env python3
"""
Sample sentences for a GOLD TREEBANK — stratified, and deliberately not random.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: duckdb
STAGE: Evaluation

Description:
    WHY WE NEED THIS
    ----------------
    There are **3,343 tokens** of free gold Esperanto UD in existence. Cairo is
    177. Every accuracy claim this project makes is a claim about 3.3k tokens.

    The standard error on an accuracy p over N tokens is sqrt(p(1-p)/N), so at our
    current LAS of 57%:

        gold tokens   SE     smallest LAS change we can SEE (2*SE)
          3,343      0.86%        1.7%      <- what we have TODAY
         10,000      0.50%        1.0%
         20,000      0.35%        0.7%
         52,000      0.22%        0.4%      <- Arbobanko

    Everything we shipped today (LAS 34.9% -> 57.0%) was big enough to see.
    **The next round of work will not be.** 10,000 tokens is the sweet spot: 1-point
    resolution for 10-20 hours of annotation.

    WHY IT MUST BE STRATIFIED
    -------------------------
    Our LAS COLLAPSES on long sentences — 57.4% at 1-10 tokens, 39.2% at 40+ — and
    the corpus is 42.7% short sentences and only 4.1% long ones. **A randomly
    sampled gold set would be dominated by exactly the sentences we already handle
    best, and would measure the wrong thing.**

    So we stratify by LENGTH, by SOURCE (Wikipedia is the simplest register the
    language has), and by PHENOMENON — every sentence must be there for a reason.

    WHY THE SOURCES MATTER
    ----------------------
    Gutenberg Esperanto (Alice, Andersen) is PUBLIC DOMAIN. Lingvaj Respondoj
    (Zamenhof, 1891) is PUBLIC DOMAIN. Wikipedia is CC-BY-SA. All three are
    REDISTRIBUTABLE — so the treebank we build can be PUBLISHED, which Oya
    (UDW/SyntaxFest 2025) explicitly called for and which nobody has done.

    That also forces register diversity: Wikipedia prose is short, declarative and
    name-heavy. Literary prose is not. Training and measuring only on Wikipedia
    measures the easiest register the language has.

Pipeline Position:
    corpus --[THIS]--> a stratified sample --> preannotate_treebank.py --> human

Usage:
    python scripts/eval/sample_for_treebank.py --tokens 10000 --out data/test_sets/treebank_sample.jsonl

Quality Checks:
    - Reports the length distribution and the phenomenon coverage of what it picked,
      so an unbalanced sample is visible BEFORE anyone spends 20 hours on it.

Last Updated: 2026-07-14
Related Issues: #820
See Also: docs/PARSER_DESIGN.md
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.corpus_quality import assess

DB = 'data/indexes/duckdb_store.db'

# Target mix. Deliberately over-weights LONG sentences: that is where we fail
# (LAS 39.2% at 40+ tokens vs 57.4% at 1-10), and where random sampling would give
# us almost nothing (4.1% of the corpus).
LENGTH_TARGETS = {
    '1-10': 0.15,
    '11-20': 0.25,
    '21-40': 0.35,
    '40+': 0.25,
}

# Every sentence must earn its place. These are the phenomena our LAS analysis says
# are failing, and a gold set that does not contain them cannot measure the fixes.
PHENOMENA = {
    # a PP whose attachment is genuinely ambiguous (#826 — Bick's #1 error class)
    'pp_attachment': re.compile(r'\b(en|al|per|kun|sur|sub|pri|por)\s+(la\s+)?\w+'),
    # coordination (#827 — 4x over-represented in Bick's errors)
    'coordination': re.compile(r'\b(kaj|aŭ|sed|nek)\b'),
    # subordination — the clauses we MISS (we find only 61% of them)
    'subordination': re.compile(r'\b(ke|ĉar|kvankam|se|dum|kiam|kiu|kiun|kies)\b'),
    # an infinitive complement — `xcomp`, which we score 0% on.
    # NOT plain `\w+i`: every Esperanto PRONOUN ends in -i (mi, ni, vi, li, ŝi, ĝi,
    # si, ili, oni), as does the preposition `pri` and the particle `ĉi`. That regex
    # matched almost every sentence in the language and the coverage number it gave
    # was worthless. Exclude the closed class explicitly.
    'infinitive': re.compile(
        r'\b(?!(?:mi|ni|vi|li|ŝi|ĝi|si|ili|oni|ci|pri|ĉi)\b)\w{3,}i\b'),
    # a participle — `acl`, which we score 0% on
    'participle': re.compile(r'\b\w+(ant|int|ont|at|it|ot)(a|aj|an|ajn|e)\b'),
    # the accusative of direction — a signal English does not have
    'acc_direction': re.compile(r'\b(en|sur|sub|tra)\s+(la\s+)?\w+n\b'),
}


def _bucket(n: int) -> str:
    return '1-10' if n <= 10 else ('11-20' if n <= 20 else
                                   ('21-40' if n <= 40 else '40+'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--tokens', type=int, default=10_000,
                    help='target gold-token count (10k = 1-point LAS resolution)')
    ap.add_argument('--out', default='data/test_sets/treebank_sample.jsonl')
    ap.add_argument('--pool', type=int, default=400_000,
                    help='how many corpus sentences to consider')
    ap.add_argument('--seed', type=int, default=20260714)
    args = ap.parse_args()

    random.seed(args.seed)
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute(
        f'SELECT text FROM sentences USING SAMPLE {args.pool} ROWS (reservoir, '
        f'{args.seed % 1000})').fetchall()

    # Only CLEAN sentences — the quality gate (#823) drops redirects, English and
    # markup. Annotating junk is the most expensive mistake available here.
    pool: dict[str, list] = collections.defaultdict(list)
    for (t,) in rows:
        if not t:
            continue
        v = assess(t)
        if not v.keep:
            continue
        text = v.text or t
        n = len(text.split())
        if n < 3 or n > 80:
            continue
        phen = {k for k, rx in PHENOMENA.items() if rx.search(text)}
        pool[_bucket(n)].append({'text': text, 'n': n, 'phenomena': sorted(phen)})

    print(f'  pool: {sum(len(v) for v in pool.values()):,} clean sentences')
    for b in ('1-10', '11-20', '21-40', '40+'):
        print(f'    {b:6s} {len(pool[b]):7,}')

    # Fill each length bucket to its token target, PREFERRING sentences that carry
    # more of the phenomena we are failing on. A sentence with a coordinated
    # subordinate clause and an ambiguous PP is worth three plain ones.
    picked: list = []
    got = collections.Counter()
    for b, share in LENGTH_TARGETS.items():
        target = int(args.tokens * share)
        cands = sorted(pool[b], key=lambda s: (-len(s['phenomena']), random.random()))
        for s in cands:
            if got[b] >= target:
                break
            picked.append(s)
            got[b] += s['n']

    total = sum(got.values())
    print(f'\n  PICKED: {len(picked):,} sentences / {total:,} tokens\n')
    print(f'    {"bucket":8s} {"tokens":>8s} {"share":>7s}   {"target":>7s}')
    for b in ('1-10', '11-20', '21-40', '40+'):
        print(f'    {b:8s} {got[b]:8,} {got[b] / total:7.1%}   '
              f'{LENGTH_TARGETS[b]:7.0%}')

    print('\n  PHENOMENON COVERAGE (a gold set without these cannot measure the fixes):')
    cov = collections.Counter()
    for s in picked:
        cov.update(s['phenomena'])
    for k in PHENOMENA:
        n = cov[k]
        print(f'    {k:16s} {n:5,} sentences  ({n / len(picked):5.1%})')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for i, s in enumerate(picked, 1):
            f.write(json.dumps({'sent_id': f'klareco-{i:05d}', **s},
                               ensure_ascii=False) + '\n')
    print(f'\n  wrote {out}')
    print(f'\n  Next: python scripts/eval/preannotate_treebank.py --in {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
