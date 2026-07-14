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

    ⚠️ AND TOKENS ARE NOT INDEPENDENT SAMPLES. The obvious sizing calculation —
    SE = sqrt(p(1-p)/N) over N tokens — is WRONG for LAS, and I shipped it before
    checking. **Attachment errors CLUSTER BY SENTENCE**: one garbled 40-token
    sentence gets a dozen arcs wrong together, a clean short one gets them all
    right. The tokens within a sentence are nothing like independent draws.

    MEASURED on Prago (scripts/eval/eval_conllu.py, per-sentence bootstrap vs
    binomial, 5,000 resamples):

        SE, binomial over 2,426 TOKENS    1.013%     <- the naive figure
        SE, bootstrap over 130 SENTENCES  1.866%     <- the truth
        DESIGN EFFECT                     3.4x       (SE inflated 1.84x)

    So the real resolution, and the lever is SENTENCE COUNT, not token count:

        gold tokens   sents    2*SE naive    2*SE REAL
          3,343         179       1.7%          3.2%    <- what we have TODAY
         10,000         536       1.0%          1.8%
         20,000       1,072       0.7%          1.3%
         52,000       2,786       0.4%          0.8%    <- Arbobanko

    **To actually SEE a 1-point LAS move takes ~33,000 tokens, not 10,000.** 10k
    still roughly halves our error bar (3.2% -> 1.8%) and is worth ~10-20 hours;
    20k is the honest target. When scoring against this set, take the CI from a
    SENTENCE-LEVEL BOOTSTRAP — never the binomial.

    (This matters more than it looks. Stratifying toward long, hard sentences —
    which we do deliberately, below — RAISES the intra-sentence correlation and so
    makes the design effect worse, not better. The stratification is still right;
    it just has to be paid for in sentences.)

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

# ─────────────────────────────────────────────────────────────────────────────
# WE SAMPLE FROM THE EXTRACTED JSONL, NOT FROM THE STORE.
#
# The store has NO provenance column (#803). Sampling from it means sampling from
# a pile in which Wikipedia outweighs everything else about 150:1 (1.6 GB vs
# ~10 MB) — so a "random" draw is ~all Wikipedia, which is the SIMPLEST REGISTER
# THE LANGUAGE HAS: short, declarative, name-heavy, few subordinate clauses.
#
# A gold set drawn that way would flatter the parser, and it would do so in
# exactly the way the length-stratification below exists to prevent. Register is
# the same trap one level up.
#
# The extracted files still know where they came from. So we read them.
# ─────────────────────────────────────────────────────────────────────────────
SOURCES = {
    # path                                                     quota  licence
    'data/extracted/wikipedia_sentences.jsonl':               (0.28, 'CC-BY-SA'),
    'data/extracted/eo/free/gutenberg_sentences.jsonl':       (0.24, 'public domain'),
    'data/extracted/eo/free/vikifontaro_sentences.jsonl':     (0.24, 'PD + CC BY-SA'),
    # ALL of Libera Folio is ORIGINAL Esperanto — the register we are starving for.
    # The Gutenberg shelf is 86 translations to 18 originals, and a translation
    # carries the SOURCE language's syntax. Weighted above its size for that reason.
    'data/extracted/eo/free/libera_folio_sentences.jsonl':    (0.16, 'CC BY 4.0'),
    'data/extracted/eo/tier0/grammar/lingvaj_respondoj_sentences.jsonl':
                                                              (0.08, 'public domain'),
}
# NOTE the old tier0 literary files (alice, andersen, krestomatio) are GONE from
# this table: all three are Gutenberg books and are now inside
# gutenberg_sentences.jsonl. Listing both would have double-counted them and
# quietly inflated their weight.
# PMEG is EXCLUDED by default. It is excellent register (technical grammatical
# prose) but it is Bertilo Wennergren's copyrighted work and its redistribution
# terms are not clear to us. The whole point of choosing these sources is that the
# resulting treebank can be PUBLISHED; quietly including a source we may not be
# able to redistribute would forfeit that. --include-pmeg if you have checked.
PMEG = 'data/extracted/eo/tier0/grammar/pmeg_sentences.jsonl'

# Deliberately over-weights LONG sentences: that is where we fail (LAS 39.2% at
# 40+ tokens vs 57.4% at 1-10), and where random sampling would give us almost
# nothing (4.1% of the corpus).
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
    ap.add_argument('--per-source', type=int, default=60_000,
                    help='how many lines to read from each source file')
    ap.add_argument('--include-pmeg', action='store_true',
                    help='include PMEG — copyrighted, redistribution UNCLEAR')
    ap.add_argument('--seed', type=int, default=20260714)
    args = ap.parse_args()

    random.seed(args.seed)

    sources = dict(SOURCES)
    if args.include_pmeg:
        print('  ⚠️  PMEG included. It is copyrighted and we have NOT verified that')
        print('      we may redistribute it. The treebank may not be publishable.\n')
        sources[PMEG] = (0.15, 'UNCLEAR — copyrighted')

    # Vikifontaro and Gutenberg BOTH host Zamenhof, Andersen and the Krestomatio.
    # The same sentence can therefore arrive twice from two "independent" sources,
    # and a gold set with duplicate sentences is a gold set that quietly weights
    # some constructions twice while looking balanced.
    seen: set[str] = set()
    n_dupe = 0

    # pool[source][bucket] -> candidate sentences
    pool: dict[str, dict[str, list]] = {}
    print('  SOURCES (register matters: Wikipedia is the simplest prose in the language)\n')
    print(f'    {"source":26s} {"read":>8s} {"clean":>8s}  {"licence":14s}')
    for path, (quota, lic) in sources.items():
        p = Path(path)
        if not p.exists():
            print(f'    {p.stem[:26]:26s} {"MISSING":>8s}          {lic:14s}')
            continue
        by_bucket: dict[str, list] = collections.defaultdict(list)
        n_read = n_clean = 0
        with open(p, encoding='utf-8') as f:
            for line in f:
                if n_read >= args.per_source:
                    break
                n_read += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # The Wikipedia extractor writes `text`; the tier-0 literary and
                # grammar extractors write `sentence`. Reading only `text` silently
                # dropped EVERY literary sentence — 100% of exactly the register
                # this source-stratification exists to include, and it looked like
                # the quality gate rejecting them.
                t = r.get('text') or r.get('sentence')
                if not t:
                    continue
                # Only CLEAN sentences — the quality gate (#823) drops redirects,
                # English and markup. Annotating junk is the most expensive mistake
                # available here.
                v = assess(t)
                if not v.keep:
                    continue
                text = v.text or t
                n = len(text.split())
                if n < 3 or n > 80:
                    continue
                key = re.sub(r'\W+', '', text.lower())
                if key in seen:
                    n_dupe += 1
                    continue
                seen.add(key)
                n_clean += 1
                phen = {k for k, rx in PHENOMENA.items() if rx.search(text)}
                by_bucket[_bucket(n)].append(
                    {'text': text, 'n': n, 'phenomena': sorted(phen),
                     'source': p.stem.replace('_sentences', ''),
                     'kind': (r.get('kind') or 'nekonata'), 'licence': lic})
        pool[path] = by_bucket
        print(f'    {p.stem.replace("_sentences","")[:26]:26s} {n_read:8,} {n_clean:8,}'
              f'  {lic:14s}')

    # Fill each (source x length) cell to its token target, PREFERRING sentences
    # that carry more of the phenomena we are failing on. A sentence with a
    # coordinated subordinate clause and an ambiguous PP is worth three plain ones.
    picked: list = []
    got: collections.Counter = collections.Counter()
    by_src: collections.Counter = collections.Counter()
    for path, (quota, _lic) in sources.items():
        if path not in pool:
            continue
        for b, share in LENGTH_TARGETS.items():
            target = int(args.tokens * quota * share)
            cands = sorted(pool[path][b],
                           key=lambda s: (-len(s['phenomena']), random.random()))
            cell = 0
            for s in cands:
                if cell >= target:
                    break
                picked.append(s)
                cell += s['n']
                got[b] += s['n']
                by_src[s['source']] += s['n']

    total = sum(got.values())
    if not total:
        print('\n  ✗ nothing sampled — are the extracted files present?')
        return 1
    print(f'\n  PICKED: {len(picked):,} sentences / {total:,} tokens\n')
    print(f'    {"bucket":8s} {"tokens":>8s} {"share":>7s}   {"target":>7s}')
    for b in ('1-10', '11-20', '21-40', '40+'):
        print(f'    {b:8s} {got[b]:8,} {got[b] / total:7.1%}   '
              f'{LENGTH_TARGETS[b]:7.0%}')

    print(f'\n    {"source":26s} {"tokens":>8s} {"share":>7s}')
    for s, n in by_src.most_common():
        print(f'    {s[:26]:26s} {n:8,} {n / total:7.1%}')
    if n_dupe:
        print(f'\n    de-duplicated across sources: {n_dupe:,} sentences')
        print('      (Vikifontaro and Gutenberg both host Zamenhof — the same')
        print('       sentence arriving twice would weight it twice)')

    # THE NUMBER THAT DECIDES WHAT THIS TREEBANK MEASURES. A translation carries the
    # SOURCE language's syntax; a gold set that is mostly translated prose measures
    # Esperanto-as-relexified-European, which is the assumption this project exists
    # NOT to make.
    by_kind: collections.Counter = collections.Counter()
    for s in picked:
        by_kind[s.get('kind', 'nekonata')] += s['n']
    print(f'\n    {"register":26s} {"tokens":>8s} {"share":>7s}')
    for k in ('originala', 'tradukita', 'nekonata'):
        print(f'    {k:26s} {by_kind[k]:8,} {by_kind[k] / total:7.1%}')
    print('      originala = written IN Esperanto. tradukita = translated INTO it,')
    print('      and therefore carrying another language\'s clause structure.')

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
