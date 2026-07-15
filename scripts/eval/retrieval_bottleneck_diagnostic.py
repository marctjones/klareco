#!/usr/bin/env python3
"""
Recall or ranking? — where the retriever actually loses. (#713/#736)

VERSION: v1.0
COMPATIBLE WITH: v2.3+ store, whoosh_v2
DEPENDENCIES: whoosh; test sets carrying source_sentence_id
STAGE: Evaluation / diagnostic

Description:
    A RERANKER CAN ONLY REORDER WHAT BM25 ALREADY RETRIEVED. Before building or
    tuning another reranker, answer the one question that decides whether reranking
    can help at all: for each test question, is the gold sentence IN the candidate
    pool, and if so, WHERE?

    Every question falls into exactly one bucket:

        RECALL MISS   gold not in BM25 top-N        -> no reranker can ever fix it;
                                                        this is a FIRST-STAGE problem.
        ALREADY R1    gold already at BM25 rank 1   -> reranking can only HOLD, not
                                                        gain (and risks demoting it).
        RERANKABLE    gold in pool, rank > 1        -> the ONLY region where a
                                                        reranker can move the number.

    The headline number is the split. If most failures are RECALL MISSES, stop
    writing rerankers and fix first-stage retrieval (index the AST: a radiko stream
    to collapse inflection, AST-role fields with boosts). If most gold sits in the
    RERANKABLE band, first stage is fine and reranking is the whole game.

    This is the measurement we did NOT have when we spent a week on rerankers.

Pipeline Position:
    whoosh + test set --[THIS]--> recall@k curve + gold-rank profile + the split

Usage:
    python scripts/eval/retrieval_bottleneck_diagnostic.py \
        --test-sets data/test_sets/clause_discriminating_qa.jsonl \
                    data/test_sets/synthetic_who_rebuild_50.jsonl \
        --max-n 200

Inputs:
    - test sets (JSONL) with a `source_sentence_id` per question (the gold sid).
      Questions without one are reported as `no-gold-id` and skipped from recall.

Outputs (stdout):
    - per test set: recall@{1,5,10,20,50,100,max}, gold-rank histogram,
      and the RECALL-MISS / ALREADY-R1 / RERANKABLE split.
    - if `question_type` is present, the same split broken down by type.

Quality Checks:
    - reports how many questions had no gold id (so recall is not silently
      computed over a shrunken denominator).

Last Updated: 2026-07-15
Related Issues: #713, #736, #737
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from whoosh import index as whoosh_index
from whoosh.qparser import OrGroup, QueryParser

WHOOSH = 'data/indexes/whoosh_v2'
_BUCKETS = (1, 5, 10, 20, 50, 100)


def _gold_rank(searcher, qp, question: str, gold_sid: str, limit: int):
    """1-based rank of the gold sid in BM25 results, or None if not in top-`limit`."""
    hits = searcher.search(qp.parse(question), limit=limit)
    for i, h in enumerate(hits):
        if str(h['id']) == str(gold_sid):
            return i + 1
    return None


def _report(name: str, ranks: list, max_n: int, no_gold: int, by_type: dict):
    n = len(ranks)
    print(f'\n{"=" * 66}\n  {name}   ({n} scored, {no_gold} without gold id)\n{"=" * 66}')
    if not n:
        print('  (nothing to score)')
        return

    # recall@k
    print('  RECALL@k  (gold anywhere in BM25 top-k):')
    for k in (*_BUCKETS, max_n):
        hit = sum(1 for r in ranks if r is not None and r <= k)
        print(f'    @{k:<4d} {hit:4d}/{n}  {hit / n:6.1%}')

    # the split that decides where to spend effort
    miss = sum(1 for r in ranks if r is None)
    at1 = sum(1 for r in ranks if r == 1)
    rerankable = sum(1 for r in ranks if r is not None and r > 1)
    print('\n  WHERE THE LOSS IS  (the split that decides the roadmap):')
    print(f'    RECALL MISS  (gold not in top-{max_n})   {miss:4d}  {miss / n:6.1%}'
          f'   -> first-stage problem; no reranker helps')
    print(f'    ALREADY R1   (gold already rank 1)      {at1:4d}  {at1 / n:6.1%}'
          f'   -> reranking can only hold, not gain')
    print(f'    RERANKABLE   (gold in pool, rank > 1)   {rerankable:4d}  '
          f'{rerankable / n:6.1%}   -> the ONLY place reranking can move MRR')

    # gold-rank histogram over the rerankable band
    hist = collections.Counter()
    for r in ranks:
        if r is None:
            hist['miss'] += 1
        elif r == 1:
            hist['1'] += 1
        elif r <= 5:
            hist['2-5'] += 1
        elif r <= 20:
            hist['6-20'] += 1
        elif r <= 50:
            hist['21-50'] += 1
        else:
            hist['51+'] += 1
    print('\n  GOLD-RANK DISTRIBUTION:')
    for b in ('1', '2-5', '6-20', '21-50', '51+', 'miss'):
        c = hist[b]
        print(f'    {b:6s} {c:4d}  {"█" * c}')
    present = sorted(r for r in ranks if r is not None)
    if present:
        print(f'    median rank (of those retrieved): {present[len(present) // 2]}')

    if by_type:
        print('\n  BY QUESTION TYPE  (rerankable / already-R1 / miss):')
        for qt, rs in sorted(by_type.items()):
            m = sum(1 for r in rs if r is None)
            a1 = sum(1 for r in rs if r == 1)
            rr = sum(1 for r in rs if r is not None and r > 1)
            print(f'    {qt:14s} n={len(rs):<4d}  rerankable={rr:<4d} '
                  f'already-R1={a1:<4d} miss={m}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--test-sets', nargs='+', required=True)
    ap.add_argument('--whoosh-dir', default=WHOOSH)
    ap.add_argument('--max-n', type=int, default=200,
                    help='deepest pool to look for gold in (the recall ceiling)')
    args = ap.parse_args()

    ix = whoosh_index.open_dir(args.whoosh_dir)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    with ix.searcher() as srch:
        for ts in args.test_sets:
            ranks: list = []
            no_gold = 0
            by_type: dict = collections.defaultdict(list)
            with open(ts, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    q = json.loads(line)
                    question = q.get('eo_question') or q.get('question')
                    gold = (q.get('source_sentence_id')
                            or q.get('sid') or q.get('gold_sid'))
                    if not question or gold is None:
                        no_gold += 1
                        continue
                    r = _gold_rank(srch, qp, question, str(gold), args.max_n)
                    ranks.append(r)
                    by_type[q.get('question_type', '?')].append(r)
            _report(Path(ts).name, ranks, args.max_n, no_gold, dict(by_type))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
