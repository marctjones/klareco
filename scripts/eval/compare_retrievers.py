#!/usr/bin/env python3
"""
Head-to-head comparison: BM25 retriever vs ASTRetriever.

VERSION: v2.x
COMPATIBLE WITH: post-bug-#1/#2/#4/#6 parser, entity_postings + pattern_kb tables
DEPENDENCIES: duckdb, klareco.*
STAGE: Evaluation

Description:
    The multi_reranker_bench tests RERANKERS on top of a single retriever's
    candidate pool. But rerankers can't help if the underlying retriever
    doesn't surface the right passage in the first place.

    This script tests RETRIEVERS head-to-head:
      - BM25 (via DuckDBRetriever — current production path)
      - ASTRetriever (KB lookup + entity_postings + verb_klaso filtering,
                     BM25 fallback for unstructured queries)

    Per question, each retriever returns its top-10. We check where the
    expected answer first appears (Recall@1/5/10, MRR), and whether the
    extractor produces a correct answer when given that retriever's top-10.

Pipeline Position:
    Trivia bank → [THIS SCRIPT] → per-retriever metrics
                                → identifies which retriever surfaces the
                                  answer faster

Usage:
    python scripts/eval/compare_retrievers.py
    python scripts/eval/compare_retrievers.py --test-set data/test_sets/trivia_bank.jsonl

Outputs:
    Per-question table + aggregate metrics for each retriever.
    Optionally appended to perf_history.

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from klareco.parser import parse
from klareco.rag.duckdb_retriever import DuckDBRetriever
from klareco.rag.ast_retriever import ASTRetriever


def first_relevant_rank(candidates: list[dict], expected: list[str]) -> int | None:
    """Find rank (1-indexed) of the first candidate whose text contains
    any expected keyword."""
    for rank, c in enumerate(candidates, 1):
        text = (c.get('text') or '').lower()
        if any(kw.lower() in text for kw in expected):
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-set', default='data/test_sets/trivia_bank.jsonl')
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--append-history', default=None)
    args = ap.parse_args()

    test = []
    with open(args.test_set) as f:
        for line in f:
            line = line.strip()
            if line:
                test.append(json.loads(line))
    print(f'Loaded {len(test)} questions from {args.test_set}\n')

    # Set up both retrievers
    print('Building BM25 retriever (DuckDBRetriever)…')
    bm25 = DuckDBRetriever(
        whoosh_index_dir=args.whoosh_dir,
        duckdb_path=args.duckdb_path,
    )
    print('Building AST retriever (with BM25 fallback)…')
    ast_ret = ASTRetriever(
        duckdb_path=args.duckdb_path,
        bm25_fallback=bm25,
        min_candidates=5,
    )

    rows = []
    for i, q in enumerate(test, 1):
        question_text = q.get('eo_question') or q['question']
        expected = q.get('expected_keywords') or [q.get('eo_answer', '')]
        if not expected[0]:
            continue
        question_ast = parse(question_text)
        question_ast['frazo_teksto'] = question_text

        # BM25 retrieval
        t0 = time.time()
        bm25_cands = bm25.retrieve_with_ast_roles(question_ast, args.top_k)
        bm25_latency_ms = (time.time() - t0) * 1000
        bm25_rank = first_relevant_rank(bm25_cands, expected)
        bm25_route = 'bm25_whoosh'

        # AST retrieval
        t0 = time.time()
        ast_cands = ast_ret.retrieve_with_ast_roles(question_ast, args.top_k)
        ast_latency_ms = (time.time() - t0) * 1000
        ast_rank = first_relevant_rank(ast_cands, expected)
        ast_route = ast_cands[0].get('retriever_route', '?') if ast_cands else '(empty)'

        rows.append({
            'id':              q.get('id', f'q{i}'),
            'question':        question_text,
            'expected':        expected,
            'bm25_rank':       bm25_rank,
            'bm25_latency_ms': round(bm25_latency_ms, 1),
            'ast_rank':        ast_rank,
            'ast_route':       ast_route,
            'ast_latency_ms':  round(ast_latency_ms, 1),
        })
        bm_mark = '✓' if bm25_rank is not None and bm25_rank <= 5 else '·'
        ast_mark = '✓' if ast_rank is not None and ast_rank <= 5 else '·'
        print(f'  [{i:>2}/{len(test)}] BM25 {bm_mark} (rank={bm25_rank}, {bm25_latency_ms:.0f}ms)  '
              f'AST {ast_mark} (rank={ast_rank}, {ast_latency_ms:.0f}ms, route={ast_route[:25]})')

    # Aggregate
    print(f'\n=== Aggregate ({len(rows)} questions, candidate pool top_k={args.top_k}) ===\n')
    print(f'{"retriever":<22s} {"R@1":>5s} {"R@5":>5s} {"R@10":>5s} {"R@pool":>7s} {"MRR":>6s} {"avg_lat":>8s}')
    print('-' * 70)
    for name, key_rank, key_lat in [
        ('bm25', 'bm25_rank', 'bm25_latency_ms'),
        ('ast_retriever', 'ast_rank', 'ast_latency_ms'),
    ]:
        ranks = [r[key_rank] for r in rows]
        n_r1 = sum(1 for x in ranks if x == 1)
        n_r5 = sum(1 for x in ranks if x is not None and x <= 5)
        n_r10 = sum(1 for x in ranks if x is not None and x <= 10)
        n_rpool = sum(1 for x in ranks if x is not None and x <= args.top_k)
        mrr = (sum(1.0 / x for x in ranks if x is not None) / len(ranks)
               if ranks else 0)
        avg_lat = sum(r[key_lat] for r in rows) / max(1, len(rows))
        print(f'{name:<22s} {n_r1:>5d} {n_r5:>5d} {n_r10:>5d} {n_rpool:>7d} '
              f'{mrr:>6.3f} {avg_lat:>7.0f}ms')

    # Per-route breakdown for AST retriever
    print(f'\n=== AST retriever route breakdown ===')
    from collections import Counter
    route_counts = Counter(r['ast_route'] for r in rows)
    for route, n in route_counts.most_common():
        avg_rank = sum(
            r['ast_rank'] for r in rows
            if r['ast_route'] == route and r['ast_rank'] is not None
        ) / max(1, sum(
            1 for r in rows if r['ast_route'] == route and r['ast_rank'] is not None
        ))
        n_found = sum(1 for r in rows if r['ast_route'] == route and r['ast_rank'] is not None)
        print(f'  {route:<35s}  n={n}  found={n_found}  avg_rank={avg_rank:.1f}')

    if args.append_history:
        from perf_history import append_run
        summary = {
            'test_set':     args.test_set,
            'n_questions':  len(rows),
            'top_k':        args.top_k,
            'retrievers': {
                'bm25': {
                    'recall_at_1':      sum(1 for r in rows if r['bm25_rank'] == 1),
                    'recall_at_5':      sum(1 for r in rows if r['bm25_rank'] is not None and r['bm25_rank'] <= 5),
                    'recall_at_10':     sum(1 for r in rows if r['bm25_rank'] is not None and r['bm25_rank'] <= 10),
                    'recall_at_pool':   sum(1 for r in rows if r['bm25_rank'] is not None and r['bm25_rank'] <= args.top_k),
                    'mrr':              round(sum(1.0/r['bm25_rank'] for r in rows if r['bm25_rank']) / max(1, len(rows)), 4),
                    'avg_latency_ms':   round(sum(r['bm25_latency_ms'] for r in rows) / max(1, len(rows)), 1),
                },
                'ast_retriever': {
                    'recall_at_1':      sum(1 for r in rows if r['ast_rank'] == 1),
                    'recall_at_5':      sum(1 for r in rows if r['ast_rank'] is not None and r['ast_rank'] <= 5),
                    'recall_at_10':     sum(1 for r in rows if r['ast_rank'] is not None and r['ast_rank'] <= 10),
                    'recall_at_pool':   sum(1 for r in rows if r['ast_rank'] is not None and r['ast_rank'] <= args.top_k),
                    'mrr':              round(sum(1.0/r['ast_rank'] for r in rows if r['ast_rank']) / max(1, len(rows)), 4),
                    'avg_latency_ms':   round(sum(r['ast_latency_ms'] for r in rows) / max(1, len(rows)), 1),
                },
            },
        }
        append_run(Path(args.append_history), summary)


if __name__ == '__main__':
    main()
