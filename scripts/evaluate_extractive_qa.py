#!/usr/bin/env python3
"""
Evaluate Klareco Extractive QA on a Test Set.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu DB, orchestrator pipeline (klareco.orchestrator)
DEPENDENCIES: Whoosh index, Kuzu DB
STAGE: Evaluation

Reports two metrics:
  1. Answer keyword match — does the generated answer contain expected keywords?
  2. Retrieval ranking — at what rank does a passage containing expected
     keywords first appear in the top-k?

Usage:
    python scripts/evaluate_extractive_qa.py
    python scripts/evaluate_extractive_qa.py --limit 10
    python scripts/evaluate_extractive_qa.py --test-set data/test_sets/qa_test_diverse_30.jsonl
    python scripts/evaluate_extractive_qa.py --top-k 20
    python scripts/evaluate_extractive_qa.py --output results/qa_eval_$(date +%Y%m%d).json

Inputs:
    Test set JSONL with one entry per line:
      {"id": int, "question": str, "expected_keywords": [str], "question_type": str}

Outputs:
    Per-question JSON with answer, retrieval ranks, latency.
    Aggregate summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Quiet retriever logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('klareco').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klareco.orchestrator import build_default_pipeline


def keywords_in_text(keywords: list[str], text: str) -> list[str]:
    """Return keywords that appear (case-insensitive substring) in text."""
    if not text:
        return []
    text_lc = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lc]


def first_relevant_rank(keywords: list[str], passages) -> int | None:
    """1-based rank of the first passage containing any expected keyword."""
    for i, p in enumerate(passages, start=1):
        if keywords_in_text(keywords, p.text):
            return i
    return None


def evaluate_question(pipeline, entry: dict) -> dict:
    """Run one test question and return per-question metrics."""
    question = entry['question']
    expected_kw = entry.get('expected_keywords', [])

    t0 = time.time()
    result = pipeline.answer(question)
    elapsed = time.time() - t0

    # Find the retrieve stage trace to inspect ranked passages
    passages = ()
    for tr in result.trace:
        if tr.stage_name == 'retrieve' and tr.delta is not None:
            passages = tr.ctx_after.symbolic.passage_asts
            break

    rank = first_relevant_rank(expected_kw, passages)
    matched = keywords_in_text(expected_kw, result.text or '')

    return {
        'id':                  entry.get('id'),
        'question':            question,
        'question_type':       entry.get('question_type'),
        'expected_keywords':   expected_kw,
        'answer':              (result.text or '').strip(),
        'matched_keywords':    matched,
        'answer_correct':      bool(matched),
        'retrieved_count':     len(passages),
        'first_relevant_rank': rank,
        'retrieval_recall@k':  rank is not None,
        'mrr':                 (1.0 / rank) if rank else 0.0,
        'latency_sec':         round(elapsed, 2),
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    n_correct = sum(1 for r in results if r['answer_correct'])
    n_recall = sum(1 for r in results if r['retrieval_recall@k'])
    mrr = sum(r['mrr'] for r in results) / n
    avg_latency = sum(r['latency_sec'] for r in results) / n

    rank_buckets = {'1': 0, '2-3': 0, '4-10': 0, '11+': 0, 'none': 0}
    for r in results:
        rk = r['first_relevant_rank']
        if rk is None:
            rank_buckets['none'] += 1
        elif rk == 1:
            rank_buckets['1'] += 1
        elif rk <= 3:
            rank_buckets['2-3'] += 1
        elif rk <= 10:
            rank_buckets['4-10'] += 1
        else:
            rank_buckets['11+'] += 1

    return {
        'n':                  n,
        'answer_accuracy':    n_correct / n,
        'retrieval_recall':   n_recall / n,
        'mrr':                mrr,
        'avg_latency_sec':    round(avg_latency, 2),
        'rank_distribution':  rank_buckets,
    }


def print_summary(summary: dict, breakdown_by_type: dict):
    print('\n' + '=' * 70)
    print('AGGREGATE RESULTS')
    print('=' * 70)
    print(f"  Questions evaluated:   {summary['n']}")
    print(f"  Answer accuracy:       {summary['answer_accuracy']:.1%}  "
          f"(answer text contains >=1 expected keyword)")
    print(f"  Retrieval recall@k:    {summary['retrieval_recall']:.1%}  "
          f"(>=1 retrieved passage contains expected keyword)")
    print(f"  Mean Reciprocal Rank:  {summary['mrr']:.3f}")
    print(f"  Avg latency / query:   {summary['avg_latency_sec']:.1f}s")
    print()
    print('  First-relevant-rank distribution:')
    rd = summary['rank_distribution']
    print(f"    rank 1     {rd['1']:>4d}   (best -- top result is relevant)")
    print(f"    rank 2-3   {rd['2-3']:>4d}")
    print(f"    rank 4-10  {rd['4-10']:>4d}")
    print(f"    rank 11+   {rd['11+']:>4d}")
    print(f"    none       {rd['none']:>4d}   (no retrieved passage contains any expected keyword)")

    if breakdown_by_type:
        print()
        print('  By question type:')
        for qt, sub in sorted(breakdown_by_type.items()):
            if not sub:
                continue
            print(f"    {qt:8s}  n={sub['n']:>3d}  "
                  f"answer={sub['answer_accuracy']:.0%}  "
                  f"recall={sub['retrieval_recall']:.0%}  "
                  f"mrr={sub['mrr']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test-set', default='data/test_sets/qa_test_set_50.jsonl',
                        help='Path to JSONL test set (default: %(default)s)')
    parser.add_argument('--whoosh-dir', default='data/indexes/whoosh_fts',
                        help='Whoosh index directory')
    parser.add_argument('--kuzu-path', default='data/indexes/v2.1_kuzu_index_full',
                        help='Kuzu DB path')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Passages to retrieve per question (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate only the first N questions')
    parser.add_argument('--output', default=None,
                        help='Optional JSON file to write per-question results')
    args = parser.parse_args()

    # Load test set
    test_path = Path(args.test_set)
    if not test_path.exists():
        print(f"ERROR: test set not found at {test_path}", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if args.limit:
        entries = entries[:args.limit]

    print(f"Loaded {len(entries)} questions from {test_path}")
    print(f"Building pipeline (Whoosh={args.whoosh_dir}, Kuzu={args.kuzu_path}, top_k={args.top_k})...")

    pipeline = build_default_pipeline(
        whoosh_index_dir=args.whoosh_dir,
        kuzu_db_path=args.kuzu_path,
        top_k=args.top_k,
    )

    print(f"\nEvaluating {len(entries)} questions...")
    print('-' * 70)

    results = []
    for i, entry in enumerate(entries, 1):
        try:
            r = evaluate_question(pipeline, entry)
            results.append(r)
            mark = 'OK' if r['answer_correct'] else '--'
            rk = r['first_relevant_rank']
            rk_str = f"rk={rk:>2d}" if rk else 'rk= -'
            print(f"  [{i:>2}/{len(entries)}] {mark} {rk_str}  "
                  f"{entry['question'][:55]:55s}  ({r['latency_sec']:.1f}s)")
        except Exception as e:
            print(f"  [{i:>2}/{len(entries)}] ERROR: {e}")
            results.append({
                'id':                  entry.get('id'),
                'question':            entry['question'],
                'error':               str(e),
                'answer_correct':      False,
                'retrieval_recall@k':  False,
                'mrr':                 0.0,
                'first_relevant_rank': None,
                'latency_sec':         0,
                'expected_keywords':   entry.get('expected_keywords', []),
                'question_type':       entry.get('question_type'),
            })

    summary = summarize(results)

    by_type = {}
    for r in results:
        qt = r.get('question_type') or 'OTHER'
        by_type.setdefault(qt, []).append(r)
    breakdown = {qt: summarize(rs) for qt, rs in by_type.items()}

    print_summary(summary, breakdown)

    if args.output:
        out = {'summary': summary, 'by_type': breakdown, 'results': results}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == '__main__':
    main()
