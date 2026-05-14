#!/usr/bin/env python3
"""
Evaluate Klareco Extractive QA on a Test Set.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu DB, orchestrator pipeline (klareco.orchestrator)
DEPENDENCIES: Whoosh index, Kuzu DB
STAGE: Evaluation

Reports three metric families:
  1. Answer keyword match — does the generated answer contain expected keywords?
  2. Retrieval ranking — at what rank does a passage containing expected
     keywords first appear in the top-k?
  3. Per-stage latency (avg / p50 / p95 / max / share%) — surfaces which
     orchestrator stage is the bottleneck. Same data is captured per-question
     under `stage_timings_ms` for downstream analysis.

Usage:
    python scripts/eval/evaluate_extractive_qa.py
    python scripts/eval/evaluate_extractive_qa.py --limit 10
    python scripts/eval/evaluate_extractive_qa.py --test-set data/test_sets/qa_test_diverse_30.jsonl
    python scripts/eval/evaluate_extractive_qa.py --top-k 20
    python scripts/eval/evaluate_extractive_qa.py --output results/qa_eval_$(date +%Y%m%d).json

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
from pathlib import Path

# Quiet retriever logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('klareco').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.orchestrator import build_default_pipeline
from klareco.eval import evaluate_question, summarize, print_summary


# Per-worker pipeline holder for multiprocessing mode. Each worker process
# builds its own pipeline once on init; subsequent tasks reuse it. The
# pipeline isn't picklable (Whoosh/Kuzu file handles), hence the initializer.
_WORKER_PIPELINE = None


def _init_worker(whoosh_dir: str, kuzu_path: str, top_k: int) -> None:
    global _WORKER_PIPELINE
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    logging.getLogger('klareco').setLevel(logging.ERROR)
    _WORKER_PIPELINE = build_default_pipeline(
        whoosh_index_dir=whoosh_dir,
        kuzu_db_path=kuzu_path,
        top_k=top_k,
    )


def _eval_in_worker(entry: dict) -> dict:
    return evaluate_question(_WORKER_PIPELINE, entry)


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
    parser.add_argument('--workers', type=int, default=1,
                        help='Run questions in parallel across N worker processes '
                             '(default 1 = serial). Each worker opens its own '
                             'Whoosh + Kuzu connection.')
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
    n_workers = max(1, args.workers)

    import time as _time
    wall_start = _time.perf_counter()

    if n_workers == 1:
        print(f"Building pipeline (Whoosh={args.whoosh_dir}, "
              f"Kuzu={args.kuzu_path}, top_k={args.top_k})...")
        pipeline = build_default_pipeline(
            whoosh_index_dir=args.whoosh_dir,
            kuzu_db_path=args.kuzu_path,
            top_k=args.top_k,
        )
        print(f"\nEvaluating {len(entries)} questions (serial)...")
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
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f"\nEvaluating {len(entries)} questions across {n_workers} workers "
              f"(each opens its own Whoosh+Kuzu connection)...")
        print('-' * 70)
        results = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(args.whoosh_dir, args.kuzu_path, args.top_k),
        ) as pool:
            for i, r in enumerate(pool.map(_eval_in_worker, entries), 1):
                results.append(r)
                mark = 'OK' if r.get('answer_correct') else '--'
                rk = r.get('first_relevant_rank')
                rk_str = f"rk={rk:>2d}" if rk else 'rk= -'
                print(f"  [{i:>3}/{len(entries)}] {mark} {rk_str}  "
                      f"{r.get('question','')[:55]:55s}  "
                      f"({r.get('latency_sec', 0):.1f}s)")

    wall = _time.perf_counter() - wall_start
    print(f"\nWall-clock total: {wall:.1f}s for {len(entries)} questions "
          f"({n_workers} worker{'s' if n_workers != 1 else ''})")

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
