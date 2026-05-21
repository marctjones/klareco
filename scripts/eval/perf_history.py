#!/usr/bin/env python3
"""
Performance history: append-only log of bench runs for over-time tracking.

VERSION: v2.x
COMPATIBLE WITH: multi_reranker_bench.py output
DEPENDENCIES: None beyond stdlib
STAGE: Evaluation / Metrics

Description:
    Bench runs are useful only if we can see whether they're improving.
    This module:
      1. Writes one record per bench run to data/perf/bench_history.jsonl
         capturing git commit, timestamp, test set, active assets,
         per-reranker and per-retriever metrics.
      2. Provides a `compare` mode that diffs two recent runs side-by-side.
      3. Can plot trends from the command line (text-mode sparklines).

    Each record is self-describing — schema can evolve forward-only by
    adding fields.

Usage:
    # Append a run (typically called from the bench script after it computes
    # metrics):
    python scripts/eval/perf_history.py append --run-summary /path/to/run.json

    # Compare the two most recent runs:
    python scripts/eval/perf_history.py compare

    # Show all runs as a table:
    python scripts/eval/perf_history.py list

    # Show a per-reranker trend over time:
    python scripts/eval/perf_history.py trend --metric mrr

Inputs:
    --history-path  data/perf/bench_history.jsonl (default)

Outputs:
    Appends to history file; prints reports to stdout.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


_HISTORY = Path('data/perf/bench_history.jsonl')


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent.parent,
            text=True,
        ).strip()
        return out
    except Exception:
        return 'unknown'


def get_git_branch() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=Path(__file__).resolve().parent.parent.parent,
            text=True,
        ).strip()
        return out
    except Exception:
        return 'unknown'


def append_run(history_path: Path, run_summary: dict) -> None:
    """Append a run record to the history file."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%S'),
        'git_commit':    get_git_commit(),
        'git_branch':    get_git_branch(),
        **run_summary,
    }
    with open(history_path, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Appended run to {history_path}: commit {record["git_commit"]}, '
          f'{len(record.get("rerankers", {}))} rerankers, '
          f'test_set={record.get("test_set", "?")}')


def load_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    runs = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    return runs


def cmd_list(args) -> None:
    runs = load_history(Path(args.history_path))
    print(f'\n=== {len(runs)} bench run(s) on file ===\n')
    print(f'{"#":<4s} {"timestamp":<20s} {"commit":<10s} {"test_set":<40s} {"rerankers":<10s}')
    print('-' * 90)
    for i, r in enumerate(runs):
        rerankers = ', '.join(r.get('rerankers', {}).keys())
        print(f'{i:<4d} {r.get("timestamp", "?"):<20s} '
              f'{r.get("git_commit", "?"):<10s} '
              f'{r.get("test_set", "?")[:38]:<40s} '
              f'{len(r.get("rerankers", {})):<10d}')


def cmd_compare(args) -> None:
    """Compare the two most recent runs (or specific indices)."""
    runs = load_history(Path(args.history_path))
    if len(runs) < 2:
        print(f'Need at least 2 runs to compare; have {len(runs)}.')
        return
    idx_a = args.a if args.a is not None else len(runs) - 2
    idx_b = args.b if args.b is not None else len(runs) - 1
    a, b = runs[idx_a], runs[idx_b]

    print(f'\nA: run #{idx_a}  commit={a.get("git_commit", "?")}  '
          f'time={a.get("timestamp", "?")}')
    print(f'B: run #{idx_b}  commit={b.get("git_commit", "?")}  '
          f'time={b.get("timestamp", "?")}')
    print()

    a_rerankers = a.get('rerankers', {})
    b_rerankers = b.get('rerankers', {})
    all_names = sorted(set(a_rerankers) | set(b_rerankers))

    print(f'{"reranker":<25s} '
          f'{"A.R@1":>6s} {"B.R@1":>6s} {"Δ":>5s} '
          f'{"A.MRR":>7s} {"B.MRR":>7s} {"Δ":>6s} '
          f'{"A.ans%":>7s} {"B.ans%":>7s} {"Δ":>6s}')
    print('-' * 100)
    for name in all_names:
        ra = a_rerankers.get(name, {})
        rb = b_rerankers.get(name, {})
        a_r1 = ra.get('recall_at_1', 0)
        b_r1 = rb.get('recall_at_1', 0)
        a_mrr = ra.get('mrr', 0)
        b_mrr = rb.get('mrr', 0)
        a_acc = ra.get('answer_accuracy', 0)
        b_acc = rb.get('answer_accuracy', 0)
        print(
            f'{name:<25s} '
            f'{a_r1:>6d} {b_r1:>6d} {b_r1 - a_r1:>+5d} '
            f'{a_mrr:>7.3f} {b_mrr:>7.3f} {b_mrr - a_mrr:>+6.3f} '
            f'{a_acc:>6.1f}% {b_acc:>6.1f}% {b_acc - a_acc:>+5.1f}%'
        )


def cmd_trend(args) -> None:
    """Show a per-reranker trend on the chosen metric."""
    runs = load_history(Path(args.history_path))
    if not runs:
        print('No runs on file.')
        return
    metric = args.metric
    # Gather per-run per-reranker values
    rerankers = sorted({
        name
        for r in runs
        for name in r.get('rerankers', {})
    })
    print(f'\nTrend on `{metric}` over {len(runs)} run(s):\n')
    print(f'{"reranker":<25s} ' + ' '.join(f'{i:>6d}' for i in range(len(runs))))
    print('-' * (25 + 7 * len(runs)))
    for name in rerankers:
        cells = []
        for r in runs:
            v = (r.get('rerankers', {}).get(name) or {}).get(metric)
            if v is None:
                cells.append('  -   ')
            elif isinstance(v, float):
                cells.append(f'{v:>6.3f}')
            else:
                cells.append(f'{v:>6d}')
        print(f'{name:<25s} ' + ' '.join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--history-path', default=str(_HISTORY))
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_append = sub.add_parser('append', help='Append a run from a JSON summary')
    ap_append.add_argument('--run-summary', required=True,
                           help='Path to a JSON file containing the run summary')

    sub.add_parser('list', help='List all recorded runs')

    ap_compare = sub.add_parser('compare', help='Compare two runs (default: last 2)')
    ap_compare.add_argument('--a', type=int, default=None, help='index of run A')
    ap_compare.add_argument('--b', type=int, default=None, help='index of run B')

    ap_trend = sub.add_parser('trend', help='Show metric trend over runs')
    ap_trend.add_argument('--metric', default='mrr',
                          choices=['recall_at_1', 'recall_at_5', 'recall_at_10',
                                   'mrr', 'answer_accuracy'])

    args = ap.parse_args()

    if args.cmd == 'append':
        with open(args.run_summary) as f:
            summary = json.load(f)
        append_run(Path(args.history_path), summary)
    elif args.cmd == 'list':
        cmd_list(args)
    elif args.cmd == 'compare':
        cmd_compare(args)
    elif args.cmd == 'trend':
        cmd_trend(args)


if __name__ == '__main__':
    main()
