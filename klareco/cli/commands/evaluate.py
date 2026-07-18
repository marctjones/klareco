"""
Evaluate against a gold set (the merge-gate ledger).

Thin wrapper today (dispatches to the existing evaluator); the target is a
first-class command that writes results to data/perf/bench_history.jsonl.
"""
from __future__ import annotations

import subprocess

from klareco.cli._base import add_common, emit


def cmd_eval(args) -> int:
    cmd = ['python', 'scripts/eval/evaluate_extractive_qa.py',
           '--test-set', args.test_set]
    emit(args, text=f"$ {' '.join(cmd)}", data={'command': cmd})
    return subprocess.call(cmd)


def register(sub) -> None:
    e = sub.add_parser('eval', help='Evaluate against a gold set (answer/retrieval metrics)')
    e.add_argument('--test-set', required=True, help='Path to a gold JSONL')
    add_common(e)
    e.set_defaults(func=cmd_eval)
