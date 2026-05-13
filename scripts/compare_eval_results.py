#!/usr/bin/env python3
"""
Compare two extractive-QA eval result JSONs and report deltas.

VERSION: v2.1
COMPATIBLE WITH: scripts/evaluate_extractive_qa.py output format
DEPENDENCIES: stdlib only
STAGE: Evaluation / reporting

Description:
    Reads two evaluate_extractive_qa.py output JSONs (the .json file the
    eval writes — top-level keys 'summary', 'results', 'by_type'). Prints
    side-by-side scalar metrics, per-type breakdown, and recall/density
    curves at K ∈ {1, 5, 10, 25, 50, 100, 500, 1000} (clamped to whatever
    K is present in both runs).

Usage:
    python scripts/compare_eval_results.py BASELINE.json NEW.json
    python scripts/compare_eval_results.py --label-baseline "old" \\
                                           --label-new      "new" \\
                                           BASELINE.json NEW.json

Outputs:
    Console table.

Last Updated: 2026-05-07
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt_num(v):
    if isinstance(v, float):
        return f'{v:.4f}'
    if isinstance(v, int):
        return f'{v:,}'
    return str(v)


def fmt_delta(old, new):
    if not (isinstance(old, (int, float)) and isinstance(new, (int, float))):
        return ''
    delta = new - old
    if isinstance(delta, float):
        return f'{delta:+.4f}'
    return f'{delta:+,}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('new')
    ap.add_argument('--label-baseline', default='OLD')
    ap.add_argument('--label-new', default='NEW')
    args = ap.parse_args()

    a = json.load(open(args.baseline))
    b = json.load(open(args.new))

    sa, sb = a.get('summary', {}), b.get('summary', {})

    print(f'BASELINE: {args.baseline}')
    print(f'NEW:      {args.new}')

    print(f'\n{"="*70}')
    print(f'{"Metric":30s}  {args.label_baseline:>14s}  {args.label_new:>14s}  {"Δ":>10s}')
    print(f'{"-"*70}')
    for key in ('n', 'answer_accuracy', 'retrieval_recall', 'mrr',
                 'avg_latency_sec', 'p50_latency_sec', 'p95_latency_sec',
                 'max_latency_sec'):
        if key not in sa and key not in sb:
            continue
        va, vb = sa.get(key), sb.get(key)
        print(f'{key:30s}  {fmt_num(va):>14s}  {fmt_num(vb):>14s}  {fmt_delta(va, vb):>10s}')

    # Recall@K
    rka = sa.get('recall_at_k') or {}
    rkb = sb.get('recall_at_k') or {}
    keys = sorted(set(rka.keys()) | set(rkb.keys()), key=lambda x: int(x))
    if keys:
        print(f'\n=== recall@K ===')
        print(f'{"K":>6s}  {args.label_baseline:>14s}  {args.label_new:>14s}  {"Δ":>10s}')
        for k in keys:
            va, vb = rka.get(k), rkb.get(k)
            print(f'{k:>6s}  {fmt_num(va):>14s}  {fmt_num(vb):>14s}  {fmt_delta(va, vb):>10s}')

    # density@K
    dka = sa.get('density_at_k') or {}
    dkb = sb.get('density_at_k') or {}
    keys = sorted(set(dka.keys()) | set(dkb.keys()), key=lambda x: int(x))
    if keys:
        print(f'\n=== density@K ===')
        print(f'{"K":>6s}  {args.label_baseline:>14s}  {args.label_new:>14s}  {"Δ":>10s}')
        for k in keys:
            va, vb = dka.get(k), dkb.get(k)
            print(f'{k:>6s}  {fmt_num(va):>14s}  {fmt_num(vb):>14s}  {fmt_delta(va, vb):>10s}')

    # By type
    bta, btb = a.get('by_type') or {}, b.get('by_type') or {}
    keys = sorted(set(bta.keys()) | set(btb.keys()))
    if keys:
        print(f'\n=== by question type ===')
        for k in keys:
            xa, xb = bta.get(k, {}), btb.get(k, {})
            if not (isinstance(xa, dict) and isinstance(xb, dict)):
                continue
            print(f'\n  type: {k}')
            for sub in ('n', 'answer_accuracy', 'retrieval_recall', 'mrr'):
                va, vb = xa.get(sub), xb.get(sub)
                if va is None and vb is None:
                    continue
                print(f'    {sub:25s}  {fmt_num(va):>14s}  '
                      f'{fmt_num(vb):>14s}  {fmt_delta(va, vb):>10s}')


if __name__ == '__main__':
    main()
