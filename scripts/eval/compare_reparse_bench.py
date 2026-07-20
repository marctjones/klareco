#!/usr/bin/env python3
"""
Before/after comparator for the #807 store reparse — is the delta real?

VERSION: v1.0
COMPATIBLE WITH: multi_reranker_bench.py --output-jsonl/--output-summary files
DEPENDENCIES: numpy, klareco.eval.bootstrap
STAGE: Evaluation

Description:
    THE MERGE-GATE NUMBER for #807. Takes two multi_reranker_bench runs on the
    SAME test set (rebaseline_500) — one against the pre-reparse store, one
    against the post-reparse store — and answers, per reranker:

      - MRR before/after, with a PAIRED bootstrap 95% CI on the delta
        (klareco.eval.bootstrap.paired_delta_ci — point deltas are not
        evidence, #726)
      - answer_accuracy before/after, with a paired bootstrap CI on the
        proportion delta

    Questions are paired by `id`; only ids present in both runs are compared
    (drops are reported loudly, never silently). The A_bm25_baseline row is
    the deterministic floor — the headline number for #807.

Pipeline Position:
    multi_reranker_bench (before) ─┐
                                   ├─> [THIS] -> stdout table
    multi_reranker_bench (after)  ─┘         -> bench_history.jsonl entry (--append-history)

Usage:
    python scripts/eval/compare_reparse_bench.py \
        --before-rows results/reparse807_before_rows.jsonl \
        --after-rows  results/reparse807_after_rows.jsonl \
        --label "#807 reparse" \
        --append-history data/perf/bench_history.jsonl

Inputs:
    --before-rows / --after-rows   per-question detail jsonl from the bench
    --label                        short tag for the history entry
    --append-history               path to bench_history.jsonl (omit = print only)

Outputs:
    Stdout comparison table; optional appended JSONL decision entry.

Quality Checks:
    - refuses to compare runs with < 50 shared question ids
    - reports unpaired ids per side
    - CI method identical to the shipped reranker gate (paired, resample
      questions, 10k iterations, seed 42)

Last Updated: 2026-07-19
Author: Claude Fable 5 (with Marc Jones)
Related Issues: #807, #871, #823, #726
See Also: scripts/pipeline/reparse_store_807.sh, klareco/eval/bootstrap.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from klareco.eval.bootstrap import paired_delta_ci


def load_rows(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r['id']] = r
    return rows


def reranker_names(rows: dict[str, dict]) -> set[str]:
    names = set()
    for r in rows.values():
        for k in r:
            if k.endswith('_rank'):
                names.add(k[:-len('_rank')])
    return names


def paired_accuracy_ci(after: np.ndarray, before: np.ndarray,
                       n_boot: int = 10_000, seed: int = 42) -> dict:
    """Paired bootstrap CI on the accuracy (proportion-correct) delta."""
    rng = np.random.default_rng(seed)
    n = len(after)
    diffs = after.astype(float) - before.astype(float)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    delta = float(diffs.mean())
    return {'delta': delta, 'lo': float(lo), 'hi': float(hi),
            'significant': bool(lo > 0 or hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--before-rows', required=True)
    ap.add_argument('--after-rows', required=True)
    ap.add_argument('--label', default='#807 store reparse')
    ap.add_argument('--append-history', default=None,
                    help='bench_history.jsonl path; omit to print only')
    args = ap.parse_args()

    before = load_rows(args.before_rows)
    after = load_rows(args.after_rows)

    shared = sorted(set(before) & set(after))
    only_b = len(before) - len(shared)
    only_a = len(after) - len(shared)
    print(f'paired questions: {len(shared)} '
          f'(before-only: {only_b}, after-only: {only_a})')
    if only_b or only_a:
        print('  WARNING: unpaired ids exist — the runs did not see identical '
              'question sets; comparison uses the intersection only.')
    if len(shared) < 50:
        print(f'FATAL: only {len(shared)} shared ids — not comparable.')
        return 2

    names = sorted(reranker_names(before) & reranker_names(after))
    if not names:
        print('FATAL: no reranker columns shared between the two runs.')
        return 2

    results = {}
    hdr = (f'{"reranker":<20} {"MRR b":>7} {"MRR a":>7} {"ΔMRR":>8} '
           f'{"95% CI":>20} {"sig":>4} {"acc b":>7} {"acc a":>7} {"Δacc":>8}')
    print('\n' + hdr)
    print('-' * len(hdr))
    for name in names:
        ranks_b = [before[i].get(f'{name}_rank') for i in shared]
        ranks_a = [after[i].get(f'{name}_rank') for i in shared]
        corr_b = np.array([bool(before[i].get(f'{name}_correct'))
                           for i in shared])
        corr_a = np.array([bool(after[i].get(f'{name}_correct'))
                           for i in shared])

        mrr_b = float(np.mean([1.0 / r if r else 0.0 for r in ranks_b]))
        mrr_a = float(np.mean([1.0 / r if r else 0.0 for r in ranks_a]))
        mrr_ci = paired_delta_ci(ranks_a, ranks_b)
        acc_ci = paired_accuracy_ci(corr_a, corr_b)
        acc_b, acc_a = float(corr_b.mean()), float(corr_a.mean())

        sig = '*' if mrr_ci['significant'] else ''
        print(f'{name:<20} {mrr_b:>7.4f} {mrr_a:>7.4f} '
              f"{mrr_ci['delta']:>+8.4f} "
              f"[{mrr_ci['lo']:>+8.4f},{mrr_ci['hi']:>+8.4f}] {sig:>4} "
              f'{acc_b:>7.2%} {acc_a:>7.2%} {acc_ci["delta"]:>+8.2%}')

        results[name] = {
            'mrr_before': round(mrr_b, 4), 'mrr_after': round(mrr_a, 4),
            'mrr_delta': round(mrr_ci['delta'], 4),
            'mrr_ci': [round(mrr_ci['lo'], 4), round(mrr_ci['hi'], 4)],
            'mrr_significant': bool(mrr_ci['significant']),
            'acc_before': round(acc_b, 4), 'acc_after': round(acc_a, 4),
            'acc_delta': round(acc_ci['delta'], 4),
            'acc_ci': [round(acc_ci['lo'], 4), round(acc_ci['hi'], 4)],
            'acc_significant': acc_ci['significant'],
        }

    # ATTRIBUTION, not one headline number.
    #
    # The naive choice was A_bm25_baseline as "the deterministic floor". That is
    # WRONG for this change: A reads no store features at all (pure BM25 over
    # `text`), while #871's win is RECOVERED SUBJECTS. Headlining A would report
    # "the reparse did nothing" while the subject signal sat unread in another
    # row. Verified 2026-07-20 by inspecting what each reranker actually queries:
    #
    #   A_bm25_baseline  no store features  -> isolates the #823 corpus-hygiene
    #                                          effect (junk rows leave the pool)
    #   I_clause_aware   clauses.subj/verb/obj_radiko, NO negation
    #                                       -> CLEAN #871 signal (AST refresh only)
    #   G_ast_aware      subj_radiko AND verb_negated
    #   H_hybrid         (same, via ast_aware)
    #                                       -> CONFOUNDED: the reparse also adds
    #                                          the verb_negated column, which
    #                                          ACTIVATES a negation hard-filter
    #                                          that is inert today (column
    #                                          missing -> guard never fires).
    #                                          Two changes, one row.
    ATTRIBUTION = [
        ('I_clause_aware', 'CLEAN #871 signal — reads clause subjects, not negation'),
        ('A_bm25_baseline', '#823 hygiene only — reads no store features'),
        ('G_ast_aware', 'CONFOUNDED — subject recovery + newly-active negation filter'),
        ('H_hybrid', 'CONFOUNDED — same as G, via ast_aware'),
    ]
    print('\n=== ATTRIBUTION ===')
    for name, note in ATTRIBUTION:
        r = results.get(name)
        if not r:
            continue
        verdict = ('MOVED (CI excludes 0)' if r['mrr_significant']
                   else 'did not move (CI includes 0)')
        print(f"  {name:<18} MRR {r['mrr_before']:.4f} -> {r['mrr_after']:.4f} "
              f"({r['mrr_delta']:+.4f})  {verdict}")
        print(f"  {'':<18} {note}")
    print('\n  The #807 merge-gate number is the I_clause_aware row: it is the '
          'only\n  reranker whose delta is attributable to the AST refresh alone.')

    if args.append_history:
        entry = {
            'date': datetime.date.today().isoformat(),
            'issue': args.label,
            'method': (f'multi_reranker_bench on rebaseline_500 before/after '
                       f'the store reparse; {len(shared)} paired questions; '
                       f'paired bootstrap 95% CI (klareco.eval.bootstrap), '
                       f'10k resamples, seed 42.'),
            'files': {'before': args.before_rows, 'after': args.after_rows},
            'result': results,
        }
        hist = Path(args.append_history)
        hist.parent.mkdir(parents=True, exist_ok=True)
        with open(hist, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f'\nappended decision entry to {hist}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
