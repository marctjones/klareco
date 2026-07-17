#!/usr/bin/env python3
"""
Assemble the versioned gold set: merge engines, validate, dedup, stratify. (#845)

VERSION: v1.0
COMPATIBLE WITH: qa_source_opentdb / qa_source_corpus / qa_answerability outputs
DEPENDENCIES: klareco.eval.qa_schema
STAGE: Evaluation / test-set construction

Description:
    The last stage. Takes one or more candidate/gold JSONL files (from either engine),
    keeps only schema-valid rows, de-duplicates by source sentence AND by question,
    stratifies, and writes a single versioned gold file. Reports per-stratum N against
    the power targets (~185 for a 0.03 MRR delta; ~67 for 0.05).

    A "frozen benchmark" subset can be split off for the merge gate — a stable ruler,
    kept separate from a fresh pool so we never overfit to it.

Usage:
    python scripts/qa/qa_build_assemble.py \
        --inputs data/staging/corpus_gold_v1.jsonl data/staging/opentdb_gold.jsonl \
        --out data/test_sets/qa_gold_v1.jsonl --freeze 60

Last Updated: 2026-07-17
Related Issues: #842, #845, #846
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.eval.qa_schema import validate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', default='data/test_sets/qa_gold_v1.jsonl')
    ap.add_argument('--freeze', type=int, default=0,
                    help='split off a frozen benchmark subset of this size')
    args = ap.parse_args()

    today = datetime.date.today().isoformat()
    # Dedup by QUESTION only — NOT by source sentence. Several distinct questions can
    # (and should) share one rich answering sentence: "Zamenhof verkis la libron de
    # Petro en 1887" answers who/when/whose. Keeping them is a genuine extraction-
    # quality probe (does the system pick the RIGHT answer among several offered).
    seen_q, seen_ids = set(), set()
    gold, invalid, dup = [], 0, 0

    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f'  ! missing input: {path}'); continue
        n_in = 0
        for line in p.open(encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            ok, errs = validate(row)
            if not ok:
                invalid += 1; continue
            qkey = row['question'].strip().lower()
            if qkey in seen_q:                 # same QUESTION twice — a true duplicate
                dup += 1; continue
            seen_q.add(qkey)
            # unique id even when questions share a sentence (gold-<sid> would collide)
            base = row.get('id') or f"gold-{row['source_sentence_id']}"
            uid, k = base, 1
            while uid in seen_ids:
                uid = f'{base}-{k}'; k += 1
            seen_ids.add(uid); row['id'] = uid
            row.setdefault('created', today)
            gold.append(row)
        print(f'  {p.name}: {n_in} rows in')

    if not gold:
        print('\n  ✗ nothing valid to assemble.'); return 1

    # deterministic order (by source then sid) so diffs are stable
    gold.sort(key=lambda r: (r.get('source', ''), str(r['source_sentence_id'])))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frozen = gold[:args.freeze] if args.freeze else []
    working = gold[args.freeze:] if args.freeze else gold
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in working:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {len(working)} gold -> {args.out}   '
          f'(skipped {invalid} invalid, {dup} dup)')
    if frozen:
        fp = args.out.replace('.jsonl', '_frozen.jsonl')
        with open(fp, 'w', encoding='utf-8') as f:
            for r in frozen:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  ✓ {len(frozen)} frozen benchmark -> {fp}')

    # stratification report
    print('\n  STRATIFICATION (source × difficulty):')
    strat = collections.Counter((r.get('source', '?'), r.get('difficulty_band', '?'))
                                for r in working)
    for k, c in sorted(strat.items()):
        print(f'    {k[0]:8s} {k[1]:11s} {c}')
    print('\n  BY QUESTION TYPE:')
    for qt, c in sorted(collections.Counter(r.get('question_type', '?')
                                            for r in working).items()):
        print(f'    {qt:6s} {c}')
    rerankable = sum(1 for r in working if r.get('difficulty_band') == 'rerankable')
    print(f'\n  rerankable band: {rerankable}  '
          f'(need ~185 for a 0.03 MRR reranker delta — the #23 stratum)')
    print(f'  total general set: {len(working)}  '
          f'(resolves ~{1.96*0.21/max(len(working),1)**0.5:.3f} pooled MRR delta)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
