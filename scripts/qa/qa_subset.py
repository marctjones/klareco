#!/usr/bin/env python3
"""
Carve fast / targeted subsets out of a gold Q&A set. (#736)

VERSION: v1.0
COMPATIBLE WITH: qa_gold_v*.jsonl (schema records with difficulty_band, question_type,
                 category, source, bm25_gold_rank)
STAGE: Evaluation / test-set construction

Description:
    A full run over ~1,000 gold pairs is slow. For iteration you want either:
      - a small FIXED smoke set — "did we break it?" in a minute, stable across runs
        so results are comparable, or
      - a TARGETED slice — "how are we doing on the DEEP band / KIO questions /
        Geography?" — to focus on one class of issue.
    This filters + (stratified-)samples a gold set into such a subset. Any evaluator
    then just takes --test-set <subset>. Deterministic (seeded) so a subset is stable.

    Classes you can target (all are record fields):
      difficulty_band  trivial | rerankable | deep     (retriever/reranker difficulty)
      question_type    KIU | KIUN | KIO | KIE | KIAM | KIOM | ...  (retrieval/extraction path)
      category         Geography | History | ...         (topic coverage)
      source           opentdb | corpus                  (authorship)

Usage:
    # fixed smoke set: 24 pairs balanced across difficulty bands
    python scripts/qa/qa_subset.py --input data/test_sets/qa_gold_v2.jsonl \
        --stratify difficulty_band --sample 24 --out data/test_sets/smoke.jsonl

    # targeted: only the DEEP (answer-present-but-hard) band
    python scripts/qa/qa_subset.py --band deep --out /tmp/deep.jsonl
    # targeted: KIO (definition) questions, 40 of them
    python scripts/qa/qa_subset.py --type KIO --sample 40 --out /tmp/kio.jsonl

Last Updated: 2026-07-17
Related Issues: #736, #737
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

_FILTERS = {'band': 'difficulty_band', 'type': 'question_type',
            'category': 'category', 'source': 'source'}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--input', default='data/test_sets/qa_gold_v2.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--band', help='keep only this difficulty_band')
    ap.add_argument('--type', help='keep only this question_type')
    ap.add_argument('--category', help='keep only this category')
    ap.add_argument('--source', help='keep only this source')
    ap.add_argument('--sample', type=int, help='cap to this many (after filtering)')
    ap.add_argument('--stratify', help='balance the --sample across values of this field '
                                       '(e.g. difficulty_band)')
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding='utf-8') if l.strip()]
    for flag, field in _FILTERS.items():
        val = getattr(args, flag)
        if val:
            rows = [r for r in rows if str(r.get(field)) == val]
    # deterministic order first (by id), so seeded selection is reproducible
    rows.sort(key=lambda r: str(r.get('id', '')))
    rng = random.Random(args.seed)

    if args.sample and args.sample < len(rows):
        if args.stratify:
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r.get(args.stratify, 'unknown')].append(r)
            keys = sorted(groups)
            per = max(1, args.sample // max(len(keys), 1))
            picked = []
            for k in keys:
                g = groups[k][:]
                rng.shuffle(g)
                picked.extend(g[:per])
            rng.shuffle(picked)
            rows = picked[:args.sample]
        else:
            rows = rows[:]
            rng.shuffle(rows)
            rows = rows[:args.sample]
        rows.sort(key=lambda r: str(r.get('id', '')))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'  ✓ {len(rows)} pairs -> {args.out}')
    print(f'    band: {dict(collections.Counter(r.get("difficulty_band") for r in rows))}')
    print(f'    type: {dict(collections.Counter(r.get("question_type") for r in rows))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
