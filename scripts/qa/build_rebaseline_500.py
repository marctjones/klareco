#!/usr/bin/env python3
"""
Grow the frozen benchmark rebaseline_210 -> rebaseline_500 (#847).

VERSION: v1.0
COMPATIBLE WITH: qa_gold_v2 pool + rebaseline_210 frozen set
DEPENDENCIES: none (stdlib)
STAGE: Evaluation / test-set construction

Description:
    rebaseline_210 (70/70/70 across trivial/rerankable/deep) is under-powered:
    band-local wins land with a paired-bootstrap CI that just includes 0 (e.g.
    #877 proper-noun boost +0.0083 MRR, CI[-0.0020,+0.0194]). More questions per
    band tighten those CIs. This grows the frozen benchmark to ~500, balanced
    (166/167/167), KEEPING all 210 existing questions as a subset (so old and new
    numbers stay comparable) and adding from qa_gold_v2's audited pool.

    Selection is DETERMINISTIC (sort by id, take the first N per band from the
    pool minus the existing set) — a frozen benchmark must be reproducible.

Usage:
    python scripts/qa/build_rebaseline_500.py --apply

Inputs:  data/test_sets/qa_gold_v2.jsonl, data/test_sets/rebaseline_210.jsonl
Outputs: data/test_sets/rebaseline_500.jsonl
Quality: reports per-band N; re-run scripts/qa/qa_audit.py on the output.

Last Updated: 2026-07-19
Author: Claude Opus 4.8
Related Issues: #847, #877 (the win this powers), #845 (assemble/freeze)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'data' / 'test_sets' / 'qa_gold_v2.jsonl'
BASE = ROOT / 'data' / 'test_sets' / 'rebaseline_210.jsonl'
OUT = ROOT / 'data' / 'test_sets' / 'rebaseline_500.jsonl'
TARGET = {'trivial': 166, 'rerankable': 167, 'deep': 167}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    pool = load(POOL)
    base = load(BASE)
    base_ids = {r['id'] for r in base}

    # existing per band (kept), and the remaining pool per band (sorted by id)
    kept = collections.defaultdict(list)
    for r in base:
        kept[r.get('difficulty_band')].append(r)
    avail = collections.defaultdict(list)
    for r in sorted(pool, key=lambda r: r['id']):
        if r['id'] not in base_ids:
            avail[r.get('difficulty_band')].append(r)

    out, report = [], {}
    for band, want in TARGET.items():
        have = list(kept.get(band, []))
        need = max(0, want - len(have))
        take = avail.get(band, [])[:need]
        chosen = have + take
        out.extend(chosen)
        report[band] = (len(have), len(take), len(chosen),
                        len(avail.get(band, [])))

    print(f"rebaseline_500 build: pool={len(pool)}, base={len(base)}")
    print(f"  {'band':12}{'kept':>6}{'added':>7}{'total':>7}{'pool_avail':>12}")
    for band, (h, t, c, av) in report.items():
        flag = '' if t <= av else '  ⚠ NOT ENOUGH IN POOL'
        print(f"  {band:12}{h:>6}{t:>7}{c:>7}{av:>12}{flag}")
    print(f"  TOTAL: {len(out)}")

    if a.apply:
        with open(OUT, 'w') as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nWROTE {OUT} ({len(out)} rows). Audit: "
              f"python scripts/qa/qa_audit.py --test-set {OUT.relative_to(ROOT)}")
    else:
        print("\nDRY-RUN — pass --apply to write.")


if __name__ == '__main__':
    main()
