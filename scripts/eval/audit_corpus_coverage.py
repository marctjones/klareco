#!/usr/bin/env python3
"""
Corpus-coverage audit (R14): how many corpus passages support a Q&A pair?

VERSION: v2.x (DuckDB + Whoosh)
COMPATIBLE WITH: any test-set JSONL with {question, expected_answer,
                 expected_keywords (optional)}
DEPENDENCIES: whoosh, duckdb
STAGE: Evaluation

Description:
    For each Q&A pair, runs a BM25 query on the question text alone
    (no answer, no source-text leakage), fetches the top-K passages
    from Whoosh, and counts how many contain any of the
    expected_keywords (diacritic-fold substring match).

    This is the "corpus support count" — independent of question
    quality. It separates two concerns:

      - R7 (discriminability): is the source sentence in the BM25
        top-K? (Yes / no — retrievability.)
      - R14 (this script): how many corpus passages support the
        answer? (A count — coverage depth.)

    A pair with support_count == 1 (only the source) is brittle:
    a single corpus edit breaks the pair, and the pipeline has no
    redundant evidence to rely on.

    A pair with support_count == 0 either has weak BM25 signal
    (question terms don't surface any answer-bearing passage) or
    a corpus-coverage gap (this fact isn't well-represented in our
    corpus). Both are useful to log.

Pipeline Position:
    <test_set>.jsonl → [THIS SCRIPT] → per-pair coverage report
                                     → (drop low-support pairs from
                                        capability set; keep them in
                                        honest-ceiling set as gap log)

Usage:
    # Capability set — enforce ≥ 3 supporting passages
    python scripts/eval/audit_corpus_coverage.py \\
        --in data/test_sets/capability_100.jsonl \\
        --min-support 3 --strict

    # Honest-ceiling set — report only, no gate
    python scripts/eval/audit_corpus_coverage.py \\
        --in data/test_sets/trivia_real_50.jsonl \\
        --report-only

Inputs:
    --in           one or more JSONL test sets
    --top-k        Whoosh top-K to scan (default 50)
    --min-support  minimum number of supporting passages for PASS (default 3)
    --strict       exit 1 if any pair fails the threshold
    --report-only  ignore --min-support; print results and exit 0

Outputs:
    Per-pair JSONL coverage log via --output (optional).
    Aggregate distribution printed to stdout.

Quality Checks:
    R14 corpus-coverage robustness. See docs/QA_TEST_SET_QUALITY_STANDARD.md.

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from whoosh.index import open_dir  # noqa: E402
from whoosh.qparser import OrGroup, QueryParser  # noqa: E402


# Function words dropped from the BM25 query — mirrors duckdb_retriever
_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis estos '
            'la de en al el ĉu por kaj aŭ ke ne je da'.split())


def _fold(s: str) -> str:
    """Diacritic-fold + lowercase for substring matching."""
    if not s:
        return ''
    decomposed = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _content_terms(q: str) -> list[str]:
    """Extract content terms from a question for BM25 query."""
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]


def support_count(searcher, qparser, question: str, expected_keywords: list[str],
                  source_sentence_id: int | None,
                  top_k: int) -> tuple[int, int | None, list[int]]:
    """Run BM25 on question terms; count passages whose text contains any
    expected_keywords (diacritic-fold match).

    Returns:
      - support_count: how many top-K passages contain an expected_keyword
      - source_rank: 1-indexed rank of source_sentence_id in the result, or None
      - supporting_sids: list of sids that supported the answer
    """
    terms = _content_terms(question)
    if not terms:
        return 0, None, []
    q_string = ' '.join(terms)
    try:
        query = qparser.parse(q_string)
    except Exception:
        return 0, None, []
    try:
        results = searcher.search(query, limit=top_k)
    except Exception:
        return 0, None, []

    folded_keywords = [_fold(k) for k in expected_keywords if k]
    n_support = 0
    source_rank: int | None = None
    supporting_sids: list[int] = []

    for rank, hit in enumerate(results, 1):
        sid_str = hit.get('sid') or hit.get('id') or hit.get('docid')
        if sid_str is None:
            continue
        try:
            sid = int(sid_str)
        except (TypeError, ValueError):
            continue
        if source_sentence_id is not None and sid == int(source_sentence_id):
            source_rank = rank

        text = hit.get('text') or hit.get('content') or ''
        folded_text = _fold(text)
        if any(fk and fk in folded_text for fk in folded_keywords):
            n_support += 1
            supporting_sids.append(sid)

    return n_support, source_rank, supporting_sids


def audit_pair(searcher, qparser, pair: dict, top_k: int, min_support: int
               ) -> dict:
    question = pair.get('question') or ''
    expected_answer = pair.get('expected_answer') or ''
    expected_keywords = pair.get('expected_keywords') or []
    if expected_answer and expected_answer not in expected_keywords:
        expected_keywords = list(expected_keywords) + [expected_answer]
    sid = pair.get('source_sentence_id')

    n_support, source_rank, supporting_sids = support_count(
        searcher, qparser, question, expected_keywords,
        sid, top_k,
    )
    verdict = 'PASS' if n_support >= min_support else 'FAIL'

    return {
        'id':               pair.get('id'),
        'question':         question,
        'expected_answer':  expected_answer,
        'expected_keywords': expected_keywords,
        'source_sentence_id': sid,
        'source_rank':       source_rank,
        'support_count':     n_support,
        'supporting_sids':   supporting_sids[:10],
        'verdict':           verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--in', dest='inputs', nargs='+', required=True)
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--min-support', type=int, default=3)
    ap.add_argument('--output', default=None)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--report-only', action='store_true',
                    help='Ignore --min-support and never exit non-zero.')
    args = ap.parse_args()

    print(f'Opening Whoosh index at {args.whoosh_dir}…')
    ix = open_dir(args.whoosh_dir)
    searcher = ix.searcher()
    qparser = QueryParser('text', schema=ix.schema, group=OrGroup)

    all_results: list[dict] = []
    per_set: dict[str, dict[str, int]] = {}

    for ts_path in args.inputs:
        path = Path(ts_path)
        if not path.exists():
            print(f'SKIP: {path} not found', file=sys.stderr)
            continue
        pairs: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        per_set[path.name] = {'pass': 0, 'fail': 0}
        for p in pairs:
            r = audit_pair(searcher, qparser, p, args.top_k, args.min_support)
            r['source_set'] = path.name
            all_results.append(r)
            per_set[path.name][r['verdict'].lower()] += 1

    n = len(all_results)
    n_pass = sum(1 for r in all_results if r['verdict'] == 'PASS')
    n_fail = n - n_pass

    print(f'\nCorpus-coverage audit ({n} pairs, top-K={args.top_k}, '
          f'min-support={args.min_support})')
    if n:
        print(f'  PASS: {n_pass} ({100*n_pass/n:.1f}%)')
        print(f'  FAIL: {n_fail} ({100*n_fail/n:.1f}%)')

    print('\nPer-set breakdown:')
    for s, c in per_set.items():
        tot = c['pass'] + c['fail']
        pct = (c['pass'] / tot * 100) if tot else 0
        print(f'  {s:<48s}  {c["pass"]:>3}/{tot:<3}  {pct:5.1f}% pass')

    # Support-count distribution
    bucket_labels = ['0', '1', '2', '3', '4', '5-9', '10-19', '20-49', '50+']
    def _bucket(c: int) -> str:
        if c < 5:
            return str(c)
        if c < 10:
            return '5-9'
        if c < 20:
            return '10-19'
        if c < 50:
            return '20-49'
        return '50+'
    dist = Counter(_bucket(r['support_count']) for r in all_results)
    print('\nSupport-count distribution:')
    for b in bucket_labels:
        n_b = dist.get(b, 0)
        bar = '█' * min(40, n_b)
        print(f'  {b:>5s}  {n_b:>4d}  {bar}')

    # Pairs with no support at all (potential corpus-coverage gaps)
    zero_support = [r for r in all_results if r['support_count'] == 0]
    if zero_support:
        print(f'\nZero-support pairs ({len(zero_support)}) — corpus coverage gaps:')
        for r in zero_support[:15]:
            print(f'  - {(r["id"] or ""):<22s} {r["question"][:60]}')
            print(f'    expected: {r["expected_answer"][:60]}')

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'\nFull audit written to {out}')

    if args.strict and not args.report_only and n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
