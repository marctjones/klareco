#!/usr/bin/env python3
"""
Batch-search candidate trivia questions against the corpus.

VERSION: v2.x
COMPATIBLE WITH: any candidate JSONL with {question, must_contain[], topic, question_type}
DEPENDENCIES: whoosh, klareco.parser
STAGE: Evaluation (test-set generation)

Description:
    Single-question search_candidate_answer.py is fine for one-off
    debugging but wasteful for batch generation: each call reopens the
    Whoosh index (~3s startup) and produces verbose output. This script
    takes a JSONL of candidates and processes them all in one process.

    For each candidate, it:
      1. Parses the question (reports parse status, marks parse-broken
         as REJECT).
      2. Runs BM25 with the question's content terms.
      3. Filters top-K hits by `must_contain` (any-of).
      4. Returns the single best (highest-BM25, shortest) matching
         sentence as the proposed answer.

    Output is a JSONL with PASS/REJECT verdicts. A PASS row carries
    enough info to be added directly to a test-set JSONL.

Pipeline Position:
    candidates JSONL → [THIS SCRIPT] → results JSONL (PASS / REJECT)
                                     → (post-process: keep PASS rows)

Usage:
    python scripts/eval/batch_search_candidates.py \\
        --in data/staging/candidates.jsonl \\
        --out data/staging/search_results.jsonl

Input JSONL row format:
    {
      "id":            "geo_001",
      "topic":         "geography",
      "question_type": "KIO",
      "question":      "Kio estas la ĉefurbo de Aŭstralio?",
      "must_contain":  ["Kanbero", "Canberra"]
    }

Output JSONL row format (PASS):
    {
      "id":              "geo_001",
      "topic":           "geography",
      "question_type":   "KIO",
      "question":        "Kio estas la ĉefurbo de Aŭstralio?",
      "expected_answer": "Kanbero",        // from must_contain
      "expected_keywords": ["Kanbero", "Canberra"],
      "source_sentence_id":   585363,
      "source_sentence_text": "Kanbero estas la ĉefurbo…",
      "bm25_rank":            9,
      "verdict":              "PASS",
      "parse_status":         "clean",
      "candidate_count":      6
    }

Output JSONL row format (REJECT):
    {
      "id": "geo_011", "verdict": "REJECT",
      "reason": "no top-50 hit contains any must_contain token",
      ...
    }

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse  # noqa: E402
from whoosh.index import open_dir  # noqa: E402
from whoosh.qparser import OrGroup, QueryParser  # noqa: E402


_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis estos '
            'la de en al el ĉu por kaj aŭ ke ne je da'.split())


def _content_terms(q: str) -> list[str]:
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]


def _fold(s: str) -> str:
    decomposed = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _parse_status(question: str) -> tuple[str, str]:
    """Returns ('clean'|'broken', reason)."""
    try:
        ast = parse(question)
    except Exception as e:
        return 'broken', f'parse raised: {e}'
    def _walk(n):
        if isinstance(n, dict):
            if n.get('tipo') == 'vorto':
                yield n
            else:
                for v in n.values():
                    yield from _walk(v)
        elif isinstance(n, list):
            for x in n:
                yield from _walk(x)
    bad = []
    for w in _walk(ast):
        st = w.get('analizstato')
        if st and st not in ('sukceso', 'propra_nomo_nekonata'):
            bad.append(f'{w.get("plena_vorto")}({st})')
    if bad:
        return 'broken', f'unparsed: {", ".join(bad[:3])}'
    return 'clean', 'ok'


def search_one(searcher, qparser, candidate: dict, top_k: int
               ) -> dict:
    """Run one candidate; return result row."""
    q = candidate.get('question', '')
    must = candidate.get('must_contain') or []
    out = {
        'id':            candidate.get('id'),
        'topic':         candidate.get('topic'),
        'question_type': candidate.get('question_type'),
        'question':      q,
        'must_contain':  must,
    }

    # 1. Parse status
    status, reason = _parse_status(q)
    out['parse_status'] = status
    if status == 'broken':
        out['verdict'] = 'REJECT'
        out['reason'] = f'parse: {reason}'
        return out

    # 2. BM25
    terms = _content_terms(q)
    if not terms:
        out['verdict'] = 'REJECT'
        out['reason'] = 'no content terms in question'
        return out
    try:
        query = qparser.parse(' '.join(terms))
        results = searcher.search(query, limit=top_k)
    except Exception as e:
        out['verdict'] = 'REJECT'
        out['reason'] = f'bm25 raised: {e}'
        return out

    folded_must = [_fold(s) for s in must]
    matched = []
    for rank, hit in enumerate(results, 1):
        sid = hit.get('sid') or hit.get('id') or hit.get('docid')
        text = hit.get('text') or hit.get('content') or ''
        if folded_must:
            folded_text = _fold(text)
            if not any(req and req in folded_text for req in folded_must):
                continue
        try:
            sid_i = int(sid)
        except (TypeError, ValueError):
            continue
        matched.append({'rank': rank, 'sid': sid_i, 'text': text})

    out['candidate_count'] = len(matched)
    if not matched:
        out['verdict'] = 'REJECT'
        out['reason'] = (f'no hit in top-{top_k} contains any of {must!r}'
                         if must else 'no BM25 hits')
        return out

    # Pick best: highest BM25 rank, break ties on shortest length
    matched.sort(key=lambda c: (c['rank'], len(c['text'])))
    best = matched[0]
    out['verdict'] = 'PASS'
    out['source_sentence_id'] = best['sid']
    out['source_sentence_text'] = best['text']
    out['bm25_rank'] = best['rank']
    out['expected_answer'] = (must[0] if must else None)
    out['expected_keywords'] = list(must)
    # Also surface alternates (rank 2-3) for human review
    out['alternates'] = [
        {'sid': m['sid'], 'rank': m['rank'], 'text': m['text'][:160]}
        for m in matched[1:4]
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--in', dest='input_path', required=True)
    ap.add_argument('--out', dest='output_path', required=True)
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    ap.add_argument('--top-k', type=int, default=50)
    args = ap.parse_args()

    candidates: list[dict] = []
    with open(args.input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f'Loaded {len(candidates)} candidates from {args.input_path}',
          file=sys.stderr)
    print(f'Opening Whoosh at {args.whoosh_dir}…', file=sys.stderr)
    ix = open_dir(args.whoosh_dir)
    searcher = ix.searcher()
    qparser = QueryParser('text', schema=ix.schema, group=OrGroup)

    n_pass = n_reject = 0
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as out_f:
        for c in candidates:
            r = search_one(searcher, qparser, c, args.top_k)
            out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
            if r['verdict'] == 'PASS':
                n_pass += 1
            else:
                n_reject += 1

    print(f'\nResults: {n_pass} PASS / {n_reject} REJECT '
          f'/ {len(candidates)} total ({100*n_pass/max(1,len(candidates)):.1f}% yield)',
          file=sys.stderr)
    print(f'Wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
