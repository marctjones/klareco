#!/usr/bin/env python3
"""
Search the corpus for a candidate answer sentence given a trivia question.

VERSION: v2.x
COMPATIBLE WITH: any test-set generation workflow that needs to verify
                 a corpus sentence supports an Esperanto trivia question
DEPENDENCIES: whoosh, duckdb, klareco.parser
STAGE: Evaluation (test-set generation)

Description:
    Drafting trivia is cheap; verifying that the answer lives in our
    corpus is the bottleneck. This helper:

      1. Parses the question and reports parse status (fail-fast on
         broken Esperanto).
      2. Runs BM25 over the question's content terms.
      3. Optionally filters by `--must-contain WORD` — e.g. the expected
         answer entity — so only sentences mentioning it surface.
      4. Prints top-K hits with sid + sentence text for human judgement.

    Use this iteratively: draft → search → if no good answer surfaces,
    pick a different question.

Pipeline Position:
    candidate question → [THIS SCRIPT] → top-K candidates →
                         (judge inline; save pair if any match) → JSONL

Usage:
    python scripts/eval/search_candidate_answer.py \\
        --question "Kio estas la ĉefurbo de Aŭstralio?" \\
        --must-contain Kanbero

    # Take multiple --must-contain to OR them
    python scripts/eval/search_candidate_answer.py \\
        --question "Kiu inventis la telefonon?" \\
        --must-contain Bell --must-contain Aleksandro

Inputs:
    --question        the Esperanto trivia question (required)
    --must-contain    a token that must appear in the candidate (repeatable)
    --top-k           how many BM25 hits to scan (default 50)
    --max-show        how many candidates to print (default 10)

Outputs:
    Stdout: parse summary + top-K matching candidate sentences with sids.

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--question', required=True)
    ap.add_argument('--must-contain', action='append', default=[],
                    help='Repeat for multiple required tokens (OR semantics).')
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--max-show', type=int, default=10)
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    args = ap.parse_args()

    # 1. Parse the question
    try:
        ast = parse(args.question)
    except Exception as e:
        print(f'PARSE ERROR: {e}')
        sys.exit(2)

    bad_words = []
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
    for w in _walk(ast):
        st = w.get('analizstato')
        if st and st != 'sukceso' and st != 'propra_nomo_nekonata':
            bad_words.append(f'{w.get("plena_vorto")}({st})')
    parse_status = 'CLEAN' if not bad_words else f'NOTE: {", ".join(bad_words)}'
    print(f'Parse: {parse_status}')

    # 2. BM25 search
    print(f'Opening Whoosh index at {args.whoosh_dir}…')
    ix = open_dir(args.whoosh_dir)
    searcher = ix.searcher()
    qparser = QueryParser('text', schema=ix.schema, group=OrGroup)
    terms = _content_terms(args.question)
    print(f'BM25 terms: {terms}')
    query = qparser.parse(' '.join(terms))
    results = searcher.search(query, limit=args.top_k)

    folded_required = [_fold(s) for s in args.must_contain]

    hits = []
    for rank, hit in enumerate(results, 1):
        sid = hit.get('sid') or hit.get('id') or hit.get('docid')
        text = hit.get('text') or hit.get('content') or ''
        if folded_required:
            folded_text = _fold(text)
            if not any(req and req in folded_text for req in folded_required):
                continue
        hits.append({'rank': rank, 'sid': sid, 'text': text})
        if len(hits) >= args.max_show:
            break

    if not hits:
        if folded_required:
            print(f'\nNo top-{args.top_k} hit contains any of '
                  f'{args.must_contain!r}. Either the corpus lacks the '
                  f'fact, or the question terms don\'t match the relevant '
                  f'passage. Try a different question.')
        else:
            print(f'\nBM25 returned zero hits. Question may be too obscure '
                  f'for the corpus.')
        sys.exit(1)

    print(f'\n=== Top {len(hits)} candidate(s) ===')
    for h in hits:
        print(f'\n[rank {h["rank"]}] sid={h["sid"]}')
        print(f'  {h["text"][:240]}')

    print(f'\nSearched {len(results)} BM25 hits; {len(hits)} matched filters.')


if __name__ == '__main__':
    main()
