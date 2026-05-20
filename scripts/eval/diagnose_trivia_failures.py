#!/usr/bin/env python3
"""
Diagnose trivia bank pipeline failures: for each `measurable` pair where the
pipeline failed, find whether there is a joint-coverage sentence (one that
contains BOTH the answer and key question terms), then look at the actual
top-10 retrieval and the extractor's output.

VERSION: v2.x
COMPATIBLE WITH: data/test_sets/trivia_bank.jsonl
DEPENDENCIES: duckdb, klareco.orchestrator
STAGE: Evaluation

Usage:
    python scripts/eval/diagnose_trivia_failures.py [--limit 5]

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from klareco.orchestrator import build_default_pipeline


_BANK = Path('data/test_sets/trivia_bank.jsonl')
_STOP = {
    'estas', 'estis', 'estos', 'la', 'de', 'en', 'al', 'el', 'kun', 'pri',
    'pro', 'por', 'sub', 'kio', 'kiu', 'kiun', 'kion', 'kiel', 'kiam',
    'kial', 'kie', 'kiom', 'kia', 'kies', 'ke', 'ne', 'aŭ', 'do', 'kaj',
    'sed', 'ankaŭ', 'ĉu', 'jen', 'tio', 'tiu', 'ĉi',
}


def question_keywords(q: str) -> list[str]:
    """Pull content tokens of length ≥4 that aren't in our stoplist."""
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q)
    return [t for t in toks if t.lower() not in _STOP and len(t) >= 4]


def find_joint_coverage_sentence(conn, answer: str, q_keywords: list[str]):
    """Find a sentence containing BOTH the answer AND at least one strong
    question keyword. Returns (sid, text, matched_keyword) or None.

    Iterates keywords by descending length (more specific first)."""
    by_specificity = sorted(set(q_keywords), key=len, reverse=True)
    for kw in by_specificity[:6]:
        row = conn.execute(
            "SELECT sid, text FROM sentences "
            "WHERE text LIKE ? AND text LIKE ? "
            "AND length(text) BETWEEN 30 AND 400 "
            "LIMIT 1",
            [f'%{answer}%', f'%{kw}%']
        ).fetchone()
        if row:
            return (row[0], row[1], kw)
    return None


def get_top_passages(pipeline, question: str, top_k: int = 10):
    """Run the orchestrator and pull retrieved passages from the retrieve
    stage's delta. The retriever stores results under
    delta.symbolic['passage_asts']. Returns list of
    (rank, sid, text-prefix, score)."""
    result = pipeline.answer(question)
    passages: list[tuple] = []
    for st in result.trace:
        if st.stage_name != 'retrieve' or st.delta is None:
            continue
        sym = getattr(st.delta, 'symbolic', None)
        cand = sym.get('passage_asts') if isinstance(sym, dict) else None
        for i, p in enumerate(cand or [], 1):
            if i > top_k:
                break
            sid = getattr(p, 'sentence_id', None)
            text = getattr(p, 'text', '') or ''
            score = getattr(p, 'score', None)
            passages.append((i, sid, text[:160], score))
        break
    return passages, result.text or ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    args = ap.parse_args()

    bank = []
    with open(_BANK) as f:
        for line in f:
            if line.strip():
                bank.append(json.loads(line))
    # Pick measurable failures (rank != 1, answer in corpus)
    targets = []
    for r in bank:
        if r['verdict'] not in ('measurable', 'sparse_corpus'):
            continue
        pr = r.get('pipeline_result') or {}
        if pr.get('first_relevant_rank') == 1 and pr.get('answer_correct'):
            continue
        targets.append(r)
    targets = targets[:args.limit]
    print(f'Diagnosing {len(targets)} failures (limit={args.limit})\n')

    conn = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
    pipeline = build_default_pipeline(whoosh_index_dir=args.whoosh_dir, top_k=10)

    for r in targets:
        q = r['eo_question']
        a = r['eo_answer']
        print('=' * 78)
        print(f'Q: {q}')
        print(f'A (expected): {a}')
        print(f'Answer-hits in corpus: {r["corpus_coverage"]["answer_hits"]}')

        kws = question_keywords(q)
        print(f'Question keywords: {kws}')

        joint = find_joint_coverage_sentence(conn, a, kws)
        if joint:
            sid, text, kw = joint
            print(f'\n  ✓ JOINT-COVERAGE SENTENCE EXISTS (matched on keyword {kw!r}):')
            print(f'    sid={sid}: {text[:300]}')
        else:
            print(f'\n  ✗ NO joint-coverage sentence found')
            print(f'    (corpus has {r["corpus_coverage"]["answer_hits"]} sentences with')
            print(f'     answer {a!r}, but none also contain any question keyword)')

        # Now run the pipeline and show top passages
        passages, final = get_top_passages(pipeline, q)
        print(f'\n  Pipeline top-{len(passages)}:')
        for rank, sid, text, score in passages:
            mark = ' ★' if a.lower() in (text or '').lower() else '  '
            print(f'    {mark}#{rank:>2}  sid={sid}  score={score}  {text}')
        if joint and not any(joint[0] == sid for _, sid, _, _ in passages):
            print(f'  ✗ The ideal joint-coverage sentence (sid={joint[0]}) is NOT in top-10')
        print(f'\n  Pipeline final answer: {final[:200]}\n')


if __name__ == '__main__':
    main()
