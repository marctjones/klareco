#!/usr/bin/env python3
"""
Extractor-isolation test: run the extract+generate stage against curated
passages to separate extractor failures from retriever failures.

VERSION: v2.x
COMPATIBLE WITH: post-bug-#1/#2/#4 parser; trivia_bank.jsonl with
                 joint_coverage_keyword tag
DEPENDENCIES: duckdb, klareco.parser, klareco.orchestrator
STAGE: Evaluation

Description:
    For each trivia-bank pair, run two variants:
      A. ORACLE   — feed the extractor JUST the joint-coverage sentence
                    (the perfect passage). Tests extractor's upper bound.
      B. MIXED    — feed the extractor BM25's top-10 PLUS the joint-coverage
                    sentence forced in at rank 5. Tests whether the extractor
                    can pick the right passage from a noisy candidate pool.

    The diff between A and B is the extractor's "passage selection" failure
    rate. If A succeeds but B fails: the extractor blindly trusts rank-1.
    If A fails: the extractor itself is broken at the span-extraction level.

Usage:
    python scripts/eval/extractor_oracle_test.py

Inputs:
    data/test_sets/trivia_bank.jsonl  (must have joint_coverage_keyword)
    data/indexes/duckdb_store.db
    data/indexes/whoosh_v2

Outputs:
    Stdout per-pair table + aggregate.
    JSONL: data/test_sets/extractor_oracle_results.jsonl

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from klareco.parser import parse
from klareco.orchestrator import build_default_pipeline
from klareco.orchestrator.context import QueryContext, ContextDelta, ParsedPassage
from klareco.orchestrator.stages.parse_question import ParseQuestionStage
from klareco.orchestrator.stages.extract_generate import ExtractAndGenerateStage
from klareco.orchestrator.stages.format_output import FormatOutputStage
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator


_BANK = Path('data/test_sets/trivia_bank.jsonl')
_RESULTS = Path('data/test_sets/extractor_oracle_results.jsonl')


def find_joint_coverage(conn, answer: str, question: str
                        ) -> tuple[int, str] | None:
    """Re-find the joint-coverage sentence (same algorithm as build_trivia_bank)."""
    _STOP = {'estas', 'estis', 'estos', 'la', 'de', 'en', 'al', 'el', 'kun',
             'pri', 'pro', 'por', 'sub', 'kio', 'kiu', 'kiun', 'kion',
             'kiel', 'kiam', 'kial', 'kie', 'kiom', 'kia', 'kies', 'ke',
             'ne', 'aŭ', 'do', 'kaj', 'sed', 'ankaŭ', 'ĉu', 'jen', 'tio',
             'tiu', 'la', 'mondo', 'jaro'}
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", question)
    kws = sorted(
        {t for t in toks if t.lower() not in _STOP and len(t) >= 4},
        key=len, reverse=True
    )[:6]
    for kw in kws:
        row = conn.execute(
            "SELECT sid, text FROM sentences WHERE text LIKE ? AND text LIKE ? "
            "AND length(text) BETWEEN 30 AND 400 LIMIT 1",
            [f'%{answer}%', f'%{kw}%']
        ).fetchone()
        if row:
            return (row[0], row[1])
    return None


def build_passage(sid: int, text: str, score: float = 10.0) -> ParsedPassage:
    """Wrap a sentence in the ParsedPassage shape the extractor expects."""
    try:
        ast = parse(text)
    except Exception:
        ast = None
    return ParsedPassage(
        sentence_id=str(sid),
        text=text,
        ast=ast,
        score=score,
        source_doc='oracle',
        source_type='oracle',
    )


def run_extractor_on_passages(question: str, passages: tuple[ParsedPassage, ...]
                              ) -> tuple[str, bool, list]:
    """Run only the parse → extract → format stages over given passages.
    Skips the retrieve and deterministic-rerank stages entirely. Returns
    (final_text, answer_correct (always False here — we'll check externally),
     trace_summary).
    """
    parse_stage = ParseQuestionStage()
    generator = ExtractiveAnswerGenerator()
    extract_stage = ExtractAndGenerateStage(generator=generator)
    format_stage = FormatOutputStage()

    ctx = QueryContext(question=question)
    delta = parse_stage.run(ctx)
    ctx = ctx.apply(delta)

    # Inject our passages as if the retriever had returned them
    ctx = ctx.apply(ContextDelta(symbolic={'passage_asts': passages}))

    delta = extract_stage.run(ctx)
    ctx = ctx.apply(delta)

    delta = format_stage.run(ctx)
    ctx = ctx.apply(delta)

    final = ctx.symbolic.final_text or ''
    return final, [type(s).__name__ for s in (parse_stage, extract_stage, format_stage)]


def get_bm25_top_k(pipeline, question: str, top_k: int = 10
                   ) -> tuple[ParsedPassage, ...]:
    """Run BM25 retrieval only, return top-K passages."""
    parse_stage = ParseQuestionStage()
    ctx = QueryContext(question=question)
    delta = parse_stage.run(ctx)
    ctx = ctx.apply(delta)
    # Find the retrieve stage in the pipeline and run it
    for stage in pipeline.stages:
        if stage.name == 'retrieve':
            delta = stage.run(ctx)
            ctx = ctx.apply(delta)
            break
    return ctx.symbolic.passage_asts or ()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    ap.add_argument('--inject-rank', type=int, default=5,
                    help='In MIXED mode, position to inject oracle (1-indexed).')
    args = ap.parse_args()

    bank = []
    with open(_BANK) as f:
        for line in f:
            if line.strip():
                bank.append(json.loads(line))
    # Only run on pairs we know have joint coverage
    bank = [r for r in bank if r.get('joint_coverage_keyword') is not None]
    print(f'Testing {len(bank)} pairs with joint coverage\n')

    conn = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
    pipeline = build_default_pipeline(whoosh_index_dir=args.whoosh_dir, top_k=10)

    results = []
    for r in bank:
        q = r['eo_question']
        a = r['eo_answer']
        joint = find_joint_coverage(conn, a, q)
        if not joint:
            continue
        oracle_sid, oracle_text = joint
        oracle_passage = build_passage(oracle_sid, oracle_text, score=15.0)

        # --- A. ORACLE ---
        a_text, _ = run_extractor_on_passages(q, (oracle_passage,))
        a_correct = a.lower() in a_text.lower()

        # --- B. MIXED ---
        bm25_passages = list(get_bm25_top_k(pipeline, q, top_k=10))
        # Inject oracle at rank N (1-indexed), removing oracle if it's already there
        bm25_passages = [p for p in bm25_passages
                         if str(p.sentence_id) != str(oracle_sid)]
        insert_pos = min(args.inject_rank - 1, len(bm25_passages))
        # Give the oracle a competitive score (above the rank-3 score)
        if len(bm25_passages) >= insert_pos:
            target_score = (bm25_passages[insert_pos - 1].score
                            if insert_pos >= 1 and insert_pos <= len(bm25_passages)
                            else 5.0)
            oracle_with_score = ParsedPassage(
                sentence_id=str(oracle_sid),
                text=oracle_text,
                ast=oracle_passage.ast,
                score=target_score,  # match its injection rank
                source_doc='oracle-injected',
                source_type='oracle-injected',
            )
            bm25_passages.insert(insert_pos, oracle_with_score)
        b_text, _ = run_extractor_on_passages(q, tuple(bm25_passages))
        b_correct = a.lower() in b_text.lower()

        results.append({
            'id': r['id'],
            'question':       q,
            'expected_answer': a,
            'oracle_sid':      oracle_sid,
            'oracle_correct':  a_correct,
            'mixed_correct':   b_correct,
            'oracle_text_snippet':  oracle_text[:120],
            'oracle_answer_text':   a_text[:160],
            'mixed_answer_text':    b_text[:160],
        })

    # Aggregate
    n = len(results)
    n_a = sum(1 for r in results if r['oracle_correct'])
    n_b = sum(1 for r in results if r['mixed_correct'])
    n_both = sum(1 for r in results if r['oracle_correct'] and r['mixed_correct'])
    n_a_only = sum(1 for r in results if r['oracle_correct'] and not r['mixed_correct'])

    print(f'\n=== Aggregate ({n} pairs tested) ===\n')
    print(f'  A. ORACLE-only success:       {n_a}/{n}  ({100*n_a/n:.0f}%)' if n else '')
    print(f'  B. MIXED  (oracle at rank {args.inject_rank}) success: {n_b}/{n}  ({100*n_b/n:.0f}%)' if n else '')
    print(f'  BOTH succeed:                 {n_both}/{n}')
    print(f'  A succeed but B fail (extractor confused by distractors): {n_a_only}/{n}')

    print(f'\n=== Per-pair ===')
    print(f'{"id":<10s} {"A":<3s} {"B":<3s} {"Q (truncated)":<60s} expected')
    print('-' * 100)
    for r in results:
        a_mark = '✓' if r['oracle_correct'] else '✗'
        b_mark = '✓' if r['mixed_correct'] else '✗'
        print(f'{r["id"]:<10s} {a_mark:<3s} {b_mark:<3s} {r["question"][:60]:<60s} {r["expected_answer"][:30]}')

    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\nFull per-pair output saved to {_RESULTS}')


if __name__ == '__main__':
    main()
