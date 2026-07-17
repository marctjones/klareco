#!/usr/bin/env python3
"""
Validate + accumulate a persistent bank of externally-sourced trivia.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store + Whoosh v2 + the (post-bug-fix) parser
DEPENDENCIES: duckdb, klareco.parser, klareco.orchestrator (when --with-pipeline)
STAGE: Evaluation

Description:
    Earlier generators extracted trivia FROM the corpus, which produced
    systematic failure modes: anchor truncation (first-name only),
    category answers ("komunumo"), and PP-governed false agents. This
    script inverts the flow: trivia is *outsourced* to a curated public
    source (OpenTriviaDB or similar), translated to Esperanto by an LLM,
    then *validated* against our parser and DuckDB store. Validation
    treats translation quality and corpus coverage as separate failure
    modes from generation quality — neither can degrade the inherent
    "is this real trivia?" property of the source.

    Every input batch is appended to a persistent bank file at
    `data/staging/trivia_bank.jsonl`, deduplicated by Esperanto
    question text. So the test set grows monotonically across sessions
    instead of being regenerated each time.

Per-pair validation:
  1. PARSE the Esperanto question. Surface any parser surprises (e.g.
     subject is a `propra_nomo` for `Kio estas X?` — should be `substantivo`
     for the head noun `X`). Flagged pairs are useful as parser regression
     tests even if they don't make the bank.
  2. CORPUS COVERAGE: count occurrences of the Esperanto answer in the
     `sentences` table. If zero or very few, the corpus can't support the
     question — `corpus_gap` verdict. The pair is still saved (it's a
     coverage signal).
  3. (Optional) PIPELINE RUN: send the question through the orchestrator
     and record the retrieval rank for the answer. Adds ~3s per pair.

Pipeline Position:
    OpenTriviaDB → LLM-translated JSONL → [THIS SCRIPT] → trivia_bank.jsonl
                                       → audit + pipeline eval

Usage:
    python scripts/eval/build_trivia_bank.py \\
        --input  /tmp/trivia_batch_2026-05-20.jsonl \\
        --with-pipeline

Input JSONL format (one per line):
    {
      "source":       "opentdb.com",
      "category":     "geography",
      "en_question":  "What is the capital of Brazil?",
      "en_answer":    "Brasília",
      "eo_question":  "Kio estas la ĉefurbo de Brazilo?",
      "eo_answer":    "Brasilio"
    }

Outputs:
    data/staging/trivia_bank.jsonl  — appended/deduped persistent bank
    stdout summary: per-pair verdict + aggregate counts

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from klareco.parser import parse


_BANK_PATH = Path('data/staging/trivia_bank.jsonl')

_INTERROGATIVES = {'Kio', 'Kion', 'Kiu', 'Kiun', 'Kie', 'Kien',
                   'Kiam', 'Kial', 'Kiel', 'Kiom', 'Kies', 'Kia', 'Kian'}


def parse_verify(eo_question: str) -> dict:
    """Parse the Esperanto question and surface key AST features for
    downstream sanity checks. Returns a dict with:
      - 'ok': bool
      - 'subjekto_pv', 'subjekto_vortspeco'
      - 'verbo_pv', 'verbo_vortspeco'
      - 'objekto_pv', 'objekto_vortspeco'
      - 'starts_with_interrogative': bool
      - 'parse_status_summary': str (notes)
    """
    notes: list[str] = []
    try:
        ast = parse(eo_question)
    except Exception as e:
        return {
            'ok': False,
            'parse_status_summary': f'parse exception: {e}',
            'starts_with_interrogative': False,
        }
    if not isinstance(ast, dict):
        return {'ok': False, 'parse_status_summary': 'no AST returned',
                'starts_with_interrogative': False}

    subj = (ast.get('subjekto') or {}).get('kerno') or {}
    verbo = ast.get('verbo') or {}
    obj = (ast.get('objekto') or {}).get('kerno') or {}

    # Accept questions whose first interrogative is at position 0 OR right
    # after a leading preposition / adverbial phrase (`En kiu lando…`,
    # `Al kiu lando…`, `Laŭ la …`, `Per kio…`). The check looks at the
    # first 4 tokens and passes if any of them is a known interrogative.
    leading_tokens = eo_question.split()[:5]
    starts_with_interrogative = any(
        t.rstrip('?,.;:').capitalize() in _INTERROGATIVES
        for t in leading_tokens
    )
    if not starts_with_interrogative:
        notes.append(f'no interrogative in leading 5 tokens: {leading_tokens}')

    # Sanity: if the question starts with `Kio estas X?` we expect the
    # subject to be a substantivo/propra_nomo (the entity X), not a
    # function-word or correlative. Surfacing surprises here gives us
    # parser regression candidates.
    s_vs = subj.get('vortspeco')
    if starts_with_interrogative and s_vs not in (
        'substantivo', 'propra_nomo', 'pronomo', 'korelativo', 'nekonata', None
    ):
        notes.append(f'subjekto.vortspeco surprising: {s_vs!r}')

    return {
        'ok': True,
        'subjekto_pv':         subj.get('plena_vorto'),
        'subjekto_vortspeco':  s_vs,
        'verbo_pv':            verbo.get('plena_vorto'),
        'verbo_vortspeco':     verbo.get('vortspeco'),
        'objekto_pv':          obj.get('plena_vorto'),
        'objekto_vortspeco':   obj.get('vortspeco'),
        'starts_with_interrogative': starts_with_interrogative,
        'parse_status_summary':      '; '.join(notes) or 'ok',
    }


def corpus_coverage(conn, eo_answer: str, eo_question: str) -> dict:
    """Check whether the answer appears in the corpus, and gather a sample.
    Also pulls top key-token occurrence counts to estimate retrieval signal.
    """
    answer_hits = conn.execute(
        "SELECT COUNT(*) FROM sentences WHERE text LIKE ?",
        [f'%{eo_answer}%']
    ).fetchone()[0]
    sample_text, sample_sid = None, None
    if answer_hits > 0:
        row = conn.execute(
            "SELECT sid, text FROM sentences WHERE text LIKE ? "
            "AND length(text) BETWEEN 25 AND 250 LIMIT 1",
            [f'%{eo_answer}%']
        ).fetchone()
        if row:
            sample_sid, sample_text = row[0], row[1]
    # Look for key non-stopword tokens from the question
    import re as _re
    q_tokens = _re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]{4,}", eo_question)
    constraint_hits = 0
    if q_tokens:
        # Take the longest token (probably the most specific) — a
        # rough but effective proxy for the question's discriminative term
        longest = sorted(q_tokens, key=len, reverse=True)[0]
        if longest.lower() not in ('estas', 'kiel', 'kiam'):
            constraint_hits = conn.execute(
                "SELECT COUNT(*) FROM sentences WHERE text LIKE ?",
                [f'%{longest}%']
            ).fetchone()[0]
    return {
        'answer_hits':      answer_hits,
        'constraint_hits':  constraint_hits,
        'sample_sentence_id':   sample_sid,
        'sample_sentence_text': sample_text,
    }


def classify_verdict(parse_info: dict, coverage: dict) -> str:
    """Single-word verdict for the pair."""
    if not parse_info.get('ok'):
        return 'unparseable_question'
    if 'surprising' in (parse_info.get('parse_status_summary') or ''):
        return 'parser_anomaly'
    if not parse_info.get('starts_with_interrogative'):
        return 'malformed_question'
    if coverage['answer_hits'] == 0:
        return 'corpus_gap'
    if coverage['answer_hits'] < 3:
        return 'sparse_corpus'
    return 'measurable'


def run_pipeline(pipeline, eo_question: str, eo_answer: str) -> dict:
    """Run one question through the orchestrator and capture retrieval rank."""
    from klareco.eval import evaluate_question
    entry = {
        'id': eo_answer,
        'question': eo_question,
        'expected_keywords': [eo_answer],
        'question_type': 'KIU',
    }
    r = evaluate_question(pipeline, entry)
    return {
        'answer_correct':       bool(r.get('answer_correct')),
        'first_relevant_rank':  r.get('first_relevant_rank'),
        'pipeline_answer':      (r.get('answer') or '')[:180],
        'latency_sec':          r.get('latency_sec'),
    }


def load_existing_bank() -> dict:
    if not _BANK_PATH.exists():
        return {}
    by_q = {}
    with open(_BANK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_q[d.get('eo_question')] = d
    return by_q


def append_to_bank(records: list[dict]) -> None:
    _BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BANK_PATH, 'a') as f:
        for d in records:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',         required=True,
                    help='JSONL with translated trivia candidates.')
    ap.add_argument('--duckdb-path',   default='data/indexes/duckdb_store.db')
    ap.add_argument('--whoosh-dir',    default='data/indexes/whoosh_v2')
    ap.add_argument('--with-pipeline', action='store_true',
                    help='Run each question through the orchestrator '
                         '(adds ~3s per pair).')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f'ERROR: input {in_path} not found', file=sys.stderr)
        sys.exit(1)

    candidates: list[dict] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f'Loaded {len(candidates)} candidate(s) from {in_path}')

    existing = load_existing_bank()
    print(f'Existing bank size: {len(existing)}')

    conn = duckdb.connect(args.duckdb_path, read_only=True)
    pipeline = None
    if args.with_pipeline:
        print('Building pipeline…')
        from klareco.orchestrator import build_default_pipeline
        pipeline = build_default_pipeline(
            whoosh_index_dir=args.whoosh_dir, top_k=10,
        )

    next_id = len(existing) + 1
    accepted: list[dict] = []
    skipped_dupes = 0

    for c in candidates:
        eo_q = c.get('eo_question')
        eo_a = c.get('eo_answer')
        if not eo_q or not eo_a:
            print(f'  SKIP: missing eo_question / eo_answer in {c}')
            continue
        if eo_q in existing:
            skipped_dupes += 1
            continue

        parse_info = parse_verify(eo_q)
        coverage = corpus_coverage(conn, eo_a, eo_q)
        verdict = classify_verdict(parse_info, coverage)
        pipeline_result = None
        if pipeline is not None:
            pipeline_result = run_pipeline(pipeline, eo_q, eo_a)

        record = {
            'id':              f'tb_{next_id:04d}',
            'fetched_at':      time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source':          c.get('source', 'opentdb.com'),
            'category':        c.get('category'),
            'en_question':     c.get('en_question'),
            'en_answer':       c.get('en_answer'),
            'eo_question':     eo_q,
            'eo_answer':       eo_a,
            'expected_keywords': [eo_a],
            'question_type':   c.get('question_type', 'KIU'),
            'translator':      c.get('translator', 'claude-opus-4-7'),
            'parse_verified':  parse_info,
            'corpus_coverage': coverage,
            'verdict':         verdict,
            'pipeline_result': pipeline_result,
        }
        accepted.append(record)
        next_id += 1

    append_to_bank(accepted)

    # Summary
    print(f'\n=== Summary ===')
    print(f'Saved: {len(accepted)}  (skipped {skipped_dupes} duplicates)')
    from collections import Counter
    verdict_counts = Counter(r['verdict'] for r in accepted)
    for v, n in sorted(verdict_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {v:<24s} {n:>3}')

    if pipeline is not None:
        print(f'\n=== Pipeline rank distribution (measurable + sparse) ===')
        rank_counts = Counter()
        for r in accepted:
            pr = r.get('pipeline_result') or {}
            rank = pr.get('first_relevant_rank')
            bucket = (str(rank) if rank in (1, 2, 3)
                      else '4-10' if rank and 4 <= rank <= 10
                      else '11+' if rank else 'none')
            rank_counts[bucket] += 1
        for b in ('1', '2', '3', '4-10', '11+', 'none'):
            print(f'  rank {b:<6s} {rank_counts.get(b, 0):>3}')

    print(f'\nBank now at: {len(existing) + len(accepted)} pairs ({_BANK_PATH})')


if __name__ == '__main__':
    main()
