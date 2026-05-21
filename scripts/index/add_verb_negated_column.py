#!/usr/bin/env python3
"""
Add the `verb_negated` column derived from AST verbo.negita (gh#731).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.ast_json` carrying the
                 frazo-level AST, post-parser-fix
DEPENDENCIES: duckdb
STAGE: Index / Schema augmentation

Description:
    GitHub issue: #731
    The parser tags negation as `verbo.negita = true` when a `ne` or
    `neniam` immediately precedes the verb. BM25 can't distinguish
    positive from negated assertions; this column materialises that
    flag so retrieval / reranking can filter or boost by polarity.

    Two-step build:
      1. ALTER TABLE sentences ADD COLUMN verb_negated BOOLEAN;
      2. UPDATE sentences SET verb_negated =
            json_extract_string(ast_json, '$.verbo.negita') = 'true';
      3. CREATE INDEX idx_verb_negated ON sentences(verb_negated);

    Closes the Johann-Sebastian-Bach class of trivia failures, where a
    question matched a negated source sentence and got a wrong answer.

Pipeline Position:
    sentences.ast_json → [THIS SCRIPT] → sentences.verb_negated
    → NegationAwareReranker (multi_reranker_bench.py)

Usage:
    python scripts/index/add_verb_negated_column.py
    python scripts/index/add_verb_negated_column.py --recompute  # force UPDATE

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db (read-write)

Outputs:
    Adds `verb_negated` column, populates it from AST, indexes it.
    Reports total / negated / null counts.

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


def column_exists(conn, table: str, col: str) -> bool:
    rows = conn.execute(
        f"SELECT column_name FROM information_schema.columns "
        f"WHERE table_name = '{table}'"
    ).fetchall()
    return any(r[0] == col for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--recompute', action='store_true',
                    help='Re-run the UPDATE even if column already exists.')
    args = ap.parse_args()

    print(f'Opening DuckDB at {args.duckdb_path} (read-write)…')
    conn = duckdb.connect(args.duckdb_path)

    n_total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'  sentences row count: {n_total:,}\n')

    have_col = column_exists(conn, 'sentences', 'verb_negated')
    if not have_col:
        print('ALTER TABLE sentences ADD COLUMN verb_negated BOOLEAN')
        conn.execute('ALTER TABLE sentences ADD COLUMN verb_negated BOOLEAN')
    elif not args.recompute:
        print('Column `verb_negated` already exists; skipping UPDATE '
              '(pass --recompute to override).')

    if not have_col or args.recompute:
        t0 = time.time()
        print('Populating verb_negated from ast_json…')
        # DuckDB's json_extract_string returns the string 'true'/'false' or NULL
        conn.execute(
            "UPDATE sentences "
            "SET verb_negated = ("
            "    json_extract_string(ast_json, '$.verbo.negita') = 'true'"
            ")"
        )
        elapsed = time.time() - t0
        print(f'  done in {elapsed:.1f}s')

    print('CREATE INDEX IF NOT EXISTS idx_verb_negated ON sentences(verb_negated)')
    t0 = time.time()
    conn.execute('CREATE INDEX IF NOT EXISTS idx_verb_negated ON sentences(verb_negated)')
    print(f'  built in {time.time()-t0:.1f}s')

    # Stats
    print('\n=== Coverage ===')
    n_true = conn.execute(
        'SELECT COUNT(*) FROM sentences WHERE verb_negated = TRUE'
    ).fetchone()[0]
    n_false = conn.execute(
        'SELECT COUNT(*) FROM sentences WHERE verb_negated = FALSE'
    ).fetchone()[0]
    n_null = conn.execute(
        'SELECT COUNT(*) FROM sentences WHERE verb_negated IS NULL'
    ).fetchone()[0]
    print(f'  verb_negated = TRUE:   {n_true:>8,}  ({100*n_true/n_total:.2f}%)')
    print(f'  verb_negated = FALSE:  {n_false:>8,}  ({100*n_false/n_total:.2f}%)')
    print(f'  verb_negated IS NULL:  {n_null:>8,}  ({100*n_null/n_total:.2f}%)  '
          '(sentences with no verbo or no negita flag)')


if __name__ == '__main__':
    main()
