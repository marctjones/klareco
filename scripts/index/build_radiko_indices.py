#!/usr/bin/env python3
"""
Build ART indices on the shredded radiko columns (gh#728).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences` table carrying
                 subj_radiko, verb_radiko, obj_radiko columns
DEPENDENCIES: duckdb
STAGE: Index

Description:
    GitHub issue: #728
    Three CREATE INDEX statements that give the sentences table
    sub-millisecond point lookup by structural role:
        SELECT * FROM sentences WHERE subj_radiko = 'zamenhof'

    This unlocks structural retrieval as a peer of BM25 retrieval —
    the foundation for the AST-based reranker / structured-retriever
    architecture in #576 and the multi-reranker bench (#733).

    Cheapest of the deterministic-indexing issues; the shredded
    columns are already populated by the corpus build + refresh, so
    this is purely an index addition.

Pipeline Position:
    sentences table (post AST-refresh) → [THIS SCRIPT] → indexed columns
    → AST-aware reranker / structured retriever

Usage:
    python scripts/index/build_radiko_indices.py
    python scripts/index/build_radiko_indices.py --drop-first  # rebuild

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db (read-write)

Outputs:
    Creates idx_subj_radiko, idx_verb_radiko, idx_obj_radiko on `sentences`.
    Reports row count and a quick benchmark of point-lookup latency.

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


INDICES = [
    ('idx_subj_radiko', 'subj_radiko'),
    ('idx_verb_radiko', 'verb_radiko'),
    ('idx_obj_radiko',  'obj_radiko'),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--drop-first', action='store_true',
                    help='DROP each index first (forces rebuild).')
    args = ap.parse_args()

    print(f'Opening DuckDB at {args.duckdb_path} (read-write)…')
    conn = duckdb.connect(args.duckdb_path)

    row_count = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'  sentences row count: {row_count:,}\n')

    if args.drop_first:
        for index_name, _ in INDICES:
            print(f'DROP INDEX IF EXISTS {index_name}')
            conn.execute(f'DROP INDEX IF EXISTS {index_name}')

    for index_name, col in INDICES:
        t0 = time.time()
        print(f'CREATE INDEX IF NOT EXISTS {index_name} ON sentences({col})')
        conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON sentences({col})')
        elapsed = time.time() - t0
        print(f'  built in {elapsed:.1f}s')

    # Quick benchmark: point-lookup latency on each indexed column
    print('\n=== Point-lookup benchmark ===')
    for index_name, col in INDICES:
        # Pick a representative value
        row = conn.execute(
            f'SELECT {col} FROM sentences WHERE {col} IS NOT NULL LIMIT 1'
        ).fetchone()
        if not row:
            print(f'  {col}: column entirely null, cannot benchmark')
            continue
        sample_value = row[0]
        t0 = time.time()
        n = conn.execute(
            f'SELECT COUNT(*) FROM sentences WHERE {col} = ?', [sample_value]
        ).fetchone()[0]
        elapsed_ms = (time.time() - t0) * 1000
        print(f'  WHERE {col}={sample_value!r:<30s}  hits={n:>6,}  {elapsed_ms:>6.2f} ms')

    print('\n=== Done ===')
    print('Indices are persistent in the DuckDB file. Subsequent CREATE INDEX')
    print('IF NOT EXISTS calls are no-ops.')


if __name__ == '__main__':
    main()
