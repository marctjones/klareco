#!/usr/bin/env python3
"""
Add the `verb_klaso` column from the VerbaKlaso ontology (gh#730).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.verb_radiko` (populated)
                 and `ontology_edges` (rel=APARTENAS_AL_VERBA_KLASO, 9 classes)
DEPENDENCIES: duckdb
STAGE: Index / Schema augmentation

Description:
    GitHub issue: #730
    Trivia questions use one verb (`inventis`) but the corpus uses
    synonyms (`kreis`, `eltrovis`, `fondis`, …). The VerbaKlaso ontology
    has 9 classes grouping synonyms; we just don't expose them as a
    queryable column on `sentences`. This script does:

      1. ALTER TABLE sentences ADD COLUMN verb_klaso VARCHAR
      2. UPDATE sentences SET verb_klaso = (SELECT class_id FROM
         ontology_edges WHERE rel = 'APARTENAS_AL_VERBA_KLASO'
         AND radiko = sentences.verb_radiko LIMIT 1)
      3. CREATE INDEX idx_verb_klaso ON sentences(verb_klaso)

    Coverage will be partial: only verbs whose radiko is in the 9
    populated VerbaKlaso classes. Other verbs leave the column NULL.
    Future ontology-population work expands the coverage; this column
    is a join target for the reranker.

Pipeline Position:
    sentences.verb_radiko + ontology_edges → [THIS SCRIPT] →
    sentences.verb_klaso → VerbClassReranker (multi_reranker_bench.py)

Usage:
    python scripts/index/add_verb_klaso_column.py
    python scripts/index/add_verb_klaso_column.py --recompute

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db (read-write)

Outputs:
    Adds + populates + indexes `verb_klaso` column.
    Coverage report by VerbaKlaso class.

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
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()

    print(f'Opening DuckDB at {args.duckdb_path} (read-write)…')
    conn = duckdb.connect(args.duckdb_path)

    n_total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    n_verbs = conn.execute(
        'SELECT COUNT(*) FROM sentences WHERE verb_radiko IS NOT NULL'
    ).fetchone()[0]
    print(f'  sentences total:   {n_total:,}')
    print(f'  with verb_radiko:  {n_verbs:,}  ({100*n_verbs/n_total:.1f}%)')

    # Confirm ontology has the expected data
    n_klaso_edges = conn.execute(
        "SELECT COUNT(*) FROM ontology_edges "
        "WHERE rel = 'APARTENAS_AL_VERBA_KLASO'"
    ).fetchone()[0]
    print(f'  VerbaKlaso edges:  {n_klaso_edges}')
    n_distinct_classes = conn.execute(
        "SELECT COUNT(DISTINCT class_id) FROM ontology_edges "
        "WHERE rel = 'APARTENAS_AL_VERBA_KLASO'"
    ).fetchone()[0]
    print(f'  distinct classes:  {n_distinct_classes}\n')

    have_col = column_exists(conn, 'sentences', 'verb_klaso')
    if not have_col:
        print('ALTER TABLE sentences ADD COLUMN verb_klaso VARCHAR')
        conn.execute('ALTER TABLE sentences ADD COLUMN verb_klaso VARCHAR')

    if not have_col or args.recompute:
        t0 = time.time()
        print('Populating verb_klaso via JOIN against ontology_edges…')
        conn.execute(
            "UPDATE sentences "
            "SET verb_klaso = ("
            "    SELECT class_id FROM ontology_edges "
            "    WHERE rel = 'APARTENAS_AL_VERBA_KLASO' "
            "      AND radiko = sentences.verb_radiko "
            "    LIMIT 1"
            ") "
            "WHERE verb_radiko IS NOT NULL"
        )
        print(f'  done in {time.time()-t0:.1f}s')

    print('CREATE INDEX IF NOT EXISTS idx_verb_klaso ON sentences(verb_klaso)')
    t0 = time.time()
    conn.execute('CREATE INDEX IF NOT EXISTS idx_verb_klaso ON sentences(verb_klaso)')
    print(f'  built in {time.time()-t0:.1f}s')

    # Coverage report
    print('\n=== Coverage by VerbaKlaso ===')
    rows = conn.execute("""
        SELECT verb_klaso, COUNT(*) AS n
        FROM sentences
        WHERE verb_klaso IS NOT NULL
        GROUP BY verb_klaso
        ORDER BY n DESC
    """).fetchall()
    total_covered = sum(r[1] for r in rows)
    for klaso, n in rows:
        print(f'  {klaso:<24s}  {n:>8,}  ({100*n/n_total:.2f}% of corpus)')
    print(f'  {"total covered":<24s}  {total_covered:>8,}  '
          f'({100*total_covered/n_total:.1f}%)')
    n_null = conn.execute(
        'SELECT COUNT(*) FROM sentences WHERE verb_klaso IS NULL AND verb_radiko IS NOT NULL'
    ).fetchone()[0]
    print(f'  {"verb but unmapped":<24s}  {n_null:>8,}  ({100*n_null/n_total:.1f}%)  '
          '(opportunity: add more verb-class edges)')


if __name__ == '__main__':
    main()
