#!/usr/bin/env python3
"""
Add cheap derived boolean columns for KIE/KIAM/KIOM filtering.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.aliaj_json` populated.
DEPENDENCIES: duckdb
STAGE: Index / Schema augmentation

Description:
    GitHub issue: #741 (Stage 2)
    The AST-aware reranker uses three structural filters that previously
    walked `aliaj_json` on the fly per candidate per query:
      - aliaj_has_loko    (KIE filter)
      - aliaj_has_jaro    (KIAM filter)
      - aliaj_has_kvant   (KIOM filter)

    JSON parsing is slow and prone to false negatives when the place /
    year / quantity lives in an unusual aliaj position. Materializing
    these as boolean columns at index time:
      - eliminates per-query JSON parsing cost
      - gives a stable, indexable signal we can refine over time
      - lets the reranker fall back to JSON parsing only when the
        boolean column is NULL (i.e., aliaj_json wasn't analyzable)

    This script does:
      1. ALTER TABLE sentences ADD COLUMN aliaj_has_loko BOOLEAN,
                                          aliaj_has_jaro BOOLEAN,
                                          aliaj_has_kvant BOOLEAN
      2. Stream every (sid, aliaj_json) row, compute the three flags,
         buffer into a staging table, then bulk-UPDATE.
      3. CREATE INDEX on each new column.

Pipeline Position:
    sentences.aliaj_json → [THIS SCRIPT] → sentences.aliaj_has_*
                                          → ASTAwareReranker

Usage:
    python scripts/index/add_aliaj_flag_columns.py
    python scripts/index/add_aliaj_flag_columns.py --recompute

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db (read-write)

Outputs:
    Adds + populates + indexes three boolean columns.
    Coverage report: how many sentences have each flag = TRUE.

Last Updated: 2026-05-22
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

# Reuse the same detection logic as the reranker so behavior is identical.
from klareco.rag.ast_aware_reranker import (
    aliaj_has_loko, aliaj_has_jaro, aliaj_has_numeral,
)


DB = 'data/indexes/duckdb_store.db'
BATCH = 50_000


#
# ⚠️ DISK-SPACE LESSON LEARNED — see EPIC #746
#
# The pattern below — ALTER TABLE ADD COLUMN + UPDATE FROM + CREATE INDEX
# in place on a 5.4M-row table — left ~30 GB of dead pages in the DuckDB
# store when this script ran in May 2026. DuckDB does NOT auto-vacuum.
#
# The preferred pattern for any future column-add is "new-table-swap":
#
#   CREATE TABLE sentences_new AS
#     SELECT s.*, f.has_loko, f.has_jaro, f.has_kvant
#     FROM sentences s LEFT JOIN _aliaj_flags f ON s.sid = f.sid;
#   -- verify count match
#   DROP TABLE sentences;
#   ALTER TABLE sentences_new RENAME TO sentences;
#   -- re-create indexes
#
# This is atomic, leaves no dead pages, and is easy to verify by row count
# before the DROP. The complication is that CREATE TABLE AS does NOT
# preserve constraints (PRIMARY KEY) or existing indexes — they must be
# explicitly re-created from a captured DDL snapshot before the swap.
#
# Until this script (and the other in-place-ALTER scripts in
# scripts/index/) are rewritten to use new-table-swap, run
# `python scripts/util/compact_duckdb.py --apply` after invoking them to
# reclaim the dead pages.
#

def add_columns(conn) -> None:
    """Add the three boolean columns if they don't exist."""
    for col in ('aliaj_has_loko', 'aliaj_has_jaro', 'aliaj_has_kvant'):
        try:
            conn.execute(
                f'ALTER TABLE sentences ADD COLUMN {col} BOOLEAN')
            print(f'  added column: {col}', flush=True)
        except duckdb.Error as e:
            if 'already exists' in str(e):
                print(f'  column {col} already exists', flush=True)
            else:
                raise


def compute_and_update(conn, recompute: bool = False) -> None:
    """Stream rows, compute flags, bulk-update."""
    if not recompute:
        n_done = conn.execute(
            "SELECT COUNT(*) FROM sentences "
            "WHERE aliaj_has_loko IS NOT NULL"
        ).fetchone()[0]
        if n_done > 0:
            print(f'  {n_done:,} sentences already have flags computed; '
                  f'rerun with --recompute to redo', flush=True)
            return

    total = conn.execute(
        "SELECT COUNT(*) FROM sentences").fetchone()[0]
    print(f'  total sentences: {total:,}', flush=True)

    # Stream rows from a read-only cursor; collect updates into staging.
    # Use a temp table so we can do one bulk UPDATE at the end.
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _aliaj_flags (
          sid INTEGER PRIMARY KEY,
          has_loko BOOLEAN,
          has_jaro BOOLEAN,
          has_kvant BOOLEAN
        )
    """)

    # Strategy: SELECT all (sid, aliaj_json) once into Python memory, then
    # bulk-insert flags into the temp table. Interleaving SELECT cursor
    # iteration with executemany INSERTs on the same connection invalidates
    # the cursor (cause of the first crash). Memory: 5.4M rows * ~100 bytes
    # aliaj_json average ≈ <1 GB. Acceptable on this 30 GB box.
    t0 = time.time()
    print('  fetching all rows…', flush=True)
    all_rows = conn.execute(
        "SELECT sid, aliaj_json FROM sentences ORDER BY sid").fetchall()
    print(f'  fetched {len(all_rows):,} rows in {time.time()-t0:.1f}s; '
          f'computing flags…', flush=True)

    t_c = time.time()
    flag_rows = []
    for i, (sid, aliaj_json) in enumerate(all_rows, 1):
        flag_rows.append((
            sid,
            aliaj_has_loko(aliaj_json),
            aliaj_has_jaro(aliaj_json),
            aliaj_has_numeral(aliaj_json),
        ))
        if i % 500_000 == 0:
            rate = i / max(time.time() - t_c, 0.001)
            eta = (len(all_rows) - i) / max(rate, 1)
            print(f'  computed {i:,}/{len(all_rows):,}  {rate:.0f}/s  '
                  f'ETA={eta/60:.1f}min', flush=True)
    print(f'  computed all flags in {time.time()-t_c:.1f}s', flush=True)
    # Free the source list before the big insert
    del all_rows

    t_i = time.time()
    print(f'  bulk-inserting into _aliaj_flags…', flush=True)
    conn.executemany(
        "INSERT INTO _aliaj_flags (sid, has_loko, has_jaro, has_kvant) "
        "VALUES (?, ?, ?, ?)",
        flag_rows,
    )
    print(f'  insert done in {time.time()-t_i:.1f}s', flush=True)
    del flag_rows

    # Bulk update
    print(f'  bulk-updating sentences from _aliaj_flags…', flush=True)
    t_u = time.time()
    conn.execute("""
        UPDATE sentences
        SET aliaj_has_loko  = f.has_loko,
            aliaj_has_jaro  = f.has_jaro,
            aliaj_has_kvant = f.has_kvant
        FROM _aliaj_flags f
        WHERE sentences.sid = f.sid
    """)
    conn.execute("CHECKPOINT")
    print(f'  update done in {time.time()-t_u:.1f}s', flush=True)


def add_indexes(conn) -> None:
    """Index each new column for fast filter pushdown."""
    for col in ('aliaj_has_loko', 'aliaj_has_jaro', 'aliaj_has_kvant'):
        try:
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{col} '
                f'ON sentences({col})')
            print(f'  indexed {col}', flush=True)
        except duckdb.Error as e:
            print(f'  index {col} failed: {e}', flush=True)


def coverage_report(conn) -> None:
    n_total = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    print(f'\n  Coverage on {n_total:,} sentences:', flush=True)
    for col in ('aliaj_has_loko', 'aliaj_has_jaro', 'aliaj_has_kvant'):
        n_true = conn.execute(
            f"SELECT COUNT(*) FROM sentences WHERE {col} = TRUE"
        ).fetchone()[0]
        n_null = conn.execute(
            f"SELECT COUNT(*) FROM sentences WHERE {col} IS NULL"
        ).fetchone()[0]
        pct = 100 * n_true / max(1, n_total)
        print(f'    {col:<22s} TRUE={n_true:>10,} ({pct:5.1f}%)  '
              f'NULL={n_null:,}', flush=True)


def _preflight() -> None:
    """Refuse to start without enough disk for the in-place UPDATE +
    temp table + index maintenance. In-place ALTER+UPDATE+INDEX on this
    5.4M-row table left ~30 GB of dead pages on the May 2026 run."""
    import subprocess
    out = subprocess.run(['df', '-k', '/'], capture_output=True, text=True)
    avail_kb = int(out.stdout.strip().split('\n')[1].split()[3])
    avail_gb = avail_kb // 1024 // 1024
    MIN_GB = 35
    if avail_gb < MIN_GB:
        print(f'\nREFUSING: only {avail_gb} GB free, need {MIN_GB} GB '
              f'for in-place UPDATE + temp table.', file=sys.stderr)
        print('  See scripts/util/cleanup_stale.sh for quick space recovery.',
              file=sys.stderr)
        sys.exit(2)
    print(f'preflight_disk: {avail_gb} GB free (need {MIN_GB} GB) — OK',
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--recompute', action='store_true',
                    help='Re-run the UPDATE even if columns already populated.')
    args = ap.parse_args()

    _preflight()
    print(f'Opening {args.duckdb_path} for write…', flush=True)
    conn = duckdb.connect(args.duckdb_path)
    conn.execute("SET memory_limit='2GB'")
    conn.execute("SET threads=4")

    print('\n=== Phase 1: add columns ===', flush=True)
    add_columns(conn)

    print('\n=== Phase 2: compute + update flags ===', flush=True)
    compute_and_update(conn, recompute=args.recompute)

    print('\n=== Phase 3: index columns ===', flush=True)
    add_indexes(conn)

    coverage_report(conn)
    conn.close()
    print('\n>>> DONE', flush=True)


if __name__ == '__main__':
    main()
