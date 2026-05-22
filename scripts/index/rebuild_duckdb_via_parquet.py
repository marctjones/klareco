#!/usr/bin/env python3
"""Option 2: EXPORT broken DB to Parquet → IMPORT to fresh DB.

VERSION: v2.x
COMPATIBLE WITH: DuckDB store with corrupted PK index
DEPENDENCIES: duckdb
STAGE: Repair

Description:
    The PK index on `sentences` is corrupted (deterministic INTERNAL
    crash on sid 2764528) and DuckDB doesn't support DROP CONSTRAINT.
    This script does the only remaining clean fix: round-trip the data
    through Parquet.

    Three phases:

      Phase A — EXPORT (read-only).
        `EXPORT DATABASE 'staging/parquet_export' (FORMAT PARQUET)` dumps
        every table to a Parquet file + a `schema.sql` + `load.sql`.
        Read-only, never touches the broken index.

      Phase B — VERIFY.
        Count rows in the Parquet export per table; compare to the
        broken DB's counts. Anything mismatched aborts before delete.

      Phase C — SWAP.
        - Move broken DB out of the way (rename, don't delete yet).
        - `IMPORT DATABASE 'staging/parquet_export'` into a fresh DB
          at the original path. This re-creates the schema and PK
          index over clean data.
        - Verify fresh-DB counts.
        - Delete the broken DB backup once fresh is confirmed.
        - Optionally delete the Parquet export.

    Aborts before any destructive operation if Phase B fails.

Pipeline Position:
    broken sentences.duckdb_store.db → [THIS SCRIPT] → fresh DB at same
                                                       path; broken DB
                                                       backed up first

Usage:
    # Dry-run (Phase A + B only)
    python scripts/index/rebuild_duckdb_via_parquet.py --dry-run

    # Full rebuild
    python scripts/index/rebuild_duckdb_via_parquet.py --apply

Last Updated: 2026-05-22
Author: Claude Code (with Marc Jones)
"""
import argparse
import duckdb
import os
import shutil
import sys
import time
from pathlib import Path


DB_PATH      = Path('data/indexes/duckdb_store.db')
EXPORT_DIR   = Path('data/staging/duckdb_parquet_export')
BACKUP_PATH  = Path('data/indexes/duckdb_store.db.broken_2026-05-22')


def phase_a_export(con) -> None:
    """EXPORT DATABASE to Parquet."""
    print(f'\n=== PHASE A: EXPORT to {EXPORT_DIR} ===', flush=True)
    if EXPORT_DIR.exists():
        print(f'  Cleaning existing {EXPORT_DIR}…', flush=True)
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    con.execute(f"EXPORT DATABASE '{EXPORT_DIR}' (FORMAT PARQUET)")
    elapsed = time.time() - t0
    print(f'  EXPORT done in {elapsed:.1f}s ({elapsed/60:.1f} min)', flush=True)


def phase_b_verify(con) -> dict:
    """Verify Parquet row counts match broken DB."""
    print(f'\n=== PHASE B: VERIFY counts ===', flush=True)

    # Get table list from broken DB
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()]
    print(f'  Tables in broken DB: {tables}', flush=True)

    broken_counts = {}
    for t in tables:
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            broken_counts[t] = n
        except Exception as e:
            print(f'  count {t} failed: {e}', flush=True)
            broken_counts[t] = None
    print(f'  broken counts: {broken_counts}', flush=True)

    # Now count in the Parquet export
    parquet_counts = {}
    for t in tables:
        pq = EXPORT_DIR / f'{t}.parquet'
        if not pq.exists():
            print(f'  WARN: {pq} not in export', flush=True)
            parquet_counts[t] = None
            continue
        try:
            con2 = duckdb.connect()
            con2.execute("SET memory_limit='2GB'")
            n = con2.execute(
                f"SELECT COUNT(*) FROM read_parquet('{pq}')"
            ).fetchone()[0]
            parquet_counts[t] = n
            con2.close()
        except Exception as e:
            print(f'  parquet count {t} failed: {e}', flush=True)
            parquet_counts[t] = None
    print(f'  parquet counts: {parquet_counts}', flush=True)

    # Compare
    mismatches = []
    for t in tables:
        b = broken_counts.get(t)
        p = parquet_counts.get(t)
        if b != p:
            mismatches.append((t, b, p))
    if mismatches:
        print(f'\n  MISMATCHES: {mismatches}', flush=True)
    else:
        print('\n  ✓ All tables match', flush=True)
    return {'tables': tables, 'broken': broken_counts,
            'parquet': parquet_counts, 'mismatches': mismatches}


def phase_c_swap(verify: dict) -> None:
    """Move broken aside, IMPORT fresh, verify, clean up."""
    print(f'\n=== PHASE C: SWAP ===', flush=True)

    # 1. Move broken aside
    print(f'\n  Moving broken DB → {BACKUP_PATH}…', flush=True)
    if BACKUP_PATH.exists():
        print(f'  WARN: backup already exists, removing it', flush=True)
        BACKUP_PATH.unlink()
    DB_PATH.rename(BACKUP_PATH)
    wal = DB_PATH.with_suffix('.db.wal')
    if wal.exists():
        wal.unlink()
    print('  ✓ broken DB renamed', flush=True)

    # 2. Import to fresh
    print(f'\n  IMPORT DATABASE into fresh {DB_PATH}…', flush=True)
    t0 = time.time()
    fresh = duckdb.connect(str(DB_PATH))
    fresh.execute("SET memory_limit='4GB'")
    fresh.execute("SET threads=4")
    fresh.execute(f"IMPORT DATABASE '{EXPORT_DIR}'")
    fresh.execute("CHECKPOINT")
    elapsed = time.time() - t0
    print(f'  IMPORT done in {elapsed:.1f}s ({elapsed/60:.1f} min)', flush=True)

    # 3. Verify
    print('\n  Verifying fresh DB row counts…', flush=True)
    fresh_counts = {}
    for t in verify['tables']:
        try:
            n = fresh.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            fresh_counts[t] = n
        except Exception as e:
            print(f'    count {t} failed: {e}', flush=True)
            fresh_counts[t] = None
    print(f'    fresh counts: {fresh_counts}', flush=True)

    # 4. Smoke test on the killer sid 2764528
    print('\n  Smoke test: UPDATE on the previously-failing sid 2764528…',
          flush=True)
    try:
        t0 = time.time()
        fresh.execute(
            "UPDATE sentences SET subj_propranoma_kat = subj_propranoma_kat "
            "WHERE sid = 2764528"
        )
        fresh.execute("CHECKPOINT")
        print(f'    ✓ ok in {time.time()-t0:.2f}s', flush=True)
    except Exception as e:
        print(f'    SMOKE TEST FAILED: {type(e).__name__}: '
              f'{str(e)[:200]}', flush=True)
        fresh.close()
        sys.exit(2)

    fresh.close()
    print('\n  ✓ Fresh DB looks healthy.', flush=True)

    # 5. Compare counts
    mismatches = []
    for t, b in verify['broken'].items():
        f = fresh_counts.get(t)
        if b != f:
            mismatches.append((t, b, f))
    if mismatches:
        print(f'\n  COUNT MISMATCH IN FRESH: {mismatches}', flush=True)
        print('  NOT deleting backup; investigate before cleanup.', flush=True)
        sys.exit(3)

    # 6. Delete backup
    print(f'\n  Deleting broken backup {BACKUP_PATH} (free ~44 GB)…',
          flush=True)
    BACKUP_PATH.unlink()
    print('  ✓ done', flush=True)

    # 7. Offer to delete the parquet export
    pq_size_gb = sum(p.stat().st_size for p in EXPORT_DIR.rglob('*')
                     if p.is_file()) / 1e9
    print(f'\n  Parquet export at {EXPORT_DIR} ({pq_size_gb:.1f} GB) '
          'preserved as additional rollback. Delete manually when '
          'confident.', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Phase A + B only; do not touch the DB on disk.')
    ap.add_argument('--apply', action='store_true',
                    help='Run all three phases.')
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error('Need --dry-run or --apply')

    if not DB_PATH.exists():
        print(f'No DB at {DB_PATH}', file=sys.stderr)
        sys.exit(1)

    # Phase A: open broken DB read-only and export
    print(f'Opening {DB_PATH} read-only…', flush=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")

    phase_a_export(con)
    verify = phase_b_verify(con)
    con.close()

    if verify['mismatches']:
        print('\nABORTING: Parquet counts diverge from broken DB. '
              'Investigate before any swap.', flush=True)
        sys.exit(4)

    if args.dry_run:
        print('\n--dry-run: stopping after Phase A + B. Phase C not '
              'attempted.', flush=True)
        return

    phase_c_swap(verify)
    print('\n>>> REBUILD COMPLETE <<<', flush=True)


if __name__ == '__main__':
    main()
