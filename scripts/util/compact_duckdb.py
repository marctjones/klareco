#!/usr/bin/env python3
"""
compact_duckdb.py — EXPORT/IMPORT round-trip to reclaim DuckDB dead pages.

VERSION: v1.0
COMPATIBLE WITH: data/indexes/duckdb_store.db (DuckDB v2.x)
DEPENDENCIES: duckdb
STAGE: Utility / Maintenance

Description:
    DuckDB doesn't auto-vacuum. Bulk ALTER + UPDATE + CREATE INDEX
    leaves behind dead pages that never get reclaimed. After the
    Stage-2 aliaj-flag work (commit c7f0d07) the store grew from
    ~32 GB to ~61 GB despite the new columns adding < 5 GB of real
    data. This script reclaims that space via the same EXPORT/IMPORT
    round-trip that the corruption-recovery tool uses, but framed
    for the routine-compaction case:

      Phase A — EXPORT all tables to Parquet (read-only on the DB)
      Phase B — VERIFY counts match between original and Parquet
      Phase C — RENAME original aside, IMPORT Parquet into fresh DB
      Phase D — VERIFY counts match in the fresh DB; delete backup

    The script will refuse to start if there isn't enough disk
    headroom for the temporary Parquet export.

Pipeline Position:
    bloated DB → [THIS SCRIPT] → compact DB at same path
                              (broken DB backup kept until verified)

Usage:
    python scripts/util/compact_duckdb.py --dry-run   # Phase A + B only
    python scripts/util/compact_duckdb.py --apply     # full round-trip

When to run:
    - After any bulk ALTER + UPDATE + CREATE INDEX operation
    - When `du -sh data/indexes/duckdb_store.db` is significantly
      larger than the expected row-count * row-size
    - As a quarterly maintenance task

Last Updated: 2026-05-26
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


DB_PATH      = Path('data/indexes/duckdb_store.db')
EXPORT_DIR   = Path('data/staging/duckdb_compact_export')
BACKUP_PATH  = Path('data/indexes/duckdb_store.db.precompact')


def disk_free_gb() -> int:
    """Free space on the filesystem holding the project (GB)."""
    out = subprocess.run(['df', '-k', '/'], capture_output=True, text=True)
    line = out.stdout.strip().split('\n')[1]
    avail_kb = int(line.split()[3])
    return avail_kb // 1024 // 1024


def preflight() -> None:
    db_size_gb = DB_PATH.stat().st_size // (1024 ** 3) + 1
    # Compaction needs room for Parquet export (~0.5x DB size) and
    # the imported fresh DB (~0.7x DB size) alongside the original
    # until verification passes. Estimate 1.5x DB as working headroom.
    needed_gb = int(db_size_gb * 1.5)
    have_gb = disk_free_gb()
    print(f'Preflight: db={db_size_gb} GB, headroom_needed={needed_gb} GB, '
          f'free={have_gb} GB', flush=True)
    if have_gb < needed_gb:
        print(f'\nREFUSING: insufficient disk headroom.', file=sys.stderr)
        print(f'  Free up at least {needed_gb - have_gb} more GB first.',
              file=sys.stderr)
        print(f'  See scripts/util/cleanup_stale.sh', file=sys.stderr)
        sys.exit(2)


def phase_a_export(con) -> None:
    print(f'\n=== PHASE A: EXPORT to {EXPORT_DIR} ===', flush=True)
    if EXPORT_DIR.exists():
        print(f'  Cleaning existing {EXPORT_DIR}…', flush=True)
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con.execute(f"EXPORT DATABASE '{EXPORT_DIR}' (FORMAT PARQUET)")
    elapsed = time.time() - t0
    print(f'  EXPORT done in {elapsed:.0f}s ({elapsed/60:.1f} min)',
          flush=True)


def phase_b_verify(con) -> dict:
    """Confirm Parquet counts equal in-DB counts. Aborts on mismatch."""
    print(f'\n=== PHASE B: VERIFY counts ===', flush=True)
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()]

    counts = {}
    mismatches = []
    for t in tables:
        in_db = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        pq = EXPORT_DIR / f'{t}.parquet'
        if not pq.exists():
            mismatches.append((t, in_db, 'parquet missing'))
            counts[t] = (in_db, None)
            continue
        scratch = duckdb.connect()
        scratch.execute("SET memory_limit='2GB'")
        in_pq = scratch.execute(
            f"SELECT COUNT(*) FROM read_parquet('{pq}')").fetchone()[0]
        scratch.close()
        counts[t] = (in_db, in_pq)
        if in_db != in_pq:
            mismatches.append((t, in_db, in_pq))
        print(f'  {t:<28s}  db={in_db:>10,}  parquet={in_pq:>10,}',
              flush=True)
    if mismatches:
        print(f'\n  MISMATCHES: {mismatches}', file=sys.stderr)
        sys.exit(3)
    print('  ✓ all tables match', flush=True)
    return counts


def phase_c_swap(counts: dict) -> None:
    """Move original aside, IMPORT fresh DB, verify, clean up."""
    print(f'\n=== PHASE C: SWAP ===', flush=True)
    if BACKUP_PATH.exists():
        print(f'  removing stale backup {BACKUP_PATH}…', flush=True)
        BACKUP_PATH.unlink()
    DB_PATH.rename(BACKUP_PATH)
    wal = DB_PATH.with_suffix('.db.wal')
    if wal.exists():
        wal.unlink()
    print(f'  ✓ original moved to {BACKUP_PATH}', flush=True)

    fresh = duckdb.connect(str(DB_PATH))
    fresh.execute("SET memory_limit='4GB'")
    fresh.execute("SET threads=4")
    print(f'  IMPORT DATABASE → {DB_PATH}…', flush=True)
    t0 = time.time()
    fresh.execute(f"IMPORT DATABASE '{EXPORT_DIR}'")
    fresh.execute("CHECKPOINT")
    print(f'  IMPORT done in {(time.time()-t0)/60:.1f} min', flush=True)

    print(f'\n=== PHASE D: VERIFY fresh counts ===', flush=True)
    for t, (in_db, _) in counts.items():
        in_fresh = fresh.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        ok = '✓' if in_fresh == in_db else '✗'
        print(f'  {ok} {t:<28s}  fresh={in_fresh:>10,}  '
              f'expected={in_db:>10,}', flush=True)
        if in_fresh != in_db:
            print(f'\n  MISMATCH in fresh DB. Original preserved at '
                  f'{BACKUP_PATH}. ABORTING.', file=sys.stderr)
            fresh.close()
            sys.exit(4)
    fresh.close()

    pre_size = BACKUP_PATH.stat().st_size // (1024**3)
    post_size = DB_PATH.stat().st_size // (1024**3)
    print(f'\n  Size before: {pre_size} GB')
    print(f'  Size after:  {post_size} GB')
    print(f'  Reclaimed:   {pre_size - post_size} GB')

    print(f'\n  Deleting backup {BACKUP_PATH}…', flush=True)
    BACKUP_PATH.unlink()
    print(f'  Deleting Parquet staging {EXPORT_DIR}…', flush=True)
    shutil.rmtree(EXPORT_DIR)
    print('  ✓ DONE', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Phase A + B only (no changes to live DB).')
    ap.add_argument('--apply', action='store_true',
                    help='Full A → B → C → D round-trip.')
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error('Need --dry-run or --apply')

    if not DB_PATH.exists():
        print(f'No DB at {DB_PATH}', file=sys.stderr)
        sys.exit(1)

    preflight()

    print(f'Opening {DB_PATH} read-only…', flush=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    phase_a_export(con)
    counts = phase_b_verify(con)
    con.close()

    if args.dry_run:
        print('\n--dry-run: stopping. Phase C/D not executed.', flush=True)
        return
    phase_c_swap(counts)


if __name__ == '__main__':
    main()
