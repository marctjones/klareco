#!/usr/bin/env python3
"""
Backfill subj_propranoma_kat (and friends) from existing ast_json — no re-parse.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.ast_json` already populated
DEPENDENCIES: duckdb (no klareco parser — pure JSON extraction)
STAGE: Index / Data integrity

Description:
    The `subj_propranoma_kat` column was created as part of the v2.1 schema
    but never populated by the corpus build / refresh path. 5.39M rows
    have subj_propranoma_kat = NULL even though their ast_json contains
    the kategorio under subjekto.kerno.kategorio.

    This script is a focused backfill: it does NOT re-parse anything. It
    just reads ast_json from rows where the column is NULL (or the row's
    subj_vortspeco indicates a propra_nomo / propranomo-bearing kerno),
    extracts the kategorio, and stages it for a bulk UPDATE.

    Two-phase to avoid blocking other reads during the long Phase A scan:

      Phase A (SCAN-ONLY, read-only):
        - SELECT sid, ast_json FROM sentences
          WHERE subj_propranoma_kat IS NULL
            AND subj_vortspeco = 'propra_nomo'
        - For each row: extract subj.kerno.kategorio
        - Stage to JSONL: {sid, kat}
        - Per-row try/except — survives the corrupt-block IO error
          present in this DB until rebuild lands.

      Phase B (APPLY, brief write lock):
        - Read staging JSONL
        - UPDATE sentences SET subj_propranoma_kat = ? WHERE sid = ?
          in batches with COMMIT every N rows

    Note: this same pattern can be reused for other "field present in
    ast_json but missing from shredded column" backfills — see Bug #6's
    verb_negated re-derive case.

Pipeline Position:
    sentences.ast_json --scan--> staging JSONL --apply--> sentences UPDATEs
    (concurrent reads OK)        (read-only)              (brief write)

Usage:
    # Phase A — read-only scan, can run alongside other DB readers:
    python scripts/index/backfill_subj_propranoma_kat.py --scan-only

    # Phase B — brief write lock, apply staged updates:
    python scripts/index/backfill_subj_propranoma_kat.py --apply

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb


def kerno(node):
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def extract_kategorio(ast_json: str | None) -> str | None:
    """Pull subjekto.kerno.kategorio out of ast_json. Returns None on
    parse failure or missing field."""
    if not ast_json:
        return None
    try:
        ast = json.loads(ast_json)
    except Exception:
        return None
    k = kerno((ast or {}).get('subjekto'))
    if not isinstance(k, dict):
        return None
    return k.get('kategorio')


def last_sid_in_staging(staging_path: Path) -> int | None:
    if not staging_path.exists():
        return None
    last = None
    with open(staging_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get('sid')
                if sid is not None and (last is None or sid > last):
                    last = sid
            except Exception:
                continue
    return last


def phase_a_scan(args) -> None:
    print(f'Opening DuckDB at {args.duckdb_path} (READ-ONLY)…')
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")

    n_candidates = conn.execute(
        "SELECT COUNT(*) FROM sentences "
        "WHERE subj_propranoma_kat IS NULL "
        "AND subj_vortspeco = 'propra_nomo'"
    ).fetchone()[0]
    print(f'Candidates (propra_nomo subj, kategorio NULL): {n_candidates:,}')

    staging_path = Path(args.staging)
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    resume_from = last_sid_in_staging(staging_path) if not args.fresh else None
    if resume_from is not None:
        print(f'Resuming from sid > {resume_from:,}')
    else:
        print(f'Fresh scan; writing to {staging_path}')

    CHUNK = 50_000
    n_scanned = 0
    n_with_kat = 0
    n_io_errors = 0
    n_no_kat = 0
    t0 = time.time()
    last_sid = resume_from

    mode = 'a' if resume_from is not None else 'w'
    with open(staging_path, mode) as out_f:
        while True:
            if last_sid is None:
                sql = (
                    "SELECT sid, ast_json FROM sentences "
                    "WHERE subj_propranoma_kat IS NULL "
                    "AND subj_vortspeco = 'propra_nomo' "
                    f"ORDER BY sid LIMIT {CHUNK}"
                )
                params: list = []
            else:
                sql = (
                    "SELECT sid, ast_json FROM sentences "
                    "WHERE sid > ? "
                    "AND subj_propranoma_kat IS NULL "
                    "AND subj_vortspeco = 'propra_nomo' "
                    f"ORDER BY sid LIMIT {CHUNK}"
                )
                params = [last_sid]

            # Corruption-tolerant chunk read: fall back to per-sid scan
            # if the bulk SELECT throws on a corrupt block.
            try:
                rows = conn.execute(sql, params).fetchall()
            except Exception as e:
                print(f'  bulk SELECT failed at sid > {last_sid} ({e}); '
                      f'falling back to per-sid scan for this chunk', flush=True)
                rows = []
                # Find candidate sids in the corrupt region by counting
                if last_sid is None:
                    last_sid = 0
                # Try the next CHUNK sids one at a time
                for try_sid in range(last_sid + 1, last_sid + CHUNK + 1):
                    try:
                        r = conn.execute(
                            "SELECT sid, ast_json FROM sentences "
                            "WHERE sid = ? AND subj_propranoma_kat IS NULL "
                            "AND subj_vortspeco = 'propra_nomo'",
                            [try_sid],
                        ).fetchone()
                        if r:
                            rows.append(r)
                    except Exception:
                        n_io_errors += 1
                        continue
            if not rows:
                # End of data OR we've fallen through the corrupt block.
                # Probe ahead to see if there's still data above the
                # corrupt region.
                if last_sid is None:
                    break
                probe = conn.execute(
                    "SELECT MAX(sid) FROM sentences"
                ).fetchone()[0]
                if probe is None or last_sid >= probe:
                    break
                # Jump past the corrupt region in 100K steps
                last_sid = min(last_sid + 100_000, probe)
                continue

            for sid, ast_json in rows:
                n_scanned += 1
                last_sid = sid
                kat = extract_kategorio(ast_json)
                if kat is None:
                    n_no_kat += 1
                    continue
                n_with_kat += 1
                out_f.write(json.dumps({
                    'sid': int(sid),
                    'kat': kat,
                }, ensure_ascii=False) + '\n')

            elapsed = time.time() - t0
            rate = n_scanned / elapsed if elapsed > 0 else 0
            eta = (n_candidates - n_scanned) / rate if rate > 0 else float('inf')
            print(
                f'  scanned={n_scanned:>8,}/{n_candidates:,}  '
                f'with_kat={n_with_kat:>7,}  '
                f'no_kat={n_no_kat:>6,}  '
                f'io_err={n_io_errors:>5}  '
                f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
                flush=True,
            )
            out_f.flush()

    elapsed = time.time() - t0
    print(f'\n=== Phase A done ===')
    print(f'Scanned: {n_scanned:,} in {elapsed/60:.1f} min')
    print(f'With kategorio (stagable): {n_with_kat:,}')
    print(f'Without kategorio: {n_no_kat:,}')
    print(f'IO errors (corrupt-block sids): {n_io_errors:,}')
    print(f'Throughput: {n_scanned/max(1,elapsed):.0f} rows/sec')
    print(f'Staging: {staging_path}')


def phase_b_apply(args) -> None:
    staging_path = Path(args.staging)
    if not staging_path.exists():
        print(f'ERROR: staging file {staging_path} not found', file=sys.stderr)
        sys.exit(1)
    n_in_file = sum(1 for _ in open(staging_path))
    print(f'Staging file: {staging_path} ({n_in_file:,} updates)\n')

    print(f'Opening DuckDB at {args.duckdb_path} (WRITE)…')
    conn = duckdb.connect(args.duckdb_path)
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")

    BATCH = args.batch_size
    n_applied = 0
    n_failed = 0
    t0 = time.time()
    conn.execute('BEGIN TRANSACTION')
    in_tx = 0
    failed_sids = []

    with open(staging_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = r.get('sid')
            kat = r.get('kat')
            if sid is None or kat is None:
                continue
            try:
                conn.execute(
                    "UPDATE sentences SET subj_propranoma_kat = ? "
                    "WHERE sid = ?",
                    [kat, sid],
                )
                n_applied += 1
                in_tx += 1
            except Exception as e:
                n_failed += 1
                failed_sids.append((sid, str(e)[:60]))
                continue
            if in_tx >= BATCH:
                try:
                    conn.execute('COMMIT')
                except Exception as e:
                    print(f'  COMMIT failed: {e}; rolling back batch', flush=True)
                    n_failed += in_tx
                    n_applied -= in_tx
                    try:
                        conn.execute('ROLLBACK')
                    except Exception:
                        pass
                conn.execute('BEGIN TRANSACTION')
                in_tx = 0
                elapsed = time.time() - t0
                rate = n_applied / elapsed if elapsed > 0 else 0
                eta = (n_in_file - n_applied) / rate if rate > 0 else float('inf')
                print(
                    f'  applied={n_applied:>8,}/{n_in_file:,}  '
                    f'failed={n_failed:>5,}  '
                    f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
                    flush=True,
                )
    if in_tx > 0:
        try:
            conn.execute('COMMIT')
        except Exception as e:
            print(f'  Final COMMIT failed: {e}', flush=True)

    elapsed = time.time() - t0
    print(f'\n=== Phase B done ===')
    print(f'Applied: {n_applied:,} in {elapsed/60:.1f} min')
    print(f'Failed:  {n_failed:,}')
    if failed_sids[:5]:
        print(f'First few failures: {failed_sids[:5]}')

    # Coverage check
    n_filled = conn.execute(
        "SELECT COUNT(*) FROM sentences WHERE subj_propranoma_kat IS NOT NULL"
    ).fetchone()[0]
    total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'\nFinal subj_propranoma_kat coverage: '
          f'{n_filled:,} / {total:,}  ({100*n_filled/total:.2f}%)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--staging',
                    default='data/staging/subj_propranoma_kat_backfill.jsonl')
    ap.add_argument('--scan-only', action='store_true',
                    help='Phase A: read-only scan + write staging.')
    ap.add_argument('--apply', action='store_true',
                    help='Phase B: bulk-UPDATE from staging.')
    ap.add_argument('--fresh', action='store_true',
                    help='Phase A: overwrite staging instead of resuming.')
    ap.add_argument('--batch-size', type=int, default=5000,
                    help='Phase B: commit batch size (default 5000).')
    args = ap.parse_args()

    if not args.scan_only and not args.apply:
        print('ERROR: must specify --scan-only and/or --apply',
              file=sys.stderr)
        sys.exit(1)

    if args.scan_only:
        phase_a_scan(args)
    if args.apply:
        phase_b_apply(args)


if __name__ == '__main__':
    main()
