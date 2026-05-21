#!/usr/bin/env python3
"""
Two-phase AST correction discovery: read-only scan + staging, then bulk apply.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.ast_json`, current klareco.parser
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index / Data integrity

Description:
    Companion to `refresh_affected_asts.py`. That script holds an exclusive
    write lock on DuckDB for the entire scan + update, blocking other
    processes for the duration. This script splits the work:

      Phase A (SCAN-ONLY, read-only DuckDB):
        - Open DuckDB in read-only mode (no write lock)
        - For each sentence (optionally pre-filtered by LIKE patterns):
            * Re-parse `text` with the current parser
            * Compare new AST to stored ast_json
            * If they differ, append a correction record to staging JSONL:
                {sid, old_ast_summary, new_ast_json, new_subj_radiko,
                 new_verb_radiko, new_obj_radiko}
        - Other processes (bench, validator, …) can read the DB
          throughout

      Phase B (APPLY, brief write lock):
        - Read the staging JSONL
        - In a single transaction: UPDATE each sid's row with the new AST +
          new shredded radikos
        - ~minutes for hundreds of thousands of staged corrections

    The staging file is durable, auditable, diffable, and bulk-apply-able
    on your own schedule. Phase A can be killed and resumed (the apply
    is gated on the file's existence; resume continues appending).

    Use cases:
      - After a parser fix, scan to find which sentences are affected
      - Routine integrity audit: are there AST drifts we haven't caught?
      - Targeted bug investigation: stage corrections, inspect, then
        decide whether to apply

Pipeline Position:
    sentences.ast_json --scan-only-->  staging JSONL  --apply-->  sentences UPDATEs
    (concurrent OK)         (read-only)                 (brief write)

Usage:
    # Phase A — anytime, no DB lock:
    python scripts/index/find_ast_corrections.py --scan-only

    # Phase A with prefilter (only sentences likely affected by known bugs):
    python scripts/index/find_ast_corrections.py --scan-only --prefilter

    # Phase B — quick write when DB is free:
    python scripts/index/find_ast_corrections.py --apply

    # Combined (acquires write lock for whole run):
    python scripts/index/find_ast_corrections.py --scan-only --apply

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db
    --staging      data/staging/ast_corrections.jsonl

Outputs:
    Phase A: JSONL staging file with one correction per line
    Phase B: UPDATEs sentences.ast_json + shredded columns in place

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


# Same patterns used by refresh_affected_asts.py for narrowing the scan
_PREFILTER_LIKE = """(
    text LIKE 'En %' OR text LIKE 'Al %' OR text LIKE 'El %'
    OR text LIKE 'Sur %' OR text LIKE 'Sub %' OR text LIKE 'Apud %'
    OR text LIKE 'Antaŭ %' OR text LIKE 'Post %' OR text LIKE 'Tra %'
    OR text LIKE 'Kun %' OR text LIKE 'Per %' OR text LIKE 'Pri %'
    OR text LIKE 'Pro %' OR text LIKE 'Laŭ %' OR text LIKE 'Malgraŭ %'
    OR text LIKE 'Anstataŭ %' OR text LIKE 'Ekde %'
    OR text LIKE 'Kaj %' OR text LIKE 'Sed %' OR text LIKE 'Tamen %'
    OR text LIKE 'Universitato %' OR text LIKE 'Konsilio %'
    OR text LIKE 'Tie %' OR text LIKE 'Tiam %'
    OR text LIKE '%En kiu %' OR text LIKE '%Al kiu %' OR text LIKE '%Per kio %'
)"""


def kerno(node):
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


def shredded_radikos(ast: dict) -> tuple[str | None, str | None, str | None]:
    """Extract subj/verb/obj radikos from an AST."""
    subj_k = kerno(ast.get('subjekto'))
    verb = ast.get('verbo')
    obj_k = kerno(ast.get('objekto'))
    return (
        (subj_k or {}).get('radiko') if isinstance(subj_k, dict) else None,
        (verb or {}).get('radiko') if isinstance(verb, dict) else None,
        (obj_k or {}).get('radiko') if isinstance(obj_k, dict) else None,
    )


def ast_summary(ast: dict) -> dict:
    """Compact summary of the role-assignment of an AST, for diff diagnostics."""
    s_k = kerno(ast.get('subjekto'))
    v = ast.get('verbo')
    o_k = kerno(ast.get('objekto'))
    return {
        'subjekto_pv':        (s_k or {}).get('plena_vorto') if isinstance(s_k, dict) else None,
        'subjekto_vortspeco': (s_k or {}).get('vortspeco')   if isinstance(s_k, dict) else None,
        'verbo_pv':           (v or {}).get('plena_vorto')   if isinstance(v, dict) else None,
        'verbo_negita':       (v or {}).get('negita')        if isinstance(v, dict) else None,
        'objekto_pv':         (o_k or {}).get('plena_vorto') if isinstance(o_k, dict) else None,
        'objekto_vortspeco':  (o_k or {}).get('vortspeco')   if isinstance(o_k, dict) else None,
    }


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
    """SCAN: for each sentence, re-parse and compare to stored AST.
    If different, append a correction record to staging JSONL."""
    print(f'Opening DuckDB at {args.duckdb_path} (READ-ONLY, no lock)…')
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")

    where_clause = ''
    if args.prefilter:
        where_clause = f'WHERE {_PREFILTER_LIKE}'
        n_total = conn.execute(
            f'SELECT COUNT(*) FROM sentences {where_clause}'
        ).fetchone()[0]
        print(f'Prefilter active; {n_total:,} candidate sentences match.')
    else:
        n_total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
        print(f'No prefilter; scanning entire corpus of {n_total:,} sentences.')

    staging_path = Path(args.staging)
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    resume_from = last_sid_in_staging(staging_path) if not args.fresh else None
    if resume_from is not None:
        print(f'Resuming from sid > {resume_from:,}')
    else:
        print(f'Fresh scan; writing to {staging_path}')

    CHUNK = 50_000
    n_scanned = 0
    n_corrections = 0
    n_unchanged = 0
    t0 = time.time()
    last_sid: int | None = resume_from

    mode = 'a' if resume_from is not None else 'w'
    with open(staging_path, mode) as out_f:
        while True:
            if last_sid is None:
                sql = (f"SELECT sid, text, ast_json FROM sentences "
                       f"{where_clause} "
                       f"ORDER BY sid LIMIT {CHUNK}")
                params: list = []
            else:
                if where_clause:
                    sql = (f"SELECT sid, text, ast_json FROM sentences "
                           f"WHERE sid > ? AND {_PREFILTER_LIKE} "
                           f"ORDER BY sid LIMIT {CHUNK}")
                else:
                    sql = (f"SELECT sid, text, ast_json FROM sentences "
                           f"WHERE sid > ? ORDER BY sid LIMIT {CHUNK}")
                params = [last_sid]
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                break
            for sid, text, old_ast_json in rows:
                n_scanned += 1
                last_sid = sid
                if not text:
                    continue
                try:
                    new_ast = parse(text)
                except Exception:
                    continue
                new_ast_json = json.dumps(new_ast, ensure_ascii=False)
                if new_ast_json == old_ast_json:
                    n_unchanged += 1
                    continue
                # Stage the correction
                try:
                    old_ast = json.loads(old_ast_json) if old_ast_json else {}
                except Exception:
                    old_ast = {}
                new_subj_r, new_verb_r, new_obj_r = shredded_radikos(new_ast)
                out_f.write(json.dumps({
                    'sid':              int(sid),
                    'old_summary':      ast_summary(old_ast),
                    'new_summary':      ast_summary(new_ast),
                    'new_ast_json':     new_ast_json,
                    'new_subj_radiko':  new_subj_r,
                    'new_verb_radiko':  new_verb_r,
                    'new_obj_radiko':   new_obj_r,
                }, ensure_ascii=False) + '\n')
                n_corrections += 1
            elapsed = time.time() - t0
            rate = n_scanned / elapsed if elapsed > 0 else 0
            eta = (n_total - n_scanned) / rate if rate > 0 else float('inf')
            print(
                f'  scanned={n_scanned:>8,}/{n_total:,}  '
                f'corrections={n_corrections:>6,}  '
                f'unchanged={n_unchanged:>6,}  '
                f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
                flush=True,
            )

    elapsed = time.time() - t0
    print(f'\n=== Phase A done ===')
    print(f'Scanned {n_scanned:,} sentences in {elapsed/60:.1f} min')
    print(f'Found {n_corrections:,} ASTs needing correction '
          f'({100*n_corrections/max(1,n_scanned):.2f}%)')
    print(f'Unchanged: {n_unchanged:,}')
    print(f'Throughput: {n_scanned/elapsed:.0f} sentences/sec')
    print(f'Staging: {staging_path}')


def phase_b_apply(args) -> None:
    """APPLY: read staging JSONL, UPDATE each sid's row in a single batched
    transaction. Brief write lock."""
    staging_path = Path(args.staging)
    if not staging_path.exists():
        print(f'ERROR: staging file {staging_path} not found.', file=sys.stderr)
        sys.exit(1)
    n_in_file = sum(1 for _ in open(staging_path))
    print(f'Staging file: {staging_path} ({n_in_file:,} corrections)\n')

    print(f'Opening DuckDB at {args.duckdb_path} (WRITE)…')
    conn = duckdb.connect(args.duckdb_path)

    BATCH = args.batch_size
    n_applied = 0
    t0 = time.time()
    conn.execute('BEGIN TRANSACTION')
    in_tx = 0

    with open(staging_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            conn.execute(
                "UPDATE sentences "
                "SET ast_json = ?, subj_radiko = ?, verb_radiko = ?, obj_radiko = ? "
                "WHERE sid = ?",
                [
                    r['new_ast_json'],
                    r.get('new_subj_radiko'),
                    r.get('new_verb_radiko'),
                    r.get('new_obj_radiko'),
                    r['sid'],
                ]
            )
            n_applied += 1
            in_tx += 1
            if in_tx >= BATCH:
                conn.execute('COMMIT')
                conn.execute('BEGIN TRANSACTION')
                in_tx = 0
                elapsed = time.time() - t0
                rate = n_applied / elapsed if elapsed > 0 else 0
                eta = (n_in_file - n_applied) / rate if rate > 0 else float('inf')
                print(
                    f'  applied={n_applied:>7,}/{n_in_file:,}  '
                    f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
                    flush=True,
                )
    if in_tx > 0:
        conn.execute('COMMIT')

    elapsed = time.time() - t0
    print(f'\n=== Phase B done ===')
    print(f'Applied {n_applied:,} corrections in {elapsed/60:.1f} min')
    print(f'Throughput: {n_applied/elapsed:.0f} updates/sec')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--staging', default='data/staging/ast_corrections.jsonl')
    ap.add_argument('--scan-only', action='store_true',
                    help='Phase A: scan + write staging (no DB writes).')
    ap.add_argument('--apply', action='store_true',
                    help='Phase B: bulk-UPDATE from staging.')
    ap.add_argument('--prefilter', action='store_true',
                    help='Phase A: narrow scan with surface-pattern LIKE filter '
                         '(fronted prepositions, common-word-as-name openers, '
                         '`prep + ki-correlative` questions). Much faster scan.')
    ap.add_argument('--fresh', action='store_true',
                    help='Phase A: overwrite staging file instead of resuming.')
    ap.add_argument('--batch-size', type=int, default=1000,
                    help='Phase B: commit batch size.')
    args = ap.parse_args()

    if not args.scan_only and not args.apply:
        print('ERROR: must specify --scan-only and/or --apply', file=sys.stderr)
        sys.exit(1)

    if args.scan_only:
        phase_a_scan(args)
    if args.apply:
        phase_b_apply(args)


if __name__ == '__main__':
    main()
