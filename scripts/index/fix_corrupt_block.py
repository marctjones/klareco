#!/usr/bin/env python3
"""
Identify and recover corrupt-block sids in the DuckDB store.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with sid-aligned data/corpus/unified_corpus.jsonl
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index / Data integrity / Recovery

Background:
    During an OOM event on 2026-05-20, a write to data/indexes/duckdb_store.db
    left a block at file offset ~30 GB with a mismatched checksum. Reads
    that touch the bad block throw _duckdb.IOException; reads elsewhere
    succeed. The retriever (klareco/rag/duckdb_retriever.py) already catches
    the exception and falls back per-sid, so the bench limps along — but
    the affected sids are silently dropped.

    This script is the surgical recovery:

    Phase A (MAP, READ-ONLY):
        Walk sids in CHUNK ranges. Try SELECT for each chunk. When a
        chunk throws, binary-narrow to identify the specific corrupt
        sids. Stage to JSONL.

    Phase B (RECOVER, WRITE):
        For each corrupt sid, look up its original text in
        data/corpus/unified_corpus.jsonl (sid = 1-indexed line number),
        re-parse with the current parser, shred all columns, and
        DELETE + INSERT the row. DELETE-then-INSERT is safer than
        UPDATE because the corrupt row's MVCC chain may itself be
        unreadable.

    Both phases are restartable; the staging file is the durable record
    of which sids are corrupt.

Why not just rebuild?
    The corrupt region is likely small (<10K sids out of 5.4M). A full
    rebuild is ~1-2 hours; this targeted recovery should run in <10 min.
    See: docs/AST_MAINTENANCE.md (todo).

Pipeline Position:
    duckdb_store.db (corrupt) --MAP--> staging/corrupt_sids.jsonl
                              --RECOVER + unified_corpus.jsonl--> duckdb_store.db (clean)

Usage:
    # Phase A: read-only scan to find corrupt sids
    python scripts/index/fix_corrupt_block.py --map-only

    # Phase B: replace those sids' rows (brief write lock)
    python scripts/index/fix_corrupt_block.py --recover

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db
    --corpus       data/corpus/unified_corpus.jsonl
    --staging      data/staging/corrupt_sids.jsonl

Outputs:
    Phase A: JSONL { "sid": N, "err": "..." } per corrupt sid
    Phase B: DELETE + INSERT for each corrupt sid; final verification SELECT

Last Updated: 2026-05-21
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


# ---------- shared shred helper, mirrors build_duckdb_store.shred() but with
# the field-name fix for subj_propranoma_kat (was `propranoma_kategorio`,
# should be `kategorio`).

def _kerno(node):
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def shred(ast: dict) -> dict:
    s = _kerno(ast.get('subjekto') or {})
    v = _kerno(ast.get('verbo') or {})
    o = _kerno(ast.get('objekto') or {})
    aliaj = []
    for a in ast.get('aliaj') or []:
        w = _kerno(a)
        if w:
            aliaj.append({'radiko': w.get('radiko'),
                          'vortspeco': w.get('vortspeco'),
                          'kazo': w.get('kazo'),
                          'plena_vorto': w.get('plena_vorto')})
    stats = ast.get('parse_statistics') or {}
    return {
        'subj_radiko': s.get('radiko'),
        'subj_vortspeco': s.get('vortspeco'),
        # NOTE: read kategorio (not propranoma_kategorio — that was the
        # original build script's bug that left subj_propranoma_kat empty).
        'subj_propranoma_kat': s.get('kategorio'),
        'subj_kazo': s.get('kazo'),
        'verb_radiko': v.get('radiko'),
        'verb_tempo': v.get('tempo'),
        'obj_radiko': o.get('radiko'),
        'obj_kazo': o.get('kazo'),
        'aliaj_json': json.dumps(aliaj, ensure_ascii=False),
        'success_rate': float(stats.get('success_rate') or 0.0),
    }


COLS = ('sid', 'text', 'subj_radiko', 'subj_vortspeco', 'subj_propranoma_kat',
        'subj_kazo', 'verb_radiko', 'verb_tempo', 'obj_radiko', 'obj_kazo',
        'aliaj_json', 'success_rate', 'ast_json')


# ---------- Phase A: map corrupt sids -------------------------------------

def _try_select_chunk(conn, lo: int, hi: int) -> tuple[bool, str | None]:
    """True if SELECT for sids in [lo, hi] succeeds. Includes ast_json
    in the projection because that's the column whose storage actually
    holds the 2026-05-20 corrupt blocks — the sid index and text column
    are intact, so a narrow SELECT would miss the corruption."""
    try:
        conn.execute(
            "SELECT sid, text, ast_json FROM sentences "
            "WHERE sid BETWEEN ? AND ?",
            [lo, hi],
        ).fetchall()
        return True, None
    except Exception as e:
        return False, str(e)


def _binary_narrow(conn, lo: int, hi: int, out_f, t_ref: float) -> int:
    """Bisect a known-bad range [lo, hi]. Returns count of corrupt sids
    found. Each corrupt sid gets one line in out_f."""
    if lo == hi:
        # Single sid is corrupt — record it.
        ok, err = _try_select_chunk(conn, lo, hi)
        if not ok:
            out_f.write(json.dumps({'sid': lo, 'err': (err or '')[:160]},
                                   ensure_ascii=False) + '\n')
            out_f.flush()
            return 1
        return 0
    mid = (lo + hi) // 2
    ok_lo, err_lo = _try_select_chunk(conn, lo, mid)
    ok_hi, err_hi = _try_select_chunk(conn, mid + 1, hi)
    n = 0
    if not ok_lo:
        n += _binary_narrow(conn, lo, mid, out_f, t_ref)
    if not ok_hi:
        n += _binary_narrow(conn, mid + 1, hi, out_f, t_ref)
    return n


def phase_a_map(args) -> None:
    print(f'Opening DuckDB at {args.duckdb_path} (READ-ONLY)…')
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")
    max_sid = conn.execute('SELECT MAX(sid) FROM sentences').fetchone()[0]
    print(f'Max sid in DB: {max_sid:,}')

    staging = Path(args.staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    CHUNK = args.chunk
    t0 = time.time()
    n_corrupt = 0
    n_clean_chunks = 0
    n_bad_chunks = 0

    with open(staging, 'w') as out_f:
        lo = 1
        while lo <= max_sid:
            hi = min(lo + CHUNK - 1, max_sid)
            ok, err = _try_select_chunk(conn, lo, hi)
            if ok:
                n_clean_chunks += 1
            else:
                n_bad_chunks += 1
                print(f'  bad chunk [{lo:,}..{hi:,}]  err={err[:80] if err else "?"}',
                      flush=True)
                n_corrupt += _binary_narrow(conn, lo, hi, out_f, t0)
            elapsed = time.time() - t0
            if (n_clean_chunks + n_bad_chunks) % 50 == 0:
                rate = (n_clean_chunks + n_bad_chunks) * CHUNK / max(0.001, elapsed)
                eta = (max_sid - hi) / max(0.001, rate)
                print(f'  scanned to sid {hi:,}  '
                      f'clean_chunks={n_clean_chunks}  '
                      f'bad_chunks={n_bad_chunks}  '
                      f'corrupt={n_corrupt:,}  '
                      f'{rate:,.0f} sids/sec  ETA {eta:.0f}s',
                      flush=True)
            lo = hi + 1

    elapsed = time.time() - t0
    print(f'\n=== Phase A done ===')
    print(f'Scanned: 1..{max_sid:,} in {elapsed:.0f}s')
    print(f'Clean chunks: {n_clean_chunks:,}')
    print(f'Bad chunks: {n_bad_chunks:,}')
    print(f'Corrupt sids identified: {n_corrupt:,}')
    print(f'Staging: {staging}')


# ---------- Phase B: recover via reparse + DELETE+INSERT -------------------

def _load_corpus_index(corpus_path: Path) -> dict[int, str]:
    """Build a {sid → text} map by reading unified_corpus.jsonl. sid is
    the 1-indexed line number (matches build_duckdb_store.py's scheme)."""
    print(f'Loading corpus from {corpus_path}…')
    idx: dict[int, str] = {}
    n = 0
    t0 = time.time()
    with open(corpus_path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get('text') or ''
            if text:
                idx[i] = text
            n += 1
            if n % 500_000 == 0:
                print(f'  loaded {n:,} lines', flush=True)
    print(f'Loaded {len(idx):,} sentences in {time.time()-t0:.0f}s')
    return idx


def phase_b_recover(args) -> None:
    staging = Path(args.staging)
    if not staging.exists():
        print(f'ERROR: staging file {staging} not found', file=sys.stderr)
        sys.exit(1)
    sids: list[int] = []
    with open(staging) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get('sid')
            if sid is not None:
                sids.append(int(sid))
    sids = sorted(set(sids))
    print(f'Corrupt sids to recover: {len(sids):,}')
    if not sids:
        print('Nothing to do.')
        return

    # Need the original texts: load corpus into memory (only need the
    # affected sids). For small corrupt sets, sparse read is cheaper;
    # for large, full load is faster. Threshold: 100K.
    corpus_path = Path(args.corpus)
    if len(sids) < 100_000:
        # Sparse read: just grab the lines we need
        wanted = set(sids)
        texts: dict[int, str] = {}
        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i not in wanted:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get('text') or ''
                if text:
                    texts[i] = text
                if len(texts) == len(wanted):
                    break
        print(f'Sparse-loaded {len(texts):,} / {len(sids):,} texts')
    else:
        texts = _load_corpus_index(corpus_path)

    print(f'Opening DuckDB at {args.duckdb_path} (WRITE)…')
    conn = duckdb.connect(args.duckdb_path)
    conn.execute("SET memory_limit = '2GB'")
    conn.execute("SET threads = 4")

    BATCH = args.batch_size
    n_applied = 0
    n_failed = 0
    n_missing_text = 0
    in_tx = 0
    t0 = time.time()
    conn.execute('BEGIN TRANSACTION')

    for sid in sids:
        text = texts.get(sid)
        if not text:
            n_missing_text += 1
            continue
        try:
            ast = parse(text)
        except Exception:
            ast = None
        if not isinstance(ast, dict):
            n_failed += 1
            continue
        shredded = shred(ast)
        ast_json = json.dumps(ast, ensure_ascii=False)
        row = (sid, text,
               shredded['subj_radiko'], shredded['subj_vortspeco'],
               shredded['subj_propranoma_kat'], shredded['subj_kazo'],
               shredded['verb_radiko'], shredded['verb_tempo'],
               shredded['obj_radiko'], shredded['obj_kazo'],
               shredded['aliaj_json'], shredded['success_rate'],
               ast_json)
        try:
            # DELETE then INSERT — UPDATE may need to read the corrupt
            # row's MVCC chain.
            conn.execute('DELETE FROM sentences WHERE sid = ?', [sid])
            conn.execute(
                "INSERT INTO sentences "
                "(sid, text, subj_radiko, subj_vortspeco, subj_propranoma_kat, "
                " subj_kazo, verb_radiko, verb_tempo, obj_radiko, obj_kazo, "
                " aliaj_json, success_rate, ast_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                list(row),
            )
            n_applied += 1
            in_tx += 1
        except Exception as e:
            n_failed += 1
            print(f'  sid={sid} FAILED: {str(e)[:120]}', flush=True)
            try:
                conn.execute('ROLLBACK')
                conn.execute('BEGIN TRANSACTION')
                in_tx = 0
            except Exception:
                pass
            continue
        if in_tx >= BATCH:
            try:
                conn.execute('COMMIT')
            except Exception as e:
                print(f'  COMMIT failed: {e}', flush=True)
            conn.execute('BEGIN TRANSACTION')
            in_tx = 0
            elapsed = time.time() - t0
            rate = n_applied / max(0.001, elapsed)
            eta = (len(sids) - n_applied - n_failed - n_missing_text) / max(0.001, rate)
            print(f'  applied={n_applied:,}/{len(sids):,}  '
                  f'failed={n_failed:,}  missing_text={n_missing_text:,}  '
                  f'{rate:.0f}/s  ETA {eta:.0f}s', flush=True)
    if in_tx > 0:
        try:
            conn.execute('COMMIT')
        except Exception as e:
            print(f'  Final COMMIT failed: {e}', flush=True)

    elapsed = time.time() - t0
    print(f'\n=== Phase B done ===')
    print(f'Applied:       {n_applied:,}')
    print(f'Failed:        {n_failed:,}')
    print(f'Missing text:  {n_missing_text:,}')
    print(f'Wall time:     {elapsed:.0f}s')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--corpus', default='data/corpus/unified_corpus.jsonl')
    ap.add_argument('--staging', default='data/staging/corrupt_sids.jsonl')
    ap.add_argument('--map-only', action='store_true',
                    help='Phase A: read-only scan to find corrupt sids.')
    ap.add_argument('--recover', action='store_true',
                    help='Phase B: re-parse + DELETE+INSERT corrupt rows.')
    ap.add_argument('--chunk', type=int, default=10000,
                    help='Phase A: chunk size for the SELECT probes.')
    ap.add_argument('--batch-size', type=int, default=500,
                    help='Phase B: commit cadence.')
    args = ap.parse_args()
    if not args.map_only and not args.recover:
        print('ERROR: specify --map-only and/or --recover', file=sys.stderr)
        sys.exit(1)
    if args.map_only:
        phase_a_map(args)
    if args.recover:
        phase_b_recover(args)


if __name__ == '__main__':
    main()
