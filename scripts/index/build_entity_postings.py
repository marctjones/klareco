#!/usr/bin/env python3
"""
Build the entity_postings inverted index in two phases (gh#729).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.ast_json` carrying
                 `multi_token_entities` annotations (post-parser fixes)
DEPENDENCIES: duckdb
STAGE: Index

Description:
    GitHub issue: #729
    The naïve single-phase build (one process holding write lock for
    the entire ~2-hour scan) blocks every other DB operation. This
    script splits the work:

      Phase A (SCAN-ONLY, read-only DB connection):
        - Open DuckDB in read-only mode (no write lock held)
        - Stream sentences in sid-range chunks
        - Extract entities (multi_token_entities + propra_nomo
          subjekto/objekto) for each
        - Write postings to a JSONL staging file
        - Other processes (bench, validator, etc.) can read DuckDB
          concurrently throughout

      Phase B (APPLY, brief write lock):
        - Open DuckDB read-write
        - DROP old entity_postings if present
        - CREATE TABLE entity_postings AS SELECT * FROM read_json(staging)
        - Build indices
        - ~30 seconds total

    The staging file (default: data/staging/entity_postings.jsonl) can
    be inspected/edited/diffed/version-controlled if needed before
    apply. Re-running scan-only resumes from the last sid in the file
    (checkpointable).

Pipeline Position:
    sentences.ast_json --scan-only-->  staging JSONL  --apply-->  entity_postings table
    (concurrent OK)         (read-only)                 (brief write)

Usage:
    # Phase A — run anytime, no DB lock contention:
    python scripts/index/build_entity_postings.py --scan-only

    # Phase B — quick write, run when DB is free:
    python scripts/index/build_entity_postings.py --apply

    # Combined (acquires write lock for whole run, like old behavior):
    python scripts/index/build_entity_postings.py --scan-only --apply

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db
    --staging      data/staging/entity_postings.jsonl

Outputs:
    Phase A: JSONL file at --staging path
    Phase B: entity_postings table + indices in DuckDB

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


def fold(s: str) -> str:
    """Lowercase + strip diacritics for diacritic-insensitive matching."""
    if not s:
        return s
    decomposed = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in decomposed if not unicodedata.combining(c)).lower()


def kerno(node):
    if not isinstance(node, dict):
        return None
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno')
    return node


# Surface markers for wiki redirect stubs and other low-value entities to skip.
_ENTITY_DENYLIST = {'REDIRECT', 'ALIDIREKTI', 'ALIDIREKTU'}


def _yield_propra_nomos(node, role_hint: str | None) -> list[tuple[str, str | None]]:
    """Yield (plena_vorto, role_hint) for every single-token propra_nomo
    reachable through a node (which may be a Vorto, Vortgrupo, or list).
    Recurses one level into vortgrupo.priskriboj — that's where adjective-
    descriptor and additional propra_nomo tokens often sit."""
    out: list[tuple[str, str | None]] = []
    if not isinstance(node, dict):
        return out
    if node.get('tipo') == 'vortgrupo':
        # The kerno itself
        kerno_node = node.get('kerno') or {}
        if isinstance(kerno_node, dict) and kerno_node.get('vortspeco') == 'propra_nomo':
            pv = kerno_node.get('plena_vorto') or ''
            if pv and ' ' not in pv:
                out.append((pv, role_hint))
        # Descriptors (priskriboj) — sometimes contain propra_nomo modifiers
        for desc in node.get('priskriboj') or []:
            if isinstance(desc, dict) and desc.get('vortspeco') == 'propra_nomo':
                pv = desc.get('plena_vorto') or ''
                if pv and ' ' not in pv:
                    out.append((pv, role_hint))
    elif node.get('vortspeco') == 'propra_nomo':
        pv = node.get('plena_vorto') or ''
        if pv and ' ' not in pv:
            out.append((pv, role_hint))
    return out


def extract_entities(ast: dict) -> list[tuple[str, int, str | None]]:
    """Return (entity_text, span_token_count, role_hint) tuples for the AST.

    Sources, in order:
      1. multi_token_entities groups (≥2 propra_nomo run)
      2. propra_nomo in subjekto / objekto (single-token only — multi-token
         forms come from #1)
      3. propra_nomo in `aliaj` items (PP-governed, modifier role — this is
         where many named entities live in encyclopedic Esperanto prose,
         e.g. `…la programisto de la ludo Fortnite` has Fortnite in aliaj)
      4. propra_nomo in priskriboj (descriptor of subjekto/objekto vortgrupos)

    Denylist: skip 'REDIRECT', 'ALIDIREKTI', 'ALIDIREKTU' which are wiki
    redirect-stub markers polluting the corpus (the top-10 entity report
    in the v1 build was dominated by 41K REDIRECT entries — these can't
    be real named entities).
    """
    entities: list[tuple[str, int, str | None]] = []

    # 1. multi-token spans
    for g in (ast.get('multi_token_entities') or []):
        span = g.get('span_tokens') or []
        if not span:
            continue
        if any(t in _ENTITY_DENYLIST for t in span):
            continue
        joined = ' '.join(span)
        entities.append((joined, len(span), None))

    # 2. + 4. subjekto / objekto kernos + priskriboj
    for role in ('subjekto', 'objekto'):
        for pv, hint in _yield_propra_nomos(ast.get(role), role):
            if pv not in _ENTITY_DENYLIST:
                entities.append((pv, 1, hint))

    # 3. aliaj items — flat scan
    for item in ast.get('aliaj') or []:
        for pv, _ in _yield_propra_nomos(item, 'aliaj'):
            if pv not in _ENTITY_DENYLIST:
                entities.append((pv, 1, 'aliaj'))

    return entities


def last_sid_in_staging(staging_path: Path) -> int | None:
    """Return the max sid in the staging file (for resume), or None if absent."""
    if not staging_path.exists():
        return None
    last = None
    with open(staging_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if last is None or obj['sid'] > last:
                    last = obj['sid']
            except Exception:
                continue
    return last


def phase_a_scan(args) -> None:
    """SCAN: stream sentences, extract entities, append to staging JSONL.
    Uses a READ-ONLY DuckDB connection — no write lock acquired."""
    print(f'Opening DuckDB at {args.duckdb_path} (READ-ONLY, no lock)…')
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    # OOM safety: cap DuckDB's working memory. The 35GB store mmap-loaded
    # without bound can chew through all available RAM, especially when
    # multiple readers run concurrently. 2GB is plenty for chunked queries.
    conn.execute(f"SET memory_limit = '{args.memory_limit}'")
    conn.execute(f"SET threads = {args.threads}")  # don't grab all cores
    conn.execute("SET preserve_insertion_order = false")

    n_total = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'Sentences in store: {n_total:,}')

    staging_path = Path(args.staging)
    staging_path.parent.mkdir(parents=True, exist_ok=True)

    resume_from = last_sid_in_staging(staging_path) if not args.fresh else None
    if resume_from is not None:
        print(f'Resuming from sid > {resume_from:,} (existing staging file '
              f'will be appended)')
    else:
        print(f'Fresh scan; writing to {staging_path}')

    # Pre-filter: only sentences with content of interest. Reduces ~40% of work.
    # The LIKE checks are cheap on DuckDB (zone-map skipping).
    pre_filter = (
        "(ast_json LIKE '%multi_token_entities%' "
        "OR ast_json LIKE '%\"vortspeco\":\"propra_nomo\"%' "
        "OR ast_json LIKE '%\"vortspeco\": \"propra_nomo\"%')"
    )

    CHUNK = args.chunk_size
    n_scanned = 0
    n_postings = 0
    t0 = time.time()
    last_sid: int | None = resume_from

    # Append mode if resuming, else fresh write
    mode = 'a' if resume_from is not None else 'w'
    with open(staging_path, mode) as out_f:
        while True:
            if last_sid is None:
                sql = (f"SELECT sid, ast_json FROM sentences "
                       f"WHERE {pre_filter} "
                       f"ORDER BY sid LIMIT {CHUNK}")
                params: list = []
            else:
                sql = (f"SELECT sid, ast_json FROM sentences "
                       f"WHERE sid > ? AND {pre_filter} "
                       f"ORDER BY sid LIMIT {CHUNK}")
                params = [last_sid]
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                break
            for sid, ast_json in rows:
                n_scanned += 1
                last_sid = sid
                if not ast_json:
                    continue
                try:
                    ast = json.loads(ast_json)
                except Exception:
                    continue
                for entity_text, span_count, role in extract_entities(ast):
                    out_f.write(json.dumps({
                        'entity_text':       entity_text,
                        'entity_normalized': fold(entity_text),
                        'span_token_count':  span_count,
                        'sid':               int(sid),
                        'role_hint':         role,
                    }, ensure_ascii=False) + '\n')
                    n_postings += 1
            elapsed = time.time() - t0
            rate = n_scanned / elapsed if elapsed > 0 else 0
            print(f'  scanned={n_scanned:>8,}  '
                  f'postings={n_postings:>8,}  '
                  f'last_sid={last_sid:>8,}  '
                  f'{rate:5.0f}/s',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\n=== Phase A done ===')
    print(f'Scanned {n_scanned:,} entity-bearing sentences in {elapsed/60:.1f} min')
    print(f'Wrote {n_postings:,} postings to {staging_path}')
    print(f'Throughput: {n_scanned/elapsed:.0f} sentences/sec')


def phase_b_apply(args) -> None:
    """APPLY: bulk-load staging JSONL into entity_postings + build indices.
    Brief write lock."""
    staging_path = Path(args.staging)
    if not staging_path.exists():
        print(f'ERROR: staging file {staging_path} does not exist. '
              'Run --scan-only first.', file=sys.stderr)
        sys.exit(1)
    n_postings_in_file = sum(1 for _ in open(staging_path))
    print(f'Staging file: {staging_path} ({n_postings_in_file:,} lines)\n')

    print(f'Opening DuckDB at {args.duckdb_path} (WRITE)…')
    conn = duckdb.connect(args.duckdb_path)
    conn.execute(f"SET memory_limit = '{args.memory_limit}'")
    conn.execute(f"SET threads = {args.threads}")
    conn.execute("SET preserve_insertion_order = false")

    print('DROP TABLE IF EXISTS entity_postings')
    conn.execute('DROP TABLE IF EXISTS entity_postings')

    print(f'Bulk-loading from {staging_path} via read_json_auto…')
    t0 = time.time()
    conn.execute(f"""
        CREATE TABLE entity_postings AS
        SELECT
            entity_text,
            entity_normalized,
            CAST(span_token_count AS INTEGER) AS span_token_count,
            CAST(sid AS BIGINT)               AS sid,
            role_hint
        FROM read_json_auto('{staging_path.absolute()}',
                            format='nd', records=true)
    """)
    n_loaded = conn.execute('SELECT COUNT(*) FROM entity_postings').fetchone()[0]
    print(f'  loaded {n_loaded:,} rows in {time.time()-t0:.1f}s')

    print('\nBuilding indices…')
    t0 = time.time()
    conn.execute('CREATE INDEX idx_entity_text ON entity_postings(entity_text)')
    conn.execute('CREATE INDEX idx_entity_normalized ON entity_postings(entity_normalized)')
    print(f'  done in {time.time()-t0:.1f}s')

    # Stats
    n_distinct = conn.execute(
        'SELECT COUNT(DISTINCT entity_text) FROM entity_postings'
    ).fetchone()[0]
    print(f'\n=== Phase B done ===')
    print(f'  distinct entity_text:    {n_distinct:,}')
    print(f'  total postings:          {n_loaded:,}')
    print(f'  average sids per entity:  {n_loaded/max(1,n_distinct):.1f}')

    print(f'\n=== Top-10 most-mentioned entities ===')
    rows = conn.execute("""
        SELECT entity_text, COUNT(*) AS n FROM entity_postings
        GROUP BY entity_text ORDER BY n DESC LIMIT 10
    """).fetchall()
    for entity_text, n in rows:
        print(f'  {entity_text:<40s}  {n:>8,}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--staging', default='data/staging/entity_postings.jsonl',
                    help='Staging JSONL path for the scan phase')
    ap.add_argument('--scan-only', action='store_true',
                    help='Phase A only: scan + write staging (no DB writes).')
    ap.add_argument('--apply', action='store_true',
                    help='Phase B: bulk-load staging into entity_postings.')
    ap.add_argument('--fresh', action='store_true',
                    help='Phase A: overwrite staging file instead of resuming.')
    ap.add_argument('--keep-staging', action='store_true',
                    help='Keep the staging file after --apply succeeds. '
                         'Default: delete it (the entity_postings table is '
                         'the source of truth once applied).')
    ap.add_argument('--chunk-size', type=int, default=20_000,
                    help='Phase A DuckDB query chunk size.')
    ap.add_argument('--memory-limit', default='4GB',
                    help='DuckDB memory limit for scan/apply phases.')
    ap.add_argument('--threads', type=int, default=2,
                    help='DuckDB worker threads for scan/apply phases.')
    args = ap.parse_args()

    if not args.scan_only and not args.apply:
        print('ERROR: must specify --scan-only and/or --apply', file=sys.stderr)
        sys.exit(1)

    if args.scan_only:
        phase_a_scan(args)
    if args.apply:
        phase_b_apply(args)
        # Clean up the staging file by default — once the data is in
        # entity_postings, the JSONL is a 1.4 GB disk-eater that's
        # regenerable from the DB via --scan-only.
        if not args.keep_staging:
            from pathlib import Path as _Path
            stage = _Path(args.staging)
            if stage.exists():
                size_mb = stage.stat().st_size // (1024 * 1024)
                stage.unlink()
                print(f'cleanup: removed {stage} ({size_mb} MB). '
                      f'Use --keep-staging to preserve.', flush=True)


if __name__ == '__main__':
    main()
