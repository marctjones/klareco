#!/usr/bin/env python3
"""
Build the entity_postings inverted index from AST multi_token_entities (gh#729).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with `sentences.ast_json` carrying
                 `multi_token_entities` annotations (post-parser fixes)
DEPENDENCIES: duckdb
STAGE: Index

Description:
    GitHub issue: #729
    Build an inverted index mapping entity_text → list of sentence ids.
    The diagnostic on 2026-05-20 showed BM25 outranks specific entities
    with generic-term-match noise (e.g. `Fortnite` lost to `senpagan ludon`).
    This index gives the retriever a direct-lookup path:
        "Find sentences mentioning Fortnite" → instant sid list
    bypassing the BM25 score-ranking step.

    Schema:
        CREATE TABLE entity_postings (
            entity_text       VARCHAR,       -- 'Béla Buzogány'
            entity_normalized VARCHAR,       -- diacritic-folded, lowercased
            span_token_count  INTEGER,       -- 1, 2, 3, …
            sid               BIGINT,
            role_hint         VARCHAR        -- subjekto/objekto/aliaj if known
        );

    Population strategy:
      - Streams sentences.ast_json one at a time (avoids OOM on 5.4M)
      - For each, extracts multi_token_entities + single-token propra_nomos
        that appear in subjekto/objekto kernos
      - INSERTs into entity_postings as a batch every N rows
      - Builds indices on entity_text and entity_normalized at the end

Pipeline Position:
    sentences.ast_json → [THIS SCRIPT] → entity_postings + indices
    → EntityPostingsReranker (multi_reranker_bench.py)
    → routing in orchestrator for entity-anchored questions

Usage:
    python scripts/index/build_entity_postings.py
    python scripts/index/build_entity_postings.py --limit 100000  # test mode

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db (read-write)

Outputs:
    Creates `entity_postings` table + 2 indices.
    Stats: distinct entities, per-entity sid count histogram.

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
    """Lowercase + strip diacritics, for diacritic-insensitive matching."""
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


def extract_entities(ast: dict) -> list[tuple[str, int, str | None]]:
    """Return (entity_text, span_token_count, role_hint) tuples for the AST.

    Sources:
      - ast['multi_token_entities'] → each group's span_tokens joined
      - ast['subjekto'/'objekto'] kernos with vortspeco=='propra_nomo'
        and a single-token plena_vorto (multi-token cases already in mte)
    """
    entities: list[tuple[str, int, str | None]] = []

    for g in (ast.get('multi_token_entities') or []):
        span = g.get('span_tokens') or []
        if not span:
            continue
        joined = ' '.join(span)
        entities.append((joined, len(span), None))

    for role in ('subjekto', 'objekto'):
        k = kerno(ast.get(role))
        if not isinstance(k, dict):
            continue
        if k.get('vortspeco') != 'propra_nomo':
            continue
        pv = k.get('plena_vorto') or ''
        if not pv or ' ' in pv:
            continue  # multi-token forms come from multi_token_entities
        entities.append((pv, 1, role))

    return entities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most this many sentences (test mode).')
    ap.add_argument('--batch-size', type=int, default=5000,
                    help='INSERT batch size (default 5000).')
    args = ap.parse_args()

    print(f'Opening DuckDB at {args.duckdb_path} (write connection)…')
    write_conn = duckdb.connect(args.duckdb_path)

    # Drop + recreate (idempotent fresh build)
    print('Recreating entity_postings table…')
    write_conn.execute('DROP TABLE IF EXISTS entity_postings')
    write_conn.execute("""
        CREATE TABLE entity_postings (
            entity_text       VARCHAR,
            entity_normalized VARCHAR,
            span_token_count  INTEGER,
            sid               BIGINT,
            role_hint         VARCHAR
        )
    """)

    # The write connection's cursor would get clobbered if we re-used it for
    # SELECT iteration. We pre-collect (sid, ast_json) tuples by chunked
    # SELECTs ordered by sid — chunked to keep memory bounded.
    n_total = write_conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    if args.limit:
        n_total = min(n_total, args.limit)
    print(f'Scanning {n_total:,} sentences in chunks of 100K…\n')

    batch: list[tuple[str, str, int, int, str | None]] = []
    n_scanned = 0
    n_postings = 0
    t0 = time.time()

    CHUNK = 100_000
    last_sid: int | None = None
    while n_scanned < n_total:
        # Fetch one chunk by sid range
        if last_sid is None:
            sql = (f'SELECT sid, ast_json FROM sentences '
                   f'ORDER BY sid LIMIT {CHUNK}')
            params: list = []
        else:
            sql = (f'SELECT sid, ast_json FROM sentences '
                   f'WHERE sid > ? ORDER BY sid LIMIT {CHUNK}')
            params = [last_sid]
        rows = write_conn.execute(sql, params).fetchall()
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
                batch.append((entity_text, fold(entity_text), span_count, sid, role))
                n_postings += 1
            if len(batch) >= args.batch_size:
                write_conn.executemany(
                    'INSERT INTO entity_postings VALUES (?, ?, ?, ?, ?)',
                    batch
                )
                batch.clear()
        elapsed = time.time() - t0
        rate = n_scanned / elapsed if elapsed > 0 else 0
        eta = (n_total - n_scanned) / rate if rate > 0 else float('inf')
        print(f'  {n_scanned:>8,} / {n_total:,}  '
              f'({100*n_scanned/n_total:5.1f}%)  '
              f'postings={n_postings:>8,}  '
              f'{rate:5.0f}/s  ETA {eta/60:5.1f}m',
              flush=True)
        if args.limit and n_scanned >= args.limit:
            break

    if batch:
        write_conn.executemany(
            'INSERT INTO entity_postings VALUES (?, ?, ?, ?, ?)',
            batch
        )
    conn = write_conn  # used by the rest of the function

    elapsed = time.time() - t0
    print(f'\nScanned {n_scanned:,} sentences in {elapsed:.1f}s')
    print(f'Generated {n_postings:,} postings')

    print('\nBuilding indices…')
    t0 = time.time()
    conn.execute('CREATE INDEX idx_entity_text ON entity_postings(entity_text)')
    conn.execute('CREATE INDEX idx_entity_normalized ON entity_postings(entity_normalized)')
    print(f'  done in {time.time()-t0:.1f}s')

    # Stats
    n_distinct = conn.execute(
        'SELECT COUNT(DISTINCT entity_text) FROM entity_postings'
    ).fetchone()[0]
    print(f'\n=== Stats ===')
    print(f'  distinct entity_text:   {n_distinct:,}')
    print(f'  total postings:         {n_postings:,}')
    print(f'  average sids per entity: {n_postings/max(1,n_distinct):.1f}')

    # Top-10 most-mentioned entities (sanity)
    print(f'\n=== Top-10 most-mentioned entities ===')
    rows = conn.execute("""
        SELECT entity_text, COUNT(*) AS n FROM entity_postings
        GROUP BY entity_text ORDER BY n DESC LIMIT 10
    """).fetchall()
    for entity_text, n in rows:
        print(f'  {entity_text:<40s}  {n:>8,}')


if __name__ == '__main__':
    main()
