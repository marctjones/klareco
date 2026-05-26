#!/usr/bin/env python3
"""
Walk the DuckDB store and extract entity facts (#745).

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: data/indexes/duckdb_store.db with shredded AST columns
DEPENDENCIES: duckdb, klareco.rag.entity_fact_patterns
STAGE: Index / Knowledge extraction

Description:
    Generalizes the per-relation pattern KB tables (capital_of, lingvo)
    to every (entity, slot, value) tuple revealed by the AST. Runs every
    pattern in klareco.rag.entity_fact_patterns over every shredded
    sentence row and emits Fact records.

    The output is a queryable lookup table:

      entity_facts(
        entity_radiko VARCHAR,    -- lower-cased radiko (e.g. 'zamenhof')
        slot          VARCHAR,    -- 'founder' / 'birth_place' / ...
        value         VARCHAR,    -- raw surface value (e.g. 'Bjalistok')
        value_radiko  VARCHAR,    -- normalized form (e.g. 'bjalistok')
        source_sid    INTEGER,
        confidence    DOUBLE,
        pattern_name  VARCHAR,
      )

    Indexed on (entity_radiko, slot) and (slot, value_radiko) for
    bidirectional lookup: "what's Zamenhof's profession?" and
    "who founded Esperanto?".

Pipeline Position:
    sentences (shredded AST) → [THIS SCRIPT] → entity_facts
                                            → EntityFactRetriever route

Usage:
    # Dry-run on first N rows; see what patterns match, no DB writes:
    python scripts/index/extract_entity_facts.py --dry-run --limit 50000

    # Apply: run all patterns over all rows, write to entity_facts table:
    python scripts/index/extract_entity_facts.py --apply

    # Apply, but rebuild the table from scratch (drops previous contents):
    python scripts/index/extract_entity_facts.py --apply --fresh

Acceptance criteria (per #745):
    - For capability_candidates_v1, R@1 should rise from 52 to 65+ once
      the EntityFactRetriever route consults this table.
    - At minimum, the table must contain a row for each of the
      pattern_capital_of and pattern_lingvo facts we already have
      (12 capital + 2 lingvo).

Last Updated: 2026-05-26
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.rag.entity_fact_patterns import (
    Fact, ALL_PATTERNS, extract_facts_from_row,
)


DB = 'data/indexes/duckdb_store.db'

_COLS = (
    'sid', 'text',
    'subj_radiko', 'subj_vortspeco', 'subj_propranoma_kat', 'subj_kazo',
    'verb_radiko', 'verb_tempo', 'verb_klaso', 'verb_negated',
    'obj_radiko', 'obj_kazo',
    'aliaj_json',
)


def ensure_schema(conn) -> None:
    """Create entity_facts table + indexes if not present."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_facts (
          entity_radiko VARCHAR NOT NULL,
          slot          VARCHAR NOT NULL,
          value         VARCHAR NOT NULL,
          value_radiko  VARCHAR NOT NULL,
          source_sid    INTEGER NOT NULL,
          confidence    DOUBLE  NOT NULL,
          pattern_name  VARCHAR NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_facts_entity "
        "ON entity_facts(entity_radiko, slot)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_facts_value "
        "ON entity_facts(slot, value_radiko)")


def stream_rows(conn, limit: int = 0):
    """Yield shredded sentence rows in sid order."""
    sql = f"SELECT {', '.join(_COLS)} FROM sentences ORDER BY sid"
    if limit:
        sql += f" LIMIT {limit}"
    cur = conn.execute(sql)
    while True:
        chunk = cur.fetchmany(20_000)
        if not chunk:
            return
        for r in chunk:
            yield dict(zip(_COLS, r))


def run_extraction(conn, limit: int, apply: bool, fresh: bool) -> Counter:
    """Walk rows, run patterns, optionally write to entity_facts. Returns
    a per-pattern Counter of facts emitted."""
    per_pattern = Counter()
    total_facts = 0
    rows_seen = 0
    rows_with_facts = 0
    t0 = time.time()

    if apply:
        if fresh:
            print('  DROP existing entity_facts (--fresh)…', flush=True)
            conn.execute('DROP TABLE IF EXISTS entity_facts')
        ensure_schema(conn)
        print('  schema ready', flush=True)

    batch: list[Fact] = []
    BATCH = 5_000

    def flush(batch):
        if not apply or not batch:
            return 0
        conn.executemany(
            "INSERT INTO entity_facts "
            "(entity_radiko, slot, value, value_radiko, "
            " source_sid, confidence, pattern_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(f.entity_radiko, f.slot, f.value, f.value_radiko,
              f.source_sid, f.confidence, f.pattern_name) for f in batch],
        )
        return len(batch)

    for row in stream_rows(conn, limit=limit):
        rows_seen += 1
        # We pass ast=None here — patterns operate on shredded columns +
        # aliaj_json walking. If a future pattern needs the full AST, we
        # can re-parse from row['text'] or store ast_json.
        facts = extract_facts_from_row(row, ast=None)
        if facts:
            rows_with_facts += 1
            for f in facts:
                per_pattern[f.pattern_name] += 1
                total_facts += 1
                if apply:
                    batch.append(f)
                    if len(batch) >= BATCH:
                        flush(batch)
                        batch = []
        if rows_seen % 200_000 == 0:
            rate = rows_seen / max(time.time() - t0, 0.001)
            print(f'  scanned {rows_seen:>9,}  facts={total_facts:>10,}  '
                  f'with-facts={rows_with_facts:>8,}  {rate:>6.0f}/s',
                  flush=True)
    if apply and batch:
        flush(batch)
    elapsed = time.time() - t0
    print(f'\n  scanned {rows_seen:,} rows in {elapsed/60:.1f} min  '
          f'({rows_seen / max(elapsed, 0.001):.0f}/s)', flush=True)
    print(f'  rows with at least one fact: {rows_with_facts:,} '
          f'({100*rows_with_facts/max(rows_seen,1):.1f}%)', flush=True)
    print(f'  total facts emitted: {total_facts:,}', flush=True)
    print(f'\n  Per-pattern breakdown:', flush=True)
    for name, n in per_pattern.most_common():
        print(f'    {name:<28s} {n:>10,}', flush=True)
    return per_pattern


def post_apply_report(conn) -> None:
    """Sample the entity_facts table to sanity-check the output."""
    print(f'\n=== entity_facts table sample ===', flush=True)
    total = conn.execute("SELECT COUNT(*) FROM entity_facts").fetchone()[0]
    print(f'  total rows: {total:,}', flush=True)

    print(f'\n  per-slot counts:', flush=True)
    for slot, n in conn.execute(
        "SELECT slot, COUNT(*) FROM entity_facts "
        "GROUP BY slot ORDER BY 2 DESC"
    ).fetchall():
        print(f'    {slot:<20s} {n:>10,}', flush=True)

    # Spot-checks: a few entities we know should appear
    for entity, slot in [('zamenhof', 'birth_place'),
                          ('zamenhof', 'birth_year'),
                          ('zamenhof', 'profession'),
                          ('esperant', 'founder'),
                          ('telefon', 'founder')]:
        rows = conn.execute(
            "SELECT value, confidence, source_sid, pattern_name "
            "FROM entity_facts WHERE entity_radiko = ? AND slot = ? "
            "ORDER BY confidence DESC LIMIT 5",
            [entity, slot]
        ).fetchall()
        if rows:
            print(f'\n  {entity}.{slot}:', flush=True)
            for val, conf, sid, pname in rows:
                print(f'    {val:<24s}  conf={conf:.2f}  sid={sid:<9d}  '
                      f'({pname})', flush=True)


def preflight() -> None:
    """Refuse to start without enough disk; warn on degraded state."""
    import subprocess
    out = subprocess.run(['df', '-k', '/'], capture_output=True, text=True)
    avail_gb = int(out.stdout.strip().split('\n')[1].split()[3]) // 1024 // 1024
    if avail_gb < 5:
        print(f'\nREFUSING: only {avail_gb} GB free. entity_facts will be '
              f'small (~50-200 MB) but DuckDB CHECKPOINT needs working space.',
              file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--limit', type=int, default=0,
                    help='Stop after N rows (0 = all). Use for smoke tests.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Run patterns, print stats; no DB writes.')
    ap.add_argument('--apply', action='store_true',
                    help='Write extracted facts to entity_facts table.')
    ap.add_argument('--fresh', action='store_true',
                    help='With --apply: DROP existing entity_facts first.')
    args = ap.parse_args()

    if not (args.dry_run or args.apply):
        ap.error('need --dry-run or --apply')

    preflight()

    mode = 'apply' if args.apply else 'dry-run'
    print(f'Opening {args.duckdb_path} ({mode})…', flush=True)
    conn = duckdb.connect(args.duckdb_path,
                          read_only=(not args.apply))
    conn.execute("SET memory_limit='2GB'")
    conn.execute("SET threads=4")

    n_sentences = conn.execute(
        "SELECT COUNT(*) FROM sentences").fetchone()[0]
    print(f'  sentences in store: {n_sentences:,}', flush=True)
    print(f'  patterns enabled:   {len(ALL_PATTERNS)}', flush=True)
    for p in ALL_PATTERNS:
        print(f'    - {p.name}', flush=True)
    print(f'  limit: {args.limit if args.limit else "ALL"}', flush=True)

    print(f'\n=== Extraction ===', flush=True)
    counter = run_extraction(conn, limit=args.limit,
                              apply=args.apply, fresh=args.fresh)

    if args.apply:
        conn.execute("CHECKPOINT")
        post_apply_report(conn)
    conn.close()
    print('\n>>> DONE', flush=True)


if __name__ == '__main__':
    main()
