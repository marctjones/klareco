#!/usr/bin/env python3
"""
Build question-shape pattern KB tables (gh#732) using the two-phase pattern.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store with sentences.text + ast_json
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index

Description:
    GitHub issue: #732
    Builds direct-lookup tables for the predictable trivia factoid
    patterns:

      pattern_capital_of           (city, country, sid, confidence)
      pattern_founded_year_of      (org, year, sid, confidence)
      pattern_official_language_of (language, country, sid, confidence)
      pattern_currency_of          (currency, country, sid, confidence)

    For each pattern, scan the corpus with a surface-text LIKE filter,
    extract the (X, Y) tuple via regex, write to a JSONL staging file.
    Phase B bulk-loads into DuckDB tables + builds indices.

    Question routing (ASTRetriever) then bypasses BM25 for any question
    matching these shapes — direct point lookup, deterministic answer.

Pipeline Position:
    sentences.text → [THIS SCRIPT] → pattern_*_of tables → ASTRetriever

Usage:
    # Phase A — read-only scan, write staging JSONL per pattern:
    python scripts/index/build_pattern_kb.py --scan-only

    # Phase B — bulk-load staging files into DuckDB tables:
    python scripts/index/build_pattern_kb.py --apply

Inputs:
    --duckdb-path  data/indexes/duckdb_store.db
    --staging-dir  data/staging/pattern_kb/  (per-pattern JSONL files)

Outputs:
    Phase A: data/staging/pattern_kb/<pattern>.jsonl
    Phase B: pattern_<name>_of tables in DuckDB + indices

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb


# Reusable proper-noun span regex
_PROPER_NOUN = (
    r'[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]{2,}'
    r'(?:\s+[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]+){0,3}'
)


class Pattern(NamedTuple):
    name:         str               # 'capital_of', 'founded_year_of', ...
    like_filter:  list[str]         # SQL LIKE patterns to narrow scan
    extractor:    Callable          # (text: str) -> tuple[str, str] | None
    table_name:   str               # 'pattern_capital_of'
    columns:      list[str]         # ['city', 'country']  (X, Y)


# ---------------------------------------------------------------------------
# Per-pattern surface-text extractors. Each returns (X, Y) or None.
# ---------------------------------------------------------------------------

_CAPITAL_RE = re.compile(
    r'(?P<entity>' + _PROPER_NOUN + r')'
    r'(?:,|\s+estas|\s+iĝis)\s+'
    r'la\s+ĉefurbo\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')',
    re.UNICODE,
)

_FOUNDED_RE = re.compile(
    r'(?P<org>' + _PROPER_NOUN + r')'
    r'(?:\s*\([^)]*\))?'
    r'\s+(?:fondiĝis|estis\s+fondita|estis\s+kreita|estis\s+establita|fondita)\s+'
    r'(?:en\s+(?:la\s+jaro\s+)?)?'
    r'(?P<year>1[0-9]{3}|20[0-2][0-9]|[789]\d{2})',
    re.UNICODE,
)

_LINGVO_A_RE = re.compile(
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)'
    r'\s+estas\s+'
    r'la\s+oficiala\s+lingvo\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')',
    re.UNICODE,
)
_LINGVO_B_RE = re.compile(
    r'la\s+oficiala\s+lingvo\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')'
    r'\s+estas\s+'
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)',
    re.UNICODE,
)

_VALUTO_A_RE = re.compile(
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ\s-]+?)'
    r'\s+estas\s+'
    r'la\s+(?:oficiala\s+)?valuto\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')',
    re.UNICODE,
)
_VALUTO_B_RE = re.compile(
    r'la\s+(?:oficiala\s+)?valuto\s+de\s+'
    r'(?P<country>' + _PROPER_NOUN + r')'
    r'\s+estas\s+'
    r'(?P<value>(?:la\s+)?[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+(?:\s+[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+){0,2})',
    re.UNICODE,
)


def extract_capital(text: str):
    m = _CAPITAL_RE.search(text)
    if not m:
        return None
    return (m.group('entity').strip(), m.group('country').strip())


def extract_founded(text: str):
    m = _FOUNDED_RE.search(text)
    if not m:
        return None
    org = m.group('org').strip()
    year = m.group('year').strip()
    y = int(year)
    if y < 700 or y > 2030:
        return None
    return (org, year)


def extract_official_language(text: str):
    m = _LINGVO_A_RE.search(text) or _LINGVO_B_RE.search(text)
    if not m:
        return None
    value = m.group('value').strip().rstrip('.,;:')
    if value.lower().startswith('la '):
        value = value[3:].strip()
    if len(value) < 3:
        return None
    country = m.group('country').strip()
    return (value, country)


def extract_currency(text: str):
    m = _VALUTO_A_RE.search(text) or _VALUTO_B_RE.search(text)
    if not m:
        return None
    value = m.group('value').strip().rstrip('.,;:')
    if value.lower().startswith('la '):
        value = value[3:].strip()
    if len(value) < 3 or len(value) > 60:
        return None
    country = m.group('country').strip()
    return (value, country)


PATTERNS: list[Pattern] = [
    Pattern(
        name='capital_of',
        like_filter=['%la ĉefurbo de%'],
        extractor=extract_capital,
        table_name='pattern_capital_of',
        columns=['city', 'country'],
    ),
    Pattern(
        name='founded_year_of',
        like_filter=['%fondiĝis en%', '%estis fondita en%',
                     '%estis kreita en%', '%estis establita en%'],
        extractor=extract_founded,
        table_name='pattern_founded_year_of',
        columns=['org', 'year'],
    ),
    Pattern(
        name='official_language_of',
        like_filter=['%la oficiala lingvo de%', '%oficiala lingvo de%'],
        extractor=extract_official_language,
        table_name='pattern_official_language_of',
        columns=['language', 'country'],
    ),
    Pattern(
        name='currency_of',
        like_filter=['%la valuto de%', '%la oficiala valuto de%'],
        extractor=extract_currency,
        table_name='pattern_currency_of',
        columns=['currency', 'country'],
    ),
]


# ---------------------------------------------------------------------------
# Phase A — scan + write staging
# ---------------------------------------------------------------------------

def phase_a_one_pattern(conn, pat: Pattern, staging_dir: Path,
                       fresh: bool = False) -> int:
    staging_path = staging_dir / f'{pat.name}.jsonl'
    staging_dir.mkdir(parents=True, exist_ok=True)
    if fresh and staging_path.exists():
        staging_path.unlink()

    where_clauses = ' OR '.join('text LIKE ?' for _ in pat.like_filter)
    params = list(pat.like_filter)

    print(f'\n--- Pattern: {pat.name} ---')
    n_candidates = conn.execute(
        f'SELECT COUNT(*) FROM sentences WHERE {where_clauses}', params
    ).fetchone()[0]
    print(f'  candidate sentences: {n_candidates:,}')

    cursor = conn.execute(
        f'SELECT sid, text FROM sentences WHERE {where_clauses}',
        params
    )

    n_extracted = 0
    t0 = time.time()
    with open(staging_path, 'w') as f:
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            sid, text = row
            if not text:
                continue
            tup = pat.extractor(text)
            if tup is None:
                continue
            x, y = tup
            record = {
                pat.columns[0]: x,
                pat.columns[1]: y,
                'sid':          int(sid),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            n_extracted += 1
    elapsed = time.time() - t0
    print(f'  extracted: {n_extracted:,} tuples in {elapsed:.1f}s '
          f'({n_extracted/elapsed:.0f}/s)')
    print(f'  staging: {staging_path}')
    return n_extracted


def phase_a(args) -> None:
    """Scan all patterns; write per-pattern staging files."""
    print(f'Opening DuckDB at {args.duckdb_path} (READ-ONLY)…')
    conn = duckdb.connect(args.duckdb_path, read_only=True)

    staging_dir = Path(args.staging_dir)
    total = 0
    for pat in PATTERNS:
        total += phase_a_one_pattern(conn, pat, staging_dir, fresh=args.fresh)

    print(f'\n=== Phase A done ===')
    print(f'Total tuples extracted across all patterns: {total:,}')


# ---------------------------------------------------------------------------
# Phase B — bulk-load staging into pattern_*_of tables
# ---------------------------------------------------------------------------

def phase_b_one_pattern(conn, pat: Pattern, staging_dir: Path) -> int:
    staging_path = staging_dir / f'{pat.name}.jsonl'
    if not staging_path.exists():
        print(f'  ✗ {pat.name}: staging file missing ({staging_path})')
        return 0
    print(f'\n--- Applying: {pat.name} ---')

    conn.execute(f'DROP TABLE IF EXISTS {pat.table_name}')
    col_x, col_y = pat.columns
    conn.execute(f"""
        CREATE TABLE {pat.table_name} AS
        SELECT
            {col_x},
            {col_y},
            CAST(sid AS BIGINT) AS sid
        FROM read_json_auto('{staging_path.absolute()}',
                            format='nd', records=true)
    """)
    n = conn.execute(f'SELECT COUNT(*) FROM {pat.table_name}').fetchone()[0]
    print(f'  loaded {n:,} rows')

    # Build index on the constraint side (Y). For capital_of that's `country`,
    # which is what queries filter on. Same for founded_year_of (`org`).
    conn.execute(
        f'CREATE INDEX idx_{pat.table_name}_{col_y} ON {pat.table_name}({col_y})'
    )
    print(f'  built index on {col_y}')
    return n


def phase_b(args) -> None:
    print(f'Opening DuckDB at {args.duckdb_path} (WRITE)…')
    conn = duckdb.connect(args.duckdb_path)
    staging_dir = Path(args.staging_dir)
    total = 0
    for pat in PATTERNS:
        total += phase_b_one_pattern(conn, pat, staging_dir)
    print(f'\n=== Phase B done ===')
    print(f'Total rows loaded across {len(PATTERNS)} pattern tables: {total:,}')

    # Quick stats per table
    print(f'\n=== Per-table stats ===')
    for pat in PATTERNS:
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM {pat.table_name}').fetchone()[0]
            n_distinct_y = conn.execute(
                f'SELECT COUNT(DISTINCT {pat.columns[1]}) FROM {pat.table_name}'
            ).fetchone()[0]
            print(f'  {pat.table_name:<35s}  {n:>7,} rows, '
                  f'{n_distinct_y:>5,} distinct {pat.columns[1]}')
        except Exception as e:
            print(f'  {pat.table_name}: {e}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--staging-dir', default='data/staging/pattern_kb')
    ap.add_argument('--scan-only', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='Phase A: overwrite per-pattern staging files')
    args = ap.parse_args()

    if not args.scan_only and not args.apply:
        print('ERROR: must specify --scan-only and/or --apply', file=sys.stderr)
        sys.exit(1)
    if args.scan_only:
        phase_a(args)
    if args.apply:
        phase_b(args)


if __name__ == '__main__':
    main()
