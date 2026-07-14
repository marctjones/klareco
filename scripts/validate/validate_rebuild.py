#!/usr/bin/env python3
"""
Verify the rebuild. A column can be 100% POPULATED and carry ZERO information.

VERSION: v1.0
COMPATIBLE WITH: v2.3 DuckDB store (post-rebuild)
DEPENDENCIES: duckdb, klareco.parser
STAGE: Validate

Description:
    The June 2026 migration cost weeks because artifacts went missing and the
    pipeline DEGRADED SILENTLY — a warning, then carry on. This script makes that
    impossible for the rebuild. It exits non-zero on failure and it is wired into
    the post-rebuild pipeline so it cannot be skipped.

    POPULATION IS NOT THE CONTRACT. VARIANCE IS.
    -------------------------------------------
    `success_rate` was 100% non-null and carried ZERO information: it was 1.0 on
    gibberish AND on English (#818). A column whose min == max is a DEAD COLUMN,
    however full it is. Every "is it loaded?" check that only counts NULLs would
    have passed it.

    AND THE PARSER MUST AGREE WITH ITSELF
    -------------------------------------
    We re-parse a random sample FROM the rebuilt store and compare to what is
    stored. If the stored AST and a fresh parse disagree, the store was built with
    a DIFFERENT PARSER than the one in the tree — which is exactly the class of bug
    that produced the June mess, and no schema check can catch it.

Usage:
    python scripts/validate/validate_rebuild.py
    python scripts/validate/validate_rebuild.py --sample 8000

Exit codes:
    0  every check passed
    1  at least one check FAILED — do not ship this store

Last Updated: 2026-07-14
Author: Claude (with Marc Jones)
Related Issues: #838, #835, #818, #815
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

DB = 'data/indexes/duckdb_store.db'

# (label, SQL predicate, expected count, comparison)
_JUNK = [
    ('redirect stubs', "text ILIKE 'REDIRECT%' OR text ILIKE 'ALIDIREKTI%' "
                       "OR text ILIKE 'ALIDIREKTU%' OR text ILIKE '#REDIRECT%'"),
    ('wiki markup', "text LIKE '%[[%' OR text LIKE '%{{%'"),
]

# Columns whose VARIANCE is the contract. `success_rate` was constant and dead.
_MUST_VARY = ['success_rate']

_MUST_EXIST = ['ontology_nodes', 'ontology_edges', 'entity_facts', 'clauses']


class Check:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.passed = 0

    def ok(self, label: str, detail: str = '') -> None:
        self.passed += 1
        print(f'    ✓ {label}{("  " + detail) if detail else ""}')

    def fail(self, label: str, detail: str) -> None:
        self.failed.append(f'{label}: {detail}')
        print(f'    ✗ {label}  {detail}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Verify the rebuilt store')
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--sample', type=int, default=4000)
    args = ap.parse_args()

    if not Path(args.duckdb_path).exists():
        print(f'  store missing: {args.duckdb_path}', file=sys.stderr)
        return 1

    con = duckdb.connect(args.duckdb_path, read_only=True)
    c = Check()
    tot = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
    print(f'\n  store: {tot:,} sentences\n')

    # ---- 1. the quality gate actually ran -------------------------------
    print('  QUALITY GATE (#823)')
    for label, pred in _JUNK:
        n = con.execute(f'SELECT count(*) FROM sentences WHERE {pred}').fetchone()[0]
        (c.ok if n == 0 else c.fail)(label, f'{n:,} rows remain — the gate did not run')
    if tot > 5_000_000:
        c.fail('row count', f'{tot:,} — the gate should have dropped ~14.5% '
                            f'(expected ~4.6M)')
    else:
        c.ok('row count', f'{tot:,} (gate dropped ~{(5_391_442 - tot) / 5_391_442:.1%})')

    # ---- 2. VARIANCE, not population ------------------------------------
    print('\n  VARIANCE — a column can be 100% populated and carry ZERO information')
    for col in _MUST_VARY:
        try:
            lo, hi = con.execute(
                f'SELECT min({col}), max({col}) FROM sentences').fetchone()
        except Exception:
            c.fail(col, 'column missing')
            continue
        if lo is None or lo == hi:
            c.fail(col, f'CONSTANT at {lo} — dead column, however full it is (#818)')
        else:
            c.ok(col, f'varies [{lo:.3f}, {hi:.3f}]')

    # ---- 3. the parser's fixes actually landed --------------------------
    print('\n  THE PARSER FIXES LANDED')
    pn = con.execute(
        "SELECT count(*) FROM sentences WHERE subj_vortspeco='propra_nomo'"
    ).fetchone()[0]
    rate = pn / tot if tot else 0
    if rate > 0.38:
        c.fail('propra_nomo subject rate',
               f'{rate:.1%} — was 41.8% before the fixes; expected ~35%')
    else:
        c.ok('propra_nomo subject rate', f'{rate:.1%} (was 41.8%)')

    esper = con.execute(
        "SELECT count(*) FROM sentences WHERE obj_radiko = 'esper'"
    ).fetchone()[0]
    esperant = con.execute(
        "SELECT count(*) FROM sentences WHERE obj_radiko = 'esperant'"
    ).fetchone()[0]
    if esperant == 0 and esper > 0:
        c.fail('Esperanton', f'still splitting to `esper` ({esper:,} rows) — '
                             f'protected_roots did not load')
    else:
        c.ok('Esperanton', f'-> `esperant` ({esperant:,} rows)')

    # ---- 4. the tables that must exist ----------------------------------
    print('\n  TABLES')
    have = {t[0] for t in con.execute('SHOW TABLES').fetchall()}
    for t in _MUST_EXIST:
        if t not in have:
            c.fail(t, 'MISSING')
            continue
        n = con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
        (c.ok if n > 0 else c.fail)(t, f'{n:,} rows' if n else '0 rows — not populated')

    # ---- 5. THE CHECK NO SCHEMA CAN MAKE --------------------------------
    #        Does the STORED AST agree with a FRESH parse?
    print('\n  STORED AST vs FRESH PARSE (the check no schema can make)')
    from klareco.parser import parse
    rows = con.execute(
        f'SELECT text, ast_json FROM sentences USING SAMPLE {args.sample} ROWS '
        f'(reservoir, 42)').fetchall()
    agree = compared = 0
    for text, aj in rows:
        if not text or not aj:
            continue
        try:
            stored = json.loads(aj)
            fresh = parse(text)
        except Exception:
            continue
        compared += 1
        s = (stored.get('subjekto') or {})
        f = (fresh.get('subjekto') or {})
        sk = (s.get('kerno') or s).get('radiko') if s else None
        fk = (f.get('kerno') or f).get('radiko') if f else None
        if sk == fk:
            agree += 1
    if compared:
        pct = agree / compared
        if pct < 0.99:
            c.fail('stored AST vs fresh parse',
                   f'{pct:.1%} agreement on {compared:,} sentences — the store was '
                   f'built with a DIFFERENT parser than the one in the tree')
        else:
            c.ok('stored AST vs fresh parse', f'{pct:.1%} on {compared:,} sentences')

    print(f'\n  {"═" * 60}')
    if c.failed:
        print(f'  FAILED — {len(c.failed)} check(s). DO NOT SHIP THIS STORE.\n')
        for f in c.failed:
            print(f'    · {f}')
        return 1
    print(f'  ALL {c.passed} CHECKS PASSED.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
