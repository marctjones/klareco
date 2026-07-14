#!/usr/bin/env python3
"""
Prune the junk rows the quality gate should have caught — using the GATE ITSELF.

VERSION: v1.0
COMPATIBLE WITH: v2.3 DuckDB store
DEPENDENCIES: duckdb, klareco.corpus_quality
STAGE: Index

Description:
    The 2026-07-14 rebuild produced a sound store — 100.0% stored-AST-vs-fresh-parse
    agreement, every table populated — but 6,103 junk rows survived the quality
    gate, because the gate had two holes:

        5,247  redirect stubs   `RedirectKantono Apencelo Ekstera`
                                The extractor GLUES the keyword to the title, and
                                the gate's regex ended in `\\b` (a word boundary),
                                so `Redirect` followed by `K` never matched.
          856  unclosed markup  `la [[Karikaturmuseum|Karikaturmusuem ...`
                                `strip_markup` needs matched brackets, so it could
                                not touch these and they sailed through.

    Both holes are fixed in `klareco.corpus_quality` (1215b49). Re-parsing 5.4M
    sentences to drop 6,103 rows would be 50 minutes to change 0.13% of the store,
    so we prune in place instead.

    ⚠️ THIS SCRIPT DOES NOT RE-IMPLEMENT THE RULES. It calls `assess()` — the SAME
    function the build uses. Writing the redirect test a second time here is exactly
    how the original bug happened: `rebuild_whoosh_from_duckdb.py` had its own copy
    of the redirect filter, and the two drifted. One gate, one definition, called
    from both places, or this recurs.

    A side effect worth having: because it runs the real gate over EVERY row, it
    also VERIFIES the fixed gate against the whole store rather than a sample.

Pipeline Position:
    duckdb_store --[THIS]--> pruned store --> rebuild_whoosh_from_duckdb.py

Usage:
    python scripts/index/prune_junk_rows.py --dry-run     # count, change nothing
    python scripts/index/prune_junk_rows.py --apply

Inputs:
    - data/indexes/duckdb_store.db

Outputs:
    - the same store, with junk rows removed from `sentences` AND `clauses`

Quality Checks:
    - reports the reason for every drop, so a gate that suddenly deletes 20% of the
      corpus is visible BEFORE --apply, not after.
    - refuses to delete more than --max-drop-pct (default 2%) without --force.
      The gate that nearly deleted 569,000 good sentences is why.

Last Updated: 2026-07-14
Related Issues: #835, #823
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.corpus_quality import assess

DB = 'data/indexes/duckdb_store.db'


def main() -> int:
    ap = argparse.ArgumentParser(description='Prune junk rows using the real gate')
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-drop-pct', type=float, default=2.0,
                    help='refuse to drop more than this share without --force')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if not args.apply and not args.dry_run:
        args.dry_run = True

    con = duckdb.connect(args.duckdb_path, read_only=args.dry_run)
    total = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
    print(f'  store: {total:,} sentences')
    print('  running the REAL gate (klareco.corpus_quality.assess) over every row…\n')

    junk: list[int] = []
    reasons: collections.Counter = collections.Counter()
    examples: dict[str, str] = {}

    cur = con.execute('SELECT sid, text FROM sentences')
    seen = 0
    while True:
        rows = cur.fetchmany(100_000)
        if not rows:
            break
        for sid, text in rows:
            seen += 1
            v = assess(text or '')
            if not v.keep:
                junk.append(sid)
                reasons[v.reason] += 1
                examples.setdefault(v.reason, (text or '')[:60])
        print(f'    …{seen:,}/{total:,}  junk so far {len(junk):,}', flush=True)

    pct = len(junk) / total * 100 if total else 0
    print(f'\n  JUNK: {len(junk):,} of {total:,} = {pct:.2f}%\n')
    print(f'  {"reason":16s} {"rows":>8s}   example')
    for r, n in reasons.most_common():
        print(f'  {r:16s} {n:8,}   {examples[r]!r}')

    if pct > args.max_drop_pct and not args.force:
        print(f'\n  ✗ REFUSING: {pct:.2f}% exceeds --max-drop-pct {args.max_drop_pct}%.')
        print('    An early version of this gate scored token PURITY and would have')
        print('    deleted ~569,000 GOOD Esperanto sentences that merely QUOTE a')
        print('    foreign title. A gate that suddenly wants a fifth of your corpus')
        print('    is a bug in the gate. Use --force only when you have looked.')
        return 1

    if args.dry_run:
        print('\n  (dry run — nothing deleted). Re-run with --apply.')
        return 0

    if not junk:
        print('\n  nothing to do.')
        return 0

    con.execute('CREATE OR REPLACE TEMP TABLE _junk (sid BIGINT)')
    con.executemany('INSERT INTO _junk VALUES (?)', [(s,) for s in junk])

    have = {t[0] for t in con.execute('SHOW TABLES').fetchall()}
    if 'clauses' in have:
        n_cl = con.execute(
            'SELECT count(*) FROM clauses WHERE sid IN (SELECT sid FROM _junk)'
        ).fetchone()[0]
        con.execute('DELETE FROM clauses WHERE sid IN (SELECT sid FROM _junk)')
        print(f'\n  deleted {n_cl:,} rows from `clauses`')

    con.execute('DELETE FROM sentences WHERE sid IN (SELECT sid FROM _junk)')
    left = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
    print(f'  deleted {total - left:,} rows from `sentences`')
    print(f'  store now: {left:,} sentences')
    print('\n  ⚠️  Whoosh is now STALE — it indexes the deleted sids.')
    print('     Next: python scripts/index/rebuild_whoosh_from_duckdb.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
