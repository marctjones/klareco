#!/usr/bin/env python3
"""
Build the CLAUSE table — from ast_json, WITHOUT reparsing.

VERSION: v1.0
COMPATIBLE WITH: v2.3 store (ast_json carries `propozicioj`)
DEPENDENCIES: duckdb
STAGE: Index

Description:
    The shredded columns keep ONE subject:

        sentences(sid, text, ast_json, subj_radiko, verb_radiko, obj_radiko, …)

    `subj_radiko` holds the MAIN clause's subject. Gold has **1.64 subjects per
    sentence** and **35.8% of sentences have two or more**. So every subordinate
    and coordinate clause was silently discarded — and `DuckDBRetriever` and all
    nine rerankers read exactly these columns.

    NO REPARSE IS NEEDED. `ast_json` carries the clause tree (`propozicioj`), so
    the shredded columns can be rebuilt FROM THE BLOB. That decouples two things
    that were wrongly coupled:

        parser changes  ->  need a reparse (cheap: ~15 min)
        schema changes  ->  read ast_json.  NO reparse.

    ⚠️ NEW-TABLE-SWAP, NOT `ALTER TABLE` + `UPDATE`. CLAUDE.md's own Stage-2
    lesson: in-place bulk changes on a 5M-row table leave ~30 GB of dead pages
    that DuckDB never reclaims.

    HONEST EXPECTATION-SETTING
    --------------------------
    Real corpus text averages **1.08 clauses per sentence**; the Prago gold
    averages 1.64. Wikipedia sentences are simpler than a political manifesto. So
    this will gain LESS on this corpus than the gold numbers suggest. It is still
    correct, and it still unblocks the rerankers — but do not promise a 60%
    retrieval jump because subject recall rose 60% on gold.

Usage:
    python scripts/index/build_clause_table.py
    python scripts/index/build_clause_table.py --dry-run

Outputs:
    - table `clauses` in data/indexes/duckdb_store.db

Last Updated: 2026-07-14
Related Issues: #836, #831, #713
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

DB = 'data/indexes/duckdb_store.db'

SCHEMA = """
CREATE TABLE clauses_new (
  sid            BIGINT,
  clause_idx     INTEGER,      -- 0 = main
  rolo           VARCHAR,      -- ĉefa | kunordigita | subordigita | rilativa
  subj_radiko    VARCHAR,  subj_vortspeco VARCHAR,  subj_kazo VARCHAR,
  verb_radiko    VARCHAR,  verb_tempo     VARCHAR,  verb_negated BOOLEAN,
  obj_radiko     VARCHAR,  obj_kazo       VARCHAR,
  verb_klaso     VARCHAR       -- populated by load_ontology.py (#837)
)
"""


def _kern(node):
    if not isinstance(node, dict):
        return {}
    return node.get('kerno', node) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description='Build the clause table from ast_json')
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--batch', type=int, default=200_000)
    args = ap.parse_args()

    con = duckdb.connect(args.duckdb_path, read_only=args.dry_run)
    tot = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
    print(f'  {tot:,} sentences')

    if not args.dry_run:
        con.execute('DROP TABLE IF EXISTS clauses_new')
        con.execute(SCHEMA)

    n_rows = n_multi = 0
    offset = 0
    while offset < tot:
        rows = con.execute(
            'SELECT sid, ast_json FROM sentences ORDER BY sid '
            f'LIMIT {args.batch} OFFSET {offset}').fetchall()
        if not rows:
            break
        batch = []
        for sid, aj in rows:
            if not aj:
                continue
            try:
                ast = json.loads(aj)
            except Exception:
                continue
            clauses = ast.get('propozicioj')
            if not clauses:
                # pre-tree AST: fall back to the flat frame as clause 0
                clauses = [{'rolo': 'ĉefa', 'subjekto': ast.get('subjekto'),
                            'verbo': ast.get('verbo'), 'objekto': ast.get('objekto')}]
            if len(clauses) > 1:
                n_multi += 1
            for i, c in enumerate(clauses):
                s, v, o = _kern(c.get('subjekto')), _kern(c.get('verbo')), _kern(c.get('objekto'))
                batch.append((
                    sid, i, c.get('rolo'),
                    s.get('radiko'), s.get('vortspeco'), s.get('kazo'),
                    v.get('radiko'), v.get('tempo'), bool(v.get('negita')),
                    o.get('radiko'), o.get('kazo'),
                    None,
                ))
        n_rows += len(batch)
        if batch and not args.dry_run:
            con.executemany(
                'INSERT INTO clauses_new VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', batch)
        offset += args.batch
        if offset % 1_000_000 == 0:
            print(f'    …{offset:,} sentences, {n_rows:,} clauses')

    print(f'\n  clauses            : {n_rows:,}')
    print(f'  clauses/sentence   : {n_rows / tot:.2f}')
    print(f'  MULTI-CLAUSE rows  : {n_multi:,}  ({n_multi / tot:.1%})')
    print(f'    -> every one of those had its subordinate clauses DISCARDED by the '
          f'flat schema')

    if args.dry_run:
        print('\n  (dry run — nothing written)')
        return 0

    # NEW-TABLE-SWAP. Never ALTER+UPDATE on 5M rows: it leaves ~30 GB of dead
    # pages that DuckDB does not reclaim (CLAUDE.md, the Stage-2 lesson).
    con.execute('DROP TABLE IF EXISTS clauses')
    con.execute('ALTER TABLE clauses_new RENAME TO clauses')
    con.execute('CREATE INDEX idx_clauses_sid ON clauses(sid)')
    con.execute('CREATE INDEX idx_clauses_subj ON clauses(subj_radiko)')
    con.execute('CREATE INDEX idx_clauses_verb ON clauses(verb_radiko)')
    print('\n  wrote table `clauses` (new-table-swap; no dead pages)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
