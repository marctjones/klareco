#!/usr/bin/env python3
"""
Index the TREE ITSELF — one row per dependency arc. (#713)

VERSION: v1.0
COMPATIBLE WITH: v2.3 store (ast_json carries `vortoj` with kapo/rolo)
DEPENDENCIES: duckdb, klareco.parser
STAGE: Index

Description:
    THE TREE IS IN THE STORE AND NOTHING INDEXES IT.

    `ast_json` carries a full dependency tree — every token with a `kapo` (head) and
    a `rolo` (relation). Retrieval never looks at it. Every reranker scores against
    the SHREDDED COLUMNS — `subj_radiko`, `verb_radiko`, `obj_radiko` — which is a
    flat record of the MAIN CLAUSE and nothing else.

    So the structure that makes Esperanto worth parsing is sitting in a JSON blob,
    unqueryable. This table makes it queryable:

        (sid, kapo_radiko, rolo, dep_radiko)

    one row per ARC. `Zamenhof fondis Esperanton` becomes

        (sid, fond, nsubj, zamenhof)
        (sid, fond, obj,   esperant)

    AND THE QUESTION HAS A TREE TOO. That is the half nobody used. `Kiu fondis
    Esperanton?` parses to

        (fond, nsubj, kiu)      <- the GAP: `kiu` is what we are looking for
        (fond, obj,   esperant) <- the CONSTRAINT: it must be about Esperanto

    Matching ARC-TO-ARC is a different question from matching bag-of-radikoj. It
    asks "does this sentence assert THAT ESPERANTO WAS THE THING FOUNDED", not
    "does this sentence mention founding and Esperanto somewhere". BM25 cannot tell
    those apart. A tree can.

    WHY THIS IS NOT JUST `clauses`
    ------------------------------
    `clauses` gives one (subj, verb, obj) frame per clause — good, and it already
    won +0.0528 MRR (I_clause_aware). But it is still a FIXED FRAME: it cannot
    express `nmod`, `acl`, `advmod`, or any modifier. A question like

        "Kiu verkis la libron DE Petro?"

    constrains an `nmod` arc that no frame column holds. Arcs hold everything.

    PUNCTUATION IS EXCLUDED. Punct arcs are ~14% of tokens and carry no retrieval
    signal — a period hanging off the root discriminates nothing.

Pipeline Position:
    sentences.ast_json --[THIS]--> dependency_arcs --> TreeAwareReranker

Usage:
    python scripts/index/build_dependency_arcs.py --dry-run
    python scripts/index/build_dependency_arcs.py --apply

Quality Checks:
    - reports arcs/sentence and the relation histogram: a table where 90% of arcs
      are `punct` or `dep` carries no signal, however many rows it has.

Last Updated: 2026-07-14
Related Issues: #713, #836
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import pandas as pd

from klareco.parser import expand_ast

DB = 'data/indexes/duckdb_store.db'

# Arcs that carry no retrieval signal. `punct` is a period hanging off the root;
# `dep` is our own "I could not work this out" label. Indexing them would triple
# the table and discriminate nothing.
_SKIP_ROLES = frozenset({'punct', 'dep', 'root'})


def main() -> int:
    ap = argparse.ArgumentParser(description='Index the dependency arcs')
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--resume', action='store_true',
                    help='keep an existing dependency_arcs table and continue '
                         'after the last sid present (kill-resilient). Default '
                         'is a fresh DROP + full rebuild.')
    ap.add_argument('--fresh', action='store_true',
                    help='force a fresh DROP + rebuild even if a table exists')
    args = ap.parse_args()
    if not args.apply:
        args.dry_run = True

    con = duckdb.connect(args.duckdb_path, read_only=args.dry_run)
    total = con.execute('SELECT count(*) FROM sentences').fetchone()[0]
    print(f'  {total:,} sentences\n')

    # ── RESUME vs FRESH ─────────────────────────────────────────────────────
    # The build kept getting killed near the end and every fresh run redoes the
    # whole DROP+rebuild (~30 min). Keyset resume makes a kill cost nothing:
    # continue after the last sid already indexed. Safe because each batch is one
    # atomic INSERT — max(sid) reflects only fully-committed batches, and re-
    # scanning a handful of trailing zero-arc sentences produces no duplicates.
    CREATE_SQL = """
            CREATE TABLE dependency_arcs (
              sid          BIGINT,
              kapo_radiko  VARCHAR,   -- the HEAD's root
              rolo         VARCHAR,   -- the relation
              dep_radiko   VARCHAR    -- the DEPENDENT's root
            )"""
    cursor = -1  # keyset: process sids strictly greater than this
    if not args.dry_run:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'dependency_arcs'").fetchone()[0]
        if args.resume and not args.fresh and exists:
            n_existing, max_sid = con.execute(
                'SELECT count(*), max(sid) FROM dependency_arcs').fetchone()
            cursor = max_sid if max_sid is not None else -1
            print(f'  RESUME: {n_existing or 0:,} arcs already present; '
                  f'continuing after sid {cursor:,}\n')
        else:
            con.execute('DROP TABLE IF EXISTS dependency_arcs')
            con.execute(CREATE_SQL)

    rel_hist: collections.Counter = collections.Counter()
    n_arcs = 0                 # arcs WRITTEN this run
    processed = 0              # sentences PROCESSED this run
    dry_cap = 50_000 if args.dry_run else None

    while True:
        take = args.batch
        if dry_cap is not None:
            take = min(take, dry_cap - processed)
            if take <= 0:
                break
        # keyset pagination — WHERE sid > cursor is O(log n) via the sid index,
        # unlike OFFSET which re-scans every skipped row on a 4.6M-row table.
        rows = con.execute(
            'SELECT sid, ast_json FROM sentences WHERE sid > ? '
            'ORDER BY sid LIMIT ?', [cursor, take]).fetchall()
        if not rows:
            break
        cursor = rows[-1][0]
        batch = []
        for sid, aj in rows:
            if not aj:
                continue
            try:
                ast = expand_ast(json.loads(aj))
            except Exception:
                continue
            toks = ast.get('vortoj') or []
            by_id = {w['id']: w for w in toks
                     if isinstance(w, dict) and w.get('id')}
            for w in toks:
                if not isinstance(w, dict):
                    continue
                rolo = w.get('rolo')
                if not rolo or rolo in _SKIP_ROLES:
                    continue
                if w.get('vortspeco') == 'interpunkcio':
                    continue
                head = by_id.get(w.get('kapo') or 0)
                if not head:
                    continue
                hr = (head.get('radiko') or '').lower()
                dr = (w.get('radiko') or '').lower()
                if not hr or not dr:
                    continue
                rel_hist[rolo] += 1
                n_arcs += 1
                batch.append((sid, hr, rolo, dr))
        if batch and not args.dry_run:
            # ⚠️ NOT `executemany`. DuckDB round-trips every tuple through Python
            # for that, and it took 38 MINUTES to get through under 1M sentences
            # (of 4.6M) — a 3-hour projection for a table we can build in minutes.
            # Register a DataFrame and let DuckDB bulk-load it in one go.
            df = pd.DataFrame(batch, columns=['sid', 'kapo_radiko', 'rolo',
                                              'dep_radiko'])
            con.register('_arc_batch', df)
            con.execute('INSERT INTO dependency_arcs SELECT * FROM _arc_batch')
            con.unregister('_arc_batch')
        processed += len(rows)
        if processed % 200_000 == 0:
            print(f'    …{processed:,} this run (sid {cursor:,}), '
                  f'{n_arcs:,} arcs', flush=True)

    seen = processed
    print(f'  arcs                : {n_arcs:,}')
    print(f'  arcs / sentence     : {n_arcs / max(seen, 1):.1f}\n')
    print('  RELATION HISTOGRAM — a table that is 90% `dep` carries no signal,')
    print('  however many rows it has:\n')
    for r, c in rel_hist.most_common(12):
        print(f'    {r:12s} {c:9,}  ({c / n_arcs:5.1%})')

    if args.dry_run:
        print(f'\n  (dry run over {seen:,} sentences — nothing written)')
        print(f'  projected full size: ~{int(n_arcs / max(seen, 1) * total):,} arcs')
        return 0

    con.execute('CREATE INDEX IF NOT EXISTS idx_arcs_sid ON dependency_arcs(sid)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_arcs_head ON dependency_arcs(kapo_radiko)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_arcs_dep ON dependency_arcs(dep_radiko)')
    total_arcs, covered = con.execute(
        'SELECT count(*), count(DISTINCT sid) FROM dependency_arcs').fetchone()
    print(f'\n  ✓ dependency_arcs built and indexed.')
    print(f'    total arcs in table : {total_arcs:,}  '
          f'(this run wrote {n_arcs:,})')
    print(f'    sentences covered   : {covered:,} / {total:,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
