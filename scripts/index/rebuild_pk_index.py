#!/usr/bin/env python3
"""Surgical fix: drop+rebuild the corrupted primary-key index on sentences.

VERSION: v2.x
COMPATIBLE WITH: DuckDB store where the PK B-tree is inconsistent
DEPENDENCIES: duckdb
STAGE: Repair

Description:
    The backfill crashes deterministically at sid 2764528 with
    "duplicate primary key" — but the underlying table has exactly
    one row with that sid. The index is corrupted; the data is clean.

    This script:
      1. Inventories the constraints / indexes on `sentences`.
      2. Drops the primary-key constraint (and its implicit index).
      3. CHECKPOINTs to flush the change.
      4. Re-adds the primary key, which scans the table to build a
         fresh index over the clean data.
      5. CHECKPOINTs again.
      6. Smoke-tests with a 10-row UPDATE to the failing region.

    If the re-ADD step fails on duplicate keys, the data isn't clean
    after all and we escalate to Option 2 (EXPORT/IMPORT rebuild).

Pipeline Position:
    crashed sentences table → [THIS SCRIPT] → restored PK index
                                           → backfill can resume

Usage:
    python scripts/index/rebuild_pk_index.py

Exit 0 → resume backfill. Exit 1 → escalate to Option 2.

Last Updated: 2026-05-22
Author: Claude Code (with Marc Jones)
"""
import duckdb
import json
import sys
import time


def main() -> None:
    print('Opening DB RW…', flush=True)
    con = duckdb.connect('data/indexes/duckdb_store.db')
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")

    # 1. Inventory
    print('\n1. Constraint inventory:', flush=True)
    rows = con.execute("""
        SELECT constraint_name, constraint_type, constraint_column_names
        FROM duckdb_constraints()
        WHERE table_name = 'sentences'
    """).fetchall()
    for r in rows:
        print(f'   {r}', flush=True)
    print('\n   Index inventory:', flush=True)
    try:
        idxs = con.execute("""
            SELECT index_name, is_primary, is_unique
            FROM duckdb_indexes()
            WHERE table_name = 'sentences'
        """).fetchall()
        for r in idxs:
            print(f'   {r}', flush=True)
    except Exception as e:
        print(f'   (duckdb_indexes() failed: {e})', flush=True)

    # 2. Drop PK — DuckDB syntax variants. Try multiple forms.
    print('\n2. Dropping PRIMARY KEY constraint…', flush=True)
    pk_constraint_name = None
    for r in rows:
        if r[1] == 'PRIMARY KEY':
            pk_constraint_name = r[0]
            break
    if not pk_constraint_name:
        print('   (no PK constraint found)', flush=True)
        sys.exit(1)
    drop_stmts = [
        f"ALTER TABLE sentences DROP CONSTRAINT {pk_constraint_name}",
        f'ALTER TABLE sentences DROP CONSTRAINT "{pk_constraint_name}"',
    ]
    dropped = False
    for stmt in drop_stmts:
        t0 = time.time()
        try:
            con.execute(stmt)
            print(f'   ok via {stmt!r} in {time.time()-t0:.1f}s', flush=True)
            dropped = True
            break
        except Exception as e:
            print(f'   {stmt!r} -> {type(e).__name__}: {str(e)[:120]}',
                  flush=True)
    if not dropped:
        print('   could not drop PK via any tried syntax', flush=True)
        sys.exit(1)

    print('   CHECKPOINT…', flush=True)
    t0 = time.time()
    con.execute("CHECKPOINT")
    print(f'   done in {time.time()-t0:.1f}s', flush=True)

    # 3. Verify no duplicate sids before re-adding PK
    print('\n3. Verifying no duplicate sids in data…', flush=True)
    t0 = time.time()
    dups = con.execute("""
        SELECT sid, COUNT(*) as n FROM sentences
        GROUP BY sid HAVING n > 1 LIMIT 5
    """).fetchall()
    print(f'   scan took {time.time()-t0:.1f}s, dups found: {len(dups)}',
          flush=True)
    if dups:
        for d in dups:
            print(f'   DUP sid={d[0]} count={d[1]}', flush=True)
        print('\n   DATA IS NOT CLEAN — cannot re-add PK. Escalate to Option 2.',
              flush=True)
        sys.exit(1)

    # 4. Re-add PK
    print('\n4. Re-adding PRIMARY KEY (forces index rebuild)…', flush=True)
    t0 = time.time()
    try:
        con.execute("ALTER TABLE sentences ADD PRIMARY KEY (sid)")
        print(f'   ok in {time.time()-t0:.1f}s', flush=True)
    except Exception as e:
        print(f'   FAILED: {type(e).__name__}: {str(e)[:300]}', flush=True)
        sys.exit(1)

    print('   CHECKPOINT…', flush=True)
    t0 = time.time()
    con.execute("CHECKPOINT")
    print(f'   done in {time.time()-t0:.1f}s', flush=True)

    # 5. Smoke test — update a few unapplied rows
    print('\n5. Smoke test: 10 UPDATEs against unapplied staging rows…',
          flush=True)
    batch = []
    with open('data/staging/subj_propranoma_kat_backfill.jsonl') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r['sid']
            cur = con.execute(
                "SELECT subj_propranoma_kat FROM sentences WHERE sid = ?",
                [sid],
            ).fetchone()
            if cur and cur[0] is None:
                batch.append((sid, r['kat']))
            if len(batch) >= 10:
                break
    print(f'   found {len(batch)} unapplied rows', flush=True)

    errors = []
    t0 = time.time()
    for sid, kat in batch:
        try:
            con.execute(
                "UPDATE sentences SET subj_propranoma_kat = ? WHERE sid = ?",
                [kat, sid],
            )
        except Exception as e:
            errors.append((sid, str(e)[:150]))
    print(f'   {time.time()-t0:.1f}s, errors: {len(errors)}', flush=True)
    for e in errors[:3]:
        print(f'   {e}', flush=True)

    if errors:
        print('\n>>> Index rebuild SUCCEEDED but smoke UPDATE still fails. <<<',
              flush=True)
        sys.exit(1)

    # 6. Specifically test sid 2764528 (the killer row)
    print('\n6. Targeted test on the killer sid 2764528…', flush=True)
    try:
        t0 = time.time()
        con.execute(
            "UPDATE sentences SET subj_propranoma_kat = subj_propranoma_kat "
            "WHERE sid = 2764528"
        )
        print(f'   no-op UPDATE on sid 2764528 ok in {time.time()-t0:.2f}s',
              flush=True)
    except Exception as e:
        print(f'   FAILED: {type(e).__name__}: {str(e)[:200]}', flush=True)
        sys.exit(1)

    print('   final CHECKPOINT…', flush=True)
    con.execute("CHECKPOINT")
    con.close()
    print('\n>>> Option 1.5 SUCCESS. Index rebuilt. Backfill can resume. <<<',
          flush=True)
    sys.exit(0)


if __name__ == '__main__':
    main()
