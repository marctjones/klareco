#!/usr/bin/env python3
"""Quick smoke test: can we run a small batch of UPDATEs after CHECKPOINT?

VERSION: v2.x
COMPATIBLE WITH: post-crash DuckDB store
STAGE: Diagnostics

Description:
    The backfill crashed with an INTERNAL "duplicate primary key" error
    during commit. The table data is clean (verified — 5.39M rows, 0
    duplicates). This script tests whether a CHECKPOINT + small batch
    of real UPDATEs now succeeds.

Usage:
    python scripts/index/test_index_recovery.py

Exit code 0 = success (proceed to full backfill resume).
Exit code 1 = failure (escalate to Option 2 rebuild).

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
import duckdb
import json
import sys
import time


def main() -> None:
    print('Open RW…', flush=True)
    con = duckdb.connect('data/indexes/duckdb_store.db')
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")

    print('Finding 10 NULL-subj_propranoma_kat staging rows…', flush=True)
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
    print(f'  found {len(batch)} unapplied rows', flush=True)

    print('\n10-row UPDATE smoke test…', flush=True)
    t0 = time.time()
    errors = []
    for sid, kat in batch:
        try:
            con.execute(
                "UPDATE sentences SET subj_propranoma_kat = ? WHERE sid = ?",
                [kat, sid],
            )
        except Exception as e:
            errors.append((sid, type(e).__name__, str(e)[:150]))
    print(f'  elapsed: {time.time()-t0:.2f}s, errors: {len(errors)}',
          flush=True)
    for e in errors[:3]:
        print(f'  {e}', flush=True)

    ok = (len(errors) == 0)
    if ok:
        print('\nFinal CHECKPOINT…', flush=True)
        t0 = time.time()
        try:
            con.execute("CHECKPOINT")
            print(f'  done in {time.time()-t0:.1f}s', flush=True)
        except Exception as e:
            print(f'  CHECKPOINT FAILED: {type(e).__name__}: '
                  f'{str(e)[:200]}', flush=True)
            ok = False

    con.close()
    print(f'\nVERDICT: {"PROCEED" if ok else "ESCALATE"}', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
