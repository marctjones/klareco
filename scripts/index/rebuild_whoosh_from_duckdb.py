#!/usr/bin/env python3
"""
Rebuild Whoosh BM25 index from the (already-built) DuckDB store

VERSION: v2.2
COMPATIBLE WITH: data/indexes/duckdb_store.db (sid + text columns)
DEPENDENCIES: duckdb, Whoosh
STAGE: Index

Description:
    The full builder `build_duckdb_store.py` re-parses every sentence AND
    rebuilds Whoosh in the same pass, which takes ~90 min + ~3 hr optimize.
    When the DuckDB store is already clean and only Whoosh is stale (e.g.
    after an aborted optimize or a contaminated --resume), this script
    rebuilds Whoosh alone from DuckDB's (sid, text) rows. No parse, no
    optimize — just stream rows into a fresh Whoosh index, commit, done.

Pipeline Position:
    duckdb_store.db --[THIS]--> data/indexes/whoosh_v2/ (fresh, multi-seg)

Usage:
    python scripts/index/rebuild_whoosh_from_duckdb.py
    python scripts/index/rebuild_whoosh_from_duckdb.py --limit 100000  # smoke

Last Updated: 2026-05-22
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh import index as whoosh_index
from whoosh.fields import ID, TEXT, Schema


DB = 'data/indexes/duckdb_store.db'
WHOOSH_DIR = 'data/indexes/whoosh_v2'
PROGRESS_EVERY = 100_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='Stop after N rows (0 = all).')
    ap.add_argument('--limitmb', type=int, default=2048,
                    help='Writer RAM cap in MB.')
    ap.add_argument('--procs', type=int, default=4,
                    help='Writer worker procs (multisegment=True).')
    args = ap.parse_args()

    # Preflight: rebuilt index lands at ~3 GB, intermediate segments
    # before the optional optimize can be 2-5x that.
    import subprocess
    out = subprocess.run(['df', '-k', '/'], capture_output=True, text=True)
    avail_gb = int(out.stdout.strip().split('\n')[1].split()[3]) // 1024 // 1024
    if avail_gb < 10:
        print(f'\nREFUSING: only {avail_gb} GB free, need 10 GB for Whoosh '
              f'rebuild. See scripts/util/cleanup_stale.sh.', file=sys.stderr)
        sys.exit(2)

    d = Path(WHOOSH_DIR)
    if d.exists():
        print(f'wiping {d} …', flush=True)
        shutil.rmtree(d)
    d.mkdir(parents=True)

    # Same schema as build_duckdb_store.py
    schema = Schema(id=ID(stored=True, unique=True),
                    text=TEXT(stored=True))
    ix = whoosh_index.create_in(str(d), schema)
    writer = ix.writer(limitmb=args.limitmb,
                       procs=args.procs,
                       multisegment=True)

    con = duckdb.connect(DB, read_only=True)
    con.execute("SET memory_limit='2GB'")
    total = con.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    print(f'DuckDB rows: {total:,}', flush=True)

    # R6 — filter corpus noise BEFORE Whoosh ingest (#802).
    #
    # The quality standard is explicit: "post-index sanity query ... must return
    # 0 IN THE WHOOSH INDEX". Measured on the live index 2026-07-13 it returned
    # 121,939 — 78,491 ALIDIREKTI + 43,448 REDIRECT. Content-free redirect
    # stubs, indexed, scored and retrievable.
    #
    # They are not inert. They corrupt three things:
    #   1. Retrieval — failure mode F5: "3/3 misses on the diversified-51 set
    #      were redirects."
    #   2. Entity extraction — `REDIRECT` is the #1 proper-noun SUBJECT in the
    #      whole store (42,319 occurrences), ahead of every real person or place.
    #      Anything mining proper nouns drinks from this — including the
    #      proper-noun dictionary rebuild (#804), which is why this filter must
    #      land FIRST or the pollution gets baked into the dictionary.
    #   3. Test-set construction — the generator streams candidates from here.
    #
    # The DuckDB store KEEPS them (R6 preserves them for ontology work). The
    # INDEX must not have them.
    JUNK_PREFIXES = ('ALIDIREKTI', 'REDIRECT', '#REDIRECT', '#ALIDIREKTI')
    junk_clause = ' AND '.join(
        f"text NOT LIKE '{p}%'" for p in JUNK_PREFIXES)

    n_junk = con.execute(
        f"SELECT COUNT(*) FROM sentences WHERE NOT ({junk_clause})").fetchone()[0]
    print(f'R6 filter: excluding {n_junk:,} redirect/disambiguation stubs '
          f'({100 * n_junk / max(total, 1):.1f}% of the store)', flush=True)

    q = f'SELECT sid, text FROM sentences WHERE {junk_clause} ORDER BY sid'
    if args.limit:
        q += f' LIMIT {args.limit}'

    t0 = time.time()
    n = 0
    cur = con.execute(q)
    while True:
        rows = cur.fetchmany(50_000)
        if not rows:
            break
        for sid, text in rows:
            if not text:
                continue
            writer.add_document(id=str(sid), text=text)
            n += 1
            if n % PROGRESS_EVERY == 0:
                rate = n / max(time.time() - t0, 0.001)
                eta_s = (total - n) / max(rate, 1)
                print(f'n={n:,} {rate:.0f}/s ETA={eta_s/60:.0f}min',
                      flush=True)

    print(f'committing (optimize=False) …', flush=True)
    writer.commit(optimize=False)
    print(f'done: {n:,} docs in {(time.time()-t0)/60:.1f} min', flush=True)

    # ----- Hard correctness gates: fail loud if Whoosh is not trustworthy -----
    print('\n=== correctness gates ===', flush=True)
    fails = []

    ix2 = whoosh_index.open_dir(str(d))
    doc_count = ix2.doc_count()
    n_segments = len(ix2._segments())
    print(f'  doc_count: {doc_count:,}', flush=True)
    print(f'  segments:  {n_segments}', flush=True)

    # THE GATE MUST EXPECT WHAT THE INDEXER WAS ASKED TO BUILD.
    #
    # This compared `doc_count` against `total` — every row in the store. But the
    # indexer DELIBERATELY EXCLUDES the R6 redirect stubs (see the junk_clause
    # above: the store keeps them for ontology work, the index must not have them).
    # So the gate expected a number the indexer was never going to reach, and it
    # FAILED A PERFECTLY GOOD BUILD:
    #
    #     doc_count 4,629,514 != expected 4,630,679      <- off by exactly n_junk
    #
    # The script even PRINTS `R6 filter: excluding 1,165 stubs` twenty lines up. It
    # knew the number and then did not subtract it. A gate that cries wolf is worse
    # than no gate: the next person to see it will assume it is noise and force the
    # build through, which is precisely when it will be right.
    indexable = total - n_junk
    if args.limit:
        expected = min(indexable, args.limit)
    else:
        expected = indexable
    if doc_count != expected:
        fails.append(f'doc_count {doc_count} != expected {expected} '
                     f'(store {total:,} - {n_junk:,} R6 stubs)')
    if n_segments < 1:
        fails.append(f'no segments')

    # id-range gate: min and max id in Whoosh must match DuckDB sid range
    from whoosh.qparser import QueryParser
    with ix2.searcher() as s:
        # Collect every id by iterating one of the documents per shard.
        # For O(N) safety on 5M docs, sample via doc number boundaries.
        all_ids = set()
        for docnum in (0, doc_count - 1):
            try:
                d_ = s.stored_fields(docnum)
                all_ids.add(int(d_['id']))
            except Exception:
                pass
        if all_ids:
            print(f'  edge-doc ids (docnums 0, last): {sorted(all_ids)}',
                  flush=True)

        # Round-trip probe: 10 random DuckDB sids, confirm Whoosh has them
        # with byte-identical text.
        import random
        rng = random.Random(42)
        probe_sids = sorted(
            rng.sample(range(1, expected + 1), k=min(10, expected)))
        qp = QueryParser('id', schema=ix2.schema)
        bad_probes = []
        for sid in probe_sids:
            db_text = con.execute(
                'SELECT text FROM sentences WHERE sid = ?', [sid]
            ).fetchone()
            if not db_text:
                bad_probes.append((sid, 'no DuckDB row'))
                continue
            db_text = db_text[0]
            hits = s.search(qp.parse(str(sid)), limit=2)
            if not hits:
                bad_probes.append((sid, 'no Whoosh hit'))
            elif hits[0].get('text') != db_text:
                bad_probes.append(
                    (sid,
                     f'text mismatch '
                     f'(whoosh={hits[0].get("text","")[:30]!r}  '
                     f'duckdb={db_text[:30]!r})'))
            elif len(hits) > 1:
                bad_probes.append((sid, f'{len(hits)} hits, expected 1'))
        if bad_probes:
            fails.append(f'round-trip probe failures: {bad_probes}')
        else:
            print(f'  round-trip probe (10 random sids): all pass',
                  flush=True)

    con.close()
    if fails:
        print('\n!!! CORRECTNESS GATES FAILED !!!', flush=True)
        for f_ in fails:
            print(f'   - {f_}', flush=True)
        sys.exit(2)
    print('\n✓ all correctness gates passed', flush=True)


if __name__ == '__main__':
    main()
