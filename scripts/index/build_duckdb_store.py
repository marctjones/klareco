#!/usr/bin/env python3
"""
Build the DuckDB AST/retrieval store (replaces Kuzu) + rebuild Whoosh

VERSION: v2.2
COMPATIBLE WITH: klareco.parser (current), data/corpus/unified_corpus.jsonl
DEPENDENCIES: duckdb, Whoosh, klareco.parser
STAGE: Index

Kuzu was retired 2026-05 (KuzuASTReconstructor ~17 s/AST; graph
traversal ~338x slower than a flat indexed store). This is the
replacement loader. It is the SINGLE id authority: it streams the
canonical corpus, assigns sid, parses once, and writes BOTH stores in
one pass so Whoosh doc-ids and DuckDB rows are consistent by
construction (no dependency on the old Kuzu-assigned ids).

Pipeline position:
    unified_corpus.jsonl --[THIS]--> data/indexes/duckdb_store.db
                                +--> data/indexes/whoosh_v2/

For every corpus sentence (in file order):
  - sid = 1-based line index (owned scheme; old Kuzu ids are abandoned)
  - parse(text) with the current fixed parser
  - shred the query-relevant AST features into indexed columns
  - keep the full AST as an ast_json blob (json.loads ~0.9 ms vs the
    retired reconstructor's ~17 s — no re-parse, no graph walk)
  - add (sid, text) to a freshly-rebuilt Whoosh index

Also loads data/ontology_export/kuzu_ontology_snapshot.json (the
Kuzu-only ontology + Tier-0 edges captured before deletion) into
DuckDB ontology tables, so semantic_bridge can be repointed in Phase 3.

Checkpointed/resumable: rows are committed per BATCH; --resume continues
after the max(sid) already in DuckDB.

Usage:
    python scripts/index/build_duckdb_store.py            # fresh
    python scripts/index/build_duckdb_store.py --resume
    python scripts/index/build_duckdb_store.py --limit 50000   # smoke

Last Updated: 2026-05-17
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
import pandas as pd
from whoosh import index as whoosh_index
from whoosh.fields import ID, TEXT, Schema

from klareco.corpus_quality import assess
from klareco.parser import parse

import os as _os
CORPUS = 'data/corpus/unified_corpus.jsonl'
# Env overrides let an isolated smoke test write to throwaway paths
# instead of the production store. Defaults are the real targets.
DUCK = _os.environ.get('KLARECO_DUCK', 'data/indexes/duckdb_store.db')
WHOOSH_DIR = _os.environ.get('KLARECO_WHOOSH', 'data/indexes/whoosh_v2')
ONTOLOGY_SNAPSHOT = 'data/ontology_export/kuzu_ontology_snapshot.json'
BATCH = 20_000          # DuckDB durability cadence (fast bulk insert)
WHOOSH_COMMIT = 100_000  # Whoosh flush cadence (merge=False, bounded RAM)
                         # over 5.4M instead of ~270 growing ones

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler('logs/build_duckdb_store.log')])
log = logging.getLogger('build_duckdb_store')


def _kerno(node):
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def shred(ast: dict) -> dict:
    """Project the query-relevant features the retrieval layer filters
    on. Everything else (full tree, aliaj list) stays in ast_json."""
    s = _kerno(ast.get('subjekto') or {})
    v = ast.get('verbo') or {}
    v = _kerno(v)
    o = _kerno(ast.get('objekto') or {})
    aliaj = []
    for a in ast.get('aliaj') or []:
        w = _kerno(a)
        if w:
            aliaj.append({'radiko': w.get('radiko'),
                          'vortspeco': w.get('vortspeco'),
                          'kazo': w.get('kazo'),
                          'plena_vorto': w.get('plena_vorto')})
    stats = ast.get('parse_statistics') or {}
    return {
        'subj_radiko': s.get('radiko'),
        'subj_vortspeco': s.get('vortspeco'),
        # FIXED 2026-05-21: was `propranoma_kategorio` — but the parser
        # stores it as `kategorio` (e.g. 'propranomo', 'propranomo_konata',
        # 'propranomo_esperantigita'). The wrong field name left
        # subj_propranoma_kat = NULL for every sentence in the first
        # build. Backfill via scripts/index/backfill_subj_propranoma_kat.py.
        'subj_propranoma_kat': s.get('kategorio'),
        'subj_kazo': s.get('kazo'),
        'verb_radiko': v.get('radiko'),
        'verb_tempo': v.get('tempo'),
        'obj_radiko': o.get('radiko'),
        'obj_kazo': o.get('kazo'),
        'aliaj_json': json.dumps(aliaj, ensure_ascii=False),
        # BUG (#805, fixed 2026-07-13): this read `success_rate`, an English key
        # the parser never emits. `parse_statistics` is keyed in ESPERANTO:
        #
        #     {'tutaj_vortoj': 3, 'esperantaj_vortoj': 3,
        #      'neesperantaj_vortoj': 0, 'sukcesoprocento': 1.0, ...}
        #
        # So `.get('success_rate')` returned None on every row and the `or 0.0`
        # silently made it a constant. Measured on the live store: success_rate
        # is EXACTLY 0.0 across all 5,391,442 rows — a column that is 100%
        # non-null and carries zero information.
        #
        # This is the SECOND time this exact bug hit this function — see the
        # `subj_propranoma_kat` comment 8 lines up, which read the wrong field
        # name and left the column NULL for every sentence in the first build.
        # A silent `.get()` on a misspelled key is how both happened.
        'success_rate': float(
            stats.get('sukcesoprocento',
                      stats.get('success_rate', 0.0)) or 0.0),
    }


def ensure_schema(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS sentences(
            sid BIGINT PRIMARY KEY, text VARCHAR,
            subj_radiko VARCHAR, subj_vortspeco VARCHAR,
            subj_propranoma_kat VARCHAR, subj_kazo VARCHAR,
            verb_radiko VARCHAR, verb_tempo VARCHAR,
            obj_radiko VARCHAR, obj_kazo VARCHAR,
            aliaj_json VARCHAR, success_rate DOUBLE,
            ast_json VARCHAR,
            -- PROVENANCE (#803). Without these we cannot cite a source
            -- article, cannot count INDEPENDENT documents (R14), and cannot
            -- hit R15's topic targets — there is no article, so no category.
            source_name VARCHAR, source_type VARCHAR,
            article_title VARCHAR, article_id VARCHAR,
            section VARCHAR, quality VARCHAR)
    """)
    con.execute("CREATE TABLE IF NOT EXISTS ontology_nodes("
                "label VARCHAR, node_json VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS ontology_edges("
                "rel VARCHAR, radiko VARCHAR, class_id VARCHAR)")


def load_ontology(con):
    p = Path(ONTOLOGY_SNAPSHOT)
    if not p.exists():
        log.warning("ontology snapshot %s missing — skipping", p)
        return
    snap = json.loads(p.read_text())
    con.execute("DELETE FROM ontology_nodes")
    con.execute("DELETE FROM ontology_edges")
    nrows = [(lbl, json.dumps(n, ensure_ascii=False, default=str))
             for lbl, v in snap.get('nodes', {}).items()
             if isinstance(v, list) for n in v]
    erows = [(rel, e[0], e[1])
             for rel, v in snap.get('edges', {}).items()
             if isinstance(v, list) for e in v]
    if nrows:
        con.executemany("INSERT INTO ontology_nodes VALUES (?,?)", nrows)
    if erows:
        con.executemany("INSERT INTO ontology_edges VALUES (?,?,?)", erows)
    log.info("ontology loaded: %d nodes, %d edges", len(nrows), len(erows))


def build_indexes(con):
    for col in ('verb_radiko', 'obj_radiko', 'subj_radiko',
                'subj_vortspeco', 'subj_propranoma_kat',
                # #803: document-level aggregation (R14 independent-document
                # counts) and per-source row-count assertions (#815).
                'article_id', 'source_name'):
        con.execute(f"CREATE INDEX IF NOT EXISTS i_{col} "
                    f"ON sentences({col})")
    log.info("DuckDB secondary indexes built")


def open_whoosh(fresh: bool):
    d = Path(WHOOSH_DIR)
    d.mkdir(parents=True, exist_ok=True)
    schema = Schema(id=ID(stored=True, unique=True),
                    text=TEXT(stored=True))
    if fresh or not whoosh_index.exists_in(str(d)):
        return whoosh_index.create_in(str(d), schema)
    return whoosh_index.open_dir(str(d))


_NULL_SHRED = {k: None for k in (
    'subj_radiko', 'subj_vortspeco', 'subj_propranoma_kat', 'subj_kazo',
    'verb_radiko', 'verb_tempo', 'obj_radiko', 'obj_kazo')}


def _worker(payload):
    """CPU-heavy per-sentence work, run in a Pool worker: parse + shred
    + serialize. Returns a fully-formed row dict (picklable)."""
    sid, text, prov = payload
    try:
        ast = parse(text)
    except Exception:
        ast = None
    if isinstance(ast, dict):
        shredded = shred(ast)
        ast_json = json.dumps(ast, ensure_ascii=False)
    else:
        shredded = dict(_NULL_SHRED)
        shredded['aliaj_json'] = '[]'
        shredded['success_rate'] = 0.0
        ast_json = None
    return {'sid': sid, 'text': text, 'ast_json': ast_json,
            **shredded, **prov}


_REJECTED: Counter = Counter()


def _corpus_payloads(start_after: int, limit):
    """Yield (sid, text) for corpus lines past the resume point. sid is
    the absolute 1-based line index — deterministic, so resume is exact."""
    with open(CORPUS, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                return
            if i <= start_after:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get('text') or ''
                # PROVENANCE (#803). The corpus HAS this — nested in `source` —
                # and the store was throwing it away. Consequences of the loss:
                # R15 topic targets unimplementable (no article => no category),
                # R14 unable to count INDEPENDENT documents, and citations that
                # read "[1] Unknown" for a project whose thesis includes
                # "answers are grounded and cited".
                src = rec.get('source') or {}
                prov = {
                    'source_name':   src.get('name'),          # wikipedia, pmeg, ...
                    'source_type':   src.get('source_type'),   # encyclopedia, ...
                    'article_title': src.get('source_name'),   # the article/work title
                    'article_id':    (str(src.get('article_id'))
                                      if src.get('article_id') is not None else None),
                    'section':       src.get('section'),
                    'quality':       src.get('quality'),
                }
            except Exception:
                continue
            if not text:
                continue

            # THE QUALITY GATE (#823) — one shared implementation, so it cannot
            # be applied to one consumer and forgotten by the next. That is
            # exactly what happened: the redirect filter lived only in
            # rebuild_whoosh_from_duckdb.py, so the Whoosh index was clean while
            # the STORE kept 123,654 redirect stubs — and `REDIRECT` became the
            # single most common proper-noun SUBJECT in the entire corpus.
            #
            # Markup is STRIPPED, not fatal: `{{DISPLAYTITLE: …}} … estas
            # asteroido` carries a real sentence, and ~27,000 rows are rescued
            # this way rather than deleted.
            verdict = assess(text)
            if not verdict.keep:
                _REJECTED[verdict.reason] += 1
                continue
            yield (i, verdict.text or text, prov)


def main() -> int:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N sentences (smoke test)')
    ap.add_argument('--workers', type=int, default=10,
                    help='parse-pool workers (box has 16 cores; leave '
                         'headroom for the writer + competing load)')
    ap.add_argument('--skip-whoosh', action='store_true',
                    help='build DuckDB only; rebuild Whoosh later from DuckDB')
    args = ap.parse_args()

    Path('logs').mkdir(exist_ok=True)
    Path(DUCK).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DUCK)
    # Defensive: cap DuckDB memory so a competing process can't OOM us
    # (the exact footgun we hit with Kuzu's 80%-of-RAM default).
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    ensure_schema(con)
    load_ontology(con)

    start_after = 0
    if args.resume:
        duck_max = con.execute(
            "SELECT max(sid) FROM sentences").fetchone()[0] or 0
        if args.skip_whoosh:
            wh_cnt = None
            start_after = duck_max
        elif whoosh_index.exists_in(WHOOSH_DIR):
            wh_cnt = whoosh_index.open_dir(WHOOSH_DIR).doc_count()
            # Clean common prefix only — Whoosh and DuckDB may disagree if a
            # crash hit mid-batch. Reconcile to min, drop the inconsistent
            # DuckDB tail; Whoosh's unique=True id makes any re-add idempotent.
            start_after = min(duck_max, wh_cnt)
            if duck_max > start_after:
                con.execute("DELETE FROM sentences WHERE sid > ?",
                            [start_after])
        else:
            wh_cnt = 0
            start_after = 0
        log.info("resume: duck_max=%d whoosh=%s -> sid<=%d",
                 duck_max,
                 'skipped' if wh_cnt is None else str(wh_cnt),
                 start_after)
    else:
        con.execute("DELETE FROM sentences")

    ix = None if args.skip_whoosh else open_whoosh(fresh=not args.resume)

    def flush(rows):
        if not rows:
            return
        df = pd.DataFrame(rows)          # noqa: F841 (used by DuckDB)
        # Columns are listed EXPLICITLY and in schema order. A positional
        # `SELECT *` here would silently mis-map the moment a column is added —
        # and this file has already shipped two silent field-name bugs
        # (subj_propranoma_kat, success_rate). Be explicit.
        con.execute(
            "INSERT INTO sentences SELECT sid, text, subj_radiko, "
            "subj_vortspeco, subj_propranoma_kat, subj_kazo, verb_radiko, "
            "verb_tempo, obj_radiko, obj_kazo, aliaj_json, success_rate, "
            "ast_json, "
            "source_name, source_type, article_title, article_id, "
            "section, quality FROM df")

    t0 = time.time()
    n = 0
    batch: list[dict] = []
    since_whoosh = 0
    # Whoosh bulk-load idiom. The pathology measured earlier: committing
    # WITH a merge every BATCH makes segment-merge cost grow super-linearly
    # (708->655/s live). The previous "fix" (one 2GB writer, 500k-doc
    # commit) cured the growth but OOM-killed the build at the first 500k
    # commit (no traceback, tmux gone, doc_count=0 mid-commit).
    # Correct idiom: bounded writer + commit(merge=False) per window
    # (flush only, NO O(index) merge -> cheap, RAM-bounded, no growth),
    # then ONE final full merge AFTER the parse pool is closed (lowest
    # memory pressure). Resume stays correct: start_after =
    # min(duck_max, whoosh_count) + DuckDB tail-delete + Whoosh unique-id
    # re-add is idempotent regardless of which store leads on a crash.
    writer = None if args.skip_whoosh else ix.writer(limitmb=512, procs=1)

    # Parse in a process pool (pure-Python CPU). The main process keeps
    # the serial, non-picklable work (Whoosh writer, DuckDB). imap
    # preserves order and streams (memory-bounded).
    with mp.Pool(args.workers) as pool:
        for row in pool.imap(_worker,
                             _corpus_payloads(start_after, args.limit),
                             chunksize=200):
            if writer is not None:
                writer.add_document(id=str(row['sid']), text=row['text'])
            batch.append(row)
            n += 1
            since_whoosh += 1
            if len(batch) >= BATCH:
                flush(batch)                       # DuckDB durable / 20k
                last_sid = batch[-1]['sid']
                batch = []
                rate = n / (time.time() - t0)
                log.info("sid=%d done=%d %.0f/s ETA=%.0f min",
                         last_sid, n, rate,
                         (5_400_000 - start_after - n) / rate / 60
                         if rate else -1)
            if writer is not None and since_whoosh >= WHOOSH_COMMIT:
                writer.commit(merge=False)         # flush only, no merge
                writer = ix.writer(limitmb=512, procs=1)
                since_whoosh = 0
                log.info("whoosh flushed @ sid~%d", n + start_after)

    if writer is not None:
        writer.commit(merge=False)                 # final partial flush
    flush(batch)
    if ix is not None:
        # Pool is closed here -> lowest memory pressure -> safe to do the
        # ONE expensive full segment merge that makes queries fast.
        log.info("whoosh: optimizing (single full merge of all segments)...")
        t_opt = time.time()
        ix.optimize()
        log.info("whoosh: optimized in %.0f s", time.time() - t_opt)
    build_indexes(con)
    cnt = con.execute("SELECT count(*) FROM sentences").fetchone()[0]
    outputs = DUCK if args.skip_whoosh else f"{DUCK} + {WHOOSH_DIR}"

    # Never drop rows silently. A quality gate whose rejections are invisible is
    # how you end up with 123,654 redirect stubs and nobody noticing (#823).
    if _REJECTED:
        total = sum(_REJECTED.values())
        log.info("QUALITY GATE rejected %d rows (%.2f%% of input):",
                 total, 100.0 * total / (total + cnt))
        for reason, n in _REJECTED.most_common():
            log.info("    %-16s %8d", reason, n)
    else:
        log.warning("QUALITY GATE rejected NOTHING — that is suspicious. "
                    "Is klareco.corpus_quality wired in?")

    log.info("DONE: %d sentences in %.0f s -> %s",
             cnt, time.time() - t0, outputs)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
