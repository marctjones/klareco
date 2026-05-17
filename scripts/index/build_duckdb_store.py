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

from klareco.parser import parse

CORPUS = 'data/corpus/unified_corpus.jsonl'
DUCK = 'data/indexes/duckdb_store.db'
WHOOSH_DIR = 'data/indexes/whoosh_v2'
ONTOLOGY_SNAPSHOT = 'data/ontology_export/kuzu_ontology_snapshot.json'
BATCH = 20_000

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
        'subj_propranoma_kat': s.get('propranoma_kategorio'),
        'subj_kazo': s.get('kazo'),
        'verb_radiko': v.get('radiko'),
        'verb_tempo': v.get('tempo'),
        'obj_radiko': o.get('radiko'),
        'obj_kazo': o.get('kazo'),
        'aliaj_json': json.dumps(aliaj, ensure_ascii=False),
        'success_rate': float(stats.get('success_rate') or 0.0),
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
            ast_json VARCHAR)
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
                'subj_vortspeco', 'subj_propranoma_kat'):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N sentences (smoke test)')
    args = ap.parse_args()

    Path('logs').mkdir(exist_ok=True)
    Path(DUCK).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DUCK)
    ensure_schema(con)
    load_ontology(con)

    start_after = 0
    if args.resume:
        row = con.execute("SELECT max(sid) FROM sentences").fetchone()
        start_after = row[0] or 0
        log.info("resume: skipping sid <= %d", start_after)
    else:
        con.execute("DELETE FROM sentences")

    ix = open_whoosh(fresh=not args.resume)
    writer = ix.writer(limitmb=512, procs=1)

    t0 = time.time()
    n = 0
    parse_fail = 0
    batch: list[dict] = []

    def flush(rows):
        if not rows:
            return
        df = pd.DataFrame(rows)
        con.execute(
            "INSERT INTO sentences SELECT sid, text, subj_radiko, "
            "subj_vortspeco, subj_propranoma_kat, subj_kazo, verb_radiko, "
            "verb_tempo, obj_radiko, obj_kazo, aliaj_json, success_rate, "
            "ast_json FROM df")

    with open(CORPUS, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if args.limit and i > args.limit:
                break
            if i <= start_after:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get('text') or ''
            except Exception:
                continue
            if not text:
                continue
            sid = i
            try:
                ast = parse(text)
            except Exception:
                ast = None
                parse_fail += 1
            shredded = shred(ast) if isinstance(ast, dict) else {
                k: None for k in (
                    'subj_radiko', 'subj_vortspeco', 'subj_propranoma_kat',
                    'subj_kazo', 'verb_radiko', 'verb_tempo', 'obj_radiko',
                    'obj_kazo')}
            if 'aliaj_json' not in shredded:
                shredded['aliaj_json'] = '[]'
                shredded['success_rate'] = 0.0
            row = {'sid': sid, 'text': text,
                   'ast_json': json.dumps(ast, ensure_ascii=False)
                   if ast else None,
                   **shredded}
            batch.append(row)
            writer.add_document(id=str(sid), text=text)
            n += 1
            if len(batch) >= BATCH:
                flush(batch)
                writer.commit()
                writer = ix.writer(limitmb=512, procs=1)
                batch = []
                rate = n / (time.time() - t0)
                log.info("sid=%d  done=%d  %.0f/s  ETA=%.0f min",
                         sid, n, rate,
                         (5_400_000 - n) / rate / 60 if rate else -1)

    flush(batch)
    writer.commit()
    build_indexes(con)
    cnt = con.execute("SELECT count(*) FROM sentences").fetchone()[0]
    log.info("DONE: %d sentences (%d parse-fail) in %.0f s -> %s + %s",
             cnt, parse_fail, time.time() - t0, DUCK, WHOOSH_DIR)
    con.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
