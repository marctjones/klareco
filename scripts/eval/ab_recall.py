#!/usr/bin/env python3
"""
A/B the retrieval tokenizations by RECALL — surface vs roots vs root+class. (#736)

VERSION: v1.0
COMPATIBLE WITH: retrieval_ab.db (built by build_retrieval_ab_index.py)
DEPENDENCIES: duckdb (fts), klareco.parser, klareco.eval.bootstrap
STAGE: Evaluation

Description:
    The diagnostic said natural-question loss is FIRST-STAGE recall, and the miss
    is morphological (`fondiĝis` vs `fondita`). This measures whether re-tokenizing
    fixes it, holding the BM25 engine constant. For each question, the SAME query
    is issued three ways — surface text, content roots, roots+ontology-class — each
    scored against its own field of one FTS index. We report recall@k per variant
    and a PAIRED bootstrap CI on MRR (roots vs surface, rootclass vs roots): a
    variant only wins if its CI excludes 0.

Usage:
    python scripts/eval/ab_recall.py \
        --test-sets data/test_sets/gold_trivia_review_queue_v1.triaged.jsonl \
                    data/test_sets/synthetic_who_rebuild_50.jsonl \
        --max-n 200

Last Updated: 2026-07-17
Related Issues: #713, #736, #737
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.parser import parse
from klareco.eval.bootstrap import paired_delta_ci, bootstrap_mrr_ci

AB = 'data/indexes/retrieval_ab.db'
STORE = 'data/indexes/duckdb_store.db'
_CONTENT = frozenset({'substantivo', 'verbo', 'adjektivo', 'adverbo',
                      'propra_nomo', 'numeralo'})
_BUCKETS = (1, 5, 10, 20, 50, 100)
_VARIANTS = ('surface', 'roots', 'rootclass')


def load_ontology(store):
    exp: dict[str, list[str]] = {}
    for rel in ('APARTENAS_AL_VERBA_KLASO', 'HAVAS_ENTECAN_TIPON'):
        for radiko, cls in store.execute(
                "SELECT radiko, class_id FROM ontology_edges WHERE rel = ?",
                [rel]).fetchall():
            exp.setdefault((radiko or '').lower(), []).append(cls)
    return exp


def q_roots(question: str) -> list[str]:
    ast = parse(question)
    out = []
    for w in (ast.get('vortoj') or []):
        if isinstance(w, dict) and w.get('vortspeco') in _CONTENT:
            r = (w.get('radiko') or '').lower()
            if r:
                out.append(r)
    return out


def _rank(con, field: str, query: str, gold: str, limit: int):
    if not query.strip():
        return None
    rows = con.execute(
        f"SELECT sid FROM ab_docs "
        f"WHERE fts_main_ab_docs.match_bm25(sid, ?, fields := '{field}') IS NOT NULL "
        f"ORDER BY fts_main_ab_docs.match_bm25(sid, ?, fields := '{field}') DESC "
        f"LIMIT ?", [query, query, limit]).fetchall()
    for i, (sid,) in enumerate(rows):
        if str(sid) == str(gold):
            return i + 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--test-sets', nargs='+', required=True)
    ap.add_argument('--ab-db', default=AB)
    ap.add_argument('--store', default=STORE)
    ap.add_argument('--max-n', type=int, default=200)
    args = ap.parse_args()

    con = duckdb.connect(args.ab_db, read_only=True)
    con.execute('LOAD fts;')
    store = duckdb.connect(args.store, read_only=True)
    onto = load_ontology(store)

    for ts in args.test_sets:
        ranks: dict = {v: [] for v in _VARIANTS}
        n = no_gold = 0
        for line in open(ts, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            question = q.get('eo_question') or q.get('question')
            gold = q.get('source_sentence_id') or q.get('sid')
            if not question or gold is None:
                no_gold += 1
                continue
            n += 1
            roots = q_roots(question)
            queries = {
                'surface': question,
                'roots': ' '.join(roots),
                'rootclass': ' '.join(roots + [c for r in roots
                                                for c in onto.get(r, ())]),
            }
            for v in _VARIANTS:
                ranks[v].append(_rank(con, v, queries[v], gold, args.max_n))

        print(f'\n{"=" * 70}\n  {Path(ts).name}   (n={n}, no-gold={no_gold})\n{"=" * 70}')
        print(f'  {"variant":10s} ' + ''.join(f'R@{k:<4d}' for k in (*_BUCKETS, args.max_n))
              + '  MRR   miss')
        for v in _VARIANTS:
            rs = ranks[v]
            cells = ''
            for k in (*_BUCKETS, args.max_n):
                hit = sum(1 for r in rs if r is not None and r <= k)
                cells += f'{hit / max(n, 1):5.0%} '
            mrr = bootstrap_mrr_ci(rs)['mrr']
            miss = sum(1 for r in rs if r is None)
            print(f'  {v:10s} {cells} {mrr:.3f}  {miss}')

        # paired CIs: does re-tokenizing actually move MRR?
        print('  paired bootstrap 95% CI of MRR delta:')
        for a, b in (('roots', 'surface'), ('rootclass', 'roots'),
                     ('rootclass', 'surface')):
            d = paired_delta_ci(ranks[a], ranks[b])
            verd = 'REAL (excludes 0)' if d['significant'] else 'inside noise'
            print(f'    {a:9s} − {b:9s}: {d["delta"]:+.4f} '
                  f'[{d["lo"]:+.4f},{d["hi"]:+.4f}]  {verd}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
