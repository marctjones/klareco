"""
Whoosh OR-query cost x quality spike (#859)

VERSION: v2.1
COMPATIBLE WITH: v2.1 whoosh_v2 index, qa_gold_v2 test sets
DEPENDENCIES: whoosh, klareco.eval.bootstrap, klareco.rag.duckdb_retriever
STAGE: Evaluation (research spike)

Description:
    The retriever's BM25 OR-query scores every doc matching ANY content term
    (~1.7 s/q, 82% of bench wall clock). Pruning it (minmatch, top-IDF term
    subsetting) CHANGES which docs are retrieved — a measured quality trade, not
    a free perf win. This spike sweeps pruning strategies and reports, per config:
    mean query latency + paired-bootstrap MRR CI vs the current OR baseline.

    Output: a speed x quality curve and a keep/reject decision. Produces a
    DECISION, not a capability — merge-gate exempt (research spike).

Pipeline Position:
    whoosh_v2 + rebaseline_210 -> [THIS SPIKE] -> speed/quality curve (stdout + json)

Usage:
    python scripts/eval/whoosh_cost_spike.py --test-set data/test_sets/rebaseline_210.jsonl

Inputs:
    - --test-set JSONL (question, source_sentence_id)

Outputs:
    - stdout table + results/whoosh_cost_spike.json

Quality Checks:
    - Paired CI vs baseline (same questions, same order) so a latency win with a
      significant MRR loss is flagged as a reject.

Last Updated: 2026-07-18
Author: Marc Jones (with Claude Fable 5)
Related Issues: #859
See Also: klareco/rag/duckdb_retriever.py, klareco/eval/bootstrap.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('KLARECO_ALLOW_DEGRADED', '1')

from whoosh import index, scoring
from whoosh.qparser import QueryParser
from whoosh.query import Or, Term

from klareco.parser import parse
from klareco.rag.duckdb_retriever import _content_terms, DuckDBRetriever
from klareco.eval.bootstrap import paired_delta_ci

IX = 'data/indexes/whoosh_v2'
LIMIT = 300  # same wide net the retriever uses (top_k*15)


def _terms(question_ast) -> list[str]:
    return _content_terms(DuckDBRetriever._question_text(question_ast))


def _rank(searcher, q, gold) -> int | None:
    ids = [int(h['id']) for h in searcher.search(q, limit=100)]
    return ids.index(gold) + 1 if gold in ids else None


def build_query(strategy: str, terms: list[str], searcher, schema):
    """Return a Whoosh query for the given pruning strategy."""
    subs = [Term('text', t) for t in terms]
    if not subs:
        return Or([])
    if strategy == 'or_all':                 # baseline: pure OR
        return Or(subs)
    if strategy == 'minmatch2':              # >=2 terms must match
        return Or(subs, minmatch=min(2, len(subs)))
    if strategy == 'minmatch_half':          # >= ceil(M/2) must match
        return Or(subs, minmatch=(len(subs) + 1) // 2)
    if strategy == 'top_idf_4':              # only the 4 rarest terms, pure OR
        ranked = sorted(terms, key=lambda t: searcher.doc_frequency('text', t))
        return Or([Term('text', t) for t in ranked[:4]])
    if strategy == 'top_idf_4_mm2':          # 4 rarest + minmatch 2
        ranked = sorted(terms, key=lambda t: searcher.doc_frequency('text', t))
        top = [Term('text', t) for t in ranked[:4]]
        return Or(top, minmatch=min(2, len(top)))
    raise ValueError(strategy)


STRATEGIES = ['or_all', 'minmatch2', 'minmatch_half', 'top_idf_4', 'top_idf_4_mm2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-set', default='data/test_sets/rebaseline_210.jsonl')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test_set)]
    rows = [r for r in rows if r.get('source_sentence_id') is not None]
    ix = index.open_dir(IX)
    s = ix.searcher(weighting=scoring.BM25F())

    # pre-parse once so parse time doesn't pollute the per-strategy latency
    parsed = [(parse(r['question']), int(r['source_sentence_id'])) for r in rows]

    out = {}
    for strat in STRATEGIES:
        ranks, lat = [], []
        for ast, gold in parsed:
            terms = _terms(ast)
            q = build_query(strat, terms, s, ix.schema)
            t0 = time.perf_counter()
            r = _rank(s, q, gold)
            lat.append((time.perf_counter() - t0) * 1000)
            ranks.append(r)
        out[strat] = {'ranks': ranks, 'lat_ms': lat}

    base = out['or_all']['ranks']
    base_lat = sum(out['or_all']['lat_ms']) / len(out['or_all']['lat_ms'])
    print(f"{'strategy':<16s} {'lat_ms':>8s} {'speedup':>8s} "
          f"{'recall@5':>9s} {'MRRΔ vs or_all':>16s} {'95% CI':>22s} {'verdict':>9s}")
    for strat in STRATEGIES:
        ranks = out[strat]['ranks']
        lat = sum(out[strat]['lat_ms']) / len(out[strat]['lat_ms'])
        r5 = sum(1 for r in ranks if r and r <= 5)
        ci = paired_delta_ci(ranks, base)  # strat minus baseline
        speedup = base_lat / lat if lat else float('nan')
        if ci['significant'] and ci['delta'] < 0:
            verdict = 'REJECT'          # significant quality loss
        elif speedup >= 1.3:
            verdict = 'KEEP?'           # meaningful speedup, no sig loss
        else:
            verdict = 'neutral'
        out[strat]['summary'] = {'lat_ms': lat, 'speedup': speedup,
                                 'recall_at_5': r5, 'mrr_delta': ci['delta'],
                                 'ci': [ci['lo'], ci['hi']],
                                 'significant': ci['significant'],
                                 'verdict': verdict}
        print(f"{strat:<16s} {lat:>8.1f} {speedup:>6.2f}x {r5:>9d} "
              f"{ci['delta']:>+16.4f} [{ci['lo']:>+.4f},{ci['hi']:>+.4f}] {verdict:>9s}")

    Path('results').mkdir(exist_ok=True)
    json.dump({k: v['summary'] for k, v in out.items()},
              open('results/whoosh_cost_spike.json', 'w'), indent=2)
    print("\nwrote results/whoosh_cost_spike.json")


if __name__ == '__main__':
    main()
