#!/usr/bin/env python3
"""Recheck a completed QA experiment after narrowly scoped parser repairs.

Prove exact question/candidate AST equality before reusing an answer. Re-run
retrieval and the pipeline for every affected question. Baseline results remain
those of the completed paired experiment; timings of reused answers are old.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import duckdb
from compare_parser_qa import FixedCandidates
from klareco.parser import parse
from klareco.eval import evaluate_question, summarize
from klareco.orchestrator.factory import build_default_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--treatment-parser', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Refusing to overwrite experiment evidence')
    experiment = json.loads(args.experiment.read_text())
    if experiment['status'] != 'complete' or not experiment['live_candidates']:
        raise ValueError('Requires a completed live-candidate paired experiment')
    source = args.treatment_parser.read_bytes()
    if hashlib.sha256(source).hexdigest() != experiment['code_hashes']['klareco/parser.py']:
        raise ValueError('Treatment parser snapshot does not match recorded source hash')
    spec = importlib.util.spec_from_file_location('klareco._previous_treatment', ROOT / 'klareco/parser.py')
    previous = importlib.util.module_from_spec(spec)
    exec(compile(source, str(args.treatment_parser), 'exec'), previous.__dict__)
    test_path = Path(experiment['test_set'])
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != experiment['test_sha256']:
        raise ValueError('Frozen QA set changed')
    entries = {r['id']: r for r in map(json.loads, test_path.read_text().splitlines())}
    ids = sorted({int(sid) for p in experiment['pairs'] for sid in p['after_candidate_ids']})
    with duckdb.connect(str(ROOT / 'data/indexes/duckdb_store.db'), read_only=True) as con:
        rows = con.execute('SELECT sid,text FROM sentences WHERE sid IN (SELECT unnest(?))', [ids]).fetchall()
    if len(rows) != len(ids):
        raise ValueError('Candidate identity missing from source store')
    changed = set()
    for i, (sid, text) in enumerate(rows):
        if previous.parse(text) != parse(text):
            changed.add(sid)
        if i % 1000 == 0:
            print(f'Compared {i}/{len(rows)} candidate ASTs', flush=True)
    pipeline = build_default_pipeline(whoosh_index_dir=ROOT / 'data/indexes/whoosh_v2')
    retrieve = next(stage for stage in pipeline.stages if stage.name == 'retrieve')
    live = retrieve.retriever
    result = deepcopy(experiment)
    result.update(status='running', parent_experiment=str(args.experiment),
                  parent_sha256=hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
                  changed_candidate_ids=sorted(changed), rechecked_questions=[], reused_questions=[],
                  verification='Exact AST input equality, with current shared projection; final parser repairs only. Reused answer timings are from the parent experiment.')
    result['code_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT / 'klareco/parser.py', ROOT / 'klareco/syntax_graph.py',
                                      ROOT / 'klareco/rag/duckdb_retriever.py', ROOT / 'klareco/rag/unified_extractor.py',
                                      ROOT / 'klareco/orchestrator/stages/parse_question.py', Path(__file__)]}
    try:
        for pair in result['pairs']:
            entry = entries[pair['id']]
            affected = (previous.parse(entry['question']) != parse(entry['question'])
                        or any(int(i) in changed for i in pair['after_candidate_ids']))
            if affected:
                candidates = live.retrieve_with_ast_roles(parse(entry['question']), retrieve.top_k)
                retrieve.retriever = FixedCandidates(candidates, parse)
                pair['after'] = evaluate_question(pipeline, entry)
                pair['after_candidate_ids'] = [r['id'] for r in candidates]
                result['rechecked_questions'].append(pair['id'])
            else:
                result['reused_questions'].append(pair['id'])
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        result['summary'] = {arm: summarize([p[arm] for p in result['pairs']]) for arm in ('before', 'after')}
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        raise
    finally:
        retrieve.retriever = live
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Rechecked', len(result['rechecked_questions']), 'reused', len(result['reused_questions']))


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
