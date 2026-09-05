#!/usr/bin/env python3
"""Reproducible parser baseline and token error inventory.

VERSION: v1.0
COMPATIBLE WITH: klareco.parser, committed UD Prago/Cairo fixtures
DEPENDENCIES: project parser lexicons; no models or production database
STAGE: Evaluation

Pipeline Position: frozen gold -> parser -> metrics and diagnostic inventory
Usage: python scripts/eval/parser_quality_report.py --output data/perf/parser.json
Inputs: committed CoNLL-U fixtures; parser and lexicon files
Outputs: JSON metrics, token diagnostics, source hashes, serialization fidelity
Quality Checks: fixed gold denominator; separate treebanks; no punctuation scoring
Last Updated: 2026-09-05
Related Issues: #832, #902
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import eval_conllu
import eval_ud_prago
import eval_ud_roles
from klareco.parser import compact_ast, expand_ast, parse


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report() -> dict:
    result = {
        'report_version': 1,
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {}, 'treebanks': {},
        'interpretation': 'Error groups are symptoms, not adjudicated root causes. '
                          'Cairo is a regression set once its errors inform a fix.',
    }
    files = [ROOT / 'klareco/parser.py', ROOT / 'klareco/morphology.py',
             ROOT / 'klareco/conllu.py', Path(__file__),
             Path(eval_conllu.__file__)]
    files += sorted((ROOT / 'data/vocabularies').glob('*.json'))
    storage = ROOT / 'klareco/ast_storage.py'
    if storage.exists():  # Optional code module, not a parser data dependency.
        files.append(storage)
    for path in files:
        result['files'][str(path.relative_to(ROOT))] = digest(path)
    for name in ('prago', 'cairo'):
        path = ROOT / f'tests/fixtures/ud/eo_{name}-ud-test.conllu'
        tokens = []
        metrics = eval_conllu.evaluate(str(path), diagnostics=tokens)
        pos = eval_ud_prago.evaluate(path)
        roles = eval_ud_roles.evaluate(eval_ud_roles.load_conllu(str(path)))
        texts = [s['text'] for s in eval_conllu.read_gold(str(path))]
        exact = 0
        raw_bytes = compact_bytes = 0
        latencies = []
        for text in texts:
            parse.cache_clear()
            start = perf_counter()
            ast = parse(text)
            latencies.append((perf_counter() - start) * 1000)
            raw = json.dumps(ast, ensure_ascii=False)
            packed = json.dumps(compact_ast(ast), ensure_ascii=False)
            exact += expand_ast(json.loads(packed)) == ast
            raw_bytes += len(raw.encode('utf-8'))
            compact_bytes += len(packed.encode('utf-8'))
        errors = [t for t in tokens if set(t['errors']) - {'pos'}]
        result['treebanks'][name] = {
            'fixture_sha256': digest(path), 'metrics': metrics,
            'pos_strict': pos['pos_strict'], 'pos_adjusted': pos['pos_adjusted'],
            'roles': roles,
            'errors_by_gold_relation': dict(Counter(
                t['gold']['dep'] for t in errors).most_common()),
            'errors_by_symptom': dict(Counter(
                e for t in tokens for e in t['errors']).most_common()),
            'storage': {'exact': exact, 'sentences': len(texts),
                        'raw_bytes': raw_bytes, 'compact_bytes': compact_bytes},
            'parse_latency_ms': {'median': sorted(latencies)[len(latencies)//2],
                                 'total': sum(latencies),
                                 'method': 'sentence cache cleared; word cache warm'},
            'tokens': tokens,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    data = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    for name, bank in data['treebanks'].items():
        print(name, json.dumps({k: v for k, v in bank.items()
                                if k not in ('tokens', 'roles')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
