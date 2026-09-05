#!/usr/bin/env python3
"""Prepare a source-stratified parser annotation queue without parser labels.

VERSION: v1.0
COMPATIBLE WITH: DuckDB sentences(sid,text,source_name,article_title)
DEPENDENCIES: duckdb; local corpus; no models
STAGE: Evaluation

Pipeline Position: source sentences -> unreviewed annotation queue -> human gold
Usage: python scripts/eval/build_parser_pilot.py --output data/test_sets/parser_pilot
Inputs: local DuckDB store, existing UD fixture texts (excluded from selection)
Outputs: development/heldout JSONL queues and a manifest; NOT a gold benchmark
Quality Checks: unique normalized texts, document-disjoint splits, no parser filter
Last Updated: 2026-09-05
Related Issues: #820, #832
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
SEED = 'parser-pilot-v1'


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def document_split(source: str, title: str) -> str:
    key = f'{SEED}\0{source}\0{title}'.encode()
    return 'heldout' if int(hashlib.sha256(key).hexdigest(), 16) % 5 == 0 else 'development'


def select(rows: list, excluded: set[str], size: int) -> list[dict]:
    """Round-robin sources, cap document reuse, never consult parser output."""
    groups = defaultdict(list)
    for sid, text, source, title in rows:
        if not text or not source or not title:
            continue
        groups[source].append((sid, text, source, title))
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(
            f'{SEED}\0{r[0]}'.encode()).hexdigest())
    seen = set(excluded)
    docs = Counter()
    chosen = []
    quotas = {'heldout': size // 5, 'development': size - size // 5}
    counts = Counter()
    while any(groups.values()) and len(chosen) < size:
        for source in sorted(groups):
            while groups[source]:
                sid, text, source, title = groups[source].pop(0)
                key = normalized(text)
                document = (source, title)
                split = document_split(source, title)
                if key in seen or docs[document] >= 4 or counts[split] >= quotas[split]:
                    continue
                seen.add(key)
                docs[document] += 1
                counts[split] += 1
                chosen.append({
                    'id': f'parser-pilot-v1-{sid}', 'source_sentence_id': sid,
                    'source_name': source, 'document_title': title,
                    'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'split': split, 'annotation_status': 'unreviewed',
                    'gold_conllu': None, 'phenomena': [], 'reviewer': None,
                })
                break
    if len(chosen) != size:
        raise ValueError(f'Only {len(chosen)}/{size} eligible rows; widen candidate pool')
    return chosen


def main() -> None:
    import duckdb
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', type=Path, default=ROOT / 'data/indexes/duckdb_store.db')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=200)
    args = ap.parse_args()
    if args.size < 5:
        ap.error('--size must be at least 5')
    if args.output.exists():
        ap.error('output directory already exists; preserve existing annotations')
    excluded = set()
    for name in ('prago', 'cairo'):
        fixture = ROOT / f'tests/fixtures/ud/eo_{name}-ud-test.conllu'
        excluded.update(normalized(line.split('=', 1)[1])
                        for line in fixture.read_text().splitlines()
                        if line.startswith('# text ='))
    with duckdb.connect(str(args.store), read_only=True) as con:
        rows = con.execute('''
            SELECT sid, text, source_name, article_title FROM sentences
            QUALIFY row_number() OVER (
                PARTITION BY source_name ORDER BY md5(CAST(sid AS VARCHAR) || ?)
            ) <= 1000
        ''', [SEED]).fetchall()
    chosen = select(rows, excluded, args.size)
    args.output.mkdir(parents=True)
    files = {}
    for split in ('development', 'heldout'):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n'
                          for row in chosen if row['split'] == split)
        filename = f'{split}.jsonl'
        (args.output / filename).write_text(content)
        files[filename] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        'version': 1, 'seed': SEED, 'store': str(args.store.resolve()),
        'status': 'unreviewed annotation queue, not gold', 'files': files,
        'sources': dict(Counter(row['source_name'] for row in chosen)),
        'splits': dict(Counter(row['split'] for row in chosen)),
        'selection': 'source round-robin, at most four per document; no parser labels',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
