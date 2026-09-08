#!/usr/bin/env python3
"""Prepare a source-stratified parser annotation queue of INDIVIDUAL
sentences, sourced from EXTRACTED-BUT-NOT-YET-INDEXED literary/news
corpora -- not the DuckDB store.

VERSION: v1.0
COMPATIBLE WITH: data/extracted/eo/free/*.jsonl schema
  {sentence, source, source_title, author, kind, licence}
DEPENDENCIES: none (pure Python + the sibling build_parser_pilot module)
STAGE: Evaluation

The sentence-level counterpart to build_literary_discourse_pilot.py (see
that file's docstring for the full rationale and the source measurement
-- gutenberg/vikifontaro/libera_folio all score 99.15-99.39% on
klareco.parser's `sukcesoprocento` in-vocabulary-Esperanto metric,
n=500 each, vs. 97.98% for Wikipedia). This script answers a different
need than the passage builder: `build_parser_pilot.py`'s existing
queues (parser_pilot_v1/v2) are built entirely from the DuckDB store,
so they are entirely Wikipedia-derived -- there is no INDIVIDUAL-
SENTENCE (not multi-sentence-passage) queue drawn from these three
higher-scoring, non-Wikipedia sources at all.

Reuses `select`, `document_split`, and `normalized` from
build_parser_pilot.py UNCHANGED -- this script only supplies a
different `rows` list of the same `(sid, text, source, title)` shape,
so the round-robin, document-reuse cap (4/document), and duplicate-
exclusion-against-the-frozen-UD-fixtures behavior are identical and
already tested (tests/test_build_parser_pilot.py, if present; the
underlying functions are covered by tests/test_build_discourse_pilot.py
via the sibling script's identical pattern).

Pipeline Position: data/extracted/eo/free/*.jsonl -> unreviewed
    annotation queue -> human gold -> tests/fixtures/ud/ or a new
    accuracy fixture
Usage: python scripts/eval/build_literary_parser_pilot.py \
    --output data/test_sets/parser_pilot_literary_v1
Inputs: data/extracted/eo/free/{gutenberg,vikifontaro,libera_folio}_sentences.jsonl,
    existing UD fixture texts (excluded from selection)
Outputs: development/heldout JSONL queues of individual sentences and a
    manifest; NOT a gold benchmark
Quality Checks: unique normalized texts, at most 4 sentences reused per
    source_title (a "document" here, same cap as build_parser_pilot.py),
    document-disjoint splits, no parser filter
Last Updated: 2026-09-07
Related Issues: #938
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.build_parser_pilot import document_split, normalized, select  # noqa: E402
from scripts.eval.build_literary_discourse_pilot import EXCLUDE_TITLES, SOURCES  # noqa: E402


def load_rows(free_dir: Path, sources: tuple[str, ...]) -> list[tuple]:
    """Every sentence from the given source files as (sid, text, source,
    title) -- the exact shape build_parser_pilot.select() expects. `sid`
    is synthetic (these sentences are not in the DuckDB store)."""
    rows = []
    for source in sources:
        path = free_dir / f'{source}_sentences.jsonl'
        with path.open(encoding='utf-8') as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                title = row.get('source_title')
                if not title or title in EXCLUDE_TITLES:
                    continue
                text = row.get('sentence', '').strip()
                if not text:
                    continue
                sid = f'{source}#{i:06d}'
                rows.append((sid, text, source, title))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--free-dir', type=Path, default=ROOT / 'data/extracted/eo/free')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=200)
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
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

    rows = load_rows(args.free_dir, tuple(args.sources))
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
        'version': 1, 'seed': 'parser-pilot-v1',
        'free_dir': str(args.free_dir.resolve()),
        'status': 'unreviewed annotation queue, not gold',
        'files': files,
        'sources_used': args.sources,
        'sources': dict(Counter(row['source_name'] for row in chosen)),
        'splits': dict(Counter(row['split'] for row in chosen)),
        'selection': 'source round-robin, at most four per document (source_title); '
                     'no parser labels; documents are NOT in the DuckDB store',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
