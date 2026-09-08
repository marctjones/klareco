#!/usr/bin/env python3
"""Prepare a multi-sentence PASSAGE annotation queue for discourse-level
gold (coreference, proper-noun classification) -- no parser labels.

VERSION: v1.0
COMPATIBLE WITH: DuckDB sentences(sid,text,article_id,article_title,section,source_name)
DEPENDENCIES: duckdb; local corpus; no models
STAGE: Evaluation

Every existing test set in this repo is built around ISOLATED sentences,
each disjoint from its neighbors by design (build_parser_pilot.py: "at
most four per document", to maximize genre/document diversity for
dependency-parsing gold). That is the right choice for parsing, and the
wrong shape for discourse phenomena: a pronoun's antecedent, or whether a
proper-noun classification holds across a document, is only checkable
against the sentences actually next to it. Checked directly (2026-09-08):
of the two existing UD fixtures, only eo_prago-ud-test.conllu is a real
continuous document (# newdoc / # newpar markers, one 1906 political
manifesto); eo_cairo-ud-test.conllu is a "parallel" example-sentence bank
(grammar-textbook sentences, individually translated, topically
unrelated to each other despite sequential numbering) and is NOT usable
for this. The parser_pilot_v1/v2 and parser_gold_candidates_v1 queues
carry no article_id/sid linkage in their rows at all. This script is the
answer: select a genuinely CONTIGUOUS run of sentences from ONE document
(by DB article_id, in store sid order, which is the article's real
reading order -- verified by inspection, not assumed) instead of
scattering individual ones.

Pipeline Position: source documents -> unreviewed passage queue -> human
    coreference/proper-noun gold -> klareco/discourse/* precision measurement
Usage: python scripts/eval/build_discourse_pilot.py --output data/test_sets/discourse_pilot_v1
Inputs: local DuckDB store, existing UD fixture texts (excluded from selection)
Outputs: development/heldout JSONL queues of PASSAGES (not individual
    sentences) and a manifest; NOT a gold benchmark
Quality Checks: document-disjoint splits, one passage per document, no
    parser filter, passages start at a document's first sentence (where
    referential setup and article-title-anaphora actually happen)
Last Updated: 2026-09-08
Related Issues: #938 (paragraph-level coreference/proper-noun consistency)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
SEED = 'discourse-pilot-v1'


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def document_split(article_id: str) -> str:
    key = f'{SEED}\0{article_id}'.encode()
    return 'heldout' if int(hashlib.sha256(key).hexdigest(), 16) % 5 == 0 else 'development'


def select(
    documents: dict[str, list[tuple]], excluded: set[str], size: int, passage_length: int,
) -> list[dict]:
    """One passage per qualifying document, source round-robin, document-
    disjoint splits. A document qualifies if it has >= passage_length
    sentences and its FIRST passage_length sentences contain no text
    already used in a frozen UD fixture (so this queue and the parser
    accuracy fixtures never overlap)."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for article_id, rows in documents.items():
        if len(rows) < passage_length:
            continue
        passage = rows[:passage_length]
        if any(normalized(text) in excluded for _sid, text, *_ in passage):
            continue
        by_source[passage[0][3]].append(article_id)   # rows[i][3] == source_name
    for source in by_source:
        by_source[source].sort(
            key=lambda aid: hashlib.sha256(f'{SEED}\0{aid}'.encode()).hexdigest())

    quotas = {'heldout': size // 5, 'development': size - size // 5}
    counts = Counter()
    chosen = []
    while any(by_source.values()) and len(chosen) < size:
        for source in sorted(by_source):
            if not by_source[source]:
                continue
            article_id = by_source[source].pop(0)
            split = document_split(article_id)
            if counts[split] >= quotas[split]:
                continue
            counts[split] += 1
            rows = documents[article_id][:passage_length]
            sentences = [
                {'sid': sid, 'text': text, 'section': section}
                for sid, text, section, _source in rows
            ]
            joined = '\n'.join(s['text'] for s in sentences)
            chosen.append({
                'id': f'{SEED}-{article_id}',
                'article_id': article_id,
                'source_name': rows[0][3],
                'document_title': None,   # filled in by the caller (needs a 2nd query)
                'sentences': sentences,
                'passage_sha256': hashlib.sha256(joined.encode()).hexdigest(),
                'split': split,
                'annotation_status': 'unreviewed',
                'coreference_gold': None,
                'proper_noun_gold': None,
                'reviewer': None,
                'phenomena': [],
            })
            if len(chosen) >= size:
                break
    if len(chosen) != size:
        raise ValueError(
            f'Only {len(chosen)}/{size} eligible documents at passage_length='
            f'{passage_length}; widen the candidate pool or shorten passages')
    return chosen


def main() -> None:
    import duckdb
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', type=Path, default=ROOT / 'data/indexes/duckdb_store.db')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=100,
                     help='number of PASSAGES (documents), not sentences')
    ap.add_argument('--passage-length', type=int, default=8,
                     help='sentences per passage, taken from the start of the document')
    args = ap.parse_args()
    if args.size < 5:
        ap.error('--size must be at least 5')
    if args.passage_length < 3:
        ap.error('--passage-length must be at least 3 to be worth annotating')
    if args.output.exists():
        ap.error('output directory already exists; preserve existing annotations')

    excluded = set()
    for name in ('prago', 'cairo'):
        fixture = ROOT / f'tests/fixtures/ud/eo_{name}-ud-test.conllu'
        excluded.update(normalized(line.split('=', 1)[1])
                        for line in fixture.read_text().splitlines()
                        if line.startswith('# text ='))

    with duckdb.connect(str(args.store), read_only=True) as con:
        # Candidate documents: enough sentences, sampled across a wide pool
        # per source (not every document with article_id needs scanning --
        # sample generously, then filter to qualifying ones below).
        candidate_ids = [r[0] for r in con.execute(
            '''
            WITH q AS (
                SELECT article_id FROM sentences WHERE article_id IS NOT NULL
                GROUP BY article_id HAVING count(*) >= ?
            )
            SELECT article_id FROM q
            USING SAMPLE reservoir(20000 ROWS) REPEATABLE (42)
            ''', [args.passage_length],
        ).fetchall()]
        documents: dict[str, list[tuple]] = {}
        titles: dict[str, str] = {}
        for aid in candidate_ids:
            rows = con.execute(
                'SELECT sid, text, section, source_name, article_title '
                'FROM sentences WHERE article_id = ? ORDER BY sid', [aid],
            ).fetchall()
            documents[aid] = [(sid, text, section, source) for sid, text, section, source, _t in rows]
            titles[aid] = rows[0][4] if rows else None

    chosen = select(documents, excluded, args.size, args.passage_length)
    for row in chosen:
        row['document_title'] = titles.get(row['article_id'])

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
        'status': 'unreviewed PASSAGE annotation queue, not gold',
        'unit': f'{args.passage_length}-sentence passage from one document, '
                f'starting at its first sentence',
        'files': files,
        'sources': dict(Counter(row['source_name'] for row in chosen)),
        'splits': dict(Counter(row['split'] for row in chosen)),
        'selection': 'one passage per document, source round-robin, '
                     'document-disjoint splits; excludes any text already '
                     'in the frozen UD Prago/Cairo fixtures; no parser labels',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
