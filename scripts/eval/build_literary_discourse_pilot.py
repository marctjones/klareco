#!/usr/bin/env python3
"""Prepare a multi-sentence PASSAGE annotation queue for discourse-level
gold, sourced from EXTRACTED-BUT-NOT-YET-INDEXED public-domain literary
translations -- not the DuckDB store.

VERSION: v1.0
COMPATIBLE WITH: data/extracted/eo/free/*.jsonl schema
  {sentence, source, source_title, author, kind, licence}
DEPENDENCIES: none (pure Python + the sibling build_discourse_pilot module)
STAGE: Evaluation

Why this script exists (found 2026-09-07): `build_discourse_pilot.py`
draws passages from the DuckDB store, but `article_id` is NULL for every
row outside `source_type='encyclopedia'` -- so it can only ever select
Wikipedia documents. Wikipedia biography/topic articles are short and
terse; they under-represent the long anaphoric chains and free indirect
narration that make coreference and proper-noun consistency hard.

Meanwhile `data/extracted/eo/free/` holds three EXTRACTED, LICENSED
corpora that were never wired into `scripts/parse/build_unified_corpus.py`
(its SOURCES list only reads tier0_filtered/, books_sentences.jsonl and
wikipedia_sentences.jsonl -- verified 2026-09-07, `books_sentences.jsonl`
does not even exist on disk) and so never reached the DuckDB store at
all:
  - gutenberg_sentences.jsonl (179,363 sentences, 111 works, licence
    'public domain'): full-length narrative translations -- Poe (6
    works: "La Falo de Uŝero-Domo", "La Puto kaj la Pendolo", "La Murdoj
    de Kadavrejo-Strato", plus a 6-novella collection), Shakespeare
    (Hamlet, Macbeth), Ibsen (8 plays), Twain, Dickens, Stevenson
    (Dr Jekyll), Baum (Wizard of Oz), Turgenev, Pushkin, Schiller,
    Goethe, Orzeszkowa's "Marta", and more.
  - vikifontaro_sentences.jsonl (187,253 sentences, 627 works, licence
    'PD + CC BY-SA 4.0'): Esperanto Wikisource, including Sienkiewicz's
    "Quo Vadis?", Prevost's "Manon Lescaut", a second "Marta" printing,
    Voltaire, plus non-narrative works (a dictionary, an encyclopedia)
    that this script's document-length filter mostly screens out on its
    own since a dictionary is one enormous "document" that would need
    an explicit exclude to avoid dominating the sample -- see EXCLUDE_TITLES.
  - libera_folio_sentences.jsonl (40,038 sentences, 832 articles, licence
    'CC BY 4.0'): a modern, professionally-edited Esperanto news site --
    articles rather than books, and CC BY (attribution required) rather
    than public domain, but included by default alongside the other two.

Included by default because a direct measurement (2026-09-07, n=500
sentences per source, klareco.parser's own `sukcesoprocento` --
fraction of words classified as in-vocabulary Esperanto, excluding
`fremda_vorto`/foreign-word classifications) found Wikipedia sentences
score consistently lower than all three of these sources: wikipedia
0.9798 vs. gutenberg 0.9929, vikifontaro 0.9915, libera_folio 0.9939. A
~1.5-2 point gap is modest, not dramatic, but it is consistent with the
expectation that curated literary translations and edited journalism
carry less untranslated technical vocabulary, foreign proper nouns, and
machine-stub content than Wikipedia's more heterogeneous, often
bot-assisted article base. This does not mean Wikipedia should be
dropped -- it remains the only source with real `article_id` linkage
into the DuckDB store -- but for a NEW discourse/parser pilot built
from scratch, these three sources are a reasonable default, not merely
an addition.

This directly answers "can we use the other Esperanto texts we have,"
"could we find public domain books like Poe," and "shouldn't we prefer
these to Wikipedia" together: the Poe (and Shakespeare, Ibsen, Twain...)
translations, Quo Vadis, and a modern edited newspaper already exist on
disk, fully extracted, measurably cleaner by this one metric, and
simply were never connected to anything downstream.

A "document" here is a maximal run of CONSECUTIVE rows sharing the same
`source_title` (checked 2026-09-07: 110/111 gutenberg titles are already
one contiguous run in file order, which IS the book's reading order --
verified by inspection of "La Falo de Uŝero-Domo"'s first 15 sentences.
The one exception, "Tri Noveloj", is reused for two unrelated
collections; splitting on any break in contiguity handles that
generically instead of special-casing it).

Reuses `document_split` and `select` from build_discourse_pilot.py
UNCHANGED -- this script only supplies a different `documents` dict of
the same shape, so the split logic, source-diversity round robin, and
duplicate-exclusion-against-the-frozen-UD-fixtures behavior are
identical and already tested (tests/test_build_discourse_pilot.py).

Pipeline Position: data/extracted/eo/free/*.jsonl -> unreviewed passage
    queue -> human coreference/proper-noun gold -> klareco/discourse/*
Usage: python scripts/eval/build_literary_discourse_pilot.py \
    --output data/test_sets/discourse_pilot_literary_v1
Inputs: data/extracted/eo/free/{gutenberg,vikifontaro}_sentences.jsonl,
    existing UD fixture texts (excluded from selection)
Outputs: development/heldout JSONL queues of PASSAGES (not individual
    sentences) and a manifest; NOT a gold benchmark. No `sid`/`ast_json`
    -- these sentences are not in the DuckDB store; parse on demand
    (`klareco.parser.parse`, ~7,384 sentences/sec) when annotating or
    measuring, same as any text outside the store.
Quality Checks: document-disjoint splits, one passage per document, no
    parser filter, passages start at a document's first sentence,
    documents shorter than passage_length are skipped
Last Updated: 2026-09-07
Related Issues: #938 (paragraph-level coreference/proper-noun consistency)
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
from scripts.eval.build_discourse_pilot import document_split, normalized, select  # noqa: E402

SOURCES = ('gutenberg', 'vikifontaro', 'libera_folio')

# Not narrative: a dictionary and an encyclopedia extracted alongside the
# literary works in vikifontaro. Including them would let one "document"
# (the entire dictionary, 9,207 sentences) dominate a source's slot in
# the round-robin with content that has no discourse structure at all.
EXCLUDE_TITLES = {
    'Kabe - Vortaro de Esperanto, 1922.pdf',
    'Enciklopedio de Esperanto, 1933.pdf',
}


def load_documents(free_dir: Path, sources: tuple[str, ...]) -> tuple[dict, dict, dict]:
    """Group each source file's rows into maximal consecutive runs of the
    same source_title. Returns (documents, titles, meta) keyed by a
    synthetic document id `f'{source}::{title}::{run_index}'`, where
    documents[id] is a list of (sid, text, section, source) matching the
    exact shape build_discourse_pilot.select() expects."""
    documents: dict[str, list[tuple]] = {}
    titles: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for source in sources:
        path = free_dir / f'{source}_sentences.jsonl'
        current_title = None
        run_index = 0
        doc_id = None
        seq = 0
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                title = row.get('source_title')
                if title in EXCLUDE_TITLES:
                    current_title = None
                    continue
                if title != current_title:
                    current_title = title
                    run_index += 1
                    doc_id = f'{source}::{title}::{run_index}'
                    documents[doc_id] = []
                    titles[doc_id] = title
                    meta[doc_id] = {'author': row.get('author'), 'kind': row.get('kind'),
                                     'licence': row.get('licence')}
                    seq = 0
                sid = f'{source}#{run_index}#{seq:05d}'
                documents[doc_id].append((sid, row['sentence'], None, source))
                seq += 1
    return documents, titles, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--free-dir', type=Path, default=ROOT / 'data/extracted/eo/free')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=100,
                     help='number of PASSAGES (documents), not sentences')
    ap.add_argument('--passage-length', type=int, default=8,
                     help='sentences per passage, taken from the start of the document')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
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

    documents, titles, meta = load_documents(args.free_dir, tuple(args.sources))
    chosen = select(documents, excluded, args.size, args.passage_length)
    for row in chosen:
        row['document_title'] = titles.get(row['article_id'])
        row.update(meta.get(row['article_id'], {}))

    args.output.mkdir(parents=True)
    files = {}
    for split in ('development', 'heldout'):
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n'
                          for row in chosen if row['split'] == split)
        filename = f'{split}.jsonl'
        (args.output / filename).write_text(content)
        files[filename] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        'version': 1, 'seed': 'discourse-pilot-v1',
        'free_dir': str(args.free_dir.resolve()),
        'status': 'unreviewed PASSAGE annotation queue, not gold',
        'unit': f'{args.passage_length}-sentence passage from one document (extracted '
                f'literary text, NOT in the DuckDB store), starting at its first sentence',
        'files': files,
        'sources_used': args.sources,
        'sources': dict(Counter(row['source_name'] for row in chosen)),
        'splits': dict(Counter(row['split'] for row in chosen)),
        'excluded_titles': sorted(EXCLUDE_TITLES),
        'selection': 'one passage per document (a maximal contiguous run of one '
                     'source_title), source round-robin, document-disjoint splits; '
                     'excludes any text already in the frozen UD Prago/Cairo fixtures; '
                     'no parser labels; documents are NOT in the DuckDB store -- parse '
                     'on demand with klareco.parser.parse',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
