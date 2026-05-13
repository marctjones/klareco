#!/usr/bin/env python3
"""
Merge Wikipedia-derived categories into the corpus-extracted proper-noun dict.

VERSION: v2.1
COMPATIBLE WITH: v2.1 parser, eo Wikipedia categories from
                 extract_wikipedia_categories.py
DEPENDENCIES: stdlib only
STAGE: Data / dictionary build

Description:
    Takes two inputs:
      A. proper_nouns_dynamic_v2.json — corpus-extracted dict (cleaned)
      B. wikipedia_categories.json    — Esperanto Wikipedia category labels

    Produces proper_nouns_dynamic_v3.json by:

      1. For each entry in A, look it up in B (case-sensitive). If found
         AND B's category is more specific than A's (i.e. not 'other'):
         override A's category with B's.
      2. For each entry in B but not in A: ADD it to v3 (Wikipedia-only
         entries — strong "this is a proper noun" signal).
      3. Preserve all other A metadata (frequency, source).

    The merge is conservative: A's category is only OVERRIDDEN by B
    when B is more specific. We don't downgrade specific A categories
    to 'other' from B, since the corpus extractor's category may have
    been right even if B doesn't classify it.

    Category specificity ranking (high → low):
        person, place, organization, work > other

Pipeline Position:
    proper_nouns_dynamic_v2.json    (cleaned corpus dict, 190K entries)
    wikipedia_categories.json       (Wiki-derived, ~hundreds of K entries)
        → [THIS SCRIPT]
        → proper_nouns_dynamic_v3.json

Usage:
    # Dry-run (default — counts only):
    python scripts/merge_proper_noun_dictionary.py

    # Apply (writes v3 + audit log):
    python scripts/merge_proper_noun_dictionary.py --apply

Inputs:
    data/proper_nouns_dynamic_v2.json
    data/wikipedia_categories.json

Outputs:
    data/proper_nouns_dynamic_v3.json
    logs/dict_cleanup/merge_<timestamp>.jsonl   (audit log)

Quality Checks:
    - Reports counts of: kept-A, A-overridden, B-added
    - Reports category distribution before/after
    - Idempotent: rerun on v3 + same B produces no changes
    - Audit log records every override for review

Last Updated: 2026-05-07
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the EXACT same stripping logic the parser's dict lookup uses,
# so merged keys collide correctly with corpus-dict keys and lookups
# resolve regardless of source.
from klareco.proper_nouns import ProperNounDictionary

_strip_endings = ProperNounDictionary()._strip_esperanto_endings

# Specificity rank: lower = more specific (preferred during override)
_SPECIFICITY = {
    'person':       1,
    'place':        1,
    'organization': 1,
    'work':         1,
    'other':        9,
    None:           99,
    '':             99,
}

_SPECIFIC_CATEGORIES = {'person', 'place', 'organization', 'work'}


def is_more_specific(new_cat: str, old_cat: str) -> bool:
    return _SPECIFICITY.get(new_cat, 99) < _SPECIFICITY.get(old_cat, 99)


def should_override(new_cat: str, old_cat: str) -> bool:
    """Decide whether to OVERRIDE old_cat with new_cat. Wikipedia categories
    (new_cat) are more authoritative than corpus-extracted categories
    (old_cat) — they come from explicit human classification rather than
    heuristic context. When BOTH are specific (e.g., corpus=place,
    wiki=person), trust Wikipedia. Only KEEP old_cat when wiki contributes
    nothing useful (other / null).
    """
    if new_cat in _SPECIFIC_CATEGORIES:
        return new_cat != old_cat   # always override when wiki has a specific cat
    return is_more_specific(new_cat, old_cat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus-dict', default='data/proper_nouns_dynamic_v2.json')
    ap.add_argument('--wiki-cats',   default='data/wikipedia_categories.json')
    ap.add_argument('--output',      default='data/proper_nouns_dynamic_v3.json')
    ap.add_argument('--apply',       action='store_true',
                    help='Write merged dict + audit log (default: dry-run)')
    ap.add_argument('--audit-dir',   default='logs/dict_cleanup')
    args = ap.parse_args()

    cd_path = Path(args.corpus_dict)
    wc_path = Path(args.wiki_cats)
    out_path = Path(args.output)

    if not cd_path.exists():
        sys.exit(f'ERROR: corpus dict not found at {cd_path}')
    if not wc_path.exists():
        sys.exit(f'ERROR: wikipedia categories not found at {wc_path}')

    print(f'Mode: {"APPLY" if args.apply else "DRY-RUN"}')
    t0 = time.time()
    with open(cd_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    with open(wc_path, 'r', encoding='utf-8') as f:
        wiki_raw = json.load(f)
    print(f'Loaded:  corpus={len(corpus):,}  wiki={len(wiki_raw):,}  ({time.time()-t0:.1f}s)')

    # Normalize Wikipedia keys via the same stripping the dict lookup uses,
    # so e.g. "Aristotelo" → key "Aristotel" matches a lookup of "Aristotelo".
    #
    # Collision handling — when multiple wiki_raw entries strip to the same
    # key (e.g. "Madrido" full-title-place + "Madrid" token-surname-person
    # both normalize to "Madrid"), prefer the FULL-TITLE entry. A full-title
    # entry represents a Wikipedia article literally about that entity
    # (Madrido = the city); a token-surname entry is inferred from articles
    # ABOUT OTHER ENTITIES that share the surname. Full-title is more
    # authoritative for the entity-as-itself classification.
    def _is_full_title_entry(entry: dict) -> bool:
        return 'is_multi_word' in entry  # only full_titles dict carries this

    def _vote_total(entry: dict) -> int:
        votes = entry.get('votes') or {}
        return sum(votes.values()) if isinstance(votes, dict) else 0

    def _entry_priority(surface: str, key: str, entry: dict) -> tuple:
        """Higher tuple → higher priority. Used to break collision ties."""
        is_full = _is_full_title_entry(entry)
        is_base = (surface == key)   # surface form already in stripped/base form
        votes = _vote_total(entry)
        return (int(is_full), int(is_base), votes)

    wiki: dict[str, dict] = {}
    for surface, entry in wiki_raw.items():
        key = _strip_endings(surface)
        if not key:
            continue
        if key in wiki:
            # Pick the higher-priority entry by:
            #   1. full-title beats token
            #   2. base-form surface (surface==key) beats inflected forms
            #      (e.g. "Mozart" wins over "Mozartoj"/"Mozarta")
            #   3. higher total vote count
            existing = wiki[key]
            existing_surface = next(iter(existing.get('orig_titles', [])), key)
            existing_priority = _entry_priority(existing_surface, key, existing)
            new_priority = _entry_priority(surface, key, entry)
            if new_priority > existing_priority:
                new_entry = dict(entry)
                new_entry.setdefault('orig_titles', set())
                if isinstance(new_entry['orig_titles'], list):
                    new_entry['orig_titles'] = set(new_entry['orig_titles'])
                new_entry['orig_titles'].add(surface)
                if isinstance(existing.get('orig_titles'), set):
                    new_entry['orig_titles'].update(existing['orig_titles'])
                wiki[key] = new_entry
            else:
                # Keep existing — just record the colliding surface
                existing.setdefault('orig_titles', set())
                if isinstance(existing['orig_titles'], list):
                    existing['orig_titles'] = set(existing['orig_titles'])
                existing['orig_titles'].add(surface)
        else:
            new_entry = dict(entry)
            new_entry['orig_titles'] = {surface}
            wiki[key] = new_entry
    # Convert orig_titles sets back to lists for JSON
    for v in wiki.values():
        if isinstance(v.get('orig_titles'), set):
            v['orig_titles'] = sorted(v['orig_titles'])[:20]
    print(f'Wiki normalized: {len(wiki):,} unique keys '
          f'(deduplicated from {len(wiki_raw):,})')

    merged: dict[str, dict] = {}
    audit_lines: list[str] = []

    n_kept_a = 0          # entries from corpus, no wiki info
    n_overridden = 0      # corpus entry, category overridden by wiki
    n_no_change = 0       # corpus entry, wiki had same / less specific cat
    n_added_b = 0         # wiki-only entries

    cat_before = Counter()
    cat_after = Counter()

    # Pass 1: walk corpus dict
    for surface, meta in corpus.items():
        if not isinstance(meta, dict):
            meta = {'category': 'other', 'source': 'corpus'}
        old_cat = meta.get('category', 'other')
        cat_before[old_cat] += 1

        wiki_entry = wiki.get(surface)
        if wiki_entry is None:
            new_meta = dict(meta)
            new_meta.setdefault('source', 'corpus')
            merged[surface] = new_meta
            cat_after[new_meta.get('category', 'other')] += 1
            n_kept_a += 1
            continue

        wiki_cat = wiki_entry.get('category', 'other')
        if should_override(wiki_cat, old_cat):
            new_meta = dict(meta)
            new_meta['category'] = wiki_cat
            new_meta['wiki_categories'] = wiki_entry.get('raw_categories', [])[:10]
            new_meta['source'] = 'corpus+wiki'
            merged[surface] = new_meta
            cat_after[wiki_cat] += 1
            n_overridden += 1
            audit_lines.append(json.dumps(
                {'op': 'override', 'surface': surface,
                 'old_cat': old_cat, 'new_cat': wiki_cat,
                 'wiki_raw': wiki_entry.get('raw_categories', [])[:5]},
                ensure_ascii=False))
        else:
            new_meta = dict(meta)
            # Even if cat unchanged, attach wiki provenance for visibility
            new_meta.setdefault('source', 'corpus')
            if wiki_cat != 'other':
                new_meta['wiki_categories'] = wiki_entry.get('raw_categories', [])[:10]
            merged[surface] = new_meta
            cat_after[new_meta.get('category', 'other')] += 1
            n_no_change += 1

    # Pass 2: add wiki-only entries
    for surface, wiki_entry in wiki.items():
        if surface in merged:
            continue
        wiki_cat = wiki_entry.get('category', 'other')
        new_meta = {
            'category':     wiki_cat,
            'source':       'wiki',
            'wiki_categories': wiki_entry.get('raw_categories', [])[:10],
        }
        merged[surface] = new_meta
        cat_after[wiki_cat] += 1
        n_added_b += 1
        audit_lines.append(json.dumps(
            {'op': 'add', 'surface': surface, 'cat': wiki_cat,
             'wiki_raw': wiki_entry.get('raw_categories', [])[:5]},
            ensure_ascii=False))

    print(f'\n=== Merge result ===')
    print(f'  Corpus entries kept (no wiki match): {n_kept_a:,}')
    print(f'  Corpus entries overridden by wiki:   {n_overridden:,}')
    print(f'  Corpus entries no change:            {n_no_change:,}')
    print(f'  Wiki-only entries added:             {n_added_b:,}')
    print(f'  Total merged size:                   {len(merged):,}')

    print(f'\n=== Category distribution: BEFORE (corpus only) ===')
    total_before = sum(cat_before.values())
    for cat in ('person', 'place', 'organization', 'work', 'other'):
        n = cat_before.get(cat, 0)
        if n:
            print(f'  {cat:14s} {n:>8,}  ({100*n/total_before:5.1f}%)')

    print(f'\n=== Category distribution: AFTER (merged) ===')
    total_after = sum(cat_after.values())
    for cat in ('person', 'place', 'organization', 'work', 'other'):
        n = cat_after.get(cat, 0)
        if n:
            print(f'  {cat:14s} {n:>8,}  ({100*n/total_after:5.1f}%)')

    if not args.apply:
        print('\n[DRY-RUN] No files written. Re-run with --apply to commit.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f'\nWrote {len(merged):,} merged entries → {out_path}')

    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f'merge_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    with open(audit_path, 'w', encoding='utf-8') as f:
        for line in audit_lines:
            f.write(line + '\n')
    print(f'Wrote {len(audit_lines):,} audit lines → {audit_path}')


if __name__ == '__main__':
    main()
