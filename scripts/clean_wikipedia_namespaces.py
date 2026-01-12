#!/usr/bin/env python3
"""
Filter out talk/discussion/meta pages from Wikipedia corpus.

Removes sentences from non-article namespaces like:
- Vikipedio: (Wikipedia meta pages)
- Diskutejo: (Talk/discussion pages)
- Uzanto: (User pages)
- Ŝablono: (Template pages)
- etc.

Usage:
    python scripts/filter_wikipedia_namespaces.py
    python scripts/filter_wikipedia_namespaces.py --input data/extracted/wikipedia_sentences.jsonl --output data/extracted/wikipedia_sentences_filtered.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Namespaces to skip (talk pages, meta pages, etc.)
SKIP_NAMESPACES = {
    'Vikipedio:',           # Wikipedia meta pages
    'Diskutejo:',           # Discussion/talk pages
    'Uzanto:',              # User pages
    'Uzanto-Diskuto:',      # User talk pages
    'Projekto:',            # Project pages
    'Projekto-Diskuto:',    # Project talk pages
    'Dosiero:',             # File pages
    'Dosiero-Diskuto:',     # File talk pages
    'MediaWiki:',           # MediaWiki system pages
    'MediaWiki-Diskuto:',   # MediaWiki talk pages
    'Ŝablono:',             # Template pages
    'Ŝablono-Diskuto:',     # Template talk pages
    'Helpo:',               # Help pages
    'Helpo-Diskuto:',       # Help talk pages
    'Kategorio:',           # Category pages
    'Kategorio-Diskuto:',   # Category talk pages
    'Portalo:',             # Portal pages
    'Portalo-Diskuto:',     # Portal talk pages
    'Modulo:',              # Module pages (Lua)
    'Modulo-Diskuto:',      # Module talk pages
    # Also skip English-style prefixes that might appear
    'Wikipedia:',
    'Talk:',
    'User:',
    'User talk:',
    'Template:',
    'Template talk:',
    'Category:',
    'Category talk:',
    'File:',
    'File talk:',
    'Help:',
    'Help talk:',
    'Portal:',
    'Portal talk:',
    'Module:',
    'Module talk:',
}


def should_skip_article(title: str) -> bool:
    """Check if article should be skipped based on namespace."""
    if not title:
        return True
    for prefix in SKIP_NAMESPACES:
        if title.startswith(prefix):
            return True
    return False


def filter_corpus(input_path: Path, output_path: Path) -> dict:
    """Filter out non-article namespace sentences."""
    stats = {
        'total_input': 0,
        'total_output': 0,
        'skipped_articles': set(),
        'skipped_sentences': 0,
    }

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Filtering {len(SKIP_NAMESPACES)} namespace prefixes...")

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            stats['total_input'] += 1

            if i % 500000 == 0 and i > 0:
                print(f"  Processed {i:,} lines, kept {stats['total_output']:,}, skipped {stats['skipped_sentences']:,}")

            try:
                entry = json.loads(line)
                title = entry.get('article_title', '')

                if should_skip_article(title):
                    stats['skipped_articles'].add(title)
                    stats['skipped_sentences'] += 1
                    continue

                fout.write(line)
                stats['total_output'] += 1

            except json.JSONDecodeError:
                continue

    return stats


def main():
    parser = argparse.ArgumentParser(description='Filter Wikipedia namespace pages')
    parser.add_argument('--input', type=Path,
                        default=Path('data/extracted/wikipedia_sentences.jsonl'),
                        help='Input JSONL file')
    parser.add_argument('--output', type=Path,
                        default=Path('data/extracted/wikipedia_sentences_filtered.jsonl'),
                        help='Output JSONL file')
    parser.add_argument('--in-place', action='store_true',
                        help='Replace input file with filtered version')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    output_path = args.input if args.in_place else args.output
    temp_output = args.output if not args.in_place else Path(str(args.input) + '.filtered.tmp')

    stats = filter_corpus(args.input, temp_output)

    # If in-place, replace original
    if args.in_place:
        import shutil
        backup_path = Path(str(args.input) + '.backup')
        print(f"\nCreating backup: {backup_path}")
        shutil.copy(args.input, backup_path)
        print(f"Replacing original with filtered version")
        shutil.move(temp_output, args.input)

    # Print summary
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Input sentences:   {stats['total_input']:,}")
    print(f"Output sentences:  {stats['total_output']:,}")
    print(f"Skipped sentences: {stats['skipped_sentences']:,} ({100*stats['skipped_sentences']/stats['total_input']:.1f}%)")
    print(f"Skipped articles:  {len(stats['skipped_articles']):,}")
    print("=" * 60)

    # Show sample of skipped article titles
    if stats['skipped_articles']:
        print("\nSample skipped article titles:")
        for title in list(stats['skipped_articles'])[:10]:
            print(f"  - {title}")


if __name__ == '__main__':
    main()
