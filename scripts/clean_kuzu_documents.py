#!/usr/bin/env python3
"""
Filter out Module/Template/Meta pages from the Kuzu documents.jsonl.

This removes Wikipedia pages that contain Lua code, templates, etc.
that pollute retrieval results.
"""

import argparse
import json
import sys
from pathlib import Path

# Article titles to skip (namespace prefixes)
SKIP_PREFIXES = {
    'Modulo:',              # Module pages (Lua code)
    'Modulo-Diskuto:',
    'Ŝablono:',             # Template pages
    'Ŝablono-Diskuto:',
    'Vikipedio:',           # Wikipedia meta pages
    'Diskutejo:',           # Discussion/talk pages
    'Uzanto:',              # User pages
    'Uzanto-Diskuto:',
    'MediaWiki:',           # System pages
    'MediaWiki-Diskuto:',
    'Kategorio:',           # Category pages
    'Kategorio-Diskuto:',
    'Module:',              # English versions
    'Template:',
    'Wikipedia:',
    'User:',
    'Category:',
}

# Content patterns that indicate code/template (not prose)
CODE_PATTERNS = [
    'function p.',
    'local function',
    'return t end',
    '-- modulo',
    'mw.getCurrentFrame()',
    'tonumber(',
    'tostring(',
]


def should_skip(entry: dict) -> tuple[bool, str]:
    """Check if document should be skipped. Returns (skip, reason)."""
    
    # Check article title prefix
    source = entry.get('source', {})
    if isinstance(source, dict):
        title = source.get('article_title', '')
    else:
        title = ''
    
    for prefix in SKIP_PREFIXES:
        if title.startswith(prefix):
            return True, f'namespace:{prefix}'
    
    # Check content for code patterns
    text = entry.get('text', entry.get('sentence', ''))
    for pattern in CODE_PATTERNS:
        if pattern in text:
            return True, f'code:{pattern[:20]}'
    
    return False, ''


def main():
    parser = argparse.ArgumentParser(description='Filter Kuzu documents.jsonl')
    parser.add_argument('--input', type=Path, 
                        default=Path('data/indexes/kuzu_index/documents.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/indexes/kuzu_index/documents_filtered.jsonl'))
    parser.add_argument('--dry-run', action='store_true',
                        help='Count what would be filtered without writing')
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)
    
    stats = {
        'total': 0,
        'kept': 0,
        'skipped': 0,
        'reasons': {},
    }
    
    print(f"Processing: {args.input}")
    if args.dry_run:
        print("DRY RUN - not writing output")
    else:
        print(f"Output: {args.output}")
    
    fout = None if args.dry_run else open(args.output, 'w', encoding='utf-8')
    
    try:
        with open(args.input, 'r', encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                stats['total'] += 1
                
                if i % 500000 == 0 and i > 0:
                    print(f"  {i:,} processed, {stats['kept']:,} kept, {stats['skipped']:,} skipped")
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                skip, reason = should_skip(entry)
                
                if skip:
                    stats['skipped'] += 1
                    stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
                else:
                    stats['kept'] += 1
                    if fout:
                        fout.write(line)
    finally:
        if fout:
            fout.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Total documents:   {stats['total']:,}")
    print(f"Kept documents:    {stats['kept']:,}")
    print(f"Skipped documents: {stats['skipped']:,} ({100*stats['skipped']/stats['total']:.2f}%)")
    
    print("\nSkip reasons:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1])[:15]:
        print(f"  {reason}: {count:,}")
    
    if not args.dry_run and args.output.exists():
        print(f"\nFiltered corpus written to: {args.output}")
        print("\nTo rebuild the Kuzu index:")
        print(f"  1. mv {args.output} {args.input}")
        print("  2. ./scripts/build_kuzu_index.sh --fresh")


if __name__ == '__main__':
    main()
