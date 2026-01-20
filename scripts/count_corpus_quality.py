#!/usr/bin/env python3
"""
Count quality distribution in corpus with robust error handling.

Usage:
    python scripts/count_corpus_quality.py \\
        --corpus data/enhanced_corpus/corpus_with_metadata.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

def count_quality_robust(corpus_path: Path):
    """Count quality distribution, skipping malformed lines."""

    quality_counts = defaultdict(int)
    total_count = 0
    error_count = 0
    error_lines = []

    print(f"Reading corpus: {corpus_path}")
    print()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                quality = entry.get('source', {}).get('quality', 'UNKNOWN')
                quality_counts[quality] += 1
                total_count += 1

                if total_count % 100000 == 0:
                    print(f"  Processed {total_count:,} sentences...")

            except json.JSONDecodeError as e:
                error_count += 1
                error_lines.append({
                    'line_num': line_num,
                    'error': str(e),
                    'preview': line[:200] if len(line) > 200 else line
                })

                if error_count <= 10:  # Show first 10 errors
                    print(f"  ⚠️  JSON error at line {line_num}: {e}")
                    print(f"     Preview: {line[:100]}...")
                    print()

    print()
    print("=" * 80)
    print("QUALITY DISTRIBUTION")
    print("=" * 80)
    print()

    print(f"Total sentences: {total_count:,}")
    if error_count > 0:
        print(f"Malformed lines: {error_count:,}")
    print()

    print("Breakdown by quality:")
    for quality in ['GOLD', 'SILVER', 'BRONZE', 'COPPER', 'UNKNOWN']:
        count = quality_counts.get(quality, 0)
        if count > 0:
            pct = (count / total_count * 100) if total_count > 0 else 0
            print(f"  {quality:8s}: {count:,} sentences ({pct:.1f}%)")

    print()

    if error_count > 0:
        print(f"Found {error_count} malformed JSON lines")
        print()
        print("Error details saved to: data/corpus_json_errors.json")

        # Save error details
        error_file = Path('data/corpus_json_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_lines, f, indent=2, ensure_ascii=False)

    return quality_counts, total_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Count quality distribution in corpus"
    )
    parser.add_argument(
        '--corpus',
        type=Path,
        default=Path('data/enhanced_corpus/corpus_with_metadata.jsonl'),
        help='Path to corpus file'
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Error: Corpus not found: {args.corpus}")
        return 1

    quality_counts, total_count, error_count = count_quality_robust(args.corpus)

    if error_count > 0:
        print()
        print("Next steps:")
        print("1. Review errors: cat data/corpus_json_errors.json | jq '.'")
        print("2. Fix malformed lines or filter them out")
        print("3. Rebuild corpus if many errors found")

    return 0


if __name__ == '__main__':
    sys.exit(main())
